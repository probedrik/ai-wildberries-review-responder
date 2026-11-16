from __future__ import annotations

import re
from typing import Optional

from src.application.llm import LLMClient
from src.application.ports import AppLogger

# Модель иногда возвращает не ответ, а служебный мусор: JSON-обёртку, блок кода,
# отказ ассистента. Такие ответы публиковать нельзя — они уходят покупателю.
_JSON_START = re.compile(r"^[\s`]*[\[{]")
_FENCE = re.compile(r"```")
_REFUSAL_MARKERS = (
    "as an ai",
    "i'm sorry, but i can",
    "i am unable to",
    "как ии",
    "как искусственный интеллект",
    "я не могу ответить",
    "не могу помочь с этим",
)
_CYRILLIC = re.compile(r"[а-яё]", re.IGNORECASE)
_SENTENCE_END = re.compile(r"[.!?…]")


def validate_reply(
    text: Optional[str],
    *,
    min_length: int = 20,
    max_length: int = 1000,
    require_cyrillic: bool = True,
) -> tuple[bool, str, str]:
    """Проверить ответ перед публикацией.

    Возвращает ``(ok, cleaned_text, reason)``. ``reason`` пуст, когда ответ годен.
    Слишком длинный ответ обрезается по последнему предложению в пределах лимита.
    """
    if not text or not text.strip():
        return False, "", "пустой ответ"

    cleaned = text.strip().strip('"').strip("'").strip()
    cleaned = cleaned.replace("**", "").replace("__", "")

    if not cleaned:
        return False, "", "пустой ответ после очистки"
    if _FENCE.search(cleaned):
        return False, "", "markdown-блок кода вместо ответа"
    if _JSON_START.match(cleaned):
        return False, "", "JSON/массив вместо текста: " + cleaned[:60]

    lowered = cleaned.lower()
    for marker in _REFUSAL_MARKERS:
        if marker in lowered:
            return False, "", "отказ модели: " + marker

    if require_cyrillic and not _CYRILLIC.search(cleaned):
        return False, "", "ответ не на русском: " + cleaned[:60]

    if len(cleaned) > max_length:
        window = cleaned[: max_length + 1]
        cut = None
        for match in _SENTENCE_END.finditer(window):
            cut = match.end()
        cleaned = (window[:cut] if cut else cleaned[:max_length]).strip()

    if len(cleaned) < min_length:
        return False, "", f"слишком короткий ответ ({len(cleaned)} симв.)"

    return True, cleaned, ""


class FallbackLLMClient(LLMClient):
    """Пробует основной провайдер, при пустом/негодном ответе — резервный.

    Нужен потому, что бесплатный лейн (100milionseek) периодически отдаёт пустой
    ``content`` или служебный мусор вместо текста, а провайдеры с оплатой — нет.
    Причина отказа логируется вместе с именем звена, чтобы сбои были видны.
    """

    def __init__(
        self,
        *,
        primary: LLMClient,
        secondary: LLMClient | None = None,
        min_reply_length: int = 20,
        max_reply_length: int = 1000,
        require_cyrillic: bool = True,
        logger: AppLogger | None = None,
        **client_kwargs: object,
    ) -> None:
        super().__init__(**client_kwargs)  # type: ignore[arg-type]
        self.primary = primary
        self.secondary = secondary
        self.min_reply_length = min_reply_length
        self.max_reply_length = max_reply_length
        self.require_cyrillic = require_cyrillic
        self.logger = logger

    def _log(self, level: str, message: str, *args: object) -> None:
        if self.logger is None:
            return
        handler = getattr(self.logger, level, None)
        if handler is not None:
            handler(message, *args)

    def _try(self, name: str, client: LLMClient, prompt: str) -> tuple[bool, str, str]:
        try:
            raw = client.generate_reply(prompt)
        except Exception as exc:  # сеть, таймаут, 5xx — звено выбывает
            return False, "", f"{name}: ошибка вызова — {type(exc).__name__}: {exc}"
        ok, cleaned, reason = validate_reply(
            raw,
            min_length=self.min_reply_length,
            max_length=self.max_reply_length,
            require_cyrillic=self.require_cyrillic,
        )
        if ok:
            return True, cleaned, ""
        return False, "", f"{name}: {reason}"

    def generate_reply(self, prompt: str) -> str:
        ok, text, reason = self._try("primary", self.primary, prompt)
        if ok:
            return text

        self._log("warning", "Основное звено не дало ответа ({}) — пробую резервное", reason)
        if self.secondary is not None:
            ok, text, fb_reason = self._try("fallback", self.secondary, prompt)
            if ok:
                return text
            self._log(
                "error",
                "Оба звена без ответа: {}; {}",
                reason,
                fb_reason,
            )
        return ""
