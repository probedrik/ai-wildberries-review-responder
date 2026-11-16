from __future__ import annotations

from src.application.llm import LLMClient
from src.application.ports import AppLogger
from src.infra.config.settings import LLMRuntimeSettings, Settings


def make_llm_client(settings: Settings, logger: AppLogger | None = None) -> LLMClient:
    """Собрать LLM-клиент: основной провайдер + необязательный резервный + валидация.

    Основной лейн — бесплатный 100milionseek (`extra_body.reasoning_effort = none`),
    резервный — платный OpenAI-совместимый провайдер. Если резервный не настроен,
    валидация ответа всё равно работает: негодный ответ превращается в пустую строку,
    и приложение пропускает отзыв вместо публикации мусора.
    """
    runtime = settings.llm
    from src.infra.llm.fallback_client import FallbackLLMClient
    from src.infra.llm.openai_client import OpenAIClient

    def build(config: LLMRuntimeSettings) -> LLMClient:
        return OpenAIClient(
            config,
            temperature=runtime.temperature,
            max_tokens=runtime.max_tokens,
            instructions=runtime.instructions,
            timeout=runtime.timeout,
        )

    primary = build(runtime)

    secondary: LLMClient | None = None
    if runtime.fallback_base_url and runtime.fallback_model:
        fallback_config = runtime.model_copy(
            update={
                "base_url": runtime.fallback_base_url,
                "api_key": runtime.fallback_api_key or runtime.api_key,
                "model": runtime.fallback_model,
                "extra_body": dict(runtime.fallback_extra_body or {}),
            }
        )
        secondary = build(fallback_config)

    return FallbackLLMClient(
        primary=primary,
        secondary=secondary,
        min_reply_length=runtime.min_reply_length,
        max_reply_length=runtime.max_reply_length,
        logger=logger,  # без него отказы звеньев не видны в логе
        temperature=runtime.temperature,
        max_tokens=runtime.max_tokens,
        instructions=runtime.instructions,
        timeout=runtime.timeout,
    )
