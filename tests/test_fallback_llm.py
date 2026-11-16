from __future__ import annotations

import pytest

from src.application.llm import LLMClient
from src.infra.llm.fallback_client import FallbackLLMClient, validate_reply


def make_client(**kwargs) -> FallbackLLMClient:
    return FallbackLLMClient(
        primary=kwargs.pop("primary"),
        secondary=kwargs.pop("secondary", None),
        temperature=0.3,
        max_tokens=600,
        instructions="test",
        timeout=10,
        **kwargs,
    )


class Stub(LLMClient):
    def __init__(self, *, text: str = "", raises: Exception | None = None) -> None:
        super().__init__(temperature=0.3, max_tokens=600, instructions="test", timeout=10)
        self.text = text
        self.raises = raises
        self.calls = 0

    def generate_reply(self, prompt: str) -> str:
        self.calls += 1
        if self.raises:
            raise self.raises
        return self.text


GOOD = "Спасибо за отзыв! Рады, что товар вам понравился. Будем ждать вас снова."


@pytest.mark.parametrize(
    "text",
    ["", "   ", "ок", '{"ton": "pos", "reply": "x"}', "```python\nprint(1)\n```", "Thank you for your feedback, we appreciate it."],
)
def test_bad_replies_are_rejected(text: str) -> None:
    ok, cleaned, reason = validate_reply(text)
    assert not ok
    assert cleaned == ""
    assert reason


def test_good_reply_passes() -> None:
    ok, cleaned, reason = validate_reply(GOOD)
    assert ok, reason
    assert cleaned == GOOD


def test_long_reply_is_truncated_at_sentence_boundary() -> None:
    text = "Первый абзац ответа покупателю. " * 60
    ok, cleaned, _ = validate_reply(text, max_length=120)
    assert ok
    assert len(cleaned) <= 121
    assert cleaned.endswith(".")


def test_fallback_used_when_primary_returns_garbage() -> None:
    primary, secondary = Stub(text='{"reply": "x"}'), Stub(text=GOOD)
    client = make_client(primary=primary, secondary=secondary)
    assert client.generate_reply("p") == GOOD
    assert primary.calls == 1 and secondary.calls == 1


def test_fallback_used_when_primary_raises() -> None:
    primary, secondary = Stub(raises=TimeoutError("boom")), Stub(text=GOOD)
    client = make_client(primary=primary, secondary=secondary)
    assert client.generate_reply("p") == GOOD


def test_empty_when_both_fail() -> None:
    primary, secondary = Stub(text=""), Stub(text="ок")
    client = make_client(primary=primary, secondary=secondary)
    assert client.generate_reply("p") == ""


def test_validation_applies_without_fallback() -> None:
    client = make_client(primary=Stub(text="Thank you for your feedback!"))
    assert client.generate_reply("p") == ""
