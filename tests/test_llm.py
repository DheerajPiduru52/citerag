"""Phase 3 tests: LLM client error handling (no network, no key)."""
import sys
from pathlib import Path

import httpx
import openai
import pytest
from pydantic import SecretStr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings, get_settings  # noqa: E402
from app.llm import (  # noqa: E402
    LLMError, LLMQuotaExceeded, LLMUpstreamError, OpenAICompatibleLLM, is_placeholder_key,
)


def status_error(status: int, code: str | None = None) -> openai.APIStatusError:
    req = httpx.Request("POST", "http://x/v1/chat/completions")
    resp = httpx.Response(status, request=req)
    body = {"error": {"code": code, "message": "m"}} if code else None
    return openai.APIStatusError("boom", response=resp, body=body)


class FakeChat:
    """Stands in for ChatOpenAI: replays a script of replies/exceptions."""

    def __init__(self, script: list) -> None:
        self.script, self.calls = list(script), []

    def invoke(self, messages):
        self.calls.append(messages)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return type("Msg", (), {"content": item})()


def make(script: list) -> tuple[OpenAICompatibleLLM, FakeChat, list[float]]:
    chat, sleeps = FakeChat(script), []
    return OpenAICompatibleLLM(Settings(), chat, sleep=sleeps.append, backoff=0.5), chat, sleeps


def test_prompt_passed_correctly() -> None:
    llm, chat, _ = make(["hello"])
    assert llm.complete("SYS", "USER") == "hello"
    assert chat.calls == [[("system", "SYS"), ("human", "USER")]]


def test_429_quota_no_retry() -> None:
    llm, chat, sleeps = make([status_error(429, "daily_quota_exceeded"), "never"])
    with pytest.raises(LLMQuotaExceeded) as ei:
        llm.complete("s", "u")
    assert ei.value.http_status == 429 and "quota" in ei.value.message.lower()
    assert len(chat.calls) == 1 and sleeps == []


def test_rate_limit_error_class_is_quota() -> None:
    req = httpx.Request("POST", "http://x")
    err = openai.RateLimitError("rl", response=httpx.Response(429, request=req), body=None)
    llm, chat, _ = make([err])
    with pytest.raises(LLMQuotaExceeded):
        llm.complete("s", "u")
    assert len(chat.calls) == 1


def test_502_retried_once_then_success() -> None:
    llm, chat, sleeps = make([status_error(502, "upstream_error"), "ok"])
    assert llm.complete("s", "u") == "ok"
    assert len(chat.calls) == 2 and sleeps == [0.5]


@pytest.mark.parametrize("status", [500, 503, 504])
def test_other_5xx_not_retried(status: int) -> None:
    llm, chat, sleeps = make([status_error(status), "never"])
    with pytest.raises(LLMUpstreamError):
        llm.complete("s", "u")
    assert len(chat.calls) == 1 and sleeps == []


def test_timeout_and_connection_error_not_retried() -> None:
    req = httpx.Request("POST", "http://x")
    for exc in (openai.APITimeoutError(req), openai.APIConnectionError(request=req)):
        llm, chat, sleeps = make([exc, "never"])
        with pytest.raises(LLMUpstreamError) as ei:
            llm.complete("s", "u")
        assert len(chat.calls) == 1 and sleeps == [] and ei.value.message


def test_list_content_blocks_joined() -> None:
    blocks = [{"type": "text", "text": "Hello "}, {"type": "image", "x": 1}, "world",
              {"type": "text", "text": "!"}]
    llm, _, _ = make([blocks])
    assert llm.complete("s", "u") == "Hello world!"


def test_upstream_error_code_without_502_status_retries() -> None:
    llm, chat, _ = make([status_error(400, "upstream_error"), "ok"])
    assert llm.complete("s", "u") == "ok"


def test_502_twice_raises_upstream() -> None:
    llm, chat, sleeps = make([status_error(502), status_error(502), "never"])
    with pytest.raises(LLMUpstreamError) as ei:
        llm.complete("s", "u")
    assert ei.value.http_status == 502
    assert len(chat.calls) == 2 and len(sleeps) == 1


def test_other_error_is_llm_error_not_retried() -> None:
    llm, chat, _ = make([status_error(401)])
    with pytest.raises(LLMError) as ei:
        llm.complete("s", "u")
    assert type(ei.value) is LLMError and len(chat.calls) == 1


@pytest.mark.parametrize("key", ["", "your-key-here", "  "])
def test_missing_or_placeholder_key_errors_at_call_time(key: str) -> None:
    llm = OpenAICompatibleLLM(Settings(llm_api_key=SecretStr(key)))  # no error at construction
    with pytest.raises(LLMError, match="LLM_API_KEY"):
        llm.complete("s", "u")


def test_placeholder_detection() -> None:
    assert is_placeholder_key("your-key-here") and not is_placeholder_key("sk-abc123")


def test_key_not_exposed_in_repr() -> None:
    s = Settings(llm_api_key=SecretStr("sk-super-secret"))
    assert "sk-super-secret" not in repr(s) and "sk-super-secret" not in str(s)


@pytest.mark.live
def test_live_llm_roundtrip() -> None:
    """Calls the real endpoint; skipped unless a real key is configured."""
    settings = get_settings()
    if is_placeholder_key(settings.llm_api_key.get_secret_value()):
        pytest.skip("no real LLM_API_KEY configured")
    out = OpenAICompatibleLLM(settings).complete("Reply with one word.", "Say: pong")
    assert isinstance(out, str) and out.strip()
