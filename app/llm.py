"""LLM client interface and an OpenAI-compatible implementation."""
import time
from collections.abc import Callable
from typing import Any, Protocol

from app.config import Settings, get_settings

PLACEHOLDER_KEYS = {"", "your-key-here", "changeme", "xxx", "none"}


class LLMError(Exception):
    """Base class for LLM failures; `http_status` is what the API should return."""

    http_status: int = 500

    def __init__(self, message: str, http_status: int | None = None) -> None:
        """Keep `message`; `http_status` overrides the class default when given."""
        super().__init__(message)
        self.message = message
        if http_status is not None:
            self.http_status = http_status


class LLMQuotaExceeded(LLMError):
    """The provider rejected the call for quota/rate reasons (HTTP 429)."""

    http_status = 429


class LLMUpstreamError(LLMError):
    """The provider failed upstream (HTTP 502 / upstream_error)."""

    http_status = 502
    retryable = False  # True only for a 502 / upstream_error, per spec


class LLMClient(Protocol):
    """Minimal interface the rest of the app depends on."""

    def complete(self, system: str, user: str) -> str:
        """Return the model's text reply for a system + user prompt."""
        ...


def is_placeholder_key(key: str) -> bool:
    """True if the key is empty or an obvious example value."""
    k = key.strip().lower()
    return k in PLACEHOLDER_KEYS or k.startswith("your-") or k.startswith("your_")


def _error_code(exc: Exception) -> str:
    """Extract a provider error code string from an openai exception, if any."""
    code = getattr(exc, "code", None)
    body: Any = getattr(exc, "body", None)
    if not code and isinstance(body, dict):
        inner = body.get("error", body)
        if isinstance(inner, dict):
            code = inner.get("code") or inner.get("type")
    return str(code or "").lower()


def _content_text(content: Any) -> str:
    """Flatten a chat reply: plain string, or a list of text blocks / strings."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        return "".join(parts)
    return str(content)


def classify_error(exc: Exception) -> LLMError:
    """Map an exception from the openai client to a typed LLMError."""
    import openai

    status = getattr(exc, "status_code", None)
    code = _error_code(exc)
    if isinstance(exc, openai.RateLimitError) or status == 429 or "quota" in code:
        return LLMQuotaExceeded(
            "LLM quota exceeded (daily quota or rate limit reached); try again later."
        )
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return LLMUpstreamError("Could not reach the LLM provider (connection error or timeout).")
    if status == 502 or code == "upstream_error":
        err = LLMUpstreamError(f"The LLM provider failed upstream (status {status}).")
        err.retryable = True
        return err
    if status in (500, 503, 504):
        return LLMUpstreamError(f"The LLM provider failed upstream (status {status}).")
    if status in (401, 403):
        return LLMError("The LLM provider rejected the API key.", 500)
    return LLMError(f"LLM call failed ({type(exc).__name__}, status {status}).", 500)


class OpenAICompatibleLLM:
    """LLMClient backed by langchain-openai ChatOpenAI; configured via Settings."""

    def __init__(
        self,
        settings: Settings | None = None,
        chat_model: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        backoff: float = 1.0,
    ) -> None:
        """`chat_model` (anything with .invoke(messages)) is injectable for tests."""
        self.settings = settings or get_settings()
        self._chat = chat_model
        self._sleep = sleep
        self._backoff = backoff

    def _model(self) -> Any:
        """Build the chat model lazily so a missing key fails at call time."""
        if self._chat is None:
            key = self.settings.llm_api_key.get_secret_value()
            if is_placeholder_key(key):
                raise LLMError(
                    "LLM_API_KEY is missing or a placeholder; set a real key in .env.", 500
                )
            from langchain_openai import ChatOpenAI

            self._chat = ChatOpenAI(
                base_url=self.settings.llm_base_url,
                api_key=self.settings.llm_api_key,
                model=self.settings.llm_model,
                temperature=self.settings.llm_temperature,
                timeout=self.settings.llm_timeout,
                max_retries=0,
            )
        return self._chat

    def _invoke(self, system: str, user: str) -> str:
        """One call to the chat model, with errors mapped to LLMError."""
        chat = self._model()
        try:
            msg = chat.invoke([("system", system), ("human", user)])
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001 - mapped to a typed error
            raise classify_error(exc) from None
        content = msg.content if hasattr(msg, "content") else msg
        return _content_text(content)

    def complete(self, system: str, user: str) -> str:
        """Return the reply; retry a 502/upstream_error exactly once, never a 429."""
        try:
            return self._invoke(system, user)
        except LLMUpstreamError as exc:
            if not exc.retryable:
                raise
            self._sleep(self._backoff)
            return self._invoke(system, user)
