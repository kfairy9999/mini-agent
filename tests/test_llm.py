from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

import pytest

from mini_agent.config import LLMConfig
from mini_agent.llm import (
    USER_AGENT,
    LLMError,
    OpenAICompatibleClient,
    RateLimitError,
    ToolValidationError,
    parse_chat_response,
    retry_after_seconds,
    urllib_transport,
)


def _config() -> LLMConfig:
    return LLMConfig(
        provider="groq",
        base_url="https://api.groq.com/openai/v1",
        model="qwen/qwen3.8-27b",
        api_key="test-key",
        timeout=5,
        max_tokens=128,
    )


def test_client_posts_chat_completions_and_parses_tool_calls() -> None:
    seen: dict[str, object] = {}

    def transport(url: str, headers: dict[str, str], payload: dict[str, object], timeout: float) -> dict[str, object]:
        seen["url"] = url
        seen["headers"] = headers
        seen["payload"] = payload
        seen["timeout"] = timeout
        return {
            "choices": [
                {
                    "message": {
                        "content": "I'll calculate it.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "calculator", "arguments": '{"expression": "2 + 2"}'},
                            }
                        ],
                    }
                }
            ]
        }

    response = OpenAICompatibleClient(_config(), transport=transport).complete(
        [{"role": "user", "content": "2 + 2"}],
        [{"type": "function", "function": {"name": "calculator"}}],
    )
    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    headers = seen["headers"]
    assert isinstance(headers, dict)
    assert headers["Authorization"] == "Bearer test-key"
    payload = seen["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "qwen/qwen3.8-27b"
    assert headers["User-Agent"] == USER_AGENT
    assert "Python-urllib" not in headers["User-Agent"]
    assert payload["temperature"] == 0
    assert "test-key" not in json.dumps({key: value for key, value in payload.items()})
    assert response.tool_calls[0].name == "calculator"
    assert response.tool_calls[0].arguments == {"expression": "2 + 2"}
    assert response.content == "I'll calculate it."


def test_parse_content_only_message() -> None:
    response = parse_chat_response({"choices": [{"message": {"content": "396"}}]})
    assert response.content == "396"
    assert response.tool_calls == []


def test_trailing_slash_on_gemini_base_url() -> None:
    seen: dict[str, str] = {}

    def transport(url: str, headers: dict[str, str], payload: dict[str, object], timeout: float) -> dict[str, object]:
        seen["url"] = url
        return {"choices": [{"message": {"content": "ok"}}]}

    config = LLMConfig(
        provider="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        model="gemini-3.8-flash",
        api_key="gem-test",
        timeout=5,
        max_tokens=32,
    )
    OpenAICompatibleClient(config, transport=transport).complete([{"role": "user", "content": "hi"}], [])
    assert seen["url"] == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"


def test_urllib_transport_surfaces_http_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> object:
        raise urllib.error.HTTPError(
            url="https://api.groq.com/openai/v1/chat/completions",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=io.BytesIO(b'{"error":{"message":"bad key"}}'),
        )

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(LLMError, match="bad key") as caught:
        urllib_transport("https://example.invalid", {}, {"model": "x"}, 1)
    assert not isinstance(caught.value, RateLimitError)


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_retry_after_parses_groq_hints() -> None:
    body = (
        "Rate limit reached. Please try again in 5.048571428s. "
        "Earlier note: Please try again in 67.499999ms."
    )
    assert retry_after_seconds(None, body) == pytest.approx(5.048571428)
    assert retry_after_seconds({"Retry-After": "1", "x-ratelimit-reset-tokens": "6m0s"}, body) == pytest.approx(360)
    assert retry_after_seconds({"x-ratelimit-reset-requests": "2s"}, "") == pytest.approx(2)


def test_client_retries_429_then_paces_the_next_call() -> None:
    clock = _Clock()
    requests: list[int] = []

    def transport(url: str, headers: dict[str, str], payload: dict[str, object], timeout: float) -> dict[str, object]:
        requests.append(1)
        if len(requests) == 1:
            raise RateLimitError(
                "LLM request failed (429): Please try again in 5.04s",
                retry_after=retry_after_seconds(None, "Please try again in 5.04s."),
            )
        return {"choices": [{"message": {"content": "396"}}]}

    client = OpenAICompatibleClient(_config(), transport=transport, sleep=clock.sleep, clock=clock.clock)
    first = client.complete([{"role": "user", "content": "What is (17 * 23) + 5?"}], [])
    assert first.content == "396"
    assert clock.sleeps[0] == pytest.approx(5.04 + 0.25)
    assert len(requests) == 2

    client.complete([{"role": "user", "content": "next"}], [])
    assert clock.sleeps[1] == pytest.approx(5.04 + 0.25)
    assert len(requests) == 3


def test_rate_limit_is_raised_after_the_retry_budget() -> None:
    clock = _Clock()

    def transport(url: str, headers: dict[str, str], payload: dict[str, object], timeout: float) -> dict[str, object]:
        raise RateLimitError("LLM request failed (429): slow down", retry_after=0.5)

    config = _config()
    config = LLMConfig(
        provider=config.provider,
        base_url=config.base_url,
        model=config.model,
        api_key=config.api_key,
        timeout=config.timeout,
        max_tokens=config.max_tokens,
        max_retries=1,
    )
    client = OpenAICompatibleClient(config, transport=transport, sleep=clock.sleep, clock=clock.clock)
    with pytest.raises(RateLimitError, match="slow down"):
        client.complete([{"role": "user", "content": "hi"}], [])
    assert len(clock.sleeps) == 1


def test_unregistered_json_tool_is_retried_with_a_correction() -> None:
    seen: list[list[dict[str, object]]] = []
    original = [{"role": "user", "content": "What is 2 + 2?"}]

    def transport(url: str, headers: dict[str, str], payload: dict[str, object], timeout: float) -> dict[str, object]:
        messages = payload["messages"]
        assert isinstance(messages, list)
        seen.append(messages)
        if len(seen) == 1:
            raise ToolValidationError(
                "LLM request failed (400): Tool call validation failed: attempted to call tool 'json' "
                "which was not in request.tools",
                tool_name="json",
            )
        return {
            "choices": [
                {
                    "message": {
                        "content": "I'll calculate it.",
                        "tool_calls": [
                            {
                                "id": "call_ok",
                                "type": "function",
                                "function": {"name": "calculator", "arguments": '{"expression": "2 + 2"}'},
                            }
                        ],
                    }
                }
            ]
        }

    response = OpenAICompatibleClient(_config(), transport=transport).complete(
        original,
        [{"type": "function", "function": {"name": "calculator"}}],
    )
    assert original == [{"role": "user", "content": "What is 2 + 2?"}]
    assert len(seen) == 2
    correction = seen[1][-1]
    assert correction["role"] == "user"
    assert "json" in str(correction["content"])
    assert "calculator" in str(correction["content"])
    assert response.tool_calls[0].name == "calculator"


def test_repeated_json_tool_validation_fails_clearly() -> None:
    def transport(url: str, headers: dict[str, str], payload: dict[str, object], timeout: float) -> dict[str, object]:
        raise ToolValidationError(
            "LLM request failed (400): attempted to call tool 'json' which was not in request.tools",
            tool_name="json",
        )

    with pytest.raises(LLMError, match="not in request.tools"):
        OpenAICompatibleClient(_config(), transport=transport).complete([{"role": "user", "content": "hi"}], [])


def test_urllib_transport_maps_429_and_json_tool_400(monkeypatch: pytest.MonkeyPatch) -> None:
    from email.message import Message

    cases: list[urllib.error.HTTPError] = []
    limited = Message()
    limited["Retry-After"] = "1"
    limited["x-ratelimit-reset-tokens"] = "6m0s"
    cases.append(
        urllib.error.HTTPError(
            url="https://api.groq.com/openai/v1/chat/completions",
            code=429,
            msg="Too Many Requests",
            hdrs=limited,
            fp=io.BytesIO(b'{"error":{"message":"Please try again in 5.048571428s."}}'),
        )
    )
    rejected = Message()
    cases.append(
        urllib.error.HTTPError(
            url="https://api.groq.com/openai/v1/chat/completions",
            code=400,
            msg="Bad Request",
            hdrs=rejected,
            fp=io.BytesIO(
                b'{"error":{"message":"Tool call validation failed: attempted to call tool \'json\' '
                b'which was not in request.tools"}}'
            ),
        )
    )

    def boom(*args: object, **kwargs: object) -> object:
        raise cases.pop(0)

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(RateLimitError, match="5.048571428s") as limited_error:
        urllib_transport("https://example.invalid", {}, {"model": "x"}, 1)
    assert limited_error.value.retry_after == pytest.approx(360)

    with pytest.raises(ToolValidationError, match="json") as rejected_error:
        urllib_transport("https://example.invalid", {}, {"model": "x"}, 1)
    assert rejected_error.value.tool_name == "json"
