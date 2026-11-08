from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

import pytest

from mini_agent.config import LLMConfig
from mini_agent.llm import LLMError, OpenAICompatibleClient, parse_chat_response, urllib_transport


def _config() -> LLMConfig:
    return LLMConfig(
        provider="groq",
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.3-70b-versatile",
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
    assert payload["model"] == "llama-3.3-70b-versatile"
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
    with pytest.raises(LLMError, match="bad key"):
        urllib_transport("https://example.invalid", {}, {"model": "x"}, 1)
