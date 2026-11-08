"""OpenAI-compatible chat completions over the standard library."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Protocol

from mini_agent.config import LLMConfig
from mini_agent.types import LLMResponse, ToolCall


class LLMError(RuntimeError):
    """The chat-completions call failed or returned something we cannot use."""


class LLMClient(Protocol):
    model: str

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        """Return the next model turn. ``tools`` uses the OpenAI function schema."""


Transport = Any


class OpenAICompatibleClient:
    """POST ``{base_url}/chat/completions`` and parse tool calls.

    Groq, Gemini's OpenAI-compatible endpoint, and OpenAI itself all speak
    this shape. The transport is injectable so tests never open a socket.
    """

    def __init__(self, config: LLMConfig, transport: Transport | None = None) -> None:
        if not config.api_key:
            raise LLMError("LLM API key is empty.")
        self.model = config.model
        self._config = config
        self._transport = transport or urllib_transport

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self._config.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        url = self._config.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        body = self._transport(url, headers, payload, self._config.timeout)
        return parse_chat_response(body)


def urllib_transport(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        message = error_message(detail) or str(exc.reason)
        raise LLMError(f"LLM request failed ({exc.code}): {message}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"LLM request failed: {exc.reason}") from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMError("LLM response was not valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise LLMError("LLM response was not a JSON object.")
    if "error" in parsed and "choices" not in parsed:
        raise LLMError(f"LLM request failed: {error_message(parsed) or parsed['error']}")
    return parsed


def parse_chat_response(body: dict[str, Any]) -> LLMResponse:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMError("LLM response did not include choices.")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise LLMError("LLM response did not include a message.")
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        content = str(content)
    tool_calls: list[ToolCall] = []
    for index, raw_call in enumerate(message.get("tool_calls") or []):
        parsed = _parse_tool_call(raw_call, index)
        if parsed is not None:
            tool_calls.append(parsed)
    return LLMResponse(content=content, tool_calls=tool_calls)


def error_message(detail: str | dict[str, Any]) -> str:
    if isinstance(detail, dict):
        error = detail.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error)
        if error is not None:
            return str(error)
        return ""
    try:
        parsed = json.loads(detail)
    except json.JSONDecodeError:
        return detail.strip()[:500]
    if isinstance(parsed, dict):
        return error_message(parsed)
    return detail.strip()[:500]


def _parse_tool_call(raw_call: Any, index: int) -> ToolCall | None:
    if not isinstance(raw_call, dict):
        return None
    function = raw_call.get("function") or {}
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    if not isinstance(name, str) or not name:
        return None
    arguments = function.get("arguments") if "arguments" in function else "{}"
    if isinstance(arguments, str):
        if not arguments.strip():
            arguments = {}
        else:
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"_unparsed": arguments}
    if not isinstance(arguments, dict):
        arguments = {"_unparsed": arguments}
    call_id = raw_call.get("id")
    if not isinstance(call_id, str) or not call_id:
        call_id = f"call_{index}"
    return ToolCall(id=call_id, name=name, arguments=arguments)
