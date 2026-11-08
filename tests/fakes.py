"""Scripted model clients for loop tests."""

from __future__ import annotations

import copy
from typing import Any

from mini_agent.types import LLMResponse, ToolCall


class ScriptedLLM:
    model = "scripted"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        self.calls.append({"messages": copy.deepcopy(messages), "tools": copy.deepcopy(tools)})
        if not self._responses:
            raise AssertionError("ScriptedLLM ran out of responses")
        return self._responses.pop(0)


def tool_call(name: str, arguments: dict[str, Any], *, content: str = "plan", call_id: str = "call_1") -> LLMResponse:
    return LLMResponse(content=content, tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)])
