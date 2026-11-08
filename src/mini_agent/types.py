"""Shared records for model calls, tool observations, and agent runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class Observation:
    tool_call_id: str
    name: str
    arguments: dict[str, Any]
    output: str


@dataclass
class Step:
    """One visible row in the trace.

    ``act``, ``draft``, and ``reflect`` are model calls. ``limit`` is a
    runtime notice written when the step budget runs out.
    """

    index: int
    kind: str
    text: str = ""
    verdict: str | None = None
    observations: list[Observation] = field(default_factory=list)


@dataclass
class AgentResult:
    answer: str
    steps: list[Step]
    stop_reason: str
    verified: bool
    model: str
    model_calls: int
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "stop_reason": self.stop_reason,
            "verified": self.verified,
            "model": self.model,
            "model_calls": self.model_calls,
            "latency_ms": round(self.latency_ms, 2),
            "steps": [_step_to_dict(step) for step in self.steps],
        }


def _step_to_dict(step: Step) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "index": step.index,
        "kind": step.kind,
        "text": step.text,
    }
    if step.verdict is not None:
        payload["verdict"] = step.verdict
    if step.observations:
        payload["observations"] = [
            {
                "tool_call_id": item.tool_call_id,
                "name": item.name,
                "arguments": item.arguments,
                "output": item.output,
            }
            for item in step.observations
        ]
    return payload
