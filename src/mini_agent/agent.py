"""The agent loop: plan, call tools, observe, self-check, then answer."""

from __future__ import annotations

import json
import time
from typing import Any

from mini_agent.llm import LLMClient
from mini_agent.prompts import REFLECTION_PROMPT, SYSTEM_PROMPT
from mini_agent.tools.registry import ToolRegistry
from mini_agent.types import AgentResult, Observation, Step

DEFAULT_MAX_STEPS = 8
_MAX_ANSWER_CHARS = 8_000


class Agent:
    """Plan, act, observe, then refuse to answer until a self-check finishes.

    Each model call is one step. Tool calls run in the registry and come
    back as observations. A text response is only a draft: the loop injects
    a self-check and spends one more step. That draft becomes the answer
    after the check accepts or revises it. If the budget runs out first,
    the result is unverified.
    """

    def __init__(self, llm: LLMClient, registry: ToolRegistry, *, max_steps: int = DEFAULT_MAX_STEPS) -> None:
        if max_steps < 1 or max_steps > 30:
            raise ValueError("max_steps must be between 1 and 30")
        self.llm = llm
        self.registry = registry
        self.max_steps = max_steps

    def run(self, question: str) -> AgentResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        steps: list[Step] = []
        awaiting_reflection = False
        draft = ""
        model_calls = 0
        started = time.perf_counter()
        schemas = self.registry.schemas()

        for index in range(1, self.max_steps + 1):
            response = self.llm.complete(messages, schemas)
            model_calls += 1
            if response.tool_calls:
                observations = self._observe(response.tool_calls)
                messages.append(_assistant_tool_message(response.content, response.tool_calls))
                messages.extend(_tool_messages(observations))
                steps.append(
                    Step(
                        index=index,
                        kind="act",
                        text=(response.content or "").strip(),
                        observations=observations,
                    )
                )
                awaiting_reflection = False
                continue

            text = (response.content or "").strip()
            if not awaiting_reflection:
                draft = text
                steps.append(Step(index=index, kind="draft", text=text))
                if index == self.max_steps:
                    steps.append(
                        Step(
                            index=index + 1,
                            kind="limit",
                            text="Step budget exhausted before the self-check.",
                        )
                    )
                    return self._finish(
                        answer=text or "Stopped before a checked answer.",
                        steps=steps,
                        stop_reason="max_steps",
                        verified=False,
                        started=started,
                        model_calls=model_calls,
                    )
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": REFLECTION_PROMPT})
                awaiting_reflection = True
                continue

            verdict, answer, notes = parse_reflection(text, draft)
            steps.append(Step(index=index, kind="reflect", text=notes, verdict=verdict))
            return self._finish(
                answer=answer,
                steps=steps,
                stop_reason="reflected",
                verified=True,
                started=started,
                model_calls=model_calls,
            )

        steps.append(
            Step(
                index=len(steps) + 1,
                kind="limit",
                text=f"Stopped after {self.max_steps} model calls.",
            )
        )
        return self._finish(
            answer="Stopped before a checked answer.",
            steps=steps,
            stop_reason="max_steps",
            verified=False,
            started=started,
            model_calls=model_calls,
        )

    def _observe(self, tool_calls: list[Any]) -> list[Observation]:
        observations: list[Observation] = []
        for call in tool_calls:
            output = self.registry.call(call.name, call.arguments)
            observations.append(
                Observation(
                    tool_call_id=call.id,
                    name=call.name,
                    arguments=dict(call.arguments),
                    output=output,
                )
            )
        return observations

    def _finish(
        self,
        *,
        answer: str,
        steps: list[Step],
        stop_reason: str,
        verified: bool,
        started: float,
        model_calls: int,
    ) -> AgentResult:
        text = answer.strip()
        if len(text) > _MAX_ANSWER_CHARS:
            text = text[:_MAX_ANSWER_CHARS] + "\n...[truncated]"
        return AgentResult(
            answer=text,
            steps=steps,
            stop_reason=stop_reason,
            verified=verified,
            model=self.llm.model,
            model_calls=model_calls,
            latency_ms=(time.perf_counter() - started) * 1000,
        )


def parse_reflection(text: str, draft: str) -> tuple[str, str, str]:
    """Pull a verdict out of the self-check turn.

    Models sometimes wrap the JSON in a fence or a sentence. If there is no
    object, the text itself is treated as an accepted answer so a chatty
    model still finishes the loop.
    """

    cleaned = _strip_fences(text)
    data = _first_object(cleaned)
    if isinstance(data, dict):
        verdict = str(data.get("verdict", "accept")).strip().lower()
        if verdict not in {"accept", "revise"}:
            verdict = "accept"
        answer = _answer_field(data.get("answer"), draft)
        notes = data.get("notes")
        note_text = notes.strip() if isinstance(notes, str) else ""
        return verdict, answer, note_text
    body = cleaned.strip() or draft.strip()
    return "accept", body, "Unstructured reflection; treating the text as the answer."


def _answer_field(raw: Any, draft: str) -> str:
    if isinstance(raw, bool):
        return draft.strip()
    if isinstance(raw, (int, float)):
        return str(raw)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return draft.strip()


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    stripped = stripped.split("\n", 1)[-1] if "\n" in stripped else stripped
    if stripped.endswith("```"):
        stripped = stripped[: stripped.rfind("```")]
    return stripped.strip()


def _first_object(text: str) -> Any:
    start = text.find("{")
    if start == -1:
        return None
    try:
        data, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return data


def _assistant_tool_message(content: str | None, tool_calls: list[Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": content or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments),
                },
            }
            for call in tool_calls
        ],
    }


def _tool_messages(observations: list[Observation]) -> list[dict[str, str]]:
    return [
        {
            "role": "tool",
            "tool_call_id": item.tool_call_id,
            "content": item.output,
        }
        for item in observations
    ]
