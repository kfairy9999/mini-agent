from __future__ import annotations

import json

import pytest

from mini_agent.agent import Agent, parse_reflection
from mini_agent.eval.scoring import score_number
from mini_agent.factory import build_agent
from mini_agent.prompts import SELF_CHECK_TAG
from mini_agent.tools import build_registry
from mini_agent.types import LLMResponse, ToolCall
from tests.fakes import ScriptedLLM, tool_call


def test_mock_math_runs_plan_observe_and_self_check() -> None:
    result = build_agent(provider="mock").run("What is (17 * 23) + 5?")
    assert [step.kind for step in result.steps] == ["act", "draft", "reflect"]
    assert result.steps[0].observations[0].name == "calculator"
    assert result.steps[0].observations[0].output == "396"
    assert result.steps[2].verdict == "accept"
    assert result.answer == "396"
    assert result.verified
    assert result.stop_reason == "reflected"
    assert result.model_calls == 3
    assert result.model == "mock"


def test_self_check_can_revise_before_the_answer_is_returned() -> None:
    llm = ScriptedLLM(
        [
            LLMResponse(content="41"),
            LLMResponse(content=json.dumps({"verdict": "revise", "answer": "42", "notes": "off by one"})),
        ]
    )
    result = Agent(llm=llm, registry=build_registry("fixture")).run("anything")
    assert llm.calls[1]["messages"][-1]["content"].startswith(SELF_CHECK_TAG)
    assert result.steps[0].text == "41"
    assert result.steps[1].verdict == "revise"
    assert result.answer == "42"
    assert result.verified


def test_fenced_reflection_json_is_accepted() -> None:
    text = '```json\n{"verdict":"accept","answer":"396","notes":"matches the tool"}\n```'
    verdict, answer, notes = parse_reflection(text, draft="0")
    assert (verdict, answer, notes) == ("accept", "396", "matches the tool")


def test_step_budget_stops_an_unfinished_loop() -> None:
    class LoopLLM:
        model = "loop"

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages: list[dict[str, object]], tools: list[dict[str, object]]) -> LLMResponse:
            self.calls += 1
            return LLMResponse(
                content="still working",
                tool_calls=[ToolCall(id=f"c{self.calls}", name="calculator", arguments={"expression": "1 + 1"})],
            )

    llm = LoopLLM()
    result = Agent(llm=llm, registry=build_registry("fixture"), max_steps=3).run("loop")
    assert llm.calls == 3
    assert result.model_calls == 3
    assert result.verified is False
    assert result.stop_reason == "max_steps"
    assert result.steps[-1].kind == "limit"
    assert result.answer == "Stopped before a checked answer."


def test_draft_on_the_last_step_is_not_treated_as_checked() -> None:
    llm = ScriptedLLM([LLMResponse(content="42")])
    result = Agent(llm=llm, registry=build_registry("fixture"), max_steps=1).run("hi")
    assert llm.calls and len(llm.calls) == 1
    assert result.verified is False
    assert "42" in result.answer


def test_tool_error_is_an_observation_and_the_model_can_retry() -> None:
    llm = ScriptedLLM(
        [
            tool_call("calculator", {"expression": "1 / 0"}, call_id="bad"),
            tool_call("calculator", {"expression": "2 + 2"}, content="retry", call_id="good"),
            LLMResponse(content="4"),
            LLMResponse(content=json.dumps({"verdict": "accept", "answer": "4", "notes": "the retry worked"})),
        ]
    )
    result = Agent(llm=llm, registry=build_registry("fixture")).run("add")
    assert "division by zero" in result.steps[0].observations[0].output
    assert result.steps[1].observations[0].output == "4"
    assert result.answer == "4"
    assert result.verified


def test_parallel_tool_calls_all_run() -> None:
    llm = ScriptedLLM(
        [
            LLMResponse(
                content="both",
                tool_calls=[
                    ToolCall("a", "calculator", {"expression": "2 + 2"}),
                    ToolCall("b", "calculator", {"expression": "3 + 3"}),
                ],
            ),
            LLMResponse(content="done"),
            LLMResponse(content=json.dumps({"verdict": "accept", "answer": "done", "notes": "both ran"})),
        ]
    )
    result = Agent(llm=llm, registry=build_registry("fixture")).run("both")
    outputs = [item.output for item in result.steps[0].observations]
    assert outputs == ["4", "6"]
    tool_messages = [message for message in llm.calls[1]["messages"] if message["role"] == "tool"]
    assert [message["tool_call_id"] for message in tool_messages] == ["a", "b"]


def test_unknown_tool_does_not_crash_the_loop() -> None:
    llm = ScriptedLLM(
        [
            tool_call("telescope", {}, content="looking"),
            LLMResponse(content="I could not use that tool."),
            LLMResponse(
                content=json.dumps(
                    {
                        "verdict": "accept",
                        "answer": "I could not use that tool.",
                        "notes": "the tool is not registered",
                    }
                )
            ),
        ]
    )
    result = Agent(llm=llm, registry=build_registry("fixture")).run("sky")
    assert "unknown tool" in result.steps[0].observations[0].output
    assert result.verified


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is 3 ** 5?", 243),
        ("What is the square root of 81?", 9),
        ("Use Python to compute the factorial of 6.", 720),
        ("Use Python to compute the sum of squares from 2 to 4.", 29),
        ("Use Python to compute the 8th Fibonacci number. The sequence starts 1, 1, 2.", 21),
        ("Use Python to count how many primes are strictly below 10.", 4),
        ('Use Python to reverse the string "hire" and print it.', "erih"),
    ],
)
def test_mock_policy_generalizes_past_the_eval_set(question: str, expected: int | str) -> None:
    result = build_agent(provider="mock").run(question)
    assert result.verified
    assert result.model_calls == 3
    if isinstance(expected, int):
        assert score_number(expected)(result.answer)
    else:
        assert expected in result.answer


def test_mock_does_not_invent_an_unknown_fact() -> None:
    result = build_agent(provider="mock").run("What is the capital of Atlantis?")
    assert result.verified
    assert "paris" not in result.answer.lower()
    assert "could not" in result.answer.lower()
    assert "source: fixture" in result.steps[0].observations[0].output


def test_a_custom_tool_is_just_a_registration() -> None:
    from mini_agent.tools.registry import Tool

    registry = build_registry("fixture")
    registry.register(
        Tool(
            name="echo",
            description="Repeat text.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            handler=lambda arguments: str(arguments["text"]),
        )
    )
    llm = ScriptedLLM(
        [
            tool_call("echo", {"text": "hello"}, content="echo it"),
            LLMResponse(content="hello"),
            LLMResponse(content=json.dumps({"verdict": "accept", "answer": "hello", "notes": "echoed"})),
        ]
    )
    result = Agent(llm=llm, registry=registry).run("say hello")
    assert result.steps[0].observations[0].output == "hello"
    assert "echo" in [schema["function"]["name"] for schema in llm.calls[0]["tools"]]
