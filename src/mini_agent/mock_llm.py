"""A deterministic tool-using policy for tests and offline demos.

The policy does not know the evaluation labels. It picks a tool from the
wording of the question. The draft it returns is the tool observation: a
calculated number, the last line of stdout, or a search snippet.
"""

from __future__ import annotations

import json
import re
from typing import Any

from mini_agent.prompts import SELF_CHECK_TAG
from mini_agent.tools.calculator import extract_expression
from mini_agent.types import LLMResponse, ToolCall

_SQRT = re.compile(r"square root of (\d+(?:\.\d+)?)", re.IGNORECASE)
_SUM_SQUARES = re.compile(r"sum of squares from (\d+) to (\d+)", re.IGNORECASE)
_FIBONACCI = re.compile(r"(\d+)(?:st|nd|rd|th)?\s+fibonacci", re.IGNORECASE)
_FACTORIAL = re.compile(r"factorial of (\d+)", re.IGNORECASE)
_PRIMES = re.compile(
    r"primes?\s+(?:are\s+)?(?:strictly\s+)?below\s+(\d+)",
    re.IGNORECASE,
)
_REVERSE = re.compile(r'reverse the string "([^"]+)"', re.IGNORECASE)


class MockLLM:
    """Stand-in model. Same ``complete`` shape as the HTTP client."""

    model = "mock"

    def __init__(self) -> None:
        self._calls = 0

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        self._calls += 1
        last = messages[-1] if messages else {"role": "user", "content": ""}
        if _is_self_check(last):
            return _reflect(messages)
        if last.get("role") == "tool":
            return LLMResponse(content=_draft(messages))
        question = _question(messages)
        return _plan(question, _tool_names(tools), self._calls)


def program_for(question: str) -> str | None:
    """Return a short program for questions the sandbox can answer, or None."""

    sum_match = _SUM_SQUARES.search(question)
    if sum_match:
        start, end = int(sum_match.group(1)), int(sum_match.group(2))
        if end < start or end - start > 100_000:
            return None
        return f"print(sum(i * i for i in range({start}, {end} + 1)))\n"

    fib_match = _FIBONACCI.search(question)
    if fib_match:
        n = int(fib_match.group(1))
        if n < 1 or n > 200:
            return None
        return (
            "def fib(n):\n"
            "    if n <= 2:\n"
            "        return 1\n"
            "    a, b = 1, 1\n"
            "    for _ in range(n - 2):\n"
            "        a, b = b, a + b\n"
            "    return b\n"
            f"print(fib({n}))\n"
        )

    factorial_match = _FACTORIAL.search(question)
    if factorial_match:
        n = int(factorial_match.group(1))
        if n < 0 or n > 200:
            return None
        return (
            f"n = {n}\n"
            "result = 1\n"
            "for i in range(2, n + 1):\n"
            "    result *= i\n"
            "print(result)\n"
        )

    prime_match = _PRIMES.search(question)
    if prime_match:
        limit = int(prime_match.group(1))
        if limit < 2 or limit > 100_000:
            return None
        return (
            f"limit = {limit}\n"
            "count = 0\n"
            "for value in range(2, limit):\n"
            "    if all(value % divisor != 0 for divisor in range(2, int(value ** 0.5) + 1)):\n"
            "        count += 1\n"
            "print(count)\n"
        )

    reverse_match = _REVERSE.search(question)
    if reverse_match:
        literal = json.dumps(reverse_match.group(1))
        return f"print({literal}[::-1])\n"
    return None


def _plan(question: str, tools: set[str], call_number: int) -> LLMResponse:
    program = program_for(question)
    if program is not None and "python" in tools:
        return _tool(
            "python",
            {"code": program},
            "This needs a short program, so I'll run it in the Python sandbox.",
            call_number,
        )
    sqrt = _SQRT.search(question)
    if sqrt and "calculator" in tools:
        expression = f"{sqrt.group(1)} ** 0.5"
        return _tool(
            "calculator",
            {"expression": expression},
            "The question asks for a square root, so I'll evaluate it with the calculator.",
            call_number,
        )
    expression = extract_expression(question) if "calculator" in tools else None
    if expression is not None:
        return _tool(
            "calculator",
            {"expression": expression},
            "The question is arithmetic, so I'll evaluate it with the calculator.",
            call_number,
        )
    if "web_search" in tools:
        return _tool(
            "web_search",
            {"query": question, "max_results": 3},
            "This is a factual lookup, so I'll search and answer from the snippets.",
            call_number,
        )
    return LLMResponse(content="I don't have a tool that can answer this.")


def _tool(name: str, arguments: dict[str, Any], plan: str, call_number: int) -> LLMResponse:
    return LLMResponse(
        content=plan,
        tool_calls=[ToolCall(id=f"call_{call_number}", name=name, arguments=arguments)],
    )


def _draft(messages: list[dict[str, Any]]) -> str:
    question = _question(messages)
    outputs = [str(message.get("content") or "") for message in messages if message.get("role") == "tool"]
    observation = outputs[-1] if outputs else ""
    if observation.startswith("Error:"):
        return "The tool failed, so I do not have a supported answer."
    if observation.startswith("No results"):
        return "I could not find a sourced answer in the search results."
    stdout = _python_stdout(observation)
    if stdout is not None:
        return stdout
    snippet = _best_snippet(question, observation)
    if snippet:
        return snippet
    return observation.strip() or "I do not have a supported answer."


def _reflect(messages: list[dict[str, Any]]) -> LLMResponse:
    draft = _last_draft(messages)
    outputs = [str(message.get("content") or "") for message in messages if message.get("role") == "tool"]
    latest = outputs[-1] if outputs else ""
    failed = latest.startswith(("Error:", "No results"))
    if failed:
        honest = "could not" in draft.lower() or "no results" in draft.lower() or draft.startswith("The tool failed")
        if draft.strip() and honest:
            return _json_response(
                "accept",
                draft.strip(),
                "The draft reports the tool outcome instead of inventing an answer.",
            )
        return _json_response(
            "revise",
            "I could not produce a supported answer from the tools.",
            "The tool did not return a usable observation.",
        )
    if not draft.strip():
        return _json_response("revise", latest.strip(), "The draft was empty, so the observation is used.")
    return _json_response("accept", draft.strip(), "The draft is grounded in the latest tool observation.")


def _json_response(verdict: str, answer: str, notes: str) -> LLMResponse:
    payload = {"verdict": verdict, "answer": answer, "notes": notes}
    return LLMResponse(content=json.dumps(payload))


def _question(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        content = str(message.get("content") or "")
        if message.get("role") == "user" and SELF_CHECK_TAG not in content:
            return content
    return ""


def _last_draft(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant" and not message.get("tool_calls"):
            return str(message.get("content") or "")
    return ""


def _is_self_check(message: dict[str, Any]) -> bool:
    return message.get("role") == "user" and SELF_CHECK_TAG in str(message.get("content") or "")


def _tool_names(tools: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for tool in tools:
        function = tool.get("function") or {}
        name = function.get("name")
        if isinstance(name, str):
            names.add(name)
    return names


def _python_stdout(observation: str) -> str | None:
    if not observation.startswith("exit_code: 0"):
        return None
    start = observation.find("--- stdout ---\n")
    end = observation.find("\n--- stderr ---")
    if start == -1 or end == -1 or end < start:
        return None
    stdout = observation[start + len("--- stdout ---\n") : end]
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        return "The program ran but printed nothing."
    return lines[-1]


def _best_snippet(question: str, observation: str) -> str | None:
    snippets: list[str] = []
    for chunk in observation.split("\n\n"):
        lines = [line.strip() for line in chunk.splitlines() if line.strip()]
        if len(lines) >= 3 and lines[0].startswith("[") and lines[1].startswith("http"):
            snippets.append(" ".join(lines[2:]))
    if not snippets:
        return None
    wanted = set(re.findall(r"[a-z0-9]+", question.lower()))

    def overlap(snippet: str) -> int:
        have = set(re.findall(r"[a-z0-9]+", snippet.lower()))
        return len(wanted & have)

    return max(snippets, key=overlap)
