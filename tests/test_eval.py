from __future__ import annotations

import json
from pathlib import Path

import pytest

from mini_agent.eval.harness import run_evaluation
from mini_agent.eval.report import render_markdown
from mini_agent.eval.scoring import extract_numbers
from mini_agent.eval.tasks import default_tasks
from mini_agent.llm import RateLimitError
from mini_agent.types import AgentResult, Step


def test_task_set_covers_math_code_and_facts() -> None:
    tasks = default_tasks()
    assert 10 <= len(tasks) <= 15
    assert len({task.id for task in tasks}) == len(tasks)
    assert {task.category for task in tasks} == {"math", "code", "factual"}


def test_number_scorer_ignores_ordinals_and_reads_a_trailing_period() -> None:
    assert extract_numbers("The 10th Fibonacci number is 55.") == [55]
    assert extract_numbers("The result is 1,024.") == [1024]


def test_mock_eval_passes_and_writes_both_reports(tmp_path: Path) -> None:
    report = run_evaluation(output_dir=tmp_path)
    failed = [(row.id, row.answer, row.stop_reason) for row in report.rows if not row.passed]
    assert failed == []
    assert report.provider == "mock"
    assert report.model == "mock"
    assert report.search == "fixture"
    assert report.success_rate == 1
    assert report.passed == report.total == 15
    assert report.rate_limited == 0
    assert report.scored == 15
    assert report.average_steps == 3
    assert report.average_latency_ms >= 0
    payload = json.loads((tmp_path / "eval_results.json").read_text(encoding="utf-8"))
    assert payload["mode"] == "mock"
    assert payload["success_rate"] == 1
    assert "average_steps" in payload
    assert "average_latency_ms" in payload
    assert "not measurements of Groq" in payload["disclaimer"]
    markdown = (tmp_path / "eval_results.md").read_text(encoding="utf-8")
    assert markdown == render_markdown(report)
    assert "Success rate" in markdown
    assert "Rate limited" in markdown
    assert "Average steps" in markdown
    assert "Average latency" in markdown
    assert "not measurements of Groq" in markdown


def _checked(answer: str) -> AgentResult:
    return AgentResult(
        answer=answer,
        steps=[Step(index=1, kind="reflect", text="ok", verdict="accept")],
        stop_reason="reflected",
        verified=True,
        model="fake",
        model_calls=3,
        latency_ms=1.0,
    )


def test_a_rate_limited_task_is_retried_then_scored() -> None:
    class Flaky:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, question: str) -> AgentResult:
            self.calls += 1
            if self.calls == 1:
                raise RateLimitError("LLM request failed (429): try again in 5s", retry_after=5)
            return _checked("396")

    flaky = Flaky()
    report = run_evaluation(agent=flaky, tasks=(default_tasks()[0],))
    assert flaky.calls == 2
    assert report.rows[0].passed
    assert report.rows[0].stop_reason == "reflected"
    assert report.rate_limited == 0
    assert report.success_rate == 1


def test_exhausted_rate_limit_is_not_a_model_failure() -> None:
    class AlwaysLimited:
        def run(self, question: str) -> AgentResult:
            if "2 ** 10" in question:
                raise RateLimitError("LLM request failed (429): still limited", retry_after=5)
            return _checked("396")

    report = run_evaluation(agent=AlwaysLimited(), tasks=default_tasks()[:2])
    assert report.rows[0].passed
    assert report.rows[1].stop_reason == "rate_limited"
    assert report.rows[1].passed is False
    assert "not a model failure" in report.rows[1].answer
    assert report.passed == 1
    assert report.rate_limited == 1
    assert report.scored == 1
    assert report.success_rate == 1
    markdown = render_markdown(report)
    assert "rate limit" in markdown


def test_eval_prints_progress_and_keeps_partial_results(tmp_path: Path) -> None:
    lines: list[str] = []

    class StopAfterFirst:
        def run(self, question: str) -> AgentResult:
            if "2 ** 10" in question:
                raise KeyboardInterrupt
            return _checked("396")

    with pytest.raises(KeyboardInterrupt):
        run_evaluation(
            agent=StopAfterFirst(),
            tasks=default_tasks()[:2],
            output_dir=tmp_path,
            on_progress=lines.append,
        )
    assert lines[0].startswith("[1/2] math-multiply-add starting")
    assert "pass" in lines[1]
    saved = json.loads((tmp_path / "eval_results.json").read_text(encoding="utf-8"))
    assert saved["partial"] is True
    assert [row["id"] for row in saved["tasks"]] == ["math-multiply-add"]
    assert "Partial" in (tmp_path / "eval_results.md").read_text(encoding="utf-8")
