from __future__ import annotations

import json
from pathlib import Path

from mini_agent.eval.harness import run_evaluation
from mini_agent.eval.report import render_markdown
from mini_agent.eval.scoring import extract_numbers
from mini_agent.eval.tasks import default_tasks


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
    assert "Average steps" in markdown
    assert "Average latency" in markdown
    assert "not measurements of Groq" in markdown
