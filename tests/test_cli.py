from __future__ import annotations

import json
from pathlib import Path

import pytest

from mini_agent.cli import main as agent_main
from mini_agent.eval.__main__ import main as eval_main


def test_cli_mock_math_trace(capsys: pytest.CaptureFixture[str]) -> None:
    code = agent_main(["--mock", "What is (17 * 23) + 5?"])
    captured = capsys.readouterr()
    assert code == 0
    assert "396" in captured.out
    assert "reflect" in captured.out
    assert "calculator" in captured.out
    assert "deterministic mock" not in captured.err


def test_cli_wraps_a_long_question(capsys: pytest.CaptureFixture[str]) -> None:
    question = "Use Python to compute the 10th Fibonacci number. The sequence starts 1, 1, 2, 3, 5."
    code = agent_main(["--mock", question])
    captured = capsys.readouterr()
    assert code == 0
    assert "3, 5." in captured.out
    assert "55" in captured.out


def test_cli_auto_without_a_key_says_it_is_using_the_mock(capsys: pytest.CaptureFixture[str]) -> None:
    code = agent_main(["What is 2 ** 10?"])
    captured = capsys.readouterr()
    assert code == 0
    assert "1024" in captured.out
    assert "deterministic mock" in captured.err


def test_cli_json_and_step_budget(capsys: pytest.CaptureFixture[str]) -> None:
    code = agent_main(["--mock", "--json", "What is 2 + 2?"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0
    assert payload["verified"] is True
    assert payload["answer"] == "4"

    stopped = agent_main(["--mock", "--max-steps", "1", "What is 2 + 2?"])
    assert stopped == 2


def test_cli_rejects_an_empty_question() -> None:
    assert agent_main(["   "]) == 2


def test_eval_command_writes_markdown_and_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = eval_main(["--output-dir", str(tmp_path)])
    captured = capsys.readouterr()
    assert code == 0
    assert "Success rate" in captured.out
    assert "Average steps" in captured.out
    assert "Average latency" in captured.out
    assert (tmp_path / "eval_results.md").is_file()
    assert (tmp_path / "eval_results.json").is_file()
    assert "eval_results.json" in captured.err
