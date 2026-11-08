from __future__ import annotations

import pytest

from mini_agent.tools.calculator import calculator_tool, evaluate, extract_expression


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("(17 * 23) + 5", "396"),
        ("2 ** 10", "1024"),
        ("144 ** 0.5", "12"),
        ("((8 + 2) * 3) - 4", "26"),
        ("100 / 4 + 7", "32"),
        ("7 // 2", "3"),
        ("7 % 2", "1"),
        ("-3 + 5", "2"),
    ],
)
def test_evaluate_arithmetic(expression: str, expected: str) -> None:
    assert evaluate(expression) == expected


@pytest.mark.parametrize(
    "expression",
    ["os.system('ls')", "__import__('os')", "a + 1", "(2).bit_length()", "2 ** 10000", "1 / 0"],
)
def test_evaluate_rejects_non_arithmetic(expression: str) -> None:
    with pytest.raises(ValueError):
        evaluate(expression)


def test_division_by_zero_is_explicit() -> None:
    with pytest.raises(ValueError, match="division by zero"):
        evaluate("1 / 0")


def test_extract_expression_picks_the_longest_valid_span() -> None:
    assert extract_expression("What is (17 * 23) + 5?") == "(17 * 23) + 5"
    assert extract_expression("What is 100 / 4 + 7?") == "100 / 4 + 7"
    assert extract_expression("no math here") is None


def test_calculator_tool_returns_errors_as_observations() -> None:
    tool = calculator_tool()
    assert tool.run({"expression": "2 + 2"}) == "4"
    assert tool.run({}) == "Error: expression is required."
    assert tool.run({"expression": "2 ** 10000"}).startswith("Error:")
