"""Arithmetic with a fail-closed AST walk. No Python ``eval``."""

from __future__ import annotations

import ast
import math
import re
from typing import Any

from mini_agent.tools.registry import Tool

_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.UAdd,
    ast.USub,
)
_CANDIDATE = re.compile(r"[0-9()+\-*/%.\s]{3,}")
_OPERATOR = re.compile(r"[+\-*/%]")
_MAX_EXPONENT = 100
_MAX_BITS = 512


def evaluate(expression: str) -> str:
    """Evaluate one arithmetic expression and return a short decimal string."""

    text = expression.strip()
    if not text:
        raise ValueError("expression is required")
    if len(text) > 200:
        raise ValueError("expression is too long")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError("expression is not valid arithmetic") from exc
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError("only arithmetic is allowed")
    value = _eval(tree.body)
    return _format(value)


def extract_expression(text: str) -> str | None:
    """Return the longest arithmetic span in ``text`` that evaluates."""

    best: str | None = None
    for match in _CANDIDATE.finditer(text):
        candidate = match.group(0).strip()
        if not _OPERATOR.search(candidate):
            continue
        try:
            evaluate(candidate)
        except ValueError:
            continue
        if best is None or len(candidate) > len(best):
            best = candidate
    return best


def calculator_tool() -> Tool:
    def run(arguments: dict[str, Any]) -> str:
        expression = arguments.get("expression")
        if not isinstance(expression, str) or not expression.strip():
            return "Error: expression is required."
        try:
            return evaluate(expression)
        except ValueError as exc:
            return f"Error: {exc}"

    return Tool(
        name="calculator",
        description=(
            "Evaluate one arithmetic expression. Supports +, -, *, /, //, %, **, "
            "parentheses, and numeric literals. For a square root, use ** 0.5."
        ),
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression, for example (17 * 23) + 5.",
                }
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
        handler=run,
    )


def _eval(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant):
        if type(node.value) not in (int, float):
            raise ValueError("only numeric literals are allowed")
        return node.value
    if isinstance(node, ast.UnaryOp):
        value = _eval(node.operand)
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return -value
        raise ValueError("only arithmetic is allowed")
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Pow):
            exponent = _eval(node.right)
            if isinstance(exponent, bool) or not isinstance(exponent, (int, float)):
                raise ValueError("exponent must be a number")
            if abs(exponent) > _MAX_EXPONENT:
                raise ValueError("exponent is too large")
            base = _eval(node.left)
            try:
                result = base ** exponent
            except OverflowError as exc:
                raise ValueError("result is too large") from exc
            return _check_result(result)
        left = _eval(node.left)
        right = _eval(node.right)
        try:
            if isinstance(node.op, ast.Add):
                result = left + right
            elif isinstance(node.op, ast.Sub):
                result = left - right
            elif isinstance(node.op, ast.Mult):
                result = left * right
            elif isinstance(node.op, ast.Div):
                result = left / right
            elif isinstance(node.op, ast.FloorDiv):
                result = left // right
            elif isinstance(node.op, ast.Mod):
                result = left % right
            else:
                raise ValueError("only arithmetic is allowed")
        except ZeroDivisionError as exc:
            raise ValueError("division by zero") from exc
        return _check_result(result)
    raise ValueError("only arithmetic is allowed")


def _check_result(value: int | float | complex) -> int | float:
    if isinstance(value, complex) or type(value) is complex:
        raise ValueError("complex results are not supported")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("result is not a finite number")
    if isinstance(value, int) and not isinstance(value, bool) and value.bit_length() > _MAX_BITS:
        raise ValueError("result is too large")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("result is not a number")
    return value


def _format(value: int | float) -> str:
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 10**15:
            return str(int(value))
        return format(value, ".12g")
    return str(value)
