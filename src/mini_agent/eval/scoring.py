"""Automatic checks for the fixed task set.

A check looks at the final answer only. It does not know which tool ran.
Passing also requires the agent loop to finish its self-check; that rule
lives in the harness, not here.
"""

from __future__ import annotations

import re
from collections.abc import Callable

_NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d+\.\d+|\d+)(?![\w])")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d)")


def extract_numbers(text: str) -> list[float]:
    cleaned = _THOUSANDS.sub("", text)
    return [float(match.group(0)) for match in _NUMBER.finditer(cleaned)]


def score_number(expected: float, *, tolerance: float = 1e-6) -> Callable[[str], bool]:
    def score(answer: str) -> bool:
        return any(abs(value - expected) <= tolerance for value in extract_numbers(answer))

    return score


def score_contains(text: str) -> Callable[[str], bool]:
    needle = text.casefold()

    def score(answer: str) -> bool:
        return needle in answer.casefold()

    return score


def score_regex(pattern: str) -> Callable[[str], bool]:
    compiled = re.compile(pattern)

    def score(answer: str) -> bool:
        return compiled.search(answer) is not None

    return score


def score_boiling_point(answer: str) -> bool:
    text = re.sub(r"\s+", " ", answer.lower().replace("°", " "))
    if re.search(r"(?<!\d)100(?!\d)", text) is None:
        return False
    return "celsius" in text or "degrees c" in text or "100 c" in text
