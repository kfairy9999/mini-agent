"""Fixed questions. The labels are used by the scorer, never by the mock."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from mini_agent.eval.scoring import score_boiling_point, score_contains, score_number, score_regex


@dataclass(frozen=True)
class Task:
    id: str
    category: str
    question: str
    expectation: str
    score: Callable[[str], bool]


def default_tasks() -> tuple[Task, ...]:
    return (
        Task(
            "math-multiply-add",
            "math",
            "What is (17 * 23) + 5?",
            "396",
            score_number(396),
        ),
        Task(
            "math-power",
            "math",
            "What is 2 ** 10?",
            "1024",
            score_number(1024),
        ),
        Task(
            "math-sqrt",
            "math",
            "What is the square root of 144?",
            "12",
            score_number(12),
        ),
        Task(
            "math-parentheses",
            "math",
            "What is ((8 + 2) * 3) - 4?",
            "26",
            score_number(26),
        ),
        Task(
            "math-division",
            "math",
            "What is 100 / 4 + 7?",
            "32",
            score_number(32),
        ),
        Task(
            "code-sum-squares",
            "code",
            "Use Python to compute the sum of squares from 1 to 10.",
            "385",
            score_number(385),
        ),
        Task(
            "code-fibonacci",
            "code",
            "Use Python to compute the 10th Fibonacci number. The sequence starts 1, 1, 2, 3, 5.",
            "55",
            score_number(55),
        ),
        Task(
            "code-factorial",
            "code",
            "Use Python to compute the factorial of 7.",
            "5040",
            score_number(5040),
        ),
        Task(
            "code-primes",
            "code",
            "Use Python to count how many primes are strictly below 50.",
            "15",
            score_number(15),
        ),
        Task(
            "code-reverse",
            "code",
            'Use Python to reverse the string "agentic" and print it.',
            "citnega",
            score_contains("citnega"),
        ),
        Task(
            "fact-france-capital",
            "factual",
            "What is the capital of France?",
            "contains Paris",
            score_contains("Paris"),
        ),
        Task(
            "fact-moon-landing",
            "factual",
            "In what year was the first crewed Moon landing?",
            "contains 1969",
            score_contains("1969"),
        ),
        Task(
            "fact-gold-symbol",
            "factual",
            "What is the chemical symbol for gold?",
            "word Au",
            score_regex(r"\bAu\b"),
        ),
        Task(
            "fact-gatsby-author",
            "factual",
            "Who wrote The Great Gatsby?",
            "contains Fitzgerald",
            score_contains("Fitzgerald"),
        ),
        Task(
            "fact-water-boiling",
            "factual",
            "What is the boiling point of water at sea level in Celsius?",
            "100 degrees Celsius",
            score_boiling_point,
        ),
    )
