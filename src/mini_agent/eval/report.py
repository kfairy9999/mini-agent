"""Markdown and JSON reports. Latency and pass/fail are measured, not assumed."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class TaskResult:
    id: str
    category: str
    question: str
    expectation: str
    passed: bool
    verified: bool
    steps: int
    latency_ms: float
    stop_reason: str
    answer: str


@dataclass
class EvalReport:
    provider: str
    model: str
    search: str
    disclaimer: str
    generated_at: str
    rows: list[TaskResult]
    partial: bool = False

    @property
    def total(self) -> int:
        return len(self.rows)

    @property
    def passed(self) -> int:
        return sum(1 for row in self.rows if row.passed)

    @property
    def rate_limited(self) -> int:
        return sum(1 for row in self.rows if row.stop_reason == "rate_limited")

    @property
    def scored(self) -> int:
        """Tasks that finished or failed for a reason other than the rate limit."""

        return self.total - self.rate_limited

    @property
    def success_rate(self) -> float:
        if self.scored == 0:
            return 0.0
        return self.passed / self.scored

    @property
    def average_steps(self) -> float:
        scored = self._scored_rows()
        if not scored:
            return 0.0
        return sum(row.steps for row in scored) / len(scored)

    @property
    def average_latency_ms(self) -> float:
        scored = self._scored_rows()
        if not scored:
            return 0.0
        return sum(row.latency_ms for row in scored) / len(scored)

    def _scored_rows(self) -> list[TaskResult]:
        return [row for row in self.rows if row.stop_reason != "rate_limited"]

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model,
            "search": self.search,
            "mode": "mock" if self.provider == "mock" else "hosted",
            "disclaimer": self.disclaimer,
            "generated_at": self.generated_at,
            "partial": self.partial,
            "success_rate": self.success_rate,
            "passed": self.passed,
            "total": self.total,
            "scored": self.scored,
            "rate_limited": self.rate_limited,
            "average_steps": self.average_steps,
            "average_latency_ms": round(self.average_latency_ms, 2),
            "categories": _categories(self.rows),
            "tasks": [
                {
                    "id": row.id,
                    "category": row.category,
                    "question": row.question,
                    "expectation": row.expectation,
                    "passed": row.passed,
                    "verified": row.verified,
                    "steps": row.steps,
                    "latency_ms": round(row.latency_ms, 2),
                    "stop_reason": row.stop_reason,
                    "answer": row.answer,
                }
                for row in self.rows
            ],
        }


def render_markdown(report: EvalReport) -> str:
    lines = [
        "# Evaluation report",
        "",
        report.disclaimer,
    ]
    if report.partial:
        lines.extend(
            [
                "",
                "Partial: this run was interrupted. Tasks below are the ones that finished.",
            ]
        )
    lines.extend(
        [
            "",
            "| Metric | Value |",
            "| --- | --- |",
        f"| Mode | {'mock' if report.provider == 'mock' else 'hosted'} |",
        f"| Provider | {report.provider} |",
        f"| Model | {report.model} |",
        f"| Search | {report.search} |",
        f"| Generated | {report.generated_at} |",
        f"| Tasks | {report.total} |",
        f"| Passed | {report.passed} |",
        f"| Rate limited | {report.rate_limited} |",
        f"| Scored | {report.scored} |",
        f"| Success rate | {_percent(report.success_rate)} |",
        f"| Average steps | {report.average_steps:.2f} |",
        f"| Average latency | {report.average_latency_ms:.2f} ms |",
        "",
        "| Category | Passed | Total |",
        "| --- | --- | --- |",
        ]
    )
    for name, bucket in _categories(report.rows).items():
        lines.append(f"| {name} | {bucket['passed']} | {bucket['total']} |")
    lines.extend(
        [
            "",
            "| id | category | pass | steps | latency_ms | answer |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in report.rows:
        if row.passed:
            mark = "yes"
        elif row.stop_reason == "rate_limited":
            mark = "rate limit"
        else:
            mark = "no"
        lines.append(
            "| "
            + " | ".join(
                [
                    row.id,
                    row.category,
                    mark,
                    str(row.steps),
                    f"{row.latency_ms:.2f}",
                    _cell(row.answer),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def write_report(report: EvalReport, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / "eval_results.md"
    json_path = output_dir / "eval_results.json"
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    return markdown_path, json_path


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def disclaimer_for(provider: str, search: str) -> str:
    if provider == "mock":
        return (
            "Mock mode uses the deterministic policy and the offline search fixture. "
            "These numbers are not measurements of Groq, Gemini, or OpenAI."
        )
        return (
            f"Hosted run with provider {provider} and search mode {search}. "
            "Results depend on the model and, when search is live, on the public web. "
            "Success rate ignores tasks that stayed rate-limited after retries; those are not model failures."
        )


def _categories(rows: list[TaskResult]) -> dict[str, dict[str, int]]:
    buckets: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = buckets.setdefault(row.category, {"passed": 0, "total": 0})
        bucket["total"] += 1
        bucket["passed"] += int(row.passed)
    return buckets


def _percent(rate: float) -> str:
    percent = rate * 100
    if abs(percent - round(percent)) < 1e-9:
        return f"{percent:.0f}%"
    return f"{percent:.1f}%"


def _cell(text: str, limit: int = 72) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) > limit:
        cleaned = cleaned[: limit - 1] + "…"
    return cleaned.replace("|", "\\|")
