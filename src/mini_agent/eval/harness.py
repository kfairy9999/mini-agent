"""Run the fixed task set and score the checked answers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

from mini_agent.agent import DEFAULT_MAX_STEPS
from mini_agent.types import AgentResult
from mini_agent.config import ConfigError, load_config, resolve_search
from mini_agent.eval.report import (
    EvalReport,
    TaskResult,
    disclaimer_for,
    utc_now,
    write_report,
)
from mini_agent.eval.tasks import Task, default_tasks
from mini_agent.factory import build_agent
from mini_agent.llm import LLMError, RateLimitError

# One client is shared, and tasks run one at a time. After a 429 the client
# spaces later calls. A task that still exhausts those retries is tried again
# here before it is recorded as rate-limited rather than as a wrong answer.
TASK_RATE_LIMIT_ATTEMPTS = 3


class TaskRunner(Protocol):
    def run(self, question: str) -> AgentResult:
        """Answer one question."""


def run_evaluation(
    *,
    provider: str = "mock",
    search: str = "auto",
    max_steps: int = DEFAULT_MAX_STEPS,
    tasks: tuple[Task, ...] | None = None,
    output_dir: Path | None = None,
    agent: TaskRunner | None = None,
) -> EvalReport:
    """Evaluate ``tasks`` (the built-in set by default).

    The default provider is mock, on purpose. A key in the environment does
    not turn this into a hosted run; pass ``provider="groq"`` (or another
    preset) for that. A task passes only when the loop completes its
    self-check and the answer satisfies the checker.
    """

    selected = tasks if tasks is not None else default_tasks()
    explicit = None if provider == "auto" else provider
    config = load_config(explicit)
    search_mode = resolve_search(search, provider=config.provider)
    if agent is None:
        agent = build_agent(provider=provider, max_steps=max_steps, search=search_mode, config=config)
    rows: list[TaskResult] = []
    for task in selected:
        rows.append(_run_task(agent, task))
    report = EvalReport(
        provider=config.provider,
        model=config.model,
        search=search_mode,
        disclaimer=disclaimer_for(config.provider, search_mode),
        generated_at=utc_now(),
        rows=rows,
    )
    if output_dir is not None:
        write_report(report, output_dir)
    return report


def _run_task(agent: TaskRunner, task: Task) -> TaskResult:
    started = time.perf_counter()
    last_limit: RateLimitError | None = None
    for _attempt in range(TASK_RATE_LIMIT_ATTEMPTS):
        try:
            result = agent.run(task.question)
        except RateLimitError as exc:
            last_limit = exc
            continue
        except (LLMError, ConfigError, ValueError) as exc:
            return _row(
                task,
                passed=False,
                verified=False,
                steps=0,
                latency_ms=(time.perf_counter() - started) * 1000,
                stop_reason="error",
                answer=str(exc),
            )
        return _row(
            task,
            passed=bool(result.verified and task.score(result.answer)),
            verified=result.verified,
            steps=result.model_calls,
            latency_ms=(time.perf_counter() - started) * 1000,
            stop_reason=result.stop_reason,
            answer=result.answer,
        )
    message = str(last_limit) if last_limit is not None else "rate limit"
    return _row(
        task,
        passed=False,
        verified=False,
        steps=0,
        latency_ms=(time.perf_counter() - started) * 1000,
        stop_reason="rate_limited",
        answer=f"Rate limited after retries; not a model failure. {message}",
    )


def _row(
    task: Task,
    *,
    passed: bool,
    verified: bool,
    steps: int,
    latency_ms: float,
    stop_reason: str,
    answer: str,
) -> TaskResult:
    return TaskResult(
        id=task.id,
        category=task.category,
        question=task.question,
        expectation=task.expectation,
        passed=passed,
        verified=verified,
        steps=steps,
        latency_ms=latency_ms,
        stop_reason=stop_reason,
        answer=answer,
    )
