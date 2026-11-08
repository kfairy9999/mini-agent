"""Render a run as a readable terminal trace."""

from __future__ import annotations

import json

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from mini_agent.types import AgentResult, Observation, Step

_OBSERVATION_LIMIT = 1_500


def render_result(console: Console, question: str, result: AgentResult) -> None:
    title = Text.assemble(("mini-agent", "bold"), "  ", (result.model, "cyan"))
    console.print(Panel(Text(question), title=title, border_style="cyan"))
    for step in result.steps:
        _render_step(console, step)
    border = "green" if result.verified else "yellow"
    label = "answer" if result.verified else "unverified answer"
    console.print(Panel(Text(result.answer or "(empty)"), title=label, border_style=border))
    footer = f"{result.model_calls} steps · {_format_ms(result.latency_ms)} · {result.stop_reason}"
    if not result.verified:
        footer += " · self-check did not finish"
    console.print(Text(footer, style="dim"))


def _render_step(console: Console, step: Step) -> None:
    heading = Text()
    if step.kind == "limit":
        heading.append("stopped", style="bold yellow")
    else:
        heading.append(f"step {step.index}  ", style="dim")
        heading.append(step.kind, style="bold")
    if step.verdict:
        color = "magenta" if step.verdict == "revise" else "green"
        heading.append(f"  {step.verdict}", style=color)
    console.print(heading)
    if step.text:
        console.print(Text(step.text))
    for observation in step.observations:
        _render_observation(console, observation)
    console.print()


def _render_observation(console: Console, observation: Observation) -> None:
    console.print(Text(f"call {observation.name}", style="bold"))
    if observation.name == "python" and isinstance(observation.arguments.get("code"), str):
        console.print(Text(observation.arguments["code"].rstrip()))
    else:
        console.print(Text(json.dumps(observation.arguments, ensure_ascii=False)))
    console.print(Text("observe", style="dim"))
    console.print(Text(_clip(observation.output)))


def _clip(text: str) -> str:
    if len(text) <= _OBSERVATION_LIMIT:
        return text
    return text[:_OBSERVATION_LIMIT] + "\n...[truncated]"


def _format_ms(value: float) -> str:
    if value < 10:
        return f"{value:.1f} ms"
    return f"{value:.0f} ms"
