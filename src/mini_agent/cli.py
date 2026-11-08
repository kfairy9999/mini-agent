"""Command line for one question and its trace."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console

from mini_agent import __version__
from mini_agent.agent import DEFAULT_MAX_STEPS
from mini_agent.config import ConfigError, load_dotenv_file
from mini_agent.factory import build_agent
from mini_agent.llm import LLMError
from mini_agent.trace import render_result


def app() -> None:
    load_dotenv_file(Path.cwd() / ".env")
    raise SystemExit(main())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mini-agent",
        description="Ask a question. The agent plans, calls tools, checks its draft, and prints the trace.",
    )
    parser.add_argument("question", help="Question to answer")
    parser.add_argument(
        "--provider",
        choices=["auto", "mock", "groq", "gemini", "openai", "custom"],
        default="auto",
        help="Model provider. auto uses a key if one is set, otherwise the mock (default: auto).",
    )
    parser.add_argument("--mock", action="store_true", help="Force the deterministic mock, even if a key is set.")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="Maximum model calls (default: 8).")
    parser.add_argument(
        "--search",
        choices=["auto", "live", "fixture"],
        default="auto",
        help="Web search backend. auto is fixture for the mock and live ddgs otherwise.",
    )
    parser.add_argument("--json", action="store_true", help="Print the result as JSON instead of a trace.")
    parser.add_argument("--version", action="version", version=f"mini-agent {__version__}")
    args = parser.parse_args(argv)

    question = args.question.strip()
    if not question:
        print("Provide a question.", file=sys.stderr)
        return 2

    provider = "mock" if args.mock else args.provider
    try:
        agent = build_agent(provider=provider, max_steps=args.max_steps, search=args.search)
        if provider == "auto" and agent.llm.model == "mock":
            print("No hosted provider selected; using the deterministic mock.", file=sys.stderr)
        result = agent.run(question)
    except (ConfigError, LLMError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.json:
        json.dump(result.to_dict(), sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        console = Console(file=sys.stdout, markup=False, highlight=False)
        render_result(console, question, result)
    return 0 if result.verified else 2


if __name__ == "__main__":
    app()
