"""``mini-agent-eval`` and ``python -m mini_agent.eval``."""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

from mini_agent.agent import DEFAULT_MAX_STEPS
from mini_agent.config import ConfigError, load_dotenv_file
from mini_agent.eval.harness import run_evaluation
from mini_agent.eval.report import render_markdown
from mini_agent.llm import LLMError


def _interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


def app() -> None:
    load_dotenv_file(Path.cwd() / ".env")
    raise SystemExit(main())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mini-agent-eval",
        description=(
            "Score the fixed task set. Defaults to the deterministic mock and the "
            "offline search fixture, even if an API key is set."
        ),
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "mock", "groq", "gemini", "openai", "custom"],
        default="mock",
        help="Model provider. Default is mock so the command stays offline.",
    )
    parser.add_argument(
        "--search",
        choices=["auto", "live", "fixture"],
        default="auto",
        help="auto is the fixture corpus for mock and live ddgs otherwise.",
    )
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory for eval_results.md and eval_results.json (default: reports).",
    )
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)

    def progress(line: str) -> None:
        print(line, file=sys.stderr, flush=True)

    previous = signal.signal(signal.SIGTERM, _interrupt)
    try:
        report = run_evaluation(
            provider=args.provider,
            search=args.search,
            max_steps=args.max_steps,
            output_dir=output_dir,
            on_progress=progress,
        )
    except KeyboardInterrupt:
        print(
            f"Interrupted. Partial results, if any task finished, are in {output_dir}.",
            file=sys.stderr,
        )
        return 130
    except (ConfigError, LLMError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)
    sys.stdout.write(render_markdown(report))
    sys.stdout.flush()
    print(f"Wrote {output_dir / 'eval_results.md'}", file=sys.stderr)
    print(f"Wrote {output_dir / 'eval_results.json'}", file=sys.stderr)
    return 0 if report.passed == report.total else 1


if __name__ == "__main__":
    app()
