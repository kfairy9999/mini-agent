"""Run model-written Python in a subprocess. Local guard, not a jail."""

from __future__ import annotations

import ast
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from typing import Any

from mini_agent.tools.registry import Tool

DEFAULT_TIMEOUT_SECONDS = 5.0
_MAX_SOURCE_CHARS = 4_000
_MAX_OUTPUT_CHARS = 4_000
_DENIED_MODULES = {
    "ctypes",
    "multiprocessing",
    "os",
    "pathlib",
    "pty",
    "shutil",
    "signal",
    "socket",
    "subprocess",
    "sys",
}
_DENIED_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "memoryview",
    "open",
    "setattr",
    "vars",
}


class SandboxError(ValueError):
    """The snippet violates the local execution policy."""


def validate_code(code: str) -> None:
    """Reject snippets that are empty, huge, or obviously hostile."""

    if not isinstance(code, str) or not code.strip():
        raise SandboxError("code is required")
    if len(code) > _MAX_SOURCE_CHARS:
        raise SandboxError("code is too long")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise SandboxError(f"invalid Python syntax: {exc.msg}") from exc
    _Policy().visit(tree)


def run_python(code: str, *, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> str:
    """Validate ``code`` and run it. The return value is an observation."""

    try:
        validate_code(code)
    except SandboxError as exc:
        return f"Error: {exc}"
    if timeout <= 0:
        return "Error: timeout must be positive."

    with tempfile.TemporaryDirectory(prefix="mini-agent-") as directory:
        script = os.path.join(directory, "snippet.py")
        with open(script, "w", encoding="utf-8") as handle:
            handle.write(code)
        command = _command(script)
        env = {
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONNOUSERSITE": "1",
        }
        try:
            returncode, stdout, stderr = _communicate(command, timeout=timeout, cwd=directory, env=env)
        except TimeoutError:
            return f"Error: execution timed out after {timeout:g}s."
        except OSError as exc:
            return f"Error: could not start Python ({exc})."

    stdout = _clip(stdout)
    stderr = _clip(stderr)
    if returncode != 0:
        detail = stderr.strip() or stdout.strip() or f"exit code {returncode}"
        return f"Error: python exited {returncode}: {detail}"
    return f"exit_code: 0\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"


def python_tool(*, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> Tool:
    def run(arguments: dict[str, Any]) -> str:
        code = arguments.get("code")
        if not isinstance(code, str):
            return "Error: code is required."
        return run_python(code, timeout=timeout)

    return Tool(
        name="python",
        description=(
            "Run a Python 3 snippet in a local subprocess with a timeout. "
            "Use it for algorithms, counting, and string work. "
            "Print the final result to stdout. "
            "Do not import os, subprocess, socket, or shutil, and do not open files."
        ),
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python source. Print the final result to stdout.",
                }
            },
            "required": ["code"],
            "additionalProperties": False,
        },
        handler=run,
    )


class _Policy(ast.NodeVisitor):
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            if root in _DENIED_MODULES:
                raise SandboxError(f"import of {root} is not allowed")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        root = (node.module or "").split(".", 1)[0]
        if root in _DENIED_MODULES:
            raise SandboxError(f"import of {root} is not allowed")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node.func)
        if name in _DENIED_CALLS:
            raise SandboxError(f"call to {name} is not allowed")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__"):
            raise SandboxError("dunder attribute access is not allowed")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("__"):
            raise SandboxError(f"name {node.id} is not allowed")
        self.generic_visit(node)


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _command(script: str) -> list[str]:
    bash = shutil.which("bash")
    if bash is None:
        return [sys.executable, "-I", script]
    # CPU-seconds, file-size (512-byte blocks), and address space (KiB).
    limits = "ulimit -t 3 && ulimit -f 2048 && ulimit -v 1048576 && exec \"$0\" \"$@\""
    return [bash, "-c", limits, sys.executable, "-I", script]


def _communicate(
    command: list[str],
    *,
    timeout: float,
    cwd: str,
    env: dict[str, str],
) -> tuple[int, str, str]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill(process)
        try:
            process.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
        raise TimeoutError(timeout) from None
    return process.returncode or 0, stdout or "", stderr or ""


def _kill(process: subprocess.Popen[str]) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
    else:
        process.kill()


def _clip(text: str) -> str:
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    return text[:_MAX_OUTPUT_CHARS] + "\n...[truncated]"
