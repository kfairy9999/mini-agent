from __future__ import annotations

import subprocess

import pytest

from mini_agent.tools.python_exec import SandboxError, run_python, validate_code


def test_python_prints_the_last_result() -> None:
    observation = run_python("print(sum(i * i for i in range(1, 11)))\n")
    assert observation.startswith("exit_code: 0")
    assert "385" in observation


def test_policy_rejects_imports_files_and_dunder_access() -> None:
    with pytest.raises(SandboxError, match="os"):
        validate_code("import os\nprint(os.getcwd())\n")
    with pytest.raises(SandboxError, match="open"):
        validate_code("open('secret.txt')\n")
    with pytest.raises(SandboxError, match="dunder"):
        validate_code("print((1).__class__)\n")
    assert run_python("import subprocess\nprint('no')\n").startswith("Error:")


def test_python_timeout() -> None:
    observation = run_python("import time\ntime.sleep(5)\nprint('late')\n", timeout=0.4)
    assert "timed out" in observation.lower()


def test_python_subprocess_gets_a_clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeProcess:
        pid = 4321
        returncode = 0

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            captured["timeout"] = timeout
            return "ok\n", ""

    def fake_popen(command: list[str], **kwargs: object) -> FakeProcess:
        captured["command"] = command
        captured["env"] = kwargs["env"]
        captured["start_new_session"] = kwargs["start_new_session"]
        return FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setenv("GROQ_API_KEY", "secret-key")
    observation = run_python("print('ok')\n", timeout=2)
    env = captured["env"]
    assert isinstance(env, dict)
    assert "GROQ_API_KEY" not in env
    assert "secret-key" not in env.values()
    assert env["PYTHONNOUSERSITE"] == "1"
    command = captured["command"]
    assert isinstance(command, list)
    assert "-I" in command
    assert captured["start_new_session"] is True
    assert captured["timeout"] == 2
    assert "ok" in observation
