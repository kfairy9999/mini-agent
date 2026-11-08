"""Keep the suite offline even if the developer shell has a provider key."""

from __future__ import annotations

import pytest

_LLM_ENV = (
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_PROVIDER",
    "LLM_TIMEOUT",
    "LLM_MAX_TOKENS",
    "LLM_MIN_INTERVAL",
    "LLM_MAX_RETRIES",
    "LLM_MAX_WAIT",
    "GROQ_API_KEY",
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "MINI_AGENT_SEARCH",
)


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _LLM_ENV:
        monkeypatch.delenv(key, raising=False)
