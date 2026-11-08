from __future__ import annotations

from pathlib import Path

import pytest

from mini_agent.config import (
    GEMINI_BASE_URL,
    GEMINI_MODEL,
    GROQ_BASE_URL,
    GROQ_MODEL,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    ConfigError,
    load_config,
    load_dotenv_file,
    resolve_search,
)


def test_no_key_selects_the_mock() -> None:
    config = load_config()
    assert config.provider == "mock"
    assert config.model == "mock"
    assert config.api_key == ""


def test_groq_key_fallback_and_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    config = load_config()
    assert config.provider == "groq"
    assert config.api_key == "gsk-test"
    assert config.base_url == GROQ_BASE_URL
    assert GROQ_MODEL == "qwen/qwen3.8-27b"
    assert config.model == GROQ_MODEL


def test_llm_api_key_wins_over_the_groq_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "from-groq")
    monkeypatch.setenv("LLM_API_KEY", "from-llm")
    monkeypatch.setenv("LLM_MODEL", "llama-3.1-8b-instant")
    config = load_config("groq")
    assert config.api_key == "from-llm"
    assert config.model == "llama-3.1-8b-instant"
    assert config.base_url == GROQ_BASE_URL


def test_explicit_provider_without_a_key_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ConfigError, match="GROQ_API_KEY"):
        load_config("groq")


def test_gemini_and_openai_presets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gem-test")
    gemini = load_config("gemini")
    assert gemini.base_url == GEMINI_BASE_URL
    assert gemini.model == GEMINI_MODEL
    assert gemini.api_key == "gem-test"

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    openai = load_config("openai")
    assert openai.base_url == OPENAI_BASE_URL
    assert openai.model == OPENAI_MODEL
    assert openai.api_key == "sk-test"


def test_base_url_infers_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "gem-test")
    monkeypatch.setenv("LLM_BASE_URL", GEMINI_BASE_URL)
    config = load_config()
    assert config.provider == "gemini"
    assert config.model == GEMINI_MODEL


def test_dotenv_fills_missing_keys_only(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text('export LLM_MODEL="from-file"\nGROQ_API_KEY=from-file\n# comment\n\n', encoding="utf-8")
    environ = {"LLM_MODEL": "from-env"}
    load_dotenv_file(path, environ)
    assert environ["LLM_MODEL"] == "from-env"
    assert environ["GROQ_API_KEY"] == "from-file"
    load_dotenv_file(tmp_path / "missing.env", environ)


def test_search_mode_follows_the_provider() -> None:
    assert resolve_search("auto", provider="mock") == "fixture"
    assert resolve_search("auto", provider="groq") == "live"
    assert resolve_search("live", provider="mock") == "live"


def test_invalid_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_TIMEOUT", "fast")
    with pytest.raises(ConfigError, match="LLM_TIMEOUT"):
        load_config("mock")
