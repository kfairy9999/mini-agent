"""Resolve provider, base URL, model, and API key from the environment."""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from dataclasses import dataclass
from pathlib import Path

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# llama-3.3-70b-versatile returns 404 on current Groq accounts.
# qwen/qwen3.8-27b is on the live model list and tool-calls cleanly.
GROQ_MODEL = "qwen/qwen3.8-27b"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL = "gemini-3.8-flash"
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MODEL = "gpt-4.1-mini"

DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_TOKENS = 1024
DEFAULT_MIN_INTERVAL = 0.0
DEFAULT_MAX_RETRIES = 8
DEFAULT_MAX_WAIT = 60.0


class ConfigError(RuntimeError):
    """The process is not configured for the provider it was asked to use."""


@dataclass(frozen=True)
class Preset:
    base_url: str
    model: str
    key_env: str


PRESETS: dict[str, Preset] = {
    "groq": Preset(GROQ_BASE_URL, GROQ_MODEL, "GROQ_API_KEY"),
    "gemini": Preset(GEMINI_BASE_URL, GEMINI_MODEL, "GEMINI_API_KEY"),
    "openai": Preset(OPENAI_BASE_URL, OPENAI_MODEL, "OPENAI_API_KEY"),
}


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    base_url: str
    model: str
    api_key: str
    timeout: float
    max_tokens: int
    min_interval: float = DEFAULT_MIN_INTERVAL
    max_retries: int = DEFAULT_MAX_RETRIES
    max_wait: float = DEFAULT_MAX_WAIT


def load_dotenv_file(
    path: Path,
    environ: MutableMapping[str, str] | None = None,
) -> None:
    """Set missing keys from a ``.env`` file. Existing variables win.

    The CLI calls this with the working directory's ``.env``. Tests and
    ``load_config`` do not, so a local key file cannot change the suite.
    """

    if not path.is_file():
        return
    target = os.environ if environ is None else environ
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in target:
            target[key] = value


def load_config(provider: str | None = None) -> LLMConfig:
    """Resolve a config.

    ``provider=None`` (and ``LLM_PROVIDER`` unset) is automatic: an explicit
    key selects that host, and no key selects the mock. Pass ``mock``,
    ``groq``, ``gemini``, ``openai``, or ``custom`` to force one.
    """

    requested = (provider or os.getenv("LLM_PROVIDER") or "auto").strip().lower()
    if requested == "auto":
        requested = _auto_provider()
    if requested == "mock":
        return LLMConfig(
            provider="mock",
            base_url="",
            model="mock",
            api_key="",
            timeout=_timeout(),
            max_tokens=_max_tokens(),
            min_interval=_min_interval(),
            max_retries=_max_retries(),
            max_wait=_max_wait(),
        )
    if requested not in {"groq", "gemini", "openai", "custom"}:
        known = ", ".join(["auto", "mock", *PRESETS, "custom"])
        raise ConfigError(f"Unknown LLM provider {requested!r}. Expected one of: {known}.")
    if requested == "custom":
        base_url = os.getenv("LLM_BASE_URL", "").strip()
        model = os.getenv("LLM_MODEL", "").strip()
        if not base_url or not model:
            raise ConfigError("Custom provider requires LLM_BASE_URL and LLM_MODEL.")
        key = os.getenv("LLM_API_KEY", "").strip()
        if not key:
            raise ConfigError("Custom provider requires LLM_API_KEY.")
        return LLMConfig(
            requested,
            base_url,
            model,
            key,
            _timeout(),
            _max_tokens(),
            _min_interval(),
            _max_retries(),
            _max_wait(),
        )

    preset = PRESETS[requested]
    base_url = os.getenv("LLM_BASE_URL", "").strip() or preset.base_url
    model = os.getenv("LLM_MODEL", "").strip() or preset.model
    key = os.getenv("LLM_API_KEY", "").strip() or os.getenv(preset.key_env, "").strip()
    if not key:
        raise ConfigError(
            f"{requested} is selected but no API key is set. "
            f"Export {preset.key_env} or LLM_API_KEY. "
            "To run offline, use --provider mock."
        )
    return LLMConfig(
        requested,
        base_url,
        model,
        key,
        _timeout(),
        _max_tokens(),
        _min_interval(),
        _max_retries(),
        _max_wait(),
    )


def resolve_search(requested: str, *, provider: str) -> str:
    """Pick live ddgs or the offline fixture corpus."""

    choice = requested.strip().lower()
    if choice == "auto":
        env = os.getenv("MINI_AGENT_SEARCH", "").strip().lower()
        if env in {"live", "fixture"}:
            choice = env
        else:
            choice = "fixture" if provider == "mock" else "live"
    if choice not in {"live", "fixture"}:
        raise ConfigError("Search mode must be auto, live, or fixture.")
    return choice


def _auto_provider() -> str:
    if os.getenv("LLM_API_KEY", "").strip():
        return _provider_from_base_url(os.getenv("LLM_BASE_URL", ""))
    if os.getenv("GROQ_API_KEY", "").strip():
        return "groq"
    if os.getenv("GEMINI_API_KEY", "").strip():
        return "gemini"
    if os.getenv("OPENAI_API_KEY", "").strip():
        return "openai"
    return "mock"


def _provider_from_base_url(base_url: str) -> str:
    lowered = base_url.lower()
    if "generativelanguage.googleapis.com" in lowered:
        return "gemini"
    if "api.openai.com" in lowered:
        return "openai"
    if base_url.strip() and "groq.com" not in lowered:
        return "custom"
    return "groq"


def _timeout() -> float:
    raw = os.getenv("LLM_TIMEOUT", "").strip()
    if not raw:
        return DEFAULT_TIMEOUT
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError("LLM_TIMEOUT must be a number of seconds.") from exc
    if value <= 0:
        raise ConfigError("LLM_TIMEOUT must be positive.")
    return value


def _min_interval() -> float:
    raw = os.getenv("LLM_MIN_INTERVAL", "").strip()
    if not raw:
        return DEFAULT_MIN_INTERVAL
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError("LLM_MIN_INTERVAL must be a number of seconds.") from exc
    if value < 0:
        raise ConfigError("LLM_MIN_INTERVAL must be zero or positive.")
    return value


def _max_retries() -> int:
    raw = os.getenv("LLM_MAX_RETRIES", "").strip()
    if not raw:
        return DEFAULT_MAX_RETRIES
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError("LLM_MAX_RETRIES must be an integer.") from exc
    if value < 0:
        raise ConfigError("LLM_MAX_RETRIES must be zero or positive.")
    return value


def _max_wait() -> float:
    raw = os.getenv("LLM_MAX_WAIT", "").strip()
    if not raw:
        return DEFAULT_MAX_WAIT
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError("LLM_MAX_WAIT must be a number of seconds.") from exc
    if value <= 0:
        raise ConfigError("LLM_MAX_WAIT must be positive.")
    return value


def _max_tokens() -> int:
    raw = os.getenv("LLM_MAX_TOKENS", "").strip()
    if not raw:
        return DEFAULT_MAX_TOKENS
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError("LLM_MAX_TOKENS must be an integer.") from exc
    if value < 1:
        raise ConfigError("LLM_MAX_TOKENS must be positive.")
    return value
