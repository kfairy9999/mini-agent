"""Wire a config, a model client, and the default tools into an agent."""

from __future__ import annotations

from mini_agent.agent import DEFAULT_MAX_STEPS, Agent
from mini_agent.config import LLMConfig, load_config, resolve_search
from mini_agent.llm import LLMClient, OpenAICompatibleClient
from mini_agent.mock_llm import MockLLM
from mini_agent.tools import ToolRegistry, build_registry


def build_agent(
    *,
    provider: str = "auto",
    max_steps: int = DEFAULT_MAX_STEPS,
    search: str = "auto",
    config: LLMConfig | None = None,
    llm: LLMClient | None = None,
    registry: ToolRegistry | None = None,
) -> Agent:
    resolved = config
    if llm is None:
        explicit = None if provider == "auto" else provider
        resolved = config or load_config(explicit)
        llm = MockLLM() if resolved.provider == "mock" else OpenAICompatibleClient(resolved)
    if registry is None:
        provider_name = resolved.provider if resolved is not None else ("mock" if isinstance(llm, MockLLM) else "custom")
        registry = build_registry(resolve_search(search, provider=provider_name))
    return Agent(llm=llm, registry=registry, max_steps=max_steps)
