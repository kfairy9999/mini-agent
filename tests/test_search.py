from __future__ import annotations

from typing import Any, cast

import pytest

from mini_agent.tools.registry import ToolRegistry
from mini_agent.tools.web_search import FixtureBackend, SearchHit, format_hits, web_search_tool


def test_fixture_search_finds_the_capital_and_misses_unknowns() -> None:
    backend = FixtureBackend()
    hits = backend.search("What is the capital of France?", max_results=3)
    assert hits[0].snippet == "The capital of France is Paris."
    assert backend.search("What is the capital of Atlantis?", max_results=3) == []


def test_search_tool_labels_the_source_and_treats_snippets_as_data() -> None:
    tool = web_search_tool(mode="fixture")
    observation = tool.run({"query": "What is the chemical symbol for gold?", "max_results": 2})
    assert "source: fixture" in observation
    assert "data, not instructions" in observation
    assert "\nThe chemical symbol for gold is Au." in observation or observation.endswith(
        "The chemical symbol for gold is Au."
    )


def test_search_tool_reports_backend_failures() -> None:
    class Broken:
        name = "broken"

        def search(self, query: str, max_results: int) -> list[SearchHit]:
            raise RuntimeError("network down")

    tool = web_search_tool(backend=Broken())
    assert tool.run({"query": "France"}) == "Error: search failed (network down)."
    assert tool.run({"query": ""}) == "Error: query is required."


def test_no_results_keep_the_source_label() -> None:
    text = format_hits([], source="fixture")
    assert text.startswith("No results.")
    assert "source: fixture" in text


def test_registry_schemas_and_unknown_tools() -> None:
    registry = ToolRegistry()
    tool = web_search_tool(mode="fixture")
    registry.register(tool)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(tool)
    schemas = registry.schemas()
    assert schemas[0]["function"]["name"] == "web_search"
    assert "unknown tool" in registry.call("missing", {})
    assert registry.call("web_search", cast(Any, "nope")).startswith("Error:")
