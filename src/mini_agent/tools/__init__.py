"""Built-in tools. Importing this package does not import the search client."""

from __future__ import annotations

from mini_agent.tools.registry import Tool, ToolRegistry

__all__ = ["Tool", "ToolRegistry", "build_registry"]


def build_registry(search: str = "live") -> ToolRegistry:
    """Register the calculator, the Python sandbox, and web search."""

    from mini_agent.tools.calculator import calculator_tool
    from mini_agent.tools.python_exec import python_tool
    from mini_agent.tools.web_search import web_search_tool

    if search not in {"live", "fixture"}:
        raise ValueError("search must be 'live' or 'fixture'")
    registry = ToolRegistry()
    registry.register(calculator_tool())
    registry.register(python_tool())
    registry.register(web_search_tool(mode=search))
    return registry
