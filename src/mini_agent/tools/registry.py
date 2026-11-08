"""A name-to-tool map. Register a tool and the model can call it."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

Handler = Callable[[dict[str, Any]], str]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Handler

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def run(self, arguments: dict[str, Any]) -> str:
        return self.handler(arguments)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not tool.name or not tool.name.replace("_", "").isalnum():
            raise ValueError(f"invalid tool name {tool.name!r}")
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(name) from exc

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.openai_schema() for tool in self._tools.values()]

    def call(self, name: str, arguments: dict[str, Any] | None) -> str:
        """Run a tool. Failures become observations the model can read."""

        tool = self._tools.get(name)
        if tool is None:
            return f"Error: unknown tool {name!r}."
        if not isinstance(arguments, dict):
            return "Error: tool arguments must be a JSON object."
        try:
            output = tool.handler(dict(arguments))
        except Exception as exc:
            # A tool bug has to become an observation. The loop stays up.
            return f"Error: {type(exc).__name__}: {exc}"
        if not isinstance(output, str):
            return "Error: tool returned a non-string result."
        return output
