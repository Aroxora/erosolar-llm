"""Tool framework: a registry the agent exposes to the model as OpenAI
function-calling specs, with safe execution (exceptions become ToolResults)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .types import ToolResult


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict       # JSON Schema (an "object" schema)
    func: Callable
    terminal: bool = False  # if True, a call ends the agent loop (e.g. `finish`)

    def to_openai(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def run(self, args: dict) -> ToolResult:
        try:
            out = self.func(**(args or {}))
        except TypeError as e:
            return ToolResult(ok=False, error=f"bad arguments for {self.name}: {e}")
        except Exception as e:  # noqa: BLE001 — surface tool errors to the model
            return ToolResult(ok=False, error=f"{type(e).__name__}: {e}")
        if isinstance(out, ToolResult):
            return out
        return ToolResult(ok=True, content="" if out is None else str(out))


class ToolRegistry:
    def __init__(self):
        self._tools: dict = {}

    def register(self, tool: Tool) -> Tool:
        self._tools[tool.name] = tool
        return tool

    def add(self, name, description, parameters, func, terminal=False) -> Tool:
        return self.register(Tool(name, description, parameters, func, terminal))

    def get(self, name):
        return self._tools.get(name)

    def names(self) -> list:
        return list(self._tools)

    def is_terminal(self, name) -> bool:
        t = self.get(name)
        return bool(t and t.terminal)

    def openai_tools(self, allowlist=None) -> list:
        tools = list(self._tools.values())
        if allowlist:
            allow = set(allowlist) | {n for n, t in self._tools.items() if t.terminal}
            tools = [t for t in tools if t.name in allow]
        return [t.to_openai() for t in tools]

    def run(self, name, args) -> ToolResult:
        t = self.get(name)
        if not t:
            return ToolResult(ok=False, error=f"unknown tool {name!r}")
        return t.run(args or {})
