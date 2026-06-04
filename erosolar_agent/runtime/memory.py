"""Working memory: the live transcript plus durable state that survives context
compaction (notes + plan + scratchpad are re-injected into the system message
every turn, so they are never lost when older transcript is summarized away)."""

from __future__ import annotations

import json
from pathlib import Path

from .tools import Tool
from .types import ToolResult, system_msg


class WorkingMemory:
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.transcript: list = []   # OpenAI-format dicts (no system msg here)
        self.notes: list = []        # durable facts/decisions
        self.plan: list = []         # subgoals
        self.scratch: dict = {}      # arbitrary durable key/values

    def add(self, msg: dict) -> None:
        self.transcript.append(msg)

    def remember(self, note: str) -> int:
        self.notes.append(note)
        return len(self.notes)

    def notes_text(self) -> str:
        return "\n".join(f"- {n}" for n in self.notes) or "(none yet)"

    def plan_text(self) -> str:
        return "\n".join(f"{i + 1}. {s}" for i, s in enumerate(self.plan)) or "(no plan)"

    def system_block(self) -> str:
        blocks = [self.system_prompt]
        if self.plan:
            blocks.append("CURRENT PLAN:\n" + self.plan_text())
        if self.notes:
            blocks.append("DURABLE NOTES (persist across compaction):\n" + self.notes_text())
        if self.scratch:
            blocks.append("SCRATCHPAD:\n" + json.dumps(self.scratch, indent=2))
        return "\n\n".join(blocks)

    def messages(self) -> list:
        return [system_msg(self.system_block())] + self.transcript

    def save(self, path) -> None:
        Path(path).write_text(json.dumps({
            "transcript": self.transcript, "notes": self.notes,
            "plan": self.plan, "scratch": self.scratch,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path) -> None:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        self.transcript = d.get("transcript", [])
        self.notes = d.get("notes", [])
        self.plan = d.get("plan", [])
        self.scratch = d.get("scratch", {})


def memory_tools(memory: WorkingMemory) -> list:
    """Tools that let the model manage its own long-horizon memory."""

    def remember(note: str) -> ToolResult:
        n = memory.remember(note)
        return ToolResult(ok=True, content=f"noted (#{n})")

    def recall() -> ToolResult:
        return ToolResult(ok=True, content=memory.notes_text())

    def set_scratch(key: str, value: str) -> ToolResult:
        memory.scratch[key] = value
        return ToolResult(ok=True, content=f"scratch[{key}] set")

    return [
        Tool("remember", "Save a durable fact/decision that must survive context compaction.",
             {"type": "object", "properties": {"note": {"type": "string"}}, "required": ["note"]},
             remember),
        Tool("recall", "List your durable notes.",
             {"type": "object", "properties": {}}, recall),
        Tool("set_scratch", "Store a durable key/value (overwrites).",
             {"type": "object", "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
              "required": ["key", "value"]}, set_scratch),
    ]
