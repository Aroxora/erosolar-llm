"""Core message helpers and trace types for the agent runtime.

The transcript is stored as a list of plain OpenAI-format dicts so it can be
passed to the chat API verbatim (avoids tool_call serialization bugs). Trace
objects are for human/debug inspection only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


# --- OpenAI chat-message constructors -------------------------------------

def system_msg(content: str) -> dict:
    return {"role": "system", "content": content}


def user_msg(content: str) -> dict:
    return {"role": "user", "content": content}


def assistant_msg(content, tool_calls=None) -> dict:
    m: dict = {"role": "assistant", "content": content or ""}
    if tool_calls:
        m["tool_calls"] = tool_calls
    return m


def tool_msg(tool_call_id: str, content: str, name: str = "") -> dict:
    m: dict = {"role": "tool", "tool_call_id": tool_call_id, "content": content}
    if name:
        m["name"] = name
    return m


# --- results / trace -------------------------------------------------------

@dataclass
class ToolResult:
    ok: bool
    content: str = ""
    error: str = ""

    def render(self) -> str:
        return self.content if self.ok else f"ERROR: {self.error}"


@dataclass
class Step:
    index: int
    kind: str  # plan | think | tool | reflect | compact | final
    tool: str = ""
    args: dict = field(default_factory=dict)
    output: str = ""
    ts: float = field(default_factory=time.time)


@dataclass
class RunResult:
    task: str
    answer: str = ""
    success: bool = False
    steps: list = field(default_factory=list)  # list[Step]
    stop_reason: str = ""

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    def trace_text(self) -> str:
        lines = [f"TASK: {self.task}", f"STOP: {self.stop_reason} (success={self.success})", ""]
        for s in self.steps:
            head = f"[{s.index}] {s.kind}"
            if s.tool:
                head += f" :: {s.tool}({', '.join(f'{k}={v!r}' for k, v in s.args.items())})"
            lines.append(head)
            if s.output:
                out = s.output if len(s.output) < 600 else s.output[:600] + "…"
                lines.append("    " + out.replace("\n", "\n    "))
        lines.append("")
        lines.append(f"ANSWER:\n{self.answer}")
        return "\n".join(lines)
