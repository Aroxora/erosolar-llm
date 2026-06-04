"""Long-horizon context management: estimate transcript size and, when it
exceeds the budget, summarize the older portion into a compact note while
keeping recent turns verbatim. Durable notes/plan live in the system block
(memory.py) and are never compacted away."""

from __future__ import annotations

import json

from .prompts import SUMMARIZE_SYSTEM
from .types import user_msg


def estimate_tokens(messages: list) -> int:
    """Cheap heuristic (~4 chars/token) — good enough for budgeting."""
    chars = 0
    for m in messages:
        chars += len(json.dumps(m, ensure_ascii=False))
    return chars // 4


def render_transcript(messages: list) -> str:
    lines = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content", "")
        if m.get("tool_calls"):
            calls = ", ".join(
                f"{tc['function']['name']}({tc['function']['arguments']})"
                for tc in m["tool_calls"]
            )
            content = (content + " " if content else "") + f"[calls: {calls}]"
        if role == "tool":
            content = f"(result of {m.get('name', 'tool')}) {content}"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _safe_keep(transcript: list, keep: int) -> int:
    """Grow `keep` until the kept slice does not start on an orphaned tool message
    (OpenAI requires a tool message to follow its assistant tool_calls)."""
    n = len(transcript)
    keep = min(keep, n)
    while keep < n and transcript[n - keep].get("role") == "tool":
        keep += 1
    return keep


def maybe_compact(memory, llm, cfg) -> bool:
    """Summarize old transcript if over budget. Returns True if it compacted."""
    if estimate_tokens(memory.messages()) <= cfg.context_token_budget:
        return False
    keep = _safe_keep(memory.transcript, cfg.keep_recent_messages)
    if len(memory.transcript) <= keep + 1:
        return False  # nothing meaningful to compress

    old = memory.transcript[:-keep]
    recent = memory.transcript[-keep:]
    summary = llm.complete(
        SUMMARIZE_SYSTEM,
        "Transcript to compress:\n\n" + render_transcript(old),
        max_tokens=700,
    )
    memory.transcript = [user_msg("[Earlier steps — summarized]\n" + summary)] + recent
    return True
