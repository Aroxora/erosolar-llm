"""Task decomposition (planning) and progress critique (reflection)."""

from __future__ import annotations

from .context import render_transcript
from .prompts import PLANNER_SYSTEM, REFLECT_SYSTEM


def make_plan(llm, task: str, max_steps: int = 8) -> list:
    text = llm.complete(PLANNER_SYSTEM, f"Task:\n{task}", temperature=0.3, max_tokens=400)
    plan = []
    for line in text.splitlines():
        cleaned = line.strip().lstrip("-*0123456789.) ").strip()
        if cleaned:
            plan.append(cleaned)
    return plan[:max_steps]


def reflect(llm, task: str, memory) -> str:
    recent = render_transcript(memory.transcript[-10:])
    prompt = (
        f"Task:\n{task}\n\nPlan:\n{memory.plan_text()}\n\n"
        f"Durable notes:\n{memory.notes_text()}\n\nRecent transcript:\n{recent}"
    )
    return llm.complete(REFLECT_SYSTEM, prompt, temperature=0.3, max_tokens=300)
