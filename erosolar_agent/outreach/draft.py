"""Personalized outreach drafting via DeepSeek (``deepseek-v4-pro``).

The drafter is grounded: it is given the lead's retrieved context and any prior
thread history (from RAG) and is instructed to make no factual claims it cannot
support, to stay concise, and to always include an opt-out line. Reply drafting
reuses the same path with the inbound message as context.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..integrations import deepseek
from ..integrations.quota import QuotaExhausted
from .prospect import Lead

_SYSTEM = """You write short, honest, high-signal outreach emails for Erosolar, a \
small-LLM research project (an honest, measurable Chain-of-Thought pipeline + an \
additive agent stack). You are emailing on behalf of the founder.

Hard rules:
- Be truthful. Never invent metrics, customers, funding, or credentials. If you \
don't have a fact, omit it.
- Audience "investors": concise, specific, what's novel + why now; one clear ask \
(a short call). Audience "marketing": partnership/collaboration angle.
- 90-160 words. Plain text. No markdown, no emoji, no fake personalization.
- One soft call-to-action. Sign as the configured sender.
- End with a single short opt-out line: "Reply 'unsubscribe' to opt out."
- Return ONLY compact JSON: {"subject": "...", "body": "..."} and nothing else."""


@dataclass
class Draft:
    subject: str
    body: str


def _coerce_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{") :]
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return {}


def draft_email(
    lead: Lead,
    *,
    model: str,
    from_name: str,
    campaign_brief: str = "",
    history_context: str = "",
    inbound: str = "",
) -> Draft:
    """Produce a subject+body. Raises QuotaExhausted if DeepSeek is out of balance."""
    persona = lead.audience or "general"
    facts = [
        f"Sender name: {from_name}",
        f"Audience: {persona}",
        f"Lead org/title: {lead.org or 'unknown'}",
        f"Lead role: {lead.role or 'unknown'}",
        f"Lead source URL: {lead.url or 'n/a'}",
        f"What we know (snippet): {lead.snippet[:500] or 'n/a'}",
    ]
    if campaign_brief:
        facts.append(f"Campaign brief: {campaign_brief}")
    if history_context:
        facts.append(f"Relevant prior context (RAG):\n{history_context[:1200]}")
    if inbound:
        facts.append(
            "This is a REPLY to the inbound message below. Answer it directly, "
            f"keep continuity, do not re-pitch from scratch:\n---\n{inbound[:1500]}\n---"
        )

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": "\n".join(facts)},
    ]
    try:
        raw = deepseek.chat(messages, model=model, temperature=0.5, max_tokens=600)
    except QuotaExhausted:
        raise
    obj = _coerce_json(raw)
    subject = (obj.get("subject") or "").strip()
    body = (obj.get("body") or "").strip()
    if not subject or not body:
        # Honest, safe fallback rather than sending a malformed email.
        subject = subject or f"Erosolar — a quick note for {lead.org or 'you'}"
        body = body or raw.strip()[:1500]
    return Draft(subject=subject, body=body)
