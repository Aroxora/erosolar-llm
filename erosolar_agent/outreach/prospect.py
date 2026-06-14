"""Agentic lead discovery via Tavily web search.

Honesty rule: Tavily returns URLs + snippets, not verified contact addresses.
This module NEVER fabricates an email. A discovered lead carries a real email
only if one is plainly present in the retrieved text; otherwise ``email=""`` and
the lead is research-only (not sendable) until a human supplies an address.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..integrations import tavily
from ..integrations.quota import QuotaExhausted

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
# Addresses we never treat as a real lead contact.
_JUNK_EMAIL = re.compile(r"(noreply|no-reply|donotreply|example\.|sentry|wixpress|@2x|\.png|\.jpg)", re.I)


@dataclass
class Lead:
    name: str = ""
    org: str = ""
    role: str = ""
    email: str = ""
    url: str = ""
    snippet: str = ""
    source: str = "tavily"
    audience: str = ""  # "investors" | "marketing" | free text
    extra: dict = field(default_factory=dict)

    def sendable(self) -> bool:
        return bool(_EMAIL_RE.fullmatch(self.email or "")) and not _JUNK_EMAIL.search(self.email)

    def dedupe_key(self) -> str:
        return (self.email or self.url or self.name).strip().lower()


def _extract_email(text: str) -> str:
    for m in _EMAIL_RE.findall(text or ""):
        if not _JUNK_EMAIL.search(m):
            return m.lower()
    return ""


def discover(brief: str, audience: str = "", max_results: int = 8) -> list[Lead]:
    """Search for prospects matching a campaign brief.

    Returns Lead objects. Raises QuotaExhausted if Tavily is out of quota — the
    caller (engine) treats that as "skip prospecting this cycle", not a crash.
    """
    if not tavily.available():
        return []
    # Bias the query toward pages that actually carry a contact address.
    base = brief if not audience else f"{brief} ({audience})"
    query = f"{base} contact email"
    try:
        # raw_content gives full page text, where real emails usually live.
        resp = tavily.search(query, max_results=max_results, include_answer=False,
                             include_raw_content=True)
    except QuotaExhausted:
        raise
    except Exception:
        return []

    leads: list[Lead] = []
    for r in resp.get("results", []) or []:
        snippet = (r.get("content") or "")[:600]
        raw = (r.get("raw_content") or "")
        title = r.get("title") or ""
        email = _extract_email(snippet + " " + title) or _extract_email(raw)
        leads.append(
            Lead(
                name="",
                org=title,
                role="",
                email=email,
                url=r.get("url", ""),
                snippet=snippet,
                audience=audience,
                extra={"score": r.get("score")},
            )
        )
    return leads


def dedupe(leads: list[Lead], seen_keys: set[str]) -> list[Lead]:
    """Drop leads we've already stored/contacted."""
    fresh, local = [], set()
    for lead in leads:
        k = lead.dedupe_key()
        if not k or k in seen_keys or k in local:
            continue
        local.add(k)
        fresh.append(lead)
    return fresh
