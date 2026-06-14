"""Inbound-mail triage: bounce/broken-address detection + follow-up judgement.

Two jobs:

1. **Bounce handling** (no LLM): detect delivery-failure notifications, extract the
   failed recipient, and attempt a conservative typo fix on the address (common
   domain misspellings only — never guesses a new address).

2. **Follow-up judgement** (``deepseek-v4-pro``): for a genuine human reply, decide
   whether a follow-up is *absolutely sensible*. If not, the model picks a
   disposition: escalate to the human owner (with required actions) or close as a
   dead end (a cheap ``deepseek-v4-flash`` summary is sent to the owner).

Honesty: the address fixer only rewrites a domain to a known-correct spelling from
a small allow-list; it never invents local-parts or domains.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..integrations import deepseek
from ..integrations.quota import QuotaExhausted
from .mail import IncomingMessage

# ── bounce detection ──────────────────────────────────────────────────────
_BOUNCE_FROM = re.compile(r"(mailer-daemon|postmaster|mail delivery|delivery (sub)?system)", re.I)
_BOUNCE_BODY = re.compile(
    r"(550|552|553|5\.1\.[0-9]|5\.2\.[0-9]|user unknown|address not found|"
    r"recipient.*(rejected|not found)|mailbox (unavailable|full)|"
    r"does not exist|no such (user|recipient)|undeliverable|delivery (has )?failed)",
    re.I,
)
_FAILED_RCPT = re.compile(
    r"(?:Final-Recipient:\s*rfc822;\s*|X-Failed-Recipients:\s*|<)([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})",
)

# Conservative domain typo allow-list (typo → correct). Never guesses beyond this.
_DOMAIN_FIX = {
    "gmial.com": "gmail.com", "gmai.com": "gmail.com", "gmail.con": "gmail.com",
    "gmail.co": "gmail.com", "gmaill.com": "gmail.com", "gnail.com": "gmail.com",
    "hotmial.com": "hotmail.com", "hotmai.com": "hotmail.com",
    "outlok.com": "outlook.com", "outlook.con": "outlook.com",
    "yahooo.com": "yahoo.com", "yaho.com": "yahoo.com", "yahoo.con": "yahoo.com",
    "iclould.com": "icloud.com", "icloud.con": "icloud.com",
    "protonmai.com": "protonmail.com", "proton.mail": "proton.me",
}


def is_bounce(msg: IncomingMessage) -> bool:
    return bool(_BOUNCE_FROM.search(msg.from_email + " " + msg.from_name)) or bool(
        _BOUNCE_BODY.search((msg.subject + " " + msg.body)[:4000])
    )


def failed_recipient(msg: IncomingMessage) -> str:
    m = _FAILED_RCPT.search(msg.body or "")
    return (m.group(1).lower() if m else "").strip("<>")


def fix_address(addr: str) -> str:
    """Return a corrected address if the domain is a known typo, else ""."""
    addr = (addr or "").strip().lower()
    if "@" not in addr:
        return ""
    local, _, domain = addr.partition("@")
    fixed = _DOMAIN_FIX.get(domain)
    return f"{local}@{fixed}" if fixed and local else ""


# ── follow-up judgement ─────────────────────────────────────────────────────
_JUDGE_SYSTEM = """You triage replies to outreach emails for a small founder-led \
project. Decide whether sending a follow-up is ABSOLUTELY SENSIBLE.

A follow-up is sensible only when the reply shows genuine interest or asks \
something we can usefully answer, AND a reply would plausibly advance the \
conversation. Auto-replies, "not interested", hostility, out-of-office, and \
spam are NOT sensible.

If a follow-up is NOT sensible, choose a disposition:
- "human": the reply needs a real person's decision/action (e.g. a meeting \
request, a question only the founder can answer, a legal/commercial ask).
- "dead_end": no value in continuing (rejection, irrelevant, bot, bounce echo).

Return ONLY compact JSON:
{"sensible": true|false, "disposition": "follow_up"|"human"|"dead_end",
 "reason": "<one sentence>", "required_human_actions": "<if human: what the \
founder must do; else empty>"}"""


@dataclass
class Judgement:
    sensible: bool
    disposition: str  # "follow_up" | "human" | "dead_end"
    reason: str
    required_human_actions: str = ""


def _coerce(text: str) -> dict:
    text = (text or "").strip().strip("`")
    s, e = text.find("{"), text.rfind("}")
    if s >= 0 and e > s:
        try:
            return json.loads(text[s : e + 1])
        except json.JSONDecodeError:
            return {}
    return {}


def judge_followup(inbound: IncomingMessage, *, model: str, history_context: str = "") -> Judgement:
    """Ask deepseek-v4-pro whether a follow-up is sensible. Conservative fallback
    (escalate to human) if the model output can't be parsed."""
    facts = [
        f"From: {inbound.from_name} <{inbound.from_email}>",
        f"Subject: {inbound.subject}",
        f"Reply body:\n{inbound.body[:2500]}",
    ]
    if history_context:
        facts.append(f"Prior context:\n{history_context[:1200]}")
    try:
        raw = deepseek.chat(
            [{"role": "system", "content": _JUDGE_SYSTEM},
             {"role": "user", "content": "\n".join(facts)}],
            model=model, temperature=0.0, max_tokens=300,
        )
    except QuotaExhausted:
        raise
    obj = _coerce(raw)
    disp = obj.get("disposition")
    if disp not in ("follow_up", "human", "dead_end"):
        # When unsure, never auto-follow-up: hand to the human.
        return Judgement(False, "human", "unparseable judgement; escalating to human", inbound.subject)
    return Judgement(
        sensible=bool(obj.get("sensible")) and disp == "follow_up",
        disposition=disp,
        reason=str(obj.get("reason", ""))[:300],
        required_human_actions=str(obj.get("required_human_actions", ""))[:600],
    )


def summarize_dead_end(inbound: IncomingMessage, *, model: str) -> str:
    """Cheap deepseek-v4-flash summary of a dead-end thread for the owner digest."""
    try:
        return deepseek.chat(
            [{"role": "system", "content": "Summarize this dead-end email reply in 1-2 plain sentences for the founder's records. No preamble."},
             {"role": "user", "content": f"From {inbound.from_email} — {inbound.subject}\n{inbound.body[:2000]}"}],
            model=model, temperature=0.2, max_tokens=160,
        ).strip()
    except QuotaExhausted:
        raise
    except Exception:
        return f"Dead-end reply from {inbound.from_email}: {inbound.subject}"
