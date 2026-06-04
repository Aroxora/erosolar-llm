"""Graceful handling for metered third-party APIs (Tavily, DeepSeek).

When a provider runs out of quota/credits we:
  1. surface a clear, friendly explanation instead of a raw HTTP error, and
  2. mark the provider disabled in-process so we stop hammering a dead endpoint
     until its reset (Tavily = monthly reset; DeepSeek = balance top-up).

The disabled flag is in-process only (resets on restart, e.g. after a top-up).
"""

from __future__ import annotations

from datetime import datetime, timezone

# Reset semantics per provider.
_MONTHLY = {"tavily"}    # monthly credit allotment that resets on the 1st
_BALANCE = {"deepseek"}  # pay-as-you-go balance; clears when you top up

# HTTP statuses + body markers that indicate "out of quota/credits".
_QUOTA_STATUS = {402, 429, 432, 433}
_QUOTA_MARKERS = (
    "insufficient balance", "insufficient_quota", "quota", "credit",
    "usage limit", "plan limit", "exceeded", "payment required", "out of credits",
    "rate limit",
)

_disabled: dict = {}  # provider(lower) -> reset_iso or None


class QuotaExhausted(RuntimeError):
    """Raised when a metered provider is out of quota/credits."""

    def __init__(self, provider: str, detail: str = ""):
        self.provider = provider
        self.detail = detail
        super().__init__(friendly_message(provider, detail))


def _next_month_reset_iso() -> str:
    now = datetime.now(timezone.utc)
    year = now.year + (1 if now.month == 12 else 0)
    month = 1 if now.month == 12 else now.month + 1
    return datetime(year, month, 1, tzinfo=timezone.utc).isoformat()


def is_quota_error(status_code: int, body_text: str = "") -> bool:
    """True if an HTTP status/body indicates quota/credit exhaustion."""
    if status_code in _QUOTA_STATUS:
        return True
    return bool(body_text) and any(m in body_text.lower() for m in _QUOTA_MARKERS)


def friendly_message(provider: str, detail: str = "") -> str:
    p = provider.lower()
    if p in _MONTHLY:
        msg = (f"{provider} is out of its monthly quota — DISABLED until the monthly reset "
               f"(≈ {_next_month_reset_iso()[:10]}). Top up this month to re-enable sooner.")
    elif p in _BALANCE:
        msg = (f"{provider} API is out of balance — DISABLED until you top up this month "
               f"(pay-as-you-go; there is no automatic monthly reset).")
    else:
        msg = f"{provider} is out of quota — temporarily disabled."
    return msg + (f" [{detail}]" if detail else "")


def mark_exhausted(provider: str) -> None:
    reset = _next_month_reset_iso() if provider.lower() in _MONTHLY else None
    _disabled[provider.lower()] = reset


def disabled_status(provider: str):
    """Return (is_disabled, message). Auto-clears once past a monthly reset time."""
    p = provider.lower()
    if p not in _disabled:
        return False, ""
    reset = _disabled[p]
    if reset and datetime.now(timezone.utc).isoformat() >= reset:
        del _disabled[p]
        return False, ""
    return True, friendly_message(provider)
