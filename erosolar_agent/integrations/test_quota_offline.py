"""Offline tests for quota handling — no network calls.

    python -m erosolar_agent.integrations.test_quota_offline   # or: pytest
"""

from __future__ import annotations

from . import deepseek, quota, tavily


def test_is_quota_error():
    assert quota.is_quota_error(402)
    assert quota.is_quota_error(429)
    assert quota.is_quota_error(200, "Insufficient Balance")
    assert quota.is_quota_error(200, "monthly usage limit exceeded")
    assert not quota.is_quota_error(200, "ok")
    assert not quota.is_quota_error(500, "internal server error")


def test_friendly_messages():
    t = quota.friendly_message("Tavily").lower()
    assert "monthly" in t and "top up" in t and "disabled" in t
    d = quota.friendly_message("DeepSeek").lower()
    assert "balance" in d and "top up" in d and "disabled" in d


def test_disable_cycle():
    quota._disabled.clear()
    assert quota.disabled_status("Tavily") == (False, "")
    quota.mark_exhausted("Tavily")
    dis, msg = quota.disabled_status("Tavily")
    assert dis and "DISABLED" in msg
    quota._disabled.clear()


def test_disabled_short_circuits_without_network():
    """Once marked exhausted, clients raise QuotaExhausted before any network/key use."""
    quota._disabled.clear()
    quota.mark_exhausted("Tavily")
    try:
        tavily.search("anything")
        assert False, "expected QuotaExhausted"
    except quota.QuotaExhausted as e:
        assert "monthly" in str(e).lower()

    quota.mark_exhausted("DeepSeek")
    try:
        deepseek.chat([{"role": "user", "content": "hi"}])
        assert False, "expected QuotaExhausted"
    except quota.QuotaExhausted as e:
        assert "balance" in str(e).lower()
    quota._disabled.clear()


def _main():
    for fn in (test_is_quota_error, test_friendly_messages, test_disable_cycle,
               test_disabled_short_circuits_without_network):
        fn()
        print(f"  ok: {fn.__name__}")
    print("ALL QUOTA TESTS PASSED")


if __name__ == "__main__":
    _main()
