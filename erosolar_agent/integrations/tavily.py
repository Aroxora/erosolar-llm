"""Tavily web-search client with graceful quota handling.

On quota/credit exhaustion it marks Tavily disabled (until the monthly reset) and
raises QuotaExhausted with a friendly message, rather than leaking a raw HTTP error.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from .. import secrets
from .quota import QuotaExhausted, disabled_status, is_quota_error, mark_exhausted

DEFAULT_URL = "https://api.tavily.com/search"


def available() -> bool:
    disabled, _ = disabled_status("Tavily")
    return bool(secrets.get_secret("TAVILY_API_KEY")) and not disabled


def search(query: str, max_results: int = 5, include_answer: bool = True) -> dict:
    """Run a Tavily search. Raises QuotaExhausted if out of monthly quota,
    RuntimeError for other failures (missing key, network, other HTTP)."""
    disabled, _ = disabled_status("Tavily")
    if disabled:
        raise QuotaExhausted("Tavily")

    key = secrets.get_secret("TAVILY_API_KEY")
    if not key:
        raise RuntimeError("TAVILY_API_KEY not set")
    url = secrets.get_secret("TAVILY_SEARCH_URL", DEFAULT_URL)

    payload = json.dumps({
        "api_key": key,
        "query": query,
        "max_results": int(max_results),
        "include_answer": include_answer,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace") if e.fp else ""
        if is_quota_error(e.code, body):
            mark_exhausted("Tavily")
            raise QuotaExhausted("Tavily", f"HTTP {e.code}") from e
        raise RuntimeError(f"Tavily HTTP {e.code}: {body[:200]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Tavily request failed: {e}") from e
