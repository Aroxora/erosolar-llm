"""DeepSeek chat client (OpenAI-compatible) with graceful quota handling.

On insufficient-balance / quota errors it marks DeepSeek disabled (until you top
up) and raises QuotaExhausted with a friendly message instead of a raw HTTP error.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from .. import secrets
from .quota import QuotaExhausted, disabled_status, is_quota_error, mark_exhausted

DEFAULT_BASE = "https://api.deepseek.com"


def available() -> bool:
    disabled, _ = disabled_status("DeepSeek")
    return bool(secrets.get_secret("DEEPSEEK_API_KEY")) and not disabled


def chat(messages: list, model: str = "deepseek-chat", temperature: float = 0.3,
         max_tokens: int = 1024) -> str:
    """Call DeepSeek chat-completions. Raises QuotaExhausted if out of balance,
    RuntimeError for other failures."""
    disabled, _ = disabled_status("DeepSeek")
    if disabled:
        raise QuotaExhausted("DeepSeek")

    key = secrets.get_secret("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    base = secrets.get_secret("DEEPSEEK_BASE_URL",
                              secrets.get_secret("DEEPSEEK_API_BASE", DEFAULT_BASE))
    url = base.rstrip("/") + "/chat/completions"

    payload = json.dumps({
        "model": model, "messages": messages,
        "temperature": temperature, "max_tokens": int(max_tokens),
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
            return data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace") if e.fp else ""
        if is_quota_error(e.code, body):
            mark_exhausted("DeepSeek")
            raise QuotaExhausted("DeepSeek", f"HTTP {e.code}") from e
        raise RuntimeError(f"DeepSeek HTTP {e.code}: {body[:200]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"DeepSeek request failed: {e}") from e
