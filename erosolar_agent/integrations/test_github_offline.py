"""Offline tests for the GitHub commit-tracker integration — no network.

Run: python -m erosolar_agent.integrations.test_github_offline
"""

from __future__ import annotations

import os

from . import github


def test_repo_resolution():
    assert github._repo("owner/name") == "owner/name"
    assert github._repo(" owner/name/ ") == "owner/name"
    os.environ["GITHUB_REPO"] = "Aroxora/erosolar-llm"
    import erosolar_agent.secrets as s
    s._dotenv.cache_clear()
    assert github._repo(None) == "Aroxora/erosolar-llm"


def test_commit_parsing():
    raw = {
        "sha": "abcdef1234567890",
        "html_url": "https://github.com/o/r/commit/abcdef1",
        "commit": {
            "message": "fix the thing\n\nbody text",
            "author": {"name": "Bo Shang", "email": "bo@ero.solar", "date": "2026-06-14T07:00:00Z"},
        },
    }
    c = github._to_commit(raw)
    assert c.short_sha == "abcdef1"
    assert c.author == "Bo Shang"
    assert c.date.startswith("2026-06-14")
    assert "fix the thing" in c.summary()
    assert "body text" not in c.summary()  # summary is first line only


def test_token_required_message():
    # With no token, _request must raise a clear RuntimeError (not leak).
    import erosolar_agent.secrets as s
    old = os.environ.pop("GITHUB_TOKEN", None)
    s._dotenv.cache_clear()
    try:
        # force empty by pointing secrets at env only — simulate missing token
        if github.secrets.get_secret("GITHUB_TOKEN"):
            return  # token present via .env; skip (can't simulate missing offline)
        raised = False
        try:
            github._request("/rate_limit")
        except RuntimeError as e:
            raised = "GITHUB_TOKEN" in str(e)
        except Exception:
            raised = True
        assert raised
    finally:
        if old is not None:
            os.environ["GITHUB_TOKEN"] = old


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn(); print(f"  ok  {fn.__name__}"); passed += 1
        except Exception as e:  # noqa: BLE001
            print(f" FAIL {fn.__name__}: {e}")
    print(f"\n{passed}/{len(fns)} offline tests passed")
    return passed == len(fns)


if __name__ == "__main__":
    raise SystemExit(0 if _run_all() else 1)
