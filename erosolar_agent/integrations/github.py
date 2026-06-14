"""GitHub commit-tracker client (read-only REST v3) with graceful errors.

Mirrors the other integrations (tavily.py / deepseek.py): secrets come from the
gitignored .env via :mod:`erosolar_agent.secrets`, the token is never logged, and
failures raise friendly ``RuntimeError`` / :class:`QuotaExhausted` instead of leaking
raw HTTP. Powers the "erosolar-commit-tracker" integration — list/inspect commits and
repo metadata for the project (or any repo the PAT can read).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from .. import secrets
from .quota import QuotaExhausted, disabled_status, is_quota_error, mark_exhausted

DEFAULT_BASE = "https://api.github.com"


def available() -> bool:
    disabled, _ = disabled_status("GitHub")
    return bool(secrets.get_secret("GITHUB_TOKEN")) and not disabled


def _repo(repo: str | None) -> str:
    r = repo or secrets.get_secret("GITHUB_REPO", "")
    if not r:
        raise RuntimeError("No repo given and GITHUB_REPO not set (owner/name).")
    return r.strip().strip("/")


def _request(path: str, params: dict | None = None) -> object:
    """GET the GitHub REST API. Raises QuotaExhausted on rate-limit, RuntimeError
    for missing token / other HTTP / network failures."""
    disabled, _ = disabled_status("GitHub")
    if disabled:
        raise QuotaExhausted("GitHub")

    token = secrets.get_secret("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN not set")
    base = secrets.get_secret("GITHUB_API_BASE", DEFAULT_BASE).rstrip("/")
    url = base + path
    if params:
        from urllib.parse import urlencode

        url += "?" + urlencode({k: v for k, v in params.items() if v is not None})

    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "erosolar-commit-tracker",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace") if e.fp else ""
        # GitHub rate-limit: 403/429 with a zero remaining or "rate limit" message.
        if e.code in (403, 429) and ("rate limit" in body.lower() or is_quota_error(e.code, body)):
            mark_exhausted("GitHub")
            raise QuotaExhausted("GitHub", f"HTTP {e.code}") from e
        if e.code == 401:
            raise RuntimeError("GitHub auth failed (401) — token invalid/expired or lacks scope.") from e
        raise RuntimeError(f"GitHub HTTP {e.code}: {body[:200]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"GitHub request failed: {e}") from e


@dataclass
class Commit:
    sha: str
    short_sha: str
    message: str
    author: str
    author_email: str
    date: str
    url: str

    def summary(self) -> str:
        return f"{self.short_sha} {self.date[:10]} {self.author}: {self.message.splitlines()[0][:72]}"


def _to_commit(raw: dict) -> Commit:
    c = raw.get("commit", {}) or {}
    a = c.get("author", {}) or {}
    sha = raw.get("sha", "") or ""
    return Commit(
        sha=sha,
        short_sha=sha[:7],
        message=c.get("message", "") or "",
        author=a.get("name", "") or "",
        author_email=a.get("email", "") or "",
        date=a.get("date", "") or "",
        url=raw.get("html_url", "") or "",
    )


def list_commits(repo: str | None = None, per_page: int = 10,
                 sha: str | None = None, since: str | None = None) -> list[Commit]:
    """Recent commits for a repo (newest first). `since` is an ISO-8601 timestamp."""
    data = _request(f"/repos/{_repo(repo)}/commits",
                    {"per_page": min(int(per_page), 100), "sha": sha, "since": since})
    return [_to_commit(r) for r in (data or [])]


def get_commit(sha: str, repo: str | None = None) -> Commit:
    return _to_commit(_request(f"/repos/{_repo(repo)}/commits/{sha}"))


def repo_info(repo: str | None = None) -> dict:
    d = _request(f"/repos/{_repo(repo)}")
    return {
        "full_name": d.get("full_name"),
        "default_branch": d.get("default_branch"),
        "private": d.get("private"),
        "pushed_at": d.get("pushed_at"),
        "open_issues": d.get("open_issues_count"),
        "stargazers": d.get("stargazers_count"),
    }


def rate_limit() -> dict:
    d = _request("/rate_limit")
    return (d.get("resources", {}) or {}).get("core", d.get("rate", {}))


def _main(argv=None) -> int:
    """python -m erosolar_agent.integrations.github [commits N | info | rate]"""
    import sys

    args = (argv if argv is not None else sys.argv[1:]) or ["commits", "10"]
    cmd = args[0]
    if not available():
        print("GitHub not available (set GITHUB_TOKEN in .env).", file=sys.stderr)
        return 2
    try:
        if cmd == "commits":
            n = int(args[1]) if len(args) > 1 else 10
            for c in list_commits(per_page=n):
                print(c.summary())
        elif cmd == "info":
            print(json.dumps(repo_info(), indent=2))
        elif cmd == "rate":
            print(json.dumps(rate_limit(), indent=2))
        else:
            print(f"unknown command: {cmd}", file=sys.stderr)
            return 2
    except QuotaExhausted:
        print("GitHub rate limit hit; try again later.", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
