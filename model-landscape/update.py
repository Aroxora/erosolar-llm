#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""model-landscape/update.py — agentically refresh the frontier-model deep-dives.

This is a *real* use of the erosolar agent stack: it drives this repo's own
`erosolar_agent.integrations` (Tavily web search + DeepSeek synthesis, both with
graceful quota handling) to keep `model-landscape/models/<slug>.md` current.

Pipeline, per model in `models.yaml`:
    1. SEARCH   Tavily for recent, primary-source material (last-N-months bias).
    2. SYNTHE   DeepSeek writes a cited markdown profile from the snippets only.
    3. WRITE    models/<slug>.md, then regenerate the AUTO block in README.md.

Honesty rules (consistent with the rest of erosolar):
    * Every model file is stamped with an "as of" date and its sources.
    * The synthesizer is instructed to NOT invent specs/benchmarks/prices, and to
      mark a model `not_found` / `rumored` / `unverified` when it cannot confirm it.
    * If no API keys are set or quota is exhausted, we DO NOT clobber the existing
      committed snapshots — we print a friendly note, refresh the README index from
      whatever files exist, and exit 0.

Usage:
    python model-landscape/update.py                 # refresh all models
    python model-landscape/update.py --only gpt-5.5  # one or more slugs
    python model-landscape/update.py --dry-run       # plan only, write nothing
    python model-landscape/update.py --reindex       # rebuild README index, no network
    python model-landscape/update.py --max-results 8 # Tavily results per model

No third-party deps beyond what erosolar already uses; PyYAML is optional (a tiny
built-in parser handles this manifest if PyYAML is absent).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
MODELS_DIR = HERE / "models"
MANIFEST = HERE / "models.yaml"
STAMP = HERE / ".last_updated.json"
README = HERE / "README.md"

AUTO_START = "<!-- AUTO:MODELS:START -->"
AUTO_END = "<!-- AUTO:MODELS:END -->"

# Make the in-repo agent integrations importable when run as a plain script.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _today() -> str:
    return _dt.date.today().isoformat()


# ── manifest loading (PyYAML if present, else a minimal parser) ──────────────
def load_manifest() -> list[dict]:
    text = MANIFEST.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
        return (yaml.safe_load(text) or {}).get("models", [])
    except Exception:
        return _mini_parse_models(text)


def _mini_parse_models(text: str) -> list[dict]:
    """Tiny fallback parser for the specific shape of models.yaml (no PyYAML)."""
    models: list[dict] = []
    cur: dict | None = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip() if not raw.strip().startswith("#") else ""
        if not line.strip():
            continue
        if re.match(r"\s*-\s+\w", line):
            if cur:
                models.append(cur)
            cur = {}
            line = re.sub(r"^\s*-\s+", "  ", line)
        if cur is None:
            continue
        m = re.match(r"\s*([A-Za-z_]+):\s*(.*)$", line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        if val.startswith("[") and val.endswith("]"):
            inner = val[1:-1].strip()
            cur[key] = [v.strip().strip('"').strip("'") for v in inner.split(",") if v.strip()]
        else:
            cur[key] = val.strip().strip('"').strip("'")
    if cur:
        models.append(cur)
    return [m for m in models if m.get("slug")]


# ── search + synthesis (graceful when keys/quota are missing) ────────────────
def gather_sources(model: dict, max_results: int) -> list[dict]:
    from erosolar_agent.integrations import tavily
    from erosolar_agent.integrations.quota import QuotaExhausted

    if not tavily.available():
        raise RuntimeError("Tavily unavailable (no TAVILY_API_KEY or disabled this month)")

    names = [model["name"], *model.get("aliases", [])]
    query = (
        f"{model['name']} {model.get('vendor', '')} language model — official announcement, "
        f"model card, context window, benchmarks, pricing, release date 2026"
    )
    out: list[dict] = []
    try:
        resp = tavily.search(query, max_results=max_results, include_answer=True)
        for r in resp.get("results", []):
            out.append({"title": r.get("title", ""), "url": r.get("url", ""),
                        "content": (r.get("content") or "")[:1200]})
        if resp.get("answer"):
            out.insert(0, {"title": "Tavily synthesis", "url": "", "content": resp["answer"][:1500]})
    except QuotaExhausted:
        raise
    return out


def synthesize(model: dict, sources: list[dict]) -> str | None:
    from erosolar_agent.integrations import deepseek
    from erosolar_agent.integrations.quota import QuotaExhausted

    if not deepseek.available():
        return None  # caller falls back to a sources-only stub

    src_block = "\n\n".join(
        f"[{i+1}] {s['title']} — {s['url']}\n{s['content']}" for i, s in enumerate(sources)
    ) or "(no sources retrieved)"

    system = (
        "You are a precise technical writer maintaining a frontier-model reference. "
        "Write ONLY from the provided sources. Never invent parameter counts, benchmark "
        "scores, dates, or prices. If the sources do not confirm the model exists, say so "
        "and label status not_found/rumored/unverified. Cite claims like [1], [2] mapping "
        "to the numbered sources. Keep it tight and factual."
    )
    user = (
        f"Model: {model['name']} (vendor: {model.get('vendor','?')}, family: {model.get('family','?')}).\n"
        f"Today is {_today()}.\n\nSOURCES:\n{src_block}\n\n"
        "Write a markdown profile with these sections: a one-line status+positioning; "
        "**Status**; **What it is**; **Architecture / context / modalities** (only what is "
        "disclosed); **Benchmarks** (name each benchmark; omit if unverified); "
        "**Access / pricing / API id**; **Caveats**. End with a numbered **Sources** list "
        "of the URLs you used."
    )
    try:
        return deepseek.chat([{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                             temperature=0.2, max_tokens=1600)
    except QuotaExhausted:
        raise


def render_model_md(model: dict, body: str, sources: list[dict]) -> str:
    head = (
        f"# {model['name']}\n\n"
        f"> Vendor: **{model.get('vendor','?')}** · Family: {model.get('family','?')} · "
        f"_auto-generated by `model-landscape/update.py` — as of {_today()}_\n\n"
        "_Figures are model-/vendor-reported via the linked sources unless a primary "
        "source is cited. This file is regenerated; edit `models.yaml` or `update.py`, not "
        "this file._\n\n---\n\n"
    )
    if not body:  # sources-only stub (no LLM available)
        lines = [head, "_No synthesis model available; raw sources only._\n"]
        for i, s in enumerate(sources, 1):
            lines.append(f"{i}. [{s['title'] or s['url']}]({s['url']})")
        return "\n".join(lines) + "\n"
    return head + body.strip() + "\n"


# ── README index (regenerated between AUTO markers; intro is preserved) ──────
def reindex_readme() -> None:
    rows = []
    for md in sorted(MODELS_DIR.glob("*.md")):
        title = md.stem
        first = ""
        for ln in md.read_text(encoding="utf-8").splitlines():
            if ln.startswith("# "):
                title = ln[2:].strip()
            elif ln.startswith(">") or (ln.strip() and not ln.startswith("#")):
                first = ln.lstrip("> ").strip()
                break
        rows.append(f"| [{title}](models/{md.name}) | {first[:110]} |")
    table = "| Model | Snapshot |\n|---|---|\n" + ("\n".join(rows) if rows else "| _(none yet)_ | |")
    block = f"{AUTO_START}\n\n_Last indexed: {_today()}_\n\n{table}\n\n{AUTO_END}"

    if README.exists():
        text = README.read_text(encoding="utf-8")
        if AUTO_START in text and AUTO_END in text:
            text = re.sub(re.escape(AUTO_START) + r".*?" + re.escape(AUTO_END), block, text, flags=re.S)
        else:
            text = text.rstrip() + "\n\n## Tracked models\n\n" + block + "\n"
        README.write_text(text, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Agentically refresh frontier-model deep-dives.")
    ap.add_argument("--only", nargs="*", default=None, help="slugs to refresh (default: all)")
    ap.add_argument("--max-results", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true", help="show plan, write nothing")
    ap.add_argument("--reindex", action="store_true", help="rebuild README index only (no network)")
    args = ap.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.reindex:
        reindex_readme()
        print("Reindexed README from existing model files.")
        return 0

    models = load_manifest()
    if args.only:
        wanted = set(args.only)
        models = [m for m in models if m.get("slug") in wanted or m.get("name") in wanted]
    if not models:
        print("No models selected (check models.yaml / --only).")
        return 1

    print(f"model-landscape updater · {len(models)} model(s) · as of {_today()}")
    if args.dry_run:
        for m in models:
            print(f"  would refresh: {m['slug']:18s} ({m['name']})")
        return 0

    from erosolar_agent.integrations.quota import QuotaExhausted

    stamp = {"updated": _today(), "models": {}}
    refreshed = 0
    for m in models:
        slug = m["slug"]
        try:
            sources = gather_sources(m, args.max_results)
            body = synthesize(m, sources)
            (MODELS_DIR / f"{slug}.md").write_text(render_model_md(m, body, sources), encoding="utf-8")
            stamp["models"][slug] = {"ok": True, "sources": len(sources), "synthesized": bool(body)}
            refreshed += 1
            print(f"  ✓ {slug}: {len(sources)} sources, synthesized={bool(body)}")
        except QuotaExhausted as e:
            print(f"  … quota exhausted ({e}); keeping existing snapshots and stopping network calls.")
            stamp["models"][slug] = {"ok": False, "reason": "quota"}
            break
        except Exception as e:  # noqa: BLE001 — degrade gracefully, never clobber on error
            print(f"  ! {slug}: {e} — keeping existing file")
            stamp["models"][slug] = {"ok": False, "reason": str(e)[:160]}

    reindex_readme()
    STAMP.write_text(json.dumps(stamp, indent=2), encoding="utf-8")
    print(f"Done. Refreshed {refreshed}/{len(models)}. Index + .last_updated.json written.")
    print("Tip: set TAVILY_API_KEY and DEEPSEEK_API_KEY in .env to enable live refresh.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
