#!/usr/bin/env python3
"""Build the SFT blend and DPO preference set from a data_blend.yaml spec.

Every source is normalized to the unified chat schema:
    {"messages": [{"role": "system|user|assistant", "content": "..."}, ...]}
DPO rows are normalized to:
    {"prompt": [...messages...], "chosen": "<text>", "rejected": "<text>"}

Design goals for an unattended run on a paid GPU box:
  * Per-source try/except — a missing/renamed/gated dataset logs a warning and is
    SKIPPED, never crashes the whole run.
  * Deterministic shuff/limit (seeded) so reruns are reproducible.
  * --dry-run reports counts without writing.

Usage:
    python -m erosolar_agent.finetune.prepare_data --stage sft --out data
    python -m erosolar_agent.finetune.prepare_data --stage dpo --out data
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
_DEFAULT_BLEND = _HERE / "configs" / "data_blend.yaml"


def _log(msg: str) -> None:
    print(f"[prepare_data] {msg}", flush=True)


# --- format normalizers ----------------------------------------------------

_ROLE_MAP = {
    "human": "user", "user": "user",
    "gpt": "assistant", "assistant": "assistant", "model": "assistant",
    "system": "system", "tool": "tool", "function": "tool", "observation": "tool",
}


def _from_sharegpt(row: dict) -> list | None:
    convo = row.get("conversations") or row.get("conversation")
    if not isinstance(convo, list):
        return None
    out = []
    for turn in convo:
        if not isinstance(turn, dict):
            return None
        role = _ROLE_MAP.get(str(turn.get("from", "")).lower())
        content = turn.get("value")
        if role is None or content is None:
            continue
        out.append({"role": role, "content": str(content)})
    return out or None


def _from_messages(row: dict) -> list | None:
    msgs = row.get("messages")
    if not isinstance(msgs, list):
        return None
    out = []
    for m in msgs:
        if not isinstance(m, dict):
            return None
        role = _ROLE_MAP.get(str(m.get("role", "")).lower(), m.get("role"))
        content = m.get("content")
        if role is None or content is None:
            continue
        # some datasets store content as a list of parts
        if isinstance(content, list):
            content = "".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in content
            )
        out.append({"role": role, "content": str(content)})
    return out or None


def _normalize_sft(row: dict, fmt: str) -> list | None:
    if fmt in ("messages", "tulu"):
        return _from_messages(row)
    if fmt == "sharegpt":
        return _from_sharegpt(row)
    # last resort: try both
    return _from_messages(row) or _from_sharegpt(row)


def _messages_to_text(messages: list) -> str:
    """Flatten messages to a single string for DPO chosen/rejected when needed."""
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def _normalize_dpo(row: dict, fmt: str) -> dict | None:
    chosen = row.get("chosen")
    rejected = row.get("rejected")
    if chosen is None or rejected is None:
        return None
    prompt = row.get("prompt")

    def _coerce(x):
        if isinstance(x, list):  # message list -> last assistant message text
            assistant = [m for m in x if isinstance(m, dict) and m.get("role") == "assistant"]
            if assistant:
                return str(assistant[-1].get("content", ""))
            return _messages_to_text([m for m in x if isinstance(m, dict)])
        return str(x)

    # derive prompt messages if not explicit: use chosen's history minus final assistant turn
    prompt_msgs = None
    if isinstance(prompt, list):
        prompt_msgs = _from_messages({"messages": prompt})
    elif isinstance(prompt, str):
        prompt_msgs = [{"role": "user", "content": prompt}]
    elif isinstance(chosen, list):
        hist = [m for m in chosen if isinstance(m, dict)]
        if hist and hist[-1].get("role") == "assistant":
            hist = hist[:-1]
        prompt_msgs = _from_messages({"messages": hist})

    if not prompt_msgs:
        return None
    return {
        "prompt": prompt_msgs,
        "chosen": _coerce(chosen),
        "rejected": _coerce(rejected),
    }


# --- loading ---------------------------------------------------------------

def _load_source(spec: dict, stage: str, rng: random.Random) -> list:
    from datasets import load_dataset  # imported lazily so --help works without HF

    name = spec["name"]
    split = spec.get("split", "train")
    limit = int(spec.get("limit", 0)) or None
    fmt = spec.get("format", "messages")
    cfg = spec.get("config")  # optional HF config/subset name

    _log(f"loading {name} (split={split}, limit={limit}, format={fmt})")
    try:
        ds = load_dataset(name, cfg, split=split) if cfg else load_dataset(name, split=split)
    except Exception as e:  # noqa: BLE001 — defensive on purpose
        _log(f"  !! SKIP {name}: {type(e).__name__}: {e}")
        return []

    rows = []
    n = len(ds)
    idxs = list(range(n))
    rng.shuffle(idxs)
    if limit:
        idxs = idxs[:limit]

    kept = 0
    for i in idxs:
        row = ds[int(i)]
        if stage == "dpo":
            norm = _normalize_dpo(row, fmt)
            if norm:
                rows.append(norm)
                kept += 1
        else:
            msgs = _normalize_sft(row, fmt)
            if msgs and any(m["role"] == "assistant" for m in msgs):
                rows.append({"messages": msgs})
                kept += 1
    _log(f"  kept {kept}/{len(idxs)} usable rows from {name}")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["sft", "dpo"], default="sft")
    ap.add_argument("--blend", default=str(_DEFAULT_BLEND))
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--max-per-source", type=int, default=0,
                    help="cap each source's row limit (for quick/smoke runs)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    blend = yaml.safe_load(Path(args.blend).read_text(encoding="utf-8")) or {}
    rng = random.Random(args.seed)
    key = "sft_sources" if args.stage == "sft" else "dpo_sources"
    sources = blend.get(key, [])
    if not sources:
        _log(f"no '{key}' in {args.blend}; nothing to do")
        return 2

    all_rows: list = []
    for spec in sources:
        if args.max_per_source:
            cur = int(spec.get("limit") or args.max_per_source)
            spec = {**spec, "limit": min(cur, args.max_per_source)}
        all_rows.extend(_load_source(spec, args.stage, rng))

    if not all_rows:
        _log("ERROR: every source was skipped/empty — check dataset names + HF_TOKEN")
        return 1

    rng.shuffle(all_rows)
    _log(f"total {args.stage} rows: {len(all_rows)}")

    if args.dry_run:
        _log("dry-run: not writing")
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "sft":
        holdout = min(int(blend.get("eval_holdout", 0)), len(all_rows) // 5)
        eval_rows = all_rows[:holdout] if holdout else []
        train_rows = all_rows[holdout:] if holdout else all_rows
        _write_jsonl(out_dir / "sft_blend.train.jsonl", train_rows)
        if eval_rows:
            _write_jsonl(out_dir / "sft_blend.eval.jsonl", eval_rows)
        _log(f"wrote {len(train_rows)} train / {len(eval_rows)} eval to {out_dir}/")
    else:
        _write_jsonl(out_dir / "dpo_pairs.train.jsonl", all_rows)
        _log(f"wrote {len(all_rows)} preference pairs to {out_dir}/")
    return 0


def _write_jsonl(path: Path, rows: list) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    sys.exit(main())
