#!/usr/bin/env python3
"""Merge the final LoRA adapter into the base and export a servable model.

    python -m erosolar_agent.finetune.merge_and_export \
        --adapter outputs/qwen3-32b-dpo \
        --out merged/erosolar-qwen3-32b \
        [--push-to-hub <user>/erosolar-qwen3-32b] \
        [--register erosolar-qwen3-32b]

Produces a merged 16-bit checkpoint (vLLM/HF-servable). Optionally pushes to the
HF Hub (so the serving box can pull it) and appends an entry to the erosolar
model registry with the new HF backend fields.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from unsloth import FastLanguageModel  # noqa: E402


def _log(msg: str) -> None:
    print(f"[merge] {msg}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True, help="final adapter dir (post-DPO)")
    ap.add_argument("--out", required=True, help="output dir for the merged model")
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--save-method", default="merged_16bit",
                    choices=["merged_16bit", "merged_4bit"])
    ap.add_argument("--push-to-hub", default="")
    ap.add_argument("--register", default="", help="registry name to add (optional)")
    ap.add_argument("--base-model", default="unsloth/Qwen3-32B")
    args = ap.parse_args(argv)

    _log(f"loading adapter {args.adapter!r}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
        token=os.environ.get("HF_TOKEN"),
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _log(f"merging -> {out} ({args.save_method})")
    model.save_pretrained_merged(str(out), tokenizer, save_method=args.save_method)
    _log("merge complete")

    if args.push_to_hub:
        _log(f"pushing to hub: {args.push_to_hub}")
        model.push_to_hub_merged(
            args.push_to_hub, tokenizer,
            save_method=args.save_method, token=os.environ.get("HF_TOKEN"),
        )

    entry = {
        "name": args.register or out.name,
        "description": "Qwen3-32B QLoRA SFT+DPO (erosolar agentic general-purpose)",
        "created": datetime.now().isoformat(),
        "backend": "hf-vllm",
        "base_model": args.base_model,
        "hf_model_id": args.push_to_hub or str(out.resolve()),
        "adapter_path": str(Path(args.adapter).resolve()),
        "max_seq_len": args.max_seq_length,
        "tags": ["qwen3", "qlora", "sft", "dpo", "agentic"],
    }
    _log("registry entry:\n" + json.dumps(entry, indent=2))

    if args.register:
        _append_registry(entry)
    return 0


def _append_registry(entry: dict) -> None:
    """Append to models/registry.json if present (additive; preserves existing)."""
    reg_path = Path(__file__).resolve().parents[2] / "models" / "registry.json"
    if not reg_path.exists():
        _log(f"(no {reg_path}; skipping auto-register — add the entry above manually)")
        return
    data = json.loads(reg_path.read_text(encoding="utf-8"))
    data[entry["name"]] = entry
    reg_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _log(f"registered {entry['name']!r} in {reg_path}")


if __name__ == "__main__":
    sys.exit(main())
