#!/usr/bin/env python3
"""
Quality Gate - Verifiable correctness checks for training data.

This script enforces a minimal, measurable quality bar by verifying
math/code answers using automated, deterministic checks. It can also
prevent regressions by comparing against a persisted baseline.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from automated_rlhf import OutcomeVerifier


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def extract_prompt_response(item: Dict) -> Optional[Tuple[str, str]]:
    if "messages" in item and isinstance(item["messages"], list):
        prompt = None
        response = None
        for msg in item["messages"]:
            if msg.get("role") == "user" and prompt is None:
                prompt = msg.get("content", "")
            if msg.get("role") == "assistant":
                response = msg.get("content", "")
        if prompt and response:
            return prompt, response
    if "prompt" in item and "response" in item:
        return item.get("prompt", ""), item.get("response", "")
    if "input" in item and "output" in item:
        return item.get("input", ""), item.get("output", "")
    return None


def load_baseline(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        with path.open() as f:
            return json.load(f)
    except Exception:
        return {}


def save_baseline(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify training data quality.")
    parser.add_argument("--input", required=True, help="JSONL input path")
    parser.add_argument("--name", required=True, help="Dataset name for baseline tracking")
    parser.add_argument("--baseline", default="cache/quality_gate/baseline.json",
                        help="Baseline metrics file")
    parser.add_argument("--min-pass-rate", type=float, default=0.6,
                        help="Minimum verified pass rate")
    parser.add_argument("--max-regression", type=float, default=0.01,
                        help="Allowed drop vs baseline")
    parser.add_argument("--min-verifiable", type=int, default=20,
                        help="Minimum verifiable examples required to enforce")
    parser.add_argument("--no-update", action="store_true",
                        help="Do not update baseline on success")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[quality_gate] missing input: {input_path}")
        return 0

    verifier = OutcomeVerifier()
    verifiable = 0
    correct = 0
    skipped = 0

    for item in iter_jsonl(input_path):
        pair = extract_prompt_response(item)
        if not pair:
            skipped += 1
            continue
        prompt, response = pair
        result = verifier.verify(prompt, response)
        if result.method == "unverified":
            skipped += 1
            continue
        verifiable += 1
        if result.is_correct:
            correct += 1

    if verifiable == 0:
        print(f"[quality_gate] no verifiable examples in {input_path}, skipping gate")
        return 0

    pass_rate = correct / verifiable
    print(f"[quality_gate] {args.name}: {correct}/{verifiable} correct "
          f"({pass_rate:.3f}), skipped {skipped}")

    if verifiable < args.min_verifiable:
        print(f"[quality_gate] only {verifiable} verifiable (<{args.min_verifiable}), "
              "skipping regression check")
        return 0

    baseline_path = Path(args.baseline)
    baseline = load_baseline(baseline_path)
    prev = baseline.get(args.name)

    if pass_rate < args.min_pass_rate:
        print(f"[quality_gate] FAIL: pass_rate {pass_rate:.3f} < min {args.min_pass_rate:.3f}")
        return 2

    if prev:
        prev_rate = prev.get("pass_rate", 0.0)
        if pass_rate + args.max_regression < prev_rate:
            print(f"[quality_gate] FAIL: regression {pass_rate:.3f} < {prev_rate:.3f} "
                  f"(max drop {args.max_regression:.3f})")
            return 2

    if not args.no_update:
        baseline[args.name] = {
            "pass_rate": pass_rate,
            "verifiable": verifiable,
            "correct": correct,
            "skipped": skipped,
        }
        save_baseline(baseline_path, baseline)
        print(f"[quality_gate] baseline updated: {baseline_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
