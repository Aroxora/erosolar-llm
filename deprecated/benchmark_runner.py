#!/usr/bin/env python3
"""
Local benchmark runner using LM Eval Harness (offline-friendly).

Usage examples:
  python benchmark_runner.py \
      --model hf \
      --model-args pretrained=Qwen/Qwen2.5-0.5B-Instruct,trust_remote_code=True \
      --tasks gpqa_diamond,arc_challenge,gsm8k,mbpp,humaneval \
      --batch-size 4 \
      --device cuda:0 \
      --offline \
      --data-dir /path/to/hf_cache \
      --output-path cache/benchmarks/results.json
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List


def _load_tasks_from_file(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Tasks file not found: {path}")
    text = path.read_text().strip()
    if not text:
        return []
    if path.suffix in {".json", ".jsonl"}:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(item) for item in data]
    return [line.strip() for line in text.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LM Eval Harness locally")
    parser.add_argument("--model", type=str, default="hf",
                        help="LM Eval model backend (default: hf)")
    parser.add_argument("--model-args", type=str, default="",
                        help="LM Eval model args (e.g., pretrained=...,trust_remote_code=True)")
    parser.add_argument("--tasks", type=str, default="",
                        help="Comma-separated LM Eval task list")
    parser.add_argument("--tasks-file", type=str, default="",
                        help="Path to a tasks file (json or newline list)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device string (e.g., cuda:0, cpu)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for evaluation")
    parser.add_argument("--num-fewshot", type=int, default=0,
                        help="Few-shot count")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of examples per task")
    parser.add_argument("--offline", action="store_true",
                        help="Set HF_DATASETS_OFFLINE=1 for local-only runs")
    parser.add_argument("--data-dir", type=str, default="",
                        help="Local HF datasets cache or data dir")
    parser.add_argument("--output-path", type=str, default="cache/benchmarks/results.json",
                        help="Output path for LM Eval results")
    args = parser.parse_args()

    tasks = []
    if args.tasks_file:
        tasks = _load_tasks_from_file(Path(args.tasks_file))
    elif args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    if not tasks:
        print("Error: no tasks specified (use --tasks or --tasks-file).")
        return 1

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", args.model,
        "--tasks", ",".join(tasks),
        "--device", args.device,
        "--batch_size", str(args.batch_size),
        "--output_path", str(output_path),
    ]
    if args.model_args:
        cmd += ["--model_args", args.model_args]
    if args.num_fewshot:
        cmd += ["--num_fewshot", str(args.num_fewshot)]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.data_dir:
        cmd += ["--data_dir", args.data_dir]

    env = os.environ.copy()
    if args.offline:
        env["HF_DATASETS_OFFLINE"] = "1"
        env["HF_HUB_OFFLINE"] = "1"
        if args.data_dir:
            env["HF_HOME"] = args.data_dir

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, env=env)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
