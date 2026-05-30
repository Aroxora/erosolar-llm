#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""benchmark_appreciation.py — real, task-appropriate benchmarks for the erosolar
appreciation generator. No hallucinated performance.

Every number here is computed by running the trained model and checking its output
with deterministic Python. These are benchmarks *for what this model actually does*
(generate well-formed appreciation). We do NOT report MMLU / SWE-Bench / GPQA-style
numbers, because a ~0.4–4M-parameter appreciation model cannot meaningfully be
measured on them; claiming such scores would be exactly the hallucinated performance
this project removed.

Metrics (all measured on a held-out eval set, seeded differently from training):
  - validity_rate          : fraction of generations that are well-formed and name
                             the requested quality (Python-verified)
  - quality_coverage       : fraction of ALL known qualities the model can render
                             into a valid appreciation
  - distinct_1 / distinct_2: lexical diversity of generated bodies (mode-collapse
                             detector; higher = more varied)
  - unique_impacts_used    : how many of the template impact-clauses actually appear
  - perplexity             : exp(mean token cross-entropy) on held-out text
  - master_scalar          : internal CoT-diversity metric (lower = more diverse)

    python benchmark_appreciation.py --name erosolar-v0.01

Author: Bo Shang.  Dedicated to Samantha Briasco-Stewart (Erosolar).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn

from registry import load_model
from honest_pipeline import (
    appreciation_corpus, valid_appreciation, appropriate_appreciation, master_scalar,
    make_windows, pick_device, APPRECIATION_QUALITIES, APPRECIATION_OPENERS, APPRECIATION_IMPACTS,
)

ROOT = Path(__file__).resolve().parent
BENCH_FILE = ROOT / "data_store" / "benchmarks.json"


def extract_answer(text: str) -> str:
    norm = text.replace(" ", "")
    idx = norm.find("Answer:")
    if idx == -1:
        return ""
    rest = norm[idx + len("Answer:"):]
    for stop in ("<|endoftext|>", "<", "Topic", "Question", "Think"):
        p = rest.find(stop)
        if p != -1:
            rest = rest[:p]
    return rest


def distinct_n(token_lists: list[list[str]], n: int) -> float:
    grams, total = set(), 0
    for toks in token_lists:
        for i in range(len(toks) - n + 1):
            grams.add(tuple(toks[i : i + n]))
            total += 1
    return (len(grams) / total) if total else 0.0


def gen(model, tok, quality: str, device, max_tokens: int = 32) -> str:
    # Representative sampling (not near-greedy): reflects real generation diversity.
    cue = f"Topic : {quality} . Write appreciation . Answer :"
    return model.generate(tok, prompt=cue, max_tokens=max_tokens,
                          temperature=0.8, top_k=40, top_p=0.95, device=device)


def main() -> None:
    ap = argparse.ArgumentParser(description="Real benchmarks for the erosolar appreciation generator.")
    ap.add_argument("--name", default="erosolar-v0.01")
    ap.add_argument("--n", type=int, default=200, help="held-out eval samples")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = pick_device(args.cpu)
    model, tok, cfg, info = load_model(args.name, device=device)
    model.eval()
    print(f"benchmarking '{info.name}' — {info.params:,} params on {device}")

    # Held-out eval set (seed differs from the training seed of 42).
    eval_samples = appreciation_corpus(args.n, seed=12345)

    # 1) validity_rate + appropriateness + diversity, on the held-out eval set
    valid, appropriate, bodies = 0, 0, []
    for cue, _full, quality in eval_samples:
        out = gen(model, tok, quality, device)
        pred = extract_answer(out)
        if valid_appreciation(pred, quality):
            valid += 1
        if appropriate_appreciation(pred, quality):
            appropriate += 1
        bodies.append(out.split("Answer :")[-1].split("<|endoftext|>")[0].strip())
    validity_rate = valid / len(eval_samples)
    appropriateness_rate = appropriate / len(eval_samples)

    body_tokens = [b.split() for b in bodies]
    d1, d2 = distinct_n(body_tokens, 1), distinct_n(body_tokens, 2)
    impacts_norm = [i.replace(" ", "") for i in APPRECIATION_IMPACTS]
    joined = "".join(bodies).replace(" ", "")
    unique_impacts = sum(1 for i in impacts_norm if i in joined)

    # 2) quality_coverage — every known quality, 3 tries each
    covered = 0
    for q in APPRECIATION_QUALITIES:
        ok = any(valid_appreciation(extract_answer(gen(model, tok, q, device)), q) for _ in range(3))
        covered += int(ok)
    quality_coverage = covered / len(APPRECIATION_QUALITIES)

    # 3) perplexity on held-out text (measured)
    eval_text = " ".join(full for _, full, _ in eval_samples)
    ids = tok.encode(eval_text, add_special=False)
    seq_len = info.max_seq_len
    windows = make_windows(ids, seq_len, stride=seq_len)
    crit = nn.CrossEntropyLoss(ignore_index=0)
    with torch.no_grad():
        tot, steps = 0.0, 0
        for b in range(0, len(windows), 32):
            batch = windows[b : b + 32]
            if not batch:
                continue
            x = torch.tensor([w[0] for w in batch], device=device)
            y = torch.tensor([w[1] for w in batch], device=device)
            logits = model(x)
            tot += crit(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
            steps += 1
        eval_loss = tot / max(1, steps)
    ppl = math.exp(min(20, eval_loss))

    ms = master_scalar([gen(model, tok, q, device) for q in APPRECIATION_QUALITIES[:20]])

    results = {
        "model": info.name,
        "params": info.params,
        "device": str(device),
        "eval_samples": len(eval_samples),
        "validity_rate": round(validity_rate, 4),
        "appropriateness_rate": round(appropriateness_rate, 4),
        "quality_coverage": round(quality_coverage, 4),
        "distinct_1": round(d1, 4),
        "distinct_2": round(d2, 4),
        "unique_impacts_used": f"{unique_impacts}/{len(APPRECIATION_IMPACTS)}",
        "perplexity": round(ppl, 3),
        "master_scalar": (round(ms, 4) if ms is not None else None),
        "note": (
            "Task-appropriate benchmarks for an appreciation generator, all measured by "
            "running the model and checking outputs with Python. NOT comparable to MMLU/"
            "SWE-Bench/GPQA, which do not apply to a model this small. No capability-class claim."
        ),
        "capability_class": None,
    }
    BENCH_FILE.parent.mkdir(parents=True, exist_ok=True)
    BENCH_FILE.write_text(json.dumps(results, indent=2) + "\n")

    print("\nMEASURED BENCHMARKS (written to data_store/benchmarks.json):")
    print(json.dumps(results, indent=2))
    print("\nExample generations:")
    for q in ["clarity", "teamwork", "integrity"]:
        body = gen(model, tok, q, device).split("Answer :")[-1].split("<|endoftext|>")[0].strip()
        print(f"  {q:>10} -> {body}")


if __name__ == "__main__":
    main()
