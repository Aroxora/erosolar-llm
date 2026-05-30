#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Invariant tests that guard the erosolar honesty + data rules.

This repo was rebuilt around one promise: no hallucinated performance, only measured,
license-clean, non-tautological, fully-mapped data. These tests fail loudly if any of
that regresses. Run:  python test_invariants.py   (exit 0 = all pass).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + ("" if cond else f"   -- {detail}"))
    if not cond:
        fails.append(name)


def main() -> int:
    print("erosolar invariant tests\n" + "=" * 50)

    # 1) No fabricated capability-class labels in the version/benchmark artifacts.
    for f in ["data_store/version.json", "version.json", "data_store/benchmarks.json", "data_store/judge_report.json"]:
        d = json.loads((ROOT / f).read_text())
        check(f"{f}: capability_class is null", d.get("capability_class", None) is None)
        blob = json.dumps(d).lower()
        # A bare 'gpt-x'/'superhuman' is only allowed inside an explicit honesty note.
        has_claim = bool(re.search(r"gpt-[4-9]|superhuman", blob))
        excused = any(w in blob for w in ["not a benchmark", "never benchmarked", "no capability", "not a capability"])
        check(f"{f}: no live GPT-x/Superhuman claim", (not has_claim) or excused)

    # 2) Model artifacts honest: no GPT-class capability ladder in the CLI.
    cli = (ROOT / "mini_the_agentic_cli.py").read_text()
    check("CLI: get_model_level capability ladder removed", "def get_model_level" not in cli)
    # 'Superhuman reasoning' is allowed ONLY inside a comment documenting its removal,
    # never in active code (e.g. an assignment to a model_level field).
    active = [ln for ln in cli.splitlines()
              if "Superhuman reasoning" in ln and not ln.lstrip().startswith("#")]
    check("CLI: no ACTIVE 'Superhuman reasoning' label", not active, f"{len(active)} active line(s)")

    # 3) Appreciation data invariants.
    import honest_pipeline as h
    qs = h.APPRECIATION_QUALITIES
    check("every quality is mapped to impacts", all(q in h.QUALITY_IMPACTS for q in qs))
    pool = set(h.APPRECIATION_IMPACTS)
    check("every mapped impact is in the global pool",
          all(i in pool for ims in h.QUALITY_IMPACTS.values() for i in ims))
    check("vocabulary is large (>= 200 qualities)", len(qs) >= 200, f"only {len(qs)}")

    # 4) No tautological impact ever reaches the generated corpus.
    sample = h.appreciation_corpus(3000, seed=99)
    taut = 0
    for _cue, full, q in sample:
        ans = full.split("Answer :", 1)[1]
        if any(h._echoes_quality(q, i) and i in ans for i in h.QUALITY_IMPACTS[q]):
            taut += 1
    check("no tautological impact in generated corpus", taut == 0, f"{taut}/{len(sample)} tautological")

    # 5) Validator consistency: every generated sample is valid AND appropriate.
    bad = 0
    for _cue, full, q in sample[:800]:
        pred = full.split("Answer :", 1)[1].replace(" ", "")
        if not (h.valid_appreciation(pred, q) and h.appropriate_appreciation(pred, q)):
            bad += 1
    check("all generated samples pass valid + appropriate", bad == 0, f"{bad} failed")

    # 6) License-clean: the default corpus is template-composed (no teacher model).
    src = (ROOT / "honest_pipeline.py").read_text()
    check("no proprietary teacher distilled by default",
          "no model distillation" in src.lower() and "teacher_model" in src)

    print("=" * 50)
    if fails:
        print(f"{len(fails)} INVARIANT(S) FAILED: " + ", ".join(fails))
        return 1
    print(f"ALL INVARIANTS PASS ({len(qs)} qualities, pool {len(pool)} impacts)")
    return 0


def test_invariants():
    """pytest entry point — asserts every honesty/data invariant holds."""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
