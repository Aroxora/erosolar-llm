#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""honest_pipeline.py — the erosolar pipeline, without hallucination.

Full path: synthesize (license-clean) data -> train -> measure -> infer -> report.

Design principles (see DEDICATION.md for why):
  1. NO hallucinated performance. Every number written by this script is measured
     during the run. Capability-class labels ("GPT-x", "Superhuman") are never
     emitted. Before a run completes, result fields are written as the string
     "pending" so nothing is ever claimed before it is measured.
  2. License-clean training data. The default corpus is generated deterministically
     by this file (arithmetic / comparison / parity problems whose answers are
     computed by Python). It is therefore correct-by-construction (no hallucinated
     labels) and distilled from no other model, so it cannot violate any model's
     distillation/terms-of-use clause.
  3. Optional open-licensed teacher. Higher-quality synthetic data may be drawn
     from an OPEN-LICENSED model on AWS Bedrock (e.g. DeepSeek-R1, MIT-licensed)
     via --bedrock. Proprietary models are never used as teachers. If credentials
     are unavailable the step is recorded as "pending", never faked.

Run:
    python honest_pipeline.py --quick          # ~1-2 min on CPU/MPS, tiny model
    python honest_pipeline.py --epochs 4
    python honest_pipeline.py --bedrock        # use DeepSeek-R1 (MIT) on Bedrock

Author: Bo Shang.  Dedicated to Samantha Briasco-Stewart (Erosolar).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from itertools import combinations
from pathlib import Path

import torch
import torch.nn as nn

from model import MiniGPT, ModelConfig
from config import Config, TrainingConfig
from tokenizer import BPETokenizer
from registry import ModelRegistry

ROOT = Path(__file__).resolve().parent
DATA_VERSION = ROOT / "data_store" / "version.json"
TOP_VERSION = ROOT / "version.json"

MASTER_SCALAR_NOTE = (
    "Internal CoT reasoning-diversity metric (lower = more diverse). "
    "NOT a benchmark and NOT a capability claim."
)


# ───────────────────────── license-clean synthetic data ─────────────────────────

def synth_corpus(n: int, seed: int = 42) -> list[tuple[str, str, str]]:
    """Deterministically generate (prompt, full_text, true_answer) triples.

    Labels are computed by Python, so they are correct by construction — there is
    no teacher model and no possibility of a hallucinated label. Numbers are space-
    separated per digit so a small word-level tokenizer can learn the pattern.
    """
    rng = random.Random(seed)
    samples: list[tuple[str, str, str]] = []

    def spaced(x: int) -> str:
        return " ".join(str(x))

    for _ in range(n):
        kind = rng.choice(["add", "sub", "cmp", "parity"])
        if kind == "add":
            a, b = rng.randint(0, 99), rng.randint(0, 99)
            ans = str(a + b)
            q = f"Question : what is {spaced(a)} plus {spaced(b)} ?"
            cot = "Think : add the ones , then the tens ."
        elif kind == "sub":
            a, b = rng.randint(0, 99), rng.randint(0, 99)
            a, b = max(a, b), min(a, b)
            ans = str(a - b)
            q = f"Question : what is {spaced(a)} minus {spaced(b)} ?"
            cot = "Think : subtract the smaller from the larger ."
        elif kind == "cmp":
            a, b = rng.randint(0, 99), rng.randint(0, 99)
            ans = "greater" if a > b else ("less" if a < b else "equal")
            q = f"Question : is {spaced(a)} greater or less than {spaced(b)} ?"
            cot = "Think : compare the tens , then the ones ."
        else:  # parity
            a = rng.randint(0, 99)
            ans = "even" if a % 2 == 0 else "odd"
            q = f"Question : is {spaced(a)} even or odd ?"
            cot = "Think : look at the last digit ."
        full = f"{q} {cot} Answer : {' '.join(ans)} <|endoftext|>"
        samples.append((q, full, ans))
    return samples


# Wholesome, general appreciation vocabulary. Deliberately about qualities and
# craft — gratitude one could sincerely offer any colleague — not about any one
# person. The dedication to Erosolar lives, tastefully, in DEDICATION.md only.
CURATED_QUALITIES = [
    "care", "craft", "effort", "patience", "clarity", "curiosity",
    "kindness", "rigor", "generosity", "courage", "focus", "honesty",
    "diligence", "creativity", "leadership", "humility", "persistence",
    "insight", "warmth", "integrity", "dedication", "attention",
    "teamwork", "optimism",
]
APPRECIATION_OPENERS = [
    "thank you for", "i am grateful for", "i really appreciate",
    "i want to thank you for", "we are thankful for", "i'm so thankful for",
    "i deeply appreciate", "many thanks for",
]
# Impacts are mapped to the qualities they actually FIT, so the appreciation is
# coherent ("clarity ... made things clearer") rather than mismatched ("kindness ...
# made the deadline reachable"). This fixes the quality-agnostic-impact weakness the
# LLM judge surfaced. Each quality keeps 4 fitting impacts, so variety stays high.
QUALITY_IMPACTS = {
    "care":        ["helped the whole team", "made everyone feel welcome", "lifted the whole mood", "made someone's day"],
    "craft":       ["made the work better", "made the result stronger", "raised the bar", "made the review smoother"],
    "effort":      ["moved the project forward", "made the deadline reachable", "made the hard part easier", "kept us on track"],
    "patience":    ["made the hard part easier", "kept us on track", "kept everyone calm", "gave us room to get it right"],
    "clarity":     ["made things clearer", "made the review smoother", "saved us a lot of time", "made the work better"],
    "curiosity":   ["moved the project forward", "made the result stronger", "inspired the rest of us", "made things clearer"],
    "kindness":    ["made everyone feel welcome", "lifted the whole mood", "helped the whole team", "made someone's day"],
    "rigor":       ["made the work better", "raised the bar", "made the result stronger", "caught the problems early"],
    "generosity":  ["helped the whole team", "made everyone feel welcome", "lifted the whole mood", "shared the load"],
    "courage":     ["set a good example", "inspired the rest of us", "raised the bar", "turned a hard week around"],
    "focus":       ["moved the project forward", "kept us on track", "made the deadline reachable", "saved us a lot of time"],
    "honesty":     ["earned our trust", "set a good example", "kept us honest", "made things clearer"],
    "diligence":   ["moved the project forward", "kept us on track", "made the deadline reachable", "made the work better"],
    "creativity":  ["made the result stronger", "inspired the rest of us", "moved the project forward", "raised the bar"],
    "leadership":  ["set a good example", "kept us on track", "inspired the rest of us", "moved the project forward"],
    "humility":    ["made everyone feel welcome", "set a good example", "shared the load", "helped the whole team"],
    "persistence": ["made the deadline reachable", "made the hard part easier", "kept us on track", "turned a hard week around"],
    "insight":     ["made things clearer", "made the result stronger", "caught the problems early", "saved us a lot of time"],
    "warmth":      ["made everyone feel welcome", "lifted the whole mood", "helped the whole team", "made someone's day"],
    "integrity":   ["set a good example", "earned our trust", "raised the bar", "kept us honest"],
    "dedication":  ["moved the project forward", "made the deadline reachable", "kept us on track", "made the result stronger"],
    "attention":   ["made the review smoother", "made things clearer", "caught the problems early", "made the work better"],
    "teamwork":    ["helped the whole team", "made everyone feel welcome", "kept us on track", "moved the project forward"],
    "optimism":    ["lifted the whole mood", "turned a hard week around", "inspired the rest of us", "made everyone feel welcome"],
}
# Global pool (union) — used by the structural validator.
APPRECIATION_IMPACTS = sorted({i for impacts in QUALITY_IMPACTS.values() for i in impacts})
# Optional second sentence — adds lexical variety.
APPRECIATION_CLOSERS = [
    "it did not go unnoticed", "thank you again", "it meant a lot to us",
    "please keep it up", "the team noticed", "it made a real difference",
    "we see it", "that is rare",
]

# Large vetted vocabulary of wholesome, single-word positive qualities (brainstormed
# across 12 categories and critic-vetted). Training on ~230 distinct qualities forces
# the model to COPY the requested quality from the cue into the answer instead of
# memorizing a handful of associations — which is what lets it generalize to qualities
# it never saw in an appreciation. The curated 24 above keep fitting impacts; these draw
# from the generic pool.
EXTRA_QUALITIES = [
    "acceptance", "accessibility", "accountability", "accuracy", "adaptability", "altruism",
    "ambition", "approachability", "articulateness", "artistry", "assertiveness", "astuteness",
    "attentiveness", "authenticity", "balance", "benevolence", "boldness", "brightness",
    "brilliance", "calmness", "candor", "carefulness", "centeredness", "charity", "cheerfulness",
    "civility", "cleanliness", "cleverness", "coachability", "coherence", "collaboration",
    "comfort", "commitment", "compassion", "competence", "composure", "comprehension",
    "concision", "confidence", "conscientiousness", "consideration", "consistency", "constancy",
    "conviction", "cooperation", "credibility", "daring", "decency", "decisiveness", "deference",
    "dependability", "determination", "devotion", "dexterity", "diplomacy", "directness",
    "discernment", "discipline", "discretion", "doggedness", "drive", "dynamism", "eagerness",
    "earnestness", "elegance", "eloquence", "empathy", "empowerment", "encouragement", "endurance",
    "energy", "enthusiasm", "equanimity", "equity", "erudition", "ethics", "expertise",
    "expressiveness", "exuberance", "fairness", "fidelity", "finesse", "flexibility", "foresight",
    "fortitude", "freshness", "friendliness", "geniality", "genius", "gentleness", "graciousness",
    "gratitude", "grit", "groundedness", "growth", "helpfulness", "honor", "hopefulness",
    "hospitality", "humanity", "imagination", "impartiality", "industriousness", "ingenuity",
    "initiative", "innovation", "inspiration", "intelligence", "introspection", "intuition",
    "inventiveness", "joyfulness", "judgment", "justice", "knowledge", "liveliness", "logic",
    "loyalty", "magnanimity", "mastery", "mentorship", "meticulousness", "mindfulness", "modesty",
    "morality", "motivation", "neatness", "neighborliness", "nurturance", "objectivity", "openness",
    "orderliness", "originality", "ownership", "passion", "perceptiveness", "perseverance",
    "perspective", "philanthropy", "playfulness", "poise", "positivity", "precision", "preparedness",
    "presence", "principle", "proactivity", "professionalism", "proficiency", "prudence",
    "punctuality", "rapport", "rationality", "reassurance", "receptiveness", "refinement",
    "reflectiveness", "reliability", "resilience", "resolve", "resourcefulness", "respect",
    "responsibility", "responsiveness", "restraint", "scholarship", "selflessness", "sensitivity",
    "serenity", "service", "sincerity", "skill", "solidarity", "spontaneity", "stability", "stamina",
    "steadfastness", "stewardship", "stoicism", "supportiveness", "sympathy", "tact", "teachability",
    "temperance", "tenacity", "tenderness", "thoroughness", "thoughtfulness", "tidiness", "tolerance",
    "tranquility", "transparency", "trustworthiness", "understanding", "versatility", "vibrancy",
    "vigilance", "vigor", "vision", "vitality", "willingness", "wisdom", "wit", "wonder", "zest",
]
# Full quality set: curated (with fitting impacts) + the large vetted vocabulary.
_curated_set = set(CURATED_QUALITIES)
APPRECIATION_QUALITIES = CURATED_QUALITIES + [q for q in EXTRA_QUALITIES if q not in _curated_set]

# 10 semantic impact-families (each maps to fitting impacts from the pool above) and the
# assignment of every extra quality to a family (a single agent categorized all 205,
# validated: each assigned exactly once). This gives APPROPRIATE impacts for ALL 229
# qualities, not just the curated 24.
IMPACT_CATEGORIES = {
    "interpersonal": ["helped the whole team", "made everyone feel welcome", "lifted the whole mood", "made someone's day"],
    "execution":     ["moved the project forward", "made the deadline reachable", "kept us on track", "made the hard part easier"],
    "clarity":       ["made things clearer", "made the review smoother", "saved us a lot of time", "caught the problems early"],
    "craft":         ["made the work better", "made the result stronger", "raised the bar", "caught the problems early"],
    "integrity":     ["earned our trust", "set a good example", "kept us honest", "raised the bar"],
    "leadership":    ["set a good example", "inspired the rest of us", "raised the bar", "turned a hard week around"],
    "creativity":    ["made the result stronger", "inspired the rest of us", "moved the project forward", "made things clearer"],
    "calm":          ["kept everyone calm", "gave us room to get it right", "made the hard part easier", "made everyone feel welcome"],
    "resilient":     ["turned a hard week around", "made the hard part easier", "kept us on track", "made the deadline reachable"],
    "generosity":    ["helped the whole team", "made everyone feel welcome", "shared the load", "made someone's day"],
    "positivity":    ["lifted the whole mood", "turned a hard week around", "inspired the rest of us", "made someone's day"],
}
CATEGORY_QUALITIES = {
    "interpersonal": ["acceptance", "approachability", "attentiveness", "benevolence", "civility", "compassion", "consideration", "deference", "empathy", "friendliness", "geniality", "gentleness", "graciousness", "neighborliness", "nurturance", "rapport", "respect", "sensitivity", "sympathy", "tact", "tenderness", "thoughtfulness", "tolerance", "understanding", "diplomacy", "comfort", "reassurance", "humanity"],
    "execution":     ["accountability", "commitment", "conscientiousness", "consistency", "constancy", "dependability", "determination", "devotion", "discipline", "doggedness", "drive", "earnestness", "fidelity", "grit", "industriousness", "loyalty", "motivation", "perseverance", "preparedness", "proactivity", "professionalism", "punctuality", "reliability", "resolve", "responsibility", "responsiveness", "steadfastness", "tenacity", "willingness", "eagerness", "vigilance"],
    "clarity":       ["articulateness", "astuteness", "coherence", "comprehension", "concision", "directness", "discernment", "eloquence", "expressiveness", "intelligence", "intuition", "judgment", "knowledge", "logic", "objectivity", "perceptiveness", "perspective", "precision", "rationality", "wisdom", "introspection", "reflectiveness"],
    "craft":         ["accuracy", "carefulness", "cleanliness", "competence", "dexterity", "elegance", "expertise", "finesse", "mastery", "meticulousness", "neatness", "orderliness", "proficiency", "refinement", "skill", "thoroughness", "tidiness", "versatility", "brilliance", "genius", "erudition", "scholarship", "openness", "receptiveness", "coachability", "teachability", "growth"],
    "integrity":     ["authenticity", "candor", "credibility", "decency", "equity", "ethics", "fairness", "honor", "impartiality", "justice", "morality", "principle", "sincerity", "transparency", "trustworthiness", "prudence", "modesty", "restraint", "temperance", "discretion"],
    "leadership":    ["ambition", "assertiveness", "boldness", "confidence", "conviction", "daring", "decisiveness", "initiative", "ownership", "vision", "foresight", "empowerment", "inspiration"],
    "creativity":    ["artistry", "cleverness", "imagination", "ingenuity", "innovation", "inventiveness", "originality", "playfulness", "resourcefulness", "spontaneity", "wit", "wonder", "freshness", "brightness"],
    "calm":          ["calmness", "centeredness", "composure", "equanimity", "groundedness", "poise", "presence", "serenity", "stability", "tranquility", "stoicism", "balance", "mindfulness"],
    "resilient":     ["adaptability", "endurance", "flexibility", "fortitude", "resilience", "stamina"],
    "generosity":    ["altruism", "charity", "collaboration", "cooperation", "encouragement", "helpfulness", "hospitality", "magnanimity", "mentorship", "philanthropy", "selflessness", "service", "solidarity", "stewardship", "supportiveness", "accessibility"],
    "positivity":    ["cheerfulness", "dynamism", "energy", "enthusiasm", "exuberance", "gratitude", "hopefulness", "joyfulness", "liveliness", "passion", "positivity", "vibrancy", "vigor", "vitality", "zest"],
}
# Extend QUALITY_IMPACTS to cover all 229 (curated keep their explicit fitting impacts).
for _cat, _qs in CATEGORY_QUALITIES.items():
    for _q in _qs:
        QUALITY_IMPACTS.setdefault(_q, IMPACT_CATEGORIES[_cat])


def appreciation_corpus(n: int, seed: int = 42, qualities: list[str] | None = None) -> list[tuple[str, str, str]]:
    """Deterministically generate (cue, full_text, quality) appreciation samples.

    License-clean by construction: these are template-composed sentences, distilled
    from no model. The model learns to produce a well-formed appreciation conditioned
    on a requested quality, choosing among several openers and impacts. Grading checks
    structural validity with Python, so the reported metric is measured, not claimed.

    `qualities` restricts which qualities are sampled (used to hold qualities out of
    training for a genuine generalization test).
    """
    qs = list(qualities) if qualities else APPRECIATION_QUALITIES
    rng = random.Random(seed)
    samples: list[tuple[str, str, str]] = []
    for _ in range(n):
        quality = rng.choice(qs)
        # Curated qualities get a fitting impact; the large extra vocabulary (which
        # exists to force the model to COPY the quality from the cue and thus
        # generalize) draws from the generic pool.
        impact = rng.choice(QUALITY_IMPACTS.get(quality, APPRECIATION_IMPACTS))
        opener = rng.choice(APPRECIATION_OPENERS)
        cue = f"Topic : {quality} . Write appreciation ."
        r = rng.random()
        if r < 0.4:
            body = f"{opener} your {quality} . it {impact} ."      # T1: opener form
        elif r < 0.75:
            body = f"your {quality} {impact} ."                     # T2: direct form
        else:
            body = f"your {quality} really shows . it {impact} ."  # T3: "really shows" form
        if rng.random() < 0.5:  # optional closer for structural variety
            body += f" {rng.choice(APPRECIATION_CLOSERS)} ."
        full = f"{cue} Answer : {body} <|endoftext|>"
        samples.append((cue, full, quality))
    return samples


def valid_appreciation(pred_norm: str, quality: str) -> bool:
    """Whether a (whitespace-stripped) generation is a well-formed appreciation that
    names the requested quality, in either sentence structure. Pure-Python and
    deterministic — so any rate computed with it is measured, not claimed."""
    q = quality.replace(" ", "")
    impacts = [i.replace(" ", "") for i in APPRECIATION_IMPACTS]
    if not any(i in pred_norm for i in impacts):
        return False
    # T1: "<opener> your <quality> . ..."
    for op in APPRECIATION_OPENERS:
        if pred_norm.startswith(op.replace(" ", "") + "your" + q + "."):
            return True
    # T2: "your <quality> <impact> ..."
    return pred_norm.startswith("your" + q)


def appropriate_appreciation(pred_norm: str, quality: str) -> bool:
    """Stricter than valid: the impact used must actually FIT the named quality
    (semantic appropriateness), addressing the quality-agnostic-impact weakness."""
    if quality.replace(" ", "") not in pred_norm:
        return False
    return any(i.replace(" ", "") in pred_norm for i in QUALITY_IMPACTS.get(quality, []))


def open_weights_synth(n: int, model_id: str, region: str) -> tuple[list | None, str]:
    """Optionally draw synthetic data from an OPEN-WEIGHTS model on AWS Bedrock.

    Policy (set by the project owner): use the 'deepseek-v4-pro' open weights, or
    no teacher model at all. We never substitute a different model (open or
    proprietary), never distill from a model whose license forbids it, and never
    fabricate samples. If the requested open-weights model is not available on
    Bedrock, we use NO teacher and the pipeline falls back to its deterministic,
    self-generated, license-clean corpus (zero distillation).

    Returns (samples_or_None, status_string).
    """
    try:
        import boto3
    except Exception:
        return None, "skipped: boto3 not installed; using self-generated data (no distillation)."
    try:
        br = boto3.client("bedrock", region_name=region)
        ids = {m["modelId"] for m in br.list_foundation_models().get("modelSummaries", [])}
        if model_id not in ids:
            present = ", ".join(sorted(i for i in ids if "deepseek" in i)) or "none"
            return None, (
                f"unavailable: open-weights model '{model_id}' was not found on Bedrock "
                f"(DeepSeek models present: {present}). Per the 'deepseek-v4-pro or nothing' "
                f"policy, no teacher was substituted; using self-generated license-clean data."
            )
        # If the requested model existed, license-clean generation would run here.
        return None, f"available: '{model_id}' present on Bedrock; generation pass not yet run."
    except Exception as e:
        return None, f"skipped: Bedrock check failed ({type(e).__name__}); using self-generated data (no distillation)."


# ───────────────────────── master scalar (dependency-free) ─────────────────────────

def _hash_embed(text: str, dim: int = 64) -> list[float]:
    """Tiny deterministic hashing embedding (no proprietary model, no distillation)."""
    v = [0.0] * dim
    for tok in text.split():
        v[hash(tok) % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def master_scalar(texts: list[str]) -> float | None:
    """Average pairwise cosine similarity of CoT texts (lower = more diverse).

    Uses a lightweight built-in hashing embedding so the metric is computable
    with zero external models. For the sentence-transformers variant, see
    master_scalar.py. Returns None if there are fewer than two texts.
    """
    embs = [_hash_embed(t) for t in texts if t.strip()]
    pairs = list(combinations(range(len(embs)), 2))
    if not pairs:
        return None
    total = 0.0
    for i, j in pairs:
        total += sum(x * y for x, y in zip(embs[i], embs[j]))
    return total / len(pairs)


# ───────────────────────── training (self-contained) ─────────────────────────

def make_windows(ids: list[int], seq_len: int, stride: int) -> list[tuple[list[int], list[int]]]:
    out = []
    for s in range(0, max(1, len(ids) - seq_len), stride):
        chunk = ids[s : s + seq_len + 1]
        if len(chunk) < 2:
            continue
        out.append((chunk[:-1], chunk[1:]))
    return out


def pick_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def write_version(status: str, measured: dict | None = None) -> None:
    """Write honest version files. `status` is 'pending' or 'measured'."""
    info = {
        "version": 1 if status == "measured" else 0,
        "version_string": "v0.01" if status == "measured" else "v0.00",
        "model_name": "erosolar-v0.01" if status == "measured" else None,
        "status": status,
        "master_scalar": (measured or {}).get("master_scalar", "pending"),
        "master_scalar_note": MASTER_SCALAR_NOTE,
        "measured": measured if status == "measured" else "pending",
        "capability_class": None,
        "training_data": "license-clean synthetic (deterministic, Python-verified labels; no model distillation)",
        "updated": "2026-05-29",
    }
    DATA_VERSION.parent.mkdir(parents=True, exist_ok=True)
    DATA_VERSION.write_text(json.dumps(info, indent=2) + "\n")


SIZES = {
    "tiny":  dict(embed_dim=96,  num_layers=2, num_heads=4, ff_dim=256,  seq_len=48),
    "small": dict(embed_dim=128, num_layers=3, num_heads=4, ff_dim=256,  seq_len=64),
    "base":  dict(embed_dim=256, num_layers=6, num_heads=8, ff_dim=1024, seq_len=96),
    "large": dict(embed_dim=384, num_layers=8, num_heads=8, ff_dim=1536, seq_len=128),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Honest erosolar pipeline (no hallucinated performance).")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--samples", type=int, default=4000)
    ap.add_argument("--quick", action="store_true", help="tiny/fast run for verification")
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    ap.add_argument("--bedrock", action="store_true", help="draw data from an open-licensed Bedrock model (DeepSeek-R1, MIT)")
    ap.add_argument("--bedrock-model", default="deepseek.r1-v1:0")
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    ap.add_argument("--name", default="erosolar-v0.01")
    ap.add_argument("--task", choices=["appreciation", "math"], default="appreciation",
                    help="appreciation = wholesome gratitude generator (default); math = grounded arithmetic demo")
    ap.add_argument("--size", choices=list(SIZES), default="base",
                    help="model size preset (default: base)")
    ap.add_argument("--holdout", type=int, default=0,
                    help="hold out the last N qualities from TRAINING to measure generalization "
                         "(appreciation only); their words are still seeded into the tokenizer")
    args = ap.parse_args()

    if args.quick:
        args.epochs, args.samples, args.size = 2, 1500, "tiny"

    # Held-out generalization experiment: train on a subset of qualities, test on the rest.
    held_out: list[str] = []
    train_qualities = list(APPRECIATION_QUALITIES)
    if args.holdout > 0 and args.task == "appreciation":
        held_out = list(APPRECIATION_QUALITIES[-args.holdout:])
        train_qualities = list(APPRECIATION_QUALITIES[: -args.holdout])
    # Only the canonical full run owns the production version.json.
    canonical = (args.name == "erosolar-v0.01" and not held_out)

    print("=" * 64)
    print("erosolar — honest pipeline (run before claiming any results)")
    print("=" * 64)

    # Mark results pending BEFORE running anything (canonical run only).
    if canonical:
        write_version("pending")
        print("[0/5] version.json -> status: pending (nothing claimed yet)")
    if held_out:
        print(f"[0/5] HOLDOUT experiment: training on {len(train_qualities)} qualities, "
              f"holding out {held_out} (model '{args.name}', production version.json untouched)")

    # 1. Data — license-clean by construction.
    bedrock_status = "not requested"
    if args.bedrock:
        _, bedrock_status = open_weights_synth(args.samples, args.bedrock_model, args.region)
        print(f"[1/5] open-weights data: {bedrock_status}")
    if args.task == "appreciation":
        samples = appreciation_corpus(args.samples, qualities=train_qualities)
        task_label = "appreciation (wholesome gratitude)" + (f" [holding out {len(held_out)}]" if held_out else "")
    else:
        samples = synth_corpus(args.samples)
        task_label = "math (grounded arithmetic)"
    split = int(len(samples) * 0.9)
    train_s, val_s = samples[:split], samples[split:]
    print(f"[1/5] synthesized {len(samples)} license-clean {task_label} samples ({len(train_s)} train / {len(val_s)} val)")

    # 2. Tokenizer (word-level; trained on a single joined string).
    corpus_text = " ".join(full for _, full, _ in train_s)
    # The tokenizer sees the held-out quality WORDS (so they aren't <unk> at eval), but the
    # MODEL never trains on a held-out appreciation — corpus_text below has no held-out data.
    tokenizer_text = corpus_text + (" " + " ".join(held_out * 40) if held_out else "")
    tok = BPETokenizer()
    tok.train(tokenizer_text, vocab_size=1024)  # headroom for the ~230-word quality vocab
    print(f"[2/5] tokenizer trained: vocab_size={tok.vocab_size}")

    # 3. Model (size preset).
    sz = SIZES[args.size]
    seq_len = sz["seq_len"]
    mc = ModelConfig(
        vocab_size=tok.vocab_size, max_seq_len=seq_len,
        embed_dim=sz["embed_dim"], num_heads=sz["num_heads"],
        num_layers=sz["num_layers"], ff_dim=sz["ff_dim"], dropout=0.0,
    )
    device = pick_device(args.cpu)
    model = MiniGPT(mc).to(device)
    print(f"[3/5] model: MiniGPT '{args.size}' {model.get_num_params():,} params on {device}")

    # 4. Train (real loop; loss is measured, computed externally).
    train_ids = tok.encode(corpus_text, add_special=False)
    val_text = " ".join(full for _, full, _ in val_s)
    val_ids = tok.encode(val_text, add_special=False)
    windows = make_windows(train_ids, seq_len, stride=seq_len // 2)
    val_windows = make_windows(val_ids, seq_len, stride=seq_len)
    crit = nn.CrossEntropyLoss(ignore_index=0)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    bs = 32
    t0 = time.time()
    last_loss = float("nan")
    for ep in range(args.epochs):
        model.train()
        random.Random(ep).shuffle(windows)
        running, steps = 0.0, 0
        for b in range(0, len(windows), bs):
            batch = windows[b : b + bs]
            if not batch:
                continue
            x = torch.tensor([w[0] for w in batch], device=device)
            y = torch.tensor([w[1] for w in batch], device=device)
            logits = model(x)
            loss = crit(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item()
            steps += 1
        last_loss = running / max(1, steps)
        print(f"      epoch {ep+1}/{args.epochs}  train_loss={last_loss:.4f}")
    train_secs = time.time() - t0

    # Validation loss (measured, held-out).
    model.eval()
    with torch.no_grad():
        vtotal, vsteps = 0.0, 0
        for b in range(0, len(val_windows), bs):
            batch = val_windows[b : b + bs]
            if not batch:
                continue
            x = torch.tensor([w[0] for w in batch], device=device)
            y = torch.tensor([w[1] for w in batch], device=device)
            logits = model(x)
            vtotal += crit(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
            vsteps += 1
        val_loss = vtotal / max(1, vsteps)
    print(f"[4/5] trained in {train_secs:.1f}s  val_loss={val_loss:.4f}  ppl={math.exp(min(20, val_loss)):.2f}")

    # 5. Inference + GROUNDED accuracy (answers are Python-checkable -> measured, not claimed).
    def extract_answer(text: str) -> str:
        """Pull the model's first answer, robust to decoder spacing ('Answer:'/'Answer :')."""
        norm = text.replace(" ", "")
        idx = norm.find("Answer:")
        if idx == -1:
            return ""
        rest = norm[idx + len("Answer:"):]
        for stop in ("<|endoftext|>", "<", "Question", "Think"):
            p = rest.find(stop)
            if p != -1:
                rest = rest[:p]
        return rest

    correct, tried, gen_texts = 0, 0, []
    for cue, _full, truth_raw in val_s[:60]:
        prompt = (cue + " Answer :") if args.task == "appreciation" else (cue + " Think :")
        try:
            out = model.generate(tok, prompt=prompt, max_tokens=32, temperature=0.7, top_k=40, top_p=0.95, device=device)
        except Exception as e:
            out = f"(generation error: {e})"
        gen_texts.append(out)
        pred = extract_answer(out)
        truth = truth_raw.replace(" ", "")
        if args.task == "appreciation":
            ok = valid_appreciation(pred, truth_raw)
        else:
            ok = (pred == truth or pred.startswith(truth))
        if ok:
            correct += 1
        tried += 1
    grounded_acc = correct / tried if tried else None
    ms = master_scalar(gen_texts)
    metric_name = "appreciation validity" if args.task == "appreciation" else "answer accuracy"
    line = f"[5/5] grounded {metric_name} on held-out: {correct}/{tried}"
    if grounded_acc is not None:
        line += f" = {grounded_acc:.1%}"
    if ms is not None:
        line += f"   master_scalar={ms:.4f}"
    print(line)
    print(f"      sample generation: {gen_texts[0][:120] if gen_texts else '(none)'!r}")

    # 5b. Held-out generalization: validity on qualities the model NEVER saw in training.
    held_out_validity = None
    held_out_examples: list[dict] = []
    if held_out:
        ho = appreciation_corpus(120, seed=999, qualities=held_out)
        hc = 0
        for cue, _f, quality in ho:
            try:
                out = model.generate(tok, prompt=cue + " Answer :", max_tokens=32,
                                     temperature=0.7, top_k=40, top_p=0.95, device=device)
            except Exception:
                out = ""
            if valid_appreciation(extract_answer(out), quality):
                hc += 1
            if len(held_out_examples) < 5:
                held_out_examples.append(
                    {"quality": quality, "text": out.split("Answer :")[-1].split("<|endoftext|>")[0].strip()})
        held_out_validity = hc / len(ho)
        print(f"[5b]  HELD-OUT generalization validity {held_out}: {hc}/{len(ho)} = {held_out_validity:.1%}")
        for ex in held_out_examples[:3]:
            print(f"       [{ex['quality']}] {ex['text'][:90]!r}")

    # Persist checkpoint via the real registry (only measured metadata).
    try:
        cfg = Config(
            vocab_size=tok.vocab_size, max_seq_len=seq_len, embed_dim=mc.embed_dim,
            num_heads=mc.num_heads, num_layers=mc.num_layers, ff_dim=mc.ff_dim,
            dropout=0.0, name=args.name, training=TrainingConfig(epochs=args.epochs, batch_size=bs),
        )
        reg = ModelRegistry()
        reg.save_model(args.name, "Honest pipeline run (license-clean synthetic data)",
                       model, tok, cfg, epochs=args.epochs, loss=last_loss,
                       training_time=train_secs / 60.0, tags=["honest", "license-clean"], preset="honest-small")
        saved = True
    except Exception as e:
        saved = False
        print(f"      (registry save skipped: {e})")

    measured = {
        "task": args.task,
        "params": model.get_num_params(),
        "train_loss": round(last_loss, 4),
        "val_loss": round(val_loss, 4),
        "val_perplexity": round(math.exp(min(20, val_loss)), 2),
        "grounded_metric": metric_name,
        "grounded_answer_accuracy": (round(grounded_acc, 4) if grounded_acc is not None else None),
        "grounded_accuracy_note": (
            "Fraction of held-out appreciation generations that are well-formed and name the "
            "requested quality, verified by Python. A real, modest number from a tiny model — not a capability claim."
            if args.task == "appreciation" else
            "Fraction of held-out questions answered correctly; answers verified by Python. "
            "A real, modest number from a tiny model — not a capability claim."
        ),
        "master_scalar": (round(ms, 4) if ms is not None else None),
        "epochs": args.epochs,
        "train_seconds": round(train_secs, 1),
        "device": str(device),
        "checkpoint_saved": saved,
        "bedrock_open_model_data": bedrock_status,
        "teacher_model": "none (deterministic Python-verified synthetic data; no distillation)",
    }
    if canonical:
        write_version("measured", measured)
        print("\nMEASURED RESULTS (written to data_store/version.json):")
        print(json.dumps(measured, indent=2))
    if held_out:
        gen_report = {
            "experiment": "held-out generalization",
            "model": args.name,
            "params": model.get_num_params(),
            "trained_on_qualities": len(train_qualities),
            "held_out_qualities": held_out,
            "held_out_validity": (round(held_out_validity, 4) if held_out_validity is not None else None),
            "in_distribution_validity": (round(grounded_acc, 4) if grounded_acc is not None else None),
            "held_out_examples": held_out_examples,
            "note": (
                "Validity on qualities whose appreciation the model NEVER saw in training (only "
                "their words were seeded into the tokenizer). Measures whether the appreciation "
                "pattern generalizes to unseen qualities. The production model erosolar-v0.01 still "
                "trains on all 24 qualities; this is a separate diagnostic, not the shipped model."
            ),
            "capability_class": None,
        }
        (ROOT / "data_store" / "generalization.json").write_text(json.dumps(gen_report, indent=2) + "\n")
        print("\nHELD-OUT GENERALIZATION (written to data_store/generalization.json):")
        print(json.dumps(gen_report, indent=2))
    print("\nNo capability-class claim is made. These are the numbers this run measured.")


if __name__ == "__main__":
    main()
