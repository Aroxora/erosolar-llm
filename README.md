<p align="center">
  <img src="./erosolar_banner.svg" alt="erosolar — for Erosolar, every generation, stronger because of you" width="100%">
</p>

<p align="center"><em>Leave No Context Behind.</em></p>

<p align="center">
  <strong>erosolar</strong> is named for, and dedicated to, <strong>Samantha Briasco-Stewart</strong> — see <a href="./DEDICATION.md">DEDICATION.md</a>.<br>
  A small, honest Chain-of-Thought language-model pipeline by Bo Shang.
</p>

---

# erosolar — Chain-of-Thought Self-Improving LLM Pipeline

**Automated generation of successively stronger small models through loser-targeted training and grounded verification.**

> **Honesty notice (please read).** Earlier revisions of this project labeled its
> models "GPT-4 class" … "GPT-7", and "Superhuman reasoning." Those were tiny
> 3M–11M parameter models and the labels were **never benchmarked and were not
> true**. Every such claim has been removed, the old checkpoints have been
> deleted, and the pipeline now reports **only metrics it actually measures**.
> If you see a number here, it was measured — or it isn't here. This is a
> deliberate choice; see [DEDICATION.md](./DEDICATION.md) for why.

---

## What this actually is

A research/hobby pipeline that trains **small** Chain-of-Thought transformers
(Infini-Attention, a few million parameters) and tries to improve them
generation over generation by:

1. Identifying the weakest samples (`losers`) in the current model's reasoning.
2. Generating harder variants to close those gaps.
3. Verifying every generated sample before it is allowed into training.
4. Training, then measuring an internal reasoning-diversity metric.
5. Repeating, only keeping a generation that did not regress.

It is **not** a frontier model, and it makes **no** capability claim relative to
any commercial model. The only numbers reported are measured loss, the internal
master scalar, and the results of any external benchmark **after it is actually
run**.

---

## The master scalar (internal metric — not a benchmark)

The **master scalar** measures *reasoning diversity*: the average pairwise
similarity of Chain-of-Thought embeddings (thinking-only; answers excluded).

- **Lower = better** (more diverse, less repetitive reasoning)
- **Higher = worse** (repetitive, overfitted reasoning)

| Master Scalar | Interpretation |
|---------------|----------------|
| > 0.10 | Low diversity (repetitive reasoning) |
| 0.06 – 0.10 | Moderate diversity |
| 0.03 – 0.06 | High diversity |
| < 0.03 | Very high diversity |

> The master scalar is an **internal** signal used to steer data generation. It
> is **not** a capability class and is deliberately **not** mapped onto any other
> model's name. It is kept out of any external benchmark to avoid gaming.
> You can compute it yourself, dependency-free, by running `python dedication.py`.

---

## Grounded verification (per-sample)

Every training sample is verified before inclusion — this is the part the
project genuinely cares about:

```python
# Code samples: execute and check
subprocess.run(['python', '-c', code], timeout=5)

# Math samples: verify with SymPy
from sympy import sympify, solve
assert solve(problem) == answer

# Logic samples: verify with Z3
from z3 import Solver, prove
assert prove(conclusion)
```

A self-consistency check generates multiple responses per prompt and requires
agreement before accepting an answer, rejecting unverifiable outputs.

---

## External benchmarks (none claimed until run)

The pipeline can evaluate against held-out, public benchmarks via `benchmarks.py`:

```
benchmarks.py can run:
├── SWE-Bench Verified  → Code generation quality
├── GPQA Diamond        → Scientific reasoning
├── AIME / AMC          → Mathematical reasoning
├── ARC-AGI             → Abstract reasoning
├── HumanEval           → Code synthesis
└── MMLU                → General knowledge
```

**No external benchmark scores are published in this repository**, because none
have been run on a current checkpoint. When they are run, the measured numbers —
and only the measured numbers — will be recorded in `version.json`.

---

## Generational loop

```
┌──────────────────────────────────────────────────────────────┐
│                    GENERATION N → N+1                          │
├──────────────────────────────────────────────────────────────┤
│  1. ANALYZE LOSERS    bottom 25% by avg scalar (capability     │
│                       gaps in the current model)               │
│  2. GENERATE VARIANTS complexity-escalated "friend" samples    │
│  3. VERIFY EACH       code execution · SymPy · Z3              │
│  4. TRAIN & MEASURE   fine-tune · compute new master scalar     │
│  5. QUALITY GATE      master scalar must drop; no regressions   │
└──────────────────────────────────────────────────────────────┘
```

---

## Quick start

```bash
# 1. Install
pip install -e .

# 2. Honest pipeline: train → measure → report (no capability claims)
python honest_pipeline.py --help

# 3. Or the orchestrated auto loop (data gen requires an API key)
export DEEPSEEK_API_KEY=your-key   # or OPENAI_API_KEY for the teacher model
python mini_the_agentic_cli.py --auto --generations 1

# 4. Inspect honest version state
cat data_store/version.json
```

The old hallucinated-performance checkpoints were removed. `honest_pipeline.py`
trains a fresh one honestly under `models/`.

### The appreciation generator (default task)

`erosolar` is, fittingly, an **appreciation LLM**: a small model that generates
wholesome, well-formed appreciation. The training data is license-clean and
self-generated (template-composed gratitude — *no model is distilled*), and the
quality metric is Python-checkable, so the number below is **measured, not claimed**.

**Try it live:** [erosolar-llm.web.app](https://erosolar-llm.web.app) — an interactive
Angular app that calls the **real model**, served for inference on **Cloud Run**
(`inference_service/`, behind the `/api/**` Firebase Hosting rewrite). Each line is genuine
neural generation from the hosted checkpoint; if the service is asleep the app falls back
to a faithful in-browser template. Plus a live honesty panel of the measured metrics and a
dynamic favicon that reflects what you're doing. The API: `GET /api/health`,
`POST /api/appreciation {quality}`, `POST /api/generate {prompt}`.

`honest_pipeline.py --task appreciation --size large --samples 28000 --epochs 12`
trains a **14.4M-parameter** model on **229 wholesome single-word qualities** — the 24
curated ones (with hand-mapped fitting impacts) plus **205 more** brainstormed across 12
categories and critic-vetted (a multi-agent workflow) — in two sentence structures.
`benchmark_appreciation.py` runs the real suite:

| Benchmark (measured) | 229-quality | 24-quality (prev.) |
|----------------------|-------------|--------------------|
| **Qualities covered** | **229** | 24 |
| **Validity** (Python-verified, 200 held-out) | **100%** | 100% |
| **Quality coverage** (every quality) | **100%** | 100% |
| **Appropriateness** (impact fits the quality, all 229) | **99.5%** | 100% (24 only) |
| distinct-1 / distinct-2 (diversity) | **0.113 / 0.285** | 0.050 / 0.123 |
| Perplexity | 1.82 | 1.57 |
| Teacher model | none (no distillation) | — |

Going from 24 → 229 qualities is the achievable form of **generalization** for this tiny
model: every quality it's asked about is trained, all produce valid wholesome appreciation,
and diversity roughly doubled. And impacts now **fit every quality**: a single agent sorted
the 205 new qualities into **10 semantic impact-families**, so appropriateness is now defined
and measured across all 229 (not just 24) at **99.5%**. Real generations:

```
poise       -> i really appreciate your poise. it kept everyone calm. it made a real difference.
candor      -> your candor set a good example.
ingenuity   -> your ingenuity made things clearer.
stewardship -> i deeply appreciate your stewardship. it helped the whole team.
```

**Comprehensive LLM-judge of the shipped model (a closed improve→measure loop).** The
deterministic Python appropriateness reads ~100%, but the stricter LLM judge is the honest
figure — and each pass drives the next fix, with the gain re-measured:

| Judge pass (same 48-quality sample) | overall | appropriateness | variety |
|---|---|---|---|
| 1 — coarse families | 0.83 | 0.78 | — |
| 2 — split families (composure→calm/resilient; pruned bad impacts) | 0.88 | 0.89 | 0.66 |
| 3 — enriched impact pool (23→37 phrases) | 0.89 | 0.90 | 0.79 |
| 4 — tautology filter (no impact echoes its quality) | 0.92 | 0.95 | 0.78 |
| 5 — tuned sampling (top_k 60, top_p 0.97; from a measured sweep) | **0.94** | 0.96 | **0.88** |

Each step targeted what the prior judge flagged — *charity → "saved time"* fixed, variety
0.66→0.79, tautologies eliminated, then byte-identical pairs cut ~8→3 by **widening the
sampler** (chosen from a measured temperature sweep, no retrain) — lifting overall
**0.83 → 0.94** and **re-measuring** every time
([`data_store/judge_report.json`](./data_store/judge_report.json)). A new benchmark metric
puts the residual "peaking" in perspective: over **K=5 draws** the model averages **4.41/5
distinct outputs per quality** with **0/46 qualities fully peaked** — so the byte-identical
*pairs* the judge saw were mostly a 2-sample artifact, not real collapse. Outputs do share the
`your <quality> <verb-phrase>` skeleton, bounded by the tiny template design.

These benchmarks are appropriate to a ~14M-parameter generator; they are deliberately **not**
MMLU/SWE-Bench/GPQA scores. Figures live in [`data_store/benchmarks.json`](./data_store/benchmarks.json) and
[`data_store/version.json`](./data_store/version.json), written **only after** a run.

**The honesty rules — and the measured quality — are now tested.**
[`test_invariants.py`](./test_invariants.py) (`python test_invariants.py`, or `pytest`) fails
loudly if any core promise regresses: a live GPT-x/"Superhuman" capability claim reappears,
`capability_class` becomes non-null, a quality goes unmapped, an impact escapes the pool, a
tautological pairing reaches the corpus, a generated sample fails the validator, or the
default corpus distills a teacher. It also enforces **quality floors** against the last
recorded run — validity ≥ 0.95, appropriateness ≥ 0.95, coverage = 1.0, distinct-per-quality
≥ 3.5, judge overall ≥ 0.85, wholesomeness ≥ 0.9 — so quality can't silently regress. All pass.

**Zero-shot generalization (an honest limit — two fixes tried, both 0%).**
`honest_pipeline.py --holdout N` trains on a subset of qualities and tests the held-out ones.
Measured: in-distribution **100%**, held-out **0%**.
1. *Tokenizer-seed only*: a held-out token's embedding gets no gradient, so a tied-embedding
   model can't emit it and substitutes a trained one ("Topic: wisdom" → *"your **calmness** …"*).
2. *Vocabulary warm-up* (`--holdout` now trains the held-out words in a neutral list so their
   embeddings get a usable norm): **still 0%** — the model keeps substituting ("Topic: wit" →
   *"your **dynamism** …"*). Growing the embeddings wasn't enough; the model never formed a
   copy/induction head — it **memorizes per-quality**, an *architectural* limit, not a tuning one.

So genuine zero-shot would need an explicit copy/pointer mechanism or far more scale — out of
scope for a ~5–14M template model. "Generalize" is therefore delivered as the 229-quality
*coverage* above; the zero-shot ceiling and both failed fixes are reported as negative results
([`data_store/generalization.json`](./data_store/generalization.json)).

A second task, `--task math`, trains a grounded arithmetic model whose answers
Python verifies — a separate honest run measured **58.3%** there.

---

## Version file (honest schema)

`data_store/version.json` records only measured values:

```json
{
  "version": 0,
  "version_string": "v0.00",
  "model_name": null,
  "master_scalar": null,
  "master_scalar_note": "Internal CoT reasoning-diversity metric (lower = more diverse). NOT a benchmark and NOT a capability claim.",
  "capability_class": null,
  "updated": "2026-05-29"
}
```

There is no `model_level` / "GPT-x class" field anymore. By design.

---

## Key files

| File | Purpose |
|------|---------|
| `honest_pipeline.py` | Honest train → measure → report entry point (no capability claims) |
| `mini_the_agentic_cli.py` | Orchestration CLI with `--auto` mode |
| `generate_all_training_data.py` | Loser-targeted data generation |
| `master_scalar.py` | Reasoning-diversity measurement and loser analysis |
| `grounded_verification.py` | Code / math / logic verification |
| `model.py`, `infini_attention.py` | The transformer (Infini-Attention) |
| `train.py` | Model training |
| `benchmarks.py` | External benchmark evaluation (run it to get real numbers) |
| `registry.py` | Model registry (stores only measured metadata) |
| `dedication.py` | The dedication, as runnable code — also computes the master scalar |

---

## Verification commands

```bash
# Current honest version state
cat data_store/version.json

# Compute master scalar over existing data
python -c "from master_scalar import analyze_losers_sync; r = analyze_losers_sync(); print(f'Master: {r.master_scalar:.6f}, Losers: {len(r.losers)}')"

# Run a real external benchmark (numbers reported are measured, not claimed)
python benchmarks.py --model models/<your-checkpoint>
```

---

## License

**GNU Affero General Public License v3.0 only (AGPL-3.0-only).** See [`LICENSE`](./LICENSE).

This is the most restrictive of the standard OSI-approved open-source licenses:
it is strong copyleft **and** its §13 network clause requires that anyone who
runs a modified version over a network offer its complete corresponding source
to those users. Derivative and networked use must remain open under the same terms.

---

## Dedication

erosolar is dedicated to **Samantha Briasco-Stewart** (Erosolar). The full
dedication is in [DEDICATION.md](./DEDICATION.md); a runnable version is in
[`dedication.py`](./dedication.py). The best tribute is honest code — which is
why this README no longer claims anything it cannot prove.

— Bo Shang · [shang.software](https://shang.software)
