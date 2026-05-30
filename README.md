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

`honest_pipeline.py --task appreciation --size large --epochs 16 --samples 24000`
trains a **14.3M-parameter** model. Data is license-clean and self-generated — *no model
distilled* — with **24 qualities × 8 openers × quality-appropriate impacts × 8 closers**
in **two sentence structures** ("thank you for your clarity. it made things clearer" and
"your clarity made the review smoother"). `benchmark_appreciation.py` runs the real
suite, and an **LLM-judge agent** scores quality across **all 24 qualities**:

| Benchmark (measured) | Value | (prev.) |
|----------------------|-------|---------|
| **Appreciation validity** (Python-verified, 200 held-out) | **100%** | 100% |
| **Appropriateness** (impact actually fits the quality) | **100%** | 100% |
| **Quality coverage** (all 24) | **100%** | 100% |
| distinct-1 / distinct-2 (diversity) | **0.050 / 0.123** | 0.045 / 0.112 |
| Unique impact-clauses used | **21 / 23** | 16 / 16 |
| Master scalar (lower = more diverse) | **0.72** | 0.69 |
| **LLM-judge overall** · per-quality accuracy | **0.90 · 0.96** | 0.89 · 0.92 |
| Teacher model | none (no distillation) | — |

Successive iterations fixed the LLM-judge's findings. First, **impacts were quality-agnostic**
(e.g. "kindness → made the deadline reachable") — impacts are now **mapped to the qualities
they fit** (appropriateness 100%). Then the judge's named weak fits — *patience → feel
welcome*, *integrity → made things clearer*, *generosity → saved time* — were **remapped**
to fitting impacts, and the pool grew 16 → 23 phrases, lifting per-quality accuracy
0.92 → **0.96** and the judge's overall to **0.90**. Real generations:

```
clarity   -> i am grateful for your clarity. it made the review smoother.
patience  -> many thanks for your patience. it gave us room to get it right.
integrity -> your integrity earned our trust. the team noticed.
```

The **LLM-judge** (`data_store/judge_report.json`) rates it **0.90** overall —
grammaticality 1.0, relevance 1.0, appropriateness 0.90, wholesomeness 1.0, variety 0.72,
per-quality accuracy 0.96 — and honestly flags the remaining ceiling (one weak *generosity*
fit, a couple within-quality duplicates). These are benchmarks
appropriate to a ~14M-parameter generator; they are deliberately **not** MMLU/SWE-Bench/
GPQA scores. Figures live in
[`data_store/benchmarks.json`](./data_store/benchmarks.json),
[`data_store/judge_report.json`](./data_store/judge_report.json), and
[`data_store/version.json`](./data_store/version.json), written **only after** a run
(`status: pending` until then). No GPT-class or "Superhuman" label is ever attached.

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
