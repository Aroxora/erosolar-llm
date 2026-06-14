<p align="center">
  <img src="./erosolar_banner.svg" alt="erosolar — for Erosolar, every generation, stronger because of you" width="100%">
</p>

<p align="center"><em>Leave No Context Behind.</em></p>

<p align="center">
  <strong>erosolar</strong> is named for <strong>Erosolar</strong> — see <a href="./DEDICATION.md">DEDICATION.md</a>.<br>
  A small, <em>honest</em> Chain-of-Thought language-model pipeline — plus an additive agentic stack — by Bo Shang.
</p>

<p align="center">
  <a href="./LICENSE"><img alt="License: AGPL-3.0-only" src="https://img.shields.io/badge/license-AGPL--3.0--only-blue.svg"></a>
  <img alt="Python 3.9+" src="https://img.shields.io/badge/python-3.9%2B-blue.svg">
  <img alt="Status: research / hobby" src="https://img.shields.io/badge/status-research%20%2F%20hobby-orange.svg">
  <img alt="No capability claims" src="https://img.shields.io/badge/capability%20claims-none-brightgreen.svg">
</p>

---

# erosolar — an honest small-LLM pipeline

**Train successively stronger *small* Chain-of-Thought models through loser-targeted
data generation and per-sample grounded verification — and report only numbers that
were actually measured.**

> ### Honesty notice (please read)
> Earlier revisions of this project labeled its models "GPT-4 class" … "GPT-7", and
> "Superhuman reasoning." Those were tiny **3M–14M parameter** models and the labels
> were **never benchmarked and were not true**. Every such claim has been removed, the
> old checkpoints deleted, and the pipeline now reports **only metrics it actually
> measures**. If you see a number here, it was measured — or it isn't here. The honesty
> rules are enforced in code by [`test_invariants.py`](./test_invariants.py), which fails
> loudly if any capability claim, tautology, or quality regression reappears. See
> [DEDICATION.md](./DEDICATION.md) for why this matters.

---

## What this actually is

Three layers, each honest about its own scale:

| Layer | What it is | Scale |
|------|-------------|-------|
| **The pipeline** | A research/hobby loop that trains **small** Chain-of-Thought transformers ([Infini-Attention](./infini_attention.py)) and improves them generation-over-generation via loser analysis + grounded verification. | ~2M–100M params |
| **The agent stack** ([`erosolar_agent/`](./erosolar_agent)) | An **additive** agentic runtime — QLoRA fine-tune of an open instruct model (Qwen3-32B) + a plan→act→observe→reflect loop + vLLM/agent serving. Nothing in the legacy pipeline is modified. | 32B base, QLoRA |
| **The web app** ([`angular-chat/`](./angular-chat)) | A live Angular app that calls the **real** model, served for inference on Cloud Run behind Firebase Hosting. | — |

It is **not** a frontier model and makes **no** capability claim relative to any
commercial model. The only numbers reported are measured loss, the internal master
scalar, and the results of task-appropriate benchmarks **after they are actually run**.

---

## The shipped model — measured, not claimed

[`honest_pipeline.py`](./honest_pipeline.py) trains an **appreciation generator**: a small
model that emits wholesome, well-formed appreciation. The data is **license-clean and
self-generated** (template-composed gratitude — *no model is distilled*), and the quality
metric is Python-checkable, so every number below is **measured**.

**`erosolar-v0.01`** — **14,375,040 parameters** · 8 layers · 8 heads · `embed_dim` 384 ·
word tokenizer (vocab 361) · trained 12 epochs on Apple MPS.
([`data_store/version.json`](./data_store/version.json),
[`benchmarks.json`](./data_store/benchmarks.json),
[`judge_report.json`](./data_store/judge_report.json))

| Metric (measured) | Value |
|---|---|
| Qualities covered | **229** |
| Validity (Python-verified, 200 held-out) | **0.99** |
| Quality coverage (every quality emits) | **1.00** |
| Appropriateness (impact fits the quality, all 229) | **0.985** |
| distinct-1 / distinct-2 (diversity) | **0.136 / 0.308** |
| Distinct outputs per quality over K=5 draws | **4.41 / 5** (0/46 fully peaked) |
| Validation perplexity | **1.92** |
| **LLM-judge overall** (96 samples) | **0.94** (gram 0.96 · relevance 1.0 · approp. 0.96 · variety 0.88 · wholesome 0.97) |
| Teacher model | **none** (no distillation) |

```
poise       -> i really appreciate your poise. it kept everyone calm. it made a real difference.
candor      -> your candor set a good example.
stewardship -> i deeply appreciate your stewardship. it helped the whole team.
```

These are benchmarks **appropriate to a ~14M-parameter generator** — deliberately *not*
MMLU / SWE-Bench / GPQA, which do not apply to a model this small.

### Zero-shot generalization — driven 0% → 53% → ~89% across five measured attempts

`honest_pipeline.py --holdout N` holds qualities out of training and tests them. The
diagnosis sharpened over five runs:

1. **Word tokenizer, seed only → 0%** — a held-out word's (tied) embedding gets no gradient, so the model literally can't emit it.
2. **Word + vocab warm-up → 0%** — growing embeddings wasn't enough.
3. **Char tokenizer → 0%**, but emittability is now fine — *no cue→answer copy head forms*.
4. **Char + copy-augmentation → 53%** — nonce qualities it can't memorize force it to copy the cue; **this induced the copy head**.
5. **Copy-aug with length-matched nonces (3–15 chars) → ~89%** — long held-out qualities now copy verbatim (*transparency, trustworthiness, thoughtfulness, …*).

The currently-recorded checkpoint ([`data_store/generalization.json`](./data_store/generalization.json)),
`erosolar-charcopy-lg` (char tokenizer, vocab 37, 24 qualities held out), measures
**held-out validity 0.733 / in-distribution 1.00** — reproduce the full recipe with
`./run.sh --generalize`. The shipped model stays the word-tokenizer 229-quality one; the
char+copy-aug model is a **separate, reproducible demonstration** of *novel*-quality
generalization. (The earlier "architectural 0% dead end" conclusion was wrong — corrected
here with measured numbers.)

> Why does a copy head matter so much for a *tiny* model? Because parametric memory is the
> expensive thing at this scale, and copying is nearly free. That logic — *what actually
> helps a small model* — is the subject of the new [architecture roadmap](./architecture-roadmap).

---

## Quick start

```bash
pip install -e .                      # install (PyTorch + a few deps)

./run.sh --quick                      # fast tiny end-to-end sanity run
./run.sh                              # train (large) -> benchmark -> invariant tests
./run.sh --generalize                 # char + copy-aug zero-shot experiment (~89% held-out)
./run.sh --task math                  # grounded arithmetic task (Python-verified answers)
./run.sh --deploy                     # also hot-swap Cloud Run + redeploy the web app

python test_invariants.py             # the honesty rules, as enforced tests
cat data_store/version.json           # current honest version state (only measured values)
```

**Try it live (current):** [erosolar.net](https://erosolar.net) — the product site + web app (Angular calling the real hosted model on Cloud Run).

**Primary .com target (register now):** **erosolarai.com** (perfect match for the iOS bundle `com.erosolarai.chat` already wired in the project).  
**Premium brand .com (for sale):** **erosoral.com** — the exact name from the dedication. See the full acquisition plan and "do it now" checklist in [DOMAINS.md](./DOMAINS.md), including iOS App Store bundle reservation steps (`com.erosolarai.chat`) and social handle claims.

The Firebase preview is still at [erosolar-llm.web.app](https://erosolar-llm.web.app). Once domains are live we will point DNS + update rewrites / custom domains. API examples (health / generate) remain the same until remapped.

---

## The agent stack (`erosolar_agent/`)

An **additive** layer — the legacy pipeline is untouched. A few hundred dollars can't
pretrain a frontier agent, so the best value is to **QLoRA-adapt a strong open instruct
model** (Qwen3-32B, Apache-2.0) and put the long-horizon behavior in a model-agnostic
**runtime**.

```
finetune/   QLoRA SFT -> DPO of Qwen3 on a Lambda H100  ->  a merged, servable model
runtime/    multi-step agent loop (plan->act->observe->reflect), durable memory, tools
serving/    vLLM (raw model) + agent_server (agent loop) behind an OpenAI-compatible API
integrations/  Tavily web search + DeepSeek, each with graceful quota handling
eval/       agentic task suite + lm-eval capability slate
```

The runtime is model-agnostic: point `--base-url` at any OpenAI-compatible endpoint (a
local vLLM, etc.) to develop against it before the 32B is trained. See
[`erosolar_agent/README.md`](./erosolar_agent/README.md). The same Tavily+DeepSeek
integrations power the [model-landscape auto-updater](./model-landscape) below.

---

## 📚 Research folders (new)

Two living research areas ship with the repo:

### [`architecture-roadmap/`](./architecture-roadmap) — where a *small* model should go
A comprehensive, **cited** survey (current to June 2026) of LLM architecture directions —
tokenization & copy heads, linear-attention / SSMs, distillation & data quality, test-time
compute, retrieval & tools, quantization / edge, scaling laws — each ending with a verdict
**for a ~2M–100M model**. The throughline, and the canonical example: **Mixture-of-Experts
needs scale; it does nothing for a tiny dense model** — so the roadmap points elsewhere
(better tokenization, copy/induction heads, retrieval, distillation, sub-quadratic attention).

### [`model-landscape/`](./model-landscape) — frontier-model deep-dives, **auto-updated agentically**
Technical profiles of current frontier models (Claude Opus 4.8, Fable 5, Mythos 5; GPT-5.5;
Gemini 3.1 Pro / 4.5 Flash; Grok 4.3; and more in [`models.yaml`](./model-landscape/models.yaml)).
[`model-landscape/update.py`](./model-landscape/update.py) drives this repo's own agent
integrations (Tavily search → DeepSeek synthesis) to refresh each profile from primary
sources, stamped with an "as of" date and citations — never inventing specs:

```bash
python model-landscape/update.py            # refresh every model (needs TAVILY + DEEPSEEK keys)
python model-landscape/update.py --only gpt-5.5 grok-4.3
python model-landscape/update.py --dry-run  # plan only, no network
```

---

## How the pipeline works

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

**The master scalar** is an *internal* reasoning-diversity signal (average pairwise
similarity of Chain-of-Thought embeddings; **lower = more diverse**). It steers data
generation and is **not** a benchmark or capability class — kept out of any external eval
to avoid gaming. Compute it dependency-free with `python dedication.py`.

**Grounded verification** runs *before* a sample is allowed into training — code is
executed, math is checked with SymPy, logic with Z3, and a self-consistency check requires
agreement across multiple generations before an answer is accepted.

---

## Repository layout

| Path | Purpose |
|------|---------|
| `honest_pipeline.py` | Honest train → measure → report entry point (no capability claims) |
| `run.sh` | One-command reproduction (train · benchmark · test · optional deploy) |
| `model.py`, `infini_attention.py` | The transformer (Infini-Attention compressive memory) |
| `config.py` | Model/size presets (tiny → large; infini-* variants) |
| `tokenizer.py`, `char_tokenizer.py` | Word/BPE and character tokenizers (char enables copy generalization) |
| `train.py` | Training loop |
| `master_scalar.py` | Reasoning-diversity measurement + loser analysis |
| `grounded_verification.py` | Code / math / logic verification |
| `benchmark_appreciation.py` | The real, task-appropriate benchmark suite |
| `test_invariants.py` | The honesty rules, as failing-tests |
| `registry.py` | Tokenizer-aware model registry (stores only measured metadata) |
| `erosolar_agent/` | The additive agentic stack (finetune · runtime · serving · eval) |
| `inference_service/` | Cloud Run inference service (build context assembled by `build.sh`) |
| `angular-chat/` | The live web app (current Firebase: erosolar-llm.web.app; primary domain target: erosolarai.com — see DOMAINS.md) |
| `architecture-roadmap/` | Architecture survey + roadmap for a small model (to June 2026) |
| `model-landscape/` | Frontier-model deep-dives + agentic auto-updater |
| `data_store/*.json` | Measured results: `version`, `benchmarks`, `generalization`, `judge_report` |
| `dedication.py` | The dedication, as runnable code — also computes the master scalar |

---

## Configuration & secrets

Real secrets live in a **gitignored** `.env` (copy [`.env.example`](./.env.example) and fill
it in); real environment variables always take precedence. The loader
([`erosolar_agent/secrets.py`](./erosolar_agent/secrets.py)) never logs raw values.

> The Firebase **web** config keys in the Angular app are *public by design* (they ship to
> every browser and are gated by Firebase Security Rules + App Check, not key secrecy).
> Server-side admin credentials (service-account JSON) are the real secret and are
> gitignored — never commit them.

---

## License

**GNU Affero General Public License v3.0 only (AGPL-3.0-only).** See [`LICENSE`](./LICENSE).
Strong copyleft **plus** the §13 network clause: anyone who runs a modified version over a
network must offer its complete corresponding source to those users.

---

## Dedication

erosolar is dedicated to **Samantha Briasco-Stewart** (Erosolar). The full dedication is in
[DEDICATION.md](./DEDICATION.md); a runnable version is in [`dedication.py`](./dedication.py).
The best tribute is honest code — which is why this README claims nothing it cannot prove.

— Bo Shang · [shang.software](https://shang.software)
