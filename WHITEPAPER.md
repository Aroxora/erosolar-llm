# Erosolar Technical Whitepaper

**An honest small Chain-of-Thought LLM pipeline and the generational upgrade path to bigger models and better architectures.**

*Leave No Context Behind.*

Current live: https://erosolar-llm.web.app (Firebase + Cloud Run) · Source: https://github.com/Aroxora/erosolar-llm

---

## 1. The Current Honest Small-LLM Pipeline

erosolar trains successively stronger *small* models (~2M–100M parameters today) using a closed, measurable loop that reports only numbers that were actually run.

**Core components (all in this repo):**

- **Model**: Small decoder-only transformer with **Infini-Attention** (compressive memory for longer effective context without quadratic cost blowup at small scale). See `model.py`, `infini_attention.py`, `config.py`.

- **Tokenizer experiments & breakthrough**:
  - Word tokenizer: 0% held-out generalization (tied embeddings for unseen qualities receive no gradient).
  - Char tokenizer + length-matched nonce copy-augmentation: induced a copy/induction head. Held-out validity jumped 0% → 53% → **~89%**.
  - This is a direct application of mechanistic interpretability findings on induction heads (the circuit that does "see A B ... A → predict B").

- **Data**: 229 wholesome qualities, license-clean, **template-composed self-generated** (no teacher model distillation). Every sample is Python-checkable for validity + appropriateness.

- **The Generational Upgrade Pipeline** (the heart of the system):

  1. **Analyze losers**: Bottom 25% by the internal *master scalar* (average pairwise similarity of Chain-of-Thought embeddings; lower = more diverse reasoning).
  2. **Generate variants**: Complexity-escalated "friend" samples targeting the diagnosed gaps.
  3. **Grounded verification** (before any sample enters training data):
     - Code execution for appreciation structure.
     - SymPy for any arithmetic/logic claims in other tasks.
     - Z3 for formal constraints.
     - Multi-sample self-consistency (multiple generations must agree).
  4. **Train & measure**: Fine-tune, recompute master scalar + external benchmarks.
  5. **Quality gate**: Master scalar must improve; no regressions on validity rate (≥0.95 floor), appropriateness (≥0.95), coverage (1.0), distinctness, or LLM-judge scores (overall ≥0.85, wholesomeness ≥0.9). See `test_invariants.py`.

**Measured results for the shipped `erosolar-v0.01` (14,375,040 params, 8 layers, word tokenizer, 12 epochs on MPS):**

- Qualities covered: **229**
- Validity (Python-verified, 200 held-out): **0.99**
- Quality coverage: **1.00**
- Appropriateness: **0.985**
- Distinct outputs per quality (K=5): **4.41 / 5** (0/46 fully peaked)
- Validation perplexity: **1.92**
- LLM-judge overall (96 samples): **0.94** (grammar 0.96, relevance 1.0, appropriateness 0.96, variety 0.88, wholesome 0.97)
- Teacher model: **none** (no distillation)

See `data_store/benchmarks.json`, `judge_report.json`, `generalization.json`, `version.json`.

The pipeline is fully reproducible with `./run.sh` (or `--generalize` for the char+copy 89% recipe).

---

## 2. The Additive Agent Stack (Nothing touches the small core)

A separate layer on top:

- **Finetune**: QLoRA SFT → DPO of Qwen3-32B (Apache-2.0) on Lambda H100.
- **Runtime**: Model-agnostic plan → act → observe → reflect loop with durable memory, tool use, and context compaction. Works against any OpenAI-compatible endpoint.
- **Serving**: vLLM (base model) + agent_server behind OpenAI-compatible API.
- **Integrations**: Tavily (search with quota handling) + DeepSeek (synthesis). Same integrations power the `model-landscape/` auto-updater.
- **Eval**: Agentic task suite + lm-eval slate.

See `erosolar_agent/`.

This is why a tiny honest core can still be useful: the long-horizon behavior lives in the runtime + tools, not purely in the weights.

---

## 3. The Upgrade Pipeline & Future Directions (Bigger Models + Better Architecture)

The current loop is deliberately architecture- and scale-agnostic in its data generation and verification. The same "loser analysis + grounded verification + quality gate" can drive improvement at larger scales.

**Current strengths at small scale**:
- Copy/induction heads (via data) are extremely high-leverage because parametric memory is expensive.
- Grounded verification acts as a strong regularizer and honesty enforcer.
- The master scalar steers diversity without gaming external benchmarks.

**Upgrade vectors** (drawn from the living [architecture-roadmap/](./architecture-roadmap) and [model-landscape/](./model-landscape)):

- **Bigger models**: Apply the identical pipeline to 100M+ and then 1B+ scales. The data loop and verification remain the same; only the base model and compute change. The small model can serve as a fast proposer or verifier in a speculative / tree-search setup.

- **Better architecture for small-to-medium**:
  - Hybrids (Mamba-2 / SSM blocks interleaved with attention or sparse attention) for linear-ish decode + precise recall where needed. (See SambaY, Nemotron Nano, Griffin, Jamba lines.)
  - Improved induction/copy mechanisms and better tokenization (char/byte-level or hybrid) to keep the generalization wins we already measured.
  - Sub-quadratic / compressive attention variants (Infini-Attention is one starting point; gated linear attention, RetNet-style retention, etc.).
  - Mixture-of-Experts is generally not useful below ~hundreds of millions of active parameters (routing overhead + under-utilization); save the capacity for depth/width or data.

- **Test-time compute as a first-class axis**: Even the current 14M model benefits from extra thinking (multiple samples + judge, tree search, code execution feedback). Small models + heavy verification can outperform naive larger models on narrow tasks.

- **Retrieval, tools & RAG**: Small models are knowledge-poor by design. Offload to retrieval and the agent runtime (already done in the additive stack). SLMs + good RAG frequently match much larger general models on specific domains.

- **Quantization & edge**: Design for 4-bit / 8-bit from the start (QAT where possible). This is the natural deployment surface for <100M models. The architecture choices (linear decode, small KV or state) matter here.

- **Distillation & self-improvement**: Use the current honest small model + verifiers to generate higher-quality data for the next generation (or to distill behaviors into the agent base). Keep everything license-clean and verifiable.

- **Model-landscape awareness**: The `model-landscape/update.py` (powered by the same agent integrations) keeps an agentically refreshed picture of frontier models. This prevents over- or under-estimating what "small" can or should do.

**Concrete next steps in the repo**:
- Run the full pipeline at larger scale (more params, more data, the same verification).
- Integrate the small core as a fast proposer/verifier inside the agent runtime.
- Experiment with hybrid blocks in the core transformer (start with a few Mamba-style layers).
- Close the loop: use the agent to propose better training distributions or verification oracles.

All numbers in this document were measured. The invariants in `test_invariants.py` will loudly fail if that stops being true.

---

## References & Living Documents in the Repo

- `honest_pipeline.py` + `grounded_verification.py` + `master_scalar.py` — the current loop.
- `generational_upgrade_pipeline.py`, `training_upgrade_pipeline.py` — the upgrade machinery.
- `architecture-roadmap/README.md` — cited survey of directions with small-model verdicts (as of June 2026).
- `model-landscape/` — agentically refreshed profiles of current frontier models (run `update.py` with keys).
- `data_store/` — all the measured artifacts (benchmarks, judge reports, generalization, version).
- `run.sh` — one-command reproduction.
- `erosolar_agent/` — the additive long-horizon layer.
- `test_invariants.py` — the honesty rules as failing tests.

This whitepaper is also available on the live site at https://erosolar-llm.web.app (embedded in the UI or linked).

The best way to understand the upgrade pipeline is to run it: `./run.sh --generalize` reproduces the 89% zero-shot result from first principles.

---

*Everything above is grounded in code and measured artifacts in this repository. No capability claims beyond what the tests and reports actually contain.*

— Bo Shang

(Updated as part of the site refresh — see DOMAINS.md for branding and domain plans.)