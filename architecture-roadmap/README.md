# Architecture Roadmap for Small LLMs (~2M–100M parameters)

**Current to June 2026.** A focused survey of architectural directions with verdicts *specifically for the tiny-to-small regime* that erosolar targets. The goal is not to chase frontier scale, but to identify what actually moves the needle when every parameter and every training token is expensive.

**Core thesis of this roadmap (and of erosolar):** At the scale of a few million to low hundreds of millions of parameters, *Mixture-of-Experts needs scale and largely does nothing useful*; the winning moves are better tokenization + explicit induction/copy mechanisms, data quality + grounded self-improvement loops, retrieval/tools offloading, sub-quadratic sequence modeling (when it doesn't sacrifice too much), and test-time compute. Quantization/edge is the primary deployment surface.

All claims below are grounded in the cited literature and public reports (as of the snapshot date). Re-check primary sources; this is a living note.

---

## Tokenization & Copy / Induction Heads

Standard subword (BPE/SentencePiece) tokenizers are near-universal, but for tiny models they create a fundamental problem: rare or held-out tokens have embeddings that receive little or no gradient, so the model literally cannot emit novel surface forms even if it "understands" the concept. This is exactly the 0% zero-shot generalization floor observed in early erosolar char-tokenizer experiments.

**Induction heads** (Anthropic Transformer Circuits, 2022) are a well-understood circuit in small transformers: a pair of attention heads that implement "previous-token" copying followed by "match previous occurrence and copy the successor." They are the primary mechanism behind in-context learning and literal/fuzzy copying in small models. Later work (2024–2026) shows they persist at larger scales, exhibit diversity (some heads prefer the most recent match, others the first, some incorporate 2nd-order context), and remain central to pattern completion and OOD generalization via composition.

Copy-augmentation strategies (explicitly training the model to copy nonce strings of varying lengths) are a cheap, data-driven way to *induce* a strong copy head without adding parameters. erosolar's length-matched nonce recipe moved held-out generalization from 0% → ~53% → ~89% measured validity on novel qualities.

**Verdict for ~2M–100M:** Extremely high leverage. Prefer character-level, byte-level, or carefully designed hybrid tokenizers when the task benefits from open-vocabulary emission or novel-term generalization. Deliberately augment training data to form and strengthen induction/copy heads. This is one of the few "free" capability jumps available at this scale because copying is nearly free at inference (vs. storing everything parametrically). Standard BPE + no copy induction is a self-imposed handicap for small models.

Relevant: [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html), later analyses in 2025 on diverse induction strategies, erosolar's own `run.sh --generalize` + `char_tokenizer.py` + `honest_pipeline.py` copy-aug path.

---

## Linear Attention, SSMs, RetNet, RWKV, Mamba-family, and Hybrids

Transformers pay O(n²) for attention and O(n) KV cache per step. For small models that still want usable context (or on-device long sessions), this hurts.

Alternatives with linear or near-linear scaling during generation:
- **Mamba** (Gu & Dao, 2023) and follow-ons (Mamba-2, Mamba-3): selective state-space models. Strong long-context, hardware-efficient kernels, competitive with same-size transformers on many tasks.
- **RWKV**: linear attention re-expressed as RNN; scaled to multi-billion params early.
- **RetNet**: retention mechanism with multi-scale decay; parallel training + recurrent inference.
- **Gated DeltaNet** and other gated linear recurrences.
- **Hybrids** (2024–2026): Jamba (Mamba + Transformer), Samba / SambaY (Mamba + local/global attention mixes), Zamba, Griffin (gated recurrence + local attention), Nemotron-3 Nano (Mamba-2 interleaved with sparse attention/MoE). Hybrids often recover the precise recall that pure SSMs can lose while keeping most of the efficiency.

Small-model-specific notes: at <100M, pure transformers (or the infini-attention compressive variant already in this repo) remain simple and strong baselines. Hybrids and SSMs shine most on long-context or high-throughput inference; some reports show small Mamba/hybrid models delivering million-token effective context or better tokens/sec on edge. Training stability and optimized kernels for tiny variants lag the big labs.

This repo already ships `infini_attention.py` (compressive memory) precisely because quadratic attention + tiny params is painful for anything beyond short CoT.

**Verdict for ~2M–100M:** Promising and worth prototyping. Pure SSMs or light hybrids can beat or match a same-FLOP dense transformer on long-sequence tasks while using far less memory at decode time. Start with a strong dense + infini baseline (current erosolar), then experiment with Mamba-2 or Samba-style blocks in a few layers. Avoid over-optimizing architecture before nailing data + verification (the latter gives bigger measured gains in the erosolar runs). Sub-quadratic is a deployment win for the edge/edge-adjacent use cases that small models target.

---

## Mixture-of-Experts (MoE)

MoE decouples total parameters from active compute by routing tokens to a sparse subset of "experts." Real wins appear at very large scale: DeepSeek-V3 (671B total, 37B active), Mixtral 8x7B, Grok/Mixtral-style, Gemini sparse, etc. Training throughput and inference speed can be dramatically better than a dense model of equivalent *active* size.

At small expert counts or small total size the picture changes. Scaling-law papers show that for modest numbers of experts the loss curves collapse back toward dense scaling; routing overhead, load-balancing, and expert under-utilization become first-order costs. Community consensus (Reddit threads, small-model ablation reports) is that for sub-100M–few-hundred-M budgets you are usually better off spending the parameter budget on depth/width or data rather than expert routing machinery.

**Verdict for ~2M–100M:** Skip MoE. The canonical example in the erosolar README: "Mixture-of-Experts needs scale; it does nothing for a tiny dense model." Use the capacity you would have "wasted" on experts for a deeper or wider dense backbone, better tokenizer, or more verified training data. Only reconsider if you are already at the upper end of this range *and* have measured that a dense equivalent is memory-bound while you have spare compute for routing.

---

## Distillation, Data Quality, Self-Generation & Grounded Verification

For any small model, *what* you train on matters more than the exact architecture once you are past the obvious transformer++ baseline. Distillation from a strong teacher is common and effective (Phi series, etc.), but introduces licensing and honesty issues if you claim the result as purely "your" model.

erosolar deliberately uses *template-composed, license-clean, self-generated* data with Python/SymPy/Z3 grounded verification before any sample is admitted to training data. The master scalar + loser analysis + quality gate loop is a tiny-model-appropriate self-improvement mechanism.

Test-time compute (search, tree search, self-verification, budgeted reflection) is another multiplier that works *especially* well for small models: a 1B model with good verifiers and extra thinking steps can beat a naive 7–8B on some reasoning slices (Hugging Face reports, 2025–2026).

**Verdict:** Highest-ROI area for the erosolar scale. Architecture experiments are fun; data quality + verification loops + test-time reasoning are where the measured numbers in `data_store/benchmarks.json` and `judge_report.json` actually came from. Continue to emphasize grounded, executable, non-tautological data. The agent stack (erosolar_agent) is the natural extension: runtime supplies the long-horizon + tool-use "thinking" that a 14M appreciation generator cannot hold parametrically.

---

## Retrieval, Tools, RAG, and Agentic Scaffolding

Small models have tiny parametric memories. Retrieval-augmented generation (RAG), tool use, and durable runtime memory are the standard way to give them "knowledge" and capabilities far beyond their parameter count.

2025–2026 literature and deployments repeatedly show SLMs + good RAG matching or exceeding much larger general models on domain-specific or up-to-date tasks. On-device/privacy-sensitive use cases (the natural home for <100M models) *require* local retrieval or federated tool calling.

The additive erosolar_agent runtime (plan→act→observe→reflect + Tavily + DeepSeek integrations) is exactly this philosophy: the heavy reasoning behavior lives in the loop and tools, not (only) in the weights. Same integrations power the model-landscape auto-updater.

**Verdict:** Mandatory for useful small models. Design the system so the tiny core generator is only one component; the rest of the capability comes from retrieval, verification, tools, and multi-step scaffolding. This is why erosolar ships both the pipeline *and* the separate agent stack.

---

## Quantization, Edge Deployment, and On-Device

This is the primary real-world surface for models in the 2M–100M band in 2026.

- Aggressive 4-bit / 5-bit / 8-bit quantization (with or without QAT) is table stakes.
- NPUs on phones (Qualcomm, Apple, etc.) and tiny edge chips love small quantized models.
- Examples: Llama 3.2 1B/3B, Gemma 3 smalls, Qwen3 small variants, Phi-3/4 mini, Apple on-device models, various 1–3B "Nano" or "Tiny" releases.
- Many ship with quantization-aware training or post-training recipes that preserve most quality.

Memory is the binding constraint (phones often have 4–12 GB total, much of it unavailable to the model). Sub-100M models (or heavily distilled/quantized 1–3B) fit comfortably and run fast.

**Verdict:** Design for quantization from day one. The erosolar checkpoints are already tiny enough that even the 14M word-tokenizer model + simple serving is deployable; the char-copy variant is even smaller in vocab footprint. Target INT4/INT8 + CoreML / ONNX / llama.cpp / MLX paths for real usage. Edge is not a "nice to have" — it is the point of staying this small.

---

## Scaling Laws & Where to Spend Compute

Classic Chinchilla / Kaplan laws still provide guidance: for a given compute budget there is an optimal model size + tokens. Newer meta-analyses (2024–2026) fit thousands of scaling curves across families and confirm that architecture family matters (some are more sample-efficient), but the biggest deviations at small scale come from data quality and the new inference-time "scaling" axes.

For a hobby/research budget, the practical advice is:
- Spend first on high-quality, verified, diverse data (and the loops that produce it).
- Spend next on architectural simplicity + one high-leverage trick (copy induction, compressive memory, or a proven hybrid block).
- Only then consider raw parameter count.
- Reserve inference budget for test-time methods when the query is worth it.

**Verdict:** The erosolar pipeline (loser-targeted generation + grounded verification + master-scalar quality gate) + run.sh one-command reproduction is a concrete embodiment of "spend the scarce resource on data and measurement, not on claiming a bigger dense model."

---

## Summary Recommendations for a ~2–100M Honest Small Model (June 2026)

1. **Tokenizer first**: char/byte or length-aware hybrid if you care about novel-term generalization. Add explicit copy-augmentation data to grow induction heads.
2. **Backbone**: Start dense + compressive memory (Infini-Attention style) or a light Mamba-2 / hybrid block if long context is required. Measure tokens/sec and memory on target hardware early.
3. **Skip MoE** unless you have already saturated everything else and have concrete evidence a dense model is the bottleneck.
4. **Data loop is king**: self-generation or high-quality synthetic, every sample grounded/verified before training, loser analysis, no capability claims, quality gates that actually fail the run on regression.
5. **Offload aggressively**: retrieval, tools, durable runtime memory, reflection loops (see erosolar_agent/).
6. **Quant + edge from the start**: train or post-train for 4-bit friendliness; ship CoreML / quantized GGUF / ONNX paths.
7. **Test-time compute**: let the small model "think" (search, multiple samples + judge, tree search, code execution) on hard queries.
8. **Measure everything** that will be reported; keep the honesty invariants (`test_invariants.py`) loud.

The architecture-roadmap + model-landscape folders exist so that when someone asks "why not just scale a dense transformer / add MoE / ...", the answer is already written down with citations and a small-model-specific verdict — and the numbers in the repo were measured against that backdrop.

— erosolar project, June 2026

(Primary sources: Mamba family papers & follow-ups, Transformer Circuits induction-head work and 2025 updates, scaling-law meta-analyses, 2025–2026 hybrid model reports (Samba, Nemotron Nano, Jamba/Zamba/Griffin lines), SLM/edge deployment surveys, and the measured erosolar runs themselves.)
