# Lambda H100 launch checklist — Qwen3-32B QLoRA (SFT → DPO)

End-to-end recipe to fine-tune **Qwen3-32B** into the erosolar agentic
general-purpose model on **one Lambda H100 80GB SXM ($4.29/hr)**. Budget target:
**under $300**. Strategy: dial the recipe cheaply on 8B first, then one clean 32B run.

> 💸 **Cost reality:** 32B QLoRA SFT+DPO on Lambda is ~$280–340 with reruns —
> right at your ceiling. The proto-on-8B-first step and **terminating between
> runs** are what keep it in bounds. A spot H100 elsewhere (~$1–2/hr) would make
> this comfortably cheaper, but this guide targets Lambda as requested.

---

## 1. Launch the instance
- **Instance type:** `1x H100 (80 GB SXM5)` — the only single Lambda GPU that holds
  a 32B QLoRA (~44 GB) without offload, and the only one with FlashAttention-3.
- **Filesystem:** create/attach a **persistent filesystem** (same region as the
  instance) for `data/`, `outputs/`, and HF cache, so you can terminate the GPU
  between runs without re-downloading.
  - ⚠️ Storage bills ~$0.20/GiB/mo **even when detached** — delete it when fully done.
- Lambda has **no spot tier** and H100s **frequently sell out** — grab capacity when it's there.

## 2. Connect + get the code
```bash
ssh ubuntu@<instance-ip>
git clone https://github.com/Aroxora/erosolar-llm.git
cd erosolar-llm
```

## 3. Secrets (optional but recommended)
`.env` is gitignored, so it won't be on the box. Create it if you need gated
datasets or W&B telemetry:
```bash
cp .env.example .env
# edit .env: set HF_TOKEN (for gated datasets) and/or WANDB_API_KEY
```
The Tülu-3 / Hermes datasets in `configs/data_blend.yaml` are public, but some
agentic datasets you may swap in are gated — that's when `HF_TOKEN` matters.

## 4. Install
```bash
bash erosolar_agent/finetune/setup.sh
```
(Installs Unsloth + a compatible trl/transformers/peft/bitsandbytes stack.)

## 5. Smoke test (~5 min, ~$0.40) — DO NOT SKIP
```bash
bash erosolar_agent/finetune/run_lambda.sh smoke
```
Confirms data builds, the model loads in 4-bit, LoRA attaches (non-zero
trainable params), the trainer steps, and saving works — before you spend hours.

## 6. Prototype on 8B (~1–2 hr, ~$5–9)
```bash
bash erosolar_agent/finetune/run_lambda.sh proto
```
Eyeball the loss curve and a few generations. Tune `configs/*.yaml`
(data mix, lr, rank, epochs) here where iteration is cheap.

## 7. Final 32B run (SFT → DPO → merge)
```bash
nohup bash erosolar_agent/finetune/run_lambda.sh full > run.log 2>&1 &
tail -f run.log
```
- SFT: `configs/qwen3-32b-sft.yaml` → adapter in `outputs/qwen3-32b-sft/`
- DPO: `configs/qwen3-32b-dpo.yaml` (continues from the SFT adapter) → `outputs/qwen3-32b-dpo/`
- merge: `merged/erosolar-qwen3-32b/` (16-bit, vLLM-servable) + a registry entry

## 8. Get the model OFF the box, then TERMINATE
```bash
# recommended: push to your HF account so the serving box can pull it
export HF_TOKEN=...   # or in .env
python -m erosolar_agent.finetune.merge_and_export \
  --adapter outputs/qwen3-32b-dpo --out merged/erosolar-qwen3-32b \
  --push-to-hub <your-hf-user>/erosolar-qwen3-32b --register erosolar-qwen3-32b
# or rsync merged/ to your machine / object storage
```
Then **TERMINATE the instance** in the Lambda console (only termination stops
GPU billing — there is no pause). Keep just the persistent filesystem if you'll
iterate, else delete it too.

## 9. Hand back to me
Tell me the **HF repo** (or where the merged model lives). I'll wire it into the
vLLM serving backend, the agent runtime, and then the website deploy.

---
### Knobs (all in `erosolar_agent/finetune/configs/`)
| File | What to tune |
|---|---|
| `data_blend.yaml` | SFT sources + limits; **the ~30% tool-use slice = agentic capability** |
| `qwen3-8b-proto.yaml` | cheap proto (raise `max_steps`/epochs for a fuller proto) |
| `qwen3-32b-sft.yaml` | final SFT: epochs, lr (2e-4), rank (32), seq len |
| `qwen3-32b-dpo.yaml` | DPO: beta (0.1), lr (5e-6), `init_adapter` must point at the SFT dir |
