# erosolar_agent

An **additive** agentic stack built alongside the existing erosolar char-model
pipeline. Nothing in the legacy paths (`honest_pipeline.py`, `model.py`,
`serve.py`, `registry.py`'s MiniGPT loader) is modified or removed — this package
adds a fine-tuned open model + a real agent runtime + serving.

```
finetune/   QLoRA SFT + DPO of Qwen3 on a Lambda H100  →  a merged servable model
runtime/    multi-step agent loop (plan→act→observe→reflect), memory, tools, long-context
serving/    vLLM (raw model) + agent_server (agent loop) behind an OpenAI API
eval/       agentic task suite + lm-eval capability slate
secrets.py  gitignored .env loader (image/video key, HF/W&B tokens)
```

## Why this shape
A few hundred dollars can't pretrain a frontier agent. Best value = QLoRA-adapt a
strong open **instruct** model (Qwen3-32B, Apache-2.0) on a general + tool-use
blend, then SFT→DPO. The "agentic / long-horizon" behavior comes mostly from the
**runtime** (durable memory, planning, reflection, context compaction), which is
model-agnostic and works against any OpenAI-compatible endpoint.

## End-to-end

**1. Train (on Lambda — see `finetune/LAMBDA_LAUNCH.md`)**
```bash
bash erosolar_agent/finetune/setup.sh
bash erosolar_agent/finetune/run_lambda.sh smoke   # ~$0.40 sanity
bash erosolar_agent/finetune/run_lambda.sh proto   # ~$5 dial-in on 8B
bash erosolar_agent/finetune/run_lambda.sh full    # Qwen3-32B SFT→DPO→merge
```

**2. Serve** (GPU box)
```bash
# raw model on :8000
python -m erosolar_agent.serving.vllm_server --name erosolar-qwen3-32b --launch
# agent loop on :8080, talking to vLLM
VLLM_MODEL=erosolar-qwen3-32b python -m erosolar_agent.serving.agent_server
```

**3. Use the agent**
```bash
python -m erosolar_agent.runtime.cli "Research X, compute Y, write a summary to report.md" \
  --base-url http://localhost:8080/v1 --trace
```

**4. Evaluate**
```bash
python -m erosolar_agent.eval.run_agent_eval --base-url http://localhost:8080/v1   # agentic
bash erosolar_agent/eval/lm_eval_slate.sh erosolar-qwen3-32b                         # capability
```

**5. Deploy** — point `angular-chat` at the agent server's `/v1/responses` and
`firebase deploy --only hosting` (the legacy `run.sh --deploy` path still works).

## Dev without a GPU
The runtime is model-agnostic. Point `--base-url` at any OpenAI-compatible server
(a local vLLM with a small model, etc.) to develop the agent before the 32B is
trained. The loop itself is covered by an offline test:
```bash
python -m erosolar_agent.runtime.test_agent_offline
```

## Secrets
`.env` (gitignored) → loaded by `erosolar_agent.secrets`. Keys: `IMAGE_VIDEO_GEN_API_KEY`,
`IMAGE_VIDEO_GEN_BASE_URL`, `HF_TOKEN`, `WANDB_API_KEY`. Real env vars override `.env`.
