#!/usr/bin/env bash
# Capability slate via lm-evaluation-harness against a running vLLM OpenAI server.
# Run on the GPU box AFTER vLLM is up (vllm_server.py). Cheap (~$10-15) and the
# honest way to confirm the fine-tune didn't regress general ability.
#
#   ./lm_eval_slate.sh <served-model-name> [host:port]
set -euo pipefail
MODEL="${1:?usage: lm_eval_slate.sh <served-model-name> [host:port]}"
HOSTPORT="${2:-localhost:8000}"

pip show lm-eval >/dev/null 2>&1 || pip install "lm-eval>=0.4.4"

# A small, representative slate: knowledge, math, instruction-following, reasoning.
lm_eval \
  --model local-chat-completions \
  --model_args "base_url=http://${HOSTPORT}/v1/chat/completions,model=${MODEL},num_concurrent=8" \
  --tasks mmlu,gsm8k,ifeval,arc_easy \
  --apply_chat_template \
  --batch_size auto \
  --output_path eval_out/

echo ">> results written under eval_out/  (compare base vs SFT vs DPO checkpoints)"
