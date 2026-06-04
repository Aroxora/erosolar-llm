#!/usr/bin/env bash
# Run the QLoRA SFT->DPO->merge pipeline on a Lambda 1x H100 80GB box.
#
#   ./run_lambda.sh proto    # cheap dial-in run on Qwen3-8B (do this FIRST)
#   ./run_lambda.sh full     # the real Qwen3-32B SFT + DPO + merge
#   ./run_lambda.sh smoke    # ~60-step sanity pass (verifies the path end to end)
#
# Run from the repo root. Assumes setup.sh / pip install already done (see
# LAMBDA_LAUNCH.md). Checkpoints land in outputs/ ; data in data/.
set -euo pipefail
cd "$(dirname "$0")/../.."          # -> repo root
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export TOKENIZERS_PARALLELISM=false
export USE_TF=0 USE_FLAX=0   # Lambda Stack ships TF+Keras3; keep transformers torch-only

MODE="${1:-full}"
FT="python3 -m erosolar_agent.finetune"
DATA_FLAGS=""
[ "$MODE" = "smoke" ] && DATA_FLAGS="--max-per-source 1500"   # tiny data for the sanity pass

echo ">> [data] building SFT blend + DPO pairs"
$FT.prepare_data --stage sft --out data $DATA_FLAGS
$FT.prepare_data --stage dpo --out data $DATA_FLAGS

case "$MODE" in
  proto)
    echo ">> [proto] Qwen3-8B SFT (dial the recipe cheaply)"
    $FT.sft_qlora --config erosolar_agent/finetune/configs/qwen3-8b-proto.yaml
    echo ">> proto done. Inspect loss/sample outputs before committing to 32B."
    ;;
  smoke)
    echo ">> [smoke] 60-step end-to-end sanity (8B)"
    SMOKE_CFG=erosolar_agent/finetune/configs/qwen3-8b-proto.yaml
    # override to a tiny step count without editing the yaml
    python3 - "$SMOKE_CFG" <<'PY'
import sys, yaml, pathlib
p = pathlib.Path(sys.argv[1]); d = yaml.safe_load(p.read_text())
d["max_steps"] = 60; d["output_dir"] = "outputs/smoke-sft"
pathlib.Path("outputs").mkdir(exist_ok=True)
pathlib.Path("configs_smoke.yaml").write_text(yaml.safe_dump(d))
print("wrote configs_smoke.yaml")
PY
    $FT.sft_qlora --config configs_smoke.yaml
    echo ">> smoke passed: data + model + trainer + save all work."
    ;;
  full)
    echo ">> [1/3] Qwen3-32B QLoRA SFT"
    $FT.sft_qlora --config erosolar_agent/finetune/configs/qwen3-32b-sft.yaml
    echo ">> [2/3] Qwen3-32B QLoRA DPO (from SFT adapter)"
    $FT.dpo_qlora --config erosolar_agent/finetune/configs/qwen3-32b-dpo.yaml
    echo ">> [3/3] DPO adapter saved -> outputs/qwen3-32b-dpo"
    if [ -n "${HF_TOKEN:-}" ] && [ -n "${HF_PUSH_REPO:-}" ]; then
      echo ">> merging + pushing to HF: $HF_PUSH_REPO"
      $FT.merge_and_export --adapter outputs/qwen3-32b-dpo \
          --out merged/erosolar-qwen3-32b --push-to-hub "$HF_PUSH_REPO" --register erosolar-qwen3-32b
    else
      echo ">> skipping 64GB local merge (no HF_TOKEN/HF_PUSH_REPO)."
      echo ">> Deliverable = LoRA adapter at outputs/qwen3-32b-dpo (small; rsync it off before terminating)."
    fi
    echo ">> DONE."
    ;;
  *)
    echo "unknown mode: $MODE (use: proto | full | smoke)"; exit 2 ;;
esac
