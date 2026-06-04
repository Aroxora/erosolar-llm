#!/usr/bin/env bash
# One-time environment setup on a fresh Lambda H100 box.
# Validated on Lambda H100 PCIe (driver 570 / CUDA 12.8), Python 3.10, June 2026.
set -euo pipefail

echo ">> python: $(python3 --version)"
echo ">> gpu:"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

python3 -m pip install --upgrade pip -q

# 1) Torch matching the driver. A plain `pip install unsloth` grabs a CUDA-13 torch
#    that a 570-series driver (CUDA 12.8) cannot run, so install the cu128 build first.
python3 -m pip install --force-reinstall --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# 2) Let unsloth resolve a coherent transformers/trl/peft, but pin torch and cap
#    transformers<5 (the stable line unsloth supports on py3.10).
printf 'torch==2.8.0\ntransformers<5\n' > /tmp/eros_constraints.txt
python3 -m pip install --no-cache-dir -c /tmp/eros_constraints.txt unsloth xformers datasets huggingface_hub pyyaml

# 3) torchvision must match torch 2.8 (unsloth requires >=0.23) — install the cu128 build.
python3 -m pip install --no-cache-dir "torchvision==0.23.0" --index-url https://download.pytorch.org/whl/cu128

# 4) Lambda Stack's system scipy/sklearn need numpy<2 (ABI); transformers imports sklearn at
#    load. Pin numpy==1.26.4, and pin fsspec (a newer one gets pulled in transitively and
#    conflicts with datasets).
python3 -m pip install --no-cache-dir "numpy==1.26.4" "fsspec<=2025.9.0"

echo ">> versions:"
USE_TF=0 USE_FLAX=0 python3 - <<'PY'
import importlib
for m in ("torch","unsloth","trl","transformers","peft","datasets","bitsandbytes","numpy"):
    try:
        mod = importlib.import_module(m); print(f"  {m}: {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  {m}: NOT IMPORTABLE ({type(e).__name__}: {e})")
import torch; print("  cuda_available:", torch.cuda.is_available(), "| cuda:", torch.version.cuda)
PY

echo ">> NOTE: training must run with USE_TF=0 USE_FLAX=0 (Lambda Stack ships TF+Keras3);"
echo ">>       run_lambda.sh sets these automatically."
echo ">> setup done. Next: bash erosolar_agent/finetune/run_lambda.sh smoke"
