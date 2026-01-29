#!/bin/bash
# ============================================
# ULTRA-FAST KIDS MODEL - 3 EPOCHS
# ============================================
# WARNING: 3 epochs is very aggressive.
# Results may be inconsistent.
# ============================================

echo "=============================================="
echo "  ULTRA-FAST KIDS MODEL - 3 EPOCHS"
echo "=============================================="
echo ""
echo "⚠️  WARNING: 3 epochs is minimal."
echo "    For best results, use 10+ epochs."
echo ""
echo "Aggressive optimizations:"
echo "  - Small model (faster convergence)"
echo "  - High learning rate (8e-4)"
echo "  - Large batch (64)"
echo "  - No label smoothing"
echo "  - Target loss: 0.5 (achievable in 3 epochs)"
echo "=============================================="

# Remove old model
rm -rf models/kids-model 2>/dev/null

# Ultra-fast training
python train_fast.py \
    --name kids-model \
    --size small \
    --epochs 3 \
    --target-loss 0.50 \
    --patience 3 \
    --kids \
    --batch 64

echo ""
echo "Done! Test with:"
echo "  python gen.py -m kids-model -p 'Why is the sky blue?'"
