#!/bin/bash
# ============================================
# KIDS MODEL - CLEAN DATA + PROPER TRAINING
# ============================================
# Fixed issues:
#   1. Removed external datasets with security content
#   2. KIDS_QA now at 1000x weight (dominant)
#   3. Large model for better knowledge retention
#   4. 15 epochs for proper convergence
#
# Previous failures:
#   3 epochs small  → 2.6/10 (garbage)
#   8 epochs medium → 2.5/10 (polluted data)
# ============================================

echo "=============================================="
echo "  KIDS MODEL - CLEAN BUILD"
echo "=============================================="
echo ""
echo "Fixed:"
echo "  ✓ Removed external datasets (had security content)"
echo "  ✓ KIDS_QA at 1000x weight (highest priority)"
echo "  ✓ Only kid-friendly curated content"
echo ""
echo "Training:"
echo "  - Model: LARGE (85M params)"
echo "  - Epochs: 15"
echo "  - Target Loss: 0.25"
echo "  - Batch: 16"
echo ""
echo "Expected: 25-40 min, score 7-8/10"
echo "=============================================="

rm -rf models/kids-model 2>/dev/null

python train_fast.py \
    --name kids-model \
    --size large \
    --epochs 15 \
    --target-loss 0.25 \
    --patience 5 \
    --kids \
    --batch 16

echo ""
echo "=============================================="
echo "  TRAINING COMPLETE"
echo "=============================================="
echo ""
echo "Quick tests:"
echo "  python gen.py -m kids-model -p 'Why is the sky blue?'"
echo "  python gen.py -m kids-model -p 'What is 47 + 18?'"
echo "  python gen.py -m kids-model -p 'Who was the first president?'"
echo "  python gen.py -m kids-model -p 'Why did dinosaurs go extinct?'"
echo ""
echo "Full evaluation:"
echo "  python generate_kids_prompts.py -m kids-model -n 50 -s 50"
