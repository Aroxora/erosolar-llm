#!/bin/bash
# ============================================
# OPTIMAL KIDS MODEL TRAINING - Minimal Epochs
# ============================================
#
# This script trains a kids-model that produces
# coherent, accurate responses with MINIMAL epochs.
#
# Key optimizations:
# 1. Rebalanced training data (KIDS_QA at 500x weight)
# 2. Reduced Constitutional data (prevents fragment overflow)
# 3. Lower target loss (0.25) for coherent output
# 4. Higher learning rate (3e-4) for faster convergence
# 5. Lower dropout (0.08) - less regularization needed
#
# ============================================

echo "=============================================="
echo "  OPTIMAL KIDS MODEL - MINIMAL EPOCHS"
echo "=============================================="
echo ""
echo "Data optimizations applied:"
echo "  - KIDS_QA: 500x weight (was 200x)"
echo "  - Constitution data: 20x (was 60x)"
echo "  - Prevents fragment overflow in responses"
echo ""
echo "Training parameters:"
echo "  - Size: LARGE (85M params)"
echo "  - Epochs: 12 (minimal for convergence)"
echo "  - Target Loss: 0.25 (coherent output)"
echo "  - Learning Rate: 3e-4 (faster convergence)"
echo "  - Batch Size: 24 (optimal for GPU memory)"
echo "=============================================="
echo ""

# Remove old model if exists
if [ -d "models/kids-model" ]; then
    echo "Removing old kids-model..."
    rm -rf models/kids-model
fi

# Train with optimal minimal-epoch parameters
python train_fast.py \
    --name kids-model \
    --size large \
    --epochs 12 \
    --target-loss 0.25 \
    --patience 6 \
    --kids \
    --batch 24

echo ""
echo "=============================================="
echo "  TRAINING COMPLETE!"
echo "=============================================="
echo ""
echo "Test the model with:"
echo "  python gen.py -m kids-model -p 'Why did dinosaurs go extinct?'"
echo "  python gen.py -m kids-model -p 'What are the three branches of government?'"
echo "  python gen.py -m kids-model -p 'Who was the first president?'"
echo ""
echo "Run full evaluation:"
echo "  python generate_kids_prompts.py -m kids-model -n 50 -s 20"
echo ""
