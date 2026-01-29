#!/bin/bash
# Erosolar Training and Deployment Script
# Runs: GPT-5.1-codex-mini weakness-targeted enhancement -> Train -> Deploy to Cloud Run + Firebase

set -e

# Configuration - uses canonical "erosolar" model that gets replaced each training
MODEL_NAME="${1:-erosolar}"
PRESET="${2:-medium-2}"
EPOCHS="${3:-10}"
GENERATIONS="${4:-5}"
PROMPTS_PER_GEN="${5:-150}"

echo "=============================================="
echo "  Erosolar Training and Deployment Pipeline"
echo "=============================================="
echo "  Model: $MODEL_NAME"
echo "  Preset: $PRESET"
echo "  Epochs: $EPOCHS"
echo "  Generations: $GENERATIONS"
echo "  Prompts/Gen: $PROMPTS_PER_GEN"
echo ""

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "ERROR: Missing required command: $1"
        exit 1
    fi
}

require_cmd python3
require_cmd docker
require_cmd gcloud
require_cmd firebase

# Set API key (required for GPT-5.1-codex-mini enhancement)
# Set OPENAI_API_KEY environment variable before running this script
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY environment variable not set"
    echo "Run: export OPENAI_API_KEY='your-api-key'"
    exit 1
fi

# Step 1: Train with GPT-5.1-codex-mini enhancement
echo "[1/4] Training with GPT-5.1-codex-mini weakness-targeted enhancement..."
python3 train.py \
    --name "$MODEL_NAME" \
    --desc "GPT-5.1-codex-mini weakness-targeted enhanced model" \
    --preset "$PRESET" \
    --epochs "$EPOCHS" \
    --generations "$GENERATIONS" \
    --prompts-per-gen "$PROMPTS_PER_GEN" \
    --upgrade-pipeline

echo ""
echo "[2/4] Building Docker image..."
docker build -t erosolar-api .
docker tag erosolar-api gcr.io/america-is-finally-back/erosolar-api

echo ""
echo "[3/4] Deploying to Cloud Run..."
gcloud auth configure-docker gcr.io --quiet
docker push gcr.io/america-is-finally-back/erosolar-api
gcloud run deploy erosolar-api \
    --image gcr.io/america-is-finally-back/erosolar-api \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --timeout 60 \
    --project america-is-finally-back

echo ""
echo "[4/4] Deploying chat interface to Firebase..."
firebase deploy --only hosting --project america-is-finally-back

echo ""
echo "=============================================="
echo "  DEPLOYMENT COMPLETE!"
echo "=============================================="
echo "  Model: $MODEL_NAME"
echo "  API: https://erosolar-api-13762901352.us-central1.run.app"
echo "  Chat: https://america-is-finally-back.web.app"
echo ""
echo "  Test locally: python3 generate.py --model $MODEL_NAME"
echo "  Test API: curl -X POST https://erosolar-api-13762901352.us-central1.run.app/api/chat -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\"}'"
echo ""
