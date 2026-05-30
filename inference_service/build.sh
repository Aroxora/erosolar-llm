#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Assemble the Cloud Run build context, then deploy the erosolar inference API.
# The model checkpoint and the imported modules are copied from the repo root at
# build time (they are not tracked here — see ../.gitignore), so the service always
# serves the current models/erosolar-v0.01 checkpoint.
set -euo pipefail
cd "$(dirname "$0")"

cp ../model.py ../infini_attention.py ../config.py ../tokenizer.py ../registry.py .
rm -rf models && mkdir -p models
cp -R ../models/erosolar-v0.01 models/
cp ../models/registry.json models/
echo "assembled build context for erosolar-v0.01"

gcloud run deploy erosolar-api \
  --source . \
  --region us-central1 \
  --project erosolar-llm \
  --allow-unauthenticated \
  --memory 2Gi --cpu 1 \
  --min-instances 0 --max-instances 2 --concurrency 4 --timeout 60 \
  --quiet
