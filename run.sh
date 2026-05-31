#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# run.sh — reproduce the honest erosolar appreciation pipeline end to end.
#
# One command for the whole, honest loop:
#   data (license-clean, self-generated) -> train -> benchmark -> invariant tests
#   -> [optional] deploy the model to Cloud Run and the app to Firebase Hosting.
#
# Honesty rules are enforced by the code itself: status is "pending" until measured,
# no capability-class labels, no model distillation (default corpus is template-composed),
# and test_invariants.py fails if any of that regresses.
#
# Usage:
#   ./run.sh                 # train (large) -> benchmark -> tests
#   ./run.sh --quick         # fast tiny run to verify the path
#   ./run.sh --size base     # pick a model size (tiny|small|base|large)
#   ./run.sh --no-train      # skip training, just benchmark + test the current checkpoint
#   ./run.sh --deploy        # also hot-swap Cloud Run + redeploy the web app
#   ./run.sh --task math     # the grounded arithmetic task instead of appreciation
#   ./run.sh --generalize    # char + copy-augmentation zero-shot experiment (~89% held-out)
set -euo pipefail
cd "$(dirname "$0")"

SIZE="large"; TASK="appreciation"; TRAIN=1; DEPLOY=0; GENERALIZE=0; EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --quick)      EXTRA+=(--quick); SIZE="tiny" ;;
    --size)       SIZE="$2"; shift ;;
    --task)       TASK="$2"; shift ;;
    --no-train)   TRAIN=0 ;;
    --deploy)     DEPLOY=1 ;;
    --generalize) GENERALIZE=1 ;;
    *) echo "unknown flag: $1"; exit 2 ;;
  esac
  shift
done

export PYTORCH_ENABLE_MPS_FALLBACK=1

# Zero-shot generalization recipe (the measured 0% -> 89% result): a char tokenizer makes
# held-out qualities emittable, and copy-augmentation with length-matched nonces (3-15 chars)
# induces the cue->answer copy head. Holds 24 qualities out and reports held-out validity to
# data_store/generalization.json. Separate diagnostic — does NOT touch the shipped model.
if [ "$GENERALIZE" -eq 1 ]; then
  echo ">> generalization experiment: char tokenizer + copy-augmentation (reproduces ~89% zero-shot)"
  python3 honest_pipeline.py --task appreciation --tokenizer char --size "$SIZE" \
    --holdout 24 --copy-aug 12000 --samples 14000 --epochs 18 --name erosolar-charcopy
  echo ">> held-out (zero-shot) validity is in data_store/generalization.json"
  exit 0
fi

if [ "$TRAIN" -eq 1 ]; then
  echo ">> [1/4] train (license-clean, self-generated data; status pending until measured)"
  python3 honest_pipeline.py --task "$TASK" --size "$SIZE" --samples 28000 --epochs 12 "${EXTRA[@]}"
fi

echo ">> [2/4] benchmark (real, task-appropriate metrics only)"
python3 benchmark_appreciation.py --name erosolar-v0.01 --n 200

echo ">> [3/4] invariant tests (no capability claims, no tautologies, all mapped, license-clean)"
python3 test_invariants.py

if [ "$DEPLOY" -eq 1 ]; then
  echo ">> [4/4] deploy: hot-swap the hosted model (Cloud Run) + redeploy the web app"
  ( cd inference_service && bash build.sh )
  ( cd angular-chat && npx ng build erosolar-web --configuration production )
  rm -rf erosolar-web-deploy/public && mkdir -p erosolar-web-deploy/public
  cp -R angular-chat/dist/erosolar-web/browser/. erosolar-web-deploy/public/
  ( cd erosolar-web-deploy && firebase deploy --only hosting --project erosolar-llm )
else
  echo ">> [4/4] deploy skipped (pass --deploy to hot-swap Cloud Run + redeploy the app)"
fi

echo ">> done. Measured numbers are in data_store/version.json, benchmarks.json, judge_report.json."
