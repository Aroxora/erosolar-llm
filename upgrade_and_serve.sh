#!/bin/bash
#
# EROSOLAR - ITERATIVE TRAINING PIPELINE
# =======================================
# Wrapper that calls the Python pipeline script.
#
# Usage:
#   ./upgrade_and_serve.sh                # Start iterative training
#   ./upgrade_and_serve.sh --local        # Train and serve locally (no deploy)
#   ./upgrade_and_serve.sh --deploy-only  # Just deploy latest model
#   ./upgrade_and_serve.sh --version 3    # Start from specific version
#   ./upgrade_and_serve.sh --once         # Run only one iteration
#

# Load .env if exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$SCRIPT_DIR/.env" ] && source "$SCRIPT_DIR/.env"

# Run Python pipeline with all arguments
exec python3 "$SCRIPT_DIR/pipeline.py" "$@"
