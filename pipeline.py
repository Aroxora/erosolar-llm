#!/usr/bin/env python3
"""
EROSOLAR - ITERATIVE TRAINING PIPELINE v2
==========================================
Generates data and trains versioned models to infinity.

Flow:
  1. Generate data to target master scalar → round_01.jsonl
  2. Train erosolar v0.01 with round_01.jsonl
  3. Generate more data to higher master scalar → round_02.jsonl
  4. Train erosolar v0.02 with round_01.jsonl + round_02.jsonl (cumulative)
  5. Repeat forever...

Each version trains on ALL previous rounds (cumulative complexity).

Usage:
  python pipeline.py                # Start iterative training (generate → train → deploy)
  python pipeline.py --train-only   # Train existing round data (no generation)
  python pipeline.py --deploy-only  # Just deploy latest model
  python pipeline.py --no-deploy    # Skip deployment (local dev)
  python pipeline.py --skip-frontend  # Deploy API only (skip Angular frontend)
  python pipeline.py --version 3    # Start from specific version
  python pipeline.py --once         # Run only one iteration
"""

import os
import sys
import json
import subprocess
import argparse
from typing import Optional
import shutil
import time
import requests
from pathlib import Path
from datetime import datetime
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# GCP backup config
GCS_BUCKET = os.environ.get("GCS_BUCKET", "erosolar-training-data")
GCP_BACKUP_ENABLED = os.environ.get("GCP_BACKUP_ENABLED", "true").lower() == "true"

# Ollama local API (no API key needed)
os.environ.setdefault("DEEPSEEK_API_KEY", "ollama")
os.environ.setdefault("DEEPSEEK_API_BASE", "http://localhost:11434/v1")
os.environ.setdefault("OPENAI_API_KEY", "ollama")

# Dynamic categories and seed tracking
try:
    from dynamic_categories import (
        DynamicCategoryManager,
        get_seed_tracker,
        inject_dynamic_categories
    )
    DYNAMIC_CATEGORIES_AVAILABLE = True
except ImportError:
    DYNAMIC_CATEGORIES_AVAILABLE = False

# Attention calculations (real CoT attention metrics used in training)
# Standard: attention = softmax(QK^T/√d)V
# Perfect:  attention = softmax(QK^T/√d + R)V
# Where R is a reasoning prior that biases attention toward similar reasoning patterns
try:
    from reasoning_consistency import (
        ReasoningConsistencyManager,
        PerfectSelfAttention,
        SharedCoTWeightOptimizer,
        ReasoningPriorComputer,
    )
    from auto_attention import (
        CodexAttentionManager,
        AutoAttentionTrainer,
        get_attention_manager,
        get_adaptive_selector,
        manage_all_existing_data,
        DataAttentionManager,
    )
    PERFECT_ATTENTION_AVAILABLE = True
except ImportError:
    PERFECT_ATTENTION_AVAILABLE = False

# Self-generation pipeline - closed-loop self-improvement
try:
    from self_generation import (
        SelfGenerationPipeline,
        run_self_generation_round,
        should_use_self_generation,
    )
    SELF_GENERATION_AVAILABLE = True
except ImportError:
    SELF_GENERATION_AVAILABLE = False

# Benchmark tracking and quality gate
try:
    from benchmarks import (
        BenchmarkRunner,
        QualityGate,
        BenchmarkHistory,
        run_post_training_benchmarks,
        should_accept_training_round,
    )
    BENCHMARKS_AVAILABLE = True
except ImportError:
    BENCHMARKS_AVAILABLE = False

# Local embeddings for loser/friend optimization
try:
    from local_embeddings import embed_batch, average_pairwise_similarity
    from mini_cot_optimizer import CoTSelfAttentionOptimizer
    LOSER_FRIEND_AVAILABLE = True
except ImportError:
    LOSER_FRIEND_AVAILABLE = False

# Grounded verification for data quality
try:
    from grounded_verification import GroundedVerifier
    GROUNDED_VERIFICATION_AVAILABLE = True
except ImportError:
    GROUNDED_VERIFICATION_AVAILABLE = False

# Colors for terminal output
class Colors:
    CYAN = '\033[0;36m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    BOLD = '\033[1m'
    NC = '\033[0m'  # No Color

def cprint(msg: str, color: str = Colors.NC):
    """Print with color."""
    print(f"{color}{msg}{Colors.NC}")

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_STORE = SCRIPT_DIR / "data_store"
VERSION_FILE = DATA_STORE / "version.json"
ROUNDS_DIR = DATA_STORE / "rounds"
CACHE_DIR = SCRIPT_DIR / "cache" / "optimal_gen"
GENERATED_FILE = DATA_STORE / "generated_training_data.jsonl"
FRONTEND_DIR = SCRIPT_DIR / "angular_app" / "dist" / "browser"
FRONTEND_INDEX = FRONTEND_DIR / "index.html"
FIREBASE_CONFIG = SCRIPT_DIR / "firebase.json"


# ════════════════════════════════════════════════════════════════════════════════
# GCP BACKUP - Persistent cloud storage for training data
# ════════════════════════════════════════════════════════════════════════════════

def check_gcloud_auth() -> bool:
    """Check if gcloud is authenticated."""
    try:
        result = subprocess.run(
            ["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"],
            capture_output=True, text=True, timeout=10
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def ensure_gcs_bucket() -> bool:
    """Ensure GCS bucket exists with versioning enabled."""
    if not GCP_BACKUP_ENABLED:
        return False

    try:
        # Check if bucket exists
        result = subprocess.run(
            ["gsutil", "ls", f"gs://{GCS_BUCKET}"],
            capture_output=True, text=True, timeout=30
        )

        if result.returncode != 0:
            # Create bucket
            cprint(f"  Creating GCS bucket: gs://{GCS_BUCKET}", Colors.YELLOW)
            project = os.environ.get("GOOGLE_CLOUD_PROJECT", "erosolar-llm")
            region = os.environ.get("GCS_REGION", "us-central1")

            subprocess.run(
                ["gsutil", "mb", "-p", project, "-l", region, f"gs://{GCS_BUCKET}"],
                check=True, timeout=60
            )
            # Enable versioning
            subprocess.run(
                ["gsutil", "versioning", "set", "on", f"gs://{GCS_BUCKET}"],
                check=True, timeout=30
            )
            cprint(f"  ✓ Created bucket with versioning enabled", Colors.GREEN)

        return True
    except Exception as e:
        cprint(f"  GCS bucket setup failed: {e}", Colors.YELLOW)
        return False


def backup_to_gcs(file_path: Path, gcs_path: str = None) -> bool:
    """Backup a file to GCS."""
    if not GCP_BACKUP_ENABLED:
        return False

    if not file_path.exists():
        return False

    if gcs_path is None:
        gcs_path = f"gs://{GCS_BUCKET}/latest/{file_path.name}"

    try:
        result = subprocess.run(
            ["gsutil", "cp", str(file_path), gcs_path],
            capture_output=True, text=True, timeout=300
        )

        if result.returncode == 0:
            cprint(f"  ✓ Backed up to {gcs_path}", Colors.GREEN)
            return True
        else:
            cprint(f"  GCS backup failed: {result.stderr}", Colors.YELLOW)
            return False
    except Exception as e:
        cprint(f"  GCS backup error: {e}", Colors.YELLOW)
        return False


def backup_round_to_gcs(round_num: int, round_file: Path) -> bool:
    """Backup a round file to GCS with proper organization."""
    if not GCP_BACKUP_ENABLED or not round_file.exists():
        return False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Backup to multiple locations for redundancy
    paths = [
        f"gs://{GCS_BUCKET}/rounds/round_{round_num:02d}.jsonl",  # Latest round
        f"gs://{GCS_BUCKET}/snapshots/{timestamp}/round_{round_num:02d}.jsonl",  # Timestamped
    ]

    success = True
    for gcs_path in paths:
        if not backup_to_gcs(round_file, gcs_path):
            success = False

    return success


def sync_data_store_to_gcs() -> bool:
    """Sync entire data_store directory to GCS."""
    if not GCP_BACKUP_ENABLED:
        return False

    if not DATA_STORE.exists():
        return False

    try:
        cprint(f"  Syncing data_store to GCS...", Colors.YELLOW)
        result = subprocess.run(
            ["gsutil", "-m", "rsync", "-r", str(DATA_STORE), f"gs://{GCS_BUCKET}/latest/"],
            capture_output=True, text=True, timeout=600
        )

        if result.returncode == 0:
            cprint(f"  ✓ Synced to gs://{GCS_BUCKET}/latest/", Colors.GREEN)
            return True
        else:
            cprint(f"  GCS sync warning: {result.stderr}", Colors.YELLOW)
            return False
    except Exception as e:
        cprint(f"  GCS sync error: {e}", Colors.YELLOW)
        return False


def load_env():
    """Load environment from .env file."""
    env_file = SCRIPT_DIR / ".env"
    if env_file.exists():
        if load_dotenv:
            load_dotenv(env_file, override=True)
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value


def get_models(api_key: str = None) -> list:
    """Return DeepSeek reasoner model."""
    return ["deepseek-reasoner"]


# ════════════════════════════════════════════════════════════════════════════════
# DATA GENERATION - Uses generate_all_training_data.py via subprocess
# ════════════════════════════════════════════════════════════════════════════════

def generate_round_data(round_num: int, target: int, model: str,
                        target_score: Optional[float] = None,
                        max_records: Optional[int] = None,
                        min_records: int = 1000) -> tuple[int, int]:
    """
    Generate data for a specific round.
    PERSISTENT: Data is always appended and never lost.

    Returns (total_records, new_records).
    """
    ROUNDS_DIR.mkdir(parents=True, exist_ok=True)
    round_file = ROUNDS_DIR / f"round_{round_num:02d}.jsonl"

    # Check existing count
    existing = 0
    if round_file.exists():
        with open(round_file) as f:
            existing = sum(1 for _ in f)

    if target_score is None:
        if existing >= target:
            cprint(f"  Round {round_num} already has {existing:,} records (target: {target:,})", Colors.GREEN)
            return existing, 0
    else:
        if max_records is not None:
            cap = max_records
            if existing >= cap:
                cprint(f"  Round {round_num} already has {existing:,} records (cap reached)", Colors.GREEN)
                return existing, 0

    if target_score is None:
        to_generate = target - existing
        cprint(f"  Generating {to_generate:,} records for round {round_num} (existing: {existing:,})", Colors.YELLOW)
    else:
        cap_note = f", cap: {max_records:,}" if max_records is not None else ""
        cprint(f"  Generating round {round_num} to target master scalar {target_score:.4f} "
               f"(existing: {existing:,}{cap_note})", Colors.YELLOW)

    # FIXED: Copy existing round data to GENERATED_FILE (no symlinks - prevents data loss)
    GENERATED_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Remove old file if exists (not symlink)
    if GENERATED_FILE.is_symlink():
        GENERATED_FILE.unlink()
    elif GENERATED_FILE.exists():
        # Backup existing generated file
        backup = GENERATED_FILE.with_suffix('.jsonl.bak')
        shutil.copy2(GENERATED_FILE, backup)
        GENERATED_FILE.unlink()

    # Copy ALL previous rounds' data to generated file (for deduplication + continuation)
    # This ensures new rounds build on cumulative data, not start fresh
    total_copied = 0
    with open(GENERATED_FILE, 'w') as outf:
        for r in range(1, round_num + 1):
            prev_round = ROUNDS_DIR / f"round_{r:02d}.jsonl"
            if prev_round.exists():
                with open(prev_round) as inf:
                    for line in inf:
                        outf.write(line)
                        total_copied += 1
    if total_copied > 0:
        cprint(f"  Copied {total_copied:,} records from rounds 1-{round_num} to generator input", Colors.CYAN)
    else:
        cprint(f"  Starting fresh for round {round_num}", Colors.CYAN)

    # Run generate_all_training_data.py
    cmd = [
        sys.executable, "generate_all_training_data.py",
        "--model", model,
        "--resume"  # Always resume - appends to existing data
    ]
    if target_score is None:
        cmd.extend(["--target", str(target)])
    else:
        cmd.extend(["--target-score", str(target_score)])
        if max_records is not None:
            cmd.extend(["--max-records", str(max_records)])
        cmd.extend(["--min-records", str(min_records)])

    cprint(f"  Generating to {GENERATED_FILE} (persistent, resumable)", Colors.CYAN)
    env = os.environ.copy()
    if round_num == 1:
        env.setdefault("LONG_FORM", "1")
        env.setdefault("LONG_FORM_OUTPUT_TOKENS", "8192")
    result = subprocess.run(cmd, cwd=SCRIPT_DIR, env=env)
    if result.returncode != 0:
        cprint(f"  Generation failed (exit {result.returncode}) - aborting before training.", Colors.RED)
        sys.exit(result.returncode)

    # Optional: append coding-only data after main generation
    coding_target = os.environ.get("MINI_CODING_TARGET")
    if coding_target is not None:
        try:
            coding_target_val = int(coding_target)
        except ValueError:
            coding_target_val = -1
    else:
        coding_target_val = -1

    if coding_target_val != 0:
        coding_script = SCRIPT_DIR / "generate_coding_only.py"
        if coding_script.exists():
            cprint("  Generating coding-only data (append-only)...", Colors.CYAN)
            coding_cmd = [sys.executable, str(coding_script), "--target", str(coding_target_val)]
            coding_result = subprocess.run(coding_cmd, cwd=SCRIPT_DIR, env=env)
            if coding_result.returncode != 0:
                cprint(f"  Coding-only generation failed (exit {coding_result.returncode})", Colors.YELLOW)
        else:
            cprint("  Coding-only generator not found, skipping", Colors.YELLOW)

    # ALWAYS copy result back to round file (persist data)
    if GENERATED_FILE.exists():
        with open(GENERATED_FILE) as f:
            final_count = sum(1 for _ in f)

        if final_count > 0:
            # Copy to round file (this is the persistent storage)
            shutil.copy2(GENERATED_FILE, round_file)
            cprint(f"  ✓ Persisted {final_count:,} records to {round_file}", Colors.GREEN)

            # Also backup locally
            backup = round_file.with_suffix('.jsonl.bak')
            shutil.copy2(round_file, backup)

            # Backup to GCS (cloud persistence)
            if GCP_BACKUP_ENABLED and check_gcloud_auth():
                backup_round_to_gcs(round_num, round_file)

            if result.returncode != 0:
                cprint(f"  Generation interrupted but data saved (resume to continue)", Colors.YELLOW)

            new_count = max(0, final_count - existing)
            return final_count, new_count

    if result.returncode != 0:
        cprint(f"  Generation failed (code {result.returncode})", Colors.RED)
        return existing, 0

    return existing, 0


def print_master_scalar_optimizer(target_score: Optional[float]) -> float:
    """Print master scalar optimizer snapshot (required before training)."""
    cmd = [sys.executable, "generate_all_training_data.py", "--print-master-scalar"]
    if target_score is not None:
        cmd.extend(["--target-score", str(target_score)])
    cprint(f"  Printing master scalar optimizer snapshot...", Colors.CYAN)
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        cprint("  Master scalar snapshot failed - aborting before training.", Colors.RED)
        sys.exit(result.returncode)
    return get_master_scalar_from_context()


def get_master_scalar_from_context() -> float:
    """Read master scalar from the loser context file."""
    ctx_file = DATA_STORE / "loser_context.json"
    if not ctx_file.exists():
        return 0.0
    try:
        with open(ctx_file) as f:
            ctx = json.load(f)
        return float(ctx.get("master_scalar", 0.0) or 0.0)
    except Exception:
        return 0.0


def combine_rounds_for_training(up_to_round: int) -> int:
    """Combine all rounds up to specified round into optimal_training.jsonl for training."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    combined_file = CACHE_DIR / "optimal_training.jsonl"

    total = 0
    with open(combined_file, 'w') as out:
        for r in range(1, up_to_round + 1):
            round_file = ROUNDS_DIR / f"round_{r:02d}.jsonl"
            if round_file.exists():
                with open(round_file) as f:
                    count = 0
                    for line in f:
                        out.write(line)
                        count += 1
                        total += 1
                cprint(f"    + Round {r}: {count:,} records", Colors.GREEN)
            else:
                cprint(f"    + Round {r}: NOT FOUND", Colors.RED)

    cprint(f"  Combined {total:,} records from {up_to_round} rounds", Colors.GREEN)
    return total


# ════════════════════════════════════════════════════════════════════════════════
# VERSION MANAGEMENT
# ════════════════════════════════════════════════════════════════════════════════

def get_version() -> int:
    """Get current version from version.json."""
    if VERSION_FILE.exists():
        with open(VERSION_FILE) as f:
            return json.load(f).get("version", 0)
    return 0


def set_version(version: int, record_count: int):
    """Save version info."""
    DATA_STORE.mkdir(parents=True, exist_ok=True)
    with open(VERSION_FILE, 'w') as f:
        json.dump({
            "version": version,
            "total_records": record_count,
            "version_string": f"v0.{version:02d}",
            "updated": datetime.now().isoformat()
        }, f, indent=2)


def format_version(version: int) -> str:
    """Format version as v0.XX."""
    return f"v0.{version:02d}"


# ════════════════════════════════════════════════════════════════════════════════
# TRAINING
# ════════════════════════════════════════════════════════════════════════════════

def get_training_config(version: int, base_preset: str, base_epochs: int,
                        auto_optimize: bool = True, force_epochs: bool = False) -> tuple:
    """
    Get progressive training config based on version.

    v0.01 uses a smaller starter set with master scalar gating.
    This keeps optimization focused on the master scalar target.
    """
    if auto_optimize and version == 1:
        # v0.01: Start small; master scalar gating controls progression
        return "infini-small", (base_epochs if force_epochs else 15)

    # Progressive model capacity - start with usable size
    presets_by_version = {
        1: "infini-small",   # v0.01: small starter phase
        2: "infini-small",   # v0.02: builds on v0.01 baseline
        3: "infini-medium",  # v0.03: 50M params - scaled up
        4: "infini-medium",  # v0.04+: 50M params - full capacity
    }

    # Progressive epochs
    epochs_by_version = {
        1: 15,  # v0.01: More epochs while gating on master scalar
        2: 10,  # v0.02: Builds on the v0.01 baseline
        3: 7,   # v0.03: 75K cumulative
        4: 5,   # v0.04+: 100K+ cumulative
    }

    preset = presets_by_version.get(version, base_preset)
    if force_epochs:
        epochs = base_epochs
    else:
        epochs = epochs_by_version.get(version, base_epochs)

    return preset, epochs


def get_adaptive_batch_size(version: int, base_batch: int, auto_optimize: bool = True) -> int:
    """
    Get adaptive batch size based on version and auto-optimization.

    v0.01 with auto-optimize uses smaller initial data, gated by master scalar.
    """
    if auto_optimize and version == 1:
        # Start with 5K records, gated by master scalar
        # This avoids brute-force data scaling before coherence improves
        return 5000

    # Later versions scale up
    adaptive_sizes = {
        1: 5000,   # v0.01: Small starter set
        2: 15000,  # v0.02: Medium
        3: 25000,  # v0.03: Full
        4: 25000,  # v0.04+: Full
    }

    return adaptive_sizes.get(version, base_batch)


def train_model(version: int, version_str: str, record_count: int, base_preset: str, base_epochs: int,
                perfect_attention: bool = True, force_epochs: bool = False) -> bool:
    """Train the erosolar model using ONLY JSONL data (no conflicting base data).

    Attention calculations (real CoT attention metrics used in training):
    - Standard: attention = softmax(QK^T/√d)V
    - Perfect:  attention = softmax(QK^T/√d + R)V
    - R = reasoning prior that biases attention toward similar reasoning patterns
    """

    # Get progressive config
    preset, epochs = get_training_config(version, base_preset, base_epochs, force_epochs=force_epochs)

    cmd = [
        sys.executable, "train.py",
        "--name", "erosolar",
        "--desc", f"erosolar {version_str} - trained on {record_count} cumulative records",
        "--preset", preset,
        "--epochs", str(epochs),
        "--balanced"  # CRITICAL: This skips base data, uses ONLY JSONL
    ]

    # Enable attention calculations (reasoning consistency)
    if perfect_attention and PERFECT_ATTENTION_AVAILABLE:
        cmd.extend([
            "--reasoning-consistency",
            "--contrastive-weight", "0.1",
            "--shared-cot-weights",
            "--shared-cot-boost", "0.5",
        ])
        cprint(f"  [AttentionCalculations] Enabled: attention = softmax(QK^T/√d + R)V", Colors.CYAN)

    cprint(f"  Training erosolar {version_str} with {record_count:,} cumulative records...", Colors.YELLOW)
    cprint(f"  Config: {preset}, {epochs} epochs", Colors.YELLOW)
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    return result.returncode == 0


# ════════════════════════════════════════════════════════════════════════════════
# DEPLOYMENT
# ════════════════════════════════════════════════════════════════════════════════

def sync_firebase_rewrites(service: str, region: str) -> bool:
    """Ensure Firebase rewrites point at the configured Cloud Run service."""
    if not FIREBASE_CONFIG.exists():
        cprint("  WARNING: firebase.json not found, skipping rewrite sync", Colors.YELLOW)
        return False

    with open(FIREBASE_CONFIG) as f:
        config = json.load(f)

    rewrites = config.get("hosting", {}).get("rewrites", [])
    updated = False
    for rewrite in rewrites:
        run_cfg = rewrite.get("run")
        source = rewrite.get("source", "")
        if not run_cfg:
            continue
        if source.startswith("/api") or source.startswith("/v1"):
            if run_cfg.get("serviceId") != service or run_cfg.get("region") != region:
                run_cfg["serviceId"] = service
                run_cfg["region"] = region
                updated = True

    if updated:
        with open(FIREBASE_CONFIG, 'w') as f:
            json.dump(config, f, indent=2)
        cprint("  ✓ Firebase rewrites updated", Colors.GREEN)
    return updated


def ensure_frontend_artifacts() -> bool:
    """Verify the Angular build output exists before deploying."""
    if not FRONTEND_DIR.exists():
        cprint(f"  ERROR: Frontend build directory missing at {FRONTEND_DIR}", Colors.RED)
        return False
    if not FRONTEND_INDEX.exists():
        cprint(f"  ERROR: Frontend build missing at {FRONTEND_INDEX}", Colors.RED)
        return False
    return True


def deploy(version_str: str, args: argparse.Namespace) -> bool:
    """Deploy to Cloud Run and Firebase Hosting."""
    cprint(f"\n[STEP 3/3] Deploying erosolar {version_str}", Colors.CYAN)

    project = args.cloud_run_project
    region = args.cloud_run_region
    service = args.cloud_run_service
    source = args.cloud_run_source
    image = args.cloud_run_image
    firebase_project = args.firebase_project or project

    if firebase_project != project:
        cprint("  WARNING: Firebase project differs from Cloud Run project", Colors.YELLOW)

    # Deploy to Cloud Run (API)
    cprint("  Deploying to Cloud Run...", Colors.YELLOW)
    if image:
        cprint(f"  Using Cloud Run image: {image}", Colors.YELLOW)
        cmd_gcloud = [
            "gcloud", "run", "deploy", service,
            "--image", image,
            "--region", region,
            "--allow-unauthenticated",
            "--memory", "2Gi",
            "--timeout", "300",
            "--project", project
        ]
    else:
        cprint(f"  Using Cloud Run source: {source}", Colors.YELLOW)
        cmd_gcloud = [
            "gcloud", "run", "deploy", service,
            "--source", source,
            "--region", region,
            "--allow-unauthenticated",
            "--memory", "2Gi",
            "--timeout", "300",
            "--project", project
        ]

    result = subprocess.run(cmd_gcloud, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        cprint("  Cloud Run deployment failed!", Colors.RED)
        return False

    cprint("  ✓ Cloud Run deployed", Colors.GREEN)

    # Deploy to Firebase Hosting (Angular frontend)
    if not args.skip_frontend:
        if not ensure_frontend_artifacts():
            return False
        sync_firebase_rewrites(service, region)
        cprint("  Deploying to Firebase Hosting...", Colors.YELLOW)
        cmd_firebase = [
            "firebase", "deploy", "--only", "hosting",
            "--project", firebase_project
        ]

        result = subprocess.run(cmd_firebase, cwd=SCRIPT_DIR)
        if result.returncode != 0:
            cprint("  Firebase deployment failed!", Colors.RED)
            return False
        cprint("  ✓ Firebase Hosting deployed", Colors.GREEN)
    else:
        cprint("  Skipping Firebase Hosting (--skip-frontend)", Colors.YELLOW)

    # Success banner
    cprint("\n╔══════════════════════════════════════════════════════════════╗", Colors.GREEN)
    cprint(f"║  DEPLOYED: erosolar {version_str}                               ║", Colors.GREEN)
    cprint("╠══════════════════════════════════════════════════════════════╣", Colors.GREEN)
    cprint(f"║  API:  Cloud Run {service} ({region})                         ║", Colors.GREEN)
    cprint(f"║  Web:  Firebase Hosting {firebase_project}                    ║", Colors.GREEN)
    cprint("╚══════════════════════════════════════════════════════════════╝", Colors.GREEN)

    return True


# ════════════════════════════════════════════════════════════════════════════════
# DYNAMIC CATEGORIES & SEED TRACKING
# ════════════════════════════════════════════════════════════════════════════════

def print_seed_coverage():
    """Print seed coverage statistics."""
    if not DYNAMIC_CATEGORIES_AVAILABLE:
        return

    tracker = get_seed_tracker()
    if not tracker:
        return

    stats = tracker.get_stats()
    cprint(f"\n{'─' * 60}", Colors.CYAN)
    cprint("SEED USAGE TRACKER", Colors.CYAN)
    cprint(f"{'─' * 60}", Colors.CYAN)
    cprint(f"  Unique seeds used: {stats['total_unique_seeds']}", Colors.GREEN)
    cprint(f"  Total uses: {stats['total_uses']}", Colors.GREEN)
    cprint(f"  Categories tracked: {stats['categories']}", Colors.GREEN)

    # Show categories with lowest coverage
    if stats.get('by_category'):
        cprint("\n  Category Coverage:", Colors.CYAN)
        sorted_cats = sorted(
            stats['by_category'].items(),
            key=lambda x: x[1]['unique_seeds'],
            reverse=True
        )[:10]
        for cat, cat_stats in sorted_cats:
            cprint(f"    {cat:<25} {cat_stats['unique_seeds']:>5} seeds, {cat_stats['total_uses']:>6} uses", Colors.GREEN)


async def expand_dynamic_categories():
    """Expand categories using LLM."""
    if not DYNAMIC_CATEGORIES_AVAILABLE:
        cprint("Dynamic categories not available", Colors.YELLOW)
        return

    manager = DynamicCategoryManager()

    # Discover from existing training data
    manager.discover_from_training_data()

    # Expand domains using LLM (if API key available)
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if api_key:
        cprint("Expanding categories with deepseek-reasoner...", Colors.YELLOW)
        await manager.auto_expand_all()

    manager.save()
    manager.print_stats()


async def manage_existing_data_attention(backup_to_gcs: bool = True):
    """
    Manage all existing training data with attention calculations.

    Uses DeepSeek reasoner for attention optimization.
    """
    if not PERFECT_ATTENTION_AVAILABLE:
        cprint("Attention calculations not available", Colors.YELLOW)
        return None

    selector = get_adaptive_selector()
    cprint(f"\n{'─' * 60}", Colors.CYAN)
    cprint("AUTO-ATTENTION DATA MANAGER", Colors.CYAN)
    cprint(f"{'─' * 60}", Colors.CYAN)
    cprint(f"  Adaptive model: {selector.get_current_model()}", Colors.GREEN)
    cprint(f"  Will auto-escalate if optimization quality < 0.5", Colors.GREEN)
    cprint(f"  Will de-escalate after 10 consecutive successes", Colors.GREEN)

    result = await manage_all_existing_data(backup_to_gcs=backup_to_gcs)

    cprint(f"\n  Analysis complete:", Colors.GREEN)
    cprint(f"    Files: {result.get('files_analyzed', 0)}", Colors.GREEN)
    cprint(f"    Records: {result.get('total_records', 0):,}", Colors.GREEN)
    cprint(f"    Final model: {selector.get_current_model()}", Colors.GREEN)
    cprint(f"    Avg quality: {selector.get_avg_quality():.2f}", Colors.GREEN)

    return result


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

def main():
    load_env()

    project_default = (
        os.environ.get("CLOUD_RUN_PROJECT")
        or os.environ.get("GCLOUD_PROJECT")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or "erosolar-llm"
    )
    firebase_default = os.environ.get("FIREBASE_PROJECT") or project_default

    parser = argparse.ArgumentParser(description="Erosolar iterative training pipeline v2")
    parser.add_argument("--no-deploy", action="store_true", help="Skip deployment (local dev only)")
    parser.add_argument("--deploy-only", action="store_true", help="Just deploy latest model")
    parser.add_argument("--train-only", action="store_true", help="Just train current round (no generation)")
    parser.add_argument("--version", type=int, default=0, help="Start from specific version")
    parser.add_argument("--batch", type=int, default=int(os.environ.get("BATCH_SIZE", 25000)),
                        help="Max records cap per round (used for record targeting or as score cap)")
    parser.add_argument("--model", default=os.environ.get("MODEL", "deepseek-reasoner"), help="Model for generation")
    parser.add_argument("--preset", default=os.environ.get("PRESET", "infini-small"), help="Model preset")
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", 10)), help="Training epochs (5-10 range, 10 default)")
    parser.add_argument("--force-epochs", action="store_true",
                        help="Force --epochs to override v0.01 defaults")
    parser.add_argument("--once", action="store_true", help="Run only one iteration")
    parser.add_argument("--cloud-run-service", default=os.environ.get("CLOUD_RUN_SERVICE", "erosolar-api"), help="Cloud Run service name")
    parser.add_argument("--cloud-run-region", default=os.environ.get("CLOUD_RUN_REGION", "us-central1"), help="Cloud Run region")
    parser.add_argument("--cloud-run-project", default=project_default, help="Cloud Run project ID")
    parser.add_argument("--cloud-run-source", default=os.environ.get("CLOUD_RUN_SOURCE", "."), help="Cloud Run deploy source directory")
    parser.add_argument("--cloud-run-image", default=os.environ.get("CLOUD_RUN_IMAGE", ""), help="Cloud Run image URI (overrides source deploy)")
    parser.add_argument("--firebase-project", default=firebase_default, help="Firebase project ID")
    parser.add_argument("--skip-frontend", action="store_true", help="Skip deploying the Angular frontend")
    parser.add_argument("--expand", action="store_true", help="Expand dynamic categories with LLM before generation")
    parser.add_argument("--stats", action="store_true", help="Show seed coverage stats and exit")
    parser.add_argument("--target-score", type=float,
                        default=float(os.environ.get("TARGET_SCORE", 0.2)),
                        help="Target master scalar per round (default: 0.2)")
    parser.add_argument("--no-score-target", dest="target_score", action="store_const", const=None,
                        help="Disable score targeting and use record counts")
    max_records_env = os.environ.get("MAX_RECORDS")
    parser.add_argument("--max-records", type=int,
                        default=int(max_records_env) if max_records_env else None,
                        help="Max records cap for score targeting (defaults to --batch)")
    min_records_env = os.environ.get("MIN_RECORDS")
    parser.add_argument("--min-records", type=int,
                        default=int(min_records_env) if min_records_env else 1000,
                        help="Minimum new records required before training (default: 1000)")
    # Attention calculations (Author: Bo Shang <bo@shang.software>)
    # Default: OFF - poor quality data gets washed out by better training data
    parser.add_argument("--perfect-attention", action="store_true", default=False,
                        help="Enable attention calculations: attention = softmax(QK^T/√d + R)V")
    parser.add_argument("--no-perfect-attention", dest="perfect_attention", action="store_false",
                        help="Disable attention calculations (default - use standard attention)")
    # Attention Calculation Data Management
    parser.add_argument("--manage-attention", action="store_true",
                        help="Have codex-mini manage attention calculations for existing data (adaptive model: mini→5.2→pro)")
    parser.add_argument("--gcs-backup", action="store_true", default=True,
                        help="Backup data to GCS during generation and attention management")
    parser.add_argument("--no-gcs-backup", dest="gcs_backup", action="store_false",
                        help="Disable GCS backup")
    # Generational upgrade mode (no target scalar)
    parser.add_argument("--generational", action="store_true",
                        help="Use generational upgrade pipeline (no master scalar target)")
    parser.add_argument("--samples-per-problem", type=int, default=5,
                        help="Samples per problem for self-consistency (generational mode)")
    parser.add_argument("--problems-per-gen", type=int, default=1000,
                        help="Problems per generation (generational mode)")
    parser.add_argument("--min-survivors", type=int, default=500,
                        help="Minimum survivors to proceed (generational mode)")
    args = parser.parse_args()

    # Stats only mode
    if args.stats:
        if DYNAMIC_CATEGORIES_AVAILABLE:
            manager = DynamicCategoryManager()
            manager.print_stats()
            print_seed_coverage()
        else:
            cprint("Dynamic categories not available", Colors.YELLOW)
        sys.exit(0)

    # Expand categories mode
    if args.expand:
        import asyncio
        asyncio.run(expand_dynamic_categories())
        if not args.deploy_only and not args.train_only and not args.manage_attention:
            sys.exit(0)

    # Attention calculations data management mode
    if args.manage_attention:
        import asyncio
        asyncio.run(manage_existing_data_attention(backup_to_gcs=args.gcs_backup))
        if not args.deploy_only and not args.train_only:
            sys.exit(0)

    # ════════════════════════════════════════════════════════════════════════════════
    # GENERATIONAL UPGRADE MODE (no target master scalar)
    # ════════════════════════════════════════════════════════════════════════════════
    if args.generational:
        import asyncio
        from generational_upgrade_pipeline import GenerationalUpgradePipeline

        cprint(f"\n{'═' * 60}", Colors.CYAN + Colors.BOLD)
        cprint("  GENERATIONAL UPGRADE PIPELINE", Colors.CYAN + Colors.BOLD)
        cprint("  Quality emerges through self-consistency - no target scalar", Colors.CYAN)
        cprint(f"{'═' * 60}\n", Colors.CYAN)

        pipeline = GenerationalUpgradePipeline(
            model=args.model,
            samples_per_problem=args.samples_per_problem,
        )

        async def run_generational_loop():
            while True:
                result = await pipeline.run_full_cycle(
                    problems_per_generation=args.problems_per_gen,
                    min_survivors=args.min_survivors,
                )

                if not result["success"]:
                    cprint(f"Generation incomplete: {result['reason']}", Colors.YELLOW)
                    if args.once:
                        return
                    cprint("Retrying in 10 seconds...", Colors.YELLOW)
                    await asyncio.sleep(10)
                    continue

                generation = result["generation"]
                ver_str = format_version(generation)

                # Train on cumulative generations
                cprint(f"\n[TRAINING] Training erosolar {ver_str}", Colors.CYAN)
                total_records = result["total_samples"]

                if not train_model(generation, ver_str, total_records, args.preset, args.epochs,
                                  perfect_attention=args.perfect_attention, force_epochs=args.force_epochs):
                    cprint("Training failed!", Colors.RED)
                    return

                set_version(generation, total_records)

                # Deploy
                if not args.no_deploy:
                    deploy(ver_str, args)

                if args.once:
                    cprint(f"\nCompleted generation {generation} (--once flag set)", Colors.GREEN)
                    return

                cprint(f"\nStarting next generation in 5 seconds...", Colors.YELLOW)
                try:
                    await asyncio.sleep(5)
                except KeyboardInterrupt:
                    cprint("\nStopped by user.", Colors.YELLOW)
                    return

        asyncio.run(run_generational_loop())
        sys.exit(0)

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")

    # Banner
    cprint(f"{Colors.BOLD}", Colors.CYAN)
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       EROSOLAR - ITERATIVE TRAINING PIPELINE v2             ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  Step 1: Generate data (codex-mini)                         ║")
    print("║  Step 2: Attention calculations (mini → 5.2 → pro)          ║")
    print("║  Step 3: Train model (reasoning consistency)                ║")
    print("║  Step 4: Deploy to Cloud Run + Firebase                     ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    cprint("", Colors.NC)

    # Deploy only mode
    if args.deploy_only:
        version = get_version()
        ver_str = format_version(version)

        if not (SCRIPT_DIR / "models" / "erosolar").exists():
            cprint("ERROR: No model found. Train first.", Colors.RED)
            sys.exit(1)

        deploy(ver_str, args)
        sys.exit(0)

    # Get starting version
    current_version = get_version()
    if args.version > 0:
        current_version = args.version - 1

    score_mode = args.target_score is not None
    base_batch = args.batch
    score_max_records = args.max_records

    cprint(f"Starting from version: {format_version(current_version)}", Colors.YELLOW)
    if score_mode:
        cprint(f"Target master scalar: {args.target_score:.4f}", Colors.YELLOW)
    else:
        cprint(f"Records per round: {args.batch:,}", Colors.YELLOW)
    cprint(f"Model: {args.model}", Colors.YELLOW)

    # Attention calculations status
    if args.perfect_attention and PERFECT_ATTENTION_AVAILABLE:
        cprint(f"\n{Colors.CYAN}{'─' * 60}{Colors.NC}", Colors.CYAN)
        cprint(f"PERFECT SELF-ATTENTION (Author: Bo Shang <bo@shang.software>)", Colors.CYAN)
        cprint(f"{'─' * 60}", Colors.CYAN)
        cprint(f"  Standard: attention = softmax(QK^T/√d)V", Colors.GREEN)
        cprint(f"  Perfect:  attention = softmax(QK^T/√d + R)V", Colors.GREEN)
        cprint(f"  R = reasoning prior (biases attention to similar reasoning)", Colors.GREEN)
        cprint(f"{'─' * 60}", Colors.CYAN)

        # Show adaptive model selector status
        cprint(f"\n{Colors.CYAN}{'─' * 60}{Colors.NC}", Colors.CYAN)
        cprint(f"ADAPTIVE MODEL SELECTION (auto-escalates/de-escalates)", Colors.CYAN)
        cprint(f"{'─' * 60}", Colors.CYAN)
        selector = get_adaptive_selector()
        cprint(f"  Using deepseek-reasoner for all generation", Colors.GREEN)
        cprint(f"  Current: {selector.get_current_model()} (level {selector.current_level})", Colors.GREEN)
        cprint(f"{'─' * 60}", Colors.CYAN)
    elif args.perfect_attention and not PERFECT_ATTENTION_AVAILABLE:
        cprint(f"Note: Attention calculations not available (import failed)", Colors.YELLOW)

    print()

    # Main training loop
    while True:
        next_version = current_version + 1
        ver_str = format_version(next_version)

        # Get adaptive batch size for this version
        # v0.01 uses smaller data with master scalar gating
        adaptive_batch = get_adaptive_batch_size(next_version, base_batch, auto_optimize=args.perfect_attention)

        cprint(f"\n{'═' * 60}", Colors.CYAN + Colors.BOLD)
        cprint(f"  ITERATION: erosolar {ver_str}", Colors.CYAN + Colors.BOLD)
        if score_mode:
            cprint(f"  SCORE-TARGETED GENERATION", Colors.CYAN + Colors.BOLD)
            cprint(f"  Target master scalar: {args.target_score:.4f}", Colors.CYAN + Colors.BOLD)
        elif next_version == 1 and args.perfect_attention:
            cprint(f"  MASTER SCALAR-GATED DATA SIZE (no other optimization targets)", Colors.CYAN + Colors.BOLD)
            cprint(f"  Initial data: {adaptive_batch:,} records (focus: attention quality)", Colors.CYAN + Colors.BOLD)
        else:
            cprint(f"  New round: {adaptive_batch:,} records", Colors.CYAN + Colors.BOLD)
        cprint(f"{'═' * 60}", Colors.CYAN + Colors.BOLD)

        # Print starting master scalar snapshot before generation
        if score_mode:
            cprint(f"\n[MASTER SCALAR] Starting snapshot (required before generation)", Colors.CYAN)
            print_master_scalar_optimizer(args.target_score)

        # ════════════════════════════════════════════════════════════════
        # SELF-GENERATION: Use trained model for data generation (version >= 2)
        # This closes the loop: trained model generates its own training data
        # ════════════════════════════════════════════════════════════════
        use_self_gen = False
        if SELF_GENERATION_AVAILABLE and should_use_self_generation(next_version) and not args.train_only:
            cprint(f"\n[SELF-GENERATION] Version {next_version} >= 2: Using self-generation", Colors.CYAN)
            cprint(f"  Model generates its own training data (closed-loop improvement)", Colors.GREEN)
            use_self_gen = True

        # Step 1: Generate data for this round (skip if --train-only)
        if not args.train_only:
            if use_self_gen:
                cprint(f"\n[STEP 1/4] Self-generating round {next_version} data", Colors.CYAN)
                try:
                    import asyncio
                    model_path = SCRIPT_DIR / "models" / "erosolar"
                    result = asyncio.run(run_self_generation_round(
                        model_path=model_path,
                        target_samples=adaptive_batch
                    ))
                    round_count = result.get("samples_generated", 0)
                    new_records = round_count
                    cprint(f"  ✓ Self-generated {round_count} verified samples", Colors.GREEN)
                    cprint(f"  Verification rate: {result.get('verification_rate', 0):.1%}", Colors.GREEN)
                except Exception as e:
                    cprint(f"  Self-generation failed: {e}", Colors.YELLOW)
                    cprint(f"  Falling back to external model generation...", Colors.YELLOW)
                    use_self_gen = False

            if not use_self_gen:
                cprint(f"\n[STEP 1/4] Generating round {next_version} data", Colors.CYAN)
                round_count, new_records = generate_round_data(
                    round_num=next_version,
                    target=adaptive_batch,
                    model=args.model,
                    target_score=args.target_score if score_mode else None,
                    max_records=score_max_records if score_mode else None,
                    min_records=args.min_records,
                )
        else:
            # Check existing round data
            round_file = ROUNDS_DIR / f"round_{next_version:02d}.jsonl"
            if round_file.exists():
                with open(round_file) as f:
                    round_count = sum(1 for _ in f)
                new_records = 0
                cprint(f"\n[STEP 1/4] Using existing round {next_version}: {round_count:,} records", Colors.CYAN)
            else:
                cprint(f"\n[STEP 1/4] ERROR: Round {next_version} not found!", Colors.RED)
                sys.exit(1)

        # ════════════════════════════════════════════════════════════════
        # AUTO SELF-ATTENTION: Analyze generated data with mini
        # ════════════════════════════════════════════════════════════════
        attention_config = None
        if args.perfect_attention and PERFECT_ATTENTION_AVAILABLE:
            cprint(f"\n[STEP 2/4] Attention calculations analysis (codex-mini reviews new data)", Colors.CYAN)
            try:
                import asyncio
                selector = get_adaptive_selector()
                result = asyncio.run(manage_existing_data_attention(backup_to_gcs=args.gcs_backup))
                if result:
                    attention_config = result.get("optimal_attention_config", {})
                    cprint(f"  ✓ Attention optimized by {selector.get_current_model()}", Colors.GREEN)
                    cprint(f"  Recommended flow: {attention_config.get('recommended_flow', 'sequential')}", Colors.GREEN)
                    cprint(f"  Cross-step: {attention_config.get('cross_step_attention', 0.7):.2f}", Colors.GREEN)
            except Exception as e:
                cprint(f"  Attention calculations warning: {e}", Colors.YELLOW)

        cprint(f"\n[MASTER SCALAR] Optimization snapshot (required before training)", Colors.CYAN)
        current_master = print_master_scalar_optimizer(args.target_score if score_mode else None)
        if score_mode and current_master < args.target_score:
            cprint(f"  Master scalar {current_master:.4f} below target {args.target_score:.4f}.", Colors.YELLOW)
            cprint("  Continuing generation; training is blocked until target is reached.", Colors.YELLOW)
            if args.train_only:
                sys.exit(1)
            if args.once:
                cprint("  --once set; exiting without training.", Colors.YELLOW)
                break
            continue

        if score_mode and new_records < args.min_records:
            cprint(f"  Only {new_records:,} new records added; minimum is {args.min_records:,}.", Colors.YELLOW)
            cprint("  Skipping training until minimum new records are reached.", Colors.YELLOW)
            if args.train_only:
                sys.exit(1)
            if args.once:
                cprint("  --once set; exiting without training.", Colors.YELLOW)
                break
            continue

        # Step 3: Combine all rounds and train
        cprint(f"\n[STEP 3/4] Training erosolar {ver_str}", Colors.CYAN)
        cprint(f"  Combining rounds 1-{next_version} for cumulative training...", Colors.YELLOW)
        total_records = combine_rounds_for_training(next_version)

        if not train_model(next_version, ver_str, total_records, args.preset, args.epochs,
                          perfect_attention=args.perfect_attention, force_epochs=args.force_epochs):
            cprint("Training failed!", Colors.RED)
            sys.exit(1)

        cprint(f"  ✓ Model trained: erosolar {ver_str}", Colors.GREEN)

        # ════════════════════════════════════════════════════════════════
        # AUTO SELF-ATTENTION: Post-training analysis
        # ════════════════════════════════════════════════════════════════
        if args.perfect_attention and PERFECT_ATTENTION_AVAILABLE:
            cprint(f"\n[AUTO-ATTENTION] Post-training verification...", Colors.CYAN)
            try:
                selector = get_adaptive_selector()
                stats = selector.get_stats()
                cprint(f"  Final model: {stats['current_model']}", Colors.GREEN)
                cprint(f"  Avg quality: {stats['avg_quality']:.2f}", Colors.GREEN)
                cprint(f"  Level: {stats['level']}/2 (0=mini, 1=5.2, 2=pro)", Colors.GREEN)
            except Exception as e:
                cprint(f"  Post-training stats warning: {e}", Colors.YELLOW)

        # ════════════════════════════════════════════════════════════════
        # BENCHMARK TRACKING: Run benchmarks and quality gate
        # ════════════════════════════════════════════════════════════════
        if BENCHMARKS_AVAILABLE:
            cprint(f"\n[BENCHMARKS] Running post-training evaluation...", Colors.CYAN)
            try:
                model_path = SCRIPT_DIR / "models" / "erosolar"
                benchmark_result = run_post_training_benchmarks(model_path, ver_str)

                cprint(f"  Math: {benchmark_result.scores.get('math', 0):.1%}", Colors.GREEN)
                cprint(f"  Code: {benchmark_result.scores.get('code', 0):.1%}", Colors.GREEN)
                cprint(f"  Reasoning: {benchmark_result.scores.get('reasoning', 0):.1%}", Colors.GREEN)
                cprint(f"  Knowledge: {benchmark_result.scores.get('knowledge', 0):.1%}", Colors.GREEN)
                cprint(f"  Overall: {benchmark_result.overall_score:.1%}", Colors.GREEN)

                # Quality gate check (only after version 2)
                if next_version >= 2:
                    should_accept, reason = should_accept_training_round(ver_str)
                    if should_accept:
                        cprint(f"  ✓ Quality gate PASSED: {reason}", Colors.GREEN)
                    else:
                        cprint(f"  ✗ Quality gate FAILED: {reason}", Colors.RED)
                        cprint(f"  Rolling back to previous version...", Colors.YELLOW)
                        # Don't update version, don't deploy
                        if args.once:
                            cprint("  --once set; exiting without deploying.", Colors.YELLOW)
                            break
                        continue
            except Exception as e:
                cprint(f"  Benchmark warning: {e}", Colors.YELLOW)

        # ════════════════════════════════════════════════════════════════
        # LOSER/FRIEND OPTIMIZATION: Improve data coherence with local embeddings
        # ════════════════════════════════════════════════════════════════
        if LOSER_FRIEND_AVAILABLE and next_version >= 2:
            cprint(f"\n[LOSER/FRIEND] Running coherence optimization...", Colors.CYAN)
            try:
                optimizer = CoTSelfAttentionOptimizer()
                samples = optimizer.load_training_data()
                losers = optimizer.find_losers(threshold=0.3)

                if losers:
                    cprint(f"  Found {len(losers)} losers (samples with low coherence)", Colors.YELLOW)
                    cprint(f"  Consider running: python mini_cot_optimizer.py --friends", Colors.YELLOW)
                else:
                    cprint(f"  ✓ No losers found - training data is well clustered", Colors.GREEN)

                # Compute master scalar using local embeddings
                master, metrics = optimizer.compute_master_scalar()
                cprint(f"  Master scalar (local embeddings): {master:.4f}", Colors.GREEN)
            except Exception as e:
                cprint(f"  Loser/friend warning: {e}", Colors.YELLOW)

        # Show seed coverage after training
        print_seed_coverage()

        # Update version
        set_version(next_version, total_records)
        current_version = next_version

        # Step 4: Deploy to Cloud Run + Firebase (Angular calls the Cloud Run API)
        if not args.no_deploy:
            cprint(f"\n[STEP 4/4] Deploying erosolar {ver_str}", Colors.CYAN)
            deploy(ver_str, args)

        # Exit if --once flag
        if args.once:
            cprint(f"\nCompleted {ver_str} (--once flag set, exiting)", Colors.GREEN)
            break

        # Brief pause before next iteration
        cprint(f"\nCompleted {ver_str}. Starting next iteration in 5 seconds...", Colors.YELLOW)
        cprint("(Press Ctrl+C to stop)", Colors.YELLOW)
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            cprint("\nStopped by user.", Colors.YELLOW)
            break


if __name__ == "__main__":
    main()
