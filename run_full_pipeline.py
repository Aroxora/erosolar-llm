#!/usr/bin/env python3
"""
Full Training Pipeline Orchestrator
Runs the complete pipeline: generate data -> validate -> train -> evaluate
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


class PipelineOrchestrator:
    """Orchestrates the full training pipeline."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = Path(config.get("output_dir", "pipeline_output"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / "pipeline.log"
        self.results = {"stages": {}, "started_at": datetime.now().isoformat()}

    def log(self, message: str):
        """Log message to console and file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        print(log_line)
        with open(self.log_file, "a") as f:
            f.write(log_line + "\n")

    def run_command(self, cmd: list, stage: str) -> bool:
        """Run a command and capture output."""
        self.log(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).parent)
            )

            if result.returncode == 0:
                self.log(f"Stage {stage} completed successfully")
                self.results["stages"][stage] = {
                    "status": "success",
                    "output": result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout
                }
                return True
            else:
                self.log(f"Stage {stage} failed: {result.stderr}")
                self.results["stages"][stage] = {
                    "status": "failed",
                    "error": result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr
                }
                return False

        except Exception as e:
            self.log(f"Stage {stage} error: {str(e)}")
            self.results["stages"][stage] = {"status": "error", "error": str(e)}
            return False

    def stage_generate_data(self) -> bool:
        """Stage 1: Generate training data using GPT-5.1-codex-mini."""
        self.log("=" * 50)
        self.log("STAGE 1: Generate Training Data")
        self.log("=" * 50)

        cmd = [
            sys.executable, "generate_superior_training.py",
            "--count", str(self.config.get("data_count", 500)),
            "--min-quality", str(self.config.get("min_quality", 0.65)),
            "--max-concurrent", str(self.config.get("max_concurrent", 10)),
            "--output-dir", str(self.output_dir / "training_data")
        ]

        return self.run_command(cmd, "generate_data")

    def stage_generate_adversarial(self) -> bool:
        """Stage 2: Generate adversarial training examples."""
        self.log("=" * 50)
        self.log("STAGE 2: Generate Adversarial Data")
        self.log("=" * 50)

        cmd = [
            sys.executable, "generate_adversarial_training.py",
            "--count", str(self.config.get("adversarial_count", 100)),
            "--output-dir", str(self.output_dir / "training_data")
        ]

        return self.run_command(cmd, "generate_adversarial")

    def stage_merge_data(self) -> bool:
        """Stage 3: Merge and deduplicate training data."""
        self.log("=" * 50)
        self.log("STAGE 3: Merge Training Data")
        self.log("=" * 50)

        data_dir = self.output_dir / "training_data"
        merged_file = data_dir / "merged_training.jsonl"

        try:
            seen_ids = set()
            merged_count = 0

            with open(merged_file, "w") as out:
                for jsonl_file in data_dir.glob("*.jsonl"):
                    if jsonl_file.name == "merged_training.jsonl":
                        continue
                    with open(jsonl_file) as f:
                        for line in f:
                            data = json.loads(line)
                            data_id = data.get("id", "")
                            if data_id not in seen_ids:
                                seen_ids.add(data_id)
                                out.write(line)
                                merged_count += 1

            self.log(f"Merged {merged_count} unique training examples")
            self.results["stages"]["merge_data"] = {
                "status": "success",
                "examples": merged_count,
                "output_file": str(merged_file)
            }
            return True

        except Exception as e:
            self.log(f"Merge failed: {e}")
            self.results["stages"]["merge_data"] = {"status": "failed", "error": str(e)}
            return False

    def stage_train_model(self) -> bool:
        """Stage 4: Train the model."""
        self.log("=" * 50)
        self.log("STAGE 4: Train Model")
        self.log("=" * 50)

        # Create training config
        config_file = self.output_dir / "training_config.yaml"

        import yaml
        train_config = {
            "model": {
                "name": self.config.get("base_model", "unsloth/Qwen2.5-7B"),
                "output_name": self.config.get("model_name", "erosolar-superior"),
                "max_seq_length": 4096,
                "load_in_4bit": True
            },
            "lora": {
                "rank": self.config.get("lora_rank", 64),
                "alpha": 128,
                "dropout": 0.05,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj"]
            },
            "training": {
                "epochs": self.config.get("epochs", 3),
                "batch_size": self.config.get("batch_size", 4),
                "gradient_accumulation_steps": 4,
                "learning_rate": self.config.get("learning_rate", "2e-5"),
                "warmup_ratio": 0.1,
                "weight_decay": 0.01,
                "lr_scheduler": "cosine",
                "max_grad_norm": 1.0,
                "fp16": True
            },
            "data": {
                "train_file": str(self.output_dir / "training_data" / "merged_training.jsonl"),
                "eval_split": 0.05
            },
            "output": {
                "dir": str(self.output_dir / "model"),
                "save_steps": 500,
                "eval_steps": 500,
                "logging_steps": 10
            }
        }

        with open(config_file, "w") as f:
            yaml.dump(train_config, f)

        cmd = [
            sys.executable, "train_superior.py",
            "--config", str(config_file)
        ]

        return self.run_command(cmd, "train_model")

    def stage_evaluate(self) -> bool:
        """Stage 5: Evaluate the trained model."""
        self.log("=" * 50)
        self.log("STAGE 5: Evaluate Model")
        self.log("=" * 50)

        cmd = [
            sys.executable, "evaluate_model.py",
            "--model-path", str(self.output_dir / "model"),
            "--benchmarks", self.config.get("benchmarks", "mmlu,gsm8k,arc"),
            "--samples", str(self.config.get("eval_samples", 100)),
            "--output", str(self.output_dir / "eval_results.json")
        ]

        return self.run_command(cmd, "evaluate")

    def run_pipeline(self, stages: list = None) -> bool:
        """Run the full pipeline or selected stages."""

        all_stages = [
            ("generate_data", self.stage_generate_data),
            ("generate_adversarial", self.stage_generate_adversarial),
            ("merge_data", self.stage_merge_data),
            ("train_model", self.stage_train_model),
            ("evaluate", self.stage_evaluate)
        ]

        if stages:
            all_stages = [(name, func) for name, func in all_stages if name in stages]

        self.log("=" * 60)
        self.log("SUPERIOR MODEL TRAINING PIPELINE")
        self.log("=" * 60)
        self.log(f"Stages to run: {[s[0] for s in all_stages]}")
        self.log(f"Output directory: {self.output_dir}")

        success = True
        for stage_name, stage_func in all_stages:
            if not stage_func():
                self.log(f"Pipeline stopped at stage: {stage_name}")
                success = False
                break

        # Save final results
        self.results["completed_at"] = datetime.now().isoformat()
        self.results["success"] = success

        with open(self.output_dir / "pipeline_results.json", "w") as f:
            json.dump(self.results, f, indent=2)

        self.log("=" * 60)
        self.log(f"PIPELINE {'COMPLETED' if success else 'FAILED'}")
        self.log("=" * 60)

        return success


def main():
    parser = argparse.ArgumentParser(description="Run full training pipeline")
    parser.add_argument("--config", type=str, help="Pipeline config JSON file")
    parser.add_argument("--data-count", type=int, default=500)
    parser.add_argument("--adversarial-count", type=int, default=100)
    parser.add_argument("--base-model", type=str, default="unsloth/Qwen2.5-7B")
    parser.add_argument("--model-name", type=str, default="erosolar-superior")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="pipeline_output")
    parser.add_argument("--stages", type=str, default=None,
                       help="Comma-separated stages to run (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be run without executing")
    args = parser.parse_args()

    # Build config
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    else:
        config = {
            "data_count": args.data_count,
            "adversarial_count": args.adversarial_count,
            "base_model": args.base_model,
            "model_name": args.model_name,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lora_rank": args.lora_rank,
            "output_dir": args.output_dir,
            "min_quality": 0.65,
            "max_concurrent": 10
        }

    if args.dry_run:
        print("DRY RUN - Pipeline configuration:")
        print(json.dumps(config, indent=2))
        return

    # Run pipeline
    orchestrator = PipelineOrchestrator(config)
    stages = args.stages.split(",") if args.stages else None
    success = orchestrator.run_pipeline(stages)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
