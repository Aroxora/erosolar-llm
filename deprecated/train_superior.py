#!/usr/bin/env python3
"""
Superior Model Training Script
Trains a model using generated training data to exceed GPT-5.1-codex-mini capabilities.
"""

import os
import json
import yaml
import argparse
from pathlib import Path
from datetime import datetime

def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)

def prepare_dataset(train_file: str, eval_split: float = 0.05):
    """Load and prepare the training dataset."""
    from datasets import Dataset
    import random

    data = []
    with open(train_file) as f:
        for line in f:
            try:
                item = json.loads(line)
                if "messages" in item:
                    # Convert to text format for training with special tokens
                    messages = item["messages"]
                    if len(messages) >= 2:
                        text = f"<|user|>\n{messages[0]['content']}\n<|end_turn|>\n<|assistant|>\n{messages[1]['content']}\n<|end_turn|>"
                        data.append({
                            "text": text,
                            "domain": item.get("domain", "general"),
                            "complexity": item.get("complexity", 5),
                            "quality_score": item.get("quality_score", 0.5)
                        })
            except json.JSONDecodeError:
                continue

    random.shuffle(data)

    # Split into train and eval
    split_idx = int(len(data) * (1 - eval_split))
    train_data = data[:split_idx]
    eval_data = data[split_idx:]

    return Dataset.from_list(train_data), Dataset.from_list(eval_data)

def train(config: dict):
    """Main training function."""
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    import torch

    print("=" * 60)
    print("SUPERIOR MODEL TRAINING")
    print("=" * 60)

    # Load model
    print(f"\n[1/5] Loading base model: {config['model']['name']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model"]["name"],
        max_seq_length=config["model"]["max_seq_length"],
        load_in_4bit=config["model"].get("load_in_4bit", True),
        dtype=None,
    )

    # Apply LoRA
    print(f"\n[2/5] Applying LoRA (rank={config['lora']['rank']})")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora"]["rank"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Load dataset
    print(f"\n[3/5] Loading training data: {config['data']['train_file']}")
    train_dataset, eval_dataset = prepare_dataset(
        config["data"]["train_file"],
        config["data"].get("eval_split", 0.05)
    )
    print(f"       Training examples: {len(train_dataset)}")
    print(f"       Evaluation examples: {len(eval_dataset)}")

    # Training arguments
    print("\n[4/5] Configuring training")
    training_args = TrainingArguments(
        output_dir=config["output"]["dir"],
        num_train_epochs=config["training"]["epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=float(config["training"]["learning_rate"]),
        warmup_ratio=config["training"]["warmup_ratio"],
        weight_decay=config["training"]["weight_decay"],
        lr_scheduler_type=config["training"]["lr_scheduler"],
        max_grad_norm=config["training"]["max_grad_norm"],
        fp16=config["training"].get("fp16", True),
        logging_steps=config["output"]["logging_steps"],
        save_steps=config["output"]["save_steps"],
        eval_strategy="steps",
        eval_steps=config["output"]["eval_steps"],
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=config["model"]["max_seq_length"],
        args=training_args,
    )

    # Train
    print("\n[5/5] Starting training...")
    print("=" * 60)

    trainer.train()

    # Save final model
    print("\nSaving model...")
    trainer.save_model(config["output"]["dir"])
    tokenizer.save_pretrained(config["output"]["dir"])

    # Save training info
    info = {
        "base_model": config["model"]["name"],
        "output_name": config["model"]["output_name"],
        "training_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
        "epochs": config["training"]["epochs"],
        "completed_at": datetime.now().isoformat()
    }

    with open(Path(config["output"]["dir"]) / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Model saved to: {config['output']['dir']}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Train superior model")
    parser.add_argument("--config", type=str, required=True, help="Training config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
