"""Shared config schema for the QLoRA SFT/DPO scripts.

One dataclass drives both stages so the YAML configs in configs/ stay
consistent with what sft_qlora.py and dpo_qlora.py actually read. Unknown keys
in a YAML file raise immediately (typos shouldn't silently no-op on a paid GPU).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class TrainConfig:
    # stage
    stage: str = "sft"  # "sft" | "dpo"

    # model
    base_model: str = "unsloth/Qwen3-8B"  # the POST-TRAINED (instruct) Qwen3, not -Base
    load_in_4bit: bool = True
    max_seq_length: int = 4096

    # LoRA — "LoRA Without Regret": adapters on ALL linear layers (incl. MLP),
    # rank big enough to hold the data, and ~10x the full-FT learning rate.
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: list = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",  # attention
            "gate_proj", "up_proj", "down_proj",      # MLP — do NOT omit
        ]
    )
    use_gradient_checkpointing: str = "unsloth"
    use_rslora: bool = False

    # DPO continues from the SFT adapter; point this at the SFT output_dir.
    init_adapter: str = ""

    # data (produced by prepare_data.py)
    dataset_path: str = "data/sft_blend.train.jsonl"
    eval_path: str = ""
    train_on_responses_only: bool = True  # mask the prompt; learn only completions

    # training
    output_dir: str = "outputs/run"
    num_train_epochs: float = 2.0
    max_steps: int = -1  # >0 overrides epochs (handy for cheap proto/smoke runs)
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 3
    seed: int = 3407

    # DPO-specific
    dpo_beta: float = 0.1

    # telemetry
    report_to: str = "none"  # set "wandb" with WANDB_API_KEY in .env
    run_name: str = "erosolar-run"

    @classmethod
    def from_yaml(cls, path) -> "TrainConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"Unknown config keys in {path}: {sorted(unknown)}. "
                f"Valid keys: {sorted(known)}"
            )
        return cls(**data)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps
