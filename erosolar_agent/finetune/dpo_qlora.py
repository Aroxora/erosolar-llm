#!/usr/bin/env python3
"""QLoRA DPO (preference optimization) continuing from the SFT adapter.

    python -m erosolar_agent.finetune.dpo_qlora --config configs/qwen3-32b-dpo.yaml

Loads the SFT LoRA (cfg.init_adapter), keeps training that same adapter so the
policy starts from the SFT model, and uses the adapter-disabled base as the
implicit reference (standard, memory-frugal LoRA-DPO). Expects preference rows
{"prompt": [...messages...], "chosen": "<text>", "rejected": "<text>"} from
prepare_data.py --stage dpo.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Unsloth first; PatchDPOTrainer patches TRL's DPOTrainer for speed/memory.
from unsloth import FastLanguageModel, PatchDPOTrainer  # noqa: E402

PatchDPOTrainer()

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from trl import DPOConfig, DPOTrainer  # noqa: E402

from erosolar_agent.finetune.config import TrainConfig  # noqa: E402
from erosolar_agent.finetune._compat import make_config, make_trainer  # noqa: E402


def _log(msg: str) -> None:
    print(f"[dpo] {msg}", flush=True)


def build_model(cfg: TrainConfig):
    src = cfg.init_adapter or cfg.base_model
    _log(f"loading policy from {src!r}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=src,
        max_seq_length=cfg.max_seq_length,
        dtype=None,
        load_in_4bit=cfg.load_in_4bit,
        token=os.environ.get("HF_TOKEN"),
    )
    if cfg.init_adapter:
        # continue training the loaded SFT adapter
        FastLanguageModel.for_training(model)
    else:
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora_r,
            target_modules=cfg.target_modules,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            use_gradient_checkpointing=cfg.use_gradient_checkpointing,
            random_state=cfg.seed,
            use_rslora=cfg.use_rslora,
        )

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_trainable == 0:
        _log("no trainable params after load — force-enabling LoRA grads")
        for name, p in model.named_parameters():
            if "lora_" in name.lower():
                p.requires_grad_(True)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_trainable == 0:
        raise RuntimeError("DPO policy has no trainable LoRA params. Aborting.")
    _log(f"trainable params: {n_trainable:,}")
    return model, tokenizer


def build_dataset(cfg: TrainConfig, tokenizer):
    ds = load_dataset("json", data_files=cfg.dataset_path, split="train")

    def to_text(ex):
        prompt_text = tokenizer.apply_chat_template(
            ex["prompt"], tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt_text, "chosen": ex["chosen"], "rejected": ex["rejected"]}

    ds = ds.map(to_text, remove_columns=ds.column_names)
    _log(f"preference pairs: {len(ds)}")
    return ds


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)

    cfg = TrainConfig.from_yaml(args.config)
    if cfg.stage != "dpo":
        _log(f"WARNING: config stage is {cfg.stage!r}, expected 'dpo'")

    model, tokenizer = build_model(cfg)
    train_ds = build_dataset(cfg, tokenizer)

    dpo_cfg, _ = make_config(
        DPOConfig,
        length_value=cfg.max_seq_length,   # max_length (TRL>=1.0) or max_seq_length (older)
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        beta=cfg.dpo_beta,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        seed=cfg.seed,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        report_to=cfg.report_to,
        run_name=cfg.run_name,
    )

    trainer = make_trainer(
        DPOTrainer, tokenizer=tokenizer,   # processing_class (TRL>=1.0) or tokenizer (older)
        model=model, ref_model=None, args=dpo_cfg, train_dataset=train_ds,
    )

    _log("starting DPO…")
    stats = trainer.train()
    _log(f"done. metrics={stats.metrics}")

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    _log(f"saved DPO adapter -> {cfg.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
