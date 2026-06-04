#!/usr/bin/env python3
"""QLoRA supervised fine-tuning of Qwen3 with Unsloth + TRL on a single H100.

    python -m erosolar_agent.finetune.sft_qlora --config configs/qwen3-32b-sft.yaml

Reads the unified chat JSONL from prepare_data.py, renders it with the model's
own chat template, and (by default) trains only on assistant completions
(prompt tokens masked). Saves the LoRA adapter to cfg.output_dir.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Unsloth must be imported before transformers/trl so its kernels patch in.
from unsloth import FastLanguageModel  # noqa: E402
from unsloth.chat_templates import train_on_responses_only  # noqa: E402

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

from erosolar_agent.finetune.config import TrainConfig  # noqa: E402
from erosolar_agent.finetune._compat import make_config, make_trainer  # noqa: E402


def _log(msg: str) -> None:
    print(f"[sft] {msg}", flush=True)


def build_model(cfg: TrainConfig):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model,
        max_seq_length=cfg.max_seq_length,
        dtype=None,            # auto: bf16 on H100
        load_in_4bit=cfg.load_in_4bit,
        token=os.environ.get("HF_TOKEN"),
    )
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
    return model, tokenizer


def build_dataset(cfg: TrainConfig, tokenizer):
    ds = load_dataset("json", data_files=cfg.dataset_path, split="train")

    def render(batch):
        texts = [
            tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            for msgs in batch["messages"]
        ]
        return {"text": texts}

    ds = ds.map(render, batched=True, remove_columns=ds.column_names)
    _log(f"dataset rows: {len(ds)}")
    return ds


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)

    cfg = TrainConfig.from_yaml(args.config)
    if cfg.stage != "sft":
        _log(f"WARNING: config stage is {cfg.stage!r}, expected 'sft'")
    _log(f"base={cfg.base_model}  effective_batch={cfg.effective_batch_size()}  "
         f"lr={cfg.learning_rate}  max_seq={cfg.max_seq_length}")

    model, tokenizer = build_model(cfg)
    train_ds = build_dataset(cfg, tokenizer)

    sft_cfg, _ = make_config(
        SFTConfig,
        length_value=cfg.max_seq_length,   # max_length (TRL>=1.0) or max_seq_length (older)
        output_dir=cfg.output_dir,
        dataset_text_field="text",
        packing=False,  # keep off so response-only masking is exact
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_ratio=cfg.warmup_ratio,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        seed=cfg.seed,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",   # 8-bit optimizer state — matters at 32B
        report_to=cfg.report_to,
        run_name=cfg.run_name,
    )

    trainer = make_trainer(
        SFTTrainer, tokenizer=tokenizer,   # processing_class (TRL>=1.0) or tokenizer (older)
        model=model, train_dataset=train_ds, args=sft_cfg,
    )

    if cfg.train_on_responses_only:
        # Qwen3 uses ChatML turn markers; learn only on assistant spans.
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )

    _smoke_check_grad(trainer)
    _smoke_check_labels(trainer)

    _log("starting training…")
    stats = trainer.train()
    _log(f"done. metrics={stats.metrics}")

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    _log(f"saved LoRA adapter -> {cfg.output_dir}")
    return 0


def _smoke_check_grad(trainer) -> None:
    """Some Unsloth/TRL version combos have shipped silent zero-grad bugs. A few
    hundred dollars is too much to find out at hour 40 — verify trainable params
    exist before the long run."""
    n_trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    _log(f"trainable params: {n_trainable:,}")
    if n_trainable == 0:
        raise RuntimeError("No trainable parameters — LoRA did not attach. Aborting.")


def _smoke_check_labels(trainer) -> None:
    """Best-effort guard: confirm response-only masking left SOMETHING to learn.
    Catches all-(-100) labels caused by over-aggressive truncation/masking before
    a multi-hour run produces zero loss."""
    try:
        ds = trainer.train_dataset
        sample = [ds[i] for i in range(min(2, len(ds)))]
        batch = trainer.data_collator(sample)
        labels = batch.get("labels") if isinstance(batch, dict) else getattr(batch, "labels", None)
        if labels is not None:
            unmasked = int((labels != -100).sum())
            _log(f"sample unmasked label tokens: {unmasked}")
            if unmasked == 0:
                raise RuntimeError(
                    "All labels masked (-100): response-only masking + truncation left "
                    "nothing to train on. Check max_length vs your data lengths."
                )
    except RuntimeError:
        raise
    except Exception as e:  # noqa: BLE001 — best-effort only
        _log(f"(label smoke-check skipped: {type(e).__name__}: {e})")


if __name__ == "__main__":
    sys.exit(main())
