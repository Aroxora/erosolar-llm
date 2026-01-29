#!/usr/bin/env python3
"""
Merge a LoRA adapter into a base model for evaluation/benchmarking.

NOTE: This is a UTILITY SCRIPT for benchmark comparisons against external models.
The main proprietary model in this repo is trained FROM SCRATCH - not a LoRA/fine-tune.

This script is only used when comparing our from-scratch model against
LoRA fine-tunes of existing models for benchmark purposes.

Example:
  python merge_lora.py \
    --base-model unsloth/Qwen2.5-0.5B \
    --adapter-dir cache/bench/model \
    --output-dir cache/bench/merged_model
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base-model", type=str, required=True, help="Base model name or path")
    parser.add_argument("--adapter-dir", type=str, required=True, help="LoRA adapter directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for merged model")
    parser.add_argument("--dtype", type=str, default="float16",
                        help="Model dtype: float16 or bfloat16")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model = model.merge_and_unload()
    model.save_pretrained(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Merged model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
