"""QLoRA fine-tuning for an open base (Qwen3) on a single Lambda H100.

Pipeline: prepare_data -> sft_qlora -> dpo_qlora -> merge_and_export.
Driven end-to-end by run_lambda.sh; see LAMBDA_LAUNCH.md for the launch checklist.
"""
