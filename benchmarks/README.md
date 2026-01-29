# Local Benchmarking Guide

This repo uses the LM Eval Harness for standardized evaluation. You can run
it entirely offline if you preload datasets into a local HuggingFace cache.

## Install (bench only)

```bash
pip install -r requirements-bench.txt
```

Or:

```bash
pip install .[bench]
```

## Offline dataset setup

LM Eval uses `datasets` under the hood. To run offline:

1) Preload the datasets into your local HuggingFace cache.
2) Set `HF_DATASETS_OFFLINE=1` and `HF_HUB_OFFLINE=1`.
3) Point `--data-dir` to your cache directory.

Example cache path:
```
~/.cache/huggingface
```

## Recommended task set (adjust as needed)

The available task names depend on your installed LM Eval version. Use:

```bash
python -m lm_eval --tasks list
```

Example task list:
```
gpqa_diamond,arc_challenge,gsm8k,mbpp,humaneval
```

## Example run

```bash
python benchmark_runner.py \
  --model hf \
  --model-args pretrained=Qwen/Qwen2.5-0.5B-Instruct,trust_remote_code=True \
  --tasks gpqa_diamond,arc_challenge,gsm8k,mbpp,humaneval \
  --device cuda:0 \
  --batch-size 4 \
  --offline \
  --data-dir ~/.cache/huggingface \
  --output-path cache/benchmarks/results.json
```

## SWE-Bench Pro

SWE-Bench Pro requires the official harness and pre-generated patch
predictions. This repo does not implement the full agent loop. Use the
official SWE-Bench tools to generate predictions, then evaluate them with
their harness locally.
