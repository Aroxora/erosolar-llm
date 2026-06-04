#!/usr/bin/env python3
"""Build (and optionally launch) a vLLM OpenAI server for an hf-vllm model.

    # print the launch command for a registered model:
    python -m erosolar_agent.serving.vllm_server --name erosolar-qwen3-32b
    # explicit repo/path, and actually launch it:
    python -m erosolar_agent.serving.vllm_server --model <hf-or-path> --launch

Tool calling (required by the agent loop) needs --enable-auto-tool-choice and a
parser; Qwen3 uses the 'hermes' parser. Serve the merged 16-bit weights, or pass
--quantization bitsandbytes/awq to fit a smaller/cheaper GPU.
"""

from __future__ import annotations

import argparse
import os
import sys


def build_command(model, port=8000, max_model_len=8192, gpu_util=0.90,
                  quantization=None, tool_parser="hermes", served_name=None) -> list:
    cmd = [
        "vllm", "serve", model,
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_util),
        "--enable-auto-tool-choice", "--tool-call-parser", tool_parser,
    ]
    if served_name:
        cmd += ["--served-model-name", served_name]
    if quantization:
        cmd += ["--quantization", quantization]
    return cmd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--name", help="registered model name (resolves hf_model_id)")
    g.add_argument("--model", help="explicit HF repo id or local path")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--quantization", default=None, help="bitsandbytes | awq | (omit for 16-bit)")
    ap.add_argument("--tool-parser", default="hermes")
    ap.add_argument("--served-name", default=None)
    ap.add_argument("--launch", action="store_true", help="exec vllm now (else just print)")
    args = ap.parse_args(argv)

    model, served = args.model, args.served_name
    if args.name:
        from .registry_ext import resolve_source
        model = resolve_source(args.name)
        served = served or args.name

    cmd = build_command(model, args.port, args.max_model_len, args.gpu_util,
                        args.quantization, args.tool_parser, served)
    print(" ".join(cmd))
    if args.launch:
        os.execvp(cmd[0], cmd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
