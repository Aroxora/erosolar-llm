"""erosolar_agent — additive agentic stack.

This package is built *alongside* the existing erosolar char-model pipeline
(honest_pipeline.py, model.py, serve.py, registry.py) without modifying or
replacing it. It provides:

  - finetune/  : QLoRA SFT + DPO of an open base (Qwen3) on Lambda H100s
  - runtime/   : a multi-step agent loop (plan -> act -> observe -> reflect),
                 persistent memory, tools, and long-context compaction
  - serving/   : a vLLM / HF backend behind the existing OpenAI-compatible API
  - eval/      : capability + agentic evaluation

Nothing here touches the legacy MiniGPT training/serving paths.
"""

__version__ = "0.1.0"
__all__ = ["secrets"]
