"""Agent runtime — a model-agnostic, multi-step, long-horizon agent loop.

The fine-tuned weights are ~30% of "agentic"; this package is the other ~70%:
  - agent.py    : plan -> act (tool calls) -> observe -> reflect -> repeat
  - planner.py  : task decomposition + replanning
  - memory.py   : working transcript + durable notes + scratchpad (survive compaction)
  - context.py  : token-budget management via transcript summarization (long-horizon)
  - tools.py    : tool registry exposed to the model as OpenAI function-calling specs
  - llm.py      : OpenAI-compatible client (points at vLLM / any chat endpoint)

It talks to ANY OpenAI-compatible chat endpoint, so you can develop against a
local vLLM (or any model) before the fine-tuned Qwen3-32B is ready.
"""
