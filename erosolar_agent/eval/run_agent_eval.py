#!/usr/bin/env python3
"""Agentic eval: run the agent on a small suite of multi-step / tool-use tasks
and score the final answers with simple checkers.

    python -m erosolar_agent.eval.run_agent_eval --base-url http://localhost:8080/v1 --model erosolar-qwen3-32b

Note: this points at the AGENT SERVER's model name / or runs the agent directly
against a vLLM endpoint. It needs a live endpoint; it is not an offline test
(see erosolar_agent/runtime/test_agent_offline.py for the offline loop test).
"""

from __future__ import annotations

import argparse
import sys

from erosolar_agent.runtime.agent import Agent
from erosolar_agent.runtime.config import AgentConfig


def _contains(*subs):
    subs = [s.lower() for s in subs]
    return lambda ans: all(s in ans.lower() for s in subs)


SUITE = [
    {
        "name": "arithmetic_multistep",
        "task": "Using the calculator tool, compute 2*(3+4)**2, then call finish with just the number.",
        "check": _contains("98"),
    },
    {
        "name": "file_roundtrip",
        "task": "Write a file notes.txt in your workspace whose only content is the word BANANA, "
                "then read it back and finish with exactly its contents.",
        "check": _contains("banana"),
    },
    {
        "name": "reasoning_fibonacci",
        "task": "Compute the 10th Fibonacci number (with F(1)=1, F(2)=1) step by step, "
                "then finish with just the number.",
        "check": _contains("55"),
    },
    {
        "name": "durable_memory",
        "task": "Remember that my project codename is ORCHID. Then, in a later step, recall it and "
                "finish with the codename.",
        "check": _contains("orchid"),
    },
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--max-steps", type=int, default=16)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)

    if args.list:
        for c in SUITE:
            print(f"  {c['name']}: {c['task']}")
        return 0

    kw = {"max_steps": args.max_steps, "verbose": False, "enable_python_tool": True}
    if args.model:
        kw["model"] = args.model
    if args.base_url:
        kw["base_url"] = args.base_url
    agent = Agent(AgentConfig(**kw))

    passed = 0
    for case in SUITE:
        try:
            res = agent.run(case["task"])
            ok = bool(res.success and case["check"](res.answer))
        except Exception as e:  # noqa: BLE001
            res, ok = None, False
            print(f"  [ERROR] {case['name']}: {e}")
        passed += ok
        status = "PASS" if ok else "FAIL"
        ans = (res.answer[:80] if res else "")
        steps = (res.n_steps if res else 0)
        print(f"  [{status}] {case['name']} ({steps} steps) -> {ans!r}")

    print(f"\n{passed}/{len(SUITE)} agentic tasks passed")
    return 0 if passed == len(SUITE) else 1


if __name__ == "__main__":
    sys.exit(main())
