"""Drive the agent from the command line.

    python -m erosolar_agent.runtime.cli "Find the population of Tokyo and Paris and compute the ratio." --trace

Points at OPENAI_BASE_URL/.env by default (a local vLLM). Override with --base-url/--model.
"""

from __future__ import annotations

import argparse
import sys

from .agent import Agent
from .config import AgentConfig


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task", help="the task for the agent to solve")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--no-plan", action="store_true", help="disable the planning step")
    ap.add_argument("--python", action="store_true", help="enable the run_python tool")
    ap.add_argument("--quiet", action="store_true", help="suppress per-step logging")
    ap.add_argument("--trace", action="store_true", help="print the full step trace")
    args = ap.parse_args(argv)

    kw = {}
    if args.model:
        kw["model"] = args.model
    if args.base_url:
        kw["base_url"] = args.base_url
    if args.max_steps:
        kw["max_steps"] = args.max_steps
    if args.no_plan:
        kw["enable_planning"] = False
    if args.python:
        kw["enable_python_tool"] = True
    if args.quiet:
        kw["verbose"] = False

    agent = Agent(AgentConfig(**kw))
    result = agent.run(args.task)

    print("\n" + ("=" * 60))
    if args.trace:
        print(result.trace_text())
    else:
        print(result.answer)
    print("=" * 60)
    print(f"[{result.stop_reason} after {result.n_steps} steps, success={result.success}]")
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
