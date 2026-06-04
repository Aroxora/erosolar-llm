"""The agent loop: plan -> act (tool calls) -> observe -> reflect -> compact -> repeat.

Talks to any OpenAI-compatible endpoint (vLLM serving the fine-tuned Qwen3, or
any chat model for development). Long-horizon behavior comes from: durable notes
that survive compaction, periodic self-reflection, and transcript summarization
when the context budget is exceeded.
"""

from __future__ import annotations

from .config import AgentConfig
from .context import maybe_compact
from .llm import LLMClient, message_to_dict, parse_tool_calls
from .memory import WorkingMemory, memory_tools
from .planner import make_plan, reflect
from .prompts import AGENT_SYSTEM
from .tools import ToolRegistry
from .tools_builtin import build_default_registry
from .types import RunResult, Step, tool_msg, user_msg


class Agent:
    def __init__(self, cfg: AgentConfig = None, llm: LLMClient = None, registry: ToolRegistry = None):
        self.cfg = cfg or AgentConfig()
        self.llm = llm or LLMClient(self.cfg)
        self.base_registry = registry or build_default_registry(self.cfg)

    def _registry_for_run(self, memory: WorkingMemory) -> ToolRegistry:
        reg = ToolRegistry()
        for t in self.base_registry._tools.values():
            reg.register(t)
        for t in memory_tools(memory):
            reg.register(t)
        return reg

    def _log(self, msg: str) -> None:
        if self.cfg.verbose:
            print(f"[agent] {msg}", flush=True)

    def run(self, task: str) -> RunResult:
        cfg = self.cfg
        mem = WorkingMemory(AGENT_SYSTEM)
        reg = self._registry_for_run(mem)
        allow = cfg.tool_allowlist or None
        steps: list = []

        if cfg.enable_planning:
            mem.plan = make_plan(self.llm, task)
            steps.append(Step(0, "plan", output=mem.plan_text()))
            self._log("plan:\n" + mem.plan_text())

        mem.add(user_msg(task))

        final_answer = ""
        stop = "max_steps"

        for i in range(1, cfg.max_steps + 1):
            if maybe_compact(mem, self.llm, cfg):
                steps.append(Step(i, "compact", output="(transcript summarized)"))
                self._log("compacted transcript")

            msg = self.llm.chat(mem.messages(), tools=reg.openai_tools(allow))
            mem.add(message_to_dict(msg))
            calls = parse_tool_calls(msg)

            if not calls:
                content = (msg.content or "").strip()
                if content:
                    final_answer, stop = content, "model_final"
                    steps.append(Step(i, "final", output=final_answer))
                    break
                mem.add(user_msg("Empty response. Take an action with a tool, "
                                 "or call `finish` with the complete answer."))
                steps.append(Step(i, "think", output="(empty; nudged)"))
                continue

            terminal_hit = False
            for call_id, name, args in calls:
                result = reg.run(name, args)
                mem.add(tool_msg(call_id, result.render(), name))
                steps.append(Step(i, "tool", tool=name, args=args, output=result.render()))
                self._log(f"{name}({args}) -> {result.render()[:160]}")
                if reg.is_terminal(name) and result.ok:
                    final_answer = args.get("answer") or result.content
                    terminal_hit = True

            if terminal_hit:
                stop = "finish"
                break

            if cfg.reflect_every and i % cfg.reflect_every == 0:
                crit = reflect(self.llm, task, mem)
                mem.add(user_msg(f"[self-reflection]\n{crit}"))
                steps.append(Step(i, "reflect", output=crit))
                self._log("reflect:\n" + crit)

        return RunResult(
            task=task,
            answer=final_answer,
            success=stop in ("finish", "model_final"),
            steps=steps,
            stop_reason=stop,
        )
