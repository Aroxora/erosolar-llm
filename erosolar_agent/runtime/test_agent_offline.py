"""Offline tests for the agent loop — no model server or openai package needed.

A scripted FakeLLM stands in for the real endpoint so we can exercise the loop,
tool execution, the terminal `finish` tool, memory, and context compaction
deterministically.

    python -m erosolar_agent.runtime.test_agent_offline   # or: pytest
"""

from __future__ import annotations

from .agent import Agent
from .config import AgentConfig
from .context import estimate_tokens, maybe_compact
from .memory import WorkingMemory
from .tools_builtin import build_default_registry
from .types import user_msg


class _Fn:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _TC:
    def __init__(self, cid, name, arguments):
        self.id = cid
        self.type = "function"
        self.function = _Fn(name, arguments)


class _Msg:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class FakeLLM:
    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.i = 0

    def chat(self, messages, tools=None, **kw):
        msg = self.scripted[min(self.i, len(self.scripted) - 1)]
        self.i += 1
        return msg

    def complete(self, system, prompt, **kw):
        return "subgoal one\nsubgoal two"


def test_tool_then_finish():
    cfg = AgentConfig(enable_planning=True, reflect_every=0, verbose=False)
    llm = FakeLLM([
        _Msg(tool_calls=[_TC("c1", "calculator", '{"expression": "2*(3+4)"}')]),
        _Msg(tool_calls=[_TC("c2", "finish", '{"answer": "14"}')]),
    ])
    agent = Agent(cfg, llm=llm, registry=build_default_registry(cfg))
    res = agent.run("compute 2*(3+4) and finish")
    assert res.success, res.stop_reason
    assert res.answer == "14", res.answer
    tool_steps = [s for s in res.steps if s.kind == "tool"]
    assert any(s.tool == "calculator" and s.output == "14" for s in tool_steps), tool_steps
    assert res.steps[0].kind == "plan"


def test_model_final_without_tool():
    cfg = AgentConfig(enable_planning=False, reflect_every=0, verbose=False)
    llm = FakeLLM([_Msg(content="The answer is blue.")])
    agent = Agent(cfg, llm=llm, registry=build_default_registry(cfg))
    res = agent.run("what color?")
    assert res.success and res.answer == "The answer is blue."
    assert res.stop_reason == "model_final"


def test_compaction():
    mem = WorkingMemory("sys")
    for k in range(20):
        mem.add(user_msg(f"message number {k} " * 20))
    cfg = AgentConfig(context_token_budget=50, keep_recent_messages=4, verbose=False)
    assert estimate_tokens(mem.messages()) > cfg.context_token_budget
    did = maybe_compact(mem, FakeLLM([]), cfg)
    assert did is True
    assert mem.transcript[0]["content"].startswith("[Earlier steps")
    assert len(mem.transcript) == 5  # summary + 4 recent


def test_calculator_rejects_code():
    reg = build_default_registry(AgentConfig(verbose=False))
    bad = reg.run("calculator", {"expression": "__import__('os').system('echo hi')"})
    assert not bad.ok
    good = reg.run("calculator", {"expression": "10 % 3 + 2**3"})  # 1 + 8 = 9
    assert good.ok and good.content == "9", good.content


def _main():
    fns = [test_tool_then_finish, test_model_final_without_tool,
           test_compaction, test_calculator_rejects_code]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print("ALL OFFLINE AGENT TESTS PASSED")


if __name__ == "__main__":
    _main()
