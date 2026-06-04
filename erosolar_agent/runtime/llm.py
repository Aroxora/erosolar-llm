"""OpenAI-compatible chat client (points at vLLM or any chat endpoint)."""

from __future__ import annotations

import json

from .config import AgentConfig


class LLMClient:
    def __init__(self, cfg: AgentConfig):
        from openai import OpenAI  # lazy import so the package loads without openai

        self.cfg = cfg
        self._client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

    def chat(self, messages, tools=None, tool_choice="auto", temperature=None, max_tokens=None):
        kw = dict(
            model=self.cfg.model,
            messages=messages,
            temperature=self.cfg.temperature if temperature is None else temperature,
            top_p=self.cfg.top_p,
            max_tokens=self.cfg.max_tokens if max_tokens is None else max_tokens,
        )
        if tools:
            kw["tools"] = tools
            kw["tool_choice"] = tool_choice
        resp = self._client.chat.completions.create(**kw)
        return resp.choices[0].message

    def complete(self, system: str, prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
        """Plain text completion (no tools) — used by planner/reflector/summarizer."""
        resp = self._client.chat.completions.create(
            model=self.cfg.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=self.cfg.top_p,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""


def message_to_dict(msg) -> dict:
    """Clean an OpenAI response message into a transcript-safe assistant dict
    (drops provider-specific extras like reasoning_content that can break replay)."""
    d = {"role": "assistant", "content": msg.content or ""}
    tcs = getattr(msg, "tool_calls", None)
    if tcs:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"},
            }
            for tc in tcs
        ]
    return d


def parse_tool_calls(msg) -> list:
    """Return [(call_id, name, args_dict), ...] from a response message."""
    out = []
    for tc in getattr(msg, "tool_calls", None) or []:
        try:
            args = json.loads(tc.function.arguments or "{}")
            if not isinstance(args, dict):
                args = {"value": args}
        except json.JSONDecodeError:
            args = {"__raw__": tc.function.arguments}
        out.append((tc.id, tc.function.name, args))
    return out
