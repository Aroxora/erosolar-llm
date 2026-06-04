#!/usr/bin/env python3
"""OpenAI-compatible server that runs the AGENT LOOP on top of a vLLM endpoint.

    # 1) start vLLM (raw model) on :8000  (see vllm_server.py)
    # 2) start this agent server on :8080
    VLLM_BASE_URL=http://localhost:8000/v1 VLLM_MODEL=<served-name> \
        python -m erosolar_agent.serving.agent_server

Exposes:
  POST /v1/chat/completions  - OpenAI chat-completions (runs the agent, returns final answer)
  POST /v1/responses         - OpenAI Responses API shape (matches the legacy serve.py;
                               reasoning summary = agent trace, message = final answer)
  GET  /api/health

The legacy MiniGPT serve.py is left untouched; this is a separate process/port.
"""

from __future__ import annotations

import json
import os
import time
import uuid

from flask import Flask, Response, jsonify, request

try:
    from flask_cors import CORS
except Exception:  # pragma: no cover - optional
    def CORS(app, **kw):
        return app

from erosolar_agent.runtime.agent import Agent
from erosolar_agent.runtime.config import AgentConfig

app = Flask(__name__)
CORS(app)

_AGENT = None


def get_agent() -> Agent:
    global _AGENT
    if _AGENT is None:
        cfg = AgentConfig(
            model=os.environ.get("VLLM_MODEL", "erosolar-qwen3-32b"),
            base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
            max_steps=int(os.environ.get("AGENT_MAX_STEPS", "24")),
            verbose=False,
        )
        _AGENT = Agent(cfg)
    return _AGENT


def _text(content) -> str:
    if isinstance(content, list):
        return "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return content if isinstance(content, str) else ""


def _task_from_messages(messages: list) -> str:
    msgs = [m for m in messages if _text(m.get("content"))]
    if not msgs:
        return ""
    history = msgs[:-1]
    task = _text(msgs[-1].get("content"))
    if history:
        convo = "\n".join(f"{m.get('role', 'user')}: {_text(m.get('content'))}" for m in history)
        task = f"Conversation so far:\n{convo}\n\nCurrent request:\n{task}"
    return task


def _reasoning_text(result) -> str:
    lines = []
    for s in result.steps:
        if s.kind == "plan":
            lines.append("Plan:\n" + s.output)
        elif s.kind == "tool":
            lines.append(f"• {s.tool}({', '.join(f'{k}={v!r}' for k, v in s.args.items())}) "
                         f"→ {s.output[:200]}")
        elif s.kind == "reflect":
            lines.append("Reflect: " + s.output[:200])
    return "\n".join(lines) or "(direct answer)"


@app.route("/v1/chat/completions", methods=["POST", "OPTIONS"])
def chat_completions():
    if request.method == "OPTIONS":
        return "", 204
    data = request.get_json(silent=True) or {}
    messages = data.get("messages", [])
    task = _task_from_messages(messages)
    if not task:
        return jsonify({"error": {"message": "no user message"}}), 400
    try:
        result = get_agent().run(task)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": {"message": str(e)}}), 500
    return jsonify({
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": data.get("model", get_agent().cfg.model),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result.answer},
            "finish_reason": "stop" if result.success else "length",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "erosolar_trace": {"steps": result.n_steps, "stop_reason": result.stop_reason},
    })


def _sse(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


@app.route("/v1/responses", methods=["POST", "OPTIONS"])
def responses():
    if request.method == "OPTIONS":
        return "", 204
    data = request.get_json(silent=True) or {}
    inp = data.get("input", "")
    if isinstance(inp, list):
        task = _task_from_messages(inp)
    else:
        task = str(inp)
    if not task:
        return jsonify({"error": {"message": "input is required"}}), 400

    model_name = data.get("model", get_agent().cfg.model)
    stream = bool(data.get("stream", False))
    try:
        result = get_agent().run(task)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": {"message": str(e)}}), 500

    reasoning = _reasoning_text(result)
    answer = result.answer
    resp_id = f"resp_{uuid.uuid4().hex[:24]}"
    # Only surface a 'reasoning' item when the agent actually took steps — otherwise
    # the frontend renders a spurious "thinking" bubble on direct answers.
    has_reasoning = bool(result.steps)

    if not stream:
        output = []
        if has_reasoning:
            output.append({"type": "reasoning", "id": f"item_{uuid.uuid4().hex[:8]}",
                           "summary": [{"type": "summary_text", "text": reasoning}]})
        output.append({"type": "message", "id": f"item_{uuid.uuid4().hex[:8]}",
                       "role": "assistant", "content": [{"type": "output_text", "text": answer}]})
        return jsonify({
            "id": resp_id, "object": "response", "created_at": int(time.time()),
            "status": "completed", "model": model_name, "output": output,
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        })

    def generate():
        message_id = f"item_{uuid.uuid4().hex[:8]}"
        msg_index = 0
        yield _sse("response.created", {"type": "response.created",
                   "response": {"id": resp_id, "status": "in_progress", "model": model_name}})
        if has_reasoning:
            reasoning_id = f"item_{uuid.uuid4().hex[:8]}"
            yield _sse("response.output_item.added", {"type": "response.output_item.added",
                       "output_index": 0, "item": {"id": reasoning_id, "type": "reasoning",
                       "status": "in_progress"}})
            yield _sse("response.output_text.delta", {"type": "response.output_text.delta",
                       "item_id": reasoning_id, "output_index": 0, "content_index": 0, "delta": reasoning})
            yield _sse("response.output_item.done", {"type": "response.output_item.done",
                       "output_index": 0, "item": {"id": reasoning_id, "type": "reasoning",
                       "status": "completed"}})
            msg_index = 1
        yield _sse("response.output_item.added", {"type": "response.output_item.added",
                   "output_index": msg_index, "item": {"id": message_id, "type": "message",
                   "role": "assistant", "status": "in_progress"}})
        yield _sse("response.output_text.delta", {"type": "response.output_text.delta",
                   "item_id": message_id, "output_index": msg_index, "content_index": 0, "delta": answer})
        yield _sse("response.output_item.done", {"type": "response.output_item.done",
                   "output_index": msg_index, "item": {"id": message_id, "type": "message",
                   "role": "assistant", "status": "completed"}})
        yield _sse("response.completed", {"type": "response.completed",
                   "response": {"id": resp_id, "status": "completed", "model": model_name}})

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "erosolar-agent-server",
                    "model": get_agent().cfg.model, "vllm": get_agent().cfg.base_url})


@app.route("/", methods=["GET"])
def root():
    return jsonify({"service": "erosolar agent server",
                    "endpoints": ["/v1/chat/completions", "/v1/responses", "/api/health"]})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"erosolar agent server on :{port} -> vLLM {os.environ.get('VLLM_BASE_URL', 'http://localhost:8000/v1')}")
    app.run(host="0.0.0.0", port=port, threaded=True)
