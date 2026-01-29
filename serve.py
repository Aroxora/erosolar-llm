#!/usr/bin/env python3
"""
Erosolar Model Inference Server

OpenAI Response API compatible with SSE streaming for thinking and answer.
Session storage via client-side Firestore (frontend handles persistence).

Run locally: python serve.py
"""

import os
import json
import time
import uuid
import threading
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from typing import Generator, Optional, Dict, Any, List
from pathlib import Path

from model import MiniGPT, InfiniGPT, ModelConfig
from tokenizer import BPETokenizer
from registry import get_registry, load_model, list_models

app = Flask(__name__)
CORS(app)

_model_cache = {}
_inference_lock = threading.Lock()


def extract_text_content(content: Any) -> str:
    """Extract plain text from Responses API-style content."""
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type in ("input_text", "output_text", "text", "summary_text"):
                text = part.get("text")
                if text:
                    parts.append(text)
        return "".join(parts)
    if isinstance(content, str):
        return content
    return ""


def format_prompt_for_generation(prompt: str, tokenizer: BPETokenizer, force_thought_tokens: bool = True) -> str:
    """Format prompt with chat tokens for generation."""
    prompt_text = prompt.strip()
    if "<|user|>" in prompt_text or "<|assistant|>" in prompt_text:
        formatted = prompt_text
        if not formatted.rstrip().endswith("<|assistant|>"):
            formatted = f"{formatted}\n<|assistant|>"
        formatted += "\n"
    else:
        formatted = f"<|user|>\n{prompt}\n<|end_turn|>\n<|assistant|>\n"
    if force_thought_tokens:
        formatted += f"{tokenizer.special_tokens.think_start}\n"
    return formatted


def count_input_tokens(formatted_prompt: str, tokenizer: BPETokenizer) -> int:
    """Count input tokens including BOS for usage reporting."""
    return 1 + len(tokenizer.encode(formatted_prompt, add_special=False))


def normalize_messages(input_data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Normalize input messages into role/content pairs."""
    messages = []
    for msg in input_data:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        content = extract_text_content(msg.get("content", ""))
        if not content:
            continue
        if role not in ("assistant", "user"):
            role = "user"
        messages.append({"role": role, "content": content})
    return messages


def build_prompt_from_messages(messages: List[Dict[str, str]]) -> str:
    """Format a multi-turn prompt using the model's chat tokens."""
    parts = []
    last_role = None
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if not content:
            continue
        if role == "assistant":
            parts.append(f"<|assistant|>\n{content}\n<|end_turn|>\n")
            last_role = "assistant"
        else:
            parts.append(f"<|user|>\n{content}\n<|end_turn|>\n")
            last_role = "user"
    if last_role != "assistant":
        parts.append("<|assistant|>\n")
    return "".join(parts)


def get_device():
    """Get compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_cached(model_name: str = "erosolar"):
    """Load model from registry with caching."""
    if model_name in _model_cache:
        return _model_cache[model_name]

    device = get_device()
    try:
        model, tokenizer, config, info = load_model(model_name, device)
        _model_cache[model_name] = (model, tokenizer, config, info, device)
        print(f"Loaded model: {model_name} ({info.params:,} params)")
        return _model_cache[model_name]
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        raise


@torch.no_grad()
def generate_stream_tokens(
    model, tokenizer, prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.3,
    top_k: int = 20,
    top_p: float = 0.85,
    device=None,
    max_seq_len: int = 256,
    force_thought_tokens: bool = True
) -> Generator[Dict[str, Any], None, None]:
    """Generate tokens with events for thinking and content."""

    prompt_text = prompt.strip()
    if "<|user|>" in prompt_text or "<|assistant|>" in prompt_text:
        formatted = prompt_text
        if not formatted.endswith("<|assistant|>"):
            formatted = f"{formatted}\n<|assistant|>"
        formatted += "\n"
    else:
        formatted = f"<|user|>\n{prompt}\n<|end_turn|>\n<|assistant|>\n"
    if force_thought_tokens:
        formatted += f"{tokenizer.special_tokens.think_start}\n"

    input_ids = [tokenizer.bos_token_id]
    input_ids.extend(tokenizer.encode(formatted, add_special=False))
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    unk_token_id = tokenizer.token_to_id.get("<|unk|>", 1)
    repetition_penalty = 1.3  # Reduced from 2.0 - gentler for v0.01 model

    think_start_id = tokenizer.token_to_id.get("<|think_start|>", -1)
    think_end_id = tokenizer.token_to_id.get("<|think_end|>", -1)
    step_id = tokenizer.token_to_id.get("<|step|>", -1)
    answer_id = tokenizer.token_to_id.get("<|answer|>", -1)
    end_turn_id = tokenizer.token_to_id.get("<|end_turn|>", -1)
    user_id = tokenizer.token_to_id.get("<|user|>", -1)

    in_thinking = force_thought_tokens
    generated_text = ""
    last_token_needs_space = False  # Track spacing between tokens

    if force_thought_tokens:
        yield {"type": "thinking_start", "content": ""}

    for _ in range(max_tokens):
        idx = input_ids[:, -max_seq_len:]
        with _inference_lock:
            logits = model(idx)[:, -1, :]

        logits[0, unk_token_id] = float('-inf')

        recent_tokens = set(input_ids[0].tolist()[-80:])
        for token_id in recent_tokens:
            if logits[0, token_id] > 0:
                logits[0, token_id] /= repetition_penalty
            else:
                logits[0, token_id] *= repetition_penalty

        logits = logits / temperature

        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumsum = torch.cumsum(probs, dim=-1)
            sorted_mask = cumsum > top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = 0
            indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
            logits[indices_to_remove] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        next_token_id = next_token.item()

        if next_token_id == tokenizer.eos_token_id:
            break

        input_ids = torch.cat([input_ids, next_token], dim=1)
        token_text = tokenizer.id_to_token.get(next_token_id, '')

        if token_text == "<|unk|>":
            continue

        if next_token_id == think_start_id:
            in_thinking = True
            last_token_needs_space = False
            yield {"type": "thinking_start", "content": ""}
            continue
        elif next_token_id == think_end_id:
            in_thinking = False
            last_token_needs_space = False
            yield {"type": "thinking_end", "content": ""}
            continue
        elif next_token_id == step_id:
            last_token_needs_space = False
            yield {"type": "thinking" if in_thinking else "content", "content": "\n"}
            continue
        elif next_token_id == answer_id:
            if in_thinking:
                in_thinking = False
                last_token_needs_space = False
                yield {"type": "thinking_end", "content": ""}
            last_token_needs_space = False
            yield {"type": "answer_start", "content": ""}
            continue
        elif next_token_id == end_turn_id or next_token_id == user_id:
            break

        # Smart spacing: add space before alphanumeric tokens only if needed
        if token_text:
            needs_space = (
                last_token_needs_space and
                token_text[0].isalnum() and
                generated_text and
                not generated_text.endswith((' ', '\n', '\t'))
            )
            if needs_space:
                token_text = ' ' + token_text
            # Track if next token needs space (after alphanumeric or closing punct)
            last_token_needs_space = token_text[-1].isalnum() if token_text else False

        generated_text += token_text

        event_type = "thinking" if in_thinking else "content"
        yield {"type": event_type, "content": token_text}

        if "<|end_turn|>" in generated_text or "<|user|>" in generated_text:
            break

    yield {"type": "done", "content": ""}


def format_sse(event_type: str, data: dict) -> str:
    """Format as SSE with event type."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


@app.route('/v1/responses', methods=['POST', 'OPTIONS'])
def responses():
    """OpenAI Response API endpoint with SSE streaming."""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.get_json(silent=True) or {}

        # Extract input (can be string or messages array)
        input_data = data.get('input', '')
        prompt = ""
        if isinstance(input_data, list):
            messages = normalize_messages(input_data)
            prompt = build_prompt_from_messages(messages)
        elif isinstance(input_data, str):
            prompt = input_data

        if not prompt:
            return jsonify({'error': {'message': 'input is required'}}), 400

        model_name = data.get('model', 'erosolar')
        max_tokens = min(data.get('max_output_tokens', 200), 500)
        temperature = max(0.1, min(data.get('temperature', 0.3), 2.0))
        stream = data.get('stream', False)
        top_p = data.get('top_p', 0.85)

        model, tokenizer, config, info, device = load_model_cached(model_name)

        response_id = f"resp_{uuid.uuid4().hex[:24]}"

        formatted_prompt = format_prompt_for_generation(prompt, tokenizer, True)
        input_tokens = count_input_tokens(formatted_prompt, tokenizer)

        if stream:
            def generate():
                # Response created event
                yield format_sse("response.created", {
                    "type": "response.created",
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "status": "in_progress",
                        "model": model_name
                    }
                })

                thinking_item_id = f"item_{uuid.uuid4().hex[:8]}"
                output_item_id = f"item_{uuid.uuid4().hex[:8]}"
                thinking_started = False
                output_started = False
                in_thinking = True
                output_tokens = 0

                for event in generate_stream_tokens(
                    model, tokenizer, prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    device=device,
                    max_seq_len=info.max_seq_len
                ):
                    event_type = event["type"]
                    content = event["content"]

                    if event_type in ("thinking", "content"):
                        output_tokens += 1

                    if event_type == "thinking_start":
                        in_thinking = True
                        if not thinking_started:
                            thinking_started = True
                            yield format_sse("response.output_item.added", {
                                "type": "response.output_item.added",
                                "output_index": 0,
                                "item": {
                                    "id": thinking_item_id,
                                    "type": "reasoning",
                                    "status": "in_progress"
                                }
                            })

                    elif event_type == "thinking":
                        yield format_sse("response.output_text.delta", {
                            "type": "response.output_text.delta",
                            "item_id": thinking_item_id,
                            "output_index": 0,
                            "content_index": 0,
                            "delta": content
                        })

                    elif event_type == "thinking_end":
                        in_thinking = False
                        if thinking_started:
                            yield format_sse("response.output_item.done", {
                                "type": "response.output_item.done",
                                "output_index": 0,
                                "item": {
                                    "id": thinking_item_id,
                                    "type": "reasoning",
                                    "status": "completed"
                                }
                            })

                    elif event_type == "content":
                        if not output_started:
                            output_started = True
                            yield format_sse("response.output_item.added", {
                                "type": "response.output_item.added",
                                "output_index": 1 if thinking_started else 0,
                                "item": {
                                    "id": output_item_id,
                                    "type": "message",
                                    "role": "assistant",
                                    "status": "in_progress"
                                }
                            })
                            yield format_sse("response.content_part.added", {
                                "type": "response.content_part.added",
                                "item_id": output_item_id,
                                "output_index": 1 if thinking_started else 0,
                                "content_index": 0,
                                "part": {"type": "text", "text": ""}
                            })

                        yield format_sse("response.output_text.delta", {
                            "type": "response.output_text.delta",
                            "item_id": output_item_id,
                            "output_index": 1 if thinking_started else 0,
                            "content_index": 0,
                            "delta": content
                        })

                    elif event_type == "done":
                        if output_started:
                            yield format_sse("response.output_item.done", {
                                "type": "response.output_item.done",
                                "output_index": 1 if thinking_started else 0,
                                "item": {
                                    "id": output_item_id,
                                    "type": "message",
                                    "role": "assistant",
                                    "status": "completed"
                                }
                            })

                        yield format_sse("response.completed", {
                            "type": "response.completed",
                            "response": {
                                "id": response_id,
                                "object": "response",
                                "status": "completed",
                                "model": model_name,
                                "usage": {
                                    "input_tokens": input_tokens,
                                    "output_tokens": output_tokens,
                                    "total_tokens": input_tokens + output_tokens
                                }
                            }
                        })

            return Response(
                generate(),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'X-Accel-Buffering': 'no'
                }
            )
        else:
            # Non-streaming
            thinking_content = []
            answer_content = []
            output_tokens = 0

            for event in generate_stream_tokens(
                model, tokenizer, prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                device=device,
                max_seq_len=info.max_seq_len
            ):
                if event["type"] in ("thinking", "content"):
                    output_tokens += 1
                if event["type"] == "thinking":
                    thinking_content.append(event["content"])
                elif event["type"] == "content":
                    answer_content.append(event["content"])

            full_thinking = ''.join(thinking_content).strip()
            full_content = ''.join(answer_content).strip()

            output = []
            if full_thinking:
                output.append({
                    "type": "reasoning",
                    "id": f"item_{uuid.uuid4().hex[:8]}",
                    "summary": [{"type": "summary_text", "text": full_thinking}]
                })

            output.append({
                "type": "message",
                "id": f"item_{uuid.uuid4().hex[:8]}",
                "role": "assistant",
                "content": [{"type": "output_text", "text": full_content}]
            })

            return jsonify({
                "id": response_id,
                "object": "response",
                "created_at": int(time.time()),
                "status": "completed",
                "model": model_name,
                "output": output,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                }
            })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': {'message': str(e)}}), 500


@app.route('/v1/models', methods=['GET'])
def v1_models():
    """List models with full info."""
    try:
        model_list = list_models()
        data = []
        for m in model_list:
            model_data = {
                "id": m.name,
                "object": "model",
                "owned_by": "erosolar",
                "params": m.params,
                "max_seq_len": m.max_seq_len
            }
            # Get version from version.json if available
            version_file = Path("data_store/version.json")
            if version_file.exists():
                with open(version_file) as f:
                    version_info = json.load(f)
                    model_data["version"] = version_info.get("version_string", "v0.01")
                    model_data["total_records"] = version_info.get("total_records", 0)
            data.append(model_data)
        return jsonify({"object": "list", "data": data})
    except Exception as e:
        return jsonify({'error': {'message': str(e)}}), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'version': '2.0'})


@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'service': 'Erosolar Inference API',
        'version': '2.0',
        'endpoints': {
            '/v1/responses': 'POST - OpenAI Response API (stream:true supported)',
            '/v1/models': 'GET - List models',
            '/api/health': 'GET - Health check'
        }
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print("Pre-loading model...")
    try:
        load_model_cached('erosolar')
        print("Ready!")
    except Exception as e:
        print(f"Warning: {e}")

    print(f"\nServer on port {port}")
    print(f"  POST /v1/responses - Response API with SSE streaming")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
