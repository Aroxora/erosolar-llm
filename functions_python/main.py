"""
Erosolar Model Inference - Python Cloud Function

Serves the trained erosolar model via HTTP endpoints.
Supports:
- SSE streaming for real-time generation (like claude.ai)
- Conversation history with user/assistant tokens
- Tavily search integration for web search

Author: Bo Shang <bo@shang.software>
"""

import os
import json
import torch
import torch.nn.functional as F
import httpx
from flask import jsonify, Response
from firebase_functions import https_fn, options
from firebase_admin import initialize_app, storage
import tempfile
from typing import List, Dict, Generator, Optional

# Initialize Firebase
initialize_app()

# Tavily API for web search
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "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")
TAVILY_SEARCH_URL = "https://api.tavily.com/search"

# Global model cache
_model_cache = {}


# ============================================================================
# TAVILY SEARCH
# ============================================================================

async def tavily_search(query: str, max_results: int = 5) -> Dict:
    """
    Search the web using Tavily API.

    Returns search results that can be injected into model context.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                TAVILY_SEARCH_URL,
                json={
                    "api_key": TAVILY_API_KEY,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": max_results,
                    "include_answer": True,
                    "include_raw_content": False
                },
                timeout=10.0
            )
            if response.status_code == 200:
                return response.json()
            return {"error": f"Tavily returned {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def tavily_search_sync(query: str, max_results: int = 5) -> Dict:
    """Synchronous version of Tavily search."""
    try:
        with httpx.Client() as client:
            response = client.post(
                TAVILY_SEARCH_URL,
                json={
                    "api_key": TAVILY_API_KEY,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": max_results,
                    "include_answer": True,
                    "include_raw_content": False
                },
                timeout=10.0
            )
            if response.status_code == 200:
                return response.json()
            return {"error": f"Tavily returned {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def format_search_results(results: Dict) -> str:
    """Format Tavily results for model context."""
    if "error" in results:
        return f"[Search failed: {results['error']}]"

    formatted = "[Web Search Results]\n"
    if results.get("answer"):
        formatted += f"Summary: {results['answer']}\n\n"

    for i, result in enumerate(results.get("results", [])[:5], 1):
        formatted += f"{i}. {result.get('title', 'No title')}\n"
        formatted += f"   URL: {result.get('url', '')}\n"
        formatted += f"   {result.get('content', '')[:200]}...\n\n"

    return formatted


# ============================================================================
# CONVERSATION HISTORY
# ============================================================================

def format_conversation_history(messages: List[Dict]) -> str:
    """
    Format conversation history into model input.

    Messages format (same as OpenAI/Claude):
    [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "<|think_start|>...<|answer|>Hi!"},
        {"role": "user", "content": "What is Python?"}
    ]
    """
    formatted = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "user":
            formatted += f"<|user|>\n{content}\n<|end_turn|>\n"
        elif role == "assistant":
            # Keep CoT tokens intact for context
            formatted += f"<|assistant|>\n{content}\n<|end_turn|>\n"
        elif role == "system":
            formatted += f"<|system|>\n{content}\n<|end_turn|>\n"

    # Add assistant turn start for generation
    formatted += "<|assistant|>\n"
    return formatted

def load_model_from_storage(model_name: str = "erosolar"):
    """Load model from Firebase Storage."""
    if model_name in _model_cache:
        return _model_cache[model_name]

    bucket = storage.bucket()

    # Download model files to temp directory
    temp_dir = tempfile.mkdtemp()

    # Download checkpoint
    checkpoint_blob = bucket.blob(f"models/{model_name}/checkpoint/model.pt")
    checkpoint_path = os.path.join(temp_dir, "model.pt")
    checkpoint_blob.download_to_filename(checkpoint_path)

    # Download tokenizer
    tokenizer_blob = bucket.blob(f"models/{model_name}/tokenizer/tokenizer.json")
    tokenizer_path = os.path.join(temp_dir, "tokenizer.json")
    tokenizer_blob.download_to_filename(tokenizer_path)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Import model class
    from model import MiniGPT, ModelConfig

    config = ModelConfig(**checkpoint.get("config", {}))
    model = MiniGPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load tokenizer
    from tokenizer import BPETokenizer
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path.replace("tokenizer.json", ""))

    _model_cache[model_name] = (model, tokenizer, config, device)
    return _model_cache[model_name]


@torch.no_grad()
def generate_response(model, tokenizer, prompt: str, max_tokens: int = 200,
                     temperature: float = 0.3, top_k: int = 20,
                     top_p: float = 0.85, device=None):
    """Generate response from model (non-streaming)."""
    formatted = f"<|user|>\n{prompt}\n<|end_turn|>\n<|assistant|>\n"

    input_ids = [tokenizer.bos_token_id]
    input_ids.extend(tokenizer.encode(formatted, add_special=False))
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    max_seq_len = getattr(model.config, 'max_seq_len', 256)
    generated_tokens = []

    for _ in range(max_tokens):
        idx = input_ids[:, -max_seq_len:]
        logits = model(idx)[:, -1, :]

        # Temperature scaling
        logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        # Top-p (nucleus) filtering
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

        generated_tokens.append(token_text)

        # Stop at end_turn or new user turn
        text = ''.join(generated_tokens)
        if "<|end_turn|>" in text or "<|user|>" in text:
            generated_tokens = [t for t in generated_tokens if t not in ["<|end_turn|>", "<|user|>"]]
            break

    # Clean up response
    response = ' '.join(generated_tokens)
    response = response.replace('  ', ' ').strip()
    for stop_tok in ["<|end_turn|>", "<|user|>", "<|assistant|>"]:
        if stop_tok in response:
            response = response.split(stop_tok)[0].strip()

    return response


@torch.no_grad()
def generate_stream(
    model, tokenizer,
    messages: List[Dict],
    max_tokens: int = 500,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    device=None,
    search_context: str = None
) -> Generator[Dict, None, None]:
    """
    Stream generation token-by-token (OpenAI-compatible SSE format).

    Yields dicts in OpenAI streaming format:
    {"choices": [{"delta": {"content": "token"}, "index": 0}]}

    CoT tokens like <|think_start|> are streamed to enable claude.ai-style
    thinking display on the frontend.
    """
    import time

    # Format conversation history
    formatted = format_conversation_history(messages)

    # Inject search context if available
    if search_context:
        formatted = f"<|system|>\n{search_context}\n<|end_turn|>\n" + formatted

    input_ids = [tokenizer.bos_token_id]
    input_ids.extend(tokenizer.encode(formatted, add_special=False))
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    max_seq_len = getattr(model.config, 'max_seq_len', 256)
    generated_text = ""
    request_id = f"chatcmpl-{int(time.time()*1000)}"

    for i in range(max_tokens):
        idx = input_ids[:, -max_seq_len:]
        logits = model(idx)[:, -1, :]

        # Temperature scaling
        logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        # Top-p filtering
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

        generated_text += token_text

        # Stop at end_turn or new user turn
        if "<|end_turn|>" in generated_text or "<|user|>" in generated_text:
            break

        # Yield OpenAI-compatible streaming chunk
        yield {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "erosolar",
            "choices": [{
                "index": 0,
                "delta": {"content": token_text},
                "finish_reason": None
            }]
        }

    # Final chunk with finish_reason
    yield {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "erosolar",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }


@https_fn.on_request(
    cors=options.CorsOptions(cors_origins="*", cors_methods=["GET", "POST"]),
    memory=options.MemoryOption.GB_1,
    timeout_sec=60
)
def chat(req: https_fn.Request) -> https_fn.Response:
    """
    Legacy chat endpoint for model inference (non-streaming).
    """
    try:
        if req.method == "OPTIONS":
            return https_fn.Response("", status=204)

        data = req.get_json(silent=True) or {}

        prompt = data.get("prompt", "")
        if not prompt:
            return jsonify({"error": "prompt is required"}), 400

        max_tokens = min(data.get("max_tokens", 200), 500)
        temperature = max(0.1, min(data.get("temperature", 0.3), 2.0))
        model_name = data.get("model", "erosolar")

        # Load model
        model, tokenizer, config, device = load_model_from_storage(model_name)

        # Generate response
        response = generate_response(
            model, tokenizer, prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            device=device
        )

        return jsonify({
            "response": response,
            "model": model_name,
            "prompt": prompt
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@https_fn.on_request(
    cors=options.CorsOptions(cors_origins="*", cors_methods=["GET", "POST"]),
    memory=options.MemoryOption.GB_1,
    timeout_sec=120
)
def chat_completions(req: https_fn.Request) -> https_fn.Response:
    """
    OpenAI-compatible /v1/chat/completions endpoint.

    Supports:
    - Conversation history (messages array)
    - SSE streaming (stream: true)
    - CoT tokens for claude.ai-style thinking display
    - Web search via Tavily (search: true or auto-detect)

    POST /v1/chat/completions
    {
        "model": "erosolar",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "<|think_start|>...<|answer|>Hi!"},
            {"role": "user", "content": "What is Python?"}
        ],
        "max_tokens": 500,
        "temperature": 0.7,
        "stream": true,
        "search": false
    }
    """
    import time

    try:
        if req.method == "OPTIONS":
            return https_fn.Response("", status=204)

        data = req.get_json(silent=True) or {}

        messages = data.get("messages", [])
        if not messages:
            return jsonify({"error": "messages array is required"}), 400

        max_tokens = min(data.get("max_tokens", 500), 2000)
        temperature = max(0.1, min(data.get("temperature", 0.7), 2.0))
        stream = data.get("stream", False)
        search_enabled = data.get("search", False)
        model_name = data.get("model", "erosolar")

        # Load model
        model, tokenizer, config, device = load_model_from_storage(model_name)

        # Optional: Run web search for context
        search_context = None
        if search_enabled:
            last_user_msg = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    last_user_msg = msg.get("content", "")
                    break
            if last_user_msg:
                results = tavily_search_sync(last_user_msg)
                search_context = format_search_results(results)

        if stream:
            # SSE streaming response
            def generate_sse():
                for chunk in generate_stream(
                    model, tokenizer, messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    device=device,
                    search_context=search_context
                ):
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return Response(
                generate_sse(),
                mimetype="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        else:
            # Non-streaming response
            formatted = format_conversation_history(messages)
            if search_context:
                formatted = f"<|system|>\n{search_context}\n<|end_turn|>\n" + formatted

            # Get last user message for legacy generate
            last_user_msg = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    last_user_msg = msg.get("content", "")
                    break

            response_text = generate_response(
                model, tokenizer, last_user_msg,
                max_tokens=max_tokens,
                temperature=temperature,
                device=device
            )

            return jsonify({
                "id": f"chatcmpl-{int(time.time()*1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(formatted.split()),
                    "completion_tokens": len(response_text.split()),
                    "total_tokens": len(formatted.split()) + len(response_text.split())
                }
            })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@https_fn.on_request(
    cors=options.CorsOptions(cors_origins="*", cors_methods=["GET", "POST"]),
    timeout_sec=30
)
def search(req: https_fn.Request) -> https_fn.Response:
    """
    Tavily search endpoint.

    POST /search
    {
        "query": "latest Python news",
        "max_results": 5
    }
    """
    try:
        if req.method == "OPTIONS":
            return https_fn.Response("", status=204)

        data = req.get_json(silent=True) or {}
        query = data.get("query", "")

        if not query:
            return jsonify({"error": "query is required"}), 400

        max_results = min(data.get("max_results", 5), 10)
        results = tavily_search_sync(query, max_results)

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@https_fn.on_request(cors=options.CorsOptions(cors_origins="*", cors_methods=["GET"]))
def health(req: https_fn.Request) -> https_fn.Response:
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "erosolar-inference",
        "version": "1.0"
    })


@https_fn.on_request(cors=options.CorsOptions(cors_origins="*", cors_methods=["GET"]))
def models(req: https_fn.Request) -> https_fn.Response:
    """List available models."""
    try:
        bucket = storage.bucket()
        blobs = bucket.list_blobs(prefix="models/")

        model_names = set()
        for blob in blobs:
            parts = blob.name.split("/")
            if len(parts) >= 2:
                model_names.add(parts[1])

        return jsonify({
            "models": list(model_names),
            "default": "erosolar"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
