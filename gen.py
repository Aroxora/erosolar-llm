#!/usr/bin/env python3
"""
Simple generation script.

Usage:
    python gen.py                           # Interactive mode (uses latest model)
    python gen.py -p "your prompt"          # Single prompt
    python gen.py --model mymodel           # Specific model
    python gen.py --list                    # List models
"""

import argparse
import sys
import torch
import torch.nn.functional as F
from registry import get_registry, list_models, load_model

# Colors
G = "\033[92m"
C = "\033[96m"
Y = "\033[93m"
D = "\033[2m"
R = "\033[0m"
B = "\033[1m"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_tokens: int = 150,
             temperature: float = 0.6, top_k: int = 40, top_p: float = 0.9,
             repetition_penalty: float = 1.2, device=None) -> str:
    """Generate text from prompt with improved sampling.

    Args:
        model: The language model
        tokenizer: Tokenizer for encoding/decoding
        prompt: Input prompt text
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (lower = more deterministic)
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling threshold
        repetition_penalty: Penalty for repeating tokens (>1.0 reduces repetition)
        device: Device to run on
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Format as conversation with special tokens
    formatted = f"<|user|>\n{prompt}\n<|end_turn|>\n<|assistant|>\n"

    # Encode
    input_ids = [tokenizer.bos_token_id]
    input_ids.extend(tokenizer.encode(formatted, add_special=False))
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    max_seq_len = model.config.max_seq_len
    eos_id = tokenizer.eos_token_id
    unk_id = tokenizer.token_to_id.get("<|unk|>", 1)
    pad_id = tokenizer.token_to_id.get("<|pad|>", 0)

    generated = []

    for _ in range(max_tokens):
        idx = input_ids[:, -max_seq_len:]
        logits = model(idx)[:, -1, :]

        # Suppress special tokens
        logits[0, unk_id] = float('-inf')
        logits[0, pad_id] = float('-inf')

        # Repetition penalty - penalize recent tokens more
        seen_tokens = input_ids[0, -50:].tolist()
        for token_id in set(seen_tokens):
            count = seen_tokens.count(token_id)
            penalty = repetition_penalty ** min(count, 3)  # Cap penalty growth
            if logits[0, token_id] > 0:
                logits[0, token_id] /= penalty
            else:
                logits[0, token_id] *= penalty

        logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_mask = cumulative_probs > top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False
            indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
            logits[indices_to_remove] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        if next_token.item() == eos_id:
            break

        input_ids = torch.cat([input_ids, next_token], dim=1)

        token_text = tokenizer.id_to_token.get(next_token.item(), '')
        if token_text == "<|unk|>":
            continue

        # Add space before words
        if token_text and token_text[0].isalnum():
            token_text = ' ' + token_text

        generated.append(token_text)

        # Stop at end_turn or new user turn (prevent continuation into next turn)
        if "<|end_turn|>" in ''.join(generated) or "<|user|>" in ''.join(generated):
            break

    result = ''.join(generated).strip()
    for stop_tok in ["<|end_turn|>", "<|user|>", "<|assistant|>"]:
        if stop_tok in result:
            result = result.split(stop_tok)[0].strip()
    return result


def stream_generate(model, tokenizer, prompt: str, max_tokens: int = 150,
                    temperature: float = 0.6, top_k: int = 40, top_p: float = 0.9,
                    repetition_penalty: float = 1.2, device=None):
    """Generate and yield tokens one at a time with improved sampling."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    formatted = f"<|user|>\n{prompt}\n<|end_turn|>\n<|assistant|>\n"
    input_ids = [tokenizer.bos_token_id]
    input_ids.extend(tokenizer.encode(formatted, add_special=False))
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    max_seq_len = model.config.max_seq_len
    eos_id = tokenizer.eos_token_id
    unk_id = tokenizer.token_to_id.get("<|unk|>", 1)
    pad_id = tokenizer.token_to_id.get("<|pad|>", 0)

    text_so_far = ""

    for _ in range(max_tokens):
        idx = input_ids[:, -max_seq_len:]

        with torch.no_grad():
            logits = model(idx)[:, -1, :]

        # Suppress special tokens
        logits[0, unk_id] = float('-inf')
        logits[0, pad_id] = float('-inf')

        # Repetition penalty
        seen_tokens = input_ids[0, -50:].tolist()
        for token_id in set(seen_tokens):
            count = seen_tokens.count(token_id)
            penalty = repetition_penalty ** min(count, 3)
            if logits[0, token_id] > 0:
                logits[0, token_id] /= penalty
            else:
                logits[0, token_id] *= penalty

        logits = logits / temperature

        # Top-k
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        # Top-p
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_mask = cumulative_probs > top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False
            indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
            logits[indices_to_remove] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        if next_token.item() == eos_id:
            break

        input_ids = torch.cat([input_ids, next_token], dim=1)

        token_text = tokenizer.id_to_token.get(next_token.item(), '')
        if token_text == "<|unk|>":
            continue

        if token_text and token_text[0].isalnum():
            token_text = ' ' + token_text

        text_so_far += token_text
        if "<|end_turn|>" in text_so_far or "<|user|>" in text_so_far:
            break

        yield token_text


def interactive(model, tokenizer, device):
    """Interactive chat mode."""
    print(f"\n{B}{C}Erosolar{R}")
    print(f"{D}Type 'quit' to exit{R}\n")

    while True:
        try:
            prompt = input(f"{B}>>> {R}").strip()
            if not prompt:
                continue
            if prompt.lower() in ('quit', 'exit', 'q'):
                break

            print(f"{G}", end="", flush=True)
            for token in stream_generate(model, tokenizer, prompt, device=device):
                print(token, end="", flush=True)
            print(f"{R}\n")

        except (KeyboardInterrupt, EOFError):
            break

    print(f"\n{Y}Goodbye!{R}\n")


def interactive_hybrid(model, tokenizer, device):
    """Interactive chat mode with smart multi-strategy responder for kids model."""
    import re

    # Try to use smart responder (multiple strategies)
    try:
        from kids_smart_responder import smart_respond
        has_smart = True
    except ImportError:
        has_smart = False
        # Fallback to basic retrieval
        try:
            from kids_retrieval import find_best_answer
            has_retrieval = True
        except ImportError:
            has_retrieval = False

    print(f"\n{B}{C}Erosolar Kids{R}")
    print(f"{D}Type 'quit' to exit{R}\n")

    while True:
        try:
            prompt = input(f"{B}>>> {R}").strip()
            if not prompt:
                continue
            if prompt.lower() in ('quit', 'exit', 'q'):
                break

            # Try smart responder first (math, jokes, retrieval, topics, templates)
            if has_smart:
                result = smart_respond(prompt)
                if result:
                    print(f"{G}{result}{R}\n")
                    continue

            # Fallback to basic retrieval
            if not has_smart and has_retrieval:
                result, score, _ = find_best_answer(prompt, threshold=0.40)
                if result and score >= 0.40:
                    result = re.sub(r'\[Generation:.*?\]', '', result).strip()
                    print(f"{G}{result}{R}\n")
                    continue

            # Last resort: neural generation
            print(f"{G}", end="", flush=True)
            for token in stream_generate(model, tokenizer, prompt, device=device):
                print(token, end="", flush=True)
            print(f"{R}\n")

        except (KeyboardInterrupt, EOFError):
            break

    print(f"\n{Y}Goodbye!{R}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate text")
    parser.add_argument("--model", "-m", type=str, help="Model name")
    parser.add_argument("--prompt", "-p", type=str, help="Single prompt")
    parser.add_argument("--list", "-l", action="store_true", help="List models")
    parser.add_argument("--max-tokens", type=int, default=150, help="Max tokens")
    parser.add_argument("--temp", "-t", type=float, default=0.6, help="Temperature (lower=deterministic)")
    args = parser.parse_args()

    if args.list:
        models = list_models()
        if not models:
            print(f"{Y}No models. Train with: python train_fast.py{R}")
            return
        print(f"\n{B}Models:{R}")
        for m in sorted(models, key=lambda x: x.created, reverse=True):
            params = f"{m.params/1e6:.1f}M" if m.params > 0 else "?"
            print(f"  {C}{m.name}{R} ({params}, loss {m.final_loss:.4f})")
        print()
        return

    # Get model
    models = list_models()
    if not models:
        print(f"{Y}No models found. Train with: python train_fast.py{R}")
        sys.exit(1)

    model_name = args.model
    if not model_name:
        # Use most recent
        model_name = sorted(models, key=lambda x: x.created, reverse=True)[0].name
        print(f"{D}Using: {model_name}{R}")

    registry = get_registry()
    if not registry.exists(model_name):
        print(f"{Y}Model '{model_name}' not found.{R}")
        print(f"{D}Available: {', '.join(m.name for m in models)}{R}")
        sys.exit(1)

    # Load
    device = get_device()
    print(f"{D}Loading {model_name}...{R}")
    model, tokenizer, config, info = load_model(model_name, device)

    if args.prompt:
        # For kids-model: use smart multi-strategy responder
        if 'kids' in model_name.lower():
            try:
                from kids_smart_responder import smart_respond
                result = smart_respond(args.prompt)
                if result:
                    print(f"{G}{result}{R}\n")
                else:
                    # Fall back to neural generation
                    print(f"{G}", end="", flush=True)
                    for token in stream_generate(model, tokenizer, args.prompt,
                                                 max_tokens=args.max_tokens,
                                                 temperature=args.temp, device=device):
                        print(token, end="", flush=True)
                    print(f"{R}\n")
            except ImportError:
                # Smart responder not available, use neural
                print(f"{G}", end="", flush=True)
                for token in stream_generate(model, tokenizer, args.prompt,
                                             max_tokens=args.max_tokens,
                                             temperature=args.temp, device=device):
                    print(token, end="", flush=True)
                print(f"{R}\n")
        else:
            # Standard neural generation for other models
            print(f"{G}", end="", flush=True)
            for token in stream_generate(model, tokenizer, args.prompt,
                                         max_tokens=args.max_tokens,
                                         temperature=args.temp, device=device):
                print(token, end="", flush=True)
            print(f"{R}\n")
    else:
        # Interactive mode
        if 'kids' in model_name.lower():
            interactive_hybrid(model, tokenizer, device)
        else:
            interactive(model, tokenizer, device)


if __name__ == "__main__":
    main()
