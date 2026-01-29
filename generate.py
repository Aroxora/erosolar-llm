#!/usr/bin/env python3
"""
Terminal generation app with model registry integration.

Usage:
    python generate.py                     # Interactive model selection
    python generate.py --model my-coder    # Use specific model
    python generate.py --list              # List all models
    python generate.py --prompt "..."      # Single prompt mode
"""

import argparse
import sys
import os
import signal
from pathlib import Path
from typing import Optional, Generator as Gen

# Enable readline for arrow keys, history, editing
try:
    import readline
    # Configure readline
    readline.parse_and_bind('set editing-mode emacs')
    readline.parse_and_bind('"\\e[D": backward-char')
    readline.parse_and_bind('"\\e[C": forward-char')
    readline.parse_and_bind('"\\e[A": previous-history')
    readline.parse_and_bind('"\\e[B": next-history')
except ImportError:
    pass  # Windows fallback

import torch
import torch.nn.functional as F

from model import MiniGPT, InfiniGPT, ModelConfig
from tokenizer import BPETokenizer
from config import Config
from registry import get_registry, list_models, load_model, ModelInfo

# Colors
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"


def setup_device() -> torch.device:
    """Setup compute device. Supports Huawei NPU, CUDA, MPS, and CPU."""
    # Use Huawei NPU compatibility layer for unified device setup
    try:
        from huawei_npu import setup_device as npu_setup_device
        return npu_setup_device(verbose=False)
    except ImportError:
        pass

    # Fallback to standard device setup
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Generator:
    """Text generator using trained model from registry.

    Supports both standard MiniGPT and InfiniGPT models.
    InfiniGPT models use compressive memory for infinite context.
    """

    def __init__(self, model, tokenizer: BPETokenizer, info: ModelInfo,
                 device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.info = info
        self.device = device
        self.max_seq_len = info.max_seq_len

        # Check if model uses Infini-attention
        self.is_infini = isinstance(model, InfiniGPT) or hasattr(model, 'memory_manager')
        if self.is_infini:
            self.segment_size = model.config.segment_size

        # Reasoning token IDs
        self.think_start_id = tokenizer.think_start_id
        self.think_end_id = tokenizer.think_end_id
        self.step_id = tokenizer.step_id
        self.answer_id = tokenizer.answer_id

    @classmethod
    def from_name(cls, name: str, device: torch.device = None) -> "Generator":
        """Load generator from model name."""
        device = device or setup_device()
        model, tokenizer, config, info = load_model(name, device)
        return cls(model, tokenizer, info, device)

    def reset_memory(self):
        """Reset compressive memory for Infini-attention models."""
        if self.is_infini and hasattr(self.model, 'reset_memory'):
            self.model.reset_memory()
            return True
        return False

    def get_memory_stats(self) -> dict:
        """Get memory statistics for Infini-attention models."""
        if self.is_infini and hasattr(self.model, 'get_memory_stats'):
            return self.model.get_memory_stats()
        return {}

    def get_compression_ratio(self, seq_len: int) -> float:
        """Get memory compression ratio for given sequence length."""
        if self.is_infini and hasattr(self.model, 'get_compression_ratio'):
            return self.model.get_compression_ratio(seq_len)
        return 1.0

    @torch.no_grad()
    def generate_stream(self, prompt: str, max_tokens: int = 200,
                        temperature: float = 0.3, top_k: int = 20,
                        top_p: float = 0.85, suppress_unk: bool = True,
                        reasoning: bool = False, show_thinking: bool = True,
                        reset_memory: bool = False) -> Gen[str, None, None]:
        """Generate tokens one at a time, yielding each as it's produced.

        Args:
            reasoning: If True, encourages chain-of-thought generation
            show_thinking: If True, shows <|think_start|> tokens; if False, hides thinking process
            reset_memory: If True, reset compressive memory before generation (Infini-attention only)
        """
        # Reset memory if requested (for Infini-attention models)
        if reset_memory and self.is_infini:
            self.reset_memory()

        # Format prompt - add reasoning hint if in reasoning mode
        if reasoning:
            formatted = f"<|user|>\n{prompt}\n<|end_turn|>\n<|assistant|>\n<|think_start|>"
        else:
            formatted = f"<|user|>\n{prompt}\n<|end_turn|>\n<|assistant|>\n"

        input_ids = [self.tokenizer.bos_token_id]
        input_ids.extend(self.tokenizer.encode(formatted, add_special=False))
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        generated_text = ""
        repetition_penalty = 2.0  # Very high penalty to prevent loops

        # Get UNK token ID for suppression
        unk_token_id = self.tokenizer.token_to_id.get("<|unk|>", 1)

        # Reasoning state tracking
        in_thinking = reasoning  # Start in thinking if reasoning mode
        thinking_content = ""

        # Get reasoning token IDs
        think_start = self.tokenizer.token_to_id.get("<|think_start|>", -1)
        think_end = self.tokenizer.token_to_id.get("<|think_end|>", -1)
        step_token = self.tokenizer.token_to_id.get("<|step|>", -1)
        answer_token = self.tokenizer.token_to_id.get("<|answer|>", -1)

        # For Infini-attention: process long prompts in segments first
        if self.is_infini and input_ids.size(1) > self.segment_size:
            num_full_segments = (input_ids.size(1) - 1) // self.segment_size
            for seg_idx in range(num_full_segments):
                start = seg_idx * self.segment_size
                end = start + self.segment_size
                _ = self.model(input_ids[:, start:end], use_memory=True, update_memory=True)

        for _ in range(max_tokens):
            # For Infini-attention: use segment window but with memory
            if self.is_infini:
                idx = input_ids[:, -self.segment_size:]
                logits = self.model(idx, use_memory=True, update_memory=True)[:, -1, :]
            else:
                idx = input_ids[:, -self.max_seq_len:]
                logits = self.model(idx)[:, -1, :]

            # Suppress UNK token to prevent <|unk|> in output
            if suppress_unk:
                logits[0, unk_token_id] = float('-inf')

            # Stronger repetition penalty with context window
            recent_tokens = set(input_ids[0].tolist()[-80:])  # Larger window
            for token_id in recent_tokens:
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

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

            if next_token_id == self.tokenizer.eos_token_id:
                break

            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Decode with context for proper spacing
            token_text = self.tokenizer.id_to_token.get(next_token_id, '')

            # Skip UNK tokens entirely in output
            if token_text == "<|unk|>":
                continue

            # Handle reasoning tokens
            if next_token_id == think_start:
                in_thinking = True
                if show_thinking:
                    yield "\n[Thinking...]\n"
                continue
            elif next_token_id == think_end:
                in_thinking = False
                if show_thinking:
                    yield "\n[/Thinking]\n"
                continue
            elif next_token_id == step_token:
                if show_thinking or not in_thinking:
                    yield "\n• "
                continue
            elif next_token_id == answer_token:
                in_thinking = False
                if show_thinking:
                    yield "\n[Answer] "
                continue

            # If in thinking and not showing, skip the content
            if in_thinking and not show_thinking:
                continue

            # Add space before word tokens (not punctuation)
            if token_text and token_text[0].isalnum():
                token_text = ' ' + token_text

            generated_text += token_text

            # Stop if we hit end_turn or new user turn (model trying to continue conversation)
            if "<|end_turn|>" in generated_text or "<|user|>" in generated_text:
                break

            yield token_text

    def generate(self, prompt: str, max_tokens: int = 200,
                 temperature: float = 0.3, top_k: int = 20,
                 top_p: float = 0.85, suppress_unk: bool = True,
                 reasoning: bool = False, show_thinking: bool = True) -> str:
        """Generate complete response."""
        tokens = list(self.generate_stream(
            prompt, max_tokens, temperature, top_k, top_p, suppress_unk,
            reasoning, show_thinking
        ))
        result = ''.join(tokens).strip()
        # Clean up spacing
        result = result.replace('  ', ' ')
        # Remove any trailing special tokens
        for stop_tok in ["<|end_turn|>", "<|user|>", "<|assistant|>"]:
            if stop_tok in result:
                result = result.split(stop_tok)[0].strip()
        return result


def print_models_table():
    """Print a formatted table of all models."""
    models = list_models()

    if not models:
        print(f"\n{YELLOW}No models found.{RESET}")
        print(f"{DIM}Train one with: python train.py --name my-model{RESET}\n")
        return False

    print(f"\n{BOLD}{'═' * 70}{RESET}")
    print(f"{BOLD}  Available Models{RESET}")
    print(f"{BOLD}{'═' * 70}{RESET}")

    print(f"{DIM}{'#':<4} {'Name':<18} {'Params':<10} {'Loss':<10} {'Description'}{RESET}")
    print(f"{DIM}{'─' * 70}{RESET}")

    sorted_models = sorted(models, key=lambda x: x.created, reverse=True)
    for i, m in enumerate(sorted_models, 1):
        params_str = f"{m.params / 1e6:.1f}M" if m.params > 0 else "?"
        loss_str = f"{m.final_loss:.4f}" if m.final_loss > 0 else "?"
        desc = (m.description[:30] + "...") if len(m.description) > 33 else m.description

        print(f"{CYAN}{i:<4}{RESET} {m.name:<18} {params_str:<10} {loss_str:<10} {DIM}{desc}{RESET}")

    print(f"{DIM}{'─' * 70}{RESET}\n")
    return sorted_models


def select_model_interactive() -> Optional[str]:
    """Interactive model selection."""
    models = print_models_table()
    if not models:
        return None

    while True:
        try:
            choice = input(f"{BLUE}{BOLD}Select model (number or name): {RESET}").strip()

            if not choice or choice.lower() in ('q', 'quit', 'exit'):
                return None

            # Try as number
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(models):
                    return models[idx].name

            # Try as name
            for m in models:
                if m.name.lower() == choice.lower():
                    return m.name

            print(f"{YELLOW}Invalid selection. Try again.{RESET}")

        except (KeyboardInterrupt, EOFError):
            return None


def print_banner(model_name: str, info: ModelInfo, generator: "Generator" = None):
    """Print startup banner."""
    print(f"\n{BOLD}{CYAN}╔══════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{CYAN}║{RESET}  {BOLD}Erosolar{RESET} - {model_name:<40} {BOLD}{CYAN}║{RESET}")
    print(f"{BOLD}{CYAN}╚══════════════════════════════════════════════════════════╝{RESET}")
    if info.description:
        print(f"{DIM}  {info.description}{RESET}")
    print(f"{DIM}  {info.params:,} params | {info.preset} preset | loss {info.final_loss:.4f}{RESET}")

    # Show Infini-attention info if applicable
    if generator and generator.is_infini:
        stats = generator.get_memory_stats()
        if stats:
            compression = generator.get_compression_ratio(10000)
            print(f"{DIM}  Infini-attention: segment={stats.get('segment_size', 'N/A')}, "
                  f"~{compression:.0f}x compression at 10K tokens{RESET}")

    print(f"\n{DIM}  Commands: 'quit' to exit, 'clear' to reset, 'switch' to change model{RESET}")
    if generator and generator.is_infini:
        print(f"{DIM}  Infini: 'memory' to view stats, 'reset' to clear context memory{RESET}")
    print(f"{DIM}{'─' * 60}{RESET}\n")


def interactive_mode(generator: Generator, stream: bool = True,
                     reasoning: bool = False, show_thinking: bool = True):
    """Run interactive chat session."""
    print_banner(generator.info.name, generator.info, generator)

    if reasoning:
        print(f"{CYAN}  [Reasoning mode enabled - model will show step-by-step thinking]{RESET}\n")

    while True:
        try:
            # Use different prompt indicator for reasoning mode
            prompt_char = "?" if reasoning else ">>>"
            # Add infinity symbol for Infini-attention models
            if generator.is_infini:
                prompt_char = "∞" if not reasoning else "∞?"
            prompt = input(f"{BLUE}{BOLD}{prompt_char} {RESET}").strip()

            if not prompt:
                continue

            if prompt.lower() in ("quit", "exit", "q"):
                print(f"\n{YELLOW}Goodbye!{RESET}")
                break

            if prompt.lower() == "clear":
                os.system('clear' if os.name == 'posix' else 'cls')
                print_banner(generator.info.name, generator.info, generator)
                if reasoning:
                    print(f"{CYAN}  [Reasoning mode enabled]{RESET}\n")
                continue

            if prompt.lower() == "switch":
                new_name = select_model_interactive()
                if new_name and new_name != generator.info.name:
                    print(f"{DIM}Loading {new_name}...{RESET}")
                    generator = Generator.from_name(new_name, generator.device)
                    print_banner(generator.info.name, generator.info, generator)
                continue

            if prompt.lower() == "reasoning":
                reasoning = not reasoning
                status = "enabled" if reasoning else "disabled"
                print(f"{CYAN}Reasoning mode {status}{RESET}\n")
                continue

            # Infini-attention specific commands
            if prompt.lower() == "reset" and generator.is_infini:
                if generator.reset_memory():
                    print(f"{CYAN}Compressive memory cleared. Context reset.{RESET}\n")
                else:
                    print(f"{YELLOW}Memory reset not available for this model.{RESET}\n")
                continue

            if prompt.lower() == "memory" and generator.is_infini:
                stats = generator.get_memory_stats()
                if stats:
                    print(f"\n{CYAN}Infini-attention Memory Stats:{RESET}")
                    print(f"{DIM}  Segment size: {stats.get('segment_size', 'N/A')} tokens{RESET}")
                    print(f"{DIM}  Memory per layer: {stats.get('memory_size_per_layer', 'N/A')} params{RESET}")
                    print(f"{DIM}  Total memory: {stats.get('total_memory_params', 'N/A')} params{RESET}")
                    print(f"{DIM}  Compression at 1K tokens: {generator.get_compression_ratio(1000):.1f}x{RESET}")
                    print(f"{DIM}  Compression at 10K tokens: {generator.get_compression_ratio(10000):.1f}x{RESET}")
                    print(f"{DIM}  Compression at 100K tokens: {generator.get_compression_ratio(100000):.1f}x{RESET}")
                    print()
                continue

            if prompt.lower() == "help":
                print(f"{DIM}Commands:{RESET}")
                print(f"{DIM}  reasoning - Toggle chain-of-thought reasoning mode{RESET}")
                print(f"{DIM}  clear     - Clear screen{RESET}")
                print(f"{DIM}  switch    - Switch to different model{RESET}")
                print(f"{DIM}  info      - Show model info{RESET}")
                if generator.is_infini:
                    print(f"{DIM}  memory    - Show Infini-attention memory stats{RESET}")
                    print(f"{DIM}  reset     - Clear compressive memory (reset context){RESET}")
                print(f"{DIM}  quit      - Exit{RESET}")
                print(f"{DIM}Example: What is 15% of 80?{RESET}\n")
                continue

            if prompt.lower() == "info":
                info = generator.info
                print(f"\n{CYAN}Model: {info.name}{RESET}")
                print(f"{DIM}  Description: {info.description}{RESET}")
                print(f"{DIM}  Parameters: {info.params:,}{RESET}")
                print(f"{DIM}  Architecture: {info.num_layers}L-{info.num_heads}H-{info.embed_dim}D{RESET}")
                if generator.is_infini:
                    print(f"{DIM}  Attention: Infini-attention (infinite context){RESET}")
                    print(f"{DIM}  Segment size: {generator.segment_size} tokens{RESET}")
                print(f"{DIM}  Trained: {info.epochs_trained} epochs, loss {info.final_loss:.4f}{RESET}")
                print(f"{DIM}  Created: {info.created[:19]}{RESET}")
                print(f"{DIM}  Reasoning mode: {'enabled' if reasoning else 'disabled'}{RESET}\n")
                continue

            print(f"{GREEN}", end="", flush=True)

            if stream:
                for token in generator.generate_stream(prompt, reasoning=reasoning,
                                                       show_thinking=show_thinking):
                    print(token, end="", flush=True)
                print(f"{RESET}\n")
            else:
                response = generator.generate(prompt, reasoning=reasoning,
                                             show_thinking=show_thinking)
                print(f"{response}{RESET}\n")

        except KeyboardInterrupt:
            print(f"\n{YELLOW}Goodbye!{RESET}")
            break
        except EOFError:
            print(f"\n{YELLOW}Goodbye!{RESET}")
            break


def main():
    parser = argparse.ArgumentParser(description="Generate with Erosolar models")
    parser.add_argument("--model", "-m", type=str, default=None, help="Model name")
    parser.add_argument("--list", "-l", action="store_true", help="List all models")
    parser.add_argument("--prompt", "-p", type=str, default=None, help="Single prompt")
    parser.add_argument("--prompts", type=str, nargs='+', help="Multiple prompts with shared context")
    parser.add_argument("--max-tokens", type=int, default=200, help="Max tokens")
    parser.add_argument("--temperature", "-t", type=float, default=0.3, help="Temperature (lower = more deterministic)")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.85, help="Top-p sampling")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding (temperature=0.01, top_k=1) for deterministic output")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming")
    parser.add_argument("--allow-unk", action="store_true", help="Allow <|unk|> tokens in output")
    parser.add_argument("--reasoning", "-r", action="store_true",
                        help="Enable chain-of-thought reasoning mode")
    parser.add_argument("--hide-thinking", action="store_true",
                        help="Hide thinking process, show only final answer")
    args = parser.parse_args()

    # List mode
    if args.list:
        print_models_table()
        return

    # Get model name
    model_name = args.model
    if not model_name:
        models = list_models()
        if not models:
            print(f"{RED}No models found.{RESET}")
            print(f"{DIM}Train one with: python train.py --name my-model{RESET}")
            sys.exit(1)
        elif len(models) == 1:
            model_name = models[0].name
            print(f"{DIM}Using only available model: {model_name}{RESET}")
        else:
            model_name = select_model_interactive()
            if not model_name:
                sys.exit(0)

    # Check model exists
    registry = get_registry()
    if not registry.exists(model_name):
        print(f"{RED}Model '{model_name}' not found.{RESET}")
        print(f"{DIM}Available models:{RESET}")
        for m in list_models():
            print(f"{DIM}  - {m.name}{RESET}")
        sys.exit(1)

    # Load model
    print(f"{DIM}Loading {model_name}...{RESET}")
    device = setup_device()
    generator = Generator.from_name(model_name, device)

    # Run
    suppress_unk = not args.allow_unk
    reasoning = args.reasoning
    show_thinking = not args.hide_thinking

    # Apply greedy decoding if requested
    temperature = args.temperature
    top_k = args.top_k
    top_p = args.top_p
    if args.greedy:
        temperature = 0.01  # Near-zero for deterministic output
        top_k = 1  # Only pick the top token
        top_p = 1.0  # Disable nucleus sampling

    if args.prompts:
        # Multi-prompt mode with shared context
        context = ""
        for i, prompt in enumerate(args.prompts):
            print(f"\n{CYAN}[Prompt {i+1}]{RESET} {prompt}")
            print(f"{GREEN}", end="", flush=True)

            # Build prompt with context
            full_prompt = f"{context}{prompt}" if context else prompt

            response = generator.generate(
                full_prompt, args.max_tokens, temperature, top_k, top_p,
                suppress_unk, reasoning, show_thinking
            )
            print(f"{response}{RESET}")

            # Accumulate context for next prompt
            context += f"{prompt}\n{response}\n\n"
        print()
    elif args.prompt:
        if args.no_stream:
            response = generator.generate(
                args.prompt, args.max_tokens, temperature, top_k, top_p,
                suppress_unk, reasoning, show_thinking
            )
            print(response)
        else:
            for token in generator.generate_stream(
                args.prompt, args.max_tokens, temperature, top_k, top_p,
                suppress_unk, reasoning, show_thinking
            ):
                print(token, end="", flush=True)
            print()
    else:
        interactive_mode(generator, stream=not args.no_stream, reasoning=reasoning,
                        show_thinking=show_thinking)


if __name__ == "__main__":
    # Clean exit on Ctrl+C
    signal.signal(signal.SIGINT, lambda s, f: (print(f"\n{YELLOW}Goodbye!{RESET}"), sys.exit(0)))
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Goodbye!{RESET}")
        sys.exit(0)
