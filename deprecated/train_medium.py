#!/usr/bin/env python3
"""
Training script for Erosolar Medium - General Purpose LLM.

This script trains a ~50M parameter model optimized for general-purpose
conversation, coding assistance, and knowledge tasks.

Usage:
    python train_medium.py                          # Train with defaults
    python train_medium.py --name my-erosolar       # Custom name
    python train_medium.py --epochs 3               # More training
    python train_medium.py --enhanced               # Use enhanced dataset
"""

import argparse
import math
import time
import gc
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from model import MiniGPT, ModelConfig, create_model
from tokenizer import BPETokenizer
from registry import get_registry, ModelInfo

# Colors
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
MAGENTA = "\033[95m"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"


# =============================================================================
# MEDIUM MODEL CONFIGURATION
# =============================================================================
@dataclass
class MediumConfig:
    """Optimized configuration for medium-sized general purpose LLM."""
    # Model architecture (~50M params)
    vocab_size: int = 16000          # Larger vocab for better coverage
    max_seq_len: int = 512           # Longer context for complex tasks
    embed_dim: int = 768             # Larger embeddings
    num_heads: int = 12              # More attention heads
    num_layers: int = 12             # Deeper model
    ff_dim: int = 3072               # 4x embed_dim (standard ratio)
    dropout: float = 0.1

    # Training hyperparameters
    batch_size: int = 4              # Smaller batch for memory
    gradient_accumulation: int = 8   # Effective batch = 32
    epochs: int = 2
    learning_rate: float = 3e-4      # Slightly lower for stability
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1        # 10% warmup
    max_grad_norm: float = 1.0
    min_lr_ratio: float = 0.1

    # Logging
    log_every_n_steps: int = 50
    save_every_n_steps: int = 500
    eval_every_n_steps: int = 200


def setup_device(seed: int = 42) -> torch.device:
    """Setup compute device with memory optimization."""
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(seed)
        # Memory optimization
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        gpu_name = torch.cuda.get_device_name()
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"{CYAN}GPU: {gpu_name} ({gpu_mem:.1f}GB){RESET}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"{CYAN}Using Apple Silicon MPS{RESET}")
    else:
        device = torch.device("cpu")
        print(f"{YELLOW}Using CPU (training will be slow){RESET}")

    return device


class TextDataset(Dataset):
    """Dataset for language modeling with stride-based windowing."""

    def __init__(self, tokens: list, seq_len: int, stride: Optional[int] = None):
        self.tokens = tokens
        self.seq_len = seq_len
        self.stride = stride or seq_len // 2
        self.num_sequences = max(0, (len(tokens) - seq_len - 1) // self.stride + 1)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.seq_len + 1
        chunk = self.tokens[start:end]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def format_time(seconds: float) -> str:
    """Format seconds into human readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins, secs = divmod(int(seconds), 60)
        return f"{mins}:{secs:02d}"
    else:
        hours, remainder = divmod(int(seconds), 3600)
        mins, secs = divmod(remainder, 60)
        return f"{hours}:{mins:02d}:{secs:02d}"


def get_cosine_schedule(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
    """Cosine learning rate schedule with warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def load_training_data(enhanced: bool = False) -> List[Tuple[str, str]]:
    """Load and prepare training data."""
    print(f"\n{BOLD}Loading Training Data...{RESET}")

    all_data = []

    # Load general training data
    if enhanced:
        try:
            from general_training_data import get_general_training_data
            general_data = get_general_training_data()
            all_data.extend(general_data)
            print(f"  {GREEN}+{RESET} {len(general_data):,} enhanced general training pairs")
        except ImportError:
            print(f"  {YELLOW}Warning: general_training_data.py not found{RESET}")

    # Load from data.py
    try:
        from data import create_training_corpus
        # We'll get the raw pairs instead
        from data import (GREETINGS, LOGIC_REASONING, QA_PAIRS, VOCABULARY_DEFINITIONS,
                         LIFE_ADVICE, CODING_TASKS, TECH_KNOWLEDGE, WHY_QUESTIONS,
                         MATH_PROBLEMS, ALGORITHM_EXPLANATIONS, CHAIN_OF_THOUGHT)

        # Add core data with appropriate weights
        all_data.extend(GREETINGS * 30)
        all_data.extend(LOGIC_REASONING * 20)
        all_data.extend(QA_PAIRS * 15)
        all_data.extend(VOCABULARY_DEFINITIONS * 50)
        all_data.extend(LIFE_ADVICE * 20)
        all_data.extend(CODING_TASKS * 10)
        all_data.extend(TECH_KNOWLEDGE * 100)
        all_data.extend(WHY_QUESTIONS * 20)
        all_data.extend(MATH_PROBLEMS * 20)
        all_data.extend(ALGORITHM_EXPLANATIONS * 15)
        all_data.extend(CHAIN_OF_THOUGHT * 30)

        print(f"  {GREEN}+{RESET} Core training data loaded")
    except ImportError as e:
        print(f"  {YELLOW}Warning: Could not load data.py: {e}{RESET}")

    # Load expanded training data
    try:
        from expanded_training_data import get_expanded_training_data
        expanded = get_expanded_training_data()
        all_data.extend(expanded)
        print(f"  {GREEN}+{RESET} {len(expanded):,} expanded training pairs")
    except ImportError:
        pass

    # Load cached instruction data
    instruction_file = Path("cache/datasets/instruction_data.json")
    if instruction_file.exists():
        try:
            import json
            with open(instruction_file, 'r', encoding='utf-8') as f:
                instruction_data = json.load(f)
            if instruction_data and isinstance(instruction_data[0], list):
                instruction_data = [(d[0], d[1]) for d in instruction_data]
            # Use more instruction data for medium model
            import random
            sample_size = min(len(instruction_data), 50000)
            sampled = random.sample(instruction_data, sample_size)
            all_data.extend(sampled)
            print(f"  {GREEN}+{RESET} {sample_size:,} instruction pairs (from {len(instruction_data):,})")
        except Exception as e:
            print(f"  {YELLOW}Warning: Could not load instruction data: {e}{RESET}")

    # Load Wikipedia data
    wiki_file = Path("cache/wikipedia/wikipedia_instructions.json")
    if wiki_file.exists():
        try:
            import json
            with open(wiki_file, 'r', encoding='utf-8') as f:
                wiki_data = json.load(f)
            if wiki_data and isinstance(wiki_data[0], list):
                wiki_data = [(d[0], d[1]) for d in wiki_data]
            import random
            sample_size = min(len(wiki_data), 30000)
            sampled = random.sample(wiki_data, sample_size)
            all_data.extend(sampled)
            print(f"  {GREEN}+{RESET} {sample_size:,} Wikipedia pairs")
        except Exception as e:
            print(f"  {YELLOW}Warning: Could not load Wikipedia data: {e}{RESET}")

    print(f"\n  {BOLD}Total: {len(all_data):,} training pairs{RESET}")
    return all_data


def create_corpus(data: List[Tuple[str, str]]) -> str:
    """Create training corpus from Q&A pairs."""
    import random
    random.shuffle(data)

    corpus_parts = []
    for question, answer in data:
        # Format as conversation with special tokens
        formatted = f"<|user|>\n{question}\n<|end_turn|>\n<|assistant|>\n{answer}\n<|end_turn|>\n\n"
        corpus_parts.append(formatted)

    return ''.join(corpus_parts)


def train_medium_model(
    name: str = "erosolar-medium",
    desc: str = "General purpose medium LLM",
    config: MediumConfig = None,
    enhanced: bool = False,
    seed: int = 42
):
    """Train the medium model."""

    if config is None:
        config = MediumConfig()

    print(f"\n{BOLD}{MAGENTA}{'═' * 60}{RESET}")
    print(f"{BOLD}{MAGENTA}  Erosolar Medium - Training{RESET}")
    print(f"{BOLD}{MAGENTA}{'═' * 60}{RESET}")

    # Setup
    device = setup_device(seed)
    registry = get_registry()

    # Check if model exists
    if registry.exists(name):
        print(f"\n{YELLOW}Model '{name}' already exists.{RESET}")
        response = input(f"{DIM}Overwrite? [y/N]: {RESET}").strip().lower()
        if response != 'y':
            print("Aborted.")
            return
        registry.delete(name)

    # Load training data
    data = load_training_data(enhanced=enhanced)

    # Create corpus
    print(f"\n{DIM}Creating training corpus...{RESET}")
    corpus = create_corpus(data)
    print(f"  Corpus size: {len(corpus):,} characters")

    # Tokenizer
    print(f"\n{DIM}Preparing tokenizer...{RESET}")
    tokenizer = BPETokenizer()
    tokenizer_cache = Path("cache/tokenizer_medium")

    if tokenizer_cache.exists():
        tokenizer.load(tokenizer_cache)
        print(f"  Loaded cached tokenizer (vocab: {tokenizer.vocab_size:,})")
    else:
        print(f"  Training new tokenizer...")
        tokenizer.train(corpus, config.vocab_size)
        tokenizer.save(tokenizer_cache)
        print(f"  Saved tokenizer (vocab: {tokenizer.vocab_size:,})")

    config.vocab_size = tokenizer.vocab_size

    # Tokenize corpus
    print(f"\n{DIM}Tokenizing corpus...{RESET}")
    tokens = tokenizer.encode(corpus, add_special=False)
    print(f"  Tokens: {len(tokens):,}")

    # Create dataset
    stride = config.max_seq_len // 2
    dataset = TextDataset(tokens, config.max_seq_len, stride)
    print(f"  Sequences: {len(dataset):,}")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == 'cuda')
    )

    # Create model
    print(f"\n{DIM}Creating model...{RESET}")
    model_config = ModelConfig(
        vocab_size=config.vocab_size,
        max_seq_len=config.max_seq_len,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        ff_dim=config.ff_dim,
        dropout=config.dropout
    )
    model = create_model(model_config, device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    print(f"  Architecture: {config.num_layers}L-{config.num_heads}H-{config.embed_dim}D")

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    steps_per_epoch = len(dataloader) // config.gradient_accumulation
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    scheduler = get_cosine_schedule(optimizer, warmup_steps, total_steps, config.min_lr_ratio)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Training info
    print(f"\n{BOLD}{CYAN}{'─' * 60}{RESET}")
    print(f"{BOLD}  Training Configuration{RESET}")
    print(f"{BOLD}{CYAN}{'─' * 60}{RESET}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size} x {config.gradient_accumulation} = {config.batch_size * config.gradient_accumulation}")
    print(f"  Steps/epoch: {steps_per_epoch:,}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Warmup steps: {warmup_steps:,}")
    print(f"  Learning rate: {config.learning_rate}")
    print()

    # Training loop
    model.train()
    start_time = time.time()
    global_step = 0
    best_loss = float('inf')
    running_loss = 0.0
    accumulated_steps = 0

    for epoch in range(config.epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_tokens = 0

        optimizer.zero_grad()

        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            # Forward pass
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss = loss / config.gradient_accumulation

            # Backward pass
            loss.backward()

            accumulated_steps += 1
            running_loss += loss.item() * config.gradient_accumulation

            batch_tokens = (y != 0).sum().item()
            epoch_loss += loss.item() * config.gradient_accumulation * batch_tokens
            epoch_tokens += batch_tokens

            # Gradient accumulation step
            if accumulated_steps >= config.gradient_accumulation:
                if config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                accumulated_steps = 0

                # Logging
                if global_step % config.log_every_n_steps == 0:
                    avg_loss = running_loss / config.log_every_n_steps
                    elapsed = time.time() - start_time
                    lr = scheduler.get_last_lr()[0]

                    progress = global_step / total_steps
                    eta = elapsed / progress - elapsed if progress > 0 else 0

                    print(f"\r{DIM}Epoch {epoch + 1}/{config.epochs} "
                          f"Step {global_step}/{total_steps} | "
                          f"Loss: {avg_loss:.4f} | "
                          f"LR: {lr:.2e} | "
                          f"Time: {format_time(elapsed)} | "
                          f"ETA: {format_time(eta)}{RESET}", end='', flush=True)

                    running_loss = 0.0

        # End of epoch
        epoch_time = time.time() - epoch_start
        epoch_avg_loss = epoch_loss / max(epoch_tokens, 1)

        print(f"\n{GREEN}Epoch {epoch + 1}/{config.epochs} complete{RESET}")
        print(f"  Loss: {epoch_avg_loss:.4f} | Time: {format_time(epoch_time)}")

        if epoch_avg_loss < best_loss:
            best_loss = epoch_avg_loss

        # Generate sample
        if epoch == config.epochs - 1:
            print(f"\n{CYAN}Sample generation:{RESET}")
            model.eval()
            with torch.no_grad():
                sample = model.generate(
                    tokenizer,
                    prompt="<|user|>\nWhat is machine learning?\n<|end_turn|>\n<|assistant|>\n",
                    max_tokens=100,
                    temperature=0.7,
                    top_k=40,
                    device=device
                )
                # Clean up sample
                for stop_tok in ["<|end_turn|>", "<|user|>"]:
                    if stop_tok in sample:
                        sample = sample.split(stop_tok)[0]
                print(f"  {sample[:400]}")
            model.train()

        # Memory cleanup
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Training complete
    total_time = (time.time() - start_time) / 60

    print(f"\n{BOLD}{GREEN}{'═' * 60}{RESET}")
    print(f"{BOLD}{GREEN}  Training Complete!{RESET}")
    print(f"{BOLD}{GREEN}{'═' * 60}{RESET}")

    # Save to registry
    from config import Config, TrainingConfig

    save_config = Config(
        vocab_size=config.vocab_size,
        max_seq_len=config.max_seq_len,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        ff_dim=config.ff_dim,
        dropout=config.dropout,
        training=TrainingConfig(epochs=config.epochs, batch_size=config.batch_size)
    )

    info = registry.save_model(
        name=name,
        description=desc,
        model=model,
        tokenizer=tokenizer,
        config=save_config,
        epochs=config.epochs,
        loss=best_loss,
        training_time=total_time,
        tags=["medium", "general-purpose"],
        preset="medium-custom"
    )

    print(f"  {GREEN}Saved: {info.name}{RESET}")
    print(f"  Parameters: {info.params:,}")
    print(f"  Final loss: {info.final_loss:.4f}")
    print(f"  Training time: {total_time:.1f} min")
    print(f"\n  {DIM}Use: python generate.py --model {info.name}{RESET}")

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Train Erosolar Medium - General Purpose LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_medium.py                     # Train with defaults
  python train_medium.py --name my-model     # Custom name
  python train_medium.py --epochs 3          # Train longer
  python train_medium.py --enhanced          # Use enhanced dataset
  python train_medium.py --small             # Smaller model for testing
        """
    )

    parser.add_argument("--name", type=str, default="erosolar-medium",
                        help="Model name (default: erosolar-medium)")
    parser.add_argument("--desc", type=str, default="General purpose medium LLM",
                        help="Model description")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of epochs (default: 2)")
    parser.add_argument("--enhanced", action="store_true",
                        help="Use enhanced training dataset")
    parser.add_argument("--small", action="store_true",
                        help="Use smaller model for testing (~25M params)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    # Configure model size
    config = MediumConfig()
    config.epochs = args.epochs

    if args.small:
        # Smaller configuration for testing
        config.embed_dim = 512
        config.num_heads = 8
        config.num_layers = 8
        config.ff_dim = 2048
        config.batch_size = 8
        config.gradient_accumulation = 4
        print(f"{YELLOW}Using smaller model configuration for testing{RESET}")

    train_medium_model(
        name=args.name,
        desc=args.desc,
        config=config,
        enhanced=args.enhanced,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
