#!/usr/bin/env python3
"""
Training script with model registry integration.

ENHANCED with:
- Curriculum Learning (foundation -> bridge -> sophisticated)
- Reasoning-Weighted Loss (thinking tokens weighted 1.0x)
- Chain-of-Thought data integration

Usage:
    python train.py --name my-coder --desc "Code generation model"
    python train.py --name math-model --epochs 2
    python train.py --curriculum              # 3-phase curriculum learning
    python train.py --reasoning-weight 1.0    # Weight thinking tokens
    python train.py --list                    # List saved models
"""

import argparse
import json
import math
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from model import MiniGPT, InfiniGPT, ModelConfig, create_model
from tokenizer import BPETokenizer, SpecialTokens
from config import Config, get_preset
from data import create_training_corpus
from registry import get_registry, list_models, ModelInfo

# Training Data Management (for chunk handling)
try:
    from manage_training_data import ensure_main_file as ensure_training_data
    TRAINING_DATA_MANAGEMENT = True
except ImportError:
    TRAINING_DATA_MANAGEMENT = False

# Training Data Upgrade Pipeline (REQUIRED)
try:
    from training_upgrade_pipeline import (
        TrainingDataUpgradePipeline,
        PipelineConfig,
        IterativeTrainer,
        get_pipeline,
        initialize_pipeline
    )
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    print(f"\033[93mWarning: Training upgrade pipeline not available\033[0m")

# Reasoning-Consistent Training (OPPOSITE of GPT-2's passive diversity approach)
# Instead of hoping reasoning emerges from random data, we explicitly reinforce
# similar reasoning patterns across different domains
try:
    from reasoning_consistency import (
        ReasoningConsistencyManager,
        ContrastiveReasoningLoss,
    )
    REASONING_CONSISTENCY_AVAILABLE = True
except ImportError:
    REASONING_CONSISTENCY_AVAILABLE = False

# Colors
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
MAGENTA = "\033[95m"

# Thinking token markers (must match tokenizer special tokens)
THINK_START = "<|think_start|>"
THINK_END = "<|think_end|>"

# Global override: all training data weights are 1x
FORCE_TRAINING_WEIGHTS_1X = True


# ==============================================================================
# REASONING-WEIGHTED LOSS
# ==============================================================================

class ReasoningWeightedLoss(nn.Module):
    """
    Cross-entropy loss that weights reasoning/thinking tokens higher.

    The key insight: if we want the model to REASON, we must make
    reasoning tokens MORE IMPORTANT in the loss function.

    Default weights:
    - Thinking tokens (<|think_start|>...<|think_end|>): 1.0x weight
    - Answer tokens: 1.0x weight
    """

    def __init__(self, tokenizer: BPETokenizer, thinking_weight: float = 1.0,
                 ignore_index: int = 0):
        super().__init__()
        self.tokenizer = tokenizer
        self.thinking_weight = thinking_weight
        self.ignore_index = ignore_index

        # Get token IDs for thinking markers
        self.think_start_ids = self._encode_marker(THINK_START)
        self.think_end_ids = self._encode_marker(THINK_END)

        print(f"{DIM}ReasoningWeightedLoss initialized: thinking_weight={thinking_weight}{RESET}")

    def _encode_marker(self, marker: str) -> List[int]:
        """Encode a marker string to token IDs."""
        try:
            return self.tokenizer.encode(marker, add_special=False)
        except:
            return []

    def _create_thinking_mask(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Create a mask indicating which tokens are inside thinking blocks.
        Returns tensor of same shape as targets with True for thinking tokens.
        """
        batch_size, seq_len = targets.shape
        mask = torch.zeros_like(targets, dtype=torch.bool)

        # If we don't have thinking markers, return all-False mask
        if not self.think_start_ids or not self.think_end_ids:
            return mask

        # For each sequence in batch, find thinking regions
        targets_list = targets.tolist()

        for b in range(batch_size):
            seq = targets_list[b]
            in_thinking = False

            for i in range(seq_len):
                # Check for start marker
                if self._matches_marker(seq, i, self.think_start_ids):
                    in_thinking = True

                if in_thinking:
                    mask[b, i] = True

                # Check for end marker
                if self._matches_marker(seq, i, self.think_end_ids):
                    in_thinking = False

        return mask

    def _matches_marker(self, seq: List[int], pos: int, marker_ids: List[int]) -> bool:
        """Check if sequence matches marker at position."""
        if not marker_ids:
            return False
        if pos + len(marker_ids) > len(seq):
            return False
        return seq[pos:pos + len(marker_ids)] == marker_ids

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted cross-entropy loss.

        Args:
            logits: [batch, seq_len, vocab_size]
            targets: [batch, seq_len]
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Flatten for cross-entropy
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)

        # If weight is 1.0, use standard cross-entropy (faster)
        if self.thinking_weight == 1.0:
            return F.cross_entropy(
                logits_flat, targets_flat,
                ignore_index=self.ignore_index
            )

        # Compute per-token loss
        loss_per_token = F.cross_entropy(
            logits_flat, targets_flat,
            ignore_index=self.ignore_index,
            reduction='none'
        )

        # Create thinking mask and apply weights
        thinking_mask = self._create_thinking_mask(targets).view(-1)

        # Weight thinking tokens higher
        weights = torch.where(
            thinking_mask,
            torch.tensor(self.thinking_weight, device=logits.device),
            torch.tensor(1.0, device=logits.device)
        )

        # Apply weights and compute mean (ignoring padding)
        valid_mask = (targets_flat != self.ignore_index)
        weighted_loss = loss_per_token * weights

        return weighted_loss[valid_mask].mean()


# ==============================================================================
# CURRICULUM LEARNING
# ==============================================================================

@dataclass
class CurriculumPhase:
    """A single phase in curriculum learning."""
    name: str
    data_path: Path
    epochs: int
    description: str
    weight: float = 1.0  # How much to weight this phase's data


class CurriculumDataset(Dataset):
    """Dataset that loads curriculum phase data from JSONL with weight support."""

    def __init__(self, data_path: Path, tokenizer: BPETokenizer, seq_len: int,
                 default_weight: float = 1.0):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.examples = []
        self.weights = []

        if not data_path.exists():
            print(f"{YELLOW}Warning: Curriculum data not found: {data_path}{RESET}")
            return

        enhanced_count = 0
        total_weight = 0.0

        # Load JSONL data
        with open(data_path) as f:
            for line in f:
                try:
                    record = json.loads(line)
                    msgs = record.get("messages", [])
                    if len(msgs) >= 2:
                        prompt = msgs[0].get("content", "")
                        response = msgs[1].get("content", "")

                        # Get weight from metadata (default to 1.0)
                        metadata = record.get("metadata", {})
                        if FORCE_TRAINING_WEIGHTS_1X:
                            weight = 1.0
                        else:
                            weight = metadata.get("weight", default_weight)
                        is_enhanced = metadata.get("is_enhanced", False)

                        if is_enhanced:
                            enhanced_count += 1

                        # Format as training example
                        text = f"<|user|>\n{prompt}\n<|end_turn|>\n<|assistant|>\n{response}\n<|end_turn|>"
                        tokens = tokenizer.encode(text)

                        if len(tokens) > 10:  # Skip very short examples
                            self.examples.append(tokens)
                            self.weights.append(weight)
                            total_weight += weight
                except:
                    continue

        # Convert weights to sampling probabilities
        if self.weights:
            total = sum(self.weights)
            self.sampling_probs = [w / total for w in self.weights]
        else:
            self.sampling_probs = []

        avg_weight = total_weight / len(self.examples) if self.examples else 0
        print(f"{DIM}  Loaded {len(self.examples)} examples from {data_path.name}{RESET}")
        if enhanced_count > 0:
            print(f"{DIM}    Enhanced: {enhanced_count}, avg weight: {avg_weight:.2f}{RESET}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens = self.examples[idx]

        # Pad or truncate to seq_len
        if len(tokens) > self.seq_len + 1:
            tokens = tokens[:self.seq_len + 1]
        else:
            tokens = tokens + [0] * (self.seq_len + 1 - len(tokens))

        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


class CurriculumTrainer:
    """
    Trainer that implements curriculum learning in phases.

    Phase 1: Foundational Knowledge (basic facts, definitions)
    Phase 2: Foundational CoT (concise reasoning on core facts)
    Phase 3: Bridge Data (connecting facts to reasoning)
    Phase 4: Bridge CoT (multi-step bridge reasoning)
    Phase 5: Sophisticated Reasoning (complex problems, CoT)

    Each phase builds on the previous, teaching the model to reason.
    """

    DEFAULT_PHASES = [
        CurriculumPhase(
            name="foundation",
            data_path=Path("cache/foundations/foundational_knowledge.jsonl"),
            epochs=1,
            description="Basic facts and definitions",
            weight=1.0
        ),
        CurriculumPhase(
            name="foundation_cot",
            data_path=Path("cache/foundations/foundational_cot.jsonl"),
            epochs=1,
            description="Foundational concepts with concise CoT",
            weight=1.0
        ),
        CurriculumPhase(
            name="bridge",
            data_path=Path("cache/bridge/bridge_data.jsonl"),
            epochs=1,
            description="Connecting foundations to reasoning",
            weight=1.0
        ),
        CurriculumPhase(
            name="bridge_cot",
            data_path=Path("cache/bridge/bridge_cot.jsonl"),
            epochs=1,
            description="Bridge data with multi-step CoT",
            weight=1.0
        ),
        CurriculumPhase(
            name="reasoning",
            data_path=Path("cache/reasoning/reasoning_chains.jsonl"),
            epochs=1,
            description="Sophisticated reasoning chains",
            weight=1.0
        ),
        CurriculumPhase(
            name="cot",
            data_path=Path("cache/cot/cot_training_data.jsonl"),
            epochs=1,
            description="Chain-of-thought training data",
            weight=1.0
        ),
    ]

    def __init__(self, model: nn.Module, tokenizer: BPETokenizer, config: Config,
                 device: torch.device, phases: List[CurriculumPhase] = None,
                 thinking_weight: float = 1.0):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.phases = phases or self.DEFAULT_PHASES
        self.thinking_weight = thinking_weight

        # Use reasoning-weighted loss
        self.criterion = ReasoningWeightedLoss(
            tokenizer, thinking_weight=thinking_weight
        )

    def train_phase(self, phase: CurriculumPhase, phase_num: int, total_phases: int) -> float:
        """Train on a single curriculum phase."""

        print(f"\n{BOLD}{MAGENTA}{'═' * 60}{RESET}")
        print(f"{BOLD}{MAGENTA}  Phase {phase_num}/{total_phases}: {phase.name.upper()}{RESET}")
        print(f"{DIM}  {phase.description}{RESET}")
        print(f"{MAGENTA}{'═' * 60}{RESET}")

        # Load phase data
        dataset = CurriculumDataset(
            phase.data_path,
            self.tokenizer,
            self.config.max_seq_len
        )

        if len(dataset) == 0:
            print(f"{YELLOW}  Skipping phase (no data){RESET}")
            return 0.0

        # Create data loader with weighted sampling if weights are present
        if hasattr(dataset, 'weights') and dataset.weights and any(w != 1.0 for w in dataset.weights):
            # Use weighted sampling for enhanced data
            sampler = WeightedRandomSampler(
                weights=dataset.weights,
                num_samples=len(dataset),
                replacement=True
            )
            loader = DataLoader(
                dataset,
                batch_size=self.config.training.batch_size,
                sampler=sampler,
                num_workers=0,
                pin_memory=(self.device.type == 'cuda')
            )
            print(f"{DIM}  Using weighted sampling for enhanced data{RESET}")
        else:
            loader = DataLoader(
                dataset,
                batch_size=self.config.training.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=(self.device.type == 'cuda')
            )

        # Optimizer for this phase (could adjust LR per phase)
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate * phase.weight,
            weight_decay=self.config.training.weight_decay
        )

        # Training loop for this phase
        import time
        self.model.train()
        total_loss = 0.0
        total_tokens = 0

        for epoch in range(phase.epochs):
            epoch_loss = 0.0
            epoch_tokens = 0
            epoch_start = time.time()

            for batch_idx, (x, y) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()

                if hasattr(self.model, 'memory_manager'):
                    logits = self.model(x, use_memory=False, update_memory=False)
                else:
                    logits = self.model(x)

                # Use reasoning-weighted loss
                loss = self.criterion(logits, y)
                loss.backward()

                if self.config.training.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.max_grad_norm
                    )

                optimizer.step()

                batch_tokens = (y != 0).sum().item()
                epoch_loss += loss.item() * batch_tokens
                epoch_tokens += batch_tokens

                if (batch_idx + 1) % 50 == 0:
                    elapsed = time.time() - epoch_start
                    batches_done = batch_idx + 1
                    batches_left = len(loader) - batches_done
                    eta_sec = (elapsed / batches_done) * batches_left if batches_done > 0 else 0
                    print(f"\r{DIM}  Epoch {epoch+1}/{phase.epochs} Batch {batches_done}/{len(loader)} "
                          f"Loss: {epoch_loss/max(epoch_tokens,1):.4f} "
                          f"ETA: {int(eta_sec)}s{RESET}    ", end='', flush=True)

            total_loss += epoch_loss
            total_tokens += epoch_tokens
            epoch_time = time.time() - epoch_start
            print(f"\n{GREEN}  Epoch {epoch+1} complete: Loss {epoch_loss/max(epoch_tokens,1):.4f} ({int(epoch_time)}s){RESET}")

        avg_loss = total_loss / max(total_tokens, 1)
        print(f"{GREEN}  Phase {phase.name} complete: Avg Loss {avg_loss:.4f}{RESET}")

        return avg_loss

    def train(self) -> float:
        """Run full curriculum training through all phases."""
        import time

        print(f"\n{BOLD}{CYAN}{'═' * 60}{RESET}")
        print(f"{BOLD}{CYAN}  CURRICULUM LEARNING{RESET}")
        print(f"{BOLD}{CYAN}{'═' * 60}{RESET}")
        print(f"{DIM}  Phases: {len(self.phases)}{RESET}")
        print(f"{DIM}  Thinking weight: {self.thinking_weight}x{RESET}")

        total_loss = 0.0
        completed_phases = 0
        phase_times = []
        start_time = time.time()

        for i, phase in enumerate(self.phases, 1):
            phase_start = time.time()
            phase_loss = self.train_phase(phase, i, len(self.phases))
            phase_elapsed = time.time() - phase_start
            phase_times.append(phase_elapsed)

            if phase_loss > 0:
                total_loss += phase_loss
                completed_phases += 1

            # Estimate time remaining
            if i < len(self.phases):
                avg_phase_time = sum(phase_times) / len(phase_times)
                remaining_phases = len(self.phases) - i
                eta_seconds = avg_phase_time * remaining_phases
                eta_min = int(eta_seconds // 60)
                eta_sec = int(eta_seconds % 60)
                elapsed = time.time() - start_time
                elapsed_min = int(elapsed // 60)
                elapsed_sec = int(elapsed % 60)
                print(f"{CYAN}  ⏱ Elapsed: {elapsed_min}m {elapsed_sec}s | ETA: ~{eta_min}m {eta_sec}s ({remaining_phases} phases left){RESET}")

        avg_loss = total_loss / max(completed_phases, 1)
        total_time = time.time() - start_time
        total_min = int(total_time // 60)
        total_sec = int(total_time % 60)

        print(f"\n{BOLD}{GREEN}{'═' * 60}{RESET}")
        print(f"{BOLD}{GREEN}  CURRICULUM COMPLETE{RESET}")
        print(f"{GREEN}{'═' * 60}{RESET}")
        print(f"{DIM}  Completed phases: {completed_phases}/{len(self.phases)}{RESET}")
        print(f"{DIM}  Average loss: {avg_loss:.4f}{RESET}")
        print(f"{DIM}  Total time: {total_min}m {total_sec}s{RESET}")

        return avg_loss


def setup_device(seed: int = 42) -> torch.device:
    """Setup compute device and seed. Supports Huawei NPU, CUDA, MPS, and CPU."""
    # Use Huawei NPU compatibility layer for unified device setup
    try:
        from huawei_npu import setup_device as npu_setup_device
        return npu_setup_device(seed=seed, verbose=True)
    except ImportError:
        pass

    # Fallback to standard device setup
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(seed)
        print(f"{DIM}Using CUDA: {torch.cuda.get_device_name()}{RESET}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"{DIM}Using Apple Silicon MPS{RESET}")
    else:
        device = torch.device("cpu")
        print(f"{DIM}Using CPU{RESET}")
    return device


class TextDataset(Dataset):
    """Simple dataset for language modeling."""

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


class ReasoningTextDataset(Dataset):
    """Sample-aware dataset with reasoning clusters and dynamic weights."""

    def __init__(self, samples: List[str], cluster_ids: List[int], weights: List[float],
                 tokenizer: BPETokenizer, seq_len: int, stride: Optional[int] = None):
        self.seq_len = seq_len
        self.stride = stride or seq_len
        self.tokenized_samples: List[List[int]] = []
        self.chunks: List[Tuple[int, int, int]] = []
        seq_cluster_ids: List[int] = []
        seq_weights: List[float] = []

        for text, cluster_id, weight in zip(samples, cluster_ids, weights):
            token_ids = tokenizer.encode(text, add_special=False)
            if len(token_ids) < 2:
                continue

            sample_idx = len(self.tokenized_samples)
            self.tokenized_samples.append(token_ids)

            for start in range(0, len(token_ids) - 1, self.stride):
                end = min(start + seq_len + 1, len(token_ids))
                if end - start < 2:
                    continue

                self.chunks.append((sample_idx, start, end))
                seq_cluster_ids.append(int(cluster_id))
                seq_weights.append(float(weight))

        self.cluster_ids = torch.tensor(seq_cluster_ids, dtype=torch.long)
        self.weights = torch.tensor(seq_weights, dtype=torch.float32)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        sample_idx, start, end = self.chunks[idx]
        token_ids = self.tokenized_samples[sample_idx]
        chunk = token_ids[start:end]
        if len(chunk) < self.seq_len + 1:
            chunk = chunk + [0] * (self.seq_len + 1 - len(chunk))
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y, self.cluster_ids[idx], self.weights[idx]


def compute_shared_cot_weights(
    sample_texts: List[str],
    base_weights: List[float],
    cluster_boost: float = 0.5,
) -> Tuple[List[int], List[float], dict]:
    """Compute dynamic weights that boost shared CoT clusters."""
    if not sample_texts:
        return [], [], {"clusters": 0, "max_cluster": 0, "weight_min": 1.0, "weight_max": 1.0}
    if not base_weights or len(base_weights) != len(sample_texts):
        base_weights = [1.0] * len(sample_texts)

    manager = ReasoningConsistencyManager()
    cluster_ids, dynamic_weights = manager.process_training_data(sample_texts)

    cluster_ids_list = [int(c) for c in cluster_ids]
    cluster_counts = Counter(cluster_ids_list)
    max_cluster = max(cluster_counts.values()) if cluster_counts else 1

    final_weights = []
    for idx, base in enumerate(base_weights):
        cluster_size = cluster_counts.get(cluster_ids_list[idx], 1)
        shared_factor = 1.0 + cluster_boost * (cluster_size / max_cluster)
        final = float(base) * float(dynamic_weights[idx]) * shared_factor
        final_weights.append(final)

    mean_weight = sum(final_weights) / len(final_weights) if final_weights else 1.0
    if mean_weight > 0:
        final_weights = [w / mean_weight for w in final_weights]

    stats = {
        "clusters": len(cluster_counts),
        "max_cluster": max_cluster,
        "weight_min": min(final_weights) if final_weights else 1.0,
        "weight_max": max(final_weights) if final_weights else 1.0,
    }
    return cluster_ids_list, final_weights, stats


def create_reasoning_dataset(
    sample_texts: List[str],
    base_weights: List[float],
    tokenizer: BPETokenizer,
    config: Config,
    cluster_boost: float = 0.5,
) -> Dataset:
    """Create a reasoning-consistent dataset with shared CoT weighting."""
    stride = int(config.max_seq_len * config.data.stride_ratio)
    cluster_ids, weights, stats = compute_shared_cot_weights(
        sample_texts, base_weights, cluster_boost=cluster_boost
    )

    dataset = ReasoningTextDataset(
        samples=sample_texts,
        cluster_ids=cluster_ids,
        weights=weights,
        tokenizer=tokenizer,
        seq_len=config.max_seq_len,
        stride=stride,
    )

    print(f"{CYAN}[ReasoningConsistency] Clusters: {stats['clusters']}, "
          f"max size: {stats['max_cluster']}{RESET}")
    print(f"{DIM}  Shared CoT weight range: "
          f"{stats['weight_min']:.2f}-{stats['weight_max']:.2f}{RESET}")
    print(f"{DIM}  Sequences: {len(dataset):,}{RESET}")

    return dataset


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


def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
    """Learning rate scheduler with warmup and cosine decay."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


@dataclass
class TrainingState:
    step: int = 0
    epoch: int = 0
    best_loss: float = float('inf')
    total_tokens: int = 0


class Trainer:
    """Handles training loop with registry integration.

    ENHANCED with Reasoning-Consistent Training:
    - Opposite of GPT-2's passive diversity approach
    - Explicitly reinforces similar reasoning patterns
    - Contrastive loss for reasoning coherence
    """

    def __init__(self, model: MiniGPT, config: Config, tokenizer: BPETokenizer,
                 train_dataset: Dataset, device: torch.device,
                 model_name: str = "unnamed", model_desc: str = "",
                 preset: str = "custom", tags: list = None,
                 reasoning_consistency: bool = True,
                 contrastive_weight: float = 0.1):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.device = device
        self.model_name = model_name
        self.model_desc = model_desc
        self.preset = preset
        self.tags = tags or []
        self.state = TrainingState()
        self.start_time = None

        # Reasoning-Consistent Training (OPPOSITE of GPT-2's passive diversity)
        self.use_reasoning_consistency = reasoning_consistency and REASONING_CONSISTENCY_AVAILABLE
        self.contrastive_weight = contrastive_weight
        self.reasoning_manager = None
        self.contrastive_loss_fn = None

        if self.use_reasoning_consistency:
            print(f"{CYAN}[ReasoningConsistency] Enabled - reinforcing consistent reasoning patterns{RESET}")
            self.contrastive_loss_fn = ContrastiveReasoningLoss(weight=contrastive_weight)
            self.contrastive_loss_fn = self.contrastive_loss_fn.to(device)

        self._setup_training()

    def _setup_training(self):
        tc = self.config.training
        sampler = None
        if hasattr(self.train_dataset, "weights") and self.train_dataset.weights is not None:
            weights = self.train_dataset.weights
            if not torch.is_tensor(weights):
                weights = torch.tensor(weights, dtype=torch.float32)
            if len(weights) > 0 and (weights != 1.0).any():
                sampler = WeightedRandomSampler(
                    weights=weights,
                    num_samples=len(weights),
                    replacement=True
                )
                print(f"{DIM}  Using weighted sampling for shared CoT weights{RESET}")

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=tc.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=0,
            pin_memory=(self.device.type == 'cuda')
        )

        self.steps_per_epoch = len(self.train_loader)
        self.total_steps = self.steps_per_epoch * tc.epochs

        self.optimizer = AdamW(
            self.model.parameters(), lr=tc.learning_rate, weight_decay=tc.weight_decay
        )
        self.scheduler = get_lr_scheduler(
            self.optimizer, tc.warmup_steps, self.total_steps, tc.min_lr_ratio
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def _generate_sample(self, prompt: str = "<|user|>\nWrite Python to add two numbers.\n<|end_turn|>\n<|assistant|>\n",
                         max_tokens: int = 80) -> str:
        self.model.eval()
        output = self.model.generate(self.tokenizer, prompt=prompt, max_tokens=max_tokens,
                                     temperature=0.7, top_k=40, top_p=0.9, device=self.device)
        self.model.train()
        if output.startswith(prompt):
            output = output[len(prompt):].strip()
        return output

    def train(self) -> TrainingState:
        tc = self.config.training
        registry = get_registry()

        print(f"\n{BOLD}{CYAN}{'═' * 60}{RESET}")
        print(f"{BOLD}  Training: {self.model_name}{RESET}")
        if self.model_desc:
            print(f"{DIM}  {self.model_desc}{RESET}")
        print(f"{BOLD}{CYAN}{'═' * 60}{RESET}")
        print(f"{DIM}  Parameters: {self.model.get_num_params():,}{RESET}")
        print(f"{DIM}  Device: {self.device}{RESET}")
        print(f"{DIM}  Epochs: {tc.epochs} | Batches/epoch: {self.steps_per_epoch}{RESET}")
        print()

        self.model.train()
        self.start_time = time.time()
        running_loss = 0.0
        log_steps = 0

        for epoch in range(tc.epochs):
            self.state.epoch = epoch
            epoch_start = time.time()
            epoch_loss = 0.0
            epoch_tokens = 0

            for batch_idx, batch in enumerate(self.train_loader):
                if isinstance(batch, (list, tuple)) and len(batch) == 4:
                    x, y, cluster_ids, sample_weights = batch
                else:
                    x, y = batch
                    cluster_ids = None
                    sample_weights = None

                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                # Disable memory for training - each batch is independent, not a
                # continuation of the previous batch. Memory is for streaming inference.
                if hasattr(self.model, 'memory_manager'):
                    logits = self.model(x, use_memory=False, update_memory=False)
                else:
                    logits = self.model(x)

                # Standard language modeling loss
                if sample_weights is not None:
                    loss_per_token = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1),
                        ignore_index=0,
                        reduction="none"
                    ).view(y.size())
                    valid_mask = (y != 0).float()
                    token_counts = valid_mask.sum(dim=1).clamp(min=1)
                    per_sample_loss = (loss_per_token * valid_mask).sum(dim=1) / token_counts
                    lm_loss = (per_sample_loss * sample_weights.to(self.device)).mean()
                else:
                    lm_loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))

                # Reasoning Consistency Loss (OPPOSITE of GPT-2's passive diversity)
                # Instead of hoping reasoning emerges from random data,
                # we explicitly reinforce similar reasoning patterns
                if self.use_reasoning_consistency and self.contrastive_loss_fn is not None:
                    # Get hidden states from model (last layer before output projection)
                    # This requires model to expose hidden states
                    if hasattr(self.model, 'last_hidden_states'):
                        hidden_states = self.model.last_hidden_states
                        if cluster_ids is None:
                            # Fallback pseudo clustering when dataset lacks reasoning clusters
                            batch_size = x.size(0)
                            cluster_ids = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                            for i in range(batch_size):
                                cluster_ids[i] = x[i].sum().item() % 16  # 16 pseudo-clusters
                        else:
                            cluster_ids = cluster_ids.to(self.device)

                        contrastive_loss = self.contrastive_loss_fn(
                            hidden_states, cluster_ids
                        )
                        loss = lm_loss + contrastive_loss
                    else:
                        loss = lm_loss
                else:
                    loss = lm_loss

                loss.backward()

                if tc.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), tc.max_grad_norm)

                self.optimizer.step()
                self.scheduler.step()

                batch_tokens = (y != 0).sum().item()
                self.state.step += 1
                self.state.total_tokens += batch_tokens
                running_loss += loss.item()
                log_steps += 1
                epoch_loss += loss.item() * batch_tokens
                epoch_tokens += batch_tokens

                if self.state.step % tc.log_every_n_steps == 0:
                    avg_loss = running_loss / log_steps
                    elapsed = time.time() - self.start_time
                    tokens_per_sec = self.state.total_tokens / elapsed
                    lr = self.scheduler.get_last_lr()[0]

                    # Calculate ETA
                    progress = self.state.step / self.total_steps
                    if progress > 0:
                        eta = elapsed / progress - elapsed
                        eta_str = format_time(eta)
                    else:
                        eta_str = "?"

                    print(f"\r{DIM}Epoch {epoch + 1}/{tc.epochs} Step {self.state.step} | "
                          f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | Tok/s: {tokens_per_sec:.0f} | "
                          f"⏱ {format_time(elapsed)} | ETA: {eta_str}{RESET}",
                          end='', flush=True)

                    running_loss = 0.0
                    log_steps = 0

            # End of epoch
            epoch_time = time.time() - epoch_start
            epoch_avg_loss = epoch_loss / max(epoch_tokens, 1)

            print(f"\n{GREEN}Epoch {epoch + 1}/{tc.epochs} complete{RESET}")
            print(f"{DIM}  Loss: {epoch_avg_loss:.4f} | Time: {epoch_time:.1f}s{RESET}")

            # Update best loss
            if epoch_avg_loss < self.state.best_loss:
                self.state.best_loss = epoch_avg_loss

            # Sample generation on last epoch
            if epoch == tc.epochs - 1:
                sample = self._generate_sample()
                print(f"\n{CYAN}Sample output:{RESET}")
                print(f"{sample[:300]}")

        # Training complete - save to registry
        total_time = (time.time() - self.start_time) / 60

        print(f"\n{BOLD}{GREEN}{'═' * 60}{RESET}")
        print(f"{BOLD}{GREEN}  Training Complete!{RESET}")
        print(f"{BOLD}{GREEN}{'═' * 60}{RESET}")

        # Save to registry
        info = registry.save_model(
            name=self.model_name,
            description=self.model_desc,
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.config,
            epochs=tc.epochs,
            loss=self.state.best_loss,
            training_time=total_time,
            tags=self.tags,
            preset=self.preset
        )

        print(f"{GREEN}  Saved: {info.name}{RESET}")
        print(f"{DIM}  Parameters: {info.params:,}{RESET}")
        print(f"{DIM}  Final loss: {info.final_loss:.4f}{RESET}")
        print(f"{DIM}  Training time: {info.training_time_mins:.1f} min{RESET}")
        print(f"\n{DIM}  Use: python generate.py --model {info.name}{RESET}")

        return self.state


def create_datasets(corpus: str, tokenizer: BPETokenizer, config: Config) -> Dataset:
    """Create training dataset."""
    seq_len = config.max_seq_len
    stride = int(seq_len * config.data.stride_ratio)
    all_ids = tokenizer.encode(corpus, add_special=False)
    print(f"{DIM}  Tokens: {len(all_ids):,}{RESET}")
    dataset = TextDataset(all_ids, seq_len, stride)
    print(f"{DIM}  Sequences: {len(dataset):,}{RESET}")
    return dataset


def train_single_generation(model, tokenizer, corpus, config, device):
    """
    Train a single model generation. Used by IterativeTrainer.

    Args:
        model: Fresh model to train
        tokenizer: Tokenizer (already trained on corpus)
        corpus: Training corpus text
        config: Training config
        device: torch device

    Returns:
        Trained model
    """
    # Create dataset
    seq_len = config.max_seq_len
    stride = int(seq_len * config.data.stride_ratio)
    all_ids = tokenizer.encode(corpus, add_special=False)
    dataset = TextDataset(all_ids, seq_len, stride)

    # Setup training
    tc = config.training
    train_loader = DataLoader(
        dataset, batch_size=tc.batch_size, shuffle=True, num_workers=0,
        pin_memory=(device.type == 'cuda')
    )

    optimizer = AdamW(model.parameters(), lr=tc.learning_rate, weight_decay=tc.weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * tc.epochs
    scheduler = get_lr_scheduler(optimizer, tc.warmup_steps, total_steps, tc.min_lr_ratio)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Training loop
    model.train()
    for epoch in range(tc.epochs):
        epoch_loss = 0.0
        epoch_tokens = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            # Disable memory for training - each batch is independent
            if hasattr(model, 'memory_manager'):
                logits = model(x, use_memory=False, update_memory=False)
            else:
                logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()

            if tc.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc.max_grad_norm)

            optimizer.step()
            scheduler.step()

            batch_tokens = (y != 0).sum().item()
            epoch_loss += loss.item() * batch_tokens
            epoch_tokens += batch_tokens

        avg_loss = epoch_loss / max(epoch_tokens, 1)
        print(f"    Epoch {epoch + 1}/{tc.epochs} - Loss: {avg_loss:.4f}")

    return model


def backup_to_gcp(local_path: str, bucket_name: str, remote_path: str) -> bool:
    """
    Backup file to GCP Storage (optional, non-blocking).

    Returns True if successful, False otherwise. Never raises.
    """
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(remote_path)
        blob.upload_from_filename(local_path)
        print(f"{DIM}  [GCP] Backed up to gs://{bucket_name}/{remote_path}{RESET}")
        return True
    except ImportError:
        print(f"{DIM}  [GCP] google-cloud-storage not installed, skipping backup{RESET}")
        return False
    except Exception as e:
        print(f"{DIM}  [GCP] Backup failed (non-blocking): {e}{RESET}")
        return False


def backup_enhanced_data_gcp(pipeline, bucket_name: str) -> None:
    """Backup all enhanced data files to GCP (non-blocking)."""
    from pathlib import Path
    from datetime import datetime

    persist_dir = Path(pipeline.config.persist_dir)
    if not persist_dir.exists():
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    files_to_backup = [
        "enhanced_batches.jsonl",
        "stats.json",
        "iteration_history.json",
        "runs.jsonl",
        "gap_probe_results.jsonl"
    ]

    for filename in files_to_backup:
        local_path = persist_dir / filename
        if local_path.exists():
            remote_path = f"enhanced_data/{timestamp}/{filename}"
            backup_to_gcp(str(local_path), bucket_name, remote_path)


def run_multi_generation_training(
    args,
    initial_corpus: str,
    config: Config,
    device: torch.device,
    tags: list
) -> None:
    """
    Run multi-generation training with GPT-5.1-codex-mini enhancement loop.

    Each generation:
    1. Trains a fresh from-scratch model on current corpus
    2. Model generates responses to diverse prompts
    3. GPT-5.1-codex-mini enhances/corrects the responses
    4. Enhanced data is added to corpus for next generation
    """
    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{CYAN}  MULTI-GENERATION TRAINING (From Scratch){RESET}")
    print(f"{CYAN}{'='*60}{RESET}")
    print(f"{DIM}  Generations: {args.generations}{RESET}")
    print(f"{DIM}  Prompts per generation: {args.prompts_per_gen}{RESET}")
    print(f"{DIM}  Enhancement ratio: {args.pipeline_enhance_ratio}{RESET}")

    # Initialize pipeline
    pipeline_config = PipelineConfig(
        enhancement_ratio=args.pipeline_enhance_ratio,
        verification_threshold=0.8,
        prompts_per_category=args.prompts_per_gen // 9 + 1  # 9 categories
    )

    # Model creator function
    def model_creator(cfg, dev):
        model_cfg = ModelConfig(
            vocab_size=cfg.vocab_size, max_seq_len=cfg.max_seq_len,
            embed_dim=cfg.embed_dim, num_heads=cfg.num_heads,
            num_layers=cfg.num_layers, ff_dim=cfg.ff_dim, dropout=cfg.dropout,
            use_infini_attention=getattr(cfg, 'use_infini_attention', False),
            segment_size=getattr(cfg, 'segment_size', 128),
            use_delta_rule=getattr(cfg, 'use_delta_rule', True),
            use_rope=getattr(cfg, 'use_rope', False)
        )
        return create_model(model_cfg, dev)

    # Run iterative trainer
    iterative_trainer = IterativeTrainer(
        base_config=config,
        pipeline_config=pipeline_config,
        max_iterations=args.generations
    )

    history = iterative_trainer.run_full_pipeline(
        initial_corpus=initial_corpus,
        tokenizer_class=BPETokenizer,
        model_creator=model_creator,
        trainer_func=train_single_generation,
        device=device,
        prompts_per_generation=args.prompts_per_gen
    )

    # Save final model to registry
    final_model = iterative_trainer.get_latest_model()
    if final_model:
        registry = get_registry()

        # Get final tokenizer
        tokenizer = BPETokenizer()
        final_corpus_size = history[-1]["corpus_size"] if history else len(initial_corpus)
        tokenizer.train(initial_corpus, config.vocab_size)  # Use base corpus for tokenizer

        tags.append(f"gen-{args.generations}")
        tags.append("gpt52-enhanced")

        info = registry.save_model(
            name=args.name,
            description=f"{args.desc} (Gen {args.generations})" if args.desc else f"Generation {args.generations} model",
            model=final_model,
            tokenizer=tokenizer,
            config=config,
            epochs=config.training.epochs * args.generations,
            loss=0.0,  # Would need to track this properly
            training_time=0.0,
            tags=tags,
            preset=args.preset
        )

        print(f"\n{GREEN}{'='*60}{RESET}")
        print(f"{GREEN}  Multi-Generation Training Complete!{RESET}")
        print(f"{GREEN}{'='*60}{RESET}")
        print(f"{GREEN}  Final model saved: {info.name}{RESET}")
        print(f"{DIM}  Generations trained: {args.generations}{RESET}")
        print(f"{DIM}  Total enhanced examples: {history[-1]['total_enhanced'] if history else 0}{RESET}")

    # Print pipeline stats
    iterative_trainer.pipeline.print_statistics()


def print_models_table():
    """Print a formatted table of all models."""
    models = list_models()

    if not models:
        print(f"{YELLOW}No models found. Train one with: python train.py --name my-model{RESET}")
        return

    print(f"\n{BOLD}{'═' * 70}{RESET}")
    print(f"{BOLD}  Saved Models{RESET}")
    print(f"{BOLD}{'═' * 70}{RESET}")

    # Header
    print(f"{DIM}{'Name':<20} {'Params':<12} {'Loss':<10} {'Preset':<10} {'Description'}{RESET}")
    print(f"{DIM}{'─' * 70}{RESET}")

    for m in sorted(models, key=lambda x: x.created, reverse=True):
        params_str = f"{m.params / 1e6:.1f}M" if m.params > 0 else "?"
        loss_str = f"{m.final_loss:.4f}" if m.final_loss > 0 else "?"
        desc = (m.description[:25] + "...") if len(m.description) > 28 else m.description

        print(f"{CYAN}{m.name:<20}{RESET} {params_str:<12} {loss_str:<10} {m.preset:<10} {DIM}{desc}{RESET}")

    print(f"{DIM}{'─' * 70}{RESET}")
    print(f"{DIM}Total: {len(models)} model(s){RESET}\n")


def main():
    parser = argparse.ArgumentParser(description="Train and manage models")
    parser.add_argument("--name", type=str, default="erosolar", help="Model name (default: erosolar - canonical model that gets replaced)")
    parser.add_argument("--desc", type=str, default="", help="Model description")
    parser.add_argument("--preset", type=str, choices=[
        "tiny-1", "tiny-2", "small-1", "small-2", "medium-1", "medium-2", "medium-plus", "large-1",
        "infini-tiny", "infini-small", "infini-medium", "infini-large", "infini-huawei"
    ], default="tiny-1", help="Model preset (infini-* presets use Infini-attention for infinite context)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tags", type=str, default="", help="Comma-separated tags")
    parser.add_argument("--list", action="store_true", help="List all saved models")
    parser.add_argument("--focused", action="store_true", help="Use focused training data (minimal, high-priority only)")
    parser.add_argument("--balanced", action="store_true", help="Balanced general-purpose model (less base data, more optimal data)")
    parser.add_argument("--upgrade-pipeline", action="store_true",
                        help="Enable GPT-5.1-codex-mini training data upgrade pipeline")
    parser.add_argument("--pipeline-enhance-ratio", type=float, default=0.3,
                        help="Fraction of data to enhance with GPT-5.1-codex-mini (default: 0.3)")
    parser.add_argument("--pipeline-iterations", type=int, default=1,
                        help="Number of generate->enhance iterations (default: 1)")
    parser.add_argument("--generations", type=int, default=1,
                        help="Number of model generations to train (iterative improvement via GPT-5.1-codex-mini)")
    parser.add_argument("--prompts-per-gen", type=int, default=100,
                        help="Prompts to generate per generation for GPT-5.1-codex-mini enhancement")
    parser.add_argument("--gcp-backup", action="store_true",
                        help="Backup enhanced data to GCP (optional, non-blocking)")
    parser.add_argument("--gcp-bucket", type=str, default="erosolar-training-data",
                        help="GCP bucket name for backups")
    # Curriculum learning and reasoning-weighted loss
    parser.add_argument("--curriculum", action="store_true",
                        help="Use multi-phase curriculum learning (foundation -> bridge -> reasoning)")
    parser.add_argument("--reasoning-weight", type=float, default=1.0,
                        help="Weight for thinking tokens in loss (default: 1.0x)")
    parser.add_argument("--foundations-path", type=str, default="cache/foundations/foundational_knowledge.jsonl",
                        help="Path to foundational knowledge JSONL")
    parser.add_argument("--foundations-cot-path", type=str, default="cache/foundations/foundational_cot.jsonl",
                        help="Path to foundational CoT JSONL")
    # Reasoning-Consistent Training (OPPOSITE of GPT-2's passive diversity)
    parser.add_argument("--reasoning-consistency", action="store_true", default=True,
                        help="Enable reasoning-consistent training (reinforces similar reasoning patterns)")
    parser.add_argument("--no-reasoning-consistency", dest="reasoning_consistency", action="store_false",
                        help="Disable reasoning-consistent training")
    parser.add_argument("--contrastive-weight", type=float, default=0.1,
                        help="Weight for contrastive reasoning loss (default: 0.1)")
    parser.add_argument("--shared-cot-weights", action="store_true", default=False,
                        help="Dynamically boost shared CoT patterns during training")
    parser.add_argument("--no-shared-cot-weights", dest="shared_cot_weights", action="store_false",
                        help="Disable shared CoT weighting")
    parser.add_argument("--shared-cot-boost", type=float, default=0.5,
                        help="Cluster size boost for shared CoT weighting (default: 0.5)")
    parser.add_argument("--bridge-path", type=str, default="cache/bridge/bridge_data.jsonl",
                        help="Path to bridge data JSONL")
    parser.add_argument("--bridge-cot-path", type=str, default="cache/bridge/bridge_cot.jsonl",
                        help="Path to bridge CoT JSONL")
    parser.add_argument("--reasoning-path", type=str, default="cache/reasoning/reasoning_chains.jsonl",
                        help="Path to reasoning chains JSONL")
    parser.add_argument("--cot-path", type=str, default="cache/cot/cot_training_data.jsonl",
                        help="Path to chain-of-thought data JSONL")
    parser.add_argument("--optimal-weight", type=float, default=1.0,
                        help="Repetition weight for optimal_training.jsonl (default: 1x)")
    parser.add_argument("--jsonl-only", action="store_true",
                        help="FAST MODE: Only use generated_training_data.jsonl, skip all base data")
    args = parser.parse_args()

    if FORCE_TRAINING_WEIGHTS_1X:
        if args.reasoning_weight != 1.0:
            print(f"{YELLOW}Forcing reasoning weight to 1.0 (all training data weights locked to 1x){RESET}")
        args.reasoning_weight = 1.0
        args.shared_cot_weights = False
        args.shared_cot_boost = 1.0
        args.optimal_weight = 1.0

    # List models mode
    if args.list:
        print_models_table()
        return

    # Check if canonical model exists - overwrite silently (this is the upgrade pattern)
    registry = get_registry()
    if registry.exists(args.name):
        if args.name == "erosolar":
            # Canonical model - always overwrite without prompt
            print(f"{CYAN}Replacing canonical model: {args.name}{RESET}")
        else:
            print(f"{YELLOW}Warning: Model '{args.name}' already exists and will be overwritten.{RESET}")
            response = input(f"{DIM}Continue? [y/N]: {RESET}").strip().lower()
            if response != 'y':
                print("Aborted.")
                return
        registry.delete(args.name)

    print(f"\n{BOLD}{CYAN}Erosolar - Training{RESET}")
    print(f"{DIM}{'─' * 40}{RESET}")

    # Setup
    device = setup_device(args.seed)
    config = get_preset(args.preset)
    if args.epochs:
        config.training.epochs = args.epochs

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    # Training Data Upgrade Pipeline (GPT-5.1-codex-mini enhancement)
    pipeline = None
    if args.upgrade_pipeline:
        if not PIPELINE_AVAILABLE:
            print(f"{RED}Error: Training upgrade pipeline not available{RESET}")
            print(f"{DIM}Install openai package: pip install openai{RESET}")
            return

        print(f"\n{CYAN}{'═' * 60}{RESET}")
        print(f"{CYAN}  Training Data Upgrade Pipeline{RESET}")
        print(f"{CYAN}{'═' * 60}{RESET}")
        print(f"{DIM}  Enhancement ratio: {args.pipeline_enhance_ratio}{RESET}")
        print(f"{DIM}  Iterations: {args.pipeline_iterations}{RESET}")

        pipeline_config = PipelineConfig(
            enhancement_ratio=args.pipeline_enhance_ratio,
            verification_threshold=0.8
        )
        pipeline = initialize_pipeline(pipeline_config)

        # For now, just mark that pipeline is active
        # Full integration requires trained model to generate, then enhance
        tags.append("upgrade-pipeline")
        print(f"{GREEN}  Pipeline initialized and ready{RESET}")
        print(f"{DIM}  After training, use pipeline to generate enhanced data for next iteration{RESET}")

    # Data
    if args.jsonl_only:
        print(f"\n{CYAN}FAST MODE: Using only generated_training_data.jsonl{RESET}")
        tags.append("jsonl-only")
        corpus = ""
    elif args.focused:
        print(f"\n{YELLOW}Using FOCUSED training mode (minimal data){RESET}")
    elif args.balanced:
        print(f"\n{CYAN}Using BALANCED training mode (general-purpose){RESET}")
        print(f"{DIM}  Reduced base data, prioritizing optimal API-generated data{RESET}")
        tags.append("balanced")

    if args.jsonl_only:
        # Skip all base data loading
        pass
    elif pipeline and not args.focused and not args.balanced:
        print(f"\n{DIM}Generating training corpus...{RESET}")
        print(f"{DIM}  Using gap-targeted repository data from upgrade pipeline{RESET}")
        try:
            corpus = pipeline.export_targeted_corpus()
            tags.append("gap-targeted")
        except Exception as e:
            print(f"{YELLOW}  Gap targeting unavailable: {e}{RESET}")
            print(f"{DIM}  Falling back to unified repository data{RESET}")
            corpus = pipeline.export_repository_corpus()
            tags.append("repo-data")
    else:
        print(f"\n{DIM}Generating training corpus...{RESET}")
        # Balanced mode: skip base data entirely, use only JSONL
        base_examples = 0 if args.balanced else 50000
        if base_examples > 0:
            corpus = create_training_corpus(num_examples=base_examples, focused=args.focused, balanced=args.balanced)
            print(f"{DIM}  Base data examples: {base_examples:,}{RESET}")
        else:
            corpus = ""
            print(f"{CYAN}  JSONL-only mode: skipping base data{RESET}")

    sample_texts: List[str] = []
    sample_base_weights: List[float] = []
    use_shared_cot_weights = (
        (not FORCE_TRAINING_WEIGHTS_1X)
        and args.shared_cot_weights
        and args.reasoning_consistency
        and REASONING_CONSISTENCY_AVAILABLE
        and args.balanced
    )
    if FORCE_TRAINING_WEIGHTS_1X and args.shared_cot_weights:
        print(f"{YELLOW}  Shared CoT weights disabled (all training data weights locked to 1x){RESET}")
    elif args.shared_cot_weights and not REASONING_CONSISTENCY_AVAILABLE:
        print(f"{YELLOW}  Shared CoT weights requested but reasoning_consistency unavailable{RESET}")
    elif args.shared_cot_weights and not args.balanced:
        print(f"{YELLOW}  Shared CoT weights require JSONL-only balanced mode; ignoring flag{RESET}")
    elif use_shared_cot_weights:
        print(f"{CYAN}  Shared CoT weights enabled (cluster boost {args.shared_cot_boost}){RESET}")

    # ALWAYS include optimal_training.jsonl if it exists (API-generated high-quality data)
    # Skip if --jsonl-only mode (only use generated_training_data.jsonl)
    optimal_path = Path("cache/optimal_gen/optimal_training.jsonl")
    if not optimal_path.exists():
        optimal_path = Path("cache/optimal_gen/optimal_training.original.jsonl")
    if optimal_path.exists() and not args.jsonl_only:
        print(f"\n{CYAN}Loading optimal training data (GPT-5.1 generated)...{RESET}")
        print(f"{DIM}  Source: {optimal_path}{RESET}")
        optimal_pairs = []
        total_lines = 0
        parse_errors = 0

        MIN_SCORE = 0.9  # Quality threshold (lowered from 0.93 for more training data)
        skipped_low_score = 0

        with open(optimal_path) as f:
            for line in f:
                total_lines += 1
                try:
                    record = json.loads(line)
                    # Check composite score if present
                    metadata = record.get("metadata", {})
                    score = metadata.get("composite_score", 1.0)  # Default to 1.0 if no score
                    if score < MIN_SCORE:
                        skipped_low_score += 1
                        continue

                    msgs = record.get("messages", [])
                    if len(msgs) >= 2:
                        prompt = msgs[0].get("content", "")
                        response = msgs[1].get("content", "")
                        if prompt and response:
                            pair = f"<|user|>\n{prompt}\n<|end_turn|>\n<|assistant|>\n{response}\n<|end_turn|>"
                            optimal_pairs.append(pair)
                except Exception as e:
                    parse_errors += 1
                    if parse_errors <= 3:
                        print(f"{YELLOW}  Parse error line {total_lines}: {e}{RESET}")

        print(f"{DIM}  JSONL lines read: {total_lines:,}{RESET}")
        print(f"{DIM}  Valid pairs (score >= {MIN_SCORE}): {len(optimal_pairs):,}{RESET}")
        if skipped_low_score > 0:
            print(f"{YELLOW}  Skipped (score < {MIN_SCORE}): {skipped_low_score}{RESET}")
        if parse_errors > 0:
            print(f"{YELLOW}  Parse errors: {parse_errors}{RESET}")

        if optimal_pairs:
            # All training data weights are forced to 1x
            weight = 1
            if use_shared_cot_weights:
                sample_texts.extend(optimal_pairs)
                sample_base_weights.extend([weight] * len(optimal_pairs))
                optimal_corpus = "\n\n\n".join(optimal_pairs)
                corpus = corpus + "\n\n\n" + optimal_corpus
            else:
                optimal_corpus = "\n\n\n".join(optimal_pairs * weight)
                corpus = corpus + "\n\n\n" + optimal_corpus

            print(f"{GREEN}  ✓ {len(optimal_pairs):,} optimal pairs (unique){RESET}")
            print(f"{GREEN}  ✓ Optimal data chars: {len(optimal_corpus):,}{RESET}")
            tags.append("optimal-data")
        else:
            print(f"{YELLOW}  No valid pairs found in JSONL{RESET}")
    else:
        print(f"{YELLOW}  No optimal_training.jsonl found at {optimal_path}{RESET}")
        print(f"{DIM}  Run './upgrade_and_serve.sh --gen-only' to generate{RESET}")

    # ALSO load upgraded base pairs if available (check both paths)
    # Skip if --jsonl-only mode
    upgraded_paths = [
        Path("cache/upgraded_base/upgraded_pairs.jsonl"),
        Path("cache/upgraded_base/upgraded_data.jsonl")
    ]
    upgraded_path = None
    if not args.jsonl_only:
        for p in upgraded_paths:
            if p.exists():
                upgraded_path = p
                break

    if upgraded_path and not args.jsonl_only:
        print(f"\n{CYAN}Loading upgraded base pairs...{RESET}")
        upgraded_pairs = []
        skipped_base = 0
        MIN_SCORE = 0.9  # Quality threshold

        with open(upgraded_path) as f:
            for line in f:
                try:
                    record = json.loads(line)
                    # Check score if present
                    metadata = record.get("metadata", {})
                    score = metadata.get("composite_score", 1.0)
                    if score < MIN_SCORE:
                        skipped_base += 1
                        continue

                    msgs = record.get("messages", [])
                    if len(msgs) >= 2:
                        prompt = msgs[0].get("content", "")
                        response = msgs[1].get("content", "")
                        if prompt and response:
                            upgraded_pairs.append(f"<|user|>\n{prompt}\n<|end_turn|>\n<|assistant|>\n{response}\n<|end_turn|>")
                except:
                    pass

        if skipped_base > 0:
            print(f"{YELLOW}  Skipped (score < {MIN_SCORE}): {skipped_base}{RESET}")

        if upgraded_pairs:
            # 1x weight for upgraded base pairs (was 5x, reduced for speed)
            weight = 1.0
            if use_shared_cot_weights:
                sample_texts.extend(upgraded_pairs)
                sample_base_weights.extend([weight] * len(upgraded_pairs))
                upgraded_corpus = "\n\n\n".join(upgraded_pairs)
            else:
                upgraded_corpus = "\n\n\n".join(upgraded_pairs)
            corpus = corpus + "\n\n\n" + upgraded_corpus
            print(f"{GREEN}  ✓ {len(upgraded_pairs)} upgraded pairs (unique){RESET}")
            tags.append("upgraded-base")

    # CRITICAL: Load generated_training_data.jsonl (mini loser-targeted generation)
    # This is the PRIMARY training data source from the self-improvement loop
    generated_path = Path("data_store/generated_training_data.jsonl")

    # Ensure main file exists - merge from chunks if needed
    if not generated_path.exists() and TRAINING_DATA_MANAGEMENT:
        print(f"{CYAN}Main JSONL missing, attempting to merge from chunks...{RESET}")
        ensure_training_data()

    if generated_path.exists():
        print(f"\n{CYAN}Loading generated training data (loser-targeted)...{RESET}")
        print(f"{DIM}  Source: {generated_path}{RESET}")
        generated_pairs = []
        gen_lines = 0
        gen_errors = 0

        with open(generated_path) as f:
            for line in f:
                gen_lines += 1
                try:
                    record = json.loads(line)
                    msgs = record.get("messages", [])
                    if len(msgs) >= 2:
                        prompt = msgs[0].get("content", "")
                        response = msgs[1].get("content", "")
                        if prompt and response:
                            pair = f"<|user|>\n{prompt}\n<|end_turn|>\n<|assistant|>\n{response}\n<|end_turn|>"
                            generated_pairs.append(pair)
                except Exception as e:
                    gen_errors += 1
                    if gen_errors <= 3:
                        print(f"{YELLOW}  Parse error line {gen_lines}: {e}{RESET}")

        print(f"{DIM}  JSONL lines read: {gen_lines:,}{RESET}")
        print(f"{DIM}  Valid pairs: {len(generated_pairs):,}{RESET}")
        if gen_errors > 0:
            print(f"{YELLOW}  Parse errors: {gen_errors}{RESET}")

        if generated_pairs:
            weight = 1.0
            if use_shared_cot_weights:
                sample_texts.extend(generated_pairs)
                sample_base_weights.extend([weight] * len(generated_pairs))
                generated_corpus = "\n\n\n".join(generated_pairs)
            else:
                generated_corpus = "\n\n\n".join(generated_pairs)
            corpus = corpus + "\n\n\n" + generated_corpus
            print(f"{GREEN}  ✓ {len(generated_pairs):,} loser-targeted pairs added{RESET}")
            print(f"{GREEN}  ✓ Generated data chars: {len(generated_corpus):,}{RESET}")
            tags.append("loser-targeted")
        else:
            print(f"{YELLOW}  No valid pairs found in generated data{RESET}")
    else:
        print(f"\n{DIM}No generated_training_data.jsonl found - run mini --auto first{RESET}")

    tokenizer_corpus = corpus
    if use_shared_cot_weights and sample_texts:
        tokenizer_corpus = "\n\n\n".join(sample_texts)
    print(f"{DIM}  Corpus: {len(tokenizer_corpus):,} chars{RESET}")

    # Multi-generation training path (from-scratch with GPT-5.1-codex-mini enhancement loop)
    if args.generations > 1:
        if not PIPELINE_AVAILABLE:
            print(f"{RED}Error: Multi-generation training requires upgrade pipeline{RESET}")
            print(f"{DIM}Install openai package: pip install openai{RESET}")
            return

        run_multi_generation_training(args, corpus, config, device, tags)

        # Optional GCP backup
        if args.gcp_backup:
            print(f"\n{DIM}Backing up enhanced data to GCP...{RESET}")
            pipeline = get_pipeline()
            backup_enhanced_data_gcp(pipeline, args.gcp_bucket)

        return

    # Single generation training path (standard)
    # Tokenizer
    print(f"\n{DIM}Preparing tokenizer...{RESET}")
    tokenizer = BPETokenizer()
    tokenizer_cache = Path("cache/tokenizer")
    if tokenizer_cache.exists():
        tokenizer.load(tokenizer_cache)
        print(f"{DIM}  Loaded cached tokenizer{RESET}")
    else:
        tokenizer.train(tokenizer_corpus, config.vocab_size)
        tokenizer.save(tokenizer_cache)

    config.vocab_size = tokenizer.vocab_size

    # Dataset
    print(f"\n{DIM}Creating dataset...{RESET}")
    if use_shared_cot_weights and sample_texts:
        dataset = create_reasoning_dataset(
            sample_texts,
            sample_base_weights,
            tokenizer,
            config,
            cluster_boost=args.shared_cot_boost,
        )
        tags.append("shared-cot-weights")
    else:
        dataset = create_datasets(corpus, tokenizer, config)

    # Validate dataset has enough data
    if len(dataset) == 0:
        print(f"\n{RED}{'═' * 60}{RESET}")
        print(f"{RED}  ERROR: Dataset has no sequences!{RESET}")
        print(f"{RED}{'═' * 60}{RESET}")
        print(f"{DIM}  Training data is too small to create sequences.{RESET}")
        print(f"{DIM}  Minimum tokens needed: {config.max_seq_len + 2}{RESET}")
        corpus_tokens = len(tokenizer.encode(corpus, add_special=False))
        print(f"{DIM}  Corpus tokens: {corpus_tokens}{RESET}")
        print(f"\n{YELLOW}Solutions:{RESET}")
        print(f"{DIM}  1. Generate more training data first{RESET}")
        print(f"{DIM}  2. Lower the sequence length (--preset tiny-1){RESET}")
        print(f"{DIM}  3. Check that data files exist and contain valid data{RESET}")
        return

    # Model
    print(f"\n{DIM}Creating model...{RESET}")
    model_config = ModelConfig(
        vocab_size=config.vocab_size, max_seq_len=config.max_seq_len,
        embed_dim=config.embed_dim, num_heads=config.num_heads,
        num_layers=config.num_layers, ff_dim=config.ff_dim, dropout=config.dropout,
        use_infini_attention=getattr(config, 'use_infini_attention', False),
        segment_size=getattr(config, 'segment_size', 128),
        use_delta_rule=getattr(config, 'use_delta_rule', True),
        use_rope=getattr(config, 'use_rope', False)
    )
    model = create_model(model_config, device)

    # Print architecture info
    arch_str = f"{config.num_layers}L-{config.num_heads}H-{config.embed_dim}D"
    if model_config.use_infini_attention:
        arch_str += f" [Infini-attention, segment={model_config.segment_size}]"
        print(f"{DIM}  Architecture: {arch_str}{RESET}")
        if hasattr(model, 'get_compression_ratio'):
            compression = model.get_compression_ratio(config.max_seq_len)
            print(f"{DIM}  Memory compression: {compression:.1f}x at {config.max_seq_len} tokens{RESET}")
    else:
        print(f"{DIM}  Architecture: {arch_str}{RESET}")

    # Train
    if args.curriculum:
        # Curriculum Learning Mode: Phase-based training
        print(f"\n{BOLD}{MAGENTA}{'═' * 60}{RESET}")
        print(f"{BOLD}{MAGENTA}  CURRICULUM LEARNING MODE{RESET}")
        print(f"{MAGENTA}{'═' * 60}{RESET}")
        print(f"{DIM}  Phase 1: Foundational Knowledge{RESET}")
        print(f"{DIM}  Phase 2: Foundational CoT{RESET}")
        print(f"{DIM}  Phase 3: Bridge Data{RESET}")
        print(f"{DIM}  Phase 4: Bridge CoT{RESET}")
        print(f"{DIM}  Phase 5: Reasoning Chains{RESET}")
        print(f"{DIM}  Phase 6: Chain-of-Thought{RESET}")
        print(f"{DIM}  Thinking weight: {args.reasoning_weight}x{RESET}")

        # Define curriculum phases
        # GENERAL-PURPOSE CORE FIRST: Foundations and Bridge get higher weights
        # Reasoning and Optimal come after with normal weights
        # This ensures the model learns general concepts before specialized reasoning

        phases = [
            # Conversational - FIRST to ensure model can handle basic greetings
            CurriculumPhase(
                name="conversational",
                data_path=Path("cache/foundations/conversational.jsonl"),
                epochs=3,  # Train multiple times on small set
                description="Basic conversational patterns (hi, hello, thanks)",
                weight=1.0
            ),
            CurriculumPhase(
                name="foundation",
                data_path=Path(args.foundations_path),
                epochs=1,
                description="General-purpose core facts and definitions",
                weight=1.0
            ),
            CurriculumPhase(
                name="foundation_cot",
                data_path=Path(args.foundations_cot_path),
                epochs=1,
                description="Foundational concepts with concise CoT",
                weight=1.0
            ),
            CurriculumPhase(
                name="bridge",
                data_path=Path(args.bridge_path),
                epochs=1,
                description="Connecting foundations to reasoning",
                weight=1.0
            ),
            CurriculumPhase(
                name="bridge_cot",
                data_path=Path(args.bridge_cot_path),
                epochs=1,
                description="Bridge data with multi-step CoT",
                weight=1.0
            ),
            CurriculumPhase(
                name="reasoning",
                data_path=Path(args.reasoning_path),
                epochs=1,
                description="Sophisticated reasoning chains",
                weight=1.0  # Normal - comes after core
            ),
            CurriculumPhase(
                name="cot",
                data_path=Path(args.cot_path),
                epochs=1,
                description="Chain-of-thought training data",
                weight=1.0  # Normal - comes after core
            ),
        ]

        # Add gap-targeted data if available (HIGH priority - targets model weaknesses)
        # This is from training_upgrade_pipeline.py weakness detection
        gap_cot_path = Path("cache/gap_targeted/gap_cot.jsonl")
        gap_raw_path = Path("cache/gap_targeted/gap_training.jsonl")
        gap_path = gap_cot_path if gap_cot_path.exists() else gap_raw_path
        if gap_path.exists():
            phases.append(CurriculumPhase(
                name="gap_targeted",
                data_path=gap_path,
                epochs=1,
                description="Weakness-targeted training (CoT)",
                weight=1.0
            ))
            cot_tag = " (CoT)" if gap_cot_path.exists() else ""
            print(f"{GREEN}  Using gap-targeted data{cot_tag} (weight: 1.0x){RESET}")

        # Add RLHF-verified data if available (HIGHEST quality - verified correct)
        rlhf_path = Path("cache/rlhf/high_quality.jsonl")
        if rlhf_path.exists():
            phases.append(CurriculumPhase(
                name="rlhf_verified",
                data_path=rlhf_path,
                epochs=1,
                description="RLHF-verified high quality data",
                weight=1.0
            ))
            print(f"{GREEN}  Using RLHF-verified data (weight: 1.0x){RESET}")

        # Add enhanced data if available (GPT-enhanced with lower weight)
        enhanced_path = Path("cache/enhanced/enhanced_training.jsonl")
        if enhanced_path.exists():
            phases.append(CurriculumPhase(
                name="enhanced",
                data_path=enhanced_path,
                epochs=1,
                description="GPT-enhanced training data",
                weight=1.0
            ))
            print(f"{GREEN}  Using GPT-enhanced data (weight: 1.0x){RESET}")

        # Add conversational and honest_safety again at the END to reinforce basic patterns
        # (Last phase often dominates model behavior)
        conv_path = Path("cache/foundations/conversational.jsonl")
        if conv_path.exists():
            phases.append(CurriculumPhase(
                name="conversational_final",
                data_path=conv_path,
                epochs=2,
                description="Reinforcing conversational patterns",
                weight=1.0
            ))


        curriculum_trainer = CurriculumTrainer(
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=device,
            phases=phases,
            thinking_weight=args.reasoning_weight
        )
        final_loss = curriculum_trainer.train()

        # Save to registry after curriculum training
        registry = get_registry()
        tags.append("curriculum")
        tags.append(f"thinking-weight-{args.reasoning_weight}")

        info = registry.save_model(
            name=args.name,
            description=args.desc or "Curriculum-trained with reasoning-weighted loss",
            model=model,
            tokenizer=tokenizer,
            config=config,
            epochs=sum(p.epochs for p in phases),
            loss=final_loss,
            training_time=(time.time() - time.time()) / 60,  # placeholder
            tags=tags,
            preset=args.preset
        )
        print(f"\n{GREEN}Saved curriculum-trained model: {info.name}{RESET}")

    else:
        # Standard training
        trainer = Trainer(
            model=model, config=config, tokenizer=tokenizer, train_dataset=dataset,
            device=device, model_name=args.name, model_desc=args.desc,
            preset=args.preset, tags=tags
        )
        trainer.train()

    # Post-training: Run upgrade pipeline iterations if enabled
    if args.upgrade_pipeline and PIPELINE_AVAILABLE and args.pipeline_iterations > 0:
        print(f"\n{CYAN}{'═' * 60}{RESET}")
        print(f"{CYAN}  Running Training Data Upgrade Pipeline{RESET}")
        print(f"{CYAN}{'═' * 60}{RESET}")

        for iteration in range(args.pipeline_iterations):
            print(f"\n{DIM}Iteration {iteration + 1}/{args.pipeline_iterations}{RESET}")

            # Get pipeline
            pipeline = get_pipeline()

            # Generate enhanced data from trained model
            print(f"{DIM}  Generating responses from trained model...{RESET}")
            sample_prompts = [
                "Write a Python function to calculate factorial",
                "Explain how binary search works",
                "What is the difference between a list and a tuple?",
                "How do you reverse a string in Python?",
                "Explain recursion with an example",
            ]

            # Generate from model
            generated = pipeline.generate_from_model(
                model, tokenizer, sample_prompts, device,
                max_tokens=256, temperature=0.8
            )

            # Enhance with GPT-5.1-codex-mini
            print(f"{DIM}  Enhancing with GPT-5.1-codex-mini...{RESET}")
            try:
                enhanced = pipeline.enhance_training_data(generated)
                print(f"{GREEN}  Enhanced {len(enhanced)} examples{RESET}")

                # Export for next iteration
                output_path = f"cache/pipeline/enhanced_iter_{iteration + 1}.txt"
                pipeline.export_training_corpus(enhanced, output_path)
                print(f"{DIM}  Saved to {output_path}{RESET}")

            except Exception as e:
                print(f"{YELLOW}  Enhancement skipped: {e}{RESET}")
                print(f"{DIM}  Set OPENAI_API_KEY to enable GPT-5.1-codex-mini enhancement{RESET}")

        # Print pipeline statistics
        pipeline.print_statistics()

        # Optional GCP backup for single-generation with pipeline
        if args.gcp_backup:
            print(f"\n{DIM}Backing up enhanced data to GCP...{RESET}")
            backup_enhanced_data_gcp(pipeline, args.gcp_bucket)


if __name__ == "__main__":
    main()
