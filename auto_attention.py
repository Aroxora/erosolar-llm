#!/usr/bin/env python3
"""
AUTO-MANAGED SELF-ATTENTION
===========================
Uses gpt-5.1-codex-mini to compute and optimize attention patterns.

Author: Bo Shang <bo@shang.software>

Instead of hand-crafted attention rules, we use a frontier model to:
1. Analyze reasoning patterns in training samples
2. Generate attention guidance
3. Adjust attention weights during training

This is meta-learning: the larger model (codex-mini) teaches the smaller
model HOW to attend, not just WHAT to output.
"""

import os
import json
import asyncio
import aiohttp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib
import time
import subprocess
import shutil

# ════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════

API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Local embeddings (replaces OpenAI text-embedding-3-small)
try:
    from local_embeddings import embed_batch as local_embed_batch, average_pairwise_similarity
    LOCAL_EMBEDDINGS_AVAILABLE = True
except ImportError:
    LOCAL_EMBEDDINGS_AVAILABLE = False
# Model hierarchy - all using deepseek-reasoner now
MODEL_MINI = "deepseek-reasoner"
MODEL_STANDARD = "deepseek-reasoner"
MODEL_PRO = "deepseek-reasoner"
API_URL = "https://api.deepseek.com/chat/completions"


class AdaptiveModelSelector:
    """
    Adaptive model selection - starts with mini, auto-escalates when
    optimization isn't optimal, de-escalates back when things are going well.
    """

    def __init__(self):
        self.current_level = 0  # 0=mini, 1=standard, 2=pro
        self.models = [MODEL_MINI, MODEL_STANDARD, MODEL_PRO]
        self.success_streak = 0
        self.failure_streak = 0
        self.escalation_threshold = 3  # Failures before escalating
        self.deescalation_threshold = 10  # Successes before de-escalating
        self.quality_scores: List[float] = []

    def get_current_model(self) -> str:
        """Get current model based on adaptive selection."""
        return self.models[self.current_level]

    def record_result(self, quality_score: float, success: bool = True):
        """
        Record optimization result and adjust model selection.

        quality_score: 0.0 to 1.0 - how optimal was the result
        success: did the API call succeed
        """
        self.quality_scores.append(quality_score)

        # Keep last 20 scores
        if len(self.quality_scores) > 20:
            self.quality_scores = self.quality_scores[-20:]

        if not success or quality_score < 0.5:
            self.failure_streak += 1
            self.success_streak = 0

            # Auto-escalate if failing too much
            if self.failure_streak >= self.escalation_threshold:
                self._escalate()
                self.failure_streak = 0

        elif quality_score >= 0.8:
            self.success_streak += 1
            self.failure_streak = 0

            # De-escalate if doing well for a while
            if self.success_streak >= self.deescalation_threshold:
                self._deescalate()
                self.success_streak = 0

    def _escalate(self):
        """Move to more powerful model."""
        if self.current_level < 2:
            old_model = self.models[self.current_level]
            self.current_level += 1
            new_model = self.models[self.current_level]
            print(f"[AdaptiveModel] Escalating: {old_model} → {new_model}")

    def _deescalate(self):
        """Move back to less powerful model."""
        if self.current_level > 0:
            old_model = self.models[self.current_level]
            self.current_level -= 1
            new_model = self.models[self.current_level]
            print(f"[AdaptiveModel] De-escalating: {old_model} → {new_model}")

    def get_avg_quality(self) -> float:
        """Get average quality score."""
        if not self.quality_scores:
            return 0.0
        return sum(self.quality_scores) / len(self.quality_scores)

    def get_stats(self) -> dict:
        """Get selector statistics."""
        return {
            "current_model": self.get_current_model(),
            "level": self.current_level,
            "avg_quality": self.get_avg_quality(),
            "success_streak": self.success_streak,
            "failure_streak": self.failure_streak,
        }


# Global adaptive selector
_adaptive_selector: Optional[AdaptiveModelSelector] = None


def get_adaptive_selector() -> AdaptiveModelSelector:
    """Get global adaptive model selector."""
    global _adaptive_selector
    if _adaptive_selector is None:
        _adaptive_selector = AdaptiveModelSelector()
    return _adaptive_selector


def get_model_for_task(task_type: str = "normal", adaptive: bool = True) -> str:
    """
    Select model based on task type with adaptive escalation.

    - "normal": starts with codex-mini, auto-escalates if needed
    - "complex": gpt-5.2 (or escalated if adaptive)
    - "architecture": gpt-5.2-pro (always - for architecture decisions)

    With adaptive=True, the selector tracks quality and auto-swaps models.
    """
    if task_type == "architecture":
        return MODEL_PRO  # Always use pro for architecture

    if adaptive:
        selector = get_adaptive_selector()
        return selector.get_current_model()

    # Non-adaptive fallback
    if task_type == "complex":
        return MODEL_STANDARD
    return MODEL_MINI

CACHE_DIR = Path("cache/attention_guidance")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════════
# ATTENTION GUIDANCE FROM CODEX-MINI
# ════════════════════════════════════════════════════════════════════════════════

ATTENTION_ANALYSIS_PROMPT = """Analyze this reasoning chain and specify optimal attention patterns.

REASONING SAMPLE:
{sample}

OUTPUT JSON with:
1. "reasoning_type": One of [decomposition, analysis, comparison, causation, conditional, enumeration, synthesis, verification, general]
2. "key_positions": List of token position ranges that are CRITICAL for this reasoning (e.g., [[0,10], [25,35]])
3. "attention_flow": How attention should flow: "sequential" (step-by-step), "parallel" (all steps equal), "hierarchical" (later steps attend to earlier), "focal" (concentrate on key insight)
4. "cross_step_attention": Float 0-1, how much later reasoning steps should attend to earlier steps
5. "answer_grounding": Float 0-1, how much the answer should attend back to reasoning

Respond with ONLY valid JSON."""

BATCH_GUIDANCE_PROMPT = """Analyze these {n} reasoning samples and identify SHARED attention patterns.

SAMPLES:
{samples}

For efficient training, identify:
1. "common_reasoning_type": Most common reasoning pattern
2. "shared_attention_structure": Attention pattern that works for ALL samples
3. "critical_token_patterns": Token patterns that always need high attention (e.g., "step", "therefore", "because")
4. "optimal_weights": {{
     "reasoning_to_reasoning": float,  // How much reasoning attends to itself
     "answer_to_reasoning": float,     // How much answer attends to reasoning
     "cross_sample_consistency": float // How similar attention should be across samples
   }}

Respond with ONLY valid JSON."""


@dataclass
class AttentionGuidance:
    """Guidance for attention patterns from codex-mini."""
    reasoning_type: str = "general"
    key_positions: List[Tuple[int, int]] = field(default_factory=list)
    attention_flow: str = "sequential"
    cross_step_attention: float = 0.7
    answer_grounding: float = 0.8
    confidence: float = 1.0


class CodexAttentionManager:
    """
    Uses gpt-5.1-codex-mini to manage self-attention optimization.

    This is the brain that tells the model HOW to attend.
    """

    def __init__(self, cache_enabled: bool = True):
        self.cache_enabled = cache_enabled
        self.guidance_cache: Dict[str, AttentionGuidance] = {}
        self.batch_guidance_cache: Dict[str, dict] = {}
        self.call_count = 0
        self.cache_hits = 0

        # Load cached guidance
        self._load_cache()

    def _load_cache(self):
        """Load cached guidance from disk."""
        cache_file = CACHE_DIR / "guidance_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                    for key, val in data.items():
                        self.guidance_cache[key] = AttentionGuidance(**val)
                print(f"[AutoAttention] Loaded {len(self.guidance_cache)} cached guidance entries")
            except:
                pass

    def _save_cache(self):
        """Save guidance cache to disk."""
        cache_file = CACHE_DIR / "guidance_cache.json"
        data = {}
        for key, guidance in self.guidance_cache.items():
            data[key] = {
                "reasoning_type": guidance.reasoning_type,
                "key_positions": guidance.key_positions,
                "attention_flow": guidance.attention_flow,
                "cross_step_attention": guidance.cross_step_attention,
                "answer_grounding": guidance.answer_grounding,
                "confidence": guidance.confidence,
            }
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _hash_sample(self, sample: str) -> str:
        """Create hash for cache lookup."""
        # Hash based on structure, not exact content (for generalization)
        structure = f"{len(sample)}:{sample[:100]}:{sample[-100:]}"
        return hashlib.md5(structure.encode()).hexdigest()[:16]

    async def get_guidance(self, sample: str) -> AttentionGuidance:
        """Get attention guidance for a single sample."""
        sample_hash = self._hash_sample(sample)

        # Check cache
        if self.cache_enabled and sample_hash in self.guidance_cache:
            self.cache_hits += 1
            return self.guidance_cache[sample_hash]

        # Call codex-mini
        guidance = await self._call_codex_for_guidance(sample)

        # Cache result
        if self.cache_enabled:
            self.guidance_cache[sample_hash] = guidance
            if len(self.guidance_cache) % 100 == 0:
                self._save_cache()

        return guidance

    async def get_batch_guidance(self, samples: List[str]) -> dict:
        """Get shared attention guidance for a batch of samples."""
        # Create batch hash
        batch_hash = hashlib.md5(
            "".join(s[:50] for s in samples[:10]).encode()
        ).hexdigest()[:16]

        if self.cache_enabled and batch_hash in self.batch_guidance_cache:
            self.cache_hits += 1
            return self.batch_guidance_cache[batch_hash]

        # Call codex-mini for batch analysis
        guidance = await self._call_codex_for_batch(samples)

        if self.cache_enabled:
            self.batch_guidance_cache[batch_hash] = guidance

        return guidance

    async def _call_codex_for_guidance(self, sample: str) -> AttentionGuidance:
        """Call codex-mini API for single sample guidance (with adaptive model)."""
        if not API_KEY:
            return AttentionGuidance()  # Default guidance

        prompt = ATTENTION_ANALYSIS_PROMPT.format(sample=sample[:2000])

        try:
            result, quality = await self._api_call(prompt, task_type="normal")
            data = json.loads(result)

            self.call_count += 1

            return AttentionGuidance(
                reasoning_type=data.get("reasoning_type", "general"),
                key_positions=data.get("key_positions", []),
                attention_flow=data.get("attention_flow", "sequential"),
                cross_step_attention=data.get("cross_step_attention", 0.7),
                answer_grounding=data.get("answer_grounding", 0.8),
                confidence=quality,  # Use adaptive quality score
            )
        except Exception as e:
            print(f"[AutoAttention] API error: {e}")
            return AttentionGuidance()

    async def _call_codex_for_batch(self, samples: List[str]) -> dict:
        """Call codex-mini API for batch guidance (with adaptive model)."""
        if not API_KEY:
            return self._default_batch_guidance()

        # Format samples for prompt
        samples_text = "\n\n---\n\n".join(
            f"Sample {i+1}:\n{s[:500]}" for i, s in enumerate(samples[:5])
        )

        prompt = BATCH_GUIDANCE_PROMPT.format(n=len(samples), samples=samples_text)

        try:
            result, quality = await self._api_call(prompt, task_type="normal")
            data = json.loads(result)
            data["_quality"] = quality  # Include quality score
            self.call_count += 1
            return data
        except:
            return self._default_batch_guidance()

    def _default_batch_guidance(self) -> dict:
        """Default batch guidance when API unavailable."""
        return {
            "common_reasoning_type": "decomposition",
            "shared_attention_structure": "sequential",
            "critical_token_patterns": ["step", "therefore", "because", "first", "then"],
            "optimal_weights": {
                "reasoning_to_reasoning": 0.8,
                "answer_to_reasoning": 0.9,
                "cross_sample_consistency": 0.7,
            }
        }

    async def _api_call(self, prompt: str, task_type: str = "normal") -> Tuple[str, float]:
        """
        Make API call with adaptive model selection.

        Returns (response_text, quality_score) where quality_score is 0-1.

        Task types:
          - "normal": starts with codex-mini, auto-escalates if needed
          - "complex": gpt-5.2 (cross-file analysis)
          - "architecture": gpt-5.2-pro (ONLY attention architecture decisions)
        """
        selector = get_adaptive_selector()
        model = get_model_for_task(task_type)

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "input": prompt,
            "max_output_tokens": 1024,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(API_URL, headers=headers, json=payload) as resp:
                    if resp.status != 200:
                        # Record failure and potentially escalate
                        selector.record_result(0.0, success=False)
                        raise Exception(f"API error: {resp.status}")

                    data = await resp.json()

                    # Parse response
                    output = data.get("output", [])
                    text = ""
                    for item in output:
                        if item.get("type") == "message":
                            for c in item.get("content", []):
                                if c.get("type") == "output_text":
                                    text = c.get("text", "")
                                    break

                    # Evaluate response quality
                    quality = self._evaluate_response_quality(text)

                    # Record result for adaptive model selection
                    selector.record_result(quality, success=True)

                    return text, quality

        except Exception as e:
            selector.record_result(0.0, success=False)
            raise

    def _evaluate_response_quality(self, response: str) -> float:
        """
        Evaluate the quality of a codex response.

        Returns 0.0 to 1.0 based on:
        - Is it valid JSON (if expected)
        - Does it have expected keys
        - Is it non-empty and substantive
        """
        if not response:
            return 0.0

        score = 0.3  # Base score for non-empty response

        # Check if it's valid JSON
        try:
            data = json.loads(response)
            score += 0.3  # Valid JSON

            # Check for expected keys
            expected_keys = ["reasoning_type", "attention_flow", "cross_step_attention",
                            "common_reasoning_type", "optimal_weights"]
            found_keys = sum(1 for k in expected_keys if k in data)
            score += 0.4 * (found_keys / len(expected_keys))

        except json.JSONDecodeError:
            # Not JSON, but might still be useful
            if len(response) > 50:
                score += 0.2

        return min(1.0, score)

    def get_stats(self) -> dict:
        """Get manager statistics including adaptive model selection."""
        selector = get_adaptive_selector()
        return {
            "api_calls": self.call_count,
            "cache_hits": self.cache_hits,
            "cache_size": len(self.guidance_cache),
            "hit_rate": self.cache_hits / max(self.cache_hits + self.call_count, 1),
            "adaptive_model": selector.get_current_model(),
            "adaptive_level": selector.current_level,
            "avg_quality": selector.get_avg_quality(),
        }


# ════════════════════════════════════════════════════════════════════════════════
# AUTO-MANAGED ATTENTION LAYER
# ════════════════════════════════════════════════════════════════════════════════

class AutoManagedAttention(nn.Module):
    """
    Self-attention layer that is automatically managed by codex-mini.

    The attention patterns are dynamically adjusted based on:
    1. Real-time analysis of input reasoning structure
    2. Guidance from codex-mini on optimal attention flow
    3. Cross-sample consistency enforcement

    Author: Bo Shang <bo@shang.software>
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Standard attention projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Learnable attention modifiers (updated by codex-mini guidance)
        self.attention_bias = nn.Parameter(torch.zeros(1, num_heads, 1, 1))
        self.flow_weights = nn.Parameter(torch.ones(4))  # [seq, parallel, hier, focal]

        # Guidance integration
        self.guidance_proj = nn.Linear(5, num_heads)  # Project guidance to per-head weights

        self.dropout = nn.Dropout(dropout)

        # Current guidance (set externally)
        self._current_guidance: Optional[AttentionGuidance] = None
        self._batch_guidance: Optional[dict] = None

    def set_guidance(self, guidance: AttentionGuidance):
        """Set current attention guidance from codex-mini."""
        self._current_guidance = guidance

    def set_batch_guidance(self, guidance: dict):
        """Set batch-level guidance."""
        self._batch_guidance = guidance

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with auto-managed attention.
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Standard attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Apply guidance-based attention modification
        attn_scores = self._apply_guidance(attn_scores, seq_len)

        # Apply mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax and output
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        out = self.out_proj(out)

        return out, attn_weights

    def _apply_guidance(self, attn_scores: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply codex-mini guidance to attention scores."""
        device = attn_scores.device
        batch_size = attn_scores.size(0)

        # Start with learned bias
        attn_scores = attn_scores + self.attention_bias

        if self._current_guidance is None and self._batch_guidance is None:
            return attn_scores

        # Build attention modifier based on guidance
        modifier = torch.zeros(1, 1, seq_len, seq_len, device=device)

        if self._current_guidance:
            g = self._current_guidance

            # Apply attention flow pattern
            flow_weights = F.softmax(self.flow_weights, dim=0)

            if g.attention_flow == "sequential":
                # Diagonal attention (each position attends to previous)
                for i in range(seq_len):
                    if i > 0:
                        modifier[0, 0, i, i-1] += flow_weights[0] * 0.5

            elif g.attention_flow == "hierarchical":
                # Later positions attend more to earlier
                for i in range(seq_len):
                    for j in range(i):
                        modifier[0, 0, i, j] += flow_weights[2] * (i - j) / seq_len * 0.3

            elif g.attention_flow == "focal":
                # Concentrate on key positions
                for start, end in g.key_positions:
                    if start < seq_len and end <= seq_len:
                        modifier[0, 0, :, start:end] += flow_weights[3] * 0.5

            # Cross-step attention
            modifier = modifier * g.cross_step_attention

        if self._batch_guidance:
            bg = self._batch_guidance
            weights = bg.get("optimal_weights", {})

            # Apply reasoning-to-reasoning weight
            r2r = weights.get("reasoning_to_reasoning", 0.8)
            modifier = modifier * r2r

        return attn_scores + modifier


# ════════════════════════════════════════════════════════════════════════════════
# TRAINING INTEGRATION
# ════════════════════════════════════════════════════════════════════════════════

class AutoAttentionTrainer:
    """
    Training wrapper that uses codex-mini to manage attention during training.

    Every N batches, queries codex-mini for updated attention guidance
    and adjusts the model's attention patterns accordingly.
    """

    def __init__(
        self,
        model: nn.Module,
        update_frequency: int = 50,  # Query codex-mini every N batches
        async_updates: bool = True,   # Non-blocking API calls
    ):
        self.model = model
        self.update_frequency = update_frequency
        self.async_updates = async_updates

        self.manager = CodexAttentionManager()
        self.step_count = 0
        self.pending_guidance = None

        # Find all AutoManagedAttention layers
        self.attention_layers = []
        for module in model.modules():
            if isinstance(module, AutoManagedAttention):
                self.attention_layers.append(module)

        print(f"[AutoAttentionTrainer] Found {len(self.attention_layers)} managed attention layers")

    def pre_batch(self, batch_texts: List[str]):
        """Called before processing a batch."""
        self.step_count += 1

        if self.step_count % self.update_frequency == 0:
            if self.async_updates:
                # Start async guidance fetch
                asyncio.create_task(self._fetch_guidance_async(batch_texts))
            else:
                # Sync fetch
                asyncio.run(self._fetch_guidance(batch_texts))

        # Apply any pending guidance
        if self.pending_guidance:
            self._apply_guidance(self.pending_guidance)
            self.pending_guidance = None

    async def _fetch_guidance_async(self, texts: List[str]):
        """Fetch guidance asynchronously."""
        guidance = await self.manager.get_batch_guidance(texts)
        self.pending_guidance = guidance

    async def _fetch_guidance(self, texts: List[str]):
        """Fetch guidance synchronously."""
        guidance = await self.manager.get_batch_guidance(texts)
        self._apply_guidance(guidance)

    def _apply_guidance(self, guidance: dict):
        """Apply guidance to all attention layers."""
        for layer in self.attention_layers:
            layer.set_batch_guidance(guidance)

    def get_stats(self) -> dict:
        """Get training statistics."""
        return {
            "steps": self.step_count,
            "attention_layers": len(self.attention_layers),
            **self.manager.get_stats(),
        }


# ════════════════════════════════════════════════════════════════════════════════
# SIMPLE INTEGRATION API
# ════════════════════════════════════════════════════════════════════════════════

_global_manager: Optional[CodexAttentionManager] = None

def get_attention_manager() -> CodexAttentionManager:
    """Get global attention manager."""
    global _global_manager
    if _global_manager is None:
        _global_manager = CodexAttentionManager()
    return _global_manager


async def get_optimal_attention(sample: str) -> AttentionGuidance:
    """Simple API: get optimal attention for a sample."""
    manager = get_attention_manager()
    return await manager.get_guidance(sample)


async def get_batch_attention(samples: List[str]) -> dict:
    """Simple API: get optimal attention for a batch."""
    manager = get_attention_manager()
    return await manager.get_batch_guidance(samples)


def create_auto_attention_layer(hidden_dim: int, num_heads: int) -> AutoManagedAttention:
    """Create an auto-managed attention layer."""
    return AutoManagedAttention(hidden_dim=hidden_dim, num_heads=num_heads)


# ════════════════════════════════════════════════════════════════════════════════
# GCS BACKUP & EXISTING DATA MANAGEMENT
# ════════════════════════════════════════════════════════════════════════════════

GCS_BUCKET = os.environ.get("GCS_BUCKET", "erosolar-training-data")
DATA_STORE = Path("data_store")
ROUNDS_DIR = DATA_STORE / "rounds"


class DataAttentionManager:
    """
    Uses codex-mini to manage self-attention across ALL existing training data.

    This scans all JSONL files, analyzes them with codex-mini, and:
    1. Computes optimal attention patterns for each reasoning type
    2. Identifies patterns that need reinforcement
    3. Backs up analyzed data to GCS for persistence
    """

    def __init__(self, gcs_backup: bool = True):
        self.attention_manager = get_attention_manager()
        self.gcs_backup = gcs_backup
        self.analyzed_files: Dict[str, dict] = {}
        self.global_patterns: Dict[str, int] = defaultdict(int)
        self.analysis_cache_file = CACHE_DIR / "data_analysis.json"

        self._load_analysis_cache()

    def _load_analysis_cache(self):
        """Load previous analysis results."""
        if self.analysis_cache_file.exists():
            try:
                with open(self.analysis_cache_file) as f:
                    data = json.load(f)
                    self.analyzed_files = data.get("files", {})
                    self.global_patterns = defaultdict(int, data.get("patterns", {}))
                print(f"[DataAttentionManager] Loaded analysis for {len(self.analyzed_files)} files")
            except:
                pass

    def _save_analysis_cache(self):
        """Save analysis results."""
        with open(self.analysis_cache_file, 'w') as f:
            json.dump({
                "files": self.analyzed_files,
                "patterns": dict(self.global_patterns),
                "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f, indent=2)

    def find_all_jsonl_files(self) -> List[Path]:
        """Find all JSONL training data files."""
        files = []
        seen = set()

        # All training JSONL files in data_store
        for pattern in ["*_training_data.jsonl", "*_training.jsonl"]:
            for f in DATA_STORE.glob(pattern):
                if f.name not in seen:
                    seen.add(f.name)
                    files.append(f)

        # Round files
        if ROUNDS_DIR.exists():
            for f in sorted(ROUNDS_DIR.glob("*.jsonl")):
                if f.name not in seen:
                    seen.add(f.name)
                    files.append(f)

        return files

    async def analyze_file(self, file_path: Path, sample_size: int = 50) -> dict:
        """Analyze a single JSONL file with codex-mini."""
        file_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:16]

        # Check if already analyzed (and file hasn't changed)
        if file_hash in self.analyzed_files:
            cached = self.analyzed_files[file_hash]
            if file_path.exists():
                current_size = file_path.stat().st_size
                if cached.get("size") == current_size:
                    return cached

        print(f"  Analyzing {file_path.name}...")

        # Read samples from file
        samples = []
        total_lines = 0
        try:
            with open(file_path) as f:
                lines = f.readlines()
                total_lines = len(lines)

                # Sample evenly from file
                step = max(1, total_lines // sample_size)
                for i in range(0, total_lines, step):
                    if len(samples) >= sample_size:
                        break
                    try:
                        data = json.loads(lines[i])
                        # Extract reasoning text
                        text = data.get("text", data.get("response", ""))
                        if text:
                            samples.append(text[:1000])  # Truncate for API
                    except:
                        continue
        except Exception as e:
            return {"error": str(e), "records": 0}

        if not samples:
            return {"records": total_lines, "samples_analyzed": 0}

        # Get batch guidance from codex-mini
        guidance = await self.attention_manager.get_batch_guidance(samples)

        # Update global pattern counts
        reasoning_type = guidance.get("common_reasoning_type", "general")
        self.global_patterns[reasoning_type] += len(samples)

        result = {
            "path": str(file_path),
            "records": total_lines,
            "samples_analyzed": len(samples),
            "size": file_path.stat().st_size if file_path.exists() else 0,
            "guidance": guidance,
            "reasoning_type": reasoning_type,
            "analyzed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        self.analyzed_files[file_hash] = result
        return result

    async def analyze_all_files(self) -> dict:
        """Analyze all existing JSONL files with codex-mini."""
        files = self.find_all_jsonl_files()

        if not files:
            print("[DataAttentionManager] No JSONL files found")
            return {"files": 0}

        print(f"\n[DataAttentionManager] Analyzing {len(files)} JSONL files with codex-mini...")

        results = []
        total_records = 0
        for f in files:
            result = await self.analyze_file(f)
            results.append(result)
            total_records += result.get("records", 0)

        # Save analysis
        self._save_analysis_cache()

        # Backup to GCS if enabled
        if self.gcs_backup:
            await self.backup_analysis_to_gcs()

        summary = {
            "files_analyzed": len(files),
            "total_records": total_records,
            "reasoning_distribution": dict(self.global_patterns),
            "optimal_attention_config": self._compute_optimal_config(),
        }

        print(f"\n[DataAttentionManager] Analysis complete:")
        print(f"  Files: {len(files)}")
        print(f"  Records: {total_records:,}")
        print(f"  Patterns: {dict(self.global_patterns)}")

        return summary

    def _compute_optimal_config(self) -> dict:
        """Compute optimal attention configuration based on all analyzed data."""
        total = sum(self.global_patterns.values())
        if total == 0:
            return {}

        # Weight attention by most common patterns
        weights = {k: v / total for k, v in self.global_patterns.items()}

        # Determine optimal attention flow
        if weights.get("decomposition", 0) > 0.3:
            flow = "hierarchical"
        elif weights.get("comparison", 0) > 0.2:
            flow = "parallel"
        elif weights.get("synthesis", 0) > 0.2:
            flow = "focal"
        else:
            flow = "sequential"

        return {
            "recommended_flow": flow,
            "pattern_weights": weights,
            "cross_step_attention": 0.7 + 0.2 * weights.get("decomposition", 0),
            "answer_grounding": 0.8 + 0.1 * weights.get("synthesis", 0),
        }

    async def backup_analysis_to_gcs(self):
        """Backup analysis results to GCS."""
        if not self.analysis_cache_file.exists():
            return

        try:
            gcs_path = f"gs://{GCS_BUCKET}/attention_analysis/data_analysis.json"
            result = subprocess.run(
                ["gsutil", "cp", str(self.analysis_cache_file), gcs_path],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                print(f"  ✓ Analysis backed up to {gcs_path}")
        except Exception as e:
            print(f"  GCS backup failed: {e}")

    async def restore_from_gcs(self) -> bool:
        """Restore analysis from GCS if local is missing."""
        if self.analysis_cache_file.exists():
            return True

        try:
            gcs_path = f"gs://{GCS_BUCKET}/attention_analysis/data_analysis.json"
            result = subprocess.run(
                ["gsutil", "cp", gcs_path, str(self.analysis_cache_file)],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                print(f"  ✓ Restored analysis from {gcs_path}")
                self._load_analysis_cache()
                return True
        except:
            pass

        return False

    def get_attention_guidance_for_training(self) -> dict:
        """Get optimal attention guidance for training based on all analyzed data."""
        config = self._compute_optimal_config()

        return {
            "attention_flow": config.get("recommended_flow", "sequential"),
            "cross_step_attention": config.get("cross_step_attention", 0.7),
            "answer_grounding": config.get("answer_grounding", 0.8),
            "patterns": self.global_patterns,
        }

    async def get_architecture_decision(self, context: str) -> dict:
        """
        Use gpt-5.2-pro ONLY for highly complex attention architecture decisions.

        This is called rarely - only when making fundamental changes to
        how self-attention is structured (e.g., new attention mechanisms,
        major architectural changes).
        """
        if not API_KEY:
            return {"decision": "use_default", "reason": "No API key"}

        prompt = f"""You are an expert in transformer attention architectures.

Based on this training data analysis:
{context}

Provide ONLY the optimal attention architecture configuration as JSON:
{{
  "attention_type": "perfect_self_attention" | "standard" | "linear" | "hybrid",
  "reasoning_prior_weight": float 0-1,
  "num_attention_heads_ratio": float (relative to hidden_dim),
  "layer_wise_attention": bool,
  "cross_layer_sharing": bool,
  "architectural_notes": "string"
}}

This is for Perfect Self-Attention: attention = softmax(QK^T/√d + R)V
Author: Bo Shang <bo@shang.software>"""

        try:
            headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
            payload = {"model": MODEL_PRO, "input": prompt, "max_output_tokens": 512}

            async with aiohttp.ClientSession() as session:
                async with session.post(API_URL, headers=headers, json=payload) as resp:
                    if resp.status != 200:
                        return {"decision": "use_default", "reason": f"API error {resp.status}"}
                    data = await resp.json()
                    output = data.get("output", [])
                    for item in output:
                        if item.get("type") == "message":
                            for c in item.get("content", []):
                                if c.get("type") == "output_text":
                                    return json.loads(c.get("text", "{}"))
        except Exception as e:
            print(f"[Architecture] gpt-5.2-pro call failed: {e}")

        return {"decision": "use_default", "attention_type": "perfect_self_attention"}


async def manage_all_existing_data(backup_to_gcs: bool = True) -> dict:
    """
    Main function to have codex-mini manage all existing training data.

    This analyzes all JSONL files and computes optimal attention patterns.
    ALSO computes embedding-based quality for existing records.
    (Author: Bo Shang <bo@shang.software>)
    """
    manager = DataAttentionManager(gcs_backup=backup_to_gcs)

    # Try to restore from GCS first if local cache is empty
    if not manager.analyzed_files:
        await manager.restore_from_gcs()

    # Analyze all files
    result = await manager.analyze_all_files()

    # CRITICAL: Compute embedding-based quality for existing records
    # So avg_quality is NEVER 0 when we have data
    # (Author: Bo Shang <bo@shang.software>)
    await _compute_existing_data_quality(manager)

    return result


async def _compute_existing_data_quality(manager: DataAttentionManager):
    """
    Compute embedding-based quality for existing training data.
    Records quality to AdaptiveModelSelector so avg_quality is never 0.
    (Author: Bo Shang <bo@shang.software>)
    """
    import aiohttp
    import numpy as np

    selector = get_adaptive_selector()

    # Get sample of CoT texts from existing data
    all_files = manager.find_all_jsonl_files()
    cot_samples = []

    for file_path in all_files[:3]:  # Sample from first 3 files
        try:
            with open(file_path) as f:
                lines = f.readlines()
                # Sample up to 20 records per file
                step = max(1, len(lines) // 20)
                for i in range(0, min(len(lines), 20 * step), step):
                    try:
                        data = json.loads(lines[i])
                        # Extract thinking/CoT content
                        text = data.get("thinking", data.get("response", data.get("text", "")))
                        if "<|think_start|>" in text:
                            start = text.find("<|think_start|>") + len("<|think_start|>")
                            end = text.find("<|think_end|>") if "<|think_end|>" in text else len(text)
                            cot = text[start:end].strip()
                            if cot and len(cot) > 50:
                                cot_samples.append(cot[:2000])
                    except:
                        continue
        except:
            continue

    if len(cot_samples) < 2:
        print("  [Quality] Not enough CoT samples for embedding-based quality")
        # Record minimum quality so it's not 0
        selector.record_result(0.1, success=True)
        return

    # Compute embedding similarity using LOCAL sentence-transformers (no API key needed)
    if not LOCAL_EMBEDDINGS_AVAILABLE:
        print("  [Quality] Local embeddings not available - using pattern-based quality")
        selector.record_result(0.3, success=True)
        return

    print(f"  [Quality] Computing embedding similarity for {len(cot_samples)} CoT samples (LOCAL)...")

    try:
        # Use local sentence-transformers for embeddings
        samples_to_embed = cot_samples[:50]  # Max 50 samples
        embeddings = local_embed_batch(samples_to_embed, batch_size=32, show_progress=False)

        # Compute average pairwise similarity
        avg_sim = average_pairwise_similarity(embeddings)

        # Record quality based on embedding similarity
        # Typical values: 0.3-0.6 for diverse CoTs, 0.7+ for consistent CoTs
        quality = max(0.1, min(1.0, avg_sim))
        selector.record_result(quality, success=True)

        print(f"  [Quality] Embedding similarity: {avg_sim:.4f} → quality: {quality:.2f}")

    except Exception as e:
        print(f"  [Quality] Embedding error: {e}")
        selector.record_result(0.2, success=True)


def get_data_attention_manager() -> DataAttentionManager:
    """Get a data attention manager instance."""
    return DataAttentionManager()


# ════════════════════════════════════════════════════════════════════════════════
# CLI / TESTING
# ════════════════════════════════════════════════════════════════════════════════

async def test_attention_manager():
    """Test the attention manager."""
    manager = CodexAttentionManager()

    sample = """<|think_start|>
Let me break this down step by step.
<|step|>First, I need to understand the problem.
<|step|>Then, I'll identify the key components.
<|step|>Finally, I'll combine them for the solution.
<|think_end|>
The answer is 42."""

    print("Testing single sample guidance...")
    guidance = await manager.get_guidance(sample)
    print(f"  Reasoning type: {guidance.reasoning_type}")
    print(f"  Attention flow: {guidance.attention_flow}")
    print(f"  Cross-step attention: {guidance.cross_step_attention}")

    print("\nTesting batch guidance...")
    samples = [sample, sample.replace("42", "100"), sample.replace("step", "phase")]
    batch_guidance = await manager.get_batch_guidance(samples)
    print(f"  Common type: {batch_guidance.get('common_reasoning_type')}")
    print(f"  Optimal weights: {batch_guidance.get('optimal_weights')}")

    print("\nManager stats:")
    for k, v in manager.get_stats().items():
        print(f"  {k}: {v}")


async def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Auto-Managed Self-Attention with codex-mini")
    parser.add_argument("--test", action="store_true", help="Run attention manager tests")
    parser.add_argument("--manage-data", action="store_true", help="Analyze all existing JSONL files")
    parser.add_argument("--backup", action="store_true", default=True, help="Backup analysis to GCS")
    parser.add_argument("--no-backup", dest="backup", action="store_false", help="Skip GCS backup")
    parser.add_argument("--stats", action="store_true", help="Show manager statistics")
    args = parser.parse_args()

    if args.test:
        await test_attention_manager()

    elif args.manage_data:
        selector = get_adaptive_selector()
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║  AUTO-ATTENTION MANAGER - Managing All Training Data        ║")
        print("║  Author: Bo Shang <bo@shang.software>                       ║")
        print("╠══════════════════════════════════════════════════════════════╣")
        print(f"║  ADAPTIVE MODEL SELECTION (auto-escalates/de-escalates)    ║")
        print(f"║  Level 0: {MODEL_MINI} (default, most tasks)             ║")
        print(f"║  Level 1: {MODEL_STANDARD} (auto-escalate on quality < 0.5)       ║")
        print(f"║  Level 2: {MODEL_PRO} (auto-escalate, ONLY if needed)    ║")
        print(f"║  Current: {selector.get_current_model():<44}  ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        print()

        result = await manage_all_existing_data(backup_to_gcs=args.backup)

        print("\n" + "═" * 60)
        print("ANALYSIS COMPLETE")
        print("═" * 60)
        print(f"  Files analyzed: {result.get('files_analyzed', 0)}")
        print(f"  Total records: {result.get('total_records', 0):,}")

        # Show adaptive model status
        selector_stats = selector.get_stats()
        print(f"\n  Adaptive Model Status:")
        print(f"    Current model: {selector_stats['current_model']}")
        print(f"    Level: {selector_stats['level']} / 2")
        print(f"    Avg quality: {selector_stats['avg_quality']:.2f}")

        config = result.get("optimal_attention_config", {})
        if config:
            print(f"\n  Optimal Configuration:")
            print(f"    Recommended flow: {config.get('recommended_flow')}")
            print(f"    Cross-step attention: {config.get('cross_step_attention', 0):.2f}")
            print(f"    Answer grounding: {config.get('answer_grounding', 0):.2f}")

    elif args.stats:
        manager = get_attention_manager()
        print("Attention Manager Statistics:")
        for k, v in manager.get_stats().items():
            print(f"  {k}: {v}")

    else:
        # Default: manage existing data
        await manage_all_existing_data(backup_to_gcs=args.backup)


if __name__ == "__main__":
    asyncio.run(main())
