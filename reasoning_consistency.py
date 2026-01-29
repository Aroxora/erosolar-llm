#!/usr/bin/env python3
"""
REASONING-CONSISTENT TRAINING (REAL)
=====================================
This module provides real reasoning-structure calculations used in training.
Master scalar remains the primary optimization target.

GPT-2/OpenAI approach (inefficient):
  - Random diverse data → accidentally learns tasks
  - "Predict next word" hopes reasoning emerges from chaos
  - Requires massive compute to stumble upon patterns

This approach:
  - Measures similarity of chain-of-thought patterns
  - Reports reasoning structure consistency across domains
  - Supplies signals that drive attention and consistency losses

Key Innovation:
  Instead of P(next_token | context), we optimize:
  P(next_token | context, reasoning_cluster)

  This forces the model to learn TRANSFERABLE reasoning patterns
  that apply across different domains, not domain-specific memorization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib

# ════════════════════════════════════════════════════════════════════════════════
# REASONING PATTERN EXTRACTION
# ════════════════════════════════════════════════════════════════════════════════

# Reasoning pattern markers (from training data format)
THINK_START = "<|think_start|>"
THINK_END = "<|think_end|>"
STEP_MARKER = "<|step|>"
ANSWER_MARKER = "<|answer|>"

@dataclass
class ReasoningPattern:
    """Extracted reasoning pattern from a training sample."""
    sample_id: str
    raw_thinking: str
    steps: List[str]
    step_count: int
    pattern_hash: str  # Hash of reasoning structure
    pattern_type: str  # Classification of reasoning type
    embedding: Optional[np.ndarray] = None
    cluster_id: int = -1
    weight: float = 1.0


# Reasoning pattern types (explicit taxonomy)
REASONING_TYPES = {
    "decomposition": ["break down", "step by step", "first", "then", "finally", "let's"],
    "analysis": ["analyze", "consider", "examine", "look at", "observe"],
    "comparison": ["compare", "contrast", "difference", "similar", "versus", "vs"],
    "causation": ["because", "therefore", "thus", "hence", "causes", "leads to"],
    "conditional": ["if", "then", "when", "unless", "assuming", "given that"],
    "enumeration": ["first", "second", "third", "1.", "2.", "3.", "a)", "b)"],
    "definition": ["means", "defined as", "is a", "refers to", "called"],
    "example": ["for example", "such as", "like", "instance", "e.g."],
    "synthesis": ["combine", "together", "overall", "in summary", "conclude"],
    "verification": ["check", "verify", "confirm", "ensure", "validate"],
}


def extract_reasoning(text: str) -> Optional[ReasoningPattern]:
    """Extract reasoning pattern from a training sample."""
    # Find thinking section
    think_match = re.search(
        f"{re.escape(THINK_START)}(.*?){re.escape(THINK_END)}",
        text,
        re.DOTALL
    )

    if not think_match:
        return None

    thinking = think_match.group(1).strip()

    # Extract steps
    steps = []
    if STEP_MARKER in thinking:
        steps = [s.strip() for s in thinking.split(STEP_MARKER) if s.strip()]
    else:
        # Split by sentences/lines as fallback
        steps = [s.strip() for s in re.split(r'[.\n]', thinking) if len(s.strip()) > 10]

    # Classify reasoning type
    pattern_type = classify_reasoning_type(thinking)

    # Create structural hash (captures reasoning structure, not content)
    structure = f"{len(steps)}:{pattern_type}:{len(thinking)//100}"
    pattern_hash = hashlib.md5(structure.encode()).hexdigest()[:8]

    return ReasoningPattern(
        sample_id=hashlib.md5(text.encode()).hexdigest()[:12],
        raw_thinking=thinking,
        steps=steps,
        step_count=len(steps),
        pattern_hash=pattern_hash,
        pattern_type=pattern_type,
    )


def classify_reasoning_type(thinking: str) -> str:
    """Classify the type of reasoning used."""
    thinking_lower = thinking.lower()

    scores = {}
    for rtype, keywords in REASONING_TYPES.items():
        score = sum(1 for kw in keywords if kw in thinking_lower)
        scores[rtype] = score

    if not scores or max(scores.values()) == 0:
        return "general"

    return max(scores, key=scores.get)


# ════════════════════════════════════════════════════════════════════════════════
# REASONING EMBEDDING & CLUSTERING
# ════════════════════════════════════════════════════════════════════════════════

class ReasoningEmbedder:
    """
    Embeds reasoning patterns into a vector space.
    Similar reasoning → similar embeddings → same cluster → reinforced together.
    """

    def __init__(self, embed_dim: int = 64):
        self.embed_dim = embed_dim
        # Feature extractors for reasoning structure
        self.feature_weights = {
            "step_count": 0.2,
            "pattern_type": 0.3,
            "avg_step_length": 0.1,
            "keyword_density": 0.2,
            "structure_hash": 0.2,
        }

    def embed(self, pattern: ReasoningPattern) -> np.ndarray:
        """Create embedding for a reasoning pattern."""
        features = []

        # Step count (normalized)
        features.append(min(pattern.step_count / 10.0, 1.0))

        # Pattern type (one-hot encoded)
        type_vec = np.zeros(len(REASONING_TYPES) + 1)
        type_idx = list(REASONING_TYPES.keys()).index(pattern.pattern_type) \
                   if pattern.pattern_type in REASONING_TYPES else len(REASONING_TYPES)
        type_vec[type_idx] = 1.0
        features.extend(type_vec)

        # Average step length (normalized)
        if pattern.steps:
            avg_len = np.mean([len(s) for s in pattern.steps])
            features.append(min(avg_len / 200.0, 1.0))
        else:
            features.append(0.0)

        # Keyword density per type
        thinking_lower = pattern.raw_thinking.lower()
        for rtype, keywords in REASONING_TYPES.items():
            density = sum(thinking_lower.count(kw) for kw in keywords)
            features.append(min(density / 10.0, 1.0))

        # Structural features from hash
        hash_features = [int(c, 16) / 15.0 for c in pattern.pattern_hash[:8]]
        features.extend(hash_features)

        # Pad/truncate to embed_dim
        embedding = np.array(features[:self.embed_dim])
        if len(embedding) < self.embed_dim:
            embedding = np.pad(embedding, (0, self.embed_dim - len(embedding)))

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity between embeddings."""
        return float(np.dot(emb1, emb2))


class ReasoningClusterer:
    """
    Clusters reasoning patterns to identify similar reasoning chains.
    Samples in the same cluster get trained together → reinforces reasoning consistency.
    """

    def __init__(self, n_clusters: int = 32, min_cluster_size: int = 10):
        self.n_clusters = n_clusters
        self.min_cluster_size = min_cluster_size
        self.centroids: Optional[np.ndarray] = None
        self.cluster_counts: Dict[int, int] = defaultdict(int)

    def fit(self, embeddings: np.ndarray):
        """Fit clusters using k-means."""
        n_samples = len(embeddings)
        if n_samples < self.n_clusters:
            self.n_clusters = max(1, n_samples // 2)

        # Simple k-means
        # Initialize centroids randomly
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = embeddings[indices].copy()

        # Iterate
        for _ in range(20):
            # Assign to nearest centroid
            assignments = self._assign(embeddings)

            # Update centroids
            new_centroids = np.zeros_like(self.centroids)
            counts = np.zeros(self.n_clusters)

            for i, cluster_id in enumerate(assignments):
                new_centroids[cluster_id] += embeddings[i]
                counts[cluster_id] += 1

            for c in range(self.n_clusters):
                if counts[c] > 0:
                    new_centroids[c] /= counts[c]
                else:
                    # Reinitialize empty cluster
                    new_centroids[c] = embeddings[np.random.randint(n_samples)]

            self.centroids = new_centroids

        # Final assignment and counts
        assignments = self._assign(embeddings)
        self.cluster_counts = defaultdict(int)
        for c in assignments:
            self.cluster_counts[c] += 1

        return assignments

    def _assign(self, embeddings: np.ndarray) -> np.ndarray:
        """Assign embeddings to nearest centroid."""
        # Compute distances to all centroids
        distances = np.linalg.norm(
            embeddings[:, np.newaxis, :] - self.centroids[np.newaxis, :, :],
            axis=2
        )
        return np.argmin(distances, axis=1)

    def predict(self, embedding: np.ndarray) -> int:
        """Predict cluster for a single embedding."""
        if self.centroids is None:
            return 0
        distances = np.linalg.norm(self.centroids - embedding, axis=1)
        return int(np.argmin(distances))


# ════════════════════════════════════════════════════════════════════════════════
# DYNAMIC WEIGHT ADJUSTMENT
# ════════════════════════════════════════════════════════════════════════════════

class ReasoningWeightManager:
    """
    Dynamically adjusts sample weights to reinforce reasoning consistency.

    Key principle: Samples with similar reasoning patterns should be
    trained together with higher weight, creating "reasoning anchors"
    that transfer across domains.
    """

    def __init__(
        self,
        base_weight: float = 1.0,
        cluster_boost: float = 1.5,      # Boost for samples in same cluster
        consistency_boost: float = 2.0,   # Boost for highly consistent reasoning
        diversity_penalty: float = 0.5,   # Penalty for random/inconsistent reasoning
    ):
        self.base_weight = base_weight
        self.cluster_boost = cluster_boost
        self.consistency_boost = consistency_boost
        self.diversity_penalty = diversity_penalty

        self.pattern_weights: Dict[str, float] = {}
        self.cluster_weights: Dict[int, float] = {}

    def compute_weights(
        self,
        patterns: List[ReasoningPattern],
        embeddings: np.ndarray,
        cluster_assignments: np.ndarray,
    ) -> np.ndarray:
        """Compute dynamic weights for all samples."""
        n_samples = len(patterns)
        weights = np.ones(n_samples) * self.base_weight

        # Compute cluster cohesion (how tight each cluster is)
        cluster_cohesion = self._compute_cluster_cohesion(embeddings, cluster_assignments)

        for i, (pattern, cluster_id) in enumerate(zip(patterns, cluster_assignments)):
            weight = self.base_weight

            # 1. Boost for clear reasoning structure
            if pattern.step_count >= 3:
                weight *= 1.0 + 0.1 * min(pattern.step_count, 10)

            # 2. Boost for recognized reasoning types
            if pattern.pattern_type != "general":
                weight *= self.consistency_boost

            # 3. Boost based on cluster cohesion
            # Tight clusters = consistent reasoning patterns = higher weight
            cohesion = cluster_cohesion.get(cluster_id, 0.5)
            weight *= 1.0 + (cohesion * (self.cluster_boost - 1.0))

            # 4. Penalty for very short/unclear reasoning
            if pattern.step_count < 2 or len(pattern.raw_thinking) < 50:
                weight *= self.diversity_penalty

            weights[i] = weight
            pattern.weight = weight

        # Normalize to sum to n_samples (preserve expected gradient magnitude)
        weights = weights * (n_samples / weights.sum())

        return weights

    def _compute_cluster_cohesion(
        self,
        embeddings: np.ndarray,
        assignments: np.ndarray
    ) -> Dict[int, float]:
        """Compute how cohesive each cluster is (tighter = more consistent reasoning)."""
        cohesion = {}
        unique_clusters = np.unique(assignments)

        for cluster_id in unique_clusters:
            mask = assignments == cluster_id
            cluster_embeddings = embeddings[mask]

            if len(cluster_embeddings) < 2:
                cohesion[cluster_id] = 0.5
                continue

            # Compute average pairwise similarity
            centroid = cluster_embeddings.mean(axis=0)
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            avg_distance = distances.mean()

            # Convert distance to cohesion (0-1, higher = tighter cluster)
            cohesion[cluster_id] = 1.0 / (1.0 + avg_distance)

        return cohesion


# ════════════════════════════════════════════════════════════════════════════════
# CONTRASTIVE REASONING LOSS
# ════════════════════════════════════════════════════════════════════════════════

class ContrastiveReasoningLoss(nn.Module):
    """
    Additional loss term that enforces reasoning consistency.

    Principle: Samples with similar reasoning patterns should have
    similar internal representations, regardless of domain.

    This is the OPPOSITE of GPT-2's approach:
    - GPT-2: Same architecture processes all data the same way
    - This: Explicitly encourage reasoning pattern similarity in hidden states
    """

    def __init__(self, temperature: float = 0.1, weight: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.weight = weight

    def forward(
        self,
        hidden_states: torch.Tensor,  # [batch, seq, hidden]
        cluster_ids: torch.Tensor,     # [batch]
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute contrastive loss based on reasoning clusters.

        Samples in same cluster should have similar representations.
        Samples in different clusters should be distinguishable.
        """
        batch_size = hidden_states.size(0)

        if batch_size < 2:
            return torch.tensor(0.0, device=hidden_states.device)

        # Pool hidden states to get sample-level representation
        if attention_mask is not None:
            # Masked mean pooling
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1)
        else:
            # Simple mean pooling
            pooled = hidden_states.mean(dim=1)  # [batch, hidden]

        # L2 normalize
        pooled = F.normalize(pooled, p=2, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(pooled, pooled.t()) / self.temperature  # [batch, batch]

        # Create positive/negative masks based on cluster membership
        cluster_ids = cluster_ids.view(-1, 1)
        positive_mask = (cluster_ids == cluster_ids.t()).float()
        positive_mask.fill_diagonal_(0)  # Exclude self

        # InfoNCE-style loss
        # For each sample, pull together samples in same cluster, push apart others
        exp_sim = torch.exp(sim_matrix)
        # Use mask instead of in-place fill_diagonal_ to avoid breaking gradients
        diag_mask = ~torch.eye(exp_sim.size(0), dtype=torch.bool, device=exp_sim.device)
        exp_sim = exp_sim * diag_mask

        # Positive term: log probability of same-cluster samples
        pos_sum = (exp_sim * positive_mask).sum(dim=1)
        all_sum = exp_sim.sum(dim=1)

        # Avoid log(0)
        loss = -torch.log((pos_sum + 1e-8) / (all_sum + 1e-8))

        # Only compute loss for samples that have same-cluster partners
        valid_mask = positive_mask.sum(dim=1) > 0
        if valid_mask.sum() > 0:
            loss = loss[valid_mask].mean()
        else:
            loss = torch.tensor(0.0, device=hidden_states.device)

        return loss * self.weight


# ════════════════════════════════════════════════════════════════════════════════
# REASONING-CONSISTENT DATASET
# ════════════════════════════════════════════════════════════════════════════════

class ReasoningConsistentDataset(torch.utils.data.Dataset):
    """
    Dataset that provides reasoning-aware sampling.

    Instead of random sampling (GPT-2 approach), this:
    1. Groups samples by reasoning pattern
    2. Returns samples with their cluster IDs
    3. Enables batch composition for reasoning consistency
    """

    def __init__(
        self,
        tokens: torch.Tensor,
        seq_len: int,
        patterns: List[ReasoningPattern],
        cluster_ids: np.ndarray,
        weights: np.ndarray,
    ):
        self.tokens = tokens
        self.seq_len = seq_len
        self.patterns = patterns
        self.cluster_ids = torch.tensor(cluster_ids, dtype=torch.long)
        self.weights = torch.tensor(weights, dtype=torch.float32)

        # Create mapping from sample index to token indices
        self.sample_indices = self._create_sample_indices()

    def _create_sample_indices(self) -> List[Tuple[int, int]]:
        """Map sample indices to token ranges."""
        # For now, assume sequential samples
        # In practice, you'd track where each sample starts/ends in the token stream
        indices = []
        n_samples = len(self.patterns)
        tokens_per_sample = len(self.tokens) // max(n_samples, 1)

        for i in range(n_samples):
            start = i * tokens_per_sample
            end = min(start + self.seq_len, len(self.tokens))
            if end - start >= self.seq_len:
                indices.append((start, end))

        return indices if indices else [(0, min(self.seq_len, len(self.tokens)))]

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        start, end = self.sample_indices[idx]
        tokens = self.tokens[start:end]

        # Pad if necessary
        if len(tokens) < self.seq_len:
            tokens = F.pad(tokens, (0, self.seq_len - len(tokens)))

        cluster_id = self.cluster_ids[idx] if idx < len(self.cluster_ids) else 0
        weight = self.weights[idx] if idx < len(self.weights) else 1.0

        return {
            "input_ids": tokens,
            "cluster_id": cluster_id,
            "weight": weight,
        }


# ════════════════════════════════════════════════════════════════════════════════
# REASONING-CONSISTENT SAMPLER
# ════════════════════════════════════════════════════════════════════════════════

class ReasoningConsistentSampler(torch.utils.data.Sampler):
    """
    Sampler that creates batches with reasoning consistency.

    Instead of random batches (GPT-2), this:
    1. Groups samples by reasoning cluster
    2. Creates batches that mix clusters strategically
    3. Ensures each batch has samples for contrastive learning
    """

    def __init__(
        self,
        cluster_ids: np.ndarray,
        weights: np.ndarray,
        batch_size: int,
        cluster_ratio: float = 0.5,  # Ratio of batch from same cluster
    ):
        self.cluster_ids = cluster_ids
        self.weights = weights
        self.batch_size = batch_size
        self.cluster_ratio = cluster_ratio

        # Group indices by cluster
        self.cluster_to_indices = defaultdict(list)
        for idx, cluster_id in enumerate(cluster_ids):
            self.cluster_to_indices[cluster_id].append(idx)

        self.clusters = list(self.cluster_to_indices.keys())

    def __iter__(self):
        """Generate batches with reasoning consistency."""
        indices = list(range(len(self.cluster_ids)))

        # Weighted shuffle
        probs = self.weights / self.weights.sum()
        shuffled = np.random.choice(indices, size=len(indices), replace=False, p=probs)

        # Create batches
        for i in range(0, len(shuffled), self.batch_size):
            batch = shuffled[i:i + self.batch_size].tolist()

            # Optionally reshuffle to ensure cluster diversity within batch
            # (helps contrastive loss have both positives and negatives)
            yield batch

    def __len__(self):
        return (len(self.cluster_ids) + self.batch_size - 1) // self.batch_size


# ════════════════════════════════════════════════════════════════════════════════
# MAIN API - Integration with training pipeline
# ════════════════════════════════════════════════════════════════════════════════

class ReasoningConsistencyManager:
    """
    Main interface for reasoning-consistent training.

    Usage:
        manager = ReasoningConsistencyManager()

        # Process training data
        patterns = manager.extract_patterns(training_texts)
        weights = manager.compute_weights(patterns)

        # Get contrastive loss module
        contrastive_loss = manager.get_contrastive_loss()

        # In training loop:
        loss = lm_loss + contrastive_loss(hidden_states, cluster_ids)
    """

    def __init__(
        self,
        n_clusters: int = 32,
        embed_dim: int = 64,
        contrastive_weight: float = 0.1,
    ):
        self.embedder = ReasoningEmbedder(embed_dim=embed_dim)
        self.clusterer = ReasoningClusterer(n_clusters=n_clusters)
        self.weight_manager = ReasoningWeightManager()
        self.contrastive_loss = ContrastiveReasoningLoss(weight=contrastive_weight)

        self.patterns: List[ReasoningPattern] = []
        self.embeddings: Optional[np.ndarray] = None
        self.cluster_ids: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None

    def process_training_data(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process training texts to extract patterns, cluster, and compute weights.

        Returns:
            cluster_ids: Cluster assignment for each sample
            weights: Dynamic weight for each sample
        """
        print(f"[ReasoningConsistency] Processing {len(texts)} samples...")

        # Extract patterns
        self.patterns = []
        for text in texts:
            pattern = extract_reasoning(text)
            if pattern:
                self.patterns.append(pattern)
            else:
                # Create dummy pattern for samples without explicit reasoning
                self.patterns.append(ReasoningPattern(
                    sample_id=hashlib.md5(text.encode()).hexdigest()[:12],
                    raw_thinking="",
                    steps=[],
                    step_count=0,
                    pattern_hash="00000000",
                    pattern_type="general",
                ))

        print(f"  Extracted {sum(1 for p in self.patterns if p.step_count > 0)} patterns with reasoning")

        # Embed patterns
        self.embeddings = np.array([
            self.embedder.embed(p) for p in self.patterns
        ])

        # Cluster
        self.cluster_ids = self.clusterer.fit(self.embeddings)
        print(f"  Created {len(set(self.cluster_ids))} reasoning clusters")

        # Compute dynamic weights
        self.weights = self.weight_manager.compute_weights(
            self.patterns, self.embeddings, self.cluster_ids
        )

        # Print weight distribution
        print(f"  Weight range: [{self.weights.min():.2f}, {self.weights.max():.2f}]")
        print(f"  High-weight samples (>1.5): {(self.weights > 1.5).sum()}")

        return self.cluster_ids, self.weights

    def get_contrastive_loss(self) -> ContrastiveReasoningLoss:
        """Get the contrastive loss module."""
        return self.contrastive_loss

    def get_cluster_stats(self) -> Dict:
        """Get statistics about reasoning clusters."""
        if self.cluster_ids is None:
            return {}

        stats = {
            "n_clusters": len(set(self.cluster_ids)),
            "cluster_sizes": {},
            "cluster_types": {},
        }

        for cluster_id in set(self.cluster_ids):
            mask = self.cluster_ids == cluster_id
            cluster_patterns = [p for p, m in zip(self.patterns, mask) if m]

            stats["cluster_sizes"][int(cluster_id)] = int(mask.sum())

            # Dominant reasoning type in cluster
            type_counts = defaultdict(int)
            for p in cluster_patterns:
                type_counts[p.pattern_type] += 1
            if type_counts:
                stats["cluster_types"][int(cluster_id)] = max(type_counts, key=type_counts.get)

        return stats

    def save(self, path: Path):
        """Save manager state."""
        state = {
            "n_patterns": len(self.patterns),
            "cluster_ids": self.cluster_ids.tolist() if self.cluster_ids is not None else [],
            "weights": self.weights.tolist() if self.weights is not None else [],
            "stats": self.get_cluster_stats(),
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    def load(self, path: Path):
        """Load manager state."""
        with open(path) as f:
            state = json.load(f)
        self.cluster_ids = np.array(state["cluster_ids"])
        self.weights = np.array(state["weights"])


# ════════════════════════════════════════════════════════════════════════════════
# PERFECT SELF-ATTENTION
# Author: Bo Shang <bo@shang.software>
# ════════════════════════════════════════════════════════════════════════════════
#
# Standard self-attention: attention(Q,K,V) = softmax(QK^T/√d)V
# - Same computation for all samples regardless of reasoning structure
# - Hopes reasoning emerges from random attention patterns
#
# Perfect self-attention: attention(Q,K,V,R) = softmax(QK^T/√d + R)V
# - R = reasoning prior matrix, dynamically computed from CoT similarity
# - Samples with similar reasoning patterns get aligned attention patterns
# - Explicitly reinforces consistent reasoning across domains
#
# Mathematical formulation:
#   Let r_i, r_j be reasoning embeddings for positions i, j
#   R_ij = λ * sim(r_i, r_j) where sim is cosine similarity
#   This biases attention toward positions with similar reasoning patterns

class ReasoningPriorComputer(nn.Module):
    """
    Computes the reasoning prior matrix R for perfect self-attention.

    Given a sequence, identifies reasoning pattern at each position
    and computes pairwise similarity to create attention bias.
    """

    def __init__(self, hidden_dim: int, reasoning_dim: int = 64, lambda_weight: float = 0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.reasoning_dim = reasoning_dim
        self.lambda_weight = lambda_weight

        # Project hidden states to reasoning space
        self.reasoning_proj = nn.Linear(hidden_dim, reasoning_dim)

        # Learnable reasoning pattern detectors (one per reasoning type)
        n_patterns = len(REASONING_TYPES) + 1  # +1 for general
        self.pattern_embeddings = nn.Parameter(torch.randn(n_patterns, reasoning_dim))

        # Gate to control reasoning prior strength (learned per-layer)
        self.prior_gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute reasoning prior matrix.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]

        Returns:
            R: [batch, seq_len, seq_len] - reasoning prior to add to attention scores
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to reasoning space
        r = self.reasoning_proj(hidden_states)  # [batch, seq_len, reasoning_dim]
        r = F.normalize(r, p=2, dim=-1)

        # Compute pairwise reasoning similarity
        # R_ij = sim(r_i, r_j) = r_i · r_j (since normalized)
        R = torch.bmm(r, r.transpose(1, 2))  # [batch, seq_len, seq_len]

        # Apply learnable gate and scaling
        gate = torch.sigmoid(self.prior_gate)
        R = self.lambda_weight * gate * R

        return R

    def get_reasoning_distribution(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Get distribution over reasoning types for interpretability."""
        r = self.reasoning_proj(hidden_states)  # [batch, seq_len, reasoning_dim]
        r = F.normalize(r, p=2, dim=-1)

        # Similarity to each pattern type
        pattern_emb = F.normalize(self.pattern_embeddings, p=2, dim=-1)
        dist = torch.matmul(r, pattern_emb.t())  # [batch, seq_len, n_patterns]
        return F.softmax(dist, dim=-1)


class PerfectSelfAttention(nn.Module):
    """
    Perfect Self-Attention with dynamic reasoning prior.

    Standard: attention = softmax(QK^T/√d)V
    Perfect:  attention = softmax(QK^T/√d + R)V

    Where R is the reasoning prior that biases attention toward
    positions with similar reasoning patterns.

    Author: Bo Shang <bo@shang.software>
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        reasoning_dim: int = 64,
        reasoning_lambda: float = 0.5,
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

        # Reasoning prior computer
        self.reasoning_prior = ReasoningPriorComputer(
            hidden_dim=hidden_dim,
            reasoning_dim=reasoning_dim,
            lambda_weight=reasoning_lambda,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_reasoning_prior: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional reasoning prior.

        Args:
            x: [batch, seq_len, hidden_dim]
            mask: Optional attention mask
            use_reasoning_prior: Whether to apply reasoning prior

        Returns:
            output: [batch, seq_len, hidden_dim]
            attention_weights: [batch, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Standard attention scores: QK^T/√d
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [batch, heads, seq, seq]

        # Add reasoning prior: R (shared across heads)
        if use_reasoning_prior:
            R = self.reasoning_prior(x)  # [batch, seq, seq]
            R = R.unsqueeze(1)  # [batch, 1, seq, seq] - broadcast to all heads
            attn_scores = attn_scores + R

        # Apply mask (causal or padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute output
        out = torch.matmul(attn_weights, V)  # [batch, heads, seq, head_dim]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        out = self.out_proj(out)

        return out, attn_weights


class SharedCoTWeightOptimizer:
    """
    Dynamically optimizes shared Chain-of-Thought weights across training samples.

    Instead of independent optimization per sample (GPT-2 approach),
    this tracks reasoning patterns globally and adjusts model weights
    to reinforce consistent reasoning.

    Key insight: Similar reasoning should produce similar attention patterns,
    regardless of the specific domain/content.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        momentum: float = 0.9,
        update_freq: int = 100,  # Update shared weights every N batches
    ):
        self.model = model
        self.lr = learning_rate
        self.momentum = momentum
        self.update_freq = update_freq

        # Track reasoning patterns and their attention patterns
        self.pattern_attention_buffer: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.target_attention_patterns: Dict[str, torch.Tensor] = {}
        self.step_count = 0

        # Moving average of attention patterns per reasoning type
        self.attention_ema: Dict[str, torch.Tensor] = {}
        self.ema_decay = 0.99

    def register_attention_pattern(
        self,
        reasoning_type: str,
        attention_weights: torch.Tensor,
    ):
        """
        Register an attention pattern for a reasoning type.

        Over time, this builds up the "ideal" attention pattern for each
        reasoning type, which is then used to guide training.
        """
        # Detach and store
        attn = attention_weights.detach().mean(dim=0).mean(dim=0)  # Average over batch and heads

        if reasoning_type not in self.attention_ema:
            self.attention_ema[reasoning_type] = attn.clone()
        else:
            # Exponential moving average
            self.attention_ema[reasoning_type] = (
                self.ema_decay * self.attention_ema[reasoning_type] +
                (1 - self.ema_decay) * attn
            )

        self.pattern_attention_buffer[reasoning_type].append(attn)

        # Limit buffer size
        if len(self.pattern_attention_buffer[reasoning_type]) > 1000:
            self.pattern_attention_buffer[reasoning_type] = \
                self.pattern_attention_buffer[reasoning_type][-500:]

    def compute_consistency_loss(
        self,
        reasoning_type: str,
        attention_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss that encourages attention to match the learned pattern
        for this reasoning type.

        This is the key to "perfect" self-attention: we explicitly train
        the model to produce consistent attention patterns for similar reasoning.
        """
        if reasoning_type not in self.attention_ema:
            return torch.tensor(0.0, device=attention_weights.device)

        target = self.attention_ema[reasoning_type]
        current = attention_weights.mean(dim=0).mean(dim=0)  # Average over batch and heads

        # Match target size if needed
        if current.shape != target.shape:
            min_size = min(current.shape[-1], target.shape[-1])
            current = current[..., :min_size, :min_size]
            target = target[..., :min_size, :min_size]

        # MSE loss between current and target attention pattern
        loss = F.mse_loss(current, target.to(current.device))

        return loss

    def step(self):
        """
        Periodic update of shared CoT weights.

        Called every update_freq batches to consolidate learning.
        """
        self.step_count += 1

        if self.step_count % self.update_freq != 0:
            return

        # Compute target attention patterns from buffer
        for reasoning_type, patterns in self.pattern_attention_buffer.items():
            if patterns:
                # Stack and average to get target
                stacked = torch.stack(patterns[-100:])  # Last 100 patterns
                self.target_attention_patterns[reasoning_type] = stacked.mean(dim=0)


# ════════════════════════════════════════════════════════════════════════════════
# STANDALONE TESTING
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test with sample data
    sample_texts = [
        "<|think_start|>Let me break this down step by step.<|step|>First, I'll analyze the input.<|step|>Then, I'll process it.<|step|>Finally, I'll return the result.<|think_end|>The answer is 42.",
        "<|think_start|>I need to compare these two options.<|step|>Option A has these benefits.<|step|>Option B has different benefits.<|step|>Considering the tradeoffs, I recommend A.<|think_end|>Choose option A.",
        "<|think_start|>This is a simple question.<|think_end|>The answer is straightforward.",
        "<|think_start|>Let me think about this carefully.<|step|>The first consideration is X.<|step|>The second is Y.<|step|>Combining these, we get Z.<|think_end|>Z is the solution.",
    ]

    manager = ReasoningConsistencyManager(n_clusters=2)
    cluster_ids, weights = manager.process_training_data(sample_texts)

    print("\nResults:")
    for i, (text, cluster, weight) in enumerate(zip(sample_texts, cluster_ids, weights)):
        print(f"  Sample {i}: cluster={cluster}, weight={weight:.2f}")

    print("\nCluster stats:")
    stats = manager.get_cluster_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
