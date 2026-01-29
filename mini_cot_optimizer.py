#!/usr/bin/env python3
"""
MINI - CoT Master Scalar Attention Calculations
===============================================

Core Principle:
  The primary optimization target is the master scalar derived from thinking-only CoT.
  Other metrics here are attention calculations that explain or refine the master scalar.

The key insight: Train HOW to think, not WHAT to know.
- Standard training: Memorize (prompt, answer) pairs
- CoT training: Learn reasoning PATTERNS that generalize

Attention calculations:
  - Cross-step attention (r→r): Reasoning steps attend to each other
  - Answer grounding (a→r): Answers attend to reasoning steps
  - Pattern centralization: Similar problems use similar reasoning structures

Author: Bo Shang <bo@shang.software>
"""

import os
import sys
import json
import time
import asyncio
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
import threading

# ============================================================================
# CONFIGURATION
# ============================================================================
WORKING_DIR = Path.cwd()
DATA_STORE = WORKING_DIR / "data_store"
TRAINING_DATA = DATA_STORE / "generated_training_data.jsonl"
EMBEDDINGS_CACHE = DATA_STORE / "embeddings.json"
COT_PATTERNS_CACHE = DATA_STORE / "cot_patterns.json"

# Model configuration - using deepseek-reasoner
MODEL = os.environ.get("MINI_MODEL", "deepseek-reasoner")

# Local embeddings (replaces OpenAI text-embedding-3-small)
try:
    from local_embeddings import embed_batch, embed, similarity, average_pairwise_similarity
    LOCAL_EMBEDDINGS_AVAILABLE = True
except ImportError:
    LOCAL_EMBEDDINGS_AVAILABLE = False
    print("[CoTOptimizer] WARNING: local_embeddings not available. pip install sentence-transformers")

# CoT Tokens
THINK_START = "<|think_start|>"
THINK_END = "<|think_end|>"
STEP = "<|step|>"
ANSWER = "<|answer|>"

# Thread pool
MAX_THREADS = 10
_thread_pool: Optional[ThreadPoolExecutor] = None

# Colors
class C:
    CYAN = '\033[0;36m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    NC = '\033[0m'


def get_thread_pool() -> ThreadPoolExecutor:
    global _thread_pool
    if _thread_pool is None:
        _thread_pool = ThreadPoolExecutor(max_workers=MAX_THREADS)
    return _thread_pool


# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

@dataclass
class CoTSample:
    """A Chain-of-Thought training sample."""
    instruction: str
    thinking: str  # Full thinking block
    steps: List[str]  # Individual reasoning steps
    answer: str
    category: str = "general"
    embedding: Optional[List[float]] = None
    attention_score: float = 0.0  # Self-attention quality
    cluster_id: int = -1  # Which reasoning pattern cluster

    def to_training_format(self) -> str:
        """Convert to training format with special tokens."""
        thinking_block = f"{THINK_START}\n"
        for i, step in enumerate(self.steps):
            thinking_block += f"{step}\n"
            if i < len(self.steps) - 1:
                thinking_block += f"{STEP}\n"
        thinking_block += f"{THINK_END}\n"
        return f"<|user|>\n{self.instruction}\n<|end_turn|>\n<|assistant|>\n{thinking_block}{ANSWER}\n{self.answer}\n<|end_turn|>"

    @classmethod
    def from_training_format(cls, text: str, category: str = "general") -> Optional['CoTSample']:
        """Parse from training format."""
        try:
            # Extract instruction
            if "<|user|>" in text and "<|end_turn|>" in text:
                user_start = text.index("<|user|>") + len("<|user|>")
                user_end = text.index("<|end_turn|>")
                instruction = text[user_start:user_end].strip()
            else:
                return None

            # Extract thinking and answer
            if THINK_START in text and THINK_END in text:
                think_start = text.index(THINK_START) + len(THINK_START)
                think_end = text.index(THINK_END)
                thinking = text[think_start:think_end].strip()
                steps = [s.strip() for s in thinking.split(STEP) if s.strip()]
            else:
                thinking = ""
                steps = []

            if ANSWER in text:
                ans_start = text.index(ANSWER) + len(ANSWER)
                ans_end = text.rfind("<|end_turn|>") if "<|end_turn|>" in text[ans_start:] else len(text)
                answer = text[ans_start:ans_end].strip()
            else:
                answer = ""

            return cls(
                instruction=instruction,
                thinking=thinking,
                steps=steps,
                answer=answer,
                category=category
            )
        except:
            return None


@dataclass
class AttentionPattern:
    """A centralized reasoning pattern that multiple samples share."""
    pattern_id: int
    description: str
    step_templates: List[str]  # Template for each step
    sample_ids: List[int] = field(default_factory=list)
    centroid: Optional[List[float]] = None  # Embedding centroid

    def similarity_to(self, embedding: List[float]) -> float:
        """Cosine similarity to pattern centroid."""
        if self.centroid is None or not embedding:
            return 0.0
        centroid = np.array(self.centroid)
        emb = np.array(embedding)
        norm_c = np.linalg.norm(centroid)
        norm_e = np.linalg.norm(emb)
        if norm_c == 0 or norm_e == 0:
            return 0.0
        return float(np.dot(centroid, emb) / (norm_c * norm_e))


@dataclass
class CoTOptimizationState:
    """Global attention state for master scalar metrics."""
    samples: List[CoTSample] = field(default_factory=list)
    patterns: List[AttentionPattern] = field(default_factory=list)
    master_scalar: float = 0.0  # Global coherence score
    cross_step_attention: float = 0.0  # r→r attention quality
    answer_grounding: float = 0.0  # a→r attention quality
    iteration: int = 0

    def save(self, path: Path = COT_PATTERNS_CACHE):
        """Save attention state."""
        data = {
            "master_scalar": self.master_scalar,
            "cross_step_attention": self.cross_step_attention,
            "answer_grounding": self.answer_grounding,
            "iteration": self.iteration,
            "patterns": [
                {
                    "id": p.pattern_id,
                    "description": p.description,
                    "templates": p.step_templates,
                    "sample_ids": p.sample_ids,
                    "centroid": p.centroid
                }
                for p in self.patterns
            ]
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path = COT_PATTERNS_CACHE) -> 'CoTOptimizationState':
        """Load attention state."""
        state = cls()
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                state.master_scalar = data.get("master_scalar", 0.0)
                state.cross_step_attention = data.get("cross_step_attention", 0.0)
                state.answer_grounding = data.get("answer_grounding", 0.0)
                state.iteration = data.get("iteration", 0)
                for p in data.get("patterns", []):
                    state.patterns.append(AttentionPattern(
                        pattern_id=p["id"],
                        description=p["description"],
                        step_templates=p["templates"],
                        sample_ids=p.get("sample_ids", []),
                        centroid=p.get("centroid")
                    ))
            except:
                pass
        return state


# ============================================================================
# COT MASTER SCALAR ATTENTION CALCULATIONS
# ============================================================================

class CoTSelfAttentionOptimizer:
    """
    Provides master scalar attention calculations for CoT reasoning consistency.

    The core insight: A model trained on consistent reasoning PATTERNS
    can generalize to solve problems it has never seen.

    Key attention calculations:
    1. Cross-step attention (r→r): Do reasoning steps reference each other?
    2. Answer grounding (a→r): Does the answer follow from the reasoning?
    3. Pattern consistency: Do similar problems use similar reasoning structures?
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.state = CoTOptimizationState.load()
        self.embeddings: Dict[int, List[float]] = {}
        self._load_embeddings()

    def _load_embeddings(self):
        """Load cached embeddings."""
        if EMBEDDINGS_CACHE.exists():
            try:
                with open(EMBEDDINGS_CACHE) as f:
                    data = json.load(f)
                self.embeddings = {int(k): v for k, v in data.items()}
            except:
                pass

    def _save_embeddings(self):
        """Save embeddings cache."""
        EMBEDDINGS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(EMBEDDINGS_CACHE, 'w') as f:
            json.dump({str(k): v for k, v in self.embeddings.items()}, f)

    def load_training_data(self) -> List[CoTSample]:
        """Load and parse all training samples from ALL JSONL files in data_store."""
        samples = []
        idx = 0

        # Load from all training JSONL files
        seen_files = set()
        for pattern in ["*_training_data.jsonl", "*_training.jsonl"]:
            for data_file in DATA_STORE.glob(pattern):
                if data_file.name in seen_files:
                    continue
                seen_files.add(data_file.name)

                try:
                    with open(data_file) as f:
                        for line in f:
                            try:
                                record = json.loads(line)
                                msgs = record.get("messages", [])
                                if len(msgs) >= 2:
                                    prompt = msgs[0].get("content", "")
                                    response = msgs[1].get("content", "")
                                    text = f"<|user|>\n{prompt}\n<|end_turn|>\n<|assistant|>\n{response}\n<|end_turn|>"
                                    category = record.get("metadata", {}).get("category", "general")
                                    sample = CoTSample.from_training_format(text, category)
                                    if sample:
                                        if idx in self.embeddings:
                                            sample.embedding = self.embeddings[idx]
                                        samples.append(sample)
                                        idx += 1
                            except:
                                continue
                except Exception:
                    continue

        self.state.samples = samples
        return samples

    async def embed_sample(self, sample: CoTSample, idx: int) -> Optional[List[float]]:
        """Get embedding for a sample's CoT reasoning using LOCAL embeddings."""
        if not LOCAL_EMBEDDINGS_AVAILABLE:
            print(f"{C.RED}Local embeddings not available{C.NC}")
            return None

        # Embed the thinking process, not just the answer
        text_to_embed = f"{sample.instruction}\n{sample.thinking}"

        try:
            # Use local sentence-transformers (no API call)
            embedding = embed(text_to_embed[:8000])  # Truncate if needed
            embedding_list = embedding.tolist()
            self.embeddings[idx] = embedding_list
            sample.embedding = embedding_list
            return embedding_list
        except Exception as e:
            print(f"{C.RED}Embedding error: {e}{C.NC}")

        return None

    async def embed_all_samples(self, batch_size: int = 64):
        """Embed all samples using LOCAL sentence-transformers (no API calls)."""
        if not LOCAL_EMBEDDINGS_AVAILABLE:
            print(f"{C.RED}Local embeddings not available - install sentence-transformers{C.NC}")
            return

        samples = self.load_training_data()
        to_embed = [(i, s) for i, s in enumerate(samples) if i not in self.embeddings]

        if not to_embed:
            print(f"{C.GREEN}All {len(samples)} samples already embedded{C.NC}")
            return

        print(f"{C.CYAN}Embedding {len(to_embed)} samples using LOCAL model...{C.NC}")

        # Prepare texts for batch embedding
        texts = [f"{s.instruction}\n{s.thinking}"[:8000] for i, s in to_embed]
        indices = [i for i, s in to_embed]

        # Batch embed using local model (much faster than API calls)
        try:
            embeddings = embed_batch(texts, batch_size=batch_size, show_progress=True)

            for idx, embedding in zip(indices, embeddings):
                embedding_list = embedding.tolist()
                self.embeddings[idx] = embedding_list
                if idx < len(samples):
                    samples[idx].embedding = embedding_list

            print(f"{C.GREEN}Embedded {len(to_embed)} samples locally{C.NC}")
        except Exception as e:
            print(f"{C.RED}Batch embedding error: {e}{C.NC}")
            # Fallback to individual embedding
            for batch_start in range(0, len(to_embed), batch_size):
                batch = to_embed[batch_start:batch_start + batch_size]
                for i, s in batch:
                    await self.embed_sample(s, i)
                print(f"  Embedded {min(batch_start + batch_size, len(to_embed))}/{len(to_embed)}")

        self._save_embeddings()
        print(f"{C.GREEN}Embeddings saved{C.NC}")

    def compute_cross_step_attention(self, sample: CoTSample) -> float:
        """
        Measure how well reasoning steps attend to each other (r→r).

        High score = Steps build on each other logically
        Low score = Steps are disconnected
        """
        if len(sample.steps) < 2:
            return 0.0

        # Heuristic: Check for referential language between steps
        referential_words = ["this", "that", "therefore", "thus", "so", "because",
                           "since", "given", "from", "using", "applying", "we have",
                           "it follows", "which means", "leading to"]

        score = 0.0
        for i, step in enumerate(sample.steps[1:], 1):
            step_lower = step.lower()
            for word in referential_words:
                if word in step_lower:
                    score += 1.0
                    break

        return score / (len(sample.steps) - 1) if len(sample.steps) > 1 else 0.0

    def compute_answer_grounding(self, sample: CoTSample) -> float:
        """
        Measure how well the answer is grounded in reasoning (a→r).

        High score = Answer directly follows from reasoning steps
        Low score = Answer seems disconnected from reasoning
        """
        if not sample.answer or not sample.steps:
            return 0.0

        answer_lower = sample.answer.lower()
        last_step_lower = sample.steps[-1].lower() if sample.steps else ""

        # Check if answer references concepts from final step
        # Simple heuristic: word overlap
        answer_words = set(answer_lower.split())
        step_words = set(last_step_lower.split())

        common = answer_words & step_words
        if not step_words:
            return 0.0

        return len(common) / len(step_words)

    def compute_master_scalar(self) -> Tuple[float, Dict[str, float]]:
        """
        Compute the master coherence scalar for all training data.

        This measures how well the training data teaches consistent reasoning.
        Higher = Model will generalize better to unseen problems.
        """
        samples = self.state.samples or self.load_training_data()
        if not samples:
            return 0.0, {}

        # Compute embedding similarities
        embedded_samples = [s for s in samples if s.embedding]
        if len(embedded_samples) < 2:
            avg_similarity = 0.0
        else:
            embeddings = np.array([s.embedding for s in embedded_samples])
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1.0)
            normalized = embeddings / norms
            sim_matrix = normalized @ normalized.T

            # Upper triangle excluding diagonal
            n = len(embedded_samples)
            upper_idx = np.triu_indices(n, k=1)
            pairwise_sims = sim_matrix[upper_idx]
            avg_similarity = float(np.mean(pairwise_sims))

        # Compute attention metrics
        cross_step_scores = [self.compute_cross_step_attention(s) for s in samples]
        answer_ground_scores = [self.compute_answer_grounding(s) for s in samples]

        avg_cross_step = np.mean(cross_step_scores) if cross_step_scores else 0.0
        avg_answer_ground = np.mean(answer_ground_scores) if answer_ground_scores else 0.0

        # Master scalar combines all metrics
        # Weighted: similarity matters most for generalization
        master = (
            0.5 * avg_similarity +
            0.3 * avg_cross_step +
            0.2 * avg_answer_ground
        )

        self.state.master_scalar = master
        self.state.cross_step_attention = avg_cross_step
        self.state.answer_grounding = avg_answer_ground

        metrics = {
            "master_scalar": master,
            "embedding_similarity": avg_similarity,
            "cross_step_attention": avg_cross_step,
            "answer_grounding": avg_answer_ground,
            "total_samples": len(samples),
            "embedded_samples": len(embedded_samples)
        }

        return master, metrics

    def find_losers(self, threshold: float = 0.3) -> List[Tuple[int, CoTSample, float]]:
        """
        Find samples with low attention scores (losers).

        These are samples that don't fit well with the overall pattern.
        They need "friends" - similar samples to form a coherent cluster.
        """
        samples = self.state.samples or self.load_training_data()
        if not samples:
            return []

        # Compute per-sample scores
        losers = []
        embedded = [s for s in samples if s.embedding]

        if len(embedded) < 2:
            return []

        for idx, sample in enumerate(samples):
            if sample.embedding is None:
                continue

            # Compute similarity to all other samples
            sample_emb = np.array(sample.embedding)
            similarities = []
            for other in embedded:
                if other.embedding and other is not sample:
                    other_emb = np.array(other.embedding)
                    sim = np.dot(sample_emb, other_emb) / (
                        np.linalg.norm(sample_emb) * np.linalg.norm(other_emb) + 1e-8
                    )
                    similarities.append(sim)

            if similarities:
                avg_sim = np.mean(similarities)
                if avg_sim < threshold:
                    losers.append((idx, sample, avg_sim))

        # Sort by similarity (lowest first)
        losers.sort(key=lambda x: x[2])
        return losers

    def cluster_patterns(self, n_clusters: int = 10) -> List[AttentionPattern]:
        """
        Cluster samples by reasoning pattern similarity.

        Samples in the same cluster should use similar reasoning structures.
        This enables the model to learn transferable reasoning patterns.
        """
        samples = [s for s in self.state.samples if s.embedding]
        if len(samples) < n_clusters:
            return []

        # Simple k-means clustering
        embeddings = np.array([s.embedding for s in samples])

        # Initialize centroids randomly
        np.random.seed(42)
        centroid_idx = np.random.choice(len(samples), n_clusters, replace=False)
        centroids = embeddings[centroid_idx].copy()

        # K-means iterations
        for _ in range(10):
            # Assign samples to nearest centroid
            distances = np.linalg.norm(embeddings[:, None] - centroids[None, :], axis=2)
            assignments = np.argmin(distances, axis=1)

            # Update centroids
            for k in range(n_clusters):
                mask = assignments == k
                if mask.sum() > 0:
                    centroids[k] = embeddings[mask].mean(axis=0)

        # Create patterns
        patterns = []
        for k in range(n_clusters):
            mask = assignments == k
            sample_ids = np.where(mask)[0].tolist()
            if not sample_ids:
                continue

            # Get representative sample for description
            cluster_samples = [samples[i] for i in sample_ids]
            representative = cluster_samples[0]

            pattern = AttentionPattern(
                pattern_id=k,
                description=f"Pattern for: {representative.category}",
                step_templates=representative.steps[:3],
                sample_ids=sample_ids,
                centroid=centroids[k].tolist()
            )
            patterns.append(pattern)

            # Update sample cluster assignments
            for i in sample_ids:
                samples[i].cluster_id = k

        self.state.patterns = patterns
        return patterns

    async def generate_friends_for_losers(self, max_friends: int = 5) -> List[CoTSample]:
        """
        Generate similar samples for isolated losers.

        This is the key to CoT centralization: losers become part of
        coherent clusters, improving overall pattern consistency.
        """
        losers = self.find_losers()
        if not losers:
            print(f"{C.GREEN}No losers found - training data is well clustered{C.NC}")
            return []

        print(f"{C.YELLOW}Found {len(losers)} loser samples{C.NC}")

        if not self.api_key:
            print(f"{C.RED}No API key - cannot generate friends{C.NC}")
            return []

        generated = []

        for idx, sample, sim_score in losers[:max_friends]:
            print(f"{C.CYAN}Generating friends for loser #{idx} (sim={sim_score:.3f}){C.NC}")

            # Generate a similar problem with the same reasoning pattern
            prompt = f"""Generate a similar problem that uses the SAME reasoning pattern.

Original problem:
{sample.instruction}

Original reasoning:
{sample.thinking}

Original answer:
{sample.answer}

Generate a NEW problem (different domain/numbers) that uses the EXACT SAME reasoning structure.
Use <|think_start|>, <|step|>, <|think_end|>, and <|answer|> tokens.
The reasoning steps should parallel the original."""

            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "https://api.deepseek.com/chat/completions",
                        headers={
                            "Authorization": f"Bearer {os.environ.get('DEEPSEEK_API_KEY', self.api_key)}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": MODEL,
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": 1000,
                            "temperature": 0.7
                        }
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            response = data["choices"][0]["message"]["content"]
                            new_sample = CoTSample.from_training_format(response, sample.category)
                            if new_sample:
                                generated.append(new_sample)
                                print(f"{C.GREEN}  Generated friend: {new_sample.instruction[:50]}...{C.NC}")
            except Exception as e:
                print(f"{C.RED}  Generation error: {e}{C.NC}")

        return generated

    def optimize(self) -> Dict[str, Any]:
        """
        Run one attention calculation iteration.

        Returns attention metrics.
        """
        print(f"\n{C.BOLD}{C.CYAN}{'═' * 60}{C.NC}")
        print(f"{C.BOLD}{C.CYAN}  COT SELF-ATTENTION OPTIMIZATION{C.NC}")
        print(f"{C.BOLD}{C.CYAN}{'═' * 60}{C.NC}")

        # Load data
        samples = self.load_training_data()
        print(f"{C.DIM}  Loaded {len(samples)} samples{C.NC}")

        # Compute metrics
        master, metrics = self.compute_master_scalar()

        print(f"\n{C.GREEN}  Master Scalar: {master:.4f}{C.NC}")
        print(f"{C.DIM}  ├─ Embedding Similarity: {metrics['embedding_similarity']:.4f}{C.NC}")
        print(f"{C.DIM}  ├─ Cross-step Attention: {metrics['cross_step_attention']:.4f}{C.NC}")
        print(f"{C.DIM}  └─ Answer Grounding: {metrics['answer_grounding']:.4f}{C.NC}")

        # Find losers
        losers = self.find_losers()
        print(f"\n{C.YELLOW}  Losers (low coherence): {len(losers)}{C.NC}")

        # Cluster patterns
        patterns = self.cluster_patterns()
        print(f"{C.CYAN}  Reasoning patterns: {len(patterns)}{C.NC}")

        self.state.iteration += 1
        self.state.save()

        metrics["losers"] = len(losers)
        metrics["patterns"] = len(patterns)
        metrics["iteration"] = self.state.iteration

        return metrics


# ============================================================================
# CLI INTERFACE
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(description="CoT Master Scalar Attention Calculations")
    parser.add_argument("--embed", action="store_true", help="Embed all training samples")
    parser.add_argument("--optimize", action="store_true", help="Run attention calculation iteration")
    parser.add_argument("--losers", action="store_true", help="Find and display losers")
    parser.add_argument("--friends", action="store_true", help="Generate friends for losers")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--max-friends", type=int, default=5, help="Max friends to generate")
    args = parser.parse_args()

    optimizer = CoTSelfAttentionOptimizer()

    if args.embed:
        await optimizer.embed_all_samples()

    elif args.optimize:
        metrics = optimizer.optimize()
        print(f"\n{C.GREEN}Attention calculations complete{C.NC}")
        print(json.dumps(metrics, indent=2))

    elif args.losers:
        optimizer.load_training_data()
        losers = optimizer.find_losers()
        print(f"\n{C.BOLD}Losers (samples with low coherence):{C.NC}")
        for idx, sample, score in losers[:20]:
            print(f"  [{idx}] sim={score:.3f} | {sample.instruction[:60]}...")

    elif args.friends:
        friends = await optimizer.generate_friends_for_losers(args.max_friends)
        print(f"\n{C.GREEN}Generated {len(friends)} friends{C.NC}")

    elif args.status:
        state = CoTOptimizationState.load()
        optimizer.load_training_data()

        print(f"\n{C.BOLD}{C.CYAN}CoT Master Scalar Status{C.NC}")
        print(f"{C.CYAN}{'─' * 40}{C.NC}")
        print(f"  Iteration: {state.iteration}")
        print(f"  Master Scalar: {state.master_scalar:.4f}")
        print(f"  Cross-step Attention: {state.cross_step_attention:.4f}")
        print(f"  Answer Grounding: {state.answer_grounding:.4f}")
        print(f"  Patterns: {len(state.patterns)}")
        print(f"  Samples: {len(optimizer.state.samples)}")
        print(f"  Embeddings cached: {len(optimizer.embeddings)}")

    else:
        # Default: run full attention calculation iteration
        await optimizer.embed_all_samples()
        metrics = optimizer.optimize()

        # Generate friends if there are losers
        losers = optimizer.find_losers()
        if losers:
            friends = await optimizer.generate_friends_for_losers(args.max_friends)
            if friends:
                print(f"\n{C.GREEN}Generated {len(friends)} friends for losers{C.NC}")
                # TODO: Save friends to training data


if __name__ == "__main__":
    asyncio.run(main())
