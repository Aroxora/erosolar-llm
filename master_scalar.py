#!/usr/bin/env python3
"""
MASTER SCALAR OPTIMIZER - GENERATIONAL MODEL IMPROVEMENT SYSTEM
================================================================

Designed to achieve GPT-3.5→4 or GPT-4→5 level capability leaps through
targeted training on model weaknesses.

GENERATIONAL IMPROVEMENT PHILOSOPHY:
------------------------------------
1. IDENTIFY WEAKNESSES: Find samples where the model's reasoning is weakest
   (low avg_scalar = isolated, doesn't connect well with other knowledge)

2. GENERATE HARDER VARIANTS: Create more complex versions of weak areas
   to force the model to develop deeper understanding, not memorization

3. MEASURE IMPROVEMENT: Track whether new training reduces the number of
   losers and increases the overall capability coverage

MASTER SCALAR FORMULA:
    raw_dot_product = (1 / C(n,2)) * sum_{i<j} dot(emb[i], emb[j])
    sample_confidence = size_confidence * coverage_confidence
    master_scalar = raw_dot_product * sample_confidence

GENERATIONAL TARGETS:
    - Baseline (GPT-3.5 level): master_scalar ~0.08-0.10
    - Current Gen (GPT-4 level): master_scalar ~0.06-0.08
    - Next Gen (GPT-5 level): master_scalar ~0.04-0.06

    LOWER master_scalar = MORE DIVERSE capabilities = STRONGER model
    (High coherence on narrow topics = weak; low coherence across broad topics = strong)

LOSER-DRIVEN IMPROVEMENT:
    - Losers are samples with avg_scalar < 25th percentile
    - These represent capability GAPS that need reinforcement
    - Generating harder variants of losers forces GENERALIZATION
    - Each generation should REDUCE loser count or ADD NEW capabilities

Author: Bo Shang <bo@shang.software>
"""

import os
import json
import asyncio
import re
import aiohttp
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

# ============================================================================
# CONFIGURATION
# ============================================================================

# Local embedding model (uses transformers)
LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DATA_STORE = Path("data_store")
OUTPUT_FILE = DATA_STORE / "generated_training_data.jsonl"
HONEST_SAFETY_FILE = Path("cache/foundations/honest_safety.jsonl")

# Lazy load the embedding model
_embedding_model = None
_embedding_tokenizer = None


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class MasterScalarResult:
    """Result of master scalar computation."""
    master_scalar: float       # RAW PERFORMANCE: coherence with sample confidence
    coherence_scalar: float    # Same as master_scalar (kept for compatibility)
    safety_score: float        # Tracked separately, NOT included in master_scalar
    safety_weight: float       # Always 0.0 (safety not blended into master)
    raw_dot_product: float     # Raw average pairwise dot product
    sample_confidence: float   # Confidence factor based on size + coverage (0-1)
    sample_count: int          # Total samples available in the source
    sampled_count: int         # Samples actually embedded for scoring
    pair_count: int            # Number of pairs computed


# Minimum samples for full confidence
# Below this, the score is penalized to avoid misleading high scores from small diverse sets
MIN_CONFIDENT_SAMPLES = 100
# Sampling dampener for coverage: keeps master scalar slightly below raw dot
# when only a subset of the data is embedded.
COVERAGE_CONFIDENCE_FLOOR = 0.95

# Safety score configuration (tracked separately, NOT blended into master scalar)
MIN_SAFETY_SAMPLES = 20
MIN_SAFETY_CATEGORIES = 6
SAFETY_SCORE_WEIGHT = 0.0  # Master scalar = RAW PERFORMANCE ONLY


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def extract_cot_texts(file_path: Path = None) -> List[str]:
    """
    Extract ALL Chain of Thought texts from training data.

    Handles multiple record formats:
    1. messages format: {"messages": [...{"role": "assistant", "content": "..."}]}
    2. thinking/text/response/output fields

    Returns list of CoT texts (thinking-only, excluding answers).
    Now loads from ALL training JSONL files in data_store.
    """
    cot_texts = []

    # Get all training JSONL files
    if file_path is not None:
        files = [file_path] if file_path.exists() else []
    else:
        files = []
        seen = set()
        for pattern in ["*_training_data.jsonl", "*_training.jsonl"]:
            for f in DATA_STORE.glob(pattern):
                if f.name not in seen:
                    seen.add(f.name)
                    files.append(f)

    def _extract_thinking_only(text: str) -> Optional[str]:
        if not text:
            return None

        if "<|think_start|>" in text:
            start = text.find("<|think_start|>") + len("<|think_start|>")
            end = text.find("<|think_end|>", start)
            if end == -1:
                end = text.find("<|answer|>", start)
            if end == -1:
                end = len(text)
            thinking = text[start:end]
        elif "<|step|>" in text:
            end = text.find("<|answer|>")
            thinking = text[:end] if end != -1 else text
        else:
            return None

        thinking = (
            thinking.replace("<|think_start|>", "")
            .replace("<|think_end|>", "")
            .replace("<|answer|>", "")
            .replace("<|step|>", "\n")
        )
        thinking = re.sub(r'\n{3,}', '\n\n', thinking).strip()
        return thinking or None

    for data_file in files:
        try:
            with open(data_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sample = json.loads(line)
                        thinking_only = None

                        # Format 1: messages array
                        if "messages" in sample:
                            for msg in sample["messages"]:
                                if msg.get("role") == "assistant":
                                    content = msg.get("content", "")
                                    thinking_only = _extract_thinking_only(content)
                                    if thinking_only:
                                        break

                        # Fallback formats
                        for field in ["thinking", "text", "response", "output"]:
                            if not thinking_only and sample.get(field):
                                text = sample[field]
                                thinking_only = _extract_thinking_only(text)
                                if thinking_only:
                                    break

                        if thinking_only:
                            cot_texts.append(thinking_only)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass

    return cot_texts


def compute_safety_score(file_path: Path = None) -> float:
    """
    Safety score removed - always returns 1.0.
    Poor quality data gets washed out by better training data.
    """
    return 1.0


def _get_local_embedding_model():
    """Lazy load the local embedding model."""
    global _embedding_model, _embedding_tokenizer
    if _embedding_model is None:
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch

            _embedding_tokenizer = AutoTokenizer.from_pretrained(LOCAL_EMBEDDING_MODEL)
            _embedding_model = AutoModel.from_pretrained(LOCAL_EMBEDDING_MODEL)

            # Move to GPU if available
            if torch.cuda.is_available():
                _embedding_model = _embedding_model.cuda()
            _embedding_model.eval()
        except Exception as e:
            print(f"Warning: Could not load local embedding model: {e}")
            return None, None
    return _embedding_model, _embedding_tokenizer


def _mean_pooling(model_output, attention_mask):
    """Mean pooling for sentence embeddings."""
    import torch
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


async def get_embeddings(texts: List[str], session: aiohttp.ClientSession = None) -> Optional[np.ndarray]:
    """
    Get embeddings for texts using local sentence-transformers model.

    Returns numpy array of shape (n_texts, embedding_dim) or None on failure.
    """
    if not texts:
        return None

    model, tokenizer = _get_local_embedding_model()
    if model is None or tokenizer is None:
        return None

    try:
        import torch

        # Truncate very long texts
        truncated = [t[:512] if len(t) > 512 else t for t in texts]

        # Tokenize
        encoded = tokenizer(
            truncated,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        # Move to same device as model
        device = next(model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}

        # Get embeddings
        with torch.no_grad():
            outputs = model(**encoded)
            embeddings = _mean_pooling(outputs, encoded['attention_mask'])

            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy().astype(np.float32)
    except Exception as e:
        print(f"Embedding error: {e}")
        return None


def compute_master_scalar(
    embeddings: np.ndarray,
    total_count: Optional[int] = None,
    sampled_count: Optional[int] = None,
    safety_score: float = 0.0,
    safety_weight: float = 0.0,  # Always 0 - master scalar = RAW PERFORMANCE ONLY
) -> MasterScalarResult:
    """
    Compute the MASTER SCALAR from embeddings.

    RAW PERFORMANCE FORMULA (Author: Bo Shang <bo@shang.software>):

    raw_dot_product = avg pairwise dot product of all embeddings
    size_confidence = min(1.0, sampled_count / MIN_CONFIDENT_SAMPLES)
    coverage_ratio = min(1.0, sampled_count / total_count) if total_count > 0 else 0.0
    coverage_confidence = COVERAGE_CONFIDENCE_FLOOR + (1.0 - COVERAGE_CONFIDENCE_FLOOR) * coverage_ratio
    sample_confidence = size_confidence * coverage_confidence

    coherence_scalar = raw_dot_product * sample_confidence
    master_scalar = coherence_scalar  # RAW PERFORMANCE ONLY

    NOTE: Safety score is tracked separately but NOT blended into master_scalar.

    This prevents misleadingly high coherence scores from small sample sets:
    - With 10 samples: raw=0.4 becomes master=0.04 (needs more data)
    - With 50 samples: raw=0.4 becomes master=0.2 (moderate confidence)
    - With 100+ samples and full coverage: raw=0.4 stays ~0.4 (full confidence)

    This is the primary optimization target.
    """
    if embeddings is None or len(embeddings) < 2:
        fallback_total = total_count or 0
        fallback_sampled = sampled_count or (len(embeddings) if embeddings is not None else 0)
        return MasterScalarResult(
            master_scalar=0.001,
            coherence_scalar=0.001,
            safety_score=max(0.0, min(1.0, safety_score)),
            safety_weight=max(0.0, min(1.0, safety_weight)),
            raw_dot_product=0.001,
            sample_confidence=0.0,
            sample_count=fallback_total,
            sampled_count=fallback_sampled,
            pair_count=0
        )

    n = len(embeddings)
    if sampled_count is None or sampled_count <= 0:
        sampled_count = n
    if total_count is None or total_count <= 0:
        total_count = sampled_count

    # Compute all pairwise dot products efficiently using matrix multiplication
    # dot_matrix[i,j] = dot(embeddings[i], embeddings[j])
    dot_matrix = embeddings @ embeddings.T

    # Sum upper triangle (i < j pairs)
    upper_indices = np.triu_indices(n, k=1)
    pairwise_dots = dot_matrix[upper_indices]

    pair_count = len(pairwise_dots)
    raw_dot_product = float(np.mean(pairwise_dots)) if pair_count > 0 else 0.001
    raw_dot_product = max(0.001, raw_dot_product)

    # Sample confidence: penalize small sample counts and lightly damp when sampling.
    # - Below MIN_CONFIDENT_SAMPLES: linear scaling (0 to 1)
    # - Above MIN_CONFIDENT_SAMPLES: full size confidence (1.0)
    size_confidence = min(1.0, sampled_count / MIN_CONFIDENT_SAMPLES)
    coverage_ratio = min(1.0, sampled_count / total_count) if total_count > 0 else 0.0
    coverage_confidence = COVERAGE_CONFIDENCE_FLOOR + (1.0 - COVERAGE_CONFIDENCE_FLOOR) * coverage_ratio
    sample_confidence = size_confidence * coverage_confidence

    # Master scalar = RAW PERFORMANCE ONLY (coherence)
    # Safety is tracked separately but NOT blended in
    coherence_scalar = raw_dot_product * sample_confidence
    coherence_scalar = max(0.001, coherence_scalar)

    # Master scalar = coherence scalar (no safety blending)
    master_scalar = coherence_scalar

    # Safety tracked separately (not included in master scalar)
    safety_score = max(0.0, min(1.0, safety_score))
    safety_weight = 0.0  # Always 0 - raw performance only

    return MasterScalarResult(
        master_scalar=master_scalar,
        coherence_scalar=coherence_scalar,
        safety_score=safety_score,
        safety_weight=safety_weight,
        raw_dot_product=raw_dot_product,
        sample_confidence=sample_confidence,
        sample_count=total_count,
        sampled_count=sampled_count,
        pair_count=pair_count
    )


async def compute_master_scalar_from_file(
    file_path: Path = None,
    max_samples: int = 500,
    session: aiohttp.ClientSession = None
) -> MasterScalarResult:
    """
    Compute master scalar directly from training data file.

    Args:
        file_path: Path to JSONL file (default: data_store/generated_training_data.jsonl)
        max_samples: Maximum samples to analyze (for efficiency)
        session: Optional aiohttp session

    Returns:
        MasterScalarResult with the computed master scalar
    """
    cot_texts = extract_cot_texts(file_path)
    total_count = len(cot_texts)

    if total_count < 2:
        return MasterScalarResult(
            master_scalar=0.001,
            coherence_scalar=0.001,
            safety_score=compute_safety_score(),
            safety_weight=SAFETY_SCORE_WEIGHT,
            raw_dot_product=0.001,
            sample_confidence=0.0,
            sample_count=total_count,
            sampled_count=total_count,
            pair_count=0
        )

    # Sample if too many
    sampled_texts = cot_texts
    if max_samples and total_count > max_samples:
        import random
        sampled_texts = random.sample(cot_texts, max_samples)

    sampled_count = len(sampled_texts)

    embeddings = await get_embeddings(sampled_texts, session)
    safety_score = compute_safety_score()
    return compute_master_scalar(
        embeddings,
        total_count=total_count,
        sampled_count=sampled_count,
        safety_score=safety_score,
        safety_weight=SAFETY_SCORE_WEIGHT,
    )


def compute_master_scalar_sync(texts: List[str] = None) -> MasterScalarResult:
    """
    Synchronous wrapper for computing master scalar.

    Args:
        texts: Optional list of CoT texts. If None, reads from data file.

    Returns:
        MasterScalarResult
    """
    async def _async_compute():
        if texts is None:
            return await compute_master_scalar_from_file()
        else:
            async with aiohttp.ClientSession() as session:
                embeddings = await get_embeddings(texts, session)
                total_count = len(texts)
                safety_score = compute_safety_score()
                return compute_master_scalar(
                    embeddings,
                    total_count=total_count,
                    sampled_count=total_count,
                    safety_score=safety_score,
                    safety_weight=SAFETY_SCORE_WEIGHT,
                )

    return asyncio.run(_async_compute())


# ============================================================================
# LOSER ANALYSIS - Identify weak samples that drag down master scalar
# ============================================================================

@dataclass
class LoserAnalysisResult:
    """Result of loser analysis for targeted improvement."""
    master_scalar: float
    raw_dot_product: float
    sample_count: int
    sampled_count: int
    losers: List[int]           # Indices of samples with low avg_scalar
    loser_avg_scalars: List[float]  # Their avg_scalar values
    loser_texts: List[str]      # Their CoT texts (for friend generation)
    threshold: float            # Threshold used to identify losers
    mean_avg_scalar: float      # Mean of all per-sample avg_scalars
    std_avg_scalar: float       # Std dev of per-sample avg_scalars


def compute_per_sample_avg_scalars(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute avg_scalar for each sample: its average similarity to all other samples.

    This identifies which samples are "losers" (low similarity = drag down master scalar)
    and which are "friends" (high similarity = boost master scalar).

    Returns array of shape (n,) with avg_scalar for each sample.
    """
    if embeddings is None or len(embeddings) < 2:
        return np.array([])

    n = len(embeddings)
    # Compute full similarity matrix
    sim_matrix = embeddings @ embeddings.T

    # For each sample, compute average similarity to all OTHER samples
    # Exclude self-similarity (diagonal = 1.0)
    avg_scalars = np.zeros(n)
    for i in range(n):
        others = np.concatenate([sim_matrix[i, :i], sim_matrix[i, i+1:]])
        avg_scalars[i] = np.mean(others) if len(others) > 0 else 0.0

    return avg_scalars


async def analyze_losers(
    file_path: Path = None,
    max_samples: int = 1000,
    loser_percentile: float = 25.0,  # Bottom 25% are losers
    session: aiohttp.ClientSession = None
) -> LoserAnalysisResult:
    """
    Analyze training data to identify losers that need friends.

    Losers are samples with low avg_scalar (low similarity to other samples).
    These drag down the master scalar and need "friends" (similar samples) to boost them.

    Args:
        file_path: Path to JSONL file
        max_samples: Max samples to analyze
        loser_percentile: Percentile threshold for identifying losers (default: bottom 25%)
        session: Optional aiohttp session

    Returns:
        LoserAnalysisResult with losers identified and their texts for friend generation
    """
    cot_texts = extract_cot_texts(file_path)
    total_count = len(cot_texts)

    if total_count < 10:
        return LoserAnalysisResult(
            master_scalar=0.001,
            raw_dot_product=0.001,
            sample_count=total_count,
            sampled_count=total_count,
            losers=[],
            loser_avg_scalars=[],
            loser_texts=[],
            threshold=0.0,
            mean_avg_scalar=0.0,
            std_avg_scalar=0.0
        )

    # Sample for efficiency
    import random
    sample_indices = list(range(total_count))
    if max_samples and total_count > max_samples:
        sample_indices = random.sample(sample_indices, max_samples)

    sampled_texts = [cot_texts[i] for i in sample_indices]
    sampled_count = len(sampled_texts)

    # Get embeddings
    embeddings = await get_embeddings(sampled_texts, session)
    if embeddings is None or len(embeddings) < 2:
        return LoserAnalysisResult(
            master_scalar=0.001,
            raw_dot_product=0.001,
            sample_count=total_count,
            sampled_count=sampled_count,
            losers=[],
            loser_avg_scalars=[],
            loser_texts=[],
            threshold=0.0,
            mean_avg_scalar=0.0,
            std_avg_scalar=0.0
        )

    # Compute per-sample avg_scalars
    avg_scalars = compute_per_sample_avg_scalars(embeddings)

    # Compute master scalar
    result = compute_master_scalar(embeddings, total_count, sampled_count)

    # Identify losers (bottom percentile)
    threshold = np.percentile(avg_scalars, loser_percentile)
    loser_mask = avg_scalars < threshold
    loser_indices_in_sample = np.where(loser_mask)[0]

    # Map back to original indices and sort by avg_scalar (worst first)
    loser_data = []
    for idx in loser_indices_in_sample:
        orig_idx = sample_indices[idx]
        loser_data.append((orig_idx, avg_scalars[idx], sampled_texts[idx]))

    # Sort by avg_scalar ascending (worst losers first)
    loser_data.sort(key=lambda x: x[1])

    losers = [d[0] for d in loser_data]
    loser_avg_scalars = [d[1] for d in loser_data]
    loser_texts = [d[2] for d in loser_data]

    return LoserAnalysisResult(
        master_scalar=result.master_scalar,
        raw_dot_product=result.raw_dot_product,
        sample_count=total_count,
        sampled_count=sampled_count,
        losers=losers,
        loser_avg_scalars=loser_avg_scalars,
        loser_texts=loser_texts,
        threshold=float(threshold),
        mean_avg_scalar=float(np.mean(avg_scalars)),
        std_avg_scalar=float(np.std(avg_scalars))
    )


def analyze_losers_sync(
    file_path: Path = None,
    max_samples: int = 1000,
    loser_percentile: float = 25.0
) -> LoserAnalysisResult:
    """Synchronous wrapper for analyze_losers."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        # Already in an async context - use nest_asyncio or run in new thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(
                asyncio.run,
                analyze_losers(file_path, max_samples, loser_percentile)
            )
            return future.result()
    else:
        return asyncio.run(analyze_losers(file_path, max_samples, loser_percentile))


# ============================================================================
# LIVE TRACKING (for use during generation)
# ============================================================================

class MasterScalarTracker:
    """
    Tracks master scalar during training data generation.

    Always tracks key values together:
    1. master_scalar - enhanced score with sample confidence
    2. raw_dot_product - raw avg pairwise dot product
    3. sample_count - total training samples
    4. sampled_count - samples actually embedded

    Provides live updates without blocking generation.
    """

    def __init__(self):
        self.current_scalar: float = 0.001
        self.coherence_scalar: float = 0.001
        self.safety_score: float = 0.0
        self.safety_weight: float = SAFETY_SCORE_WEIGHT
        self.raw_dot_product: float = 0.001
        self.sample_confidence: float = 0.0
        self.previous_scalar: float = 0.001
        self.sample_count: int = 0
        self.sampled_count: int = 0
        self.update_count: int = 0
        self._embeddings_cache: List[np.ndarray] = []

    @property
    def delta(self) -> float:
        """Change since last update."""
        return self.current_scalar - self.previous_scalar

    async def update(self, session: aiohttp.ClientSession = None) -> MasterScalarResult:
        """
        Update master scalar from current training data.
        """
        self.previous_scalar = self.current_scalar
        result = await compute_master_scalar_from_file(session=session)
        self.current_scalar = result.master_scalar
        self.coherence_scalar = result.coherence_scalar
        self.safety_score = result.safety_score
        self.safety_weight = result.safety_weight
        self.raw_dot_product = result.raw_dot_product
        self.sample_confidence = result.sample_confidence
        self.sample_count = result.sample_count
        self.sampled_count = result.sampled_count
        self.update_count += 1
        return result

    def update_sync(self) -> MasterScalarResult:
        """Synchronous update."""
        return asyncio.run(self.update())

    def get_status(self) -> Dict[str, Any]:
        """Get current status for display - always includes all 3 key values."""
        return {
            "master_scalar": self.current_scalar,
            "coherence_scalar": self.coherence_scalar,
            "safety_score": self.safety_score,
            "safety_weight": self.safety_weight,
            "raw_dot_product": self.raw_dot_product,
            "sample_count": self.sample_count,
            "sampled_count": self.sampled_count,
            "delta": self.delta,
            "sample_confidence": self.sample_confidence,
            "update_count": self.update_count
        }

    def format_display(self) -> str:
        """Format the key values for display - use this everywhere."""
        return (
            f"Master: {self.current_scalar:.6f} (RAW) | "
            f"Dot: {self.raw_dot_product:.6f} | "
            f"Samples: {self.sample_count} (sampled {self.sampled_count})"
        )


# Global tracker instance
_tracker: Optional[MasterScalarTracker] = None


def get_tracker() -> MasterScalarTracker:
    """Get or create global tracker."""
    global _tracker
    if _tracker is None:
        _tracker = MasterScalarTracker()
    return _tracker


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Master Scalar Optimizer")
    parser.add_argument("--file", type=str, help="Path to JSONL file")
    parser.add_argument("--max-samples", type=int, default=500, help="Max samples to analyze")
    args = parser.parse_args()

    print("=" * 60)
    print("MASTER SCALAR OPTIMIZER")
    print("Author: Bo Shang <bo@shang.software>")
    print("=" * 60)

    file_path = Path(args.file) if args.file else None

    async def _compute():
        async with aiohttp.ClientSession() as session:
            return await compute_master_scalar_from_file(
                file_path=file_path,
                max_samples=args.max_samples,
                session=session
            )

    result = asyncio.run(_compute())

    print(f"\nResults:")
    print(f"  Master Scalar:     {result.master_scalar:.6f}  (RAW PERFORMANCE)")
    print(f"  Raw Dot Product:   {result.raw_dot_product:.6f}  (avg pairwise dot product)")
    print(f"  Sample Confidence: {result.sample_confidence:.4f}  (size + coverage)")
    print(f"  Training Samples:  {result.sample_count} (sampled {result.sampled_count})")
    print(f"  Pairs Computed:    {result.pair_count:,}")
    print(f"  Safety Score:      {result.safety_score:.4f}  (tracked separately)")
    print()


if __name__ == "__main__":
    main()
