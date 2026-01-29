#!/usr/bin/env python3
"""
LOCAL EMBEDDINGS MODULE
========================
Provides local embedding computation using sentence-transformers.
Replaces all OpenAI embedding API calls with local models.

Model: sentence-transformers/all-MiniLM-L6-v2
- 384-dimensional embeddings
- Fast inference (~14K sentences/sec on GPU)
- Good quality for semantic similarity

Usage:
    from local_embeddings import embed, embed_batch, similarity, get_model

    # Single text
    emb = embed("Hello world")

    # Batch (more efficient)
    embs = embed_batch(["Hello", "World"])

    # Similarity
    sim = similarity(emb1, emb2)

    # Test
    python local_embeddings.py --test

Author: Bo Shang <bo@shang.software>
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import List, Union, Optional
import threading

# Model configuration
MODEL_NAME = os.environ.get("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 produces 384-dim embeddings

# Thread-safe model singleton
_model = None
_model_lock = threading.Lock()


def get_model():
    """
    Get the sentence-transformers model (lazy loading, thread-safe).

    Returns:
        SentenceTransformer model instance
    """
    global _model

    if _model is not None:
        return _model

    with _model_lock:
        # Double-check after acquiring lock
        if _model is not None:
            return _model

        try:
            from sentence_transformers import SentenceTransformer
            print(f"[LocalEmbeddings] Loading model: {MODEL_NAME}")
            _model = SentenceTransformer(MODEL_NAME)
            print(f"[LocalEmbeddings] Model loaded successfully (dim={EMBEDDING_DIM})")
            return _model
        except ImportError:
            print("[LocalEmbeddings] ERROR: sentence-transformers not installed")
            print("[LocalEmbeddings] Install with: pip install sentence-transformers")
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install with: pip install sentence-transformers"
            )


def embed(text: str) -> np.ndarray:
    """
    Embed a single text string.

    Args:
        text: Text to embed

    Returns:
        Normalized embedding as numpy array (384 dimensions)
    """
    model = get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return np.array(embedding, dtype=np.float32)


def embed_batch(texts: List[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
    """
    Embed a batch of texts efficiently.

    Args:
        texts: List of texts to embed
        batch_size: Batch size for encoding (default: 64)
        show_progress: Show progress bar

    Returns:
        Normalized embeddings as numpy array (N x 384)
    """
    if not texts:
        return np.array([], dtype=np.float32).reshape(0, EMBEDDING_DIM)

    model = get_model()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=show_progress
    )
    return np.array(embeddings, dtype=np.float32)


def similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.
    Since embeddings are normalized, this is just the dot product.

    Args:
        emb1: First embedding
        emb2: Second embedding

    Returns:
        Cosine similarity (-1 to 1, typically 0 to 1 for text)
    """
    emb1 = np.array(emb1, dtype=np.float32).flatten()
    emb2 = np.array(emb2, dtype=np.float32).flatten()
    return float(np.dot(emb1, emb2))


def pairwise_similarities(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute all pairwise similarities for a set of embeddings.

    Args:
        embeddings: N x D embedding matrix (normalized)

    Returns:
        N x N similarity matrix
    """
    embeddings = np.array(embeddings, dtype=np.float32)
    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(1, -1)
    return embeddings @ embeddings.T


def average_pairwise_similarity(embeddings: np.ndarray) -> float:
    """
    Compute average pairwise similarity (excluding self-similarity).
    This is used for computing master scalar coherence.

    Args:
        embeddings: N x D embedding matrix (normalized)

    Returns:
        Average pairwise similarity
    """
    embeddings = np.array(embeddings, dtype=np.float32)
    n = len(embeddings)

    if n < 2:
        return 0.0

    # Compute similarity matrix
    sim_matrix = pairwise_similarities(embeddings)

    # Get upper triangle indices (excluding diagonal)
    upper_idx = np.triu_indices(n, k=1)
    pairwise_sims = sim_matrix[upper_idx]

    return float(np.mean(pairwise_sims))


def find_nearest(query_embedding: np.ndarray, corpus_embeddings: np.ndarray, top_k: int = 5) -> List[tuple]:
    """
    Find nearest neighbors in corpus.

    Args:
        query_embedding: Query embedding (D,)
        corpus_embeddings: Corpus embeddings (N x D)
        top_k: Number of neighbors to return

    Returns:
        List of (index, similarity) tuples sorted by similarity descending
    """
    query = np.array(query_embedding, dtype=np.float32).flatten()
    corpus = np.array(corpus_embeddings, dtype=np.float32)

    if len(corpus) == 0:
        return []

    # Compute similarities
    similarities = corpus @ query

    # Get top-k indices
    top_k = min(top_k, len(similarities))
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    return [(int(idx), float(similarities[idx])) for idx in top_indices]


def compute_outliers(embeddings: np.ndarray, threshold: float = 0.3) -> List[tuple]:
    """
    Find outlier embeddings (low similarity to corpus).
    These are "losers" that need "friends" to improve coherence.

    Args:
        embeddings: N x D embedding matrix
        threshold: Similarity threshold below which a sample is an outlier

    Returns:
        List of (index, avg_similarity) tuples for outliers, sorted by similarity ascending
    """
    embeddings = np.array(embeddings, dtype=np.float32)
    n = len(embeddings)

    if n < 2:
        return []

    # Compute all similarities
    sim_matrix = pairwise_similarities(embeddings)

    outliers = []
    for i in range(n):
        # Average similarity to all other samples
        others_mask = np.ones(n, dtype=bool)
        others_mask[i] = False
        avg_sim = float(np.mean(sim_matrix[i, others_mask]))

        if avg_sim < threshold:
            outliers.append((i, avg_sim))

    # Sort by similarity (lowest first)
    outliers.sort(key=lambda x: x[1])
    return outliers


class LocalEmbeddingCache:
    """
    Simple file-based cache for embeddings.
    Avoids recomputing embeddings for already-seen texts.
    """

    def __init__(self, cache_path: Optional[Path] = None):
        self.cache_path = cache_path or Path("data_store/local_embeddings_cache.npz")
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache = {}
        self._hashes = {}
        self._load_cache()

    def _hash_text(self, text: str) -> str:
        """Create hash for cache lookup."""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()[:16]

    def _load_cache(self):
        """Load cache from disk."""
        if self.cache_path.exists():
            try:
                data = np.load(self.cache_path, allow_pickle=True)
                self._hashes = dict(data.get("hashes", np.array({})).item())
                embeddings = data.get("embeddings", None)
                if embeddings is not None and len(embeddings) > 0:
                    for i, h in enumerate(self._hashes.keys()):
                        if i < len(embeddings):
                            self._cache[h] = embeddings[i]
                print(f"[LocalEmbeddings] Loaded {len(self._cache)} cached embeddings")
            except Exception as e:
                print(f"[LocalEmbeddings] Cache load error: {e}")

    def _save_cache(self):
        """Save cache to disk."""
        if not self._cache:
            return
        try:
            hashes = {h: i for i, h in enumerate(self._cache.keys())}
            embeddings = np.array(list(self._cache.values()))
            np.savez(self.cache_path, hashes=hashes, embeddings=embeddings)
        except Exception as e:
            print(f"[LocalEmbeddings] Cache save error: {e}")

    def get(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding for text."""
        h = self._hash_text(text)
        return self._cache.get(h)

    def put(self, text: str, embedding: np.ndarray):
        """Cache embedding for text."""
        h = self._hash_text(text)
        self._cache[h] = embedding

    def embed_with_cache(self, texts: List[str]) -> np.ndarray:
        """
        Embed texts, using cache when available.

        Args:
            texts: List of texts to embed

        Returns:
            Embeddings (N x D)
        """
        results = []
        to_embed = []
        to_embed_indices = []

        # Check cache first
        for i, text in enumerate(texts):
            cached = self.get(text)
            if cached is not None:
                results.append((i, cached))
            else:
                to_embed.append(text)
                to_embed_indices.append(i)

        # Embed missing texts
        if to_embed:
            new_embeddings = embed_batch(to_embed)
            for text, emb, orig_idx in zip(to_embed, new_embeddings, to_embed_indices):
                self.put(text, emb)
                results.append((orig_idx, emb))

            # Save cache periodically
            if len(self._cache) % 100 == 0:
                self._save_cache()

        # Sort by original index and return
        results.sort(key=lambda x: x[0])
        return np.array([r[1] for r in results])

    def save(self):
        """Force save cache to disk."""
        self._save_cache()


# Global cache instance
_embedding_cache: Optional[LocalEmbeddingCache] = None


def get_embedding_cache() -> LocalEmbeddingCache:
    """Get global embedding cache instance."""
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = LocalEmbeddingCache()
    return _embedding_cache


def embed_with_cache(texts: List[str]) -> np.ndarray:
    """
    Embed texts using global cache.

    Args:
        texts: List of texts

    Returns:
        Embeddings (N x D)
    """
    cache = get_embedding_cache()
    return cache.embed_with_cache(texts)


# ============================================================================
# CLI / TESTING
# ============================================================================

def test_embeddings():
    """Test the local embedding functionality."""
    print("=" * 60)
    print("LOCAL EMBEDDINGS TEST")
    print("=" * 60)

    # Test 1: Single embedding
    print("\n1. Testing single embedding...")
    text = "The quick brown fox jumps over the lazy dog."
    emb = embed(text)
    print(f"   Text: {text[:50]}...")
    print(f"   Embedding shape: {emb.shape}")
    print(f"   Embedding norm: {np.linalg.norm(emb):.4f} (should be ~1.0)")

    # Test 2: Batch embedding
    print("\n2. Testing batch embedding...")
    texts = [
        "Machine learning is fascinating.",
        "Deep learning uses neural networks.",
        "The weather is nice today.",
        "I love programming in Python.",
        "Natural language processing is cool."
    ]
    embs = embed_batch(texts)
    print(f"   Batch size: {len(texts)}")
    print(f"   Embeddings shape: {embs.shape}")

    # Test 3: Similarity
    print("\n3. Testing similarity...")
    pairs = [
        (0, 1, "ML vs DL (similar)"),
        (0, 2, "ML vs Weather (different)"),
        (3, 4, "Python vs NLP (somewhat related)"),
    ]
    for i, j, desc in pairs:
        sim = similarity(embs[i], embs[j])
        print(f"   {desc}: {sim:.4f}")

    # Test 4: Average pairwise similarity
    print("\n4. Testing average pairwise similarity...")
    avg_sim = average_pairwise_similarity(embs)
    print(f"   Average pairwise similarity: {avg_sim:.4f}")

    # Test 5: Find nearest
    print("\n5. Testing nearest neighbor search...")
    query = embed("I enjoy coding")
    nearest = find_nearest(query, embs, top_k=3)
    print(f"   Query: 'I enjoy coding'")
    for idx, sim in nearest:
        print(f"   Nearest #{idx}: '{texts[idx][:40]}...' (sim={sim:.4f})")

    # Test 6: Find outliers
    print("\n6. Testing outlier detection...")
    outliers = compute_outliers(embs, threshold=0.5)
    print(f"   Outliers (threshold=0.5): {len(outliers)}")
    for idx, avg_sim in outliers[:3]:
        print(f"   Outlier #{idx}: '{texts[idx][:40]}...' (avg_sim={avg_sim:.4f})")

    # Test 7: Cache
    print("\n7. Testing embedding cache...")
    cache = LocalEmbeddingCache(Path("data_store/test_cache.npz"))
    cached_embs = cache.embed_with_cache(texts[:3])
    print(f"   First call: embedded {len(texts[:3])} texts")
    cached_embs2 = cache.embed_with_cache(texts[:3])
    print(f"   Second call: used cache for {len(texts[:3])} texts")
    print(f"   Embeddings match: {np.allclose(cached_embs, cached_embs2)}")
    cache.save()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)

    return True


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Local Embeddings Module")
    parser.add_argument("--test", action="store_true", help="Run embedding tests")
    parser.add_argument("--embed", type=str, help="Embed a single text")
    parser.add_argument("--similarity", nargs=2, help="Compute similarity between two texts")
    args = parser.parse_args()

    if args.test:
        test_embeddings()
    elif args.embed:
        emb = embed(args.embed)
        print(f"Embedding shape: {emb.shape}")
        print(f"First 10 values: {emb[:10]}")
    elif args.similarity:
        emb1 = embed(args.similarity[0])
        emb2 = embed(args.similarity[1])
        sim = similarity(emb1, emb2)
        print(f"Text 1: {args.similarity[0]}")
        print(f"Text 2: {args.similarity[1]}")
        print(f"Similarity: {sim:.4f}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
