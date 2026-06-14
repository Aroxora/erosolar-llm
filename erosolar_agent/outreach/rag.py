"""RAG memory over the full outreach history.

Every contact, sent message, and reply is embedded and appended to a local index
(``data_store/outreach_rag/index.jsonl``) so the drafter can retrieve relevant
prior context for personalization and dedupe. The same records are mirrored to
Firestore by the store layer — this index is the fast local retrieval path.

Embeddings use the repo's :mod:`local_embeddings`. If sentence-transformers/numpy
are unavailable the index still records text (vector omitted) and retrieval falls
back to recency + keyword overlap, so the worker never hard-fails on RAG.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INDEX_DIR = _REPO_ROOT / "data_store" / "outreach_rag"
_INDEX_PATH = _INDEX_DIR / "index.jsonl"

_WORD = re.compile(r"[a-z0-9]+")


def _embed(text: str):
    try:
        import local_embeddings  # heavy; optional

        return [float(x) for x in local_embeddings.embed(text).tolist()]
    except Exception:
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _keyword_overlap(q: str, t: str) -> float:
    qs, ts = set(_WORD.findall(q.lower())), set(_WORD.findall(t.lower()))
    if not qs or not ts:
        return 0.0
    return len(qs & ts) / math.sqrt(len(qs) * len(ts))


class OutreachRAG:
    """Append-only local vector memory of the full outreach history."""

    def __init__(self, path: Path = _INDEX_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[dict] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    self._records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    def add(self, doc_id: str, text: str, meta: dict | None = None) -> dict:
        """Embed + persist one record. Returns the stored record (incl. vector)."""
        rec = {
            "id": doc_id,
            "text": (text or "")[:4000],
            "meta": meta or {},
            "vector": _embed(text or ""),
        }
        self._records.append(rec)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def retrieve(self, query: str, k: int = 4, kind: str | None = None) -> list[dict]:
        """Top-k relevant prior records (cosine if embedded, else keyword)."""
        pool = [r for r in self._records if not kind or r.get("meta", {}).get("kind") == kind]
        if not pool:
            return []
        qv = _embed(query)
        scored = []
        for r in pool:
            if qv and r.get("vector"):
                s = _cosine(qv, r["vector"])
            else:
                s = _keyword_overlap(query, r.get("text", ""))
            scored.append((s, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for s, r in scored[:k] if s > 0]

    def context_for(self, query: str, k: int = 3) -> str:
        """Compact, prompt-ready context block from retrieved records."""
        hits = self.retrieve(query, k=k)
        if not hits:
            return ""
        lines = []
        for h in hits:
            m = h.get("meta", {})
            tag = m.get("kind", "note")
            lines.append(f"[{tag}] {h.get('text', '')[:400]}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._records)
