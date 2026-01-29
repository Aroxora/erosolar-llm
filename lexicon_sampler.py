#!/usr/bin/env python3
"""
Lexicon Sampler - rotate through large vocabularies without enumerating all.

This creates a stable, windowed sample from a lexicon file (e.g., 80k nouns)
and persists offset state so successive runs cover more of the list.
"""

import argparse
import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import List, Dict


DEFAULT_OUTPUT = "optional_unverified_concepts/lexicon_sample.json"
DEFAULT_STATE = "cache/lexicon_sampler/state.json"

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "so",
    "of", "in", "on", "at", "to", "for", "from", "by", "with", "about",
    "as", "is", "are", "was", "were", "be", "been", "being",
}


def read_lexicon(path: Path) -> List[str]:
    if not path.exists():
        return []
    if path.suffix.lower() == ".json":
        with path.open() as f:
            data = json.load(f)
        if isinstance(data, dict):
            return list(data.keys())
        if isinstance(data, list):
            return [str(x) for x in data]
        return []
    words = []
    with path.open() as f:
        for line in f:
            w = line.strip()
            if w:
                words.append(w)
    return words


def normalize(words: List[str], min_len: int = 3, max_len: int = 40) -> List[str]:
    cleaned = []
    for w in words:
        w = w.strip()
        if not w:
            continue
        if len(w) < min_len or len(w) > max_len:
            continue
        if not re.fullmatch(r"[A-Za-z][A-Za-z\\-']*", w):
            continue
        if w.lower() in STOP_WORDS:
            continue
        if w.islower():
            w = w.title()
        cleaned.append(w)
    return sorted(set(cleaned))


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_state(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        with path.open() as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def sample_window(words: List[str], sample_size: int, seed: int, state: Dict) -> List[str]:
    rng = random.Random(seed)
    words = words[:]
    rng.shuffle(words)

    offset = state.get("offset", 0)
    if offset >= len(words):
        offset = 0

    sample = []
    for i in range(min(sample_size, len(words))):
        sample.append(words[(offset + i) % len(words)])

    state["offset"] = (offset + sample_size) % len(words)
    return sample


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample a rotating lexicon slice.")
    parser.add_argument("--lexicon", required=False, help="Path to lexicon file")
    parser.add_argument("--sample-size", type=int, default=500,
                        help="Number of words to sample")
    parser.add_argument("--seed", type=int, default=42,
                        help="Shuffle seed for stable rotation")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output JSON path")
    parser.add_argument("--state", default=DEFAULT_STATE,
                        help="State file to persist offset")
    parser.add_argument("--min-length", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=40)
    args = parser.parse_args()

    lexicon_path = Path(args.lexicon) if args.lexicon else None
    if not lexicon_path or not lexicon_path.exists():
        default_path = Path("/usr/share/dict/words")
        if default_path.exists():
            lexicon_path = default_path
        else:
            print("[lexicon_sampler] no lexicon file found, skipping")
            return 0

    raw_words = read_lexicon(lexicon_path)
    words = normalize(raw_words, min_len=args.min_length, max_len=args.max_length)
    if not words:
        print("[lexicon_sampler] no valid words after filtering")
        return 0

    state_path = Path(args.state)
    state = load_state(state_path)
    source_hash = file_hash(lexicon_path)
    state_key = f"{lexicon_path}:{source_hash}:{args.seed}:{args.sample_size}"
    if state.get("key") != state_key:
        state = {"key": state_key, "offset": 0}

    sample = sample_window(words, args.sample_size, args.seed, state)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(sample, f, indent=2)

    save_state(state_path, state)
    print(f"[lexicon_sampler] sampled {len(sample)} words -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
