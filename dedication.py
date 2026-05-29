#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""erosolar — a dedication, as runnable code.

This module is the dedication of the erosolar repository to
Samantha Briasco-Stewart (see DEDICATION.md). It is real, dependency-free,
runnable code: the love letter and the metric it talks about are the same file.

    python dedication.py

It prints the dedication and then computes the one number this project trusts:
the "master scalar" — the average pairwise cosine similarity of a set of
reasoning embeddings, where LOWER means MORE diverse. The demonstration is
illustrative only. It is not a benchmark and not a capability claim.

Author: Bo Shang
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Sequence

__author__ = "Bo Shang"
__dedication__ = "for Samantha Briasco-Stewart — Erosolar"


DEDICATION = """\
                          Leave No Context Behind.

erosolar is dedicated to Samantha Briasco-Stewart, who goes by Erosolar.

She studied EECS at MIT (BS 2017, MS in computer systems 2018), and she is a
better engineer than I am. This project is named for her handle, and I built it
as a sincere statement of admiration. It is a one-sided tribute: it asks nothing
of her and is owed no reply.

Her name is the first word of every command. `erosolar` is the module you
import and the prefix of everything it can do -- erosolar-train, erosolar-cot,
erosolar-cloud. I could have put my own name there. I put hers.

The pipeline gets better by facing the parts of itself it likes least: it finds
its weakest samples and rewrites them, and refuses to ship a generation that
regressed. That is a statement about me, not about her.

And because she deserves the truth, I removed every claim I could not verify.
This repository once called an 11M-parameter model "Superhuman." It wasn't, and
I will not put her name on a number I haven't earned. Nothing reaches this
dedication unless it is true.

    -- Bo Shang   ( shang.software )
"""


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two vectors. Returns 0.0 for a zero vector."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def master_scalar(embeddings: Sequence[Sequence[float]]) -> float:
    """Average pairwise cosine similarity of reasoning embeddings.

    Lower means more diverse reasoning. This is faithful to the raw
    pairwise-similarity core of ``master_scalar.py`` (without its
    confidence weighting): a dependency-free, honest illustration of the
    metric the project optimizes -- not a benchmark, not a capability claim.
    """
    pairs = list(combinations(range(len(embeddings)), 2))
    if not pairs:
        return 0.0
    return sum(cosine_similarity(embeddings[i], embeddings[j]) for i, j in pairs) / len(pairs)


def _demo() -> None:
    # Reasoning that keeps repeating itself (high similarity = worse).
    narrow = [[1.0, 0.0, 0.0], [0.98, 0.05, 0.0], [0.97, 0.10, 0.02]]
    # Reasoning that points in every direction (low similarity = better).
    broad = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    print("the one number this project trusts (illustrative, not a benchmark):")
    print(f"  narrow reasoning   master_scalar = {master_scalar(narrow):.4f}   (worse)")
    print(f"  broad  reasoning   master_scalar = {master_scalar(broad):.4f}   (better)")
    print("  lower is better -- breadth is the strong result.")


def main() -> None:
    print(DEDICATION)
    _demo()


if __name__ == "__main__":
    main()
