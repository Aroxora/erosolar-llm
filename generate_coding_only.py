#!/usr/bin/env python3
"""
Generate coding-only training data with long-form explanations and code.

This script:
- Uses a curated, comprehensive prompt catalog of coding tasks
- Generates responses via the DeepSeek Responses API
- Appends to data_store/generated_training_data.jsonl
- Persists state and a seen index to avoid duplicates across runs
"""

import argparse
import asyncio
import dbm
import hashlib
import json
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp

THINK_START = "<|think_start|>"
THINK_END = "<|think_end|>"
ANSWER_MARKER = "<|answer|>"
STEP_MARKER = "<|step|>"

# ════════════════════════════════════════════════════════════════════════════
# CHAIN-OF-THOUGHT MASTER SCALAR CALCULATIONS (EMBEDDING-FREE)
# Same algorithm as mini uses for embedding-free master scalar calculations
# ════════════════════════════════════════════════════════════════════════════

REFERENTIAL_WORDS = {
    "therefore", "thus", "hence", "so", "because", "since",
    "as a result", "consequently", "this means", "which means",
    "we can see", "we find", "we get", "this gives", "leading to",
    "it follows", "given that", "knowing that", "recall that",
    "from this", "using this", "applying", "substituting",
    "first", "second", "third", "next", "then", "finally",
    "step", "now", "let's", "let us", "we need", "we must",
    "the answer", "the result", "the solution", "in conclusion"
}


def compute_cot_attention_score(content: str) -> dict:
    """
    Compute master scalar proxy attention calculations for generated content.

    Metrics (NO embeddings required):
    - cross_step (r→r): Steps reference each other
    - answer_ground (a→r): Answer follows from reasoning
    - structural: Well-formed CoT structure

    Returns dict with scores and weighted total.
    """
    # Extract thinking and answer parts
    if THINK_START in content and THINK_END in content:
        start = content.find(THINK_START) + len(THINK_START)
        end = content.find(THINK_END)
        reasoning_text = content[start:end]
    else:
        reasoning_text = content

    # Extract answer
    if ANSWER_MARKER in content:
        output = content.split(ANSWER_MARKER, 1)[1].strip()
    else:
        output = ""

    # Split into steps
    if STEP_MARKER in reasoning_text:
        steps = [s.strip() for s in reasoning_text.split(STEP_MARKER) if s.strip()]
    else:
        steps = [s.strip() for s in reasoning_text.split("\n") if s.strip() and len(s.strip()) > 20]

    # 1. CROSS-STEP ATTENTION (r→r)
    cross_step_score = 0.0
    if len(steps) >= 2:
        referential_count = 0
        word_overlap_scores = []

        for i, step in enumerate(steps):
            step_lower = step.lower()
            for ref_word in REFERENTIAL_WORDS:
                if ref_word in step_lower:
                    referential_count += 1

            if i > 0:
                current_words = set(step_lower.split())
                prev_words = set(" ".join(steps[:i]).lower().split())
                if current_words and prev_words:
                    overlap = len(current_words & prev_words)
                    union = len(current_words | prev_words)
                    word_overlap_scores.append(overlap / union if union > 0 else 0)

        ref_score = min(1.0, referential_count / 5.0)
        overlap_score = sum(word_overlap_scores) / len(word_overlap_scores) if word_overlap_scores else 0
        overlap_score = min(1.0, overlap_score / 0.3)
        cross_step_score = (ref_score * 0.6 + overlap_score * 0.4)

    # 2. ANSWER GROUNDING (a→r)
    answer_ground_score = 0.0
    if output and reasoning_text:
        output_words = set(output.lower().split())
        reasoning_words = set(reasoning_text.lower().split())
        if output_words and reasoning_words:
            overlap = len(output_words & reasoning_words)
            answer_ground_score = min(1.0, overlap / (len(output_words) * 0.5))
        if steps and output.lower()[:50] in steps[-1].lower():
            answer_ground_score = min(1.0, answer_ground_score + 0.2)

    # 3. STRUCTURAL PATTERNS
    structural_score = 0.0
    structural_score += 0.25 if THINK_START in content else 0
    structural_score += 0.25 if THINK_END in content else 0
    structural_score += 0.3 if STEP_MARKER in content or len(steps) >= 2 else 0
    structural_score += 0.2 if output.strip() else 0
    if len(steps) >= 3:
        structural_score = min(1.0, structural_score + 0.1)
    if len(steps) >= 5:
        structural_score = min(1.0, structural_score + 0.1)

    # WEIGHTED MASTER SCORE
    weights = {"cross_step": 0.30, "answer_ground": 0.25, "structural": 0.45}
    weighted_score = (
        cross_step_score * weights["cross_step"] +
        answer_ground_score * weights["answer_ground"] +
        structural_score * weights["structural"]
    )

    return {
        "cross_step": cross_step_score,
        "answer_ground": answer_ground_score,
        "structural": structural_score,
        "weighted_score": weighted_score,
        "num_steps": len(steps)
    }

def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default

DEFAULT_MODEL = os.environ.get("MODEL", "deepseek-reasoner")
DEFAULT_OUTPUT = Path("data_store/coding_training_data.jsonl")
DEFAULT_STATE = Path("data_store/coding_only_state.json")
DEFAULT_PROMPTS_LOG = Path("data_store/coding_only_prompts.jsonl")
DEFAULT_SEEN_DB = Path("data_store/coding_only_seen.db")
DEFAULT_MAX_OUTPUT_TOKENS = _env_int("LONG_FORM_OUTPUT_TOKENS", _env_int("MAX_OUTPUT_TOKENS", 1600))
LONG_FORM = os.environ.get("LONG_FORM", "").lower() in {"1", "true", "yes"}

SYSTEM_PROMPT = """You are generating training data for a coding assistant.
Only discuss software programming, computer science, or developer tooling.
If the prompt is not coding-related, say so briefly and redirect to a coding answer.

Output format (exact):
<|think_start|>
<|step|>...
<|think_end|>
<|answer|>
...

Rules:
- Use clear, practical explanations and at least one code block when relevant.
- Always wrap code in triple backticks with a language tag (python, javascript, typescript, java, cpp, go, rust, csharp, sql, bash).
- Keep the response factual and focused on implementation details.
"""

if LONG_FORM:
    SYSTEM_PROMPT += (
        "\nLONG-FORM MODE:\n"
        "- Provide expansive, detailed answers with examples and edge cases.\n"
        "- Use available output capacity without adding filler."
    )


def get_api_base() -> str:
    base = os.environ.get("DEEPSEEK_API_BASE", os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    return base.rstrip("/")


def get_chat_completions_url() -> str:
    base = get_api_base()
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return f"{base}/chat/completions"


def parse_chat_completions_output(result: dict) -> str:
    """Parse DeepSeek Chat Completions API response.

    For deepseek-reasoner model:
    - reasoning_content: Contains the CoT reasoning steps
    - content: Contains the final answer

    We combine both for complete response with thinking markers.
    """
    choices = result.get("choices", [])
    for choice in choices:
        message = choice.get("message", {})
        reasoning = message.get("reasoning_content", "")
        content = message.get("content", "")

        # For deepseek-reasoner: combine reasoning + answer
        if reasoning:
            # Format with CoT markers
            parts = []
            parts.append(f"{THINK_START}")
            # Split reasoning into steps
            reasoning_lines = reasoning.strip().split("\n\n")
            for line in reasoning_lines:
                if line.strip():
                    parts.append(f"{STEP_MARKER} {line.strip()}")
            parts.append(f"{THINK_END}")
            if content:
                parts.append(f"{ANSWER_MARKER} {content.strip()}")
            return "\n".join(parts)

        # Fallback for non-reasoner models
        if content:
            return content.strip()
    return ""


def normalize_cot(content: str) -> str:
    content = content.strip()
    if THINK_START not in content:
        content = f"{THINK_START}\n{content}"
    if THINK_END not in content:
        if ANSWER_MARKER in content:
            content = content.replace(ANSWER_MARKER, f"{THINK_END}\n{ANSWER_MARKER}", 1)
        else:
            content = f"{content}\n{THINK_END}"
    if ANSWER_MARKER not in content:
        content = f"{content}\n{ANSWER_MARKER}"
    return content


def extract_answer_text(content: str) -> str:
    if ANSWER_MARKER in content:
        return content.split(ANSWER_MARKER, 1)[1].strip()
    return content.strip()


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", text))


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def prompt_key(prompt: str) -> bytes:
    return prompt_hash(prompt).encode("ascii")


def open_seen_db(path: Path, mode: str = "c"):
    path.parent.mkdir(parents=True, exist_ok=True)
    return dbm.open(str(path), mode)


@dataclass
class CatalogItem:
    prompt: str
    category: str
    language: Optional[str] = None
    topic: Optional[str] = None


def build_prompt_catalog() -> List[CatalogItem]:
    catalog: List[CatalogItem] = []

    def add(prompt: str, category: str, language: Optional[str] = None, topic: Optional[str] = None) -> None:
        catalog.append(CatalogItem(prompt=prompt, category=category, language=language, topic=topic))

    # All major programming languages for comprehensive coverage
    all_langs = [
        "Python", "JavaScript", "TypeScript", "Java", "C++", "Go", "Rust",
        "C#", "Kotlin", "Swift", "Ruby", "PHP", "Scala", "Haskell",
        "Elixir", "Clojure", "F#", "OCaml", "Lua", "Perl", "R",
        "Julia", "Dart", "Zig", "Nim", "Crystal", "V", "Odin"
    ]
    impl_langs = all_langs
    ds_langs = all_langs[:15]  # Most common for DS
    concept_langs = all_langs
    web_langs = ["Python", "JavaScript", "TypeScript", "Go", "Java", "C#", "Ruby", "PHP", "Kotlin", "Elixir", "Rust"]
    test_langs = all_langs[:12]
    pattern_langs = all_langs[:15]
    system_langs = ["Python", "Go", "Java", "Rust", "C++", "TypeScript", "Kotlin", "Scala"]

    algorithms = [
        "binary search",
        "merge sort",
        "quick sort",
        "heap sort",
        "breadth-first search (BFS)",
        "depth-first search (DFS)",
        "Dijkstra's algorithm",
        "A* search",
        "topological sort",
        "union-find (disjoint set)",
        "LRU cache",
        "KMP string search",
        "sliding window maximum",
        "two-pointer technique",
        "0/1 knapsack (dynamic programming)",
        "coin change (dynamic programming)",
        "longest increasing subsequence",
        "edit distance (Levenshtein)",
        "interval merging",
        "prefix sum array",
        # Extended algorithms
        "bubble sort",
        "insertion sort",
        "selection sort",
        "counting sort",
        "radix sort",
        "bucket sort",
        "shell sort",
        "Tim sort",
        "Floyd-Warshall algorithm",
        "Bellman-Ford algorithm",
        "Prim's algorithm",
        "Kruskal's algorithm",
        "Tarjan's algorithm",
        "Kosaraju's algorithm",
        "Boyer-Moore voting algorithm",
        "Kadane's algorithm",
        "Rabin-Karp string matching",
        "Z algorithm",
        "manacher's algorithm",
        "reservoir sampling",
        "Fisher-Yates shuffle",
        "binary exponentiation",
        "matrix exponentiation",
        "Euclidean algorithm (GCD)",
        "extended Euclidean algorithm",
        "sieve of Eratosthenes",
        "Miller-Rabin primality test",
        "convex hull (Graham scan)",
        "line sweep algorithm",
        "Mo's algorithm",
        "sqrt decomposition",
        "heavy-light decomposition",
        "centroid decomposition",
        "ternary search",
        "meet in the middle",
        "bitmasking DP",
        "digit DP",
        "tree DP",
        "probability DP",
        "longest common subsequence",
        "longest palindromic subsequence",
        "matrix chain multiplication",
        "egg dropping problem",
        "word break problem",
        "palindrome partitioning",
        "subset sum",
        "partition problem",
        "rod cutting",
        "maximum subarray (circular)",
        "stock buy sell problems",
        "house robber variants",
    ]

    data_structures = [
        "stack",
        "queue",
        "deque",
        "singly linked list",
        "doubly linked list",
        "hash map",
        "binary search tree",
        "binary heap",
        "trie",
        "graph adjacency list",
        "graph adjacency matrix",
        "segment tree",
        "Fenwick tree (binary indexed tree)",
        "Bloom filter",
        "ring buffer",
        # Extended data structures
        "AVL tree",
        "red-black tree",
        "B-tree",
        "B+ tree",
        "skip list",
        "treap",
        "splay tree",
        "suffix array",
        "suffix tree",
        "persistent data structure",
        "rope (for strings)",
        "interval tree",
        "k-d tree",
        "quad tree",
        "octree",
        "R-tree",
        "min-max heap",
        "Fibonacci heap",
        "binomial heap",
        "pairing heap",
        "disjoint sparse table",
        "sparse table",
        "wavelet tree",
        "link-cut tree",
        "Euler tour tree",
        "van Emde Boas tree",
        "count-min sketch",
        "HyperLogLog",
        "cuckoo filter",
        "XOR linked list",
        "unrolled linked list",
        "self-organizing list",
        "circular buffer with overwrite",
        "priority deque",
        "indexed priority queue",
    ]

    concepts = [
        "recursion",
        "memoization",
        "time complexity and Big-O notation",
        "space complexity",
        "stable vs unstable sorting",
        "in-place algorithms",
        "immutability",
        "idempotency",
        "garbage collection",
        "memory leaks",
        "references vs values",
        "concurrency vs parallelism",
        "race conditions",
        "deadlocks",
        "async/await",
        "event loop",
        "thread pools",
        "serialization vs deserialization",
        "hashing and collisions",
        "caching strategies",
        # Extended concepts
        "closures and lexical scoping",
        "higher-order functions",
        "currying and partial application",
        "lazy evaluation",
        "generators and iterators",
        "coroutines",
        "promises and futures",
        "reactive programming",
        "functional composition",
        "monads and functors",
        "tail call optimization",
        "copy-on-write",
        "reference counting",
        "mark and sweep GC",
        "generational GC",
        "weak references",
        "memory pools",
        "object pooling",
        "flyweight pattern",
        "prototype pattern",
        "dependency injection",
        "inversion of control",
        "aspect-oriented programming",
        "duck typing",
        "structural typing",
        "type inference",
        "variance (covariance/contravariance)",
        "phantom types",
        "newtype pattern",
        "smart pointers",
        "RAII pattern",
        "move semantics",
        "zero-cost abstractions",
        "compile-time computation",
        "template metaprogramming",
        "reflection and introspection",
        "code generation",
        "JIT compilation",
        "AOT compilation",
        "bytecode interpretation",
        "stack vs heap allocation",
        "memory alignment",
        "cache locality",
        "branch prediction",
        "SIMD operations",
        "lock-free programming",
        "wait-free algorithms",
        "compare-and-swap (CAS)",
        "memory barriers",
        "happens-before relationship",
        "eventual consistency",
        "linearizability",
        "serializability",
    ]

    # Design patterns
    design_patterns = [
        "singleton pattern",
        "factory pattern",
        "abstract factory pattern",
        "builder pattern",
        "prototype pattern",
        "adapter pattern",
        "bridge pattern",
        "composite pattern",
        "decorator pattern",
        "facade pattern",
        "proxy pattern",
        "chain of responsibility",
        "command pattern",
        "interpreter pattern",
        "iterator pattern",
        "mediator pattern",
        "memento pattern",
        "observer pattern",
        "state pattern",
        "strategy pattern",
        "template method pattern",
        "visitor pattern",
        "null object pattern",
        "object pool pattern",
        "lazy initialization",
        "multiton pattern",
        "servant pattern",
        "specification pattern",
        "repository pattern",
        "unit of work pattern",
        "data mapper pattern",
        "active record pattern",
        "data transfer object (DTO)",
        "value object pattern",
        "aggregate pattern",
        "domain event pattern",
        "CQRS pattern",
        "event sourcing",
        "saga pattern",
        "circuit breaker pattern",
        "bulkhead pattern",
        "retry pattern",
        "timeout pattern",
        "throttling pattern",
        "rate limiter pattern",
        "sidecar pattern",
        "ambassador pattern",
        "anti-corruption layer",
    ]

    # System design topics
    system_design = [
        "load balancer implementation",
        "rate limiter design",
        "URL shortener design",
        "web crawler design",
        "chat system design",
        "notification system design",
        "search autocomplete",
        "distributed cache design",
        "message queue design",
        "distributed lock design",
        "consistent hashing",
        "bloom filter for caching",
        "leaderboard system",
        "recommendation system",
        "newsfeed system",
        "video streaming system",
        "file storage system",
        "key-value store design",
        "distributed counter",
        "unique ID generator",
        "API gateway design",
        "service discovery",
        "configuration management",
        "logging system design",
        "metrics collection system",
        "distributed tracing",
        "health check system",
        "feature flag system",
        "A/B testing framework",
        "canary deployment system",
    ]

    # Database topics
    database_topics = [
        "SQL joins (inner, outer, cross)",
        "SQL window functions",
        "SQL CTEs (Common Table Expressions)",
        "database indexing strategies",
        "query optimization techniques",
        "database normalization",
        "database denormalization",
        "ACID properties implementation",
        "transaction isolation levels",
        "optimistic locking",
        "pessimistic locking",
        "database sharding",
        "database replication",
        "master-slave replication",
        "multi-master replication",
        "database connection pooling",
        "prepared statements",
        "stored procedures",
        "database triggers",
        "materialized views",
        "database partitioning",
        "time-series database design",
        "graph database queries",
        "document database modeling",
        "key-value store patterns",
        "column-family database design",
        "full-text search implementation",
        "geospatial queries",
        "database migration strategies",
        "schema versioning",
    ]

    # Testing topics
    testing_topics = [
        "unit testing best practices",
        "integration testing strategies",
        "end-to-end testing",
        "test-driven development (TDD)",
        "behavior-driven development (BDD)",
        "property-based testing",
        "mutation testing",
        "fuzz testing",
        "snapshot testing",
        "contract testing",
        "performance testing",
        "load testing",
        "stress testing",
        "chaos engineering",
        "mocking and stubbing",
        "test fixtures",
        "test data factories",
        "parameterized tests",
        "test coverage analysis",
        "code coverage metrics",
    ]

    for lang in impl_langs:
        for algo in algorithms:
            add(
                f"Implement {algo} in {lang}. Provide time and space complexity and a small test.",
                "algorithm_implementation",
                language=lang,
                topic=algo,
            )

    for lang in ds_langs:
        for ds in data_structures:
            add(
                f"Explain how a {ds} works and implement basic operations in {lang}. Include a small example.",
                "data_structure",
                language=lang,
                topic=ds,
            )

    for lang in concept_langs:
        for concept in concepts:
            add(
                f"Explain {concept} in programming terms and show a short {lang} example.",
                "concepts",
                language=lang,
                topic=concept,
            )

    web_tasks = [
        "pagination for a GET /items endpoint",
        "rate limiting middleware for an API",
        "JWT authentication for login and refresh",
        "CORS handling for a browser-facing API",
        "webhook verification and replay protection",
        "file upload handling with size limits",
        "input validation and error responses",
        "idempotent POST requests with idempotency keys",
        "structured logging for request tracing",
        "graceful error handling with HTTP status codes",
    ]

    for lang in web_langs:
        for task in web_tasks:
            add(
                f"Create a minimal {lang} HTTP handler for {task}. Include route, validation, and response codes.",
                "web_api",
                language=lang,
                topic=task,
            )

    sql_prompts = [
        "Write a SQL query to return the top 5 customers by total spend in the last 90 days.",
        "Write a SQL query that finds duplicate emails in a users table.",
        "Write a SQL query to compute a 7-day rolling average of daily sales.",
        "Write a SQL query to return orders with no matching shipment record.",
        "Write a SQL query that ranks products by category using window functions.",
        "Write a SQL query to upsert a user record by email (PostgreSQL).",
        "Write a SQL query to find the most recent login per user.",
        "Write a SQL query to compute retention by cohort month.",
    ]
    for prompt in sql_prompts:
        add(prompt, "sql", language="SQL")

    testing_tasks = [
        "a function that normalizes phone numbers",
        "a function that parses a CSV string into objects",
        "a function that validates a password policy",
        "a function that merges overlapping intervals",
        "a function that debounces a callback",
    ]
    for lang in test_langs:
        for task in testing_tasks:
            add(
                f"Write unit tests in {lang} for {task}. Include edge cases.",
                "testing",
                language=lang,
                topic=task,
            )

    debug_cases = [
        {
            "language": "Python",
            "prompt": (
                "Debug this Python function and fix the off-by-one error. Return corrected code and explain the fix.\n\n"
                "```python\n"
                "def sum_first_n(n):\n"
                "    total = 0\n"
                "    for i in range(1, n):\n"
                "        total += i\n"
                "    return total\n"
                "```\n"
            ),
        },
        {
            "language": "JavaScript",
            "prompt": (
                "Debug this JavaScript function and fix the loop bug. Return corrected code and explain the fix.\n\n"
                "```javascript\n"
                "function lastItem(arr) {\n"
                "  for (let i = 0; i <= arr.length; i++) {\n"
                "    if (i === arr.length) return arr[i];\n"
                "  }\n"
                "}\n"
                "```\n"
            ),
        },
        {
            "language": "Go",
            "prompt": (
                "Debug this Go snippet and fix the panic. Return corrected code and explain the fix.\n\n"
                "```go\n"
                "package main\n"
                "\n"
                "func main() {\n"
                "    var m map[string]int\n"
                "    m[\"a\"] = 1\n"
                "}\n"
                "```\n"
            ),
        },
        {
            "language": "Python",
            "prompt": (
                "Debug this Python function and fix the mutation bug. Return corrected code and explain the fix.\n\n"
                "```python\n"
                "def append_item(item, items=[]):\n"
                "    items.append(item)\n"
                "    return items\n"
                "```\n"
            ),
        },
        {
            "language": "Java",
            "prompt": (
                "Debug this Java method and fix the NullPointerException. Return corrected code and explain the fix.\n\n"
                "```java\n"
                "public int lengthOfName(User user) {\n"
                "    return user.getName().length();\n"
                "}\n"
                "```\n"
            ),
        },
    ]
    for case in debug_cases:
        add(case["prompt"], "debugging", language=case["language"])

    refactor_cases = [
        {
            "language": "Python",
            "prompt": (
                "Refactor this Python function to be clearer and more efficient. Return the improved code and explain changes.\n\n"
                "```python\n"
                "def unique_items(items):\n"
                "    out = []\n"
                "    for i in range(len(items)):\n"
                "        if items[i] not in out:\n"
                "            out.append(items[i])\n"
                "    return out\n"
                "```\n"
            ),
        },
        {
            "language": "JavaScript",
            "prompt": (
                "Refactor this JavaScript function to avoid nested loops. Return the improved code and explain changes.\n\n"
                "```javascript\n"
                "function hasCommon(a, b) {\n"
                "  for (let i = 0; i < a.length; i++) {\n"
                "    for (let j = 0; j < b.length; j++) {\n"
                "      if (a[i] === b[j]) return true;\n"
                "    }\n"
                "  }\n"
                "  return false;\n"
                "}\n"
                "```\n"
            ),
        },
    ]
    for case in refactor_cases:
        add(case["prompt"], "refactor", language=case["language"])

    tooling_prompts = [
        "Explain how to resolve a Git merge conflict with a concrete step-by-step example.",
        "Explain the difference between git rebase and git merge and when to use each.",
        "Write a minimal Dockerfile for a Python web app and explain each layer.",
        "Write a minimal docker-compose.yml for a web app and a database.",
        "Explain how to use git bisect to find a regression with a concrete example.",
        "Explain how to set up a basic CI pipeline for tests and linting.",
    ]
    for prompt in tooling_prompts:
        add(prompt, "tooling")

    # Design patterns across languages
    for lang in pattern_langs:
        for pattern in design_patterns:
            add(
                f"Implement the {pattern} in {lang} with a practical example and explain when to use it.",
                "design_pattern",
                language=lang,
                topic=pattern,
            )

    # System design with code examples
    for lang in system_langs:
        for topic in system_design:
            add(
                f"Design and implement a basic {topic} in {lang}. Include key components and code.",
                "system_design",
                language=lang,
                topic=topic,
            )

    # Database topics
    for topic in database_topics:
        add(
            f"Explain {topic} with practical SQL examples and best practices.",
            "database",
            language="SQL",
            topic=topic,
        )

    # Testing topics across languages
    for lang in test_langs:
        for topic in testing_topics:
            add(
                f"Demonstrate {topic} in {lang} with concrete examples and best practices.",
                "testing_advanced",
                language=lang,
                topic=topic,
            )

    return catalog


def catalog_hash(catalog: List[CatalogItem]) -> str:
    h = hashlib.sha256()
    for item in catalog:
        h.update(item.prompt.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def seed_seen_db(seen_db, prompts_log: Path) -> int:
    seed_marker = b"__seeded__"
    if seed_marker in seen_db:
        return 0
    count = 0
    if prompts_log.exists():
        with open(prompts_log, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt_id = item.get("prompt_id")
                prompt = item.get("prompt")
                if prompt_id:
                    seen_db[str(prompt_id).encode("ascii", "ignore")] = b"1"
                    count += 1
                elif prompt:
                    seen_db[prompt_key(prompt)] = b"1"
                    count += 1
    seen_db[seed_marker] = b"1"
    return count


def load_state(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(path: Path, state: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, ensure_ascii=True, indent=2)


def select_prompts(
    catalog: List[CatalogItem],
    state: Dict[str, object],
    seen_db,
    target: int,
    resume: bool,
) -> List[CatalogItem]:
    if not catalog:
        return []

    if target <= 0:
        selected = []
        for item in catalog:
            if seen_db is None or prompt_key(item.prompt) not in seen_db:
                selected.append(item)
        state["cursor"] = 0
        state["completed"] = len(selected) == 0
        return selected

    start = int(state.get("cursor", 0)) if resume else 0
    idx = start
    scanned = 0
    total = len(catalog)
    selected: List[CatalogItem] = []
    while len(selected) < target and scanned < total:
        item = catalog[idx]
        if seen_db is None or prompt_key(item.prompt) not in seen_db:
            selected.append(item)
        idx = (idx + 1) % total
        scanned += 1

    state["cursor"] = idx
    state["completed"] = scanned >= total and len(selected) == 0
    return selected


async def call_deepseek(
    session: aiohttp.ClientSession,
    api_key: str,
    model: str,
    prompt: str,
    max_output_tokens: int,
    temperature: float,
) -> str:
    """Call DeepSeek Chat Completions API."""
    url = get_chat_completions_url()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_output_tokens,
        "temperature": temperature,
        "stream": False
    }
    async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=180)) as resp:
        resp.raise_for_status()
        data = await resp.json()
    return parse_chat_completions_output(data)


async def generate_record(
    session: aiohttp.ClientSession,
    item: CatalogItem,
    api_key: str,
    model: str,
    min_words: int,
    max_output_tokens: int,
    temperature: float,
    max_retries: int,
) -> Dict[str, object]:
    prompt = item.prompt
    last_content = ""
    for attempt in range(max_retries + 1):
        extra = ""
        if attempt > 0:
            extra = "\nPlease expand with more detail, edge cases, and testing guidance."
        try:
            content = await call_deepseek(
                session=session,
                api_key=api_key,
                model=model,
                prompt=prompt + extra,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            )
            content = normalize_cot(content)
            last_content = content
            answer_text = extract_answer_text(content)
            if min_words <= 0 or word_count(answer_text) >= min_words:
                break
        except Exception as e:
            return {"success": False, "prompt": prompt, "error": str(e)}

    # Compute master scalar proxy (embedding-free calculation)
    cot_scores = compute_cot_attention_score(last_content)

    metadata = {
        "source": "gpt-generated",
        "category": "coding_only",
        "type": "coding_only",
        "coding_category": item.category,
        "language": item.language,
        "topic": item.topic,
        "prompt_id": prompt_hash(prompt),
        "weight": 1.0,
        "has_thinking": THINK_START in last_content,
        "has_answer": ANSWER_MARKER in last_content,
        "has_step": STEP_MARKER in last_content,
        "generated_at": datetime.now().isoformat(),
        "model": model,
        # Master scalar attention metrics (embedding-free)
        "cot_cross_step": cot_scores["cross_step"],
        "cot_answer_ground": cot_scores["answer_ground"],
        "cot_structural": cot_scores["structural"],
        "cot_master_score": cot_scores["weighted_score"],
        "cot_num_steps": cot_scores["num_steps"],
    }
    record = {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": last_content},
        ],
        "metadata": metadata,
    }
    return {"success": True, "prompt": prompt, "record": record}


def append_jsonl(path: Path, records: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def append_prompts_log(path: Path, items: List[CatalogItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for item in items:
            payload = asdict(item)
            payload["prompt_id"] = prompt_hash(item.prompt)
            payload["generated_at"] = datetime.now().isoformat()
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")


async def run(args: argparse.Namespace) -> int:
    if not args.dry_run:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            # Try loading from secrets file
            secrets_file = Path.home() / ".agi" / "secrets.json"
            if secrets_file.exists():
                try:
                    with open(secrets_file) as f:
                        secrets = json.load(f)
                        api_key = secrets.get("DEEPSEEK_API_KEY", "")
                except:
                    pass
        if not api_key:
            raise SystemExit("DEEPSEEK_API_KEY is required (or use --dry-run).")
    else:
        api_key = ""

    catalog = build_prompt_catalog()
    state = load_state(Path(args.state)) if args.resume else {}
    catalog_id = catalog_hash(catalog)
    if state.get("catalog_hash") != catalog_id:
        state["catalog_hash"] = catalog_id
        if not args.resume:
            state["cursor"] = 0

    seen_db = None
    if args.skip_seen:
        seen_db_path = Path(args.seen_db)
        if args.dry_run and not seen_db_path.exists():
            seen_db = None
        else:
            db_mode = "r" if args.dry_run else "c"
            seen_db = open_seen_db(seen_db_path, db_mode)
            if not args.dry_run:
                seed_seen_db(seen_db, Path(args.prompts_log))

    try:
        selected = select_prompts(
            catalog=catalog,
            state=state,
            seen_db=seen_db,
            target=args.target,
            resume=args.resume,
        )

        if args.dry_run:
            for item in selected:
                label = item.category
                lang = f" ({item.language})" if item.language else ""
                print(f"{label}{lang}: {item.prompt}")
            print(f"Dry run complete. Prompts listed: {len(selected)}.")
            return 0

        if not selected:
            save_state(Path(args.state), {
                **state,
                "updated_at": datetime.now().isoformat(),
            })
            print("No new coding prompts to generate (all seen).")
            return 0

        # Process in batches and write incrementally for persistence
        batch_size = args.workers * 2  # Small batches for frequent writes
        total_success = 0
        output_path = Path(args.output)

        async with aiohttp.ClientSession() as session:
            sem = asyncio.Semaphore(args.workers)

            async def bounded_generate(item: CatalogItem) -> Dict[str, object]:
                async with sem:
                    await asyncio.sleep(args.rate_limit_delay)
                    return await generate_record(
                        session=session,
                        item=item,
                        api_key=api_key,
                        model=args.model,
                        min_words=args.min_words,
                        max_output_tokens=args.max_output_tokens,
                        temperature=args.temperature,
                        max_retries=args.max_retries,
                    )

            # Process in batches
            for batch_start in range(0, len(selected), batch_size):
                batch_items = selected[batch_start:batch_start + batch_size]
                tasks = [bounded_generate(item) for item in batch_items]
                results = await asyncio.gather(*tasks)

                # Write successful records immediately
                success_records = [r["record"] for r in results if r.get("success")]
                if success_records:
                    append_jsonl(output_path, success_records)
                    total_success += len(success_records)
                    print(f"Batch {batch_start // batch_size + 1}: wrote {len(success_records)} records (total: {total_success})")

                # Mark as seen immediately
                if seen_db is not None:
                    for item in batch_items:
                        seen_db[prompt_key(item.prompt)] = b"1"

                # Save state after each batch
                state["generated_count"] = int(state.get("generated_count", 0)) + len(success_records)
                state["updated_at"] = datetime.now().isoformat()
                save_state(Path(args.state), state)

        success_records = []  # Already written above

        if args.log_prompts:
            append_prompts_log(Path(args.prompts_log), selected)

        if seen_db is not None:
            for item in selected:
                seen_db[prompt_key(item.prompt)] = b"1"

        state["generated_count"] = int(state.get("generated_count", 0)) + len(success_records)
        state["updated_at"] = datetime.now().isoformat()
        save_state(Path(args.state), state)

        print(f"Generation complete. Records written: {len(success_records)}.")
        return 0
    finally:
        if seen_db is not None:
            seen_db.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate coding-only training data.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help="Output JSONL path (append-only).")
    parser.add_argument("--state", type=str, default=str(DEFAULT_STATE),
                        help="Persistent generator state.")
    parser.add_argument("--prompts-log", type=str, default=str(DEFAULT_PROMPTS_LOG),
                        help="Append-only prompt log.")
    parser.add_argument("--seen-db", type=str, default=str(DEFAULT_SEEN_DB),
                        help="Persistent seen index (dbm).")
    parser.add_argument("--target", type=int, default=10000,
                        help="How many records to generate (<=0 means all).")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True,
                        help="Resume using saved state (default: true).")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Do not resume; start from the beginning.")
    parser.add_argument("--skip-seen", dest="skip_seen", action="store_true", default=True,
                        help="Skip prompts already seen (default: true).")
    parser.add_argument("--no-skip-seen", dest="skip_seen", action="store_false",
                        help="Do not skip seen prompts.")
    parser.add_argument("--log-prompts", action="store_true", default=True,
                        help="Append prompts to the log (default: true).")
    parser.add_argument("--no-log-prompts", dest="log_prompts", action="store_false",
                        help="Disable prompt logging.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="DeepSeek model to use.")
    parser.add_argument("--min-words", type=int, default=150,
                        help="Minimum words required in the answer section.")
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS,
                        help="Max output tokens for the DeepSeek response.")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Sampling temperature.")
    parser.add_argument("--max-retries", type=int, default=1,
                        help="Retries if the answer is too short.")
    parser.add_argument("--workers", type=int, default=100,
                        help="Concurrent DeepSeek requests.")
    parser.add_argument("--rate-limit-delay", type=float, default=0.05,
                        help="Delay (seconds) between request starts.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only list prompts (no DeepSeek calls or output writes).")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
