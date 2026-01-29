#!/usr/bin/env python3
"""
Chain-of-Thought Optimization Data Generator using DeepSeek-Reasoner

FULLY AUTOMATED with Master Scalar Improvement Tracking

Every generation:
1. Uses deepseek-reasoner to generate training samples
2. Computes master scalar scores for quality filtering
3. Tracks improvements across generations
4. Appends to JSONL immediately after each sample
5. Saves state persistently

Author: Bo Shang <bo@shang.software>
"""

import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import aiohttp
except ImportError:
    print("Installing aiohttp...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp", "-q"])
    import aiohttp

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

THINK_START = "<|think_start|>"
THINK_END = "<|think_end|>"
ANSWER_MARKER = "<|answer|>"
STEP_MARKER = "<|step|>"

DEFAULT_MODEL = os.environ.get("MODEL", "deepseek-reasoner")
DATA_STORE = Path("data_store")
DEFAULT_OUTPUT = DATA_STORE / "cot_optimization_training_data.jsonl"
DEFAULT_STATE = DATA_STORE / "cot_optimization_state.json"
MASTER_SCALAR_LOG = DATA_STORE / "master_scalar_log.jsonl"

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

SYSTEM_PROMPT = """You are generating training data for a reasoning assistant.

Your responses must demonstrate EXPLICIT step-by-step reasoning where each step:
1. Builds logically on previous steps (use words like "therefore", "thus", "from this")
2. References prior conclusions explicitly
3. Shows clear logical progression toward the answer

Output format (EXACT - use these markers):
<|think_start|>
<|step|> [First reasoning step]
<|step|> [Second step - reference step 1]
<|step|> [Continue...]
<|think_end|>
<|answer|>
[Final answer from reasoning]

Rules:
- Each step MUST reference or build on previous steps
- Use: "Therefore...", "From step 1...", "This means...", "Combining..."
- Include 3-5+ reasoning steps
"""


# ════════════════════════════════════════════════════════════════════════════
# MASTER SCALAR COMPUTATION
# ════════════════════════════════════════════════════════════════════════════

def compute_cot_attention_score(content: str) -> dict:
    """Compute master scalar proxy attention scores."""
    if THINK_START in content and THINK_END in content:
        start = content.find(THINK_START) + len(THINK_START)
        end = content.find(THINK_END)
        reasoning_text = content[start:end]
    else:
        reasoning_text = content

    output = content.split(ANSWER_MARKER, 1)[1].strip() if ANSWER_MARKER in content else ""

    if STEP_MARKER in reasoning_text:
        steps = [s.strip() for s in reasoning_text.split(STEP_MARKER) if s.strip()]
    else:
        steps = [s.strip() for s in reasoning_text.split("\n") if s.strip() and len(s.strip()) > 20]

    # Cross-step attention (r→r)
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
        ref_score = min(1.0, referential_count / (len(steps) * 2))
        overlap_score = sum(word_overlap_scores) / len(word_overlap_scores) if word_overlap_scores else 0
        overlap_score = min(1.0, overlap_score / 0.25)
        cross_step_score = ref_score * 0.6 + overlap_score * 0.4

    # Answer grounding (a→r)
    answer_ground_score = 0.0
    if output and reasoning_text:
        output_words = set(output.lower().split())
        reasoning_words = set(reasoning_text.lower().split())
        if output_words and reasoning_words:
            overlap = len(output_words & reasoning_words)
            answer_ground_score = min(1.0, overlap / (len(output_words) * 0.4))
        if steps:
            final_words = set(steps[-1].lower().split())
            if len(output_words & final_words) > 3:
                answer_ground_score = min(1.0, answer_ground_score + 0.2)

    # Structural patterns
    structural_score = 0.0
    structural_score += 0.20 if THINK_START in content else 0
    structural_score += 0.20 if THINK_END in content else 0
    structural_score += 0.25 if STEP_MARKER in content else 0
    structural_score += 0.15 if output.strip() else 0
    if 3 <= len(steps) <= 7:
        structural_score = min(1.0, structural_score + 0.20)
    elif len(steps) >= 2:
        structural_score = min(1.0, structural_score + 0.10)

    # Master scalar = RAW PERFORMANCE (no safety blending)
    weighted_score = cross_step_score * 0.45 + answer_ground_score * 0.30 + structural_score * 0.25

    return {
        "cross_step": round(cross_step_score, 4),
        "answer_ground": round(answer_ground_score, 4),
        "structural": round(structural_score, 4),
        "master_scalar": round(weighted_score, 4),
        "num_steps": len(steps),
        "quality": "excellent" if weighted_score >= 0.7 else "good" if weighted_score >= 0.5 else "fair"
    }


# ════════════════════════════════════════════════════════════════════════════
# PROMPT CATALOG (5000+ unique CoT prompts)
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class CoTPromptItem:
    prompt: str
    category: str
    difficulty: str
    expected_steps: int
    topic: Optional[str] = None


def build_cot_prompt_catalog() -> List[CoTPromptItem]:
    """Build 5000+ unique CoT prompts."""
    catalog: List[CoTPromptItem] = []

    def add(prompt: str, category: str, difficulty: str, expected_steps: int, topic: str = None):
        catalog.append(CoTPromptItem(prompt, category, difficulty, expected_steps, topic))

    # Math - Linear equations (~630)
    for a in range(2, 12):
        for b in range(1, 10):
            for c in range(1, 8):
                add(f"Solve: {a}x + {b} = {a*5 + b + c}. Show steps.", "math", "easy", 4, "linear")

    # Math - Quadratic (~300)
    for a in range(1, 6):
        for b in range(-5, 6):
            for c in range(-10, 11, 2):
                if b*b - 4*a*c >= 0:
                    add(f"Solve: {a}x² + {b}x + {c} = 0", "math", "medium", 5, "quadratic")

    # Math - Systems (~80)
    for a1 in range(1, 6):
        for b1 in range(1, 5):
            for a2 in range(1, 5):
                if a1 != a2:
                    add(f"Solve: {a1}x + {b1}y = {a1*3+b1*2}, {a2}x - y = {a2*3-2}", "math", "medium", 5, "systems")

    # Math - Sequences (~700)
    for first in range(1, 20):
        for diff in range(1, 10):
            for n in range(10, 30, 5):
                add(f"Sum of first {n} terms: start={first}, diff={diff}", "math", "easy", 4, "arithmetic")

    # Math - Geometry (~150)
    for a in range(3, 15):
        for b in range(3, 15):
            if a + b > max(a, b) + 1:
                add(f"Right triangle legs {a},{b}. Find hypotenuse, area, perimeter.", "math", "easy", 4, "geometry")

    # Math - Probability (~64)
    for red in range(2, 10):
        for blue in range(2, 10):
            add(f"Bag: {red} red, {blue} blue. P(both same) w/o replacement?", "math", "medium", 5, "probability")

    # Math - Number theory (~240)
    for a in range(50, 200, 7):
        for b in range(30, 150, 11):
            if a != b:
                add(f"GCD({a}, {b}) using Euclidean algorithm", "math", "medium", 5, "gcd")

    # Logic - Syllogisms (~720)
    subjects = ["mammals", "birds", "reptiles", "fish", "insects", "scientists", "artists", "engineers", "teachers", "students"]
    props = ["intelligent", "creative", "hardworking", "logical", "patient", "curious", "adaptable", "warm-blooded"]
    for s1 in subjects:
        for s2 in subjects:
            if s1 != s2:
                for p in props:
                    add(f"All {s1} are {p}. All {s2} are {s1}. Conclude about {s2}?", "logic", "easy", 3, "syllogism")

    # Logic - Ordering (~35)
    names = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Henry"]
    for i in range(len(names) - 3):
        for j in range(i + 1, len(names) - 2):
            for k in range(j + 1, len(names) - 1):
                a, b, c, d = names[i], names[j], names[k], names[k+1]
                add(f"{a} > {b}, {c} < {b}, {d} > {a}. Order tallest to shortest.", "logic", "medium", 5, "ordering")

    # Algorithms - Sorting (~400)
    for _ in range(100):
        arr = random.sample(range(1, 50), 6)
        add(f"Bubble sort {arr}. Show passes.", "algo", "easy", 5, "bubble")
        add(f"Selection sort {arr}. Show swaps.", "algo", "easy", 5, "selection")
        add(f"Insertion sort {arr}. Show steps.", "algo", "easy", 5, "insertion")
        add(f"Merge sort {arr}. Show divide/merge.", "algo", "medium", 6, "merge")

    # Algorithms - Search (~100)
    for _ in range(100):
        arr = sorted(random.sample(range(1, 100), 10))
        target = random.choice(arr + [arr[0]-1, arr[-1]+1])
        add(f"Binary search {target} in {arr}. Show comparisons.", "algo", "easy", 4, "binary_search")

    # Algorithms - Recursion (~30)
    for n in range(5, 20):
        add(f"Trace factorial({n}) recursively.", "algo", "easy", 4, "factorial")
        add(f"Fibonacci({n}): trace and explain memoization.", "algo", "medium", 5, "fibonacci")

    # Physics (~700)
    for v0 in range(0, 30, 5):
        for a in range(-5, 6):
            if a != 0:
                for t in range(1, 10):
                    add(f"v0={v0}m/s, a={a}m/s². Find v,d after {t}s.", "physics", "easy", 4, "kinematics")

    # Word problems (~100)
    for d in range(100, 500, 50):
        for r in range(30, 80, 10):
            add(f"Car travels {d}mi at {r}mph. Time?", "word", "easy", 3, "rate_time")

    for t1 in range(2, 10):
        for t2 in range(2, 10):
            if t1 != t2:
                add(f"A finishes in {t1}h, B in {t2}h. Together?", "word", "medium", 5, "work_rate")

    return catalog


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


# ════════════════════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ════════════════════════════════════════════════════════════════════════════

class GenerationState:
    """Persistent state for automated generation."""

    def __init__(self, state_path: Path = DEFAULT_STATE):
        self.state_path = state_path
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()

    def _load(self) -> dict:
        if self.state_path.exists():
            try:
                with open(self.state_path) as f:
                    return json.load(f)
            except:
                pass
        return {
            "generation": 0,
            "total_samples": 0,
            "seen_prompts": [],
            "master_scalar_history": [],
            "current_avg_score": 0.0,
            "best_avg_score": 0.0,
            "created_at": datetime.now().isoformat(),
        }

    def save(self):
        with open(self.state_path, "w") as f:
            json.dump(self.data, f, indent=2)

    @property
    def generation(self) -> int:
        return self.data.get("generation", 0)

    @generation.setter
    def generation(self, val: int):
        self.data["generation"] = val
        self.save()

    @property
    def seen_prompts(self) -> set:
        return set(self.data.get("seen_prompts", []))

    def mark_seen(self, prompt_id: str):
        if prompt_id not in self.data["seen_prompts"]:
            self.data["seen_prompts"].append(prompt_id)

    def record_generation(self, samples: int, avg_score: float):
        self.data["generation"] += 1
        self.data["total_samples"] += samples
        self.data["current_avg_score"] = avg_score
        if avg_score > self.data.get("best_avg_score", 0):
            self.data["best_avg_score"] = avg_score
        self.data["master_scalar_history"].append({
            "gen": self.data["generation"],
            "samples": samples,
            "avg_score": avg_score,
            "timestamp": datetime.now().isoformat()
        })
        self.data["updated_at"] = datetime.now().isoformat()
        self.save()


# ════════════════════════════════════════════════════════════════════════════
# API INTEGRATION
# ════════════════════════════════════════════════════════════════════════════

def get_api_key() -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        secrets = Path.home() / ".agi" / "secrets.json"
        if secrets.exists():
            try:
                with open(secrets) as f:
                    api_key = json.load(f).get("DEEPSEEK_API_KEY", "")
            except:
                pass
    return api_key


def get_api_url() -> str:
    base = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com")
    base = base.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return f"{base}/chat/completions"


def parse_response(result: dict) -> str:
    choices = result.get("choices", [])
    for choice in choices:
        msg = choice.get("message", {})
        reasoning = msg.get("reasoning_content", "")
        content = msg.get("content", "")

        if reasoning:
            parts = [THINK_START]
            for line in reasoning.strip().split("\n\n"):
                if line.strip():
                    parts.append(f"{STEP_MARKER} {line.strip()}")
            parts.append(THINK_END)
            if content:
                parts.append(f"{ANSWER_MARKER} {content.strip()}")
            return "\n".join(parts)
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


# ════════════════════════════════════════════════════════════════════════════
# GENERATION
# ════════════════════════════════════════════════════════════════════════════

async def generate_one(
    session: aiohttp.ClientSession,
    api_key: str,
    model: str,
    prompt: str,
    min_score: float = 0.3,
    max_retries: int = 2
) -> Optional[Tuple[str, dict]]:
    """Generate one sample. Returns (response, scores) or None."""
    url = get_api_url()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for attempt in range(max_retries + 1):
        try:
            p = prompt
            if attempt > 0:
                p += "\n\nBe explicit about logical connections between steps."

            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": p}
                ],
                "max_tokens": 2000,
                "temperature": 0.1,
                "stream": False
            }

            async with session.post(url, headers=headers, json=payload,
                                   timeout=aiohttp.ClientTimeout(total=120)) as resp:
                resp.raise_for_status()
                data = await resp.json()

            response = parse_response(data)
            if not response:
                continue

            response = normalize_cot(response)
            scores = compute_cot_attention_score(response)

            if scores["master_scalar"] >= min_score:
                return response, scores

        except Exception as e:
            print(f"  Attempt {attempt+1} error: {e}")
            await asyncio.sleep(1)

    return None


async def run_generation(
    target: int = 5000,
    workers: int = 20,
    min_score: float = 0.3,
    model: str = DEFAULT_MODEL,
    output_path: Path = DEFAULT_OUTPUT,
    state_path: Path = DEFAULT_STATE
):
    """Fully automated generation with master scalar tracking."""

    api_key = get_api_key()
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY required")
        print("Set via environment or ~/.agi/secrets.json")
        return 1

    # Initialize
    state = GenerationState(state_path)
    catalog = build_cot_prompt_catalog()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"═══════════════════════════════════════════════════════════")
    print(f"  CoT Optimization Generator - Generation {state.generation + 1}")
    print(f"═══════════════════════════════════════════════════════════")
    print(f"  Model: {model}")
    print(f"  Catalog: {len(catalog)} prompts")
    print(f"  Previous samples: {state.data.get('total_samples', 0)}")
    print(f"  Best avg score: {state.data.get('best_avg_score', 0):.4f}")
    print(f"  Target: {target} samples")
    print(f"  Output: {output_path}")
    print()

    # Filter prompts
    seen = state.seen_prompts
    available = [p for p in catalog if prompt_hash(p.prompt) not in seen]

    if len(available) < target:
        print(f"Only {len(available)} unseen prompts available")
        target = len(available)

    if target == 0:
        print("No new prompts to generate.")
        return 0

    prompts = available[:target]
    print(f"Generating {len(prompts)} samples...")
    print()

    # Generate
    success_count = 0
    total_score = 0.0

    async with aiohttp.ClientSession() as session:
        sem = asyncio.Semaphore(workers)

        async def bounded_gen(item: CoTPromptItem) -> Optional[dict]:
            async with sem:
                result = await generate_one(session, api_key, model, item.prompt, min_score)
                if result:
                    response, scores = result
                    return {
                        "messages": [
                            {"role": "user", "content": item.prompt},
                            {"role": "assistant", "content": response}
                        ],
                        "metadata": {
                            "source": "deepseek-reasoner",
                            "category": item.category,
                            "difficulty": item.difficulty,
                            "topic": item.topic,
                            "prompt_id": prompt_hash(item.prompt),
                            "model": model,
                            "generation": state.generation + 1,
                            "generated_at": datetime.now().isoformat(),
                            "has_thinking": True,
                            "has_answer": True,
                            "has_step": True,
                            "cot_master_scalar": scores["master_scalar"],
                            "cot_cross_step": scores["cross_step"],
                            "cot_answer_ground": scores["answer_ground"],
                            "cot_structural": scores["structural"],
                            "cot_num_steps": scores["num_steps"],
                            "cot_quality": scores["quality"],
                            "weight": 1.0 + scores["master_scalar"]
                        }
                    }
                return None

        # Process in batches, save immediately
        batch_size = workers * 2
        for batch_start in range(0, len(prompts), batch_size):
            batch = prompts[batch_start:batch_start + batch_size]
            tasks = [bounded_gen(item) for item in batch]
            results = await asyncio.gather(*tasks)

            # Write each record immediately
            records = [r for r in results if r]
            if records:
                with open(output_path, "a") as f:
                    for rec in records:
                        f.write(json.dumps(rec, ensure_ascii=True) + "\n")

                success_count += len(records)
                batch_scores = [r["metadata"]["cot_master_scalar"] for r in records]
                batch_avg = sum(batch_scores) / len(batch_scores)
                total_score += sum(batch_scores)

                # Mark as seen
                for item in batch:
                    state.mark_seen(prompt_hash(item.prompt))
                state.save()

                print(f"  Batch {batch_start//batch_size + 1}: +{len(records)} | avg: {batch_avg:.3f} | total: {success_count}")

    # Record generation stats
    avg_score = total_score / success_count if success_count > 0 else 0
    state.record_generation(success_count, avg_score)

    # Log master scalar
    with open(MASTER_SCALAR_LOG, "a") as f:
        f.write(json.dumps({
            "generation": state.generation,
            "samples": success_count,
            "avg_master_scalar": avg_score,
            "total_samples": state.data["total_samples"],
            "best_avg_score": state.data["best_avg_score"],
            "timestamp": datetime.now().isoformat()
        }) + "\n")

    print()
    print(f"═══════════════════════════════════════════════════════════")
    print(f"  Generation {state.generation} Complete!")
    print(f"═══════════════════════════════════════════════════════════")
    print(f"  Samples generated: {success_count}")
    print(f"  Average master scalar: {avg_score:.4f}")
    print(f"  Total samples: {state.data['total_samples']}")
    print(f"  Best avg score: {state.data['best_avg_score']:.4f}")
    improvement = avg_score - state.data.get("master_scalar_history", [{}])[-2].get("avg_score", 0) if len(state.data.get("master_scalar_history", [])) > 1 else 0
    if improvement > 0:
        print(f"  Improvement: +{improvement:.4f}")
    print()

    return 0


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Automated CoT Optimization Data Generator using DeepSeek-Reasoner"
    )
    parser.add_argument("--target", type=int, default=5000, help="Samples per generation")
    parser.add_argument("--workers", type=int, default=20, help="Concurrent requests")
    parser.add_argument("--min-score", type=float, default=0.3, help="Min master scalar")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if args.dry_run:
        catalog = build_cot_prompt_catalog()
        print(f"Catalog: {len(catalog)} prompts")
        by_cat = {}
        for p in catalog:
            by_cat.setdefault(p.category, 0)
            by_cat[p.category] += 1
        for cat, cnt in sorted(by_cat.items()):
            print(f"  {cat}: {cnt}")
        return 0

    return asyncio.run(run_generation(
        target=args.target,
        workers=args.workers,
        min_score=args.min_score,
        model=args.model,
        output_path=args.output,
        state_path=args.state
    ))


if __name__ == "__main__":
    raise SystemExit(main())
