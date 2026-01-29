#!/usr/bin/env python3
"""
GENERATIONAL UPGRADE PIPELINE
==============================
Each version gets progressively better through emergent quality,
NOT through targeting a master scalar.

Principles:
  1. NO TARGET SCALAR - Quality emerges through generational selection
  2. SELF-CONSISTENCY - Multiple reasoning paths must converge
  3. DIFFICULTY PROGRESSION - Each generation handles harder problems
  4. GENERATIONAL DISTILLATION - Best outputs become next generation's training

Flow:
  Generation N:
    1. Generate candidate solutions (k samples per problem)
    2. Filter by self-consistency (agreement across samples)
    3. Filter by correctness (for verifiable problems)
    4. Rank by difficulty (harder problems weighted higher)
    5. Top survivors become training data for Generation N+1

  Generation N+1:
    - Trains on curated data from Generation N
    - Generates candidates for even harder problems
    - Repeat forever...

Author: Bo Shang <bo@shang.software>
"""

import os
import sys
import json
import asyncio
import hashlib
import random
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import aiohttp

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_STORE = SCRIPT_DIR / "data_store"
GENERATIONS_DIR = DATA_STORE / "generations"
GENERATION_STATE_FILE = DATA_STORE / "generational_state.json"
DIFFICULTY_BANK_FILE = DATA_STORE / "difficulty_bank.json"

# Generation config
DEFAULT_SAMPLES_PER_PROBLEM = 5  # k samples for self-consistency
MIN_CONSISTENCY_RATIO = 0.6  # 3/5 must agree for a problem to pass
MIN_SURVIVORS_PER_GEN = 500  # minimum survivors to proceed to training
MAX_DIFFICULTY_LEVEL = 10  # difficulty progression cap


class Colors:
    CYAN = '\033[0;36m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    BOLD = '\033[1m'
    NC = '\033[0m'


def cprint(msg: str, color: str = Colors.NC):
    print(f"{color}{msg}{Colors.NC}")


@dataclass
class GenerationState:
    """Tracks state across generations."""
    current_generation: int = 0
    total_survivors: int = 0
    difficulty_level: float = 1.0
    consistency_threshold: float = MIN_CONSISTENCY_RATIO
    generations_completed: List[int] = field(default_factory=list)
    metrics_history: List[Dict] = field(default_factory=list)

    def save(self):
        GENERATIONS_DIR.mkdir(parents=True, exist_ok=True)
        with open(GENERATION_STATE_FILE, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls) -> 'GenerationState':
        if GENERATION_STATE_FILE.exists():
            with open(GENERATION_STATE_FILE) as f:
                data = json.load(f)
            return cls(**data)
        return cls()


@dataclass
class CandidateSolution:
    """A single candidate solution with metadata."""
    problem_id: str
    problem_text: str
    reasoning: str  # CoT reasoning
    answer: str  # Final answer
    sample_index: int  # Which sample (0 to k-1)
    difficulty: float = 1.0

    def answer_hash(self) -> str:
        """Hash the answer for consistency comparison."""
        normalized = self.answer.strip().lower()
        return hashlib.md5(normalized.encode()).hexdigest()[:16]


@dataclass
class SurvivorSample:
    """A sample that passed consistency and correctness checks."""
    problem_id: str
    problem_text: str
    winning_reasoning: str
    winning_answer: str
    consistency_score: float  # How many agreed
    difficulty: float
    generation: int

    def to_training_format(self) -> Dict:
        """Convert to training JSONL format."""
        return {
            "messages": [
                {"role": "user", "content": self.problem_text},
                {"role": "assistant", "content": f"<|think_start|>{self.winning_reasoning}<|think_end|><|answer|>{self.winning_answer}"}
            ],
            "metadata": {
                "generation": self.generation,
                "difficulty": self.difficulty,
                "consistency_score": self.consistency_score,
                "source": "generational_upgrade"
            }
        }


class DifficultyProgressionManager:
    """
    Manages difficulty progression across generations.
    Each generation unlocks harder problem types.
    """

    DIFFICULTY_TIERS = {
        1: ["arithmetic", "simple_logic", "basic_comprehension"],
        2: ["algebra", "multi_step_logic", "inference"],
        3: ["word_problems", "constraint_satisfaction", "analysis"],
        4: ["optimization", "proof_sketches", "synthesis"],
        5: ["abstract_reasoning", "meta_problems", "novel_domains"],
        6: ["adversarial", "edge_cases", "ambiguous"],
        7: ["multi_domain", "long_horizon", "planning"],
        8: ["creative_synthesis", "open_ended", "discovery"],
        9: ["frontier", "unsolved", "research_level"],
        10: ["transcendent", "paradigm_shifting", "unknown_unknowns"],
    }

    def __init__(self):
        self.current_tier = 1
        self.problem_bank: Dict[int, List[Dict]] = defaultdict(list)
        self._load_bank()

    def _load_bank(self):
        if DIFFICULTY_BANK_FILE.exists():
            with open(DIFFICULTY_BANK_FILE) as f:
                data = json.load(f)
                for tier_str, problems in data.items():
                    self.problem_bank[int(tier_str)] = problems

    def _save_bank(self):
        with open(DIFFICULTY_BANK_FILE, 'w') as f:
            json.dump({str(k): v for k, v in self.problem_bank.items()}, f, indent=2)

    def get_unlocked_categories(self, generation: int) -> List[str]:
        """Get categories unlocked by this generation."""
        tier = min(generation, MAX_DIFFICULTY_LEVEL)
        categories = []
        for t in range(1, tier + 1):
            categories.extend(self.DIFFICULTY_TIERS.get(t, []))
        return categories

    def calculate_difficulty(self, problem: Dict, generation: int) -> float:
        """Calculate difficulty score for a problem."""
        base_difficulty = problem.get("tier", 1)
        complexity_bonus = len(problem.get("constraints", [])) * 0.1
        novelty_bonus = 0.2 if problem.get("novel", False) else 0.0
        return min(10.0, base_difficulty + complexity_bonus + novelty_bonus)

    def add_problem(self, tier: int, problem: Dict):
        """Add a problem to the bank."""
        self.problem_bank[tier].append(problem)
        self._save_bank()

    def get_problems_for_generation(self, generation: int, count: int) -> List[Dict]:
        """Get problems appropriate for this generation."""
        unlocked_tiers = range(1, min(generation + 1, MAX_DIFFICULTY_LEVEL + 1))

        # Weight toward harder problems as generations increase
        all_problems = []
        for tier in unlocked_tiers:
            weight = tier / generation if generation > 0 else 1.0
            for p in self.problem_bank.get(tier, []):
                p_copy = p.copy()
                p_copy["tier"] = tier
                p_copy["weight"] = weight
                all_problems.append(p_copy)

        if not all_problems:
            return self._generate_seed_problems(generation, count)

        # Weighted sample
        weights = [p["weight"] for p in all_problems]
        total = sum(weights)
        if total == 0:
            return random.sample(all_problems, min(count, len(all_problems)))

        normalized = [w / total for w in weights]
        selected = []
        for _ in range(min(count, len(all_problems))):
            r = random.random()
            cumulative = 0
            for i, w in enumerate(normalized):
                cumulative += w
                if r <= cumulative:
                    selected.append(all_problems[i])
                    break

        return selected

    def _generate_seed_problems(self, generation: int, count: int) -> List[Dict]:
        """Generate seed problems for bootstrapping."""
        seeds = []
        categories = self.get_unlocked_categories(generation)

        for i in range(count):
            cat = random.choice(categories) if categories else "general"
            tier = min(generation, MAX_DIFFICULTY_LEVEL)

            seeds.append({
                "id": f"seed_{generation}_{i}",
                "category": cat,
                "tier": tier,
                "text": f"[SEED PROBLEM - Category: {cat}, Difficulty: {tier}]",
                "requires_generation": True,
            })

        return seeds


class SelfConsistencyChecker:
    """
    Implements self-consistency checking across multiple samples.
    No external scalar target - consistency emerges from agreement.
    """

    def __init__(self, min_ratio: float = MIN_CONSISTENCY_RATIO):
        self.min_ratio = min_ratio

    def check_consistency(self, candidates: List[CandidateSolution]) -> Optional[SurvivorSample]:
        """
        Check if candidates agree on an answer.
        Returns a SurvivorSample if consistency threshold is met, None otherwise.
        """
        if not candidates:
            return None

        # Group by answer hash
        answer_groups: Dict[str, List[CandidateSolution]] = defaultdict(list)
        for c in candidates:
            answer_groups[c.answer_hash()].append(c)

        # Find the most common answer
        best_hash = max(answer_groups.keys(), key=lambda h: len(answer_groups[h]))
        best_group = answer_groups[best_hash]
        consistency_ratio = len(best_group) / len(candidates)

        if consistency_ratio < self.min_ratio:
            return None  # No consensus

        # Pick the best reasoning from the winning group (shortest that's complete)
        best_candidate = min(
            best_group,
            key=lambda c: (
                0 if "<|think_end|>" in c.reasoning else 1,  # Prefer complete
                len(c.reasoning)  # Then prefer concise
            )
        )

        return SurvivorSample(
            problem_id=best_candidate.problem_id,
            problem_text=best_candidate.problem_text,
            winning_reasoning=best_candidate.reasoning,
            winning_answer=best_candidate.answer,
            consistency_score=consistency_ratio,
            difficulty=best_candidate.difficulty,
            generation=0,  # Set by caller
        )

    def batch_check(
        self,
        all_candidates: Dict[str, List[CandidateSolution]],
        generation: int
    ) -> List[SurvivorSample]:
        """Check consistency for a batch of problems."""
        survivors = []

        for problem_id, candidates in all_candidates.items():
            survivor = self.check_consistency(candidates)
            if survivor:
                survivor.generation = generation
                survivors.append(survivor)

        return survivors


class CorrectnessVerifier:
    """
    Verifies correctness for problems with known answers.
    This is an optional filter - not all problems have verifiable answers.
    """

    def __init__(self):
        self.verifiable_categories = {
            "arithmetic", "algebra", "logic", "constraint_satisfaction"
        }

    def is_verifiable(self, problem: Dict) -> bool:
        """Check if this problem type has verifiable answers."""
        category = problem.get("category", "")
        return category in self.verifiable_categories or "expected_answer" in problem

    def verify(self, survivor: SurvivorSample, problem: Dict) -> bool:
        """Verify a survivor's answer against expected answer."""
        if not self.is_verifiable(problem):
            return True  # Can't verify, assume correct

        expected = problem.get("expected_answer", "").strip().lower()
        if not expected:
            return True

        actual = survivor.winning_answer.strip().lower()

        # Exact match
        if actual == expected:
            return True

        # Numeric comparison (handles formatting differences)
        try:
            if abs(float(actual) - float(expected)) < 1e-6:
                return True
        except (ValueError, TypeError):
            pass

        # Substring match (for text answers)
        if expected in actual or actual in expected:
            return True

        return False

    def filter_survivors(
        self,
        survivors: List[SurvivorSample],
        problems: Dict[str, Dict]
    ) -> List[SurvivorSample]:
        """Filter survivors by correctness verification."""
        verified = []
        for s in survivors:
            problem = problems.get(s.problem_id, {})
            if self.verify(s, problem):
                verified.append(s)
        return verified


class GenerationalUpgradePipeline:
    """
    Main pipeline orchestrator.
    Each generation produces progressively better models through:
    - Self-consistency filtering
    - Correctness verification
    - Difficulty progression

    Uses DeepSeek reasoner for all generation.
    """

    def __init__(
        self,
        model: str = "deepseek-reasoner",
        samples_per_problem: int = DEFAULT_SAMPLES_PER_PROBLEM,
    ):
        self.model = model
        self.samples_per_problem = samples_per_problem
        self.state = GenerationState.load()
        self.difficulty_manager = DifficultyProgressionManager()
        self.consistency_checker = SelfConsistencyChecker()
        self.correctness_verifier = CorrectnessVerifier()
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        self.api_base = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com")

    async def generate_candidate(
        self,
        session: aiohttp.ClientSession,
        problem: Dict,
        sample_index: int,
    ) -> Optional[CandidateSolution]:
        """Generate a single candidate solution using DeepSeek reasoner."""
        problem_text = problem.get("text", problem.get("prompt", ""))
        problem_id = problem.get("id", hashlib.md5(problem_text.encode()).hexdigest()[:16])

        prompt = f"""Solve this problem step by step.

Problem: {problem_text}

Think through this carefully, showing your reasoning."""

        try:
            async with session.post(
                f"{self.api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2048,
                },
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    if sample_index == 0:  # Only log once per problem
                        cprint(f"  DeepSeek API error: {error[:100]}", Colors.YELLOW)
                    return None
                data = await resp.json()

                # DeepSeek reasoner returns reasoning_content + content
                message = data.get("choices", [{}])[0].get("message", {})
                reasoning = message.get("reasoning_content", "")
                answer = message.get("content", "")

                return CandidateSolution(
                    problem_id=problem_id,
                    problem_text=problem_text,
                    reasoning=reasoning,
                    answer=answer,
                    sample_index=sample_index,
                    difficulty=self.difficulty_manager.calculate_difficulty(
                        problem, self.state.current_generation
                    ),
                )
        except Exception as e:
            cprint(f"  Generation error: {e}", Colors.YELLOW)
            return None

    def _parse_response(self, content: str) -> Tuple[str, str]:
        """Parse response into reasoning and answer."""
        # Look for explicit answer markers
        answer_markers = ["answer:", "therefore:", "thus:", "result:", "="]

        lines = content.strip().split("\n")
        reasoning_lines = []
        answer = ""

        for i, line in enumerate(lines):
            line_lower = line.lower().strip()

            # Check for answer markers
            found_answer = False
            for marker in answer_markers:
                if marker in line_lower:
                    # Everything after this is the answer
                    idx = line_lower.find(marker)
                    answer = line[idx + len(marker):].strip()
                    found_answer = True
                    break

            if found_answer:
                break
            else:
                reasoning_lines.append(line)

        reasoning = "\n".join(reasoning_lines)

        # If no explicit answer found, use last non-empty line
        if not answer:
            for line in reversed(lines):
                if line.strip():
                    answer = line.strip()
                    break

        return reasoning, answer

    async def generate_candidates_for_problem(
        self,
        session: aiohttp.ClientSession,
        problem: Dict,
    ) -> List[CandidateSolution]:
        """Generate k candidate solutions for a single problem."""
        tasks = [
            self.generate_candidate(session, problem, i)
            for i in range(self.samples_per_problem)
        ]

        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]

    async def run_generation(
        self,
        problems: List[Dict],
        batch_size: int = 10,
    ) -> List[SurvivorSample]:
        """Run a full generation cycle."""
        generation = self.state.current_generation + 1

        cprint(f"\n{'═' * 60}", Colors.CYAN + Colors.BOLD)
        cprint(f"  GENERATION {generation}", Colors.CYAN + Colors.BOLD)
        cprint(f"  Problems: {len(problems)}", Colors.CYAN)
        cprint(f"  Samples per problem: {self.samples_per_problem}", Colors.CYAN)
        cprint(f"  Consistency threshold: {self.consistency_checker.min_ratio:.0%}", Colors.CYAN)
        cprint(f"{'═' * 60}", Colors.CYAN + Colors.BOLD)

        all_candidates: Dict[str, List[CandidateSolution]] = defaultdict(list)
        problem_map: Dict[str, Dict] = {}

        async with aiohttp.ClientSession() as session:
            # Process in batches
            for batch_start in range(0, len(problems), batch_size):
                batch = problems[batch_start:batch_start + batch_size]
                batch_num = batch_start // batch_size + 1
                total_batches = (len(problems) + batch_size - 1) // batch_size

                cprint(f"\n  Batch {batch_num}/{total_batches}...", Colors.YELLOW)

                # Generate candidates for all problems in batch
                tasks = [
                    self.generate_candidates_for_problem(session, p)
                    for p in batch
                ]

                batch_results = await asyncio.gather(*tasks)

                for problem, candidates in zip(batch, batch_results):
                    problem_id = problem.get("id", hashlib.md5(
                        problem.get("text", "").encode()
                    ).hexdigest()[:16])
                    all_candidates[problem_id].extend(candidates)
                    problem_map[problem_id] = problem

                # Progress
                total_candidates = sum(len(c) for c in all_candidates.values())
                cprint(f"    Generated {total_candidates} candidates so far", Colors.GREEN)

        # Apply self-consistency filter
        cprint(f"\n  Checking self-consistency...", Colors.YELLOW)
        survivors = self.consistency_checker.batch_check(all_candidates, generation)
        cprint(f"    Consistency survivors: {len(survivors)}/{len(all_candidates)}", Colors.GREEN)

        # Apply correctness filter
        cprint(f"  Verifying correctness...", Colors.YELLOW)
        verified = self.correctness_verifier.filter_survivors(survivors, problem_map)
        cprint(f"    Verified survivors: {len(verified)}/{len(survivors)}", Colors.GREEN)

        # Update state
        self.state.current_generation = generation
        self.state.total_survivors += len(verified)
        self.state.generations_completed.append(generation)
        self.state.metrics_history.append({
            "generation": generation,
            "problems": len(problems),
            "candidates": sum(len(c) for c in all_candidates.values()),
            "consistency_survivors": len(survivors),
            "verified_survivors": len(verified),
            "timestamp": datetime.now().isoformat(),
        })
        self.state.save()

        return verified

    def save_generation_data(self, survivors: List[SurvivorSample], generation: int):
        """Save survivors as training data for this generation."""
        GENERATIONS_DIR.mkdir(parents=True, exist_ok=True)
        gen_file = GENERATIONS_DIR / f"generation_{generation:03d}.jsonl"

        with open(gen_file, 'w') as f:
            for s in survivors:
                f.write(json.dumps(s.to_training_format()) + "\n")

        cprint(f"\n  Saved {len(survivors)} survivors to {gen_file}", Colors.GREEN)
        return gen_file

    def combine_generations_for_training(self, up_to_generation: int) -> Tuple[Path, int]:
        """Combine all generations into cumulative training file."""
        cache_dir = SCRIPT_DIR / "cache" / "optimal_gen"
        cache_dir.mkdir(parents=True, exist_ok=True)
        combined_file = cache_dir / "optimal_training.jsonl"

        total = 0
        with open(combined_file, 'w') as out:
            for gen in range(1, up_to_generation + 1):
                gen_file = GENERATIONS_DIR / f"generation_{gen:03d}.jsonl"
                if gen_file.exists():
                    with open(gen_file) as f:
                        count = 0
                        for line in f:
                            out.write(line)
                            count += 1
                            total += 1
                    cprint(f"    + Generation {gen}: {count} samples", Colors.GREEN)

        cprint(f"  Combined {total} samples from {up_to_generation} generations", Colors.GREEN)
        return combined_file, total

    async def run_full_cycle(
        self,
        problems_per_generation: int = 1000,
        min_survivors: int = MIN_SURVIVORS_PER_GEN,
    ) -> Dict:
        """Run a complete generation cycle."""
        generation = self.state.current_generation + 1

        # Get problems for this generation (with difficulty progression)
        problems = self.difficulty_manager.get_problems_for_generation(
            generation, problems_per_generation
        )

        if not problems:
            cprint(f"No problems available for generation {generation}", Colors.RED)
            return {"success": False, "reason": "no_problems"}

        # Generate candidates and filter
        survivors = await self.run_generation(problems)

        if len(survivors) < min_survivors:
            cprint(
                f"\n  Only {len(survivors)} survivors (need {min_survivors}). "
                f"Generation incomplete.",
                Colors.YELLOW
            )
            return {
                "success": False,
                "reason": "insufficient_survivors",
                "survivors": len(survivors),
                "required": min_survivors,
            }

        # Save generation data
        gen_file = self.save_generation_data(survivors, generation)

        # Combine for training
        combined_file, total = self.combine_generations_for_training(generation)

        cprint(f"\n{'═' * 60}", Colors.GREEN + Colors.BOLD)
        cprint(f"  GENERATION {generation} COMPLETE", Colors.GREEN + Colors.BOLD)
        cprint(f"{'═' * 60}", Colors.GREEN)
        cprint(f"  Survivors: {len(survivors)}", Colors.GREEN)
        cprint(f"  Cumulative training samples: {total}", Colors.GREEN)
        cprint(f"  Ready for training: {combined_file}", Colors.GREEN)

        return {
            "success": True,
            "generation": generation,
            "survivors": len(survivors),
            "total_samples": total,
            "training_file": str(combined_file),
        }


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generational Upgrade Pipeline - No target scalar required"
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL", "gpt-5.1-codex-mini"),
        help="Model for generation"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES_PER_PROBLEM,
        help="Samples per problem for consistency"
    )
    parser.add_argument(
        "--problems",
        type=int,
        default=1000,
        help="Problems per generation"
    )
    parser.add_argument(
        "--min-survivors",
        type=int,
        default=MIN_SURVIVORS_PER_GEN,
        help="Minimum survivors to proceed"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run only one generation"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current status and exit"
    )

    args = parser.parse_args()

    pipeline = GenerationalUpgradePipeline(
        model=args.model,
        samples_per_problem=args.samples,
    )

    if args.status:
        state = pipeline.state
        cprint(f"\n{'═' * 60}", Colors.CYAN)
        cprint(f"GENERATIONAL UPGRADE PIPELINE STATUS", Colors.CYAN)
        cprint(f"{'═' * 60}", Colors.CYAN)
        cprint(f"  Current generation: {state.current_generation}", Colors.GREEN)
        cprint(f"  Total survivors: {state.total_survivors}", Colors.GREEN)
        cprint(f"  Difficulty level: {state.difficulty_level:.1f}", Colors.GREEN)
        cprint(f"  Generations completed: {state.generations_completed}", Colors.GREEN)
        if state.metrics_history:
            last = state.metrics_history[-1]
            cprint(f"\n  Last generation metrics:", Colors.CYAN)
            cprint(f"    Problems: {last.get('problems', 0)}", Colors.GREEN)
            cprint(f"    Candidates: {last.get('candidates', 0)}", Colors.GREEN)
            cprint(f"    Survivors: {last.get('verified_survivors', 0)}", Colors.GREEN)
        return

    # Banner
    cprint(f"\n{'═' * 60}", Colors.CYAN + Colors.BOLD)
    cprint(f"  GENERATIONAL UPGRADE PIPELINE", Colors.CYAN + Colors.BOLD)
    cprint(f"  No target scalar - quality emerges through selection", Colors.CYAN)
    cprint(f"{'═' * 60}", Colors.CYAN)
    cprint(f"  Model: {args.model}", Colors.GREEN)
    cprint(f"  Samples per problem: {args.samples}", Colors.GREEN)
    cprint(f"  Problems per generation: {args.problems}", Colors.GREEN)
    cprint(f"  Minimum survivors: {args.min_survivors}", Colors.GREEN)
    cprint(f"{'═' * 60}\n", Colors.CYAN)

    while True:
        result = await pipeline.run_full_cycle(
            problems_per_generation=args.problems,
            min_survivors=args.min_survivors,
        )

        if not result["success"]:
            cprint(f"\nGeneration incomplete: {result['reason']}", Colors.YELLOW)
            if args.once:
                break
            cprint("Retrying in 10 seconds...", Colors.YELLOW)
            await asyncio.sleep(10)
            continue

        if args.once:
            cprint(f"\nCompleted generation {result['generation']} (--once flag set)", Colors.GREEN)
            break

        cprint(f"\nStarting next generation in 5 seconds...", Colors.YELLOW)
        cprint("(Press Ctrl+C to stop)", Colors.YELLOW)
        try:
            await asyncio.sleep(5)
        except KeyboardInterrupt:
            cprint("\nStopped by user.", Colors.YELLOW)
            break


if __name__ == "__main__":
    asyncio.run(main())
