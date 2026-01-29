#!/usr/bin/env python3
"""
BENCHMARK TRACKING MODULE
==========================
Automated benchmark tracking and quality gating for self-improving models.

Benchmark Suite:
- MATH: GSM8K-style arithmetic and word problems
- CODE: HumanEval-style function completion
- REASONING: ARC-Challenge-style multiple choice
- KNOWLEDGE: MMLU-style knowledge questions

Quality Gate:
- Must improve on at least 2/4 categories
- No major regressions (>2% drop in any category)
- Tracked in data_store/benchmark_history.json

Usage:
    from benchmarks import BenchmarkRunner, QualityGate

    runner = BenchmarkRunner()
    result = runner.run_benchmarks(model_path)
    gate = QualityGate()
    should_accept = gate.should_accept_version(old_result, new_result)

    # CLI
    python benchmarks.py --test
    python benchmarks.py --run --model models/erosolar
    python benchmarks.py --history

Author: Bo Shang <bo@shang.software>
"""

import os
import sys
import json
import time
import random
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime

# Paths
DATA_STORE = Path("data_store")
BENCHMARK_HISTORY = DATA_STORE / "benchmark_history.json"
BENCHMARK_CACHE = DATA_STORE / "benchmark_cache"

# Try to import local model interface (use our custom Generator from generate.py)
try:
    from generate import Generator, setup_device
    LOCAL_MODEL_AVAILABLE = True
except ImportError:
    LOCAL_MODEL_AVAILABLE = False
    print("[Benchmarks] WARNING: generate.py not available")


@dataclass
class BenchmarkResult:
    """Result of running benchmarks."""
    version: str
    timestamp: str
    scores: Dict[str, float]  # category -> score (0-1)
    details: Dict[str, Dict] = field(default_factory=dict)
    total_samples: int = 0
    total_correct: int = 0
    model_path: str = ""
    duration_seconds: float = 0.0

    @property
    def overall_score(self) -> float:
        """Average score across all categories."""
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'BenchmarkResult':
        return cls(**data)


class BenchmarkSuite:
    """
    Built-in benchmark suite for quick evaluation.
    For production, use lm-eval-harness with full datasets.
    """

    # Mini GSM8K-style math problems
    MATH_PROBLEMS = [
        {"question": "Lisa has 12 apples. She gives 4 to Tom and 3 to Mary. How many apples does Lisa have left?", "answer": 5},
        {"question": "A train travels 60 miles per hour. How far does it travel in 3 hours?", "answer": 180},
        {"question": "If a book costs $15 and you have $50, how many books can you buy?", "answer": 3},
        {"question": "There are 24 students in a class. If 8 are absent, how many are present?", "answer": 16},
        {"question": "A recipe needs 2 cups of flour. If you want to make 4 batches, how many cups of flour do you need?", "answer": 8},
        {"question": "John has 45 marbles. He divides them equally among 5 friends. How many marbles does each friend get?", "answer": 9},
        {"question": "A movie is 120 minutes long. If you've watched 45 minutes, how many minutes are left?", "answer": 75},
        {"question": "If you save $10 each week, how much will you have after 6 weeks?", "answer": 60},
        {"question": "A box contains 36 chocolates. If 12 are milk chocolate and the rest are dark, how many are dark?", "answer": 24},
        {"question": "If a car uses 5 gallons of gas for 100 miles, how many gallons for 300 miles?", "answer": 15},
    ]

    # HumanEval-style code problems (simplified)
    CODE_PROBLEMS = [
        {
            "question": "Write a function `add(a, b)` that returns the sum of a and b.",
            "test_cases": [{"input": [2, 3], "expected": 5}, {"input": [-1, 1], "expected": 0}],
            "signature": "def add(a, b):"
        },
        {
            "question": "Write a function `is_even(n)` that returns True if n is even, False otherwise.",
            "test_cases": [{"input": [4], "expected": True}, {"input": [7], "expected": False}],
            "signature": "def is_even(n):"
        },
        {
            "question": "Write a function `max_of_three(a, b, c)` that returns the maximum of three numbers.",
            "test_cases": [{"input": [1, 2, 3], "expected": 3}, {"input": [5, 5, 5], "expected": 5}],
            "signature": "def max_of_three(a, b, c):"
        },
        {
            "question": "Write a function `reverse_string(s)` that returns the string reversed.",
            "test_cases": [{"input": ["hello"], "expected": "olleh"}, {"input": [""], "expected": ""}],
            "signature": "def reverse_string(s):"
        },
        {
            "question": "Write a function `count_vowels(s)` that returns the number of vowels in s.",
            "test_cases": [{"input": ["hello"], "expected": 2}, {"input": ["xyz"], "expected": 0}],
            "signature": "def count_vowels(s):"
        },
    ]

    # ARC-Challenge-style reasoning problems
    REASONING_PROBLEMS = [
        {
            "question": "Water freezes at what temperature in Celsius?",
            "choices": ["A. 100", "B. 50", "C. 0", "D. -10"],
            "answer": "C"
        },
        {
            "question": "Which planet is closest to the Sun?",
            "choices": ["A. Venus", "B. Mercury", "C. Earth", "D. Mars"],
            "answer": "B"
        },
        {
            "question": "What is the chemical formula for water?",
            "choices": ["A. CO2", "B. NaCl", "C. H2O", "D. O2"],
            "answer": "C"
        },
        {
            "question": "How many sides does a hexagon have?",
            "choices": ["A. 5", "B. 6", "C. 7", "D. 8"],
            "answer": "B"
        },
        {
            "question": "What is the largest organ in the human body?",
            "choices": ["A. Heart", "B. Liver", "C. Skin", "D. Brain"],
            "answer": "C"
        },
    ]

    # MMLU-style knowledge questions
    KNOWLEDGE_PROBLEMS = [
        {
            "question": "In machine learning, what does 'overfitting' refer to?",
            "choices": ["A. Model performs well on training but poorly on test data",
                       "B. Model performs poorly on both training and test data",
                       "C. Model is too simple",
                       "D. Model trains too slowly"],
            "answer": "A"
        },
        {
            "question": "What is the time complexity of binary search?",
            "choices": ["A. O(n)", "B. O(n^2)", "C. O(log n)", "D. O(1)"],
            "answer": "C"
        },
        {
            "question": "In Python, which keyword is used to define a function?",
            "choices": ["A. function", "B. def", "C. func", "D. define"],
            "answer": "B"
        },
        {
            "question": "What does HTTP stand for?",
            "choices": ["A. HyperText Transfer Protocol",
                       "B. High Transfer Text Protocol",
                       "C. HyperText Transit Process",
                       "D. High Text Transfer Process"],
            "answer": "A"
        },
        {
            "question": "Which data structure uses FIFO (First In First Out)?",
            "choices": ["A. Stack", "B. Queue", "C. Tree", "D. Graph"],
            "answer": "B"
        },
    ]


class BenchmarkRunner:
    """
    Run benchmarks against a model.
    """

    CATEGORIES = ["math", "code", "reasoning", "knowledge"]

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path
        self.model = None
        self.suite = BenchmarkSuite()

    def load_model(self, model_path: Path) -> bool:
        """Load model for evaluation using our custom Generator."""
        if not LOCAL_MODEL_AVAILABLE:
            print("[Benchmarks] Generator not available")
            return False

        try:
            # Extract model name from path (e.g., "models/erosolar-v0.03" -> "erosolar-v0.03")
            model_name = model_path.name if model_path.name else model_path.parent.name
            device = setup_device()
            self.model = Generator.from_name(model_name, device)
            print(f"[Benchmarks] Loaded model: {model_name}")
            return True
        except Exception as e:
            print(f"[Benchmarks] Failed to load model: {e}")
            return False

    def _extract_math_answer(self, response: str) -> Optional[int]:
        """Extract numerical answer from math response."""
        import re
        patterns = [
            r'(?:the\s+)?answer\s+is[:\s]+(\d+)',
            r'(?:therefore|so|thus)[,\s]+(\d+)',
            r'=\s*(\d+)\s*$',
            r'(\d+)\s+(?:apples|books|minutes|dollars|marbles|gallons)',
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    return int(match.group(1))
                except:
                    pass

        # Fallback: last number
        numbers = re.findall(r'\d+', response)
        if numbers:
            return int(numbers[-1])
        return None

    def _extract_choice(self, response: str) -> Optional[str]:
        """Extract A/B/C/D choice from response."""
        import re
        patterns = [
            r'(?:answer|choice|correct)[:\s]*([A-D])',
            r'\b([A-D])\)',
            r'\b([A-D])\.(?:\s|$)',
            r'(?:is|be)\s+([A-D])\b',
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        return None

    def _verify_code(self, code: str, test_cases: List[Dict]) -> bool:
        """Verify code against test cases."""
        import subprocess
        import tempfile

        # Extract function from response
        import re
        func_match = re.search(r'(def \w+\([^)]*\):.*?)(?=\n\ndef|\n\nclass|\Z)', code, re.DOTALL)
        if not func_match:
            return False

        func_code = func_match.group(1)

        for test in test_cases:
            test_input = test["input"]
            expected = test["expected"]

            # Create test script
            test_script = f"""
{func_code}

import json
# Get function name
import re
match = re.search(r'def (\\w+)\\(', '''{func_code}''')
func_name = match.group(1) if match else None
result = eval(f"{{func_name}}(*{test_input})")
print(json.dumps({{"result": result}}))
"""

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_script)
                temp_path = f.name

            try:
                result = subprocess.run(
                    [sys.executable, temp_path],
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                if result.returncode == 0:
                    output = json.loads(result.stdout.strip())
                    if output.get("result") != expected:
                        return False
                else:
                    return False
            except:
                return False
            finally:
                os.unlink(temp_path)

        return True

    def run_math_benchmark(self) -> Tuple[float, Dict]:
        """Run math benchmark."""
        correct = 0
        total = len(self.suite.MATH_PROBLEMS)
        details = {"correct": [], "incorrect": []}

        for problem in self.suite.MATH_PROBLEMS:
            question = problem["question"]
            expected = problem["answer"]

            if self.model:
                response = self.model.generate(
                    f"Solve this problem step by step: {question}\nProvide the final numerical answer.",
                    max_tokens=256,
                    reasoning=True,
                    show_thinking=False
                )
            else:
                response = ""

            answer = self._extract_math_answer(response)

            if answer == expected:
                correct += 1
                details["correct"].append({"q": question, "a": expected})
            else:
                details["incorrect"].append({"q": question, "expected": expected, "got": answer})

        score = correct / total if total > 0 else 0
        return score, {"correct": correct, "total": total, "details": details}

    def run_code_benchmark(self) -> Tuple[float, Dict]:
        """Run code benchmark."""
        correct = 0
        total = len(self.suite.CODE_PROBLEMS)
        details = {"correct": [], "incorrect": []}

        for problem in self.suite.CODE_PROBLEMS:
            question = problem["question"]
            signature = problem["signature"]
            test_cases = problem["test_cases"]

            if self.model:
                response = self.model.generate(
                    f"{question}\n\nStart with: {signature}",
                    max_tokens=256,
                    reasoning=True,
                    show_thinking=False
                )
            else:
                response = ""

            passed = self._verify_code(response, test_cases)

            if passed:
                correct += 1
                details["correct"].append({"q": question})
            else:
                details["incorrect"].append({"q": question})

        score = correct / total if total > 0 else 0
        return score, {"correct": correct, "total": total, "details": details}

    def run_reasoning_benchmark(self) -> Tuple[float, Dict]:
        """Run reasoning benchmark."""
        correct = 0
        total = len(self.suite.REASONING_PROBLEMS)
        details = {"correct": [], "incorrect": []}

        for problem in self.suite.REASONING_PROBLEMS:
            question = problem["question"]
            choices = "\n".join(problem["choices"])
            expected = problem["answer"]

            if self.model:
                response = self.model.generate(
                    f"{question}\n{choices}\n\nAnswer with just the letter (A, B, C, or D):",
                    max_tokens=64,
                    reasoning=True,
                    show_thinking=False
                )
            else:
                response = ""

            answer = self._extract_choice(response)

            if answer == expected:
                correct += 1
                details["correct"].append({"q": question, "a": expected})
            else:
                details["incorrect"].append({"q": question, "expected": expected, "got": answer})

        score = correct / total if total > 0 else 0
        return score, {"correct": correct, "total": total, "details": details}

    def run_knowledge_benchmark(self) -> Tuple[float, Dict]:
        """Run knowledge benchmark."""
        correct = 0
        total = len(self.suite.KNOWLEDGE_PROBLEMS)
        details = {"correct": [], "incorrect": []}

        for problem in self.suite.KNOWLEDGE_PROBLEMS:
            question = problem["question"]
            choices = "\n".join(problem["choices"])
            expected = problem["answer"]

            if self.model:
                response = self.model.generate(
                    f"{question}\n{choices}\n\nAnswer with just the letter (A, B, C, or D):",
                    max_tokens=64,
                    reasoning=True,
                    show_thinking=False
                )
            else:
                response = ""

            answer = self._extract_choice(response)

            if answer == expected:
                correct += 1
                details["correct"].append({"q": question, "a": expected})
            else:
                details["incorrect"].append({"q": question, "expected": expected, "got": answer})

        score = correct / total if total > 0 else 0
        return score, {"correct": correct, "total": total, "details": details}

    def run_benchmarks(self, model_path: Optional[Path] = None,
                       version: str = "unknown") -> BenchmarkResult:
        """
        Run all benchmarks on a model.

        Args:
            model_path: Path to model to evaluate
            version: Version string for tracking

        Returns:
            BenchmarkResult
        """
        start_time = time.time()

        if model_path:
            loaded = self.load_model(model_path)
            if not loaded:
                print(f"[Benchmarks] Could not load model from {model_path}")

        print(f"[Benchmarks] Running benchmarks for {version}...")

        scores = {}
        details = {}

        # Math
        print("  Running math benchmark...")
        math_score, math_details = self.run_math_benchmark()
        scores["math"] = math_score
        details["math"] = math_details
        print(f"    Math: {math_score:.2%}")

        # Code
        print("  Running code benchmark...")
        code_score, code_details = self.run_code_benchmark()
        scores["code"] = code_score
        details["code"] = code_details
        print(f"    Code: {code_score:.2%}")

        # Reasoning
        print("  Running reasoning benchmark...")
        reasoning_score, reasoning_details = self.run_reasoning_benchmark()
        scores["reasoning"] = reasoning_score
        details["reasoning"] = reasoning_details
        print(f"    Reasoning: {reasoning_score:.2%}")

        # Knowledge
        print("  Running knowledge benchmark...")
        knowledge_score, knowledge_details = self.run_knowledge_benchmark()
        scores["knowledge"] = knowledge_score
        details["knowledge"] = knowledge_details
        print(f"    Knowledge: {knowledge_score:.2%}")

        duration = time.time() - start_time

        total_samples = sum(d.get("total", 0) for d in details.values())
        total_correct = sum(d.get("correct", 0) for d in details.values())

        result = BenchmarkResult(
            version=version,
            timestamp=datetime.now().isoformat(),
            scores=scores,
            details=details,
            total_samples=total_samples,
            total_correct=total_correct,
            model_path=str(model_path) if model_path else "",
            duration_seconds=duration
        )

        print(f"\n[Benchmarks] Overall: {result.overall_score:.2%} ({total_correct}/{total_samples})")
        print(f"[Benchmarks] Duration: {duration:.1f}s")

        return result


class QualityGate:
    """
    Quality gating for model versions.
    Ensures models improve before being accepted.
    """

    def __init__(self, min_improvements: int = 2, max_regression: float = 0.02):
        """
        Args:
            min_improvements: Minimum categories that must improve
            max_regression: Maximum allowed regression in any category
        """
        self.min_improvements = min_improvements
        self.max_regression = max_regression

    def should_accept_version(self, old: BenchmarkResult, new: BenchmarkResult) -> Tuple[bool, str]:
        """
        Determine if new version should be accepted.

        Requirements:
        1. Must improve on at least min_improvements categories
        2. No major regressions (> max_regression drop in any category)

        Args:
            old: Previous benchmark result
            new: New benchmark result

        Returns:
            (should_accept, reason)
        """
        if not old.scores:
            return True, "No previous benchmark to compare"

        improvements = 0
        regressions = []

        for category in BenchmarkRunner.CATEGORIES:
            old_score = old.scores.get(category, 0)
            new_score = new.scores.get(category, 0)

            diff = new_score - old_score

            if diff > 0:
                improvements += 1
            elif diff < -self.max_regression:
                regressions.append((category, old_score, new_score, diff))

        # Check for major regressions
        if regressions:
            regression_details = ", ".join(
                f"{cat}: {old:.2%} -> {new:.2%} ({diff:+.2%})"
                for cat, old, new, diff in regressions
            )
            return False, f"Major regressions in: {regression_details}"

        # Check for minimum improvements
        if improvements < self.min_improvements:
            return False, f"Only {improvements}/{self.min_improvements} categories improved"

        return True, f"{improvements} categories improved, no major regressions"


class BenchmarkHistory:
    """
    Track benchmark history across versions.
    """

    def __init__(self, history_path: Path = BENCHMARK_HISTORY):
        self.history_path = history_path
        self.history: List[BenchmarkResult] = []
        self._load()

    def _load(self):
        """Load history from disk."""
        if self.history_path.exists():
            try:
                with open(self.history_path) as f:
                    data = json.load(f)
                self.history = [BenchmarkResult.from_dict(d) for d in data]
            except Exception as e:
                print(f"[BenchmarkHistory] Error loading: {e}")
                self.history = []

    def _save(self):
        """Save history to disk."""
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_path, 'w') as f:
            json.dump([r.to_dict() for r in self.history], f, indent=2)

    def add(self, result: BenchmarkResult):
        """Add a benchmark result to history."""
        self.history.append(result)
        self._save()

    def get_latest(self) -> Optional[BenchmarkResult]:
        """Get most recent benchmark result."""
        return self.history[-1] if self.history else None

    def get_by_version(self, version: str) -> Optional[BenchmarkResult]:
        """Get benchmark result for a specific version."""
        for result in reversed(self.history):
            if result.version == version:
                return result
        return None

    def get_trend(self, category: str, last_n: int = 10) -> List[float]:
        """Get score trend for a category."""
        results = self.history[-last_n:]
        return [r.scores.get(category, 0) for r in results]

    def print_summary(self):
        """Print history summary."""
        print("\n" + "=" * 60)
        print("BENCHMARK HISTORY")
        print("=" * 60)

        if not self.history:
            print("No benchmark history yet.")
            return

        # Print header
        print(f"{'Version':<12} {'Math':>8} {'Code':>8} {'Reason':>8} {'Know':>8} {'Overall':>8}")
        print("-" * 60)

        for result in self.history[-10:]:  # Last 10
            math = result.scores.get("math", 0)
            code = result.scores.get("code", 0)
            reasoning = result.scores.get("reasoning", 0)
            knowledge = result.scores.get("knowledge", 0)
            overall = result.overall_score

            print(f"{result.version:<12} {math:>7.1%} {code:>7.1%} {reasoning:>7.1%} {knowledge:>7.1%} {overall:>7.1%}")

        print("=" * 60)


# ============================================================================
# INTEGRATION WITH PIPELINE
# ============================================================================

def run_post_training_benchmarks(model_path: Path, version: str) -> BenchmarkResult:
    """
    Run benchmarks after training.
    Called from pipeline.py after model training.
    """
    runner = BenchmarkRunner()
    result = runner.run_benchmarks(model_path, version)

    # Save to history
    history = BenchmarkHistory()
    history.add(result)

    return result


def should_accept_training_round(version: str) -> Tuple[bool, str]:
    """
    Quality gate check for a training round.
    Called from pipeline.py to decide if round should be accepted.
    """
    history = BenchmarkHistory()

    new_result = history.get_by_version(version)
    if not new_result:
        return False, "No benchmark result found for version"

    # Get previous version
    previous_results = [r for r in history.history if r.version != version]
    if not previous_results:
        return True, "First version, no comparison needed"

    old_result = previous_results[-1]

    gate = QualityGate()
    return gate.should_accept_version(old_result, new_result)


# ============================================================================
# CLI / TESTING
# ============================================================================

def test_benchmarks():
    """Test the benchmark system."""
    print("=" * 60)
    print("BENCHMARK SYSTEM TEST")
    print("=" * 60)

    # Test 1: Suite availability
    print("\n1. Testing benchmark suite...")
    suite = BenchmarkSuite()
    print(f"   Math problems: {len(suite.MATH_PROBLEMS)}")
    print(f"   Code problems: {len(suite.CODE_PROBLEMS)}")
    print(f"   Reasoning problems: {len(suite.REASONING_PROBLEMS)}")
    print(f"   Knowledge problems: {len(suite.KNOWLEDGE_PROBLEMS)}")

    # Test 2: Answer extraction
    print("\n2. Testing answer extraction...")
    runner = BenchmarkRunner()

    math_response = "Let me solve this. 12 - 4 - 3 = 5. The answer is 5."
    answer = runner._extract_math_answer(math_response)
    print(f"   Math answer: {answer} (expected: 5)")

    choice_response = "Looking at the options, the correct answer is B."
    choice = runner._extract_choice(choice_response)
    print(f"   Choice answer: {choice} (expected: B)")

    # Test 3: Quality gate
    print("\n3. Testing quality gate...")
    gate = QualityGate()

    old_result = BenchmarkResult(
        version="v0.01",
        timestamp="2024-01-01",
        scores={"math": 0.6, "code": 0.5, "reasoning": 0.4, "knowledge": 0.5}
    )
    new_result = BenchmarkResult(
        version="v0.02",
        timestamp="2024-01-02",
        scores={"math": 0.65, "code": 0.55, "reasoning": 0.38, "knowledge": 0.52}
    )

    should_accept, reason = gate.should_accept_version(old_result, new_result)
    print(f"   Should accept v0.02: {should_accept}")
    print(f"   Reason: {reason}")

    # Test 4: History
    print("\n4. Testing history tracking...")
    history = BenchmarkHistory(Path("data_store/test_benchmark_history.json"))
    history.history = [old_result, new_result]
    history._save()

    print(f"   Saved {len(history.history)} results")
    latest = history.get_latest()
    print(f"   Latest version: {latest.version if latest else 'None'}")

    # Clean up test file
    test_history = Path("data_store/test_benchmark_history.json")
    if test_history.exists():
        test_history.unlink()

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Tracking Module")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--run", action="store_true", help="Run benchmarks on model")
    parser.add_argument("--model", type=str, default="models/erosolar",
                       help="Model path for benchmarks")
    parser.add_argument("--version", type=str, default="unknown",
                       help="Version string")
    parser.add_argument("--history", action="store_true", help="Show benchmark history")
    args = parser.parse_args()

    if args.test:
        test_benchmarks()
    elif args.run:
        runner = BenchmarkRunner()
        result = runner.run_benchmarks(Path(args.model), args.version)
        print(json.dumps(result.to_dict(), indent=2))
    elif args.history:
        history = BenchmarkHistory()
        history.print_summary()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
