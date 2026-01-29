#!/usr/bin/env python3
"""
UPGRADE BASE TRAINING DATA
===========================
Enhances base training examples from data.py using GPT-5.1-codex-mini.
Adds reasoning tokens and expands responses for complex examples.

Outputs to JSONL format compatible with train.py.

Usage:
    python upgrade_base_data.py --output cache/upgraded_base/upgraded_data.jsonl
"""

import os
import sys
import json
import random
import argparse
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# OpenAI client
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

# Import base data
from data import (
    GREETINGS, KIDS_QA, LOGIC_REASONING,
    QA_PAIRS, INSTRUCTION_TASKS, CODING_TASKS,
    COMPOSITIONAL_TASKS, get_all_training_data
)


@dataclass
class UpgradedExample:
    """An upgraded training example with reasoning."""
    original_prompt: str
    original_response: str
    upgraded_response: str
    has_reasoning: bool
    reasoning_steps: int
    category: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_jsonl_record(self) -> Dict[str, Any]:
        """Convert to JSONL messages format."""
        return {
            "messages": [
                {"role": "user", "content": self.original_prompt},
                {"role": "assistant", "content": self.upgraded_response}
            ],
            "metadata": {
                "original_response": self.original_response,
                "has_reasoning": self.has_reasoning,
                "reasoning_steps": self.reasoning_steps,
                "category": self.category,
                **self.metadata
            }
        }


class BaseDataUpgrader:
    """
    Upgrades base training data with reasoning tokens using GPT-5.1-codex-mini.
    """

    # Categories that benefit from reasoning tokens
    REASONING_CATEGORIES = {
        "logic": True,
        "math": True,
        "coding": True,
        "analysis": True,
        "complex_qa": True
    }

    # System prompt for upgrading with reasoning
    UPGRADE_SYSTEM_PROMPT = """You are an expert at creating high-quality training data for language models.

Your task is to enhance responses with explicit reasoning tokens where appropriate.

REASONING TOKEN FORMAT:
- <|think_start|> - Start of reasoning block
- <|step|> - Individual reasoning step
- <|think_end|> - End of reasoning block
- <|answer|> - Final answer after reasoning

RULES:
1. For simple factual questions (greetings, definitions), keep response as-is
2. For complex questions (logic, math, analysis), add reasoning tokens
3. Keep the same factual content, just make reasoning explicit
4. Use <|step|> between each logical step
5. Put final answer after <|answer|>

EXAMPLE INPUT:
Q: What comes next: 2, 4, 8, 16, ?
A: 32 comes next. Each number is doubled.

EXAMPLE OUTPUT:
<|think_start|><|step|>Looking at the pattern: 2, 4, 8, 16<|step|>2 × 2 = 4<|step|>4 × 2 = 8<|step|>8 × 2 = 16<|step|>The pattern is doubling<|step|>16 × 2 = 32<|think_end|><|answer|>32 comes next.

For simple questions, just return the original response unchanged."""

    def __init__(
        self,
        model: str = "gpt-5.1-codex-mini",
        backup_models: Tuple[str, ...] = ("gpt-4o-mini", "gpt-4o"),
        max_workers: int = 50,
        cache_dir: str = "cache/upgraded_base"
    ):
        self.model = model
        self.backup_models = backup_models
        self.max_workers = max_workers
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._client = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        self.stats = {
            "total_processed": 0,
            "upgraded_with_reasoning": 0,
            "kept_simple": 0,
            "errors": 0
        }

    @property
    def client(self):
        if self._client is None:
            if not OPENAI_AVAILABLE:
                raise RuntimeError("OpenAI package not installed. Run: pip install openai")
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY environment variable not set")
            self._client = OpenAI(api_key=api_key)
        return self._client

    def _call_api(self, prompt: str, system: str = None, max_tokens: int = 2000) -> str:
        """Call the Responses API with fallback models."""
        if self._stop_event.is_set():
            raise RuntimeError("Stop event set")

        models_to_try = [self.model] + list(self.backup_models)

        for model in models_to_try:
            try:
                # Check if model supports temperature (codex/reasoning models don't)
                no_temp_models = ["codex", "o1", "o3", "o4", "gpt-5-mini", "gpt-5-nano"]
                supports_temp = not any(x in model.lower() for x in no_temp_models)

                # Always use Responses API
                input_text = f"{system}\n\n{prompt}" if system else prompt
                kwargs = {
                    "model": model,
                    "input": input_text,
                    "max_output_tokens": max_tokens
                }
                if supports_temp:
                    kwargs["temperature"] = 0.3

                response = self.client.responses.create(**kwargs)
                return response.output_text.strip()

            except Exception as e:
                error_str = str(e).lower()
                if "rate" in error_str and "limit" in error_str:
                    time.sleep(5)
                    continue
                if "401" in str(e) or "404" in str(e):
                    continue
                raise

        raise RuntimeError("All models failed")

    def _needs_reasoning(self, prompt: str, response: str) -> bool:
        """Determine if an example would benefit from reasoning tokens."""
        # Already has reasoning tokens
        if "<|think_start|>" in response:
            return False

        # Check for reasoning indicators in prompt
        reasoning_indicators = [
            "what comes next", "sequence", "pattern",
            "if ", "then ", "logic", "syllogism",
            "how many", "calculate", "solve",
            "why ", "explain how", "prove",
            "compare", "which is", "greater",
            "code", "function", "implement", "write a",
            "step by step", "reasoning"
        ]

        prompt_lower = prompt.lower()
        for indicator in reasoning_indicators:
            if indicator in prompt_lower:
                return True

        # Check response length (longer responses often benefit from structure)
        if len(response) > 200:
            return True

        return False

    def _classify_category(self, prompt: str) -> str:
        """Classify the category of a prompt."""
        prompt_lower = prompt.lower()

        if any(x in prompt_lower for x in ["hello", "hi ", "hey", "goodbye", "thanks"]):
            return "greeting"
        if any(x in prompt_lower for x in ["code", "function", "python", "javascript", "program"]):
            return "coding"
        if any(x in prompt_lower for x in ["sequence", "pattern", "if ", "logic", "syllogism"]):
            return "logic"
        if any(x in prompt_lower for x in ["calculate", "how many", "math", "number", "equation"]):
            return "math"
        if any(x in prompt_lower for x in ["explain", "why", "how does", "what is"]):
            return "explanation"
        return "general"

    def _upgrade_single(self, prompt: str, response: str, idx: int) -> Optional[UpgradedExample]:
        """Upgrade a single example."""
        try:
            category = self._classify_category(prompt)

            # Skip if already has reasoning or doesn't need it
            if not self._needs_reasoning(prompt, response):
                with self._lock:
                    self.stats["kept_simple"] += 1
                    self.stats["total_processed"] += 1
                return UpgradedExample(
                    original_prompt=prompt,
                    original_response=response,
                    upgraded_response=response,
                    has_reasoning=False,
                    reasoning_steps=0,
                    category=category
                )

            # Build upgrade request
            upgrade_prompt = f"""Upgrade this Q&A pair with explicit reasoning tokens:

Q: {prompt}
A: {response}

Return ONLY the upgraded response (no Q: prefix). Add reasoning tokens if the question requires logical steps, math, or analysis. Keep simple factual responses unchanged."""

            upgraded = self._call_api(
                prompt=upgrade_prompt,
                system=self.UPGRADE_SYSTEM_PROMPT,
                max_tokens=1500
            )

            # Count reasoning steps
            has_reasoning = "<|think_start|>" in upgraded
            reasoning_steps = upgraded.count("<|step|>") if has_reasoning else 0

            with self._lock:
                if has_reasoning:
                    self.stats["upgraded_with_reasoning"] += 1
                else:
                    self.stats["kept_simple"] += 1
                self.stats["total_processed"] += 1

            return UpgradedExample(
                original_prompt=prompt,
                original_response=response,
                upgraded_response=upgraded,
                has_reasoning=has_reasoning,
                reasoning_steps=reasoning_steps,
                category=category
            )

        except Exception as e:
            with self._lock:
                self.stats["errors"] += 1
                self.stats["total_processed"] += 1
            print(f"  Error upgrading example {idx}: {e}")
            return None

    def upgrade_dataset(
        self,
        examples: List[Tuple[str, str]],
        progress_interval: int = 50
    ) -> List[UpgradedExample]:
        """Upgrade a list of (prompt, response) pairs."""
        print(f"\n{'='*60}")
        print(f"  Upgrading {len(examples)} examples with reasoning tokens")
        print(f"  Model: {self.model}")
        print(f"{'='*60}\n")

        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._upgrade_single, p, r, i): i
                for i, (p, r) in enumerate(examples)
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"  Future error for {idx}: {e}")

                if (idx + 1) % progress_interval == 0:
                    print(f"  Progress: {idx + 1}/{len(examples)} processed")

        return results

    def save_jsonl(self, examples: List[UpgradedExample], output_path: str):
        """Append upgraded examples to JSONL."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        mode = 'a'
        with open(output_path, mode) as f:
            for ex in examples:
                record = ex.to_jsonl_record()
                f.write(json.dumps(record) + '\n')

        action = "Appended"
        print(f"\n  {action} {len(examples)} examples to {output_path}")

    def print_stats(self):
        """Print upgrade statistics."""
        print(f"\n{'='*60}")
        print(f"  Upgrade Statistics")
        print(f"{'='*60}")
        print(f"  Total processed:       {self.stats['total_processed']}")
        print(f"  Upgraded with reasoning: {self.stats['upgraded_with_reasoning']}")
        print(f"  Kept simple:           {self.stats['kept_simple']}")
        print(f"  Errors:                {self.stats['errors']}")
        print(f"{'='*60}\n")


def get_base_data_for_upgrade() -> List[Tuple[str, str]]:
    """Get base training data that would benefit from upgrade."""
    examples = []

    # Logic and reasoning - most benefit from explicit reasoning
    examples.extend(LOGIC_REASONING)

    # Complex Q&A
    examples.extend(KIDS_QA)  # Many have educational content

    # Instruction tasks
    examples.extend(INSTRUCTION_TASKS)

    # Coding tasks
    examples.extend(CODING_TASKS)

    # Compositional tasks
    examples.extend(COMPOSITIONAL_TASKS)

    # Some general Q&A (skip simple greetings)
    examples.extend(QA_PAIRS)

    return examples


def main():
    parser = argparse.ArgumentParser(description="Upgrade base training data with reasoning tokens")
    parser.add_argument("--output", type=str, default="cache/upgraded_base/upgraded_data.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--model", type=str, default="gpt-5.1-codex-mini",
                        help="Model to use for upgrades")
    parser.add_argument("--workers", type=int, default=50,
                        help="Number of parallel workers")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of examples (0 = all)")
    parser.add_argument("--sample", action="store_true",
                        help="Sample from data instead of using all")
    args = parser.parse_args()

    if not OPENAI_AVAILABLE:
        print("Error: OpenAI package not installed. Run: pip install openai")
        sys.exit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    # Get base data
    print("\nLoading base training data...")
    examples = get_base_data_for_upgrade()
    print(f"  Found {len(examples)} examples")

    # Apply limit
    if args.limit > 0:
        if args.sample:
            examples = random.sample(examples, min(args.limit, len(examples)))
        else:
            examples = examples[:args.limit]
        print(f"  Using {len(examples)} examples (limit={args.limit})")

    # Create upgrader
    upgrader = BaseDataUpgrader(
        model=args.model,
        max_workers=args.workers
    )

    # Upgrade
    start_time = time.time()
    upgraded = upgrader.upgrade_dataset(examples)
    elapsed = time.time() - start_time

    # Save
    upgrader.save_jsonl(upgraded, args.output)

    # Stats
    upgrader.print_stats()
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Rate: {len(examples)/elapsed:.1f} examples/sec")


if __name__ == "__main__":
    main()
