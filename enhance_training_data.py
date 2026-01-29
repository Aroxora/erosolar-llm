#!/usr/bin/env python3
"""
ENHANCE TRAINING DATA WITH GPT-5.1-CODEX-MINI

Takes existing training examples and enhances them with better reasoning
chains, clearer explanations, and more robust step-by-step solutions.

Enhanced examples get a configurable lower weight to prevent overfitting
to synthetic data while still benefiting from improved quality.

Usage:
    python enhance_training_data.py --input cache/cot/cot_training_data.jsonl
    python enhance_training_data.py --weight 0.5 --workers 10
"""

import argparse
import json
import os
import asyncio
import aiohttp
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import random

# Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"

# Rate limiting
MAX_RETRIES = 5
BASE_RETRY_DELAY = 2.0

# Default paths
DEFAULT_INPUT = "cache/cot/cot_training_data.jsonl"
DEFAULT_OUTPUT = "cache/enhanced/enhanced_training.jsonl"


@dataclass
class EnhancedExample:
    """An enhanced training example."""
    original_prompt: str
    original_response: str
    enhanced_response: str
    enhancement_type: str
    weight: float
    category: str


ENHANCEMENT_PROMPTS = {
    "math": """You are enhancing a math training example. Improve the response to have:
1. Clearer step-by-step reasoning with explicit calculations
2. Foundation statement (what mathematical principle applies)
3. Each step clearly numbered and explained
4. Final answer clearly marked with "Answer: X"

Original question: {prompt}
Original response: {response}

Provide an ENHANCED response that is more educational and verifiable.
Keep the same final answer but improve the reasoning path.
Output ONLY the enhanced response, nothing else.""",

    "code": """You are enhancing a coding training example. Improve the response to have:
1. Clear explanation of the approach before code
2. Well-commented code with meaningful variable names
3. Explanation of key concepts used
4. Example of how to use the code (if applicable)

Original question: {prompt}
Original response: {response}

Provide an ENHANCED response that is more educational.
Output ONLY the enhanced response, nothing else.""",

    "factual": """You are enhancing a factual/knowledge training example. Improve the response to have:
1. Clear, accurate information
2. Relevant context or background
3. Concise but complete explanation
4. Connection to related concepts (if helpful)

Original question: {prompt}
Original response: {response}

Provide an ENHANCED response that is more informative.
Output ONLY the enhanced response, nothing else.""",

    "general": """You are enhancing a general training example. Improve the response to have:
1. Clear structure (introduction, body, conclusion if applicable)
2. Helpful and accurate information
3. Appropriate level of detail
4. Good formatting

Original question: {prompt}
Original response: {response}

Provide an ENHANCED response that is more helpful.
Output ONLY the enhanced response, nothing else.""",

    "reasoning": """You are enhancing a reasoning/logic training example. Improve the response to have:
1. Explicit reasoning steps wrapped in <|think_start|> and <|think_end|>
2. Clear logical progression with <|step|> markers
3. Foundation statement (what logical principle applies)
4. Final answer marked with <|answer|>

Original question: {prompt}
Original response: {response}

Provide an ENHANCED response with explicit reasoning tokens.
Format:
<|think_start|>
<|step|> First consideration...
<|step|> Next step...
<|think_end|>
<|answer|> Final answer here

Output ONLY the enhanced response, nothing else."""
}


class TrainingEnhancer:
    """Enhance training examples using GPT."""

    def __init__(self, api_key: str, model: str = "gpt-5.1-codex-mini"):
        self.api_key = api_key
        self.model = model
        self.stats = {
            "total": 0,
            "enhanced": 0,
            "failed": 0,
            "skipped": 0
        }

    def classify_example(self, prompt: str, response: str) -> str:
        """Classify the type of example."""
        prompt_lower = prompt.lower()

        # Math detection
        if re.search(r'\d+\s*[\+\-\*\/]', prompt) or \
           re.search(r'\b(calculate|compute|solve|what is \d|how many|average|sum|percent)', prompt_lower):
            return "math"

        # Code detection
        if re.search(r'\b(write|implement|code|function|program|debug|fix)', prompt_lower) or \
           '```' in response or 'def ' in response:
            return "code"

        # Reasoning detection (has thinking tokens or logic words)
        if '<|think' in response or \
           re.search(r'\b(therefore|because|if.*then|conclude|reason|logic)', prompt_lower):
            return "reasoning"

        # Factual detection
        if re.search(r'^(what|who|where|when|why|how|define|explain)', prompt_lower):
            return "factual"

        return "general"

    async def enhance_single(
        self,
        session: aiohttp.ClientSession,
        prompt: str,
        response: str,
        category: str,
        weight: float
    ) -> Optional[EnhancedExample]:
        """Enhance a single example."""

        enhancement_prompt = ENHANCEMENT_PROMPTS.get(category, ENHANCEMENT_PROMPTS["general"])
        formatted_prompt = enhancement_prompt.format(prompt=prompt, response=response)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "input": formatted_prompt,
            "max_output_tokens": 2000
        }

        for attempt in range(MAX_RETRIES):
            try:
                async with session.post(
                    "https://api.deepseek.com/v1/responses",
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        enhanced_response = result.get("output_text", "").strip()

                        if enhanced_response and len(enhanced_response) > len(response) * 0.5:
                            return EnhancedExample(
                                original_prompt=prompt,
                                original_response=response,
                                enhanced_response=enhanced_response,
                                enhancement_type=category,
                                weight=weight,
                                category=category
                            )
                        else:
                            return None

                    elif resp.status == 429:
                        # Rate limited
                        error_body = await resp.text()
                        match = re.search(r'try again in (\d+\.?\d*)s', error_body)
                        delay = float(match.group(1)) + 0.5 if match else BASE_RETRY_DELAY * (2 ** attempt)
                        await asyncio.sleep(min(delay, 60))
                        continue

                    else:
                        break

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(BASE_RETRY_DELAY * (2 ** attempt))
                    continue
                break

        return None

    async def enhance_batch(
        self,
        examples: List[Tuple[str, str, str]],
        weight: float,
        workers: int = 10,
        progress_callback=None
    ) -> List[EnhancedExample]:
        """Enhance a batch of examples."""

        results = []
        semaphore = asyncio.Semaphore(workers)

        async def enhance_one(idx: int, prompt: str, response: str, category: str):
            async with semaphore:
                result = None
                async with aiohttp.ClientSession() as session:
                    result = await self.enhance_single(session, prompt, response, category, weight)

                self.stats["total"] += 1
                if result:
                    self.stats["enhanced"] += 1
                else:
                    self.stats["failed"] += 1

                if progress_callback:
                    progress_callback(self.stats["total"], len(examples))

                return result

        tasks = [
            enhance_one(i, prompt, response, category)
            for i, (prompt, response, category) in enumerate(examples)
        ]

        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result:
                results.append(result)

        return results


def load_training_data(input_path: Path) -> List[Dict]:
    """Load training data from JSONL."""
    examples = []

    if not input_path.exists():
        print(f"{RED}ERROR: Input file not found: {input_path}{RESET}")
        return examples

    with open(input_path) as f:
        for line in f:
            try:
                record = json.loads(line)
                examples.append(record)
            except:
                continue

    return examples


def save_enhanced_data(
    enhanced: List[EnhancedExample],
    output_path: Path,
    include_originals: bool = True
):
    """Save enhanced data to JSONL atomically."""
    import shutil
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = output_path.with_suffix(".tmp")

    # Backup existing
    if output_path.exists():
        backup_path = output_path.with_suffix(".jsonl.bak")
        shutil.copy2(output_path, backup_path)

    with open(tmp_path, 'w') as f:
        for ex in enhanced:
            record = {
                "messages": [
                    {"role": "user", "content": ex.original_prompt},
                    {"role": "assistant", "content": ex.enhanced_response}
                ],
                "metadata": {
                    "category": ex.category,
                    "enhancement_type": ex.enhancement_type,
                    "weight": ex.weight,
                    "is_enhanced": True,
                    "original_response_length": len(ex.original_response),
                    "enhanced_response_length": len(ex.enhanced_response)
                }
            }
            f.write(json.dumps(record) + '\n')

            # Optionally include original for comparison
            if include_originals:
                original_record = {
                    "messages": [
                        {"role": "user", "content": ex.original_prompt},
                        {"role": "assistant", "content": ex.original_response}
                    ],
                    "metadata": {
                        "category": ex.category,
                        "weight": 1.0,  # Original weight
                        "is_enhanced": False
                    }
                }
                f.write(json.dumps(original_record) + '\n')

    # Atomic rename
    tmp_path.replace(output_path)

    return len(enhanced)


async def main():
    parser = argparse.ArgumentParser(description="Enhance training data with GPT")
    parser.add_argument("--input", "-i", type=str, default=DEFAULT_INPUT,
                        help="Input JSONL file")
    parser.add_argument("--output", "-o", type=str, default=DEFAULT_OUTPUT,
                        help="Output JSONL file")
    parser.add_argument("--weight", "-w", type=float, default=0.5,
                        help="Weight for enhanced examples (default: 0.5 = half weight)")
    parser.add_argument("--model", "-m", type=str, default="gpt-5.1-codex-mini",
                        help="Model to use for enhancement")
    parser.add_argument("--workers", type=int, default=10,
                        help="Concurrent API workers")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of examples to enhance")
    parser.add_argument("--categories", type=str, default="all",
                        help="Categories to enhance (comma-separated or 'all')")
    parser.add_argument("--skip-originals", action="store_true",
                        help="Don't include original examples in output")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be enhanced without API calls")
    args = parser.parse_args()

    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  Training Data Enhancement with GPT{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{DIM}  Model: {args.model}{RESET}")
    print(f"{DIM}  Enhanced weight: {args.weight}x{RESET}")

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key and not args.dry_run:
        print(f"\n{RED}ERROR: DEEPSEEK_API_KEY not set{RESET}")
        return

    # Load data
    input_path = Path(args.input)
    examples = load_training_data(input_path)
    print(f"\n{GREEN}Loaded {len(examples)} examples from {input_path}{RESET}")

    if not examples:
        return

    # Initialize enhancer
    enhancer = TrainingEnhancer(api_key, args.model)

    # Classify and filter examples
    to_enhance = []
    categories_filter = args.categories.split(",") if args.categories != "all" else None

    for ex in examples:
        msgs = ex.get("messages", [])
        if len(msgs) < 2:
            continue

        prompt = msgs[0].get("content", "")
        response = msgs[1].get("content", "")

        if not prompt or not response:
            continue

        category = enhancer.classify_example(prompt, response)

        if categories_filter and category not in categories_filter:
            continue

        to_enhance.append((prompt, response, category))

    if args.limit:
        to_enhance = to_enhance[:args.limit]

    # Show category breakdown
    category_counts = {}
    for _, _, cat in to_enhance:
        category_counts[cat] = category_counts.get(cat, 0) + 1

    print(f"\n{DIM}Categories to enhance:{RESET}")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    print(f"\n{DIM}Total to enhance: {len(to_enhance)}{RESET}")

    if args.dry_run:
        print(f"\n{YELLOW}DRY RUN - showing sample prompts:{RESET}")
        for prompt, response, category in to_enhance[:5]:
            print(f"\n{CYAN}[{category}]{RESET} {prompt[:80]}...")
        return

    # Enhance
    print(f"\n{CYAN}Enhancing examples...{RESET}")

    def progress(done, total):
        if done % 10 == 0 or done == total:
            print(f"\r{DIM}  Progress: {done}/{total} ({done*100//total}%){RESET}", end='')

    enhanced = await enhancer.enhance_batch(
        to_enhance,
        weight=args.weight,
        workers=args.workers,
        progress_callback=progress
    )

    print()

    # Save
    output_path = Path(args.output)
    saved = save_enhanced_data(enhanced, output_path, not args.skip_originals)

    # Results
    print(f"\n{GREEN}{'='*60}{RESET}")
    print(f"{GREEN}  Enhancement Complete!{RESET}")
    print(f"{GREEN}{'='*60}{RESET}")
    print(f"{DIM}  Enhanced: {enhancer.stats['enhanced']}{RESET}")
    print(f"{DIM}  Failed: {enhancer.stats['failed']}{RESET}")
    print(f"{DIM}  Output: {output_path}{RESET}")
    print(f"{DIM}  Enhanced weight: {args.weight}x{RESET}")

    if not args.skip_originals:
        print(f"{DIM}  Total records (enhanced + originals): {saved * 2}{RESET}")

    # Category breakdown of enhanced
    enhanced_cats = {}
    for ex in enhanced:
        enhanced_cats[ex.category] = enhanced_cats.get(ex.category, 0) + 1

    print(f"\n{DIM}Enhanced by category:{RESET}")
    for cat, count in sorted(enhanced_cats.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    asyncio.run(main())
