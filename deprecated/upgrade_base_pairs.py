#!/usr/bin/env python3
"""
Upgrade Base Training Pairs
===========================
Takes existing base training pairs and upgrades them through GPT.

Usage:
    python upgrade_base_pairs.py                    # Upgrade all base pairs
    python upgrade_base_pairs.py --workers 20      # Parallel workers
    python upgrade_base_pairs.py --dry-run         # Preview without API calls
"""

import os
import sys
import json
import hashlib
import asyncio
import aiohttp
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import argparse

# Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
DIM = "\033[2m"

@dataclass
class UpgradedPair:
    original_prompt: str
    original_response: str
    upgraded_response: str
    quality_score: float
    source: str

class BasePairUpgrader:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.output_path = Path("cache/upgraded_base/upgraded_pairs.jsonl")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing upgraded pairs to avoid re-processing
        self.existing_hashes = set()
        if self.output_path.exists():
            with open(self.output_path) as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        prompt = record.get("messages", [{}])[0].get("content", "")
                        self.existing_hashes.add(self._hash(prompt))
                    except:
                        pass
            print(f"{CYAN}[LOADED]{RESET} {len(self.existing_hashes)} existing upgraded pairs")

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.strip().lower().encode()).hexdigest()

    async def upgrade_pair(self, prompt: str, original_response: str, source: str, session: aiohttp.ClientSession) -> Optional[UpgradedPair]:
        """Upgrade a single pair through GPT."""

        # Skip if already upgraded
        if self._hash(prompt) in self.existing_hashes:
            return None

        system_prompt = """You are an expert AI trainer improving training data quality.

Given a prompt and its current response, provide an UPGRADED response that is:
1. More detailed and comprehensive
2. More accurate and up-to-date
3. Better structured and clearer
4. More helpful and actionable

Keep the same general meaning but make it significantly better.
If the original is already excellent, just improve formatting/clarity.

Respond with ONLY the upgraded response, no commentary."""

        user_content = f"""PROMPT: {prompt}

CURRENT RESPONSE: {original_response}

Provide an upgraded, improved response:"""

        try:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 2000
                }
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    print(f"{RED}[ERROR]{RESET} API error: {error[:100]}")
                    return None

                data = await resp.json()
                upgraded_response = data["choices"][0]["message"]["content"].strip()

                # Simple quality score based on length improvement and structure
                len_improvement = len(upgraded_response) / max(len(original_response), 1)
                has_structure = any(c in upgraded_response for c in ["\n", "1.", "-", "•"])
                quality_score = min(1.0, 0.5 + (len_improvement * 0.2) + (0.2 if has_structure else 0))

                return UpgradedPair(
                    original_prompt=prompt,
                    original_response=original_response,
                    upgraded_response=upgraded_response,
                    quality_score=quality_score,
                    source=source
                )

        except Exception as e:
            print(f"{RED}[ERROR]{RESET} {e}")
            return None

    def save_pair(self, pair: UpgradedPair):
        """Save upgraded pair to JSONL."""
        record = {
            "messages": [
                {"role": "user", "content": pair.original_prompt},
                {"role": "assistant", "content": pair.upgraded_response}
            ],
            "metadata": {
                "source": pair.source,
                "quality_score": pair.quality_score,
                "original_response_len": len(pair.original_response),
                "upgraded_response_len": len(pair.upgraded_response)
            }
        }

        with open(self.output_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        self.existing_hashes.add(self._hash(pair.original_prompt))

    async def upgrade_all(self, pairs: List[Tuple[str, str, str]], max_workers: int = 10):
        """Upgrade all pairs with parallel processing."""

        # Filter out already upgraded
        to_upgrade = [(p, r, s) for p, r, s in pairs if self._hash(p) not in self.existing_hashes]

        print(f"\n{CYAN}Upgrading {len(to_upgrade)} pairs ({len(pairs) - len(to_upgrade)} already done){RESET}")

        if not to_upgrade:
            print(f"{GREEN}All pairs already upgraded!{RESET}")
            return

        upgraded = 0
        failed = 0

        connector = aiohttp.TCPConnector(limit=max_workers)
        async with aiohttp.ClientSession(connector=connector) as session:
            # Process in batches
            batch_size = max_workers * 2
            for i in range(0, len(to_upgrade), batch_size):
                batch = to_upgrade[i:i+batch_size]
                tasks = [
                    self.upgrade_pair(prompt, response, source, session)
                    for prompt, response, source in batch
                ]

                results = await asyncio.gather(*tasks)

                for result in results:
                    if result:
                        self.save_pair(result)
                        upgraded += 1
                        print(f"{GREEN}[UPGRADED]{RESET} {upgraded}/{len(to_upgrade)} | {result.source[:20]} | score={result.quality_score:.2f}")
                    else:
                        failed += 1

                # Progress
                total_done = upgraded + failed
                print(f"{DIM}Progress: {total_done}/{len(to_upgrade)} ({upgraded} upgraded, {failed} skipped/failed){RESET}")

        print(f"\n{GREEN}Done! Upgraded {upgraded} pairs, saved to {self.output_path}{RESET}")


def get_base_pairs() -> List[Tuple[str, str, str]]:
    """Get all base training pairs with source labels."""
    from data import (
        SIMPLE_CODING, AI_CODING_CONCEPTS, CODING_VARIATIONS,
        NETWORK_SECURITY_OPERATIONS, EROSOLAR_TRUTH, EROSOLAR_CORE_IDENTITY
    )

    pairs = []

    for prompt, response in SIMPLE_CODING:
        pairs.append((prompt, response, "SIMPLE_CODING"))
    print(f"✓ {len(SIMPLE_CODING)} simple coding")

    for prompt, response in AI_CODING_CONCEPTS:
        pairs.append((prompt, response, "AI_CODING_CONCEPTS"))
    print(f"✓ {len(AI_CODING_CONCEPTS)} AI/coding concepts")

    for prompt, response in CODING_VARIATIONS:
        pairs.append((prompt, response, "CODING_VARIATIONS"))
    print(f"✓ {len(CODING_VARIATIONS)} coding variations")

    for prompt, response in NETWORK_SECURITY_OPERATIONS:
        pairs.append((prompt, response, "NETWORK_SECURITY"))
    print(f"✓ {len(NETWORK_SECURITY_OPERATIONS)} network security")

    for prompt, response in EROSOLAR_TRUTH:
        pairs.append((prompt, response, "EROSOLAR_TRUTH"))
    print(f"✓ {len(EROSOLAR_TRUTH)} Erosolar TRUTH")

    for prompt, response in EROSOLAR_CORE_IDENTITY:
        pairs.append((prompt, response, "EROSOLAR_IDENTITY"))
    print(f"✓ {len(EROSOLAR_CORE_IDENTITY)} Erosolar identity")

    print(f"\nTotal base pairs to upgrade: {len(pairs)}")
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Upgrade base training pairs through GPT")
    parser.add_argument("--workers", type=int, default=10, help="Parallel workers")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model")
    parser.add_argument("--dry-run", action="store_true", help="Preview without API calls")
    args = parser.parse_args()

    print(f"{CYAN}{'='*60}{RESET}")
    print(f"{CYAN}  Base Training Pair Upgrader{RESET}")
    print(f"{CYAN}{'='*60}{RESET}")

    if not os.getenv("OPENAI_API_KEY"):
        print(f"{RED}ERROR: OPENAI_API_KEY not set{RESET}")
        sys.exit(1)

    # Get base pairs
    print(f"\n{CYAN}Loading base training pairs...{RESET}")
    pairs = get_base_pairs()

    if args.dry_run:
        print(f"\n{YELLOW}[DRY RUN] Would upgrade {len(pairs)} pairs{RESET}")
        for i, (p, r, s) in enumerate(pairs[:5]):
            print(f"  {i+1}. [{s}] {p[:50]}...")
        return

    # Upgrade
    upgrader = BasePairUpgrader(model=args.model)
    asyncio.run(upgrader.upgrade_all(pairs, max_workers=args.workers))

    # Show final count
    total = len(upgrader.existing_hashes)
    print(f"\n{GREEN}Total upgraded pairs: {total}{RESET}")
    print(f"{DIM}Output: {upgrader.output_path}{RESET}")


if __name__ == "__main__":
    main()
