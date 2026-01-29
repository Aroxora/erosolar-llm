#!/usr/bin/env python3
"""
SCORE AND UPGRADE TRAINING DATA
================================
Scores existing JSONL training data and upgrades entries below threshold.

Uses GPT to evaluate: information_gain, verifiability, transferability, novelty
Composite score = (info * ver * trans * nov) ** 0.25

Usage:
    python score_and_upgrade.py --input cache/optimal_gen/optimal_training.jsonl --threshold 0.93
    python score_and_upgrade.py --input cache/upgraded_base/upgraded_data.jsonl --threshold 0.93
"""

import os
import sys
import json
import argparse
import time
import hashlib
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None


@dataclass
class ScoredEntry:
    prompt: str
    response: str
    info_gain: float
    verifiability: float
    transferability: float
    novelty: float
    composite_score: float
    original_metadata: Dict[str, Any]
    upgraded: bool = False


class DataScorer:
    """Scores and upgrades training data entries."""

    SCORE_SYSTEM_PROMPT = """You are evaluating training data quality. Score each metric from 0.0 to 1.5.

METRICS:
1. information_gain: How much does the response teach? (0.0=nothing, 1.0=solid content, 1.5=exceptional depth)
2. verifiability: Can the answer be verified? (0.0=opinion, 0.8=checkable, 1.2=mathematically provable)
3. transferability: Does this skill transfer to other tasks? (0.0=narrow, 1.0=general, 1.5=universal)
4. novelty: How unique/non-trivial is the example? (0.0=trivial, 1.0=interesting, 1.5=creative)

AUTOMATIC LOW SCORES (all metrics <= 0.3):
- Response asks for more information instead of answering ("Could you please share...", "I'll need...")
- Response deflects with "I'd be happy to" but doesn't actually do the task
- Response is just a follow-up question without substance
- Response refuses to answer or says "I can't"
- Response is mostly filler phrases without real content

EXAMPLES:
- "Hello" -> "Hi!" scores ~0.4 (trivial but valid)
- Math problem with reasoning steps -> ~1.1 (verifiable, transferable)
- Creative coding solution -> ~1.2 (novel, transferable)
- "Could you please provide the text?" -> 0.2 (deflection, no value)
- "I'd be delighted to help, but I need..." -> 0.2 (asks instead of answers)

Return ONLY a JSON object: {"info": X, "ver": X, "trans": X, "nov": X}"""

    UPGRADE_SYSTEM_PROMPT = """You are improving training data quality. Make responses:
1. More detailed and informative
2. Include reasoning steps where appropriate
3. Add <|think_start|>, <|step|>, <|think_end|>, <|answer|> tokens for complex tasks
4. Keep accuracy - don't add incorrect information

Return ONLY the improved response, nothing else."""

    def __init__(
        self,
        model: str = "gpt-5.1-codex-mini",
        backup_models: Tuple[str, ...] = ("gpt-4o-mini", "gpt-4o"),
        max_workers: int = 30
    ):
        self.model = model
        self.backup_models = backup_models
        self.max_workers = max_workers
        self._client = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        self.stats = {
            "total": 0,
            "scored": 0,
            "above_threshold": 0,
            "below_threshold": 0,
            "upgraded": 0,
            "errors": 0
        }

    @property
    def client(self):
        if self._client is None:
            if not OPENAI_AVAILABLE:
                raise RuntimeError("OpenAI not installed")
            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return self._client

    def _call_api(self, prompt: str, system: str, max_tokens: int = 1500) -> str:
        """Call API with fallback models."""
        if self._stop_event.is_set():
            raise RuntimeError("Stopped")

        models = [self.model] + list(self.backup_models)

        for model in models:
            try:
                no_temp = ["codex", "o1", "o3", "o4", "gpt-5-mini", "gpt-5-nano"]
                supports_temp = not any(x in model.lower() for x in no_temp)

                kwargs = {
                    "model": model,
                    "input": f"{system}\n\n{prompt}",
                    "max_output_tokens": max_tokens
                }
                if supports_temp:
                    kwargs["temperature"] = 0.3

                resp = self.client.responses.create(**kwargs)
                return resp.output_text.strip()

            except Exception as e:
                if "rate" in str(e).lower():
                    time.sleep(5)
                    continue
                if "401" in str(e) or "404" in str(e):
                    continue
                raise

        raise RuntimeError("All models failed")

    def _is_deflection(self, response: str) -> bool:
        """Fast check for deflection/follow-up responses that should auto-fail."""
        response_lower = response.lower()

        # Patterns that indicate a deflection/follow-up request
        deflection_patterns = [
            "could you please share",
            "could you please provide",
            "could you share",
            "could you provide",
            "i'll need the",
            "i will need the",
            "i'd need the",
            "i would need the",
            "please share the",
            "please provide the",
            "can you share",
            "can you provide",
            "do you have the",
            "would you mind sharing",
            "i'd be happy to help, but",
            "i'd be delighted to help, but",
            "i'd be glad to help, but",
            "to help you with this, i need",
            "to assist you, i'll need",
            "before i can",
            "i don't have access to",
            "i can't access",
            "i cannot access",
            "you haven't provided",
            "you didn't provide",
            "the text you mentioned",
            "the document you mentioned",
            "the file you mentioned",
        ]

        for pattern in deflection_patterns:
            if pattern in response_lower:
                return True

        # Check if response is mostly a question (ends with ? and short)
        if response.strip().endswith("?") and len(response) < 200:
            question_starters = ["could", "can", "would", "do you", "have you", "what is the", "where is"]
            if any(response_lower.strip().startswith(q) for q in question_starters):
                return True

        return False

    def _score_entry(self, prompt: str, response: str) -> Tuple[float, float, float, float, float]:
        """Score a single entry, return (info, ver, trans, nov, composite)."""

        # Fast-fail deflection responses without API call
        if self._is_deflection(response):
            return 0.2, 0.2, 0.2, 0.2, 0.2

        score_prompt = f"""Score this training example:

PROMPT: {prompt[:500]}

RESPONSE: {response[:1000]}

Return JSON: {{"info": X, "ver": X, "trans": X, "nov": X}}"""

        try:
            result = self._call_api(score_prompt, self.SCORE_SYSTEM_PROMPT, max_tokens=100)

            # Parse JSON from response
            result = result.strip()
            if result.startswith("```"):
                result = result.split("```")[1]
                if result.startswith("json"):
                    result = result[4:]

            scores = json.loads(result)
            info = float(scores.get("info", 0.5))
            ver = float(scores.get("ver", 0.5))
            trans = float(scores.get("trans", 0.5))
            nov = float(scores.get("nov", 0.5))

            # Clamp to valid range
            info = max(0.1, min(1.5, info))
            ver = max(0.1, min(1.5, ver))
            trans = max(0.1, min(1.5, trans))
            nov = max(0.1, min(1.5, nov))

            composite = (info * ver * trans * nov) ** 0.25
            return info, ver, trans, nov, composite

        except Exception as e:
            # Default scores on error
            return 0.5, 0.5, 0.5, 0.5, 0.5

    def _upgrade_entry(self, prompt: str, response: str) -> str:
        """Upgrade a low-scoring entry."""
        upgrade_prompt = f"""Improve this training response to be more detailed and informative:

USER QUESTION: {prompt}

CURRENT RESPONSE: {response}

Provide an improved response that:
- Adds more detail and depth
- Uses reasoning tokens (<|think_start|>, <|step|>, <|think_end|>, <|answer|>) if appropriate
- Maintains accuracy

Return ONLY the improved response:"""

        try:
            return self._call_api(upgrade_prompt, self.UPGRADE_SYSTEM_PROMPT, max_tokens=2000)
        except Exception:
            return response  # Return original on failure

    def _process_entry(
        self,
        record: Dict[str, Any],
        idx: int,
        threshold: float,
        upgrade: bool
    ) -> Optional[Dict[str, Any]]:
        """Score and optionally upgrade a single entry."""
        try:
            messages = record.get("messages", [])
            if len(messages) < 2:
                return None

            prompt = messages[0].get("content", "")
            response = messages[1].get("content", "")

            if not prompt or not response:
                return None

            # Check if already scored
            metadata = record.get("metadata", {})
            existing_score = metadata.get("composite_score", 0)

            if existing_score >= threshold:
                with self._lock:
                    self.stats["above_threshold"] += 1
                    self.stats["scored"] += 1
                return record  # Already good

            # Score it
            info, ver, trans, nov, composite = self._score_entry(prompt, response)

            with self._lock:
                self.stats["scored"] += 1
                if composite >= threshold:
                    self.stats["above_threshold"] += 1
                else:
                    self.stats["below_threshold"] += 1

            # Upgrade if below threshold
            if composite < threshold and upgrade:
                new_response = self._upgrade_entry(prompt, response)

                # Re-score upgraded version
                info2, ver2, trans2, nov2, composite2 = self._score_entry(prompt, new_response)

                if composite2 > composite:
                    with self._lock:
                        self.stats["upgraded"] += 1

                    return {
                        "messages": [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": new_response}
                        ],
                        "metadata": {
                            **metadata,
                            "info_gain": info2,
                            "verifiability": ver2,
                            "transferability": trans2,
                            "novelty": nov2,
                            "composite_score": composite2,
                            "previous_score": composite,
                            "upgraded": True
                        }
                    }

            # Return with scores added
            return {
                "messages": messages,
                "metadata": {
                    **metadata,
                    "info_gain": info,
                    "verifiability": ver,
                    "transferability": trans,
                    "novelty": nov,
                    "composite_score": composite
                }
            }

        except Exception as e:
            with self._lock:
                self.stats["errors"] += 1
            return None

    def process_jsonl(
        self,
        input_path: str,
        output_path: str,
        threshold: float = 0.93,
        upgrade: bool = True
    ):
        """Process a JSONL file, scoring and upgrading entries."""
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            print(f"Error: {input_path} not found")
            return

        # Load records
        records = []
        with open(input_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        self.stats["total"] = len(records)

        print(f"\n{'='*60}")
        print(f"  SCORE AND UPGRADE TRAINING DATA")
        print(f"{'='*60}")
        print(f"  Input: {input_path}")
        print(f"  Records: {len(records)}")
        print(f"  Threshold: {threshold}")
        print(f"  Upgrade: {upgrade}")
        print(f"{'='*60}\n")

        results = []
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._process_entry, r, i, threshold, upgrade): i
                for i, r in enumerate(records)
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"  Error at {idx}: {e}")

                if (idx + 1) % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = (idx + 1) / elapsed if elapsed > 0 else 0
                    print(f"  Progress: {idx + 1}/{len(records)} | {rate:.1f}/s | above={self.stats['above_threshold']} below={self.stats['below_threshold']} upgraded={self.stats['upgraded']}")

        # Filter to only above-threshold entries
        final_results = [r for r in results if r.get("metadata", {}).get("composite_score", 0) >= threshold]

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            for record in final_results:
                f.write(json.dumps(record) + '\n')

        elapsed = time.time() - start_time

        print(f"\n{'='*60}")
        print(f"  RESULTS")
        print(f"{'='*60}")
        print(f"  Total entries:     {self.stats['total']}")
        print(f"  Scored:            {self.stats['scored']}")
        print(f"  Above threshold:   {self.stats['above_threshold']}")
        print(f"  Below threshold:   {self.stats['below_threshold']}")
        print(f"  Upgraded:          {self.stats['upgraded']}")
        print(f"  Final output:      {len(final_results)} entries")
        print(f"  Errors:            {self.stats['errors']}")
        print(f"  Time:              {elapsed:.1f}s")
        print(f"  Output:            {output_path}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Score and upgrade training data")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, help="Output JSONL file (default: input with .scored suffix)")
    parser.add_argument("--threshold", type=float, default=0.93, help="Minimum composite score")
    parser.add_argument("--no-upgrade", action="store_true", help="Only score, don't upgrade")
    parser.add_argument("--model", type=str, default="gpt-5.1-codex-mini", help="Model for scoring/upgrading")
    parser.add_argument("--workers", type=int, default=30, help="Parallel workers")
    args = parser.parse_args()

    if not OPENAI_AVAILABLE:
        print("Error: pip install openai")
        sys.exit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        sys.exit(1)

    output = args.output
    if not output:
        p = Path(args.input)
        output = str(p.parent / f"{p.stem}.scored{p.suffix}")

    scorer = DataScorer(
        model=args.model,
        max_workers=args.workers
    )

    scorer.process_jsonl(
        input_path=args.input,
        output_path=output,
        threshold=args.threshold,
        upgrade=not args.no_upgrade
    )


if __name__ == "__main__":
    main()
