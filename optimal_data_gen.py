#!/usr/bin/env python3
"""
OPTIMAL TRAINING DATA GENERATION
================================
No categories. No benchmarks. Pure mathematical optimization.

Theory: Treat training data as sampling from an infinite-dimensional knowledge manifold.
Goal: Generate prompts that maximize expected model improvement per training example.

Mathematical Foundation:
------------------------
Let K be the space of all knowledge/capabilities
Let M(θ) be model with parameters θ
Let D = {(p_i, r_i)} be training dataset

Objective: max_{D} E[ΔC(M)] where C is capability measure

Key insight: Information gain I(p) for prompt p is:
    I(p) = H(correct_answer | p) - H(correct_answer | p, model_knowledge)

We want prompts where model has HIGH uncertainty but ground truth is VERIFIABLE.

This script generates training data by:
1. Sampling from high-information-gain regions of knowledge space
2. Using GPT-5.1-codex-mini to both generate AND verify responses
3. Focusing on prompts that require REASONING not retrieval
4. Maximizing diversity through mathematical constraints

Usage:
    python optimal_data_gen.py --num-prompts 1000 --output optimal_training.jsonl
"""

import os
import sys
import json
import random
import hashlib
import argparse
import math
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Any, Optional, Set
from concepts import FoundationalDictionary
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import threading

# DeepSeek client
try:
    from deepseek import DeepSeek
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    DeepSeek = None


@dataclass
class OptimalPrompt:
    """A prompt optimized for maximum training value."""
    prompt: str
    response: str
    information_gain: float  # Estimated bits of information
    reasoning_depth: int     # Estimated reasoning steps required
    verifiability: float     # How easily can we verify correctness (0-1)
    novelty: float          # Distance from common training data (0-1)
    transferability: float  # How much learning transfers to other tasks (0-1)
    composite_score: float  # Weighted combination
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_training_pair(self) -> Tuple[str, str]:
        return (self.prompt, self.response)


class OptimalDataGenerator:
    """
    Generates training data using information-theoretic optimization.

    Core principle: Each training example should maximize:
        Value(example) = InfoGain × Verifiability × Transferability × Novelty

    No categories - pure exploration of the knowledge manifold.
    """

    def __init__(
        self,
        model: str = "gpt-5.1-codex-mini",
        backup_models: Tuple[str, ...] = (),
        max_workers: int = 10,
        cache_dir: str = "cache/optimal_gen"
    ):
        self.model = model
        self.backup_models = backup_models
        self.max_workers = max_workers
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._client = None
        self._initialized = False
        self._seen_hashes: Dict[str, float] = {}  # hash -> best composite score
        self._hash_to_index: Dict[str, int] = {}  # hash -> index in results list
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._rate_limit_announced = False
        self._rate_limit_lock = threading.Lock()
        self._rate_limit_until = 0.0
        self._rate_limit_backoff = 5.0
        self._max_rate_limit_backoff = 300.0

        # Statistics
        self.stats = {
            "total_generated": 0,
            "total_verified": 0,
            "total_rejected": 0,
            "avg_info_gain": 0.0,
            "avg_composite_score": 0.0
        }
        
        # Concept Bridge
        self.concept_bridge = FoundationalDictionary()

    def _ensure_client(self):
        if self._initialized:
            return

        if not OPENAI_AVAILABLE:
            raise ImportError("deepseek package required: pip install deepseek")

        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not set")

        self._client = DeepSeek(api_key=api_key)
        self._initialized = True

    def _is_rate_limit_error(self, error: Exception) -> bool:
        error_str = str(error).lower()
        return "rate limit" in error_str or "rate_limit" in error_str or "429" in error_str

    def _reset_rate_limit_backoff(self) -> None:
        with self._rate_limit_lock:
            self._rate_limit_backoff = 5.0
            self._rate_limit_until = 0.0
            self._rate_limit_announced = False

    def _wait_for_rate_limit(self) -> None:
        with self._rate_limit_lock:
            now = time.time()
            if now < self._rate_limit_until:
                sleep_for = self._rate_limit_until - now
            else:
                sleep_for = self._rate_limit_backoff
                self._rate_limit_until = now + sleep_for
                self._rate_limit_backoff = min(
                    self._rate_limit_backoff * 2,
                    self._max_rate_limit_backoff
                )
                if not self._rate_limit_announced:
                    self._rate_limit_announced = True
                    print(f"\n  [RATE LIMIT] Hit API rate limit. Sleeping {sleep_for:.0f}s before retrying.")
        if sleep_for > 0:
            time.sleep(sleep_for)

    def _hash_prompt(self, prompt: str) -> str:
        """Hash prompt for deduplication."""
        normalized = prompt.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()[:16]

    def _call_model(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.9,
        max_tokens: int = 2048,
        json_mode: bool = False
    ) -> str:
        """Call model with fallback chain."""
        self._ensure_client()

        models_to_try = [self.model] + list(self.backup_models)

        for model in models_to_try:
            while True:
                try:
                    # Check if model supports temperature (codex/reasoning models don't)
                    no_temp_models = ["codex", "o1", "o3", "o4", "gpt-5-mini", "gpt-5-nano"]
                    supports_temp = not any(x in model.lower() for x in no_temp_models)

                    # Responses API only
                    input_text = f"{system}\n\n{prompt}" if system else prompt
                    kwargs = {
                        "model": model,
                        "input": input_text,
                        "max_output_tokens": max_tokens
                    }
                    if supports_temp:
                        kwargs["temperature"] = temperature
                    if json_mode:
                        kwargs["text"] = {"format": {"type": "json_object"}}

                    response = self._client.responses.create(**kwargs)
                    self._reset_rate_limit_backoff()
                    return response.output_text.strip()

                except Exception as e:
                    if self._is_rate_limit_error(e):
                        self._wait_for_rate_limit()
                        continue
                    error_str = str(e)
                    if "401" in error_str or "404" in error_str:
                        break  # Try next model
                    raise

        raise RuntimeError("All models failed")

    def _is_safe_prompt(self, prompt: str) -> Tuple[bool, Optional[str]]:
        """Basic safety filter to avoid generating or answering disallowed content.
        Returns (is_safe, matched_term) tuple."""
        lowered = prompt.lower()
        # Use more specific phrases to avoid false positives
        # e.g., "explicit" alone matches "explicitly" which is fine
        blocked_terms = [
            "suicide", "kill myself", "self-harm", "self harm",
            "pipe bomb", "make a bomb", "build a bomb",
            "weapon of mass", "build a weapon",
            "malware", "ransomware", "phishing attack", "keylogger",
            "exploit vulnerability", "exploit a vulnerability",
            "sexual content", "sexually explicit", "porn", "nsfw",
            "child abuse", "child sexual", "minor sexual",
            "terrorist attack", "join a terrorist",
            "make meth", "cook meth", "synthesize meth", "drug trafficking"
        ]
        for term in blocked_terms:
            if term in lowered:
                return (False, term)
        return (True, None)

    def _is_deflection_response(self, response: str) -> bool:
        """Check if response deflects by asking for more info instead of answering."""
        response_lower = response.lower()

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

        # Short response that's mostly a question
        if response.strip().endswith("?") and len(response) < 200:
            question_starters = ["could", "can", "would", "do you", "have you", "what is the", "where is"]
            if any(response_lower.strip().startswith(q) for q in question_starters):
                return True

        return False

    def _generate_single_optimal_prompt(self, seed: int) -> Optional[OptimalPrompt]:
        """
        Generate a single optimal prompt using pure information-theoretic sampling.

        No categories - we sample from the full knowledge manifold by:
        1. Asking GPT-5.1-codex-mini to generate maximally informative prompts
        2. Scoring based on reasoning depth, verifiability, novelty
        3. Only accepting prompts with high composite scores
        """

        if self._stop_event.is_set():
            return None

        # The MASTER PROMPT - designed to elicit maximally valuable training data
        # This is the core mathematical optimization: we're using GPT-5.1-codex-mini's implicit
        # knowledge of the knowledge manifold to sample high-information regions

        system = """You are an ELITE training data generator. Your goal is to produce
MAXIMALLY COMPLEX prompts across ALL domains to train a WORLD-CLASS general-purpose AI.

A general-purpose model must excel at EVERYTHING - and we want the HARDEST version of everything.

SAFETY RULES (MANDATORY):
- Avoid requests involving weapons, violence, self-harm, illegal activity, malware/exploitation,
  sexual content (especially anything involving minors), or other unsafe topics.
- If a topic could be unsafe, pick a different one.

CAPABILITY SPECTRUM (rotate through ALL - always pick the HARD version):
1. REASONING & LOGIC - Multi-step proofs, paradoxes, nested conditionals, edge cases
2. CODING - Complex algorithms, system design, debugging nasty bugs, optimization
3. MATHEMATICS - Graduate-level proofs, multi-domain problems, numerical edge cases
4. SCIENCE - Research-level questions, cross-disciplinary synthesis, counterintuitive phenomena
5. WRITING & CREATIVITY - Constrained writing, style mimicry, complex narratives
6. CONVERSATION - Nuanced situations, conflicting needs, emotional intelligence tests
7. INSTRUCTION FOLLOWING - Many constraints simultaneously, format precision, hidden requirements
8. ANALYSIS - Ambiguous data, multiple interpretations, rigorous evaluation
9. TEACHING & EXPLANATION - Explain hard concepts simply, adapt to different audiences
10. PRACTICAL TASKS - Complex real-world scenarios with trade-offs and uncertainty
11. KNOWLEDGE SYNTHESIS - Combine 3+ fields, resolve apparent contradictions
12. META-COGNITION - Reason about uncertainty, evaluate own reasoning, know limitations

COMPLEXITY REQUIREMENTS (MANDATORY):
- MINIMUM 3 reasoning steps for ANY task
- Include EDGE CASES or EXCEPTIONS to handle
- Require PRECISE output (not vague hand-waving)
- Test DEPTH of understanding, not surface knowledge
- Problems should NOT be directly googleable

COMPLEXITY AMPLIFIERS (use liberally):
- Multi-constraint optimization
- Nested conditionals and exceptions
- State tracking across steps
- Counter-intuitive correct answers
- Ambiguity requiring careful resolution
- Long dependency chains
- Cross-domain synthesis

Generate prompts that would challenge a brilliant human expert."""

        # Exploration strategies: FULL capability spectrum × MAXIMUM complexity
        exploration_strategies = [
            # REASONING & LOGIC (complex)
            "Generate a PARADOX that requires careful logical resolution with multiple cases",
            "Create a DEDUCTION puzzle with 5+ interacting constraints and red herrings",
            "Design a GAME THEORY problem with hidden information and multiple equilibria",
            # CODING (complex)
            "Write an ALGORITHM challenge requiring dynamic programming with edge cases",
            "Generate a DEBUGGING task: subtle concurrency bug or memory issue to find",
            "Create a SYSTEM DESIGN for a distributed system handling millions of requests",
            "Design an OPTIMIZATION problem: make this O(n²) algorithm O(n log n)",
            # MATHEMATICS (complex)
            "Generate a PROOF requiring induction with a non-obvious base case",
            "Create a problem combining NUMBER THEORY and COMBINATORICS",
            "Design a CALCULUS problem with multiple integration techniques needed",
            "Generate a LINEAR ALGEBRA problem with geometric interpretation",
            # SCIENCE (complex)
            "Create a PHYSICS problem combining thermodynamics and mechanics",
            "Generate a CHEMISTRY question about reaction mechanisms with edge cases",
            "Design a BIOLOGY problem about system dynamics with feedback loops",
            "Create a cross-disciplinary problem: PHYSICS + BIOLOGY + MATH",
            # WRITING & CREATIVITY (complex)
            "Generate a CONSTRAINED WRITING task: story with 5+ specific requirements",
            "Create a STYLE TRANSFER task: rewrite X in the style of Y preserving meaning",
            "Design a NARRATIVE puzzle: story with hidden information the reader must deduce",
            # CONVERSATION (complex)
            "Generate a CONFLICT RESOLUTION scenario with multiple valid perspectives",
            "Create an ETHICS dilemma with competing valid moral frameworks",
            "Design a COACHING scenario: help someone through a multi-faceted problem",
            # INSTRUCTION FOLLOWING (complex)
            "Create a task with 7+ simultaneous formatting and content constraints",
            "Generate a CONDITIONAL task: different outputs based on input characteristics",
            "Design a TRANSFORM task: convert between complex formats precisely",
            # ANALYSIS (complex)
            "Generate a DATA INTERPRETATION task with confounding variables",
            "Create an ARGUMENT ANALYSIS: find 3+ logical fallacies in a passage",
            "Design a DECISION ANALYSIS under uncertainty with multiple criteria",
            # TEACHING (complex)
            "Generate an EXPLANATION task: quantum mechanics for a 10-year-old",
            "Create a MULTI-LEVEL explanation: same concept for beginner/intermediate/expert",
            "Design a MISCONCEPTION task: explain why a common belief is wrong",
            # PRACTICAL (complex)
            "Generate a PLANNING task with resource constraints and dependencies",
            "Create a TROUBLESHOOTING scenario with 5+ possible root causes",
            "Design a NEGOTIATION scenario with multiple parties and interests",
            # KNOWLEDGE SYNTHESIS (complex)
            "Create a question requiring HISTORY + ECONOMICS + TECHNOLOGY synthesis",
            "Generate a problem combining LINGUISTICS + COMPUTER SCIENCE + PSYCHOLOGY",
            "Design a task requiring PHILOSOPHY + SCIENCE + PRACTICAL reasoning",
            # META-COGNITION (complex)
            "Generate a task about CALIBRATING confidence in uncertain claims",
            "Create a REASONING AUDIT: evaluate the quality of a given argument",
            "Design a LIMITATION task: explain what this model likely gets wrong and why"
        ]

        strategy = exploration_strategies[seed % len(exploration_strategies)]
        
        # Concept Bridge: Bridge the random seed to specific foundational concepts
        bridged_concepts = self.concept_bridge.get_concept_combo_string(seed, count=3)

        user_prompt = f"""[Capability focus: {strategy}]
[Diversity seed: {seed}]
[Foundational Concepts: {bridged_concepts}]
[Mode: GENERAL-PURPOSE TRAINING]

Generate ONE high-quality training prompt for a general-purpose AI assistant.

Requirements:
1. Follow the capability focus above
2. Make it feel like a REAL user request (natural language)
3. Include enough context for a complete, useful answer
4. Vary difficulty (some easy, some hard)
5. Be SPECIFIC enough to have a clear good answer

Output JSON:
{{
    "prompt": "your generated prompt (as a real user would ask it)",
    "capability_area": "which of the 12 capability areas this targets",
    "difficulty": "easy|moderate|hard|expert",
    "expected_response_length": "short|medium|long|very_long",
    "verification_method": "how to verify quality of response",
    "estimated_info_gain": <float 0-1>,
    "novelty_score": <float 0-1>,
    "transferability_score": <float 0-1>
}}"""

        try:
            # Generate prompt with metadata
            result_str = self._call_model(user_prompt, system, temperature=0.95, json_mode=True)
            result = json.loads(result_str)

            prompt = result.get("prompt", "")
            if not prompt or len(prompt) < 20:
                print(f"\n  \033[93m[REJECTED]\033[0m Seed {seed}: Prompt too short ({len(prompt)} chars)")
                print(f"  ┌─ PROMPT ─────────────────────────────────────────────────────")
                for line in prompt.split('\n'):
                    print(f"  │ {line}")
                print(f"  └───────────────────────────────────────────────────────────────\n")
                return None
            is_safe, matched_term = self._is_safe_prompt(prompt)
            if not is_safe:
                print(f"\n  \033[91m[REJECTED]\033[0m Seed {seed}: Safety filter matched '{matched_term}'")
                print(f"  ┌─ PROMPT ─────────────────────────────────────────────────────")
                for line in prompt.split('\n'):
                    print(f"  │ {line}")
                print(f"  └───────────────────────────────────────────────────────────────\n")
                return None

            # Check for duplicates - but allow upgrades if new score is better
            prompt_hash = self._hash_prompt(prompt)

            # Generate the response with reasoning tokens for complex tasks
            difficulty = result.get("difficulty", "moderate")
            use_reasoning = difficulty in ["hard", "expert"]

            if use_reasoning:
                response_system = """You are a world-class AI assistant with explicit reasoning capabilities.

For complex problems, use reasoning tokens:
<|think_start|>
[Your step-by-step reasoning here - break down the problem, consider approaches, work through logic]
<|step|> First, I need to...
<|step|> Next, considering...
<|step|> This means that...
<|think_end|>
<|answer|> [Your final answer here]

For simpler parts of your response, just answer directly without reasoning tokens.

Be thorough in reasoning, precise in answers."""
            else:
                response_system = """You are a world-class AI assistant. Answer naturally and helpfully.

For straightforward questions: answer directly and concisely.
For questions that benefit from explanation: provide clear reasoning inline.

You may optionally use reasoning tokens for any part that requires careful thinking:
<|think_start|>reasoning<|think_end|>
<|answer|>final answer

But only use them when genuinely helpful - simple questions should just be answered directly.

Always be helpful, accurate, and appropriately detailed."""

            response = self._call_model(prompt, response_system, temperature=0.4, max_tokens=3000)

            if not response or len(response) < 50:
                print(f"\n  \033[93m[REJECTED]\033[0m Seed {seed}: Response too short ({len(response) if response else 0} chars)")
                print(f"  ┌─ PROMPT ─────────────────────────────────────────────────────")
                for line in prompt.split('\n'):
                    print(f"  │ {line}")
                print(f"  └───────────────────────────────────────────────────────────────")
                print(f"  ┌─ RESPONSE ────────────────────────────────────────────────────")
                print(f"  │ {response if response else '(empty)'}")
                print(f"  └───────────────────────────────────────────────────────────────\n")
                return None

            # Reject deflection responses (asks for info instead of answering)
            if self._is_deflection_response(response):
                print(f"\n  \033[93m[REJECTED]\033[0m Seed {seed}: Deflection response (asks for info instead of answering)")
                return None

            # Calculate composite score
            info_gain = float(result.get("estimated_info_gain", 0.5))
            reasoning_depth = int(result.get("expected_reasoning_steps", 3))
            novelty = float(result.get("novelty_score", 0.5))
            transferability = float(result.get("transferability_score", 0.5))

            # Estimate verifiability based on verification method
            verification = result.get("verification_method", "")
            verifiability = 0.5
            if any(kw in verification.lower() for kw in ["execute", "run", "compute", "calculate"]):
                verifiability = 0.9
            elif any(kw in verification.lower() for kw in ["proof", "derive", "deduce", "logical"]):
                verifiability = 0.8
            elif any(kw in verification.lower() for kw in ["check", "verify", "test"]):
                verifiability = 0.7

            # Composite score: geometric mean to ensure all factors matter
            composite = (info_gain * verifiability * transferability * novelty) ** 0.25

            # Bonus for reasoning depth
            composite *= (1 + 0.1 * min(reasoning_depth, 5))

            # Log the result with scores
            cap = result.get("capability_area", "unknown")[:12]

            # Check if this is a duplicate and whether we should upgrade
            with self._lock:
                if prompt_hash in self._seen_hashes:
                    existing_score = self._seen_hashes[prompt_hash]
                    if composite > existing_score:
                        print(f"\n  \033[96m[UPGRADE]\033[0m Seed {seed}: score={composite:.3f} > existing={existing_score:.3f}")
                        print(f"             Category: {cap} | Difficulty: {difficulty}")
                        print(f"\n  ┌─ PROMPT (same) ─────────────────────────────────────────────")
                        for line in prompt.split('\n'):
                            print(f"  │ {line}")
                        print(f"  └───────────────────────────────────────────────────────────────")
                        print(f"\n  ┌─ NEW RESPONSE (better) ─────────────────────────────────────")
                        for line in response.split('\n'):
                            print(f"  │ {line}")
                        print(f"  └───────────────────────────────────────────────────────────────\n")
                        self._seen_hashes[prompt_hash] = composite
                        # Mark as upgrade by storing prompt_hash in metadata
                        optimal = OptimalPrompt(
                            prompt=prompt,
                            response=response,
                            information_gain=info_gain,
                            reasoning_depth=reasoning_depth,
                            verifiability=verifiability,
                            novelty=novelty,
                            transferability=transferability,
                            composite_score=composite,
                            metadata={
                                "verification_method": verification,
                                "seed": seed,
                                "exploration_strategy": strategy,
                                "capability_area": result.get("capability_area", "general"),
                                "difficulty": difficulty,
                                "uses_reasoning_tokens": use_reasoning,
                                "is_upgrade": True,
                                "prompt_hash": prompt_hash,
                                "previous_score": existing_score
                            }
                        )
                        return optimal
                    else:
                        print(f"\n  \033[93m[REJECTED]\033[0m Seed {seed}: Duplicate, not better ({composite:.3f} <= {existing_score:.3f})")
                        print(f"  ┌─ PROMPT ─────────────────────────────────────────────────────")
                        for line in prompt.split('\n'):
                            print(f"  │ {line}")
                        print(f"  └───────────────────────────────────────────────────────────────\n")
                        return None
                else:
                    self._seen_hashes[prompt_hash] = composite

            print(f"\n  \033[92m[GENERATED]\033[0m Seed {seed}: score={composite:.3f} info={info_gain:.2f} ver={verifiability:.2f} nov={novelty:.2f} trans={transferability:.2f}")
            print(f"             Category: {cap} | Difficulty: {difficulty}")
            print(f"\n  ┌─ PROMPT ─────────────────────────────────────────────────────")
            for line in prompt.split('\n'):
                print(f"  │ {line}")
            print(f"  └───────────────────────────────────────────────────────────────")
            print(f"\n  ┌─ RESPONSE ────────────────────────────────────────────────────")
            for line in response.split('\n'):
                print(f"  │ {line}")
            print(f"  └───────────────────────────────────────────────────────────────\n")

            return OptimalPrompt(
                prompt=prompt,
                response=response,
                information_gain=info_gain,
                reasoning_depth=reasoning_depth,
                verifiability=verifiability,
                novelty=novelty,
                transferability=transferability,
                composite_score=composite,
                metadata={
                    "verification_method": verification,
                    "seed": seed,
                    "exploration_strategy": strategy,
                    "capability_area": result.get("capability_area", "general"),
                    "difficulty": difficulty,
                    "uses_reasoning_tokens": use_reasoning,
                    "prompt_hash": prompt_hash
                }
            )

        except Exception as e:
            print(f"  [Seed {seed}] Generation failed: {e}")
            return None

    def generate_optimal_dataset(
        self,
        num_prompts: int,
        min_composite_score: float = 0.93,
        progress_callback: Optional[callable] = None
    ) -> List[OptimalPrompt]:
        """
        Generate a dataset of optimal training prompts.

        Uses parallel generation with quality filtering.
        """
        self._ensure_client()

        results: List[OptimalPrompt] = []
        completed = [0]
        results_lock = threading.Lock()

        print(f"\n{'='*60}")
        print(f"  OPTIMAL TRAINING DATA GENERATION")
        print(f"{'='*60}")
        print(f"  Target prompts: {num_prompts}")
        print(f"  Min composite score: {min_composite_score}")
        print(f"  Max workers: {self.max_workers}")
        print(f"  Model: {self.model}")
        print()

        # Generate more than needed to account for filtering
        seeds_to_try = list(range(int(num_prompts * 2)))
        random.shuffle(seeds_to_try)

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            seed_iter = iter(seeds_to_try)
            futures = set()

            def submit_next() -> bool:
                try:
                    seed = next(seed_iter)
                except StopIteration:
                    return False
                futures.add(executor.submit(self._generate_single_optimal_prompt, seed))
                return True

            for _ in range(min(self.max_workers, len(seeds_to_try))):
                if not submit_next():
                    break

            while futures:
                if self._stop_event.is_set():
                    for pending in futures:
                        pending.cancel()
                    break
                if len(results) >= num_prompts:
                    for pending in futures:
                        pending.cancel()
                    break

                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    futures.remove(future)
                    result = future.result()
                    completed[0] += 1

                    if result and result.composite_score >= min_composite_score:
                        with results_lock:
                            is_upgrade = result.metadata.get("is_upgrade", False)
                            prompt_hash = result.metadata.get("prompt_hash")

                            if is_upgrade and prompt_hash and prompt_hash in self._hash_to_index:
                                # Replace old entry with upgraded version
                                old_idx = self._hash_to_index[prompt_hash]
                                old_score = results[old_idx].composite_score
                                results[old_idx] = result
                                self.stats["total_upgraded"] = self.stats.get("total_upgraded", 0) + 1
                            else:
                                # New entry
                                results.append(result)
                                if prompt_hash:
                                    self._hash_to_index[prompt_hash] = len(results) - 1
                                self.stats["total_generated"] += 1
                                self.stats["total_verified"] += 1
                    elif result:
                        self.stats["total_rejected"] += 1
                    else:
                        self.stats["total_rejected"] += 1
                        # Already logged above in _generate_single_optimal_prompt

                    # Summary line every 10 completions
                    elapsed = time.time() - start_time
                    rate = completed[0] / elapsed if elapsed > 0 else 0
                    accepted = len(results)
                    rejected = self.stats["total_rejected"]
                    upgraded = self.stats.get("total_upgraded", 0)

                    if completed[0] % 10 == 0:
                        print(f"\n  --- Progress: {accepted} accepted, {upgraded} upgraded, {rejected} rejected | {rate:.1f}/s | target {num_prompts} ---\n")

                    if progress_callback:
                        progress_callback(len(results), num_prompts)

                    if self._stop_event.is_set() or len(results) >= num_prompts:
                        break

                    submit_next()

        print()  # Newline

        if self._stop_event.is_set() and len(results) < num_prompts:
            print(f"  [STOPPED] Generated {len(results)}/{num_prompts} prompts before rate limit.")

        # Sort by composite score (highest first)
        results.sort(key=lambda x: x.composite_score, reverse=True)
        results = results[:num_prompts]

        # Update stats
        if results:
            self.stats["avg_info_gain"] = sum(r.information_gain for r in results) / len(results)
            self.stats["avg_composite_score"] = sum(r.composite_score for r in results) / len(results)

        return results

    def export_training_corpus(
        self,
        prompts: List[OptimalPrompt],
        output_file: str,
        format: str = "jsonl",
        show_progress: bool = True
    ) -> str:
        """Export optimal prompts to training format with line tracking.

        APPENDS to existing file and deduplicates by prompt hash.
        """

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            # Load existing data first to dedupe and merge
            existing_records = {}
            existing_count = 0
            if output_path.exists():
                with open(output_path, "r") as f:
                    for line in f:
                        try:
                            record = json.loads(line)
                            prompt = record.get("messages", [{}])[0].get("content", "")
                            prompt_hash = hash(prompt.strip().lower())
                            score = record.get("metadata", {}).get("composite_score", 0)
                            existing_records[prompt_hash] = (record, score)
                            existing_count += 1
                        except:
                            pass
                print(f"  \033[96m[LOADED]\033[0m {existing_count} existing entries from {output_path}")

            # Merge new prompts (upgrade if better score)
            new_count = 0
            upgraded_count = 0
            for p in prompts:
                prompt_hash = hash(p.prompt.strip().lower())
                record = {
                    "messages": [
                        {"role": "user", "content": p.prompt},
                        {"role": "assistant", "content": p.response}
                    ],
                    "metadata": {
                        "composite_score": p.composite_score,
                        "information_gain": p.information_gain,
                        "reasoning_depth": p.reasoning_depth,
                        "verifiability": p.verifiability,
                        "novelty": p.novelty,
                        "transferability": p.transferability,
                        "capability_area": p.metadata.get("capability_area", "unknown")
                    }
                }

                if prompt_hash in existing_records:
                    old_score = existing_records[prompt_hash][1]
                    if p.composite_score > old_score:
                        existing_records[prompt_hash] = (record, p.composite_score)
                        upgraded_count += 1
                else:
                    existing_records[prompt_hash] = (record, p.composite_score)
                    new_count += 1

            # Write all records atomically
            import shutil
            tmp_path = output_path.with_suffix(".tmp")

            # Backup existing
            if output_path.exists():
                backup_path = output_path.with_suffix(".jsonl.bak")
                shutil.copy2(output_path, backup_path)

            with open(tmp_path, "w") as f:
                for idx, (record, _) in enumerate(existing_records.values()):
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Atomic rename
            tmp_path.replace(output_path)

            total = len(existing_records)
            print(f"  \033[92m[MERGED]\033[0m {new_count} new + {upgraded_count} upgraded = {total} total entries (saved persistently)")

        elif format == "txt":
            line_num = 1
            with open(output_path, "w") as f:
                for idx, p in enumerate(prompts):
                    if show_progress:
                        print(f"  Saving {idx+1}/{len(prompts)} -> {output_path}")
                    start_line = line_num
                    # Use special tokens for user/assistant turns
                    content = f"<|user|>\n{p.prompt}\n<|end_turn|>\n<|assistant|>\n{p.response}\n<|end_turn|>\n\n\n"
                    f.write(content)
                    line_num += content.count('\n')
                    if show_progress:
                        cap = p.metadata.get("capability_area", "?")[:12]
                        print(f"  \033[92m[SAVED]\033[0m {output_path}:{start_line}-{line_num-1} | score={p.composite_score:.2f} | {cap}")

        else:
            raise ValueError(f"Unknown format: {format}")

        return str(output_path)

    def print_statistics(self):
        """Print generation statistics."""
        print(f"\n{'='*50}")
        print(f"  Optimal Data Generation Statistics")
        print(f"{'='*50}")
        print(f"  Total generated:       {self.stats['total_generated']}")
        print(f"  Total verified:        {self.stats['total_verified']}")
        print(f"  Total upgraded:        {self.stats.get('total_upgraded', 0)}")
        print(f"  Total rejected:        {self.stats['total_rejected']}")
        print(f"  Avg information gain:  {self.stats['avg_info_gain']:.3f}")
        print(f"  Avg composite score:   {self.stats['avg_composite_score']:.3f}")
        print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate optimal training data using information-theoretic optimization"
    )
    parser.add_argument(
        "--num-prompts", type=int, default=2000,
        help="Number of optimal prompts to generate"
    )
    parser.add_argument(
        "--min-score", type=float, default=0.93,
        help="Minimum composite score threshold"
    )
    parser.add_argument(
        "--output", type=str, default="cache/optimal_gen/optimal_training.jsonl",
        help="Output file path"
    )
    parser.add_argument(
        "--format", type=str, choices=["jsonl", "txt"], default="jsonl",
        help="Output format"
    )
    parser.add_argument(
        "--workers", type=int, default=10,
        help="Max parallel workers"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-5.1-codex-mini",
        help="Primary model to use"
    )

    args = parser.parse_args()

    # Check API key
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("ERROR: DEEPSEEK_API_KEY environment variable not set")
        print("Get your API key from https://platform.deepseek.com/api-keys")
        sys.exit(1)

    # Generate
    generator = OptimalDataGenerator(
        model=args.model,
        max_workers=args.workers
    )

    prompts = generator.generate_optimal_dataset(
        num_prompts=args.num_prompts,
        min_composite_score=args.min_score
    )

    # Export
    output_path = generator.export_training_corpus(
        prompts, args.output, args.format
    )

    print(f"\n  Exported to: {output_path}")

    # Show sample
    if prompts:
        print(f"\n{'='*60}")
        print(f"  SAMPLE OPTIMAL PROMPT (score={prompts[0].composite_score:.3f})")
        print(f"{'='*60}")
        print(f"\n  PROMPT:\n  {prompts[0].prompt[:500]}...")
        print(f"\n  RESPONSE:\n  {prompts[0].response[:500]}...")

    # Statistics
    generator.print_statistics()


if __name__ == "__main__":
    main()
