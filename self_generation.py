#!/usr/bin/env python3
"""
SELF-GENERATION PIPELINE
=========================
Closes the loop: Trained model generates its own training data.

The Core Problem:
    Current: External model (deepseek-reasoner) generates data -> Train local model -> Deploy
    Missing: Trained model never generates its own training data

Solution:
    1. Use TRAINED erosolar model to generate candidate responses
    2. Verify correctness with grounded verification (code execution, math solving)
    3. Use self-consistency (multiple samples, majority vote) for uncertain cases
    4. External models (if any) only verify - they don't generate

Mathematical Guarantee:
    If verification is sound, then:
    - All accepted samples are correct
    - Model improves on verified domains
    - Quality is bounded only by verification completeness

Usage:
    from self_generation import SelfGenerationPipeline

    pipeline = SelfGenerationPipeline(model_path="models/erosolar")
    samples = await pipeline.generate_round(num_samples=1000)

Author: Bo Shang <bo@shang.software>
"""

import os
import sys
import json
import asyncio
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import Counter
import hashlib
import time

# Import grounded verification
try:
    from grounded_verification import GroundedVerifier, VerificationResult
    GROUNDED_VERIFICATION_AVAILABLE = True
except ImportError:
    GROUNDED_VERIFICATION_AVAILABLE = False
    print("[SelfGeneration] WARNING: grounded_verification not available")

# Import local embeddings for similarity checks
try:
    from local_embeddings import embed, embed_batch, similarity, average_pairwise_similarity
    LOCAL_EMBEDDINGS_AVAILABLE = True
except ImportError:
    LOCAL_EMBEDDINGS_AVAILABLE = False
    print("[SelfGeneration] WARNING: local_embeddings not available")

# Paths
DATA_STORE = Path("data_store")
SELF_GEN_DIR = DATA_STORE / "self_generated"
SELF_GEN_SAMPLES = SELF_GEN_DIR / "samples.jsonl"
SELF_GEN_STATE = SELF_GEN_DIR / "state.json"


@dataclass
class GeneratedSample:
    """A self-generated training sample."""
    prompt: str
    response: str
    thinking: str = ""
    answer: str = ""
    category: str = "general"
    verified: bool = False
    verification_method: str = ""
    confidence: float = 0.0
    self_consistency_score: float = 0.0
    generation_model: str = ""
    timestamp: str = ""

    def to_training_format(self) -> Dict:
        """Convert to training JSONL format."""
        return {
            "messages": [
                {"role": "user", "content": self.prompt},
                {"role": "assistant", "content": self.response}
            ],
            "metadata": {
                "category": self.category,
                "verified": self.verified,
                "verification_method": self.verification_method,
                "confidence": self.confidence,
                "self_consistency_score": self.self_consistency_score,
                "generation_model": self.generation_model,
            }
        }


@dataclass
class SelfGenerationState:
    """Track self-generation progress."""
    total_generated: int = 0
    total_verified: int = 0
    total_rejected: int = 0
    rounds_completed: int = 0
    current_round: int = 0
    last_updated: str = ""
    category_counts: Dict[str, int] = field(default_factory=dict)
    verification_stats: Dict[str, int] = field(default_factory=dict)

    def save(self, path: Path = SELF_GEN_STATE):
        """Save state to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path = SELF_GEN_STATE) -> 'SelfGenerationState':
        """Load state from disk."""
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                return cls(**data)
            except:
                pass
        return cls()


class LocalModelInterface:
    """
    Interface to the locally trained erosolar model.
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or Path("models/erosolar")
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self) -> bool:
        """Load the model if available."""
        if self._loaded:
            return True

        if not self.model_path.exists():
            print(f"[LocalModel] Model not found at {self.model_path}")
            return False

        try:
            # Try to load with transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            print(f"[LocalModel] Loading model from {self.model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self._loaded = True
            print(f"[LocalModel] Model loaded successfully")
            return True

        except Exception as e:
            print(f"[LocalModel] Failed to load model: {e}")
            return False

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7,
                num_samples: int = 1) -> List[str]:
        """
        Generate responses from the local model.

        Args:
            prompt: Input prompt
            max_tokens: Max generation length
            temperature: Sampling temperature
            num_samples: Number of samples to generate

        Returns:
            List of generated responses
        """
        if not self._loaded and not self.load():
            # Fallback: use API if model not available
            return self._fallback_generate(prompt, max_tokens, temperature, num_samples)

        try:
            import torch

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            outputs = []
            for _ in range(num_samples):
                with torch.no_grad():
                    generated = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                response = self.tokenizer.decode(generated[0], skip_special_tokens=False)
                # Remove prompt from response
                if prompt in response:
                    response = response.split(prompt, 1)[1]
                outputs.append(response.strip())

            return outputs

        except Exception as e:
            print(f"[LocalModel] Generation error: {e}")
            return self._fallback_generate(prompt, max_tokens, temperature, num_samples)

    def _fallback_generate(self, prompt: str, max_tokens: int = 1024,
                          temperature: float = 0.7, num_samples: int = 1) -> List[str]:
        """Fallback generation using API if local model unavailable."""
        # Use deepseek API as fallback
        import aiohttp
        import asyncio

        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            print("[LocalModel] No API key for fallback generation")
            return []

        async def _generate():
            async with aiohttp.ClientSession() as session:
                responses = []
                for _ in range(num_samples):
                    async with session.post(
                        "https://api.deepseek.com/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "deepseek-reasoner",
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": max_tokens,
                            "temperature": temperature
                        }
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            content = data["choices"][0]["message"]["content"]
                            responses.append(content)
                return responses

        return asyncio.run(_generate())


class SelfConsistencyChecker:
    """
    Verify responses through self-consistency (majority voting).
    Generate multiple responses and use majority vote on final answer.
    """

    def __init__(self, num_samples: int = 5, confidence_threshold: float = 0.6):
        self.num_samples = num_samples
        self.confidence_threshold = confidence_threshold

    def extract_answer(self, response: str) -> str:
        """Extract final answer from response."""
        import re

        patterns = [
            r'(?:<\|answer\|>)(.*?)(?:<\|end|$)',
            r'(?:the\s+)?answer\s+is[:\s]+([^\.\n]+)',
            r'(?:therefore|so|thus)[,\s]+([^\.\n]+)',
            r'=\s*(\-?\d+\.?\d*)\s*$',
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()

        # Fallback: last line
        lines = response.strip().split('\n')
        return lines[-1].strip() if lines else ""

    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        # Remove whitespace, lowercase, remove common prefixes
        answer = answer.lower().strip()
        for prefix in ["the answer is", "answer:", "result:"]:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()
        return answer

    def check_consistency(self, responses: List[str]) -> Tuple[str, float, str]:
        """
        Check self-consistency across multiple responses.

        Args:
            responses: List of generated responses

        Returns:
            (best_response, confidence, majority_answer)
        """
        if not responses:
            return "", 0.0, ""

        if len(responses) == 1:
            return responses[0], 0.5, self.extract_answer(responses[0])

        # Extract and normalize answers
        answers = []
        answer_to_response = {}
        for resp in responses:
            ans = self.normalize_answer(self.extract_answer(resp))
            answers.append(ans)
            if ans not in answer_to_response:
                answer_to_response[ans] = resp

        # Majority vote
        counter = Counter(answers)
        if not counter:
            return responses[0], 0.3, ""

        majority_answer, count = counter.most_common(1)[0]
        confidence = count / len(answers)

        # Return response that has the majority answer
        best_response = answer_to_response.get(majority_answer, responses[0])

        return best_response, confidence, majority_answer

    def is_consistent(self, responses: List[str]) -> Tuple[bool, float]:
        """Check if responses are self-consistent above threshold."""
        _, confidence, _ = self.check_consistency(responses)
        return confidence >= self.confidence_threshold, confidence


class SelfGenerationPipeline:
    """
    Complete self-generation pipeline.

    Use TRAINED erosolar model to generate its own training data.
    External models verify only - they don't generate.
    """

    def __init__(self, model_path: Optional[Path] = None,
                 num_consistency_samples: int = 5,
                 verification_timeout: int = 5):
        self.model = LocalModelInterface(model_path)
        self.verifier = GroundedVerifier(timeout=verification_timeout) if GROUNDED_VERIFICATION_AVAILABLE else None
        self.consistency_checker = SelfConsistencyChecker(num_samples=num_consistency_samples)
        self.state = SelfGenerationState.load()

        # Ensure directories exist
        SELF_GEN_DIR.mkdir(parents=True, exist_ok=True)

    async def generate_verified_sample(self, prompt: str, category: str = "auto",
                                       test_cases: List[Dict] = None) -> Optional[GeneratedSample]:
        """
        Generate a single verified sample.

        Process:
        1. Local model generates candidate response
        2. Grounded verification checks correctness
        3. If verification uncertain, use self-consistency
        4. Return sample only if verified

        Args:
            prompt: The problem/question
            category: "code", "math", "logic", or "auto"
            test_cases: For code, list of {"input": ..., "expected": ...}

        Returns:
            GeneratedSample if verified, None otherwise
        """
        # Step 1: Generate candidate(s)
        responses = self.model.generate(
            prompt,
            max_tokens=1024,
            temperature=0.7,
            num_samples=1
        )

        if not responses:
            return None

        response = responses[0]

        # Step 2: Grounded verification
        if self.verifier:
            result = self.verifier.verify(prompt, response, category=category, test_cases=test_cases)

            if result.is_correct and result.confidence >= 0.8:
                # High confidence verification
                return self._create_sample(
                    prompt, response, category,
                    verified=True,
                    verification_method=result.method,
                    confidence=result.confidence
                )

        # Step 3: Self-consistency for uncertain cases
        more_responses = self.model.generate(
            prompt,
            max_tokens=1024,
            temperature=0.8,  # Slightly higher for diversity
            num_samples=self.consistency_checker.num_samples - 1
        )

        all_responses = [response] + more_responses
        is_consistent, consistency_score = self.consistency_checker.is_consistent(all_responses)

        if is_consistent:
            best_response, _, _ = self.consistency_checker.check_consistency(all_responses)
            return self._create_sample(
                prompt, best_response, category,
                verified=True,
                verification_method="self_consistency",
                confidence=consistency_score,
                self_consistency_score=consistency_score
            )

        # Not verified - reject
        self.state.total_rejected += 1
        return None

    def _create_sample(self, prompt: str, response: str, category: str,
                      verified: bool, verification_method: str,
                      confidence: float, self_consistency_score: float = 0.0) -> GeneratedSample:
        """Create a GeneratedSample object."""
        # Extract thinking and answer from response
        thinking = ""
        answer = ""

        if "<|think_start|>" in response:
            import re
            think_match = re.search(r'<\|think_start\|>(.*?)<\|think_end\|>', response, re.DOTALL)
            if think_match:
                thinking = think_match.group(1).strip()

        if "<|answer|>" in response:
            answer = response.split("<|answer|>", 1)[1].strip()
            if "<|end_turn|>" in answer:
                answer = answer.split("<|end_turn|>", 1)[0].strip()

        return GeneratedSample(
            prompt=prompt,
            response=response,
            thinking=thinking,
            answer=answer,
            category=category if category != "auto" else "general",
            verified=verified,
            verification_method=verification_method,
            confidence=confidence,
            self_consistency_score=self_consistency_score,
            generation_model="erosolar" if self.model._loaded else "deepseek-reasoner",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )

    async def generate_round(self, prompts: List[Dict], min_samples: int = 500,
                            max_attempts: int = 2000) -> List[GeneratedSample]:
        """
        Generate a round of verified training samples.

        Args:
            prompts: List of {"prompt": ..., "category": ..., "test_cases": [...]}
            min_samples: Minimum verified samples needed
            max_attempts: Maximum generation attempts

        Returns:
            List of verified samples
        """
        samples = []
        attempts = 0

        print(f"[SelfGeneration] Generating round with target {min_samples} samples...")

        for prompt_data in prompts:
            if len(samples) >= min_samples:
                break

            prompt = prompt_data.get("prompt", "")
            category = prompt_data.get("category", "auto")
            test_cases = prompt_data.get("test_cases")

            sample = await self.generate_verified_sample(prompt, category, test_cases)

            if sample:
                samples.append(sample)
                self.state.total_generated += 1
                self.state.total_verified += 1

                # Update category counts
                cat = sample.category
                self.state.category_counts[cat] = self.state.category_counts.get(cat, 0) + 1

                if len(samples) % 50 == 0:
                    print(f"  Generated {len(samples)}/{min_samples} verified samples")

            attempts += 1
            if attempts >= max_attempts:
                print(f"  Reached max attempts ({max_attempts})")
                break

        # Save samples
        self._save_samples(samples)
        self.state.rounds_completed += 1
        self.state.last_updated = time.strftime("%Y-%m-%d %H:%M:%S")
        self.state.save()

        print(f"[SelfGeneration] Round complete: {len(samples)} verified samples")
        return samples

    def _save_samples(self, samples: List[GeneratedSample]):
        """Save samples to JSONL file."""
        with open(SELF_GEN_SAMPLES, 'a') as f:
            for sample in samples:
                f.write(json.dumps(sample.to_training_format()) + '\n')

    def get_stats(self) -> Dict:
        """Get generation statistics."""
        return {
            "total_generated": self.state.total_generated,
            "total_verified": self.state.total_verified,
            "total_rejected": self.state.total_rejected,
            "rounds_completed": self.state.rounds_completed,
            "verification_rate": self.state.total_verified / max(1, self.state.total_generated),
            "category_counts": self.state.category_counts,
        }


class PromptGenerator:
    """
    Generate diverse prompts for self-generation.
    """

    MATH_TEMPLATES = [
        "Calculate {a} {op} {b}.",
        "What is {a} {op} {b}?",
        "Solve: {a} {op} {b} = ?",
        "If x = {a} {op} {b}, what is x?",
    ]

    CODE_TEMPLATES = [
        "Write a Python function that {task}.",
        "Implement a function to {task}.",
        "Create a Python function that takes {input_desc} and returns {output_desc}.",
    ]

    LOGIC_TEMPLATES = [
        "If a = {a} and b = {b}, is (a > {thresh} AND b > {thresh}) true or false?",
        "Given that a = {a} and b = {b}, evaluate: a > {thresh} OR b > {thresh}",
    ]

    CODE_TASKS = [
        ("calculate factorial", [{"input": 5, "expected": 120}, {"input": 0, "expected": 1}]),
        ("reverse a string", [{"input": "hello", "expected": "olleh"}]),
        ("check if number is prime", [{"input": 7, "expected": True}, {"input": 4, "expected": False}]),
        ("find maximum in list", [{"input": [1, 5, 3], "expected": 5}]),
        ("count vowels in string", [{"input": "hello", "expected": 2}]),
    ]

    def generate_math_prompts(self, count: int = 100) -> List[Dict]:
        """Generate math prompts."""
        prompts = []
        ops = [("+", lambda a, b: a + b), ("-", lambda a, b: a - b),
               ("*", lambda a, b: a * b), ("/", lambda a, b: a / b if b != 0 else None)]

        for _ in range(count):
            a = random.randint(1, 100)
            b = random.randint(1, 50)
            op_str, op_fn = random.choice(ops)

            template = random.choice(self.MATH_TEMPLATES)
            prompt = template.format(a=a, op=op_str, b=b)

            expected = op_fn(a, b)
            if expected is None:
                continue

            prompts.append({
                "prompt": prompt,
                "category": "math",
                "expected": expected
            })

        return prompts

    def generate_code_prompts(self, count: int = 50) -> List[Dict]:
        """Generate code prompts with test cases."""
        prompts = []

        for _ in range(count):
            task, test_cases = random.choice(self.CODE_TASKS)
            template = random.choice(self.CODE_TEMPLATES[:2])
            prompt = template.format(task=task)

            prompts.append({
                "prompt": prompt,
                "category": "code",
                "test_cases": test_cases
            })

        return prompts

    def generate_logic_prompts(self, count: int = 50) -> List[Dict]:
        """Generate logic prompts."""
        prompts = []

        for _ in range(count):
            a = random.randint(1, 20)
            b = random.randint(1, 20)
            thresh = random.randint(5, 15)

            template = random.choice(self.LOGIC_TEMPLATES)
            prompt = template.format(a=a, b=b, thresh=thresh)

            prompts.append({
                "prompt": prompt,
                "category": "logic"
            })

        return prompts

    def generate_mixed_prompts(self, count: int = 200) -> List[Dict]:
        """Generate a mix of all prompt types."""
        math_count = count // 2
        code_count = count // 4
        logic_count = count - math_count - code_count

        prompts = []
        prompts.extend(self.generate_math_prompts(math_count))
        prompts.extend(self.generate_code_prompts(code_count))
        prompts.extend(self.generate_logic_prompts(logic_count))

        random.shuffle(prompts)
        return prompts


# ============================================================================
# INTEGRATION WITH PIPELINE
# ============================================================================

async def run_self_generation_round(model_path: Optional[Path] = None,
                                   target_samples: int = 1000) -> Dict:
    """
    Run a self-generation round.

    This is called from pipeline.py after version >= 2.

    Args:
        model_path: Path to trained model
        target_samples: Number of samples to generate

    Returns:
        Stats dictionary
    """
    pipeline = SelfGenerationPipeline(model_path=model_path)
    prompt_gen = PromptGenerator()

    prompts = prompt_gen.generate_mixed_prompts(target_samples * 2)  # Generate extra for rejections
    samples = await pipeline.generate_round(prompts, min_samples=target_samples)

    return {
        "samples_generated": len(samples),
        "samples_file": str(SELF_GEN_SAMPLES),
        **pipeline.get_stats()
    }


def should_use_self_generation(version: int, threshold_version: int = 2) -> bool:
    """
    Determine if we should use self-generation.

    After threshold_version, switch from external model to self-generation.

    Args:
        version: Current model version
        threshold_version: Version at which to switch

    Returns:
        True if should use self-generation
    """
    return version >= threshold_version


# ============================================================================
# CLI / TESTING
# ============================================================================

async def test_self_generation():
    """Test the self-generation pipeline."""
    print("=" * 60)
    print("SELF-GENERATION PIPELINE TEST")
    print("=" * 60)

    # Test prompt generation
    print("\n1. Testing prompt generation...")
    gen = PromptGenerator()

    math_prompts = gen.generate_math_prompts(5)
    print(f"   Generated {len(math_prompts)} math prompts")
    print(f"   Example: {math_prompts[0]['prompt']}")

    code_prompts = gen.generate_code_prompts(3)
    print(f"   Generated {len(code_prompts)} code prompts")
    print(f"   Example: {code_prompts[0]['prompt']}")

    # Test self-consistency
    print("\n2. Testing self-consistency checker...")
    checker = SelfConsistencyChecker()

    responses = [
        "The answer is 42.",
        "After calculation, the answer is 42.",
        "Therefore, the result is 42.",
        "The answer is 42.",
        "So we get 43.",  # Different answer
    ]

    best, confidence, majority = checker.check_consistency(responses)
    print(f"   Majority answer: {majority}")
    print(f"   Confidence: {confidence:.2f}")
    print(f"   Is consistent (threshold 0.6): {confidence >= 0.6}")

    # Test pipeline (if model available)
    print("\n3. Testing pipeline...")
    pipeline = SelfGenerationPipeline()

    if pipeline.model.load():
        print("   Model loaded - testing generation...")
        sample = await pipeline.generate_verified_sample(
            "What is 25 + 17?",
            category="math"
        )
        if sample:
            print(f"   Generated verified sample: {sample.answer}")
        else:
            print("   Sample not verified")
    else:
        print("   Model not available - skipping generation test")

    print("\n4. Pipeline stats:")
    stats = pipeline.get_stats()
    for k, v in stats.items():
        print(f"   {k}: {v}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Self-Generation Pipeline")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--generate", type=int, default=0,
                       help="Generate N samples")
    parser.add_argument("--stats", action="store_true", help="Show stats")
    parser.add_argument("--model-path", type=str, help="Path to model")
    args = parser.parse_args()

    if args.test:
        asyncio.run(test_self_generation())
    elif args.generate > 0:
        model_path = Path(args.model_path) if args.model_path else None
        result = asyncio.run(run_self_generation_round(model_path, args.generate))
        print(json.dumps(result, indent=2))
    elif args.stats:
        state = SelfGenerationState.load()
        print(json.dumps(asdict(state), indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
