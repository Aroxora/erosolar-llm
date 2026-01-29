#!/usr/bin/env python3
"""
Transform Training Data to 100% Chain-of-Thought Format

CRITICAL INSIGHT: Self-attention learns by seeing patterns.
To learn reasoning, it must see: FOUNDATION -> REASONING -> ANSWER

This script:
1. Loads concept foundations (what the model "knows")
2. Transforms Q&A to explicitly reference foundations in thinking
3. Creates training data that teaches reasoning patterns

The chain for self-attention to learn:
  "I know [foundation]" -> "Therefore [reasoning step]" -> "So [answer]"

Usage:
    python transform_to_cot.py                    # Transform all data
    python transform_to_cot.py --check            # Preview transformations
    python transform_to_cot.py --use-api          # High-quality API transforms
"""

import argparse
import json
import os
import asyncio
import aiohttp
import re
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import random

# ANSI Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"

# Thinking tokens (must match tokenizer special tokens)
THINK_START = "<|think_start|>"
THINK_END = "<|think_end|>"
STEP_TOKEN = "<|step|>"
ANSWER_TOKEN = "<|answer|>"

# Rate limiting
MAX_RETRIES = 5
BASE_RETRY_DELAY = 1.5

# Paths
CONCEPT_FOUNDATIONS_PATH = Path("cache/foundations/concept_foundations.json")
FOUNDATIONAL_CONCEPTS_PATH = Path("optional_unverified_concepts/foundational_concepts.json")


class ConceptKnowledgeBase:
    """
    Loads and provides access to concept foundations.
    This is what the model "knows" - used to ground reasoning.
    """

    def __init__(self):
        self.concepts: Dict[str, Dict] = {}
        self.concept_list: List[str] = []
        self._load_foundations()

    def _load_foundations(self):
        """Load concept foundations from JSON files."""
        # Load detailed foundations
        if CONCEPT_FOUNDATIONS_PATH.exists():
            try:
                with open(CONCEPT_FOUNDATIONS_PATH) as f:
                    data = json.load(f)
                for concept, info in data.items():
                    self.concepts[concept.lower()] = {
                        "concept": concept,
                        "definition": info.get("definition", ""),
                        "examples": info.get("examples", []),
                        "category": info.get("category", "concept")
                    }
            except Exception as e:
                print(f"{YELLOW}Warning loading foundations: {e}{RESET}")

        # Load concept list
        if FOUNDATIONAL_CONCEPTS_PATH.exists():
            try:
                with open(FOUNDATIONAL_CONCEPTS_PATH) as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self.concept_list = list(data.keys())
                elif isinstance(data, list):
                    self.concept_list = data
            except Exception:
                pass

        print(f"{DIM}Loaded {len(self.concepts)} concept foundations{RESET}")

    def find_relevant_concepts(self, text: str, max_concepts: int = 3) -> List[Dict]:
        """Find concepts from the knowledge base that are relevant to the text."""
        text_lower = text.lower()
        relevant = []

        for key, info in self.concepts.items():
            # Check if concept appears in text
            if key in text_lower or info.get("concept", "").lower() in text_lower:
                relevant.append(info)
                if len(relevant) >= max_concepts:
                    break

        return relevant

    def get_foundation_for_concept(self, concept: str) -> Optional[str]:
        """Get the foundational definition for a concept."""
        info = self.concepts.get(concept.lower())
        if info and info.get("definition"):
            return info["definition"]
        return None

    def format_as_known_fact(self, concept: str) -> Optional[str]:
        """Format a concept as a 'known fact' for reasoning."""
        definition = self.get_foundation_for_concept(concept)
        if definition:
            return f"I know that {concept}: {definition}"
        return None


def get_api_url() -> str:
    """Get OpenAI Responses API URL."""
    base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    return f"{base.rstrip('/')}/responses"


def parse_response(result: dict) -> str:
    """Parse OpenAI Responses API output."""
    for out in result.get("output", []):
        if out.get("type") == "message":
            for content in out.get("content", []):
                if content.get("type") == "output_text":
                    return content.get("text", "").strip()
    return ""


@dataclass
class CoTExample:
    """A chain-of-thought training example."""
    prompt: str
    thinking: str  # The reasoning steps
    answer: str    # The final answer
    category: str

    def to_response(self) -> str:
        """Format as response with thinking tokens."""
        return f"{THINK_START} {self.thinking} {THINK_END} {ANSWER_TOKEN} {self.answer}"

    def to_dict(self) -> dict:
        return {
            "messages": [
                {"role": "user", "content": self.prompt},
                {"role": "assistant", "content": self.to_response()}
            ],
            "metadata": {
                "category": self.category,
                "has_cot": True,
                "thinking_length": len(self.thinking)
            }
        }


class CoTTransformer:
    """
    Transform Q&A pairs into chain-of-thought format.

    KEY: Uses concept foundations to ground reasoning.
    Pattern: "I know [foundation]" -> "Therefore [step]" -> "So [answer]"
    """

    # Categories that need different transformation strategies
    MATH_PATTERNS = [
        r'\d+\s*[\+\-\*\/]\s*\d+',  # arithmetic
        r'how many', r'how much', r'calculate', r'what is \d',
        r'sum|difference|product|quotient',
    ]

    CODE_PATTERNS = [
        r'write.*function', r'write.*code', r'implement',
        r'python|javascript|java|code|program',
        r'def |class |function|algorithm',
    ]

    FACTUAL_PATTERNS = [
        r'^what is', r'^who is', r'^where is', r'^when',
        r'explain|describe|define',
    ]

    def __init__(self, api_key: str = None, model: str = "gpt-5.1-codex-mini",
                 knowledge_base: ConceptKnowledgeBase = None,
                 max_steps: int = None, max_thinking_chars: int = None):
        self.api_key = api_key
        self.model = model
        self.kb = knowledge_base or ConceptKnowledgeBase()
        self.max_steps = max_steps
        self.max_thinking_chars = max_thinking_chars

    def classify_example(self, prompt: str, response: str) -> str:
        """Classify the type of example for appropriate transformation."""
        prompt_lower = prompt.lower()

        # Check patterns
        for pattern in self.MATH_PATTERNS:
            if re.search(pattern, prompt_lower):
                return "math"

        for pattern in self.CODE_PATTERNS:
            if re.search(pattern, prompt_lower):
                return "code"

        for pattern in self.FACTUAL_PATTERNS:
            if re.search(pattern, prompt_lower):
                return "factual"

        # Check response content
        if "```" in response or "def " in response or "function" in response:
            return "code"

        if any(c.isdigit() for c in response[:50]):
            return "math"

        return "general"

    def transform_static(self, prompt: str, response: str, category: str) -> CoTExample:
        """
        Transform using static rules (no API needed).

        KEY PATTERN for self-attention to learn:
        1. "I know [foundation]" - reference what we know
        2. "Therefore [reasoning]" - apply that knowledge
        3. "[answer]" - produce the result
        """

        # Find relevant concepts from knowledge base
        relevant_concepts = self.kb.find_relevant_concepts(prompt + " " + response)

        # Build foundation-grounded thinking
        thinking_parts = []

        # Add foundation references (what the model "knows")
        for concept_info in relevant_concepts[:2]:
            concept = concept_info.get("concept", "")
            definition = concept_info.get("definition", "")
            if definition and len(definition) > 10:
                # Truncate long definitions
                short_def = definition[:100] + "..." if len(definition) > 100 else definition
                thinking_parts.append(f"I know that {concept}: {short_def}")

        # Add category-specific reasoning
        if category == "math":
            thinking_parts.extend(self._generate_math_thinking_parts(prompt, response))
        elif category == "code":
            thinking_parts.extend(self._generate_code_thinking_parts(prompt, response))
        elif category == "factual":
            thinking_parts.extend(self._generate_factual_thinking_parts(prompt, response))
        else:
            thinking_parts.extend(self._generate_general_thinking_parts(prompt, response))

        # Combine into coherent thinking with explicit steps
        if thinking_parts:
            if self.max_steps:
                thinking_parts = thinking_parts[:self.max_steps]
            thinking = f" {STEP_TOKEN} ".join(thinking_parts)
        else:
            thinking = self._generate_minimal_thinking(prompt)

        if self.max_thinking_chars and len(thinking) > self.max_thinking_chars:
            trimmed = thinking[:self.max_thinking_chars]
            last_space = trimmed.rfind(" ")
            if last_space > 60:
                trimmed = trimmed[:last_space]
            thinking = trimmed.rstrip()

        return CoTExample(
            prompt=prompt,
            thinking=thinking,
            answer=response,
            category=category
        )

    def _generate_minimal_thinking(self, prompt: str) -> str:
        """Generate minimal thinking when no foundations match."""
        return f"Let me think about this. Analyzing the request to provide the best response."

    def _generate_math_thinking_parts(self, prompt: str, response: str) -> List[str]:
        """Generate thinking parts for math problems."""
        parts = []

        # Extract numbers from prompt
        numbers = re.findall(r'\d+', prompt)

        # Look for operation
        if '+' in prompt or 'add' in prompt.lower() or 'sum' in prompt.lower():
            op = "addition"
        elif '-' in prompt or 'subtract' in prompt.lower():
            op = "subtraction"
        elif '*' in prompt or 'x' in prompt or 'multiply' in prompt.lower():
            op = "multiplication"
        elif '/' in prompt or 'divide' in prompt.lower():
            op = "division"
        else:
            op = "calculation"

        parts.append(f"This is a {op} problem.")

        if numbers:
            parts.append(f"Numbers involved: {', '.join(numbers[:3])}.")

        # Add step-by-step reasoning
        if len(numbers) >= 2:
            parts.append(f"Step 1: Identify the operation - {op}.")
            parts.append(f"Step 2: Apply {op} to the numbers.")

            # Extract answer and show reasoning
            answer_match = re.search(r'(\d+)', response)
            if answer_match:
                parts.append(f"Therefore, the result is {answer_match.group(1)}.")

        return parts

    def _generate_code_thinking_parts(self, prompt: str, response: str) -> List[str]:
        """Generate thinking parts for code problems."""
        parts = []

        # Identify what's being asked
        if "function" in prompt.lower():
            parts.append("I need to write a function.")
        elif "class" in prompt.lower():
            parts.append("I need to define a class.")
        elif "fix" in prompt.lower() or "bug" in prompt.lower():
            parts.append("I need to debug/fix existing code.")
        else:
            parts.append("I need to write code to solve this.")

        # Add planning steps
        parts.append("Planning: 1) Understand requirements, 2) Identify inputs/outputs, 3) Implement.")

        # Look for language
        if "python" in prompt.lower() or "def " in response:
            parts.append("Using Python for this solution.")
        elif "javascript" in prompt.lower() or "function(" in response:
            parts.append("Using JavaScript for this solution.")

        parts.append("Therefore, implementing the solution:")

        return parts

    def _generate_factual_thinking_parts(self, prompt: str, response: str) -> List[str]:
        """Generate thinking parts for factual questions."""
        parts = []

        # Identify question type
        if prompt.lower().startswith("what is"):
            topic = prompt[7:].strip().rstrip("?")
            parts.append(f"This is a definition question about: {topic}.")
        elif prompt.lower().startswith("who"):
            parts.append("This is a question about a person or entity.")
        elif prompt.lower().startswith("why"):
            parts.append("This is a question asking for an explanation.")
        elif prompt.lower().startswith("how"):
            parts.append("This is a question about a process or method.")
        else:
            parts.append("Analyzing this question.")

        # Add retrieval/reasoning step
        parts.append("Retrieving relevant knowledge from my foundations.")

        # Extract key concepts to show reasoning
        words = response.split()[:20]
        key_words = [w for w in words if len(w) > 4 and w.isalpha()][:3]
        if key_words:
            parts.append(f"Key concepts involved: {', '.join(key_words)}.")

        parts.append("Therefore, the answer is:")

        return parts

    def _generate_general_thinking_parts(self, prompt: str, response: str) -> List[str]:
        """Generate thinking parts for general responses."""
        parts = []

        # Check if it's a greeting
        greetings = ["hi", "hello", "hey", "good morning", "good afternoon"]
        if any(g in prompt.lower() for g in greetings):
            parts.append("This is a greeting. I should respond warmly and offer help.")

        # Check if it's a thanks
        elif "thank" in prompt.lower():
            parts.append("The user is expressing gratitude. I should acknowledge and offer further help.")

        # Check if it's asking for help
        elif "help" in prompt.lower():
            parts.append("The user needs assistance. I should ask what they need help with.")

        else:
            parts.append("Analyzing the request to provide the best response.")

        parts.append("Therefore, responding with:")

        return parts

    async def transform_with_api(
        self,
        prompt: str,
        response: str,
        category: str,
        session: aiohttp.ClientSession
    ) -> Optional[CoTExample]:
        """Transform using API for high-quality thinking generation."""

        api_prompt = f"""Transform this Q&A pair into chain-of-thought format.

Original Question: {prompt}
Original Answer: {response}

Generate ONLY the thinking/reasoning steps that would lead to this answer.
The thinking should:
1. Break down the problem
2. Show step-by-step reasoning
3. Connect to the final answer

Output ONLY the thinking steps (2-4 sentences), nothing else."""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "input": api_prompt
        }

        for attempt in range(MAX_RETRIES):
            try:
                async with session.post(
                    get_api_url(),
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        thinking = parse_response(result)

                        # Clean up thinking
                        thinking = thinking.strip()
                        if thinking.startswith('"') and thinking.endswith('"'):
                            thinking = thinking[1:-1]
                        if self.max_thinking_chars and len(thinking) > self.max_thinking_chars:
                            trimmed = thinking[:self.max_thinking_chars]
                            last_space = trimmed.rfind(" ")
                            if last_space > 60:
                                trimmed = trimmed[:last_space]
                            thinking = trimmed.rstrip()

                        return CoTExample(
                            prompt=prompt,
                            thinking=thinking,
                            answer=response,
                            category=category
                        )

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

        # Fallback to static transformation
        return self.transform_static(prompt, response, category)


async def transform_all_data(
    use_api: bool = False,
    api_key: str = None,
    model: str = "gpt-5.1-codex-mini",
    workers: int = 10,
    limit: int = None,
    input_file: str = None,
    max_steps: int = None,
    max_thinking_chars: int = None
) -> List[Dict]:
    """Transform all training data to CoT format."""

    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  Chain-of-Thought Data Transformation{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")

    # Load training data - from input file or data.py
    if input_file and Path(input_file).exists():
        # Load from JSONL file (gap-targeted, etc.)
        print(f"{CYAN}Loading from: {input_file}{RESET}")
        all_data = []
        with open(input_file) as f:
            for line in f:
                try:
                    record = json.loads(line)
                    msgs = record.get("messages", [])
                    if len(msgs) >= 2:
                        prompt = msgs[0].get("content", "")
                        response = msgs[1].get("content", "")
                        if prompt and response:
                            all_data.append((prompt, response))
                except:
                    continue
        print(f"{GREEN}Loaded {len(all_data)} examples from file{RESET}")
    else:
        try:
            from data import get_all_training_data
            all_data = get_all_training_data(balanced=True)
            print(f"{GREEN}Loaded {len(all_data)} training examples{RESET}")
        except ImportError as e:
            print(f"{RED}Error loading data: {e}{RESET}")
            return []

    if limit:
        all_data = all_data[:limit]
        print(f"{YELLOW}Limited to {limit} examples{RESET}")

    transformer = CoTTransformer(
        api_key,
        model,
        max_steps=max_steps,
        max_thinking_chars=max_thinking_chars
    )
    results = []

    if use_api and api_key:
        print(f"\n{CYAN}Using API for high-quality transformations...{RESET}")
        print(f"{DIM}Model: {model}{RESET}")

        semaphore = asyncio.Semaphore(workers)

        async def transform_one(prompt: str, response: str) -> Dict:
            async with semaphore:
                category = transformer.classify_example(prompt, response)
                async with aiohttp.ClientSession() as session:
                    example = await transformer.transform_with_api(
                        prompt, response, category, session
                    )
                    return example.to_dict() if example else None

        tasks = [transform_one(p, r) for p, r in all_data]

        completed = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result:
                results.append(result)
            completed += 1
            if completed % 100 == 0:
                print(f"\r{DIM}  Transformed {completed}/{len(all_data)}...{RESET}", end='')

        print()

    else:
        print(f"\n{CYAN}Using static transformation (no API)...{RESET}")

        for i, (prompt, response) in enumerate(all_data):
            category = transformer.classify_example(prompt, response)
            example = transformer.transform_static(prompt, response, category)
            results.append(example.to_dict())

            if (i + 1) % 500 == 0:
                print(f"\r{DIM}  Transformed {i+1}/{len(all_data)}...{RESET}", end='')

        print()

    print(f"\n{GREEN}Transformed {len(results)} examples to CoT format{RESET}")

    # Stats
    categories = {}
    for r in results:
        cat = r.get("metadata", {}).get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    print(f"\n{DIM}Categories:{RESET}")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    return results


def save_cot_data(results: List[Dict], output_path: Path) -> int:
    """Save transformed data to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = output_path.with_name(output_path.name + ".tmp")
    if output_path.exists():
        backup_path = output_path.with_name(output_path.name + ".bak")
        shutil.copy2(output_path, backup_path)
    with open(tmp_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    tmp_path.replace(output_path)

    return len(results)


def update_data_py_imports():
    """Generate code to update data.py to use CoT data."""

    code = '''
# Add to data.py to use CoT-transformed data:

def load_cot_training_data() -> List[Tuple[str, str]]:
    """Load chain-of-thought transformed training data."""
    cot_path = Path("cache/cot/cot_training_data.jsonl")

    if not cot_path.exists():
        print("Warning: CoT data not found, using original data")
        return get_all_training_data()

    pairs = []
    with open(cot_path) as f:
        for line in f:
            record = json.loads(line)
            msgs = record.get("messages", [])
            if len(msgs) >= 2:
                prompt = msgs[0].get("content", "")
                response = msgs[1].get("content", "")
                pairs.append((prompt, response))

    return pairs
'''
    return code


async def main():
    parser = argparse.ArgumentParser(description="Transform training data to chain-of-thought format")
    parser.add_argument("--output", "-o", type=str, default="cache/cot/cot_training_data.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--use-api", action="store_true",
                        help="Use API for high-quality transformations")
    parser.add_argument("--model", "-m", type=str, default="gpt-5.1-codex-mini",
                        help="Model for API transformations")
    parser.add_argument("--workers", "-w", type=int, default=10,
                        help="Concurrent API workers")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of examples to transform")
    parser.add_argument("--check", action="store_true",
                        help="Preview transformations without saving")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Input JSONL file (optional - uses data.py if not specified)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Max thinking steps to include (optional)")
    parser.add_argument("--max-thinking-chars", type=int, default=None,
                        help="Max thinking characters (optional)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY") if args.use_api else None

    if args.use_api and not api_key:
        print(f"{YELLOW}Warning: --use-api specified but OPENAI_API_KEY not set{RESET}")
        print(f"{YELLOW}Falling back to static transformation{RESET}")

    results = await transform_all_data(
        use_api=args.use_api and bool(api_key),
        api_key=api_key,
        model=args.model,
        workers=args.workers,
        limit=args.limit if not args.check else 10,
        input_file=args.input,
        max_steps=args.max_steps,
        max_thinking_chars=args.max_thinking_chars
    )

    if args.check:
        print(f"\n{CYAN}Preview of transformations:{RESET}")
        for i, result in enumerate(results[:5]):
            msgs = result.get("messages", [])
            if len(msgs) >= 2:
                print(f"\n{BOLD}Example {i+1}:{RESET}")
                print(f"{DIM}Q: {msgs[0]['content'][:100]}...{RESET}")
                print(f"{GREEN}A: {msgs[1]['content'][:200]}...{RESET}")
        return

    # Save
    output_path = Path(args.output)
    saved = save_cot_data(results, output_path)

    print(f"\n{GREEN}{'='*60}{RESET}")
    print(f"{GREEN}  Transformation Complete!{RESET}")
    print(f"{GREEN}{'='*60}{RESET}")
    print(f"{DIM}  Saved: {saved} CoT examples{RESET}")
    print(f"{DIM}  Output: {args.output}{RESET}")
    print(f"\n{DIM}  To use in training, update data.py or use --cot flag{RESET}")


if __name__ == "__main__":
    asyncio.run(main())
