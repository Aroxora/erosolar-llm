#!/usr/bin/env python3
"""
Upgrade and Serve - Complete Training Data Generation Pipeline

This script generates the best possible training data for a curriculum learning approach:
1. Phase 1: Foundational Knowledge - Generate training data for all concepts without values
2. Phase 1.5: Bridge Data - Connect foundational knowledge to sophisticated reasoning
3. Phase 2 (Optional): Sophisticated Reasoning Chains - Deep reasoning patterns

Usage:
    python upgrade_and_serve.py                           # Full pipeline
    python upgrade_and_serve.py --skip-reasoning          # Skip Phase 2
    python upgrade_and_serve.py --check-only              # Just report missing
    python upgrade_and_serve.py --model gpt-4o --workers 30

Author: Training Data Pipeline
"""

import argparse
import json
import os
import asyncio
import aiohttp
import re
import random
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, asdict
import time

from bridge_synthetic import generate_synthetic_bridge_pairs

# Rate limiting configuration
MAX_RETRIES = 5
BASE_RETRY_DELAY = 1.5  # seconds
MAX_RETRY_DELAY = 60.0  # seconds
RATE_LIMIT_BUFFER = 0.1  # Add small delay between requests to avoid hitting limits

# Foundation completeness thresholds
MIN_TRAINING_PAIRS_DEFAULT = 5
MIN_TRAINING_PAIRS = MIN_TRAINING_PAIRS_DEFAULT

# ANSI Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"
MAGENTA = "\033[95m"

# Paths
FOUNDATIONAL_CONCEPTS_PATH = Path("optional_unverified_concepts/foundational_concepts.json")
CONCEPT_FOUNDATIONS_PATH = Path("cache/foundations/concept_foundations.json")
FOUNDATIONAL_KNOWLEDGE_PATH = Path("cache/foundations/foundational_knowledge.jsonl")
BRIDGE_DATA_PATH = Path("cache/bridge/bridge_data.jsonl")
REASONING_CHAINS_PATH = Path("cache/reasoning/reasoning_chains.jsonl")


def get_api_url(endpoint: str = "responses") -> str:
    """Get OpenAI API URL."""
    base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    return f"{base.rstrip('/')}/{endpoint}"


def parse_responses_api(result: dict) -> str:
    """Parse OpenAI Responses API output."""
    for out in result.get("output", []):
        if out.get("type") == "message":
            for content in out.get("content", []):
                if content.get("type") == "output_text":
                    return content.get("text", "").strip()
    return ""


def parse_chat_completion(result: dict) -> str:
    """Parse OpenAI Chat Completion API output."""
    try:
        return result["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        return ""


async def api_call_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    headers: dict,
    data: dict,
    timeout: int = 60
) -> Tuple[Optional[dict], Optional[str]]:
    """
    Make API call with exponential backoff retry for rate limits.
    Returns (result_dict, error_string) tuple.
    """
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            # Add small jitter to avoid thundering herd
            if attempt > 0:
                jitter = random.uniform(0, 0.5)
                await asyncio.sleep(jitter)

            async with session.post(
                url,
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                if resp.status == 200:
                    return await resp.json(), None

                elif resp.status == 429:
                    # Rate limited - extract retry delay from response
                    error_body = await resp.text()

                    # Try to parse retry-after from error message
                    retry_after = BASE_RETRY_DELAY * (2 ** attempt)

                    # Look for "Please try again in X.XXXs" pattern
                    import re as regex
                    match = regex.search(r'try again in (\d+\.?\d*)s', error_body)
                    if match:
                        retry_after = float(match.group(1)) + 0.5  # Add buffer

                    retry_after = min(retry_after, MAX_RETRY_DELAY)

                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(retry_after)
                        continue
                    else:
                        last_error = f"Rate limit exceeded after {MAX_RETRIES} retries"

                else:
                    error_text = await resp.text()
                    last_error = f"API error {resp.status}: {error_text[:200]}"

                    # Don't retry on client errors (except rate limits)
                    if 400 <= resp.status < 500 and resp.status != 429:
                        break

        except asyncio.TimeoutError:
            last_error = "Request timeout"
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(BASE_RETRY_DELAY * (2 ** attempt))

        except aiohttp.ClientError as e:
            last_error = f"Connection error: {str(e)}"
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(BASE_RETRY_DELAY * (2 ** attempt))

        except Exception as e:
            last_error = f"Unexpected error: {str(e)}"
            break

    return None, last_error


@dataclass
class ConceptFoundation:
    """Foundational knowledge for a single concept."""
    concept: str
    category: str
    definition: str
    examples: List[str]
    training_pairs: List[Dict]

    def to_dict(self):
        return asdict(self)

    def is_complete(self) -> bool:
        """Check if this foundation has sufficient data."""
        return (
            len(self.definition) > 10 and
            len(self.training_pairs) >= MIN_TRAINING_PAIRS
        )

    def missing_fields(self) -> List[str]:
        """Return a short list of what's missing for completion."""
        missing = []
        if len(self.definition) <= 10:
            missing.append("definition")
        if len(self.training_pairs) < MIN_TRAINING_PAIRS:
            missing.append("training_pairs")
        return missing


class ConceptExtractor:
    """Extract and categorize concepts from training data."""

    STOP_WORDS = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        'this', 'that', 'these', 'those', 'it', 'its', 'if', 'then', 'else',
        'so', 'up', 'out', 'just', 'also', 'very', 'too', 'much', 'more',
        'most', 'other', 'some', 'any', 'no', 'not', 'only', 'same', 'than',
        'such', 'like', 'into', 'over', 'after', 'before', 'between', 'under',
        'through', 'during', 'without', 'within', 'along', 'following',
        'across', 'behind', 'beyond', 'plus', 'except', 'about', 'around',
        'i', 'me', 'my', 'you', 'your', 'he', 'she', 'we', 'they', 'them',
    }

    TECH_TERMS = {
        'python', 'javascript', 'java', 'html', 'css', 'sql', 'git', 'docker',
        'linux', 'windows', 'api', 'json', 'http', 'https', 'url', 'cpu', 'gpu',
        'memory', 'server', 'client', 'database', 'function', 'class', 'method',
        'variable', 'loop', 'array', 'object', 'string', 'integer', 'boolean',
        'list', 'dict', 'tuple', 'set', 'import', 'export', 'module', 'package',
        'kubernetes', 'azure', 'aws', 'cloud', 'network', 'security', 'firewall',
        'encryption', 'algorithm', 'machine', 'learning', 'neural', 'model',
    }

    def __init__(self):
        self.concepts: Dict[str, Dict] = {}

    def _categorize(self, word: str) -> str:
        """Categorize a word/concept."""
        word_lower = word.lower()

        if word_lower.isdigit() or word_lower in {
            'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
            'eight', 'nine', 'ten', 'eleven', 'twelve', 'hundred', 'thousand'
        }:
            return 'number'

        if word_lower in {'hi', 'hello', 'hey', 'goodbye', 'bye', 'thanks', 'please', 'sorry'}:
            return 'greeting'

        if word_lower in {'what', 'who', 'why', 'how', 'when', 'where', 'which'}:
            return 'question_word'

        if word_lower in self.TECH_TERMS:
            return 'tech_term'

        return 'concept'

    def load_from_json(self, path: Path) -> Dict[str, Dict]:
        """Load concepts from JSON list/dict."""
        if not path.exists():
            print(f"{YELLOW}Warning: {path} not found{RESET}")
            return {}

        with open(path) as f:
            concepts_list = json.load(f)

        for concept in concepts_list:
            key = concept.lower()
            self.concepts[key] = {
                'canonical': concept,
                'category': self._categorize(concept),
                'count': 1,
                'source': str(path)
            }

        return self.concepts

    def extract_from_training_data(self) -> Dict[str, Dict]:
        """Extract concepts from training corpus."""
        try:
            from data import get_all_training_data

            print(f"{DIM}Loading training corpus...{RESET}")
            all_data = get_all_training_data(balanced=True)

            print(f"{DIM}Tokenizing {len(all_data):,} training pairs...{RESET}")

            for prompt, response in all_data:
                tokens = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_]*\b', prompt + " " + response)
                for token in tokens:
                    if token.lower() not in self.STOP_WORDS and len(token) >= 3:
                        key = token.lower()
                        if key not in self.concepts:
                            self.concepts[key] = {
                                'canonical': token.title(),
                                'category': self._categorize(token),
                                'count': 0,
                                'source': 'training_data'
                            }
                        self.concepts[key]['count'] += 1

        except ImportError as e:
            print(f"{YELLOW}Warning: Could not import training data: {e}{RESET}")

        return self.concepts

    def get_all_concepts(self) -> List[Tuple[str, str]]:
        """Get all concepts sorted by frequency."""
        sorted_concepts = sorted(
            self.concepts.items(),
            key=lambda x: -x[1].get('count', 0)
        )
        return [(v['canonical'], v['category']) for k, v in sorted_concepts]


class FoundationGenerator:
    """Generate high-quality foundational knowledge for concepts."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()

    def _get_prompt(self, concept: str, category: str) -> str:
        """Get the best prompt for generating concept foundations."""

        if category == 'number':
            return f"""Generate concise foundational knowledge for the number "{concept}".

Provide:
1. DEFINITION: 1-2 sentences, precise and clear
2. EXAMPLES: 2-3 real-world uses of this number
3. TRAINING PAIRS: 5 Q&A pairs, each answer 1-2 sentences:
   Q: [question about {concept}]
   A: [concise, educational answer]

Cover key properties and common uses without filler."""

        elif category == 'greeting':
            return f"""Generate concise foundational knowledge for the greeting/phrase "{concept}".

Provide:
1. DEFINITION: 1-2 sentences on meaning and usage
2. EXAMPLES: 2-3 conversational contexts
3. TRAINING PAIRS: 5 Q&A pairs, each answer 1-2 sentences:
   Q: [using {concept} in conversation or asking about it]
   A: [concise, natural response]

Be conversational but succinct."""

        elif category == 'tech_term':
            return f"""Generate concise foundational knowledge for the technical term "{concept}".

Provide:
1. DEFINITION: 1-2 sentences, precise and technical
2. EXAMPLES: 2-3 practical use cases (brief)
3. TRAINING PAIRS: 5 Q&A pairs, each answer 1-2 sentences:
   Q: [question about {concept}]
   A: [concise, accurate answer]

Cover what it is, why it's used, and a key pattern without filler."""

        else:
            return f"""Generate concise foundational knowledge for the concept "{concept}".

Provide:
1. DEFINITION: 1-2 sentences, clear and educational
2. EXAMPLES: 2-3 real-world examples or use cases
3. TRAINING PAIRS: 5 Q&A pairs, each answer 1-2 sentences:
   Q: [question about {concept}]
   A: [informative, concise answer]

Be broad in coverage but short in form."""

    async def generate_for_concept(
        self,
        concept: str,
        category: str,
        semaphore: asyncio.Semaphore,
        strict: bool = False
    ) -> Optional[ConceptFoundation]:
        """Generate foundation for a single concept with retry logic."""

        async with semaphore:
            # Add small delay to spread out requests
            await asyncio.sleep(RATE_LIMIT_BUFFER)

            system_prompt = "You are an expert educator creating high-quality training data. Be thorough, accurate, and educational."
            user_prompt = self._get_prompt(concept, category)
            if strict:
                user_prompt += (
                    "\n\nReturn ONLY in this exact format:\n"
                    "DEFINITION: <2-3 sentences>\n"
                    "EXAMPLES:\n"
                    "- <example 1>\n"
                    "- <example 2>\n"
                    "- <example 3>\n"
                    "TRAINING PAIRS:\n"
                    "Q1: <question>\n"
                    "A1: <answer>\n"
                    "Q2: <question>\n"
                    "A2: <answer>\n"
                    "Q3: <question>\n"
                    "A3: <answer>\n"
                    "Q4: <question>\n"
                    "A4: <answer>\n"
                    "Q5: <question>\n"
                    "A5: <answer>\n"
                    "Do not omit any Q/A pairs.\n"
                )

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            # Use OpenAI Responses API format with explicit roles
            data = {
                "model": self.model,
                "input": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }

            result, error = await api_call_with_retry(
                self.session,
                get_api_url("responses"),
                headers,
                data,
                timeout=60
            )

            if result:
                response_text = parse_responses_api(result)
                return self._parse_foundation(concept, category, response_text)
            else:
                # Only print errors for non-rate-limit issues (rate limits are retried)
                if error and "rate limit" not in error.lower():
                    print(f"\n{RED}  Error for {concept}: {error}{RESET}")
                return None

    def _parse_foundation(self, concept: str, category: str, text: str) -> ConceptFoundation:
        """Parse API response into structured foundation."""

        # Extract definition
        definition = ""
        lines = text.split('\n')
        in_definition = False

        for i, line in enumerate(lines):
            if 'DEFINITION' in line.upper() or 'definition' in line.lower():
                in_definition = True
                # Check if definition is on same line
                parts = line.split(':', 1)
                if len(parts) > 1 and len(parts[1].strip()) > 10:
                    definition = parts[1].strip()
                    break
                continue
            elif in_definition and line.strip() and not any(x in line.upper() for x in ['EXAMPLE', 'Q:', 'TRAINING']):
                definition = line.strip().lstrip('- ').lstrip('* ')
                break

        # Fallback: first substantial line
        if not definition:
            for line in lines:
                clean = line.strip()
                if len(clean) > 30 and not clean.startswith(('Q:', 'A:', '#', '*', '-', '1.', '2.')):
                    definition = clean
                    break

        # Extract examples
        examples = []
        in_examples = False
        for line in lines:
            if 'EXAMPLE' in line.upper():
                in_examples = True
                continue
            if in_examples and line.strip().startswith(('-', '*', '1.', '2.', '3.')):
                example = re.sub(r'^[-*\d.]+\s*', '', line.strip())
                if len(example) > 10:
                    examples.append(example)
                if len(examples) >= 3:
                    break
            if 'TRAINING' in line.upper() or 'Q:' in line:
                break

        # Extract Q&A pairs (robust to Q1:, Question 1:, bullets, numbering, etc.)
        pairs = []
        current_q = None

        qa_text = re.sub(r'^[\s>*•-]+', '', text, flags=re.MULTILINE)
        qa_text = re.sub(r'^[ \t]*\d+[\.\)]\s*', '', qa_text, flags=re.MULTILINE)
        qa_text = qa_text.replace("**", "").replace("__", "")
        qa_lines = qa_text.splitlines()

        qa_q_re = re.compile(r'^(Q|Question)\s*[\d\.\)]*\s*[:\.\)]\s*', re.IGNORECASE)
        qa_a_re = re.compile(r'^(A|Answer)\s*[\d\.\)]*\s*[:\.\)]\s*', re.IGNORECASE)

        for line in qa_lines:
            line = line.strip()
            if qa_q_re.match(line):
                current_q = qa_q_re.sub('', line).strip()
            elif qa_a_re.match(line) and current_q:
                answer = qa_a_re.sub('', line).strip()
                if current_q and answer:
                    pairs.append({"user": current_q, "assistant": answer})
                current_q = None

        inline_re = re.compile(
            r'(?:^|\n)\s*(?:Q|Question)\s*[\d\.\)]*\s*[:\.\)]\s*(.+?)\s*'
            r'(?:A|Answer)\s*[\d\.\)]*\s*[:\.\)]\s*(.+?)(?=\n|$)',
            re.IGNORECASE | re.DOTALL
        )
        for q, a in inline_re.findall(qa_text):
            q_clean = " ".join(q.strip().split())
            a_clean = " ".join(a.strip().split())
            if q_clean and a_clean:
                pairs.append({"user": q_clean, "assistant": a_clean})

        block_re = re.compile(
            r'(?:^|\n)\s*(?:Q|Question)\s*[\d\.\)]*\s*[:\.\)]\s*(.+?)\n\s*'
            r'(?:A|Answer)\s*[\d\.\)]*\s*[:\.\)]\s*(.+?)(?=\n\s*(?:Q|Question)\s*[\d\.\)]*\s*[:\.\)]|\Z)',
            re.IGNORECASE | re.DOTALL
        )
        for q, a in block_re.findall(qa_text):
            q_clean = " ".join(q.strip().split())
            a_clean = " ".join(a.strip().split())
            if q_clean and a_clean:
                pairs.append({"user": q_clean, "assistant": a_clean})

        seen_q = set()
        unique_pairs = []
        for pair in pairs:
            q = pair.get("user", "").strip()
            a = pair.get("assistant", "").strip()
            key = q.lower()
            if not q or not a or key in seen_q:
                continue
            seen_q.add(key)
            unique_pairs.append({"user": q, "assistant": a})
        pairs = unique_pairs

        def add_pair(question: str, answer: str) -> None:
            q = question.strip()
            a = answer.strip()
            if not q or not a:
                return
            key = q.lower()
            if key in seen_q:
                return
            seen_q.add(key)
            pairs.append({"user": q, "assistant": a})

        # Fallback: synthesize Q&A from definition/examples to reach minimum pairs
        if len(pairs) < MIN_TRAINING_PAIRS:
            for ex in examples:
                if len(pairs) >= MIN_TRAINING_PAIRS:
                    break
                add_pair(f"Give an example of {concept}.", ex)

            if definition:
                add_pair(f"What is {concept}?", definition)

                brief = definition
                if len(brief) > 200:
                    brief = brief[:200].rsplit(" ", 1)[0]
                add_pair(f"Why is {concept} important?", brief)

                if examples:
                    add_pair(f"Where is {concept} used?", examples[0])
                add_pair(f"What is a key property of {concept}?", brief)
                add_pair(f"Explain {concept} in one sentence.", brief)

        return ConceptFoundation(
            concept=concept,
            category=category,
            definition=definition,
            examples=examples,
            training_pairs=pairs
        )

    async def generate_all(
        self,
        concepts: List[Tuple[str, str]],
        workers: int = 20,
        on_result=None,
        on_progress=None,
        retry_incomplete: int = 1
    ) -> Dict[str, ConceptFoundation]:
        """Generate foundations for all concepts."""

        semaphore = asyncio.Semaphore(workers)
        foundations = {}

        print(f"\n{CYAN}Generating foundations for {len(concepts)} concepts...{RESET}")
        print(f"{DIM}Model: {self.model}{RESET}")

        async def run_one(concept: str, category: str, strict_mode: bool) -> Tuple[str, str, Optional[ConceptFoundation]]:
            result = await self.generate_for_concept(concept, category, semaphore, strict=strict_mode)
            return concept, category, result

        async def run_pass(
            pending: List[Tuple[str, str]],
            strict: bool
        ) -> Tuple[List[Tuple[str, str]], Dict[str, ConceptFoundation]]:
            tasks = [
                asyncio.create_task(run_one(concept, category, strict))
                for concept, category in pending
            ]
            completed = 0
            total = len(pending)
            still_incomplete: List[Tuple[str, str]] = []
            incomplete_results: Dict[str, ConceptFoundation] = {}

            for task in asyncio.as_completed(tasks):
                concept, category, result = await task
                completed += 1
                if result and result.is_complete():
                    foundations[result.concept] = result
                    if on_result:
                        on_result(result, completed, total)
                    else:
                        print(f"{DIM}  [{completed}/{total}] {result.concept}: {len(result.training_pairs)} pairs{RESET}")
                    if on_progress:
                        on_progress(concept, "complete", completed, total, result)
                else:
                    if result:
                        incomplete_results[concept.lower()] = result
                    still_incomplete.append((concept, category))
                    status = "incomplete" if result else "failed"
                    if on_progress:
                        on_progress(concept, status, completed, total, result)

            return still_incomplete, incomplete_results

        remaining = concepts
        strict = False
        last_incomplete_results: Dict[str, ConceptFoundation] = {}
        for attempt in range(retry_incomplete + 1):
            remaining, last_incomplete_results = await run_pass(remaining, strict=strict)
            if not remaining:
                break
            if attempt < retry_incomplete:
                print(f"\n{YELLOW}Retrying {len(remaining)} incomplete concepts with strict format (attempt {attempt + 2}/{retry_incomplete + 1})...{RESET}")
                strict = True
            else:
                print(f"\n{YELLOW}Still incomplete after retries: {len(remaining)}{RESET}")
                for concept, category in remaining[:20]:
                    foundation = last_incomplete_results.get(concept.lower())
                    if foundation:
                        missing = ", ".join(foundation.missing_fields()) or "unknown"
                        print(f"{YELLOW}  - {concept} ({category}) missing {missing}{RESET}")
                    else:
                        print(f"{YELLOW}  - {concept} ({category}) missing training_pairs/definition{RESET}")

        return foundations


# ============================================================================
# BRIDGE DATA GENERATION (Phase 1.5)
# ============================================================================

@dataclass
class BridgeConcept:
    """Bridge concept connecting foundational to sophisticated knowledge."""
    name: str
    description: str
    prompts: List[str]
    system_prompt: str


BRIDGE_CATEGORIES = {
    "multi_step_math": BridgeConcept(
        name="multi_step_math",
        description="Multi-step arithmetic building on foundational facts",
        prompts=[
            "We know 2+2=4. Now calculate 2+2+5",
            "Building on 5+5=10, what is 5+5+5+5?",
            "If I have 5 apples and buy 3 more, then eat 2, how many do I have?",
            "Using 3*3=9, calculate 3*3+3*3",
            "Foundation: 10-3=7. Now what is 10-3-3?",
            "We know 20/4=5. What is 20/4+20/4?",
            "Building on addition: 2+2+3+3+4+4=?",
            "If x=5 (variable stores 5), what is x+x+x?",
            "Chain: Start with 10, subtract 3, add 5. What's the result?",
            "Using 100/10=10, calculate 100/10+100/10",
            "Foundation: half of 10 is 5. What is half of 10 plus half of 10?",
            "Build from 6+6=12: what is 6+6+6?",
            "We know 4*5=20. What is 4*5-4?",
            "Chain calculation: 2*2=4, 4*2=8, 8*2=?",
            "Using 15/3=5, what is 15/3+15/3+15/3?",
        ],
        system_prompt="""You are bridging foundational math to multi-step reasoning.

IMPORTANT: Each response must:
1. Reference the foundational fact being used
2. Show step-by-step reasoning that builds on it
3. State the final answer clearly

Format:
Foundation: [state the basic fact]
Step 1: [apply foundation]
Step 2: [next step]
Answer: [final result]

Keep responses under 80 words. Show the BRIDGE from simple to complex."""
    ),

    "word_problems": BridgeConcept(
        name="word_problems",
        description="Simple word problems requiring comprehension and math",
        prompts=[
            "A book costs $5. How much do 3 books cost?",
            "Sara has 12 stickers. She gives 4 to Tom. How many does Sara have now?",
            "There are 8 birds on a tree. 3 fly away. How many are left?",
            "Mom bought 2 boxes of cookies. Each box has 6 cookies. How many cookies total?",
            "A train has 4 cars. Each car has 10 seats. How many seats total?",
            "I walked 2 miles on Monday and 3 miles on Tuesday. How far did I walk?",
            "There are 15 students. They form 3 equal groups. How many in each group?",
            "A pizza has 8 slices. We ate half. How many slices are left?",
            "John is 8 years old. His sister is 3 years older. How old is his sister?",
            "A bag has 20 marbles. 5 are red, 5 are blue. How many are other colors?",
        ],
        system_prompt="""You are helping a student with word problems. For each:
1. Identify what we know
2. Show the calculation
3. State the answer with units
Keep responses under 60 words. Build logical reasoning skills."""
    ),

    "simple_patterns": BridgeConcept(
        name="simple_patterns",
        description="Pattern recognition and sequence completion",
        prompts=[
            "What comes next: 2, 4, 6, 8, ?",
            "Complete the pattern: A, B, A, B, A, ?",
            "What's missing: 1, 2, _, 4, 5",
            "Continue: red, blue, red, blue, ?",
            "Next number: 5, 10, 15, 20, ?",
            "Pattern: 1, 1, 2, 2, 3, 3, ?",
            "What comes after: Mon, Tue, Wed, ?",
            "Complete: 10, 20, 30, ?, 50",
            "Sequence: 2, 4, 8, 16, ?",
            "Fill in: cat, dog, cat, dog, ?",
            "Next: 100, 90, 80, 70, ?",
            "Pattern: ABC, DEF, GHI, ?",
        ],
        system_prompt="""You recognize patterns. For each:
1. Identify the pattern rule
2. Apply it to find the next element
3. State the answer clearly
Keep responses under 50 words. Focus on pattern reasoning."""
    ),

    "simple_coding": BridgeConcept(
        name="simple_coding",
        description="Basic coding concepts and simple functions",
        prompts=[
            "Write a function to add two numbers in Python",
            "How do I print 'Hello World' in Python?",
            "What does a for loop do?",
            "Write code to check if a number is even",
            "How do I create a list in Python?",
            "What is a variable?",
            "Write a function that returns the bigger of two numbers",
            "How do I comment code in Python?",
            "What does 'return' do in a function?",
            "Write code to print numbers 1 to 5",
            "How do I check if two strings are equal?",
            "What is an if statement?",
            "Write a function to double a number",
            "How do I get the length of a string?",
            "What is indentation in Python?",
        ],
        system_prompt="""You are a beginner-friendly coding tutor. For each:
1. Explain the concept simply
2. Provide a small working code example
3. Keep code under 5 lines when possible
Build from basic programming understanding. Be concise."""
    ),

    "cause_effect": BridgeConcept(
        name="cause_effect",
        description="Simple cause and effect reasoning",
        prompts=[
            "What happens when water freezes?",
            "Why do plants need sunlight?",
            "What happens if you don't sleep enough?",
            "Why does ice melt in the sun?",
            "What causes rain?",
            "Why do we wear coats in winter?",
            "What happens when you mix red and blue paint?",
            "Why do birds fly south in winter?",
            "What causes a shadow?",
            "Why do we brush our teeth?",
            "What happens when a battery dies?",
            "Why does metal feel cold?",
        ],
        system_prompt="""Explain cause and effect relationships simply.
1. State the cause clearly
2. Explain the effect
3. Keep the connection logical and simple
Responses under 60 words. Build reasoning from basic knowledge."""
    ),

    "comparisons": BridgeConcept(
        name="comparisons",
        description="Comparing and contrasting concepts",
        prompts=[
            "What's the difference between a cat and a dog?",
            "How are addition and multiplication related?",
            "Compare day and night",
            "What's similar about circles and squares?",
            "Difference between hot and cold",
            "How is reading different from writing?",
            "Compare a bicycle and a car",
            "What do summer and winter have in common?",
            "Difference between a question and an answer",
            "How are letters and numbers different?",
            "Compare walking and running",
            "What's the difference between a list and a dictionary in Python?",
        ],
        system_prompt="""Compare and contrast concepts clearly.
1. State key similarities
2. State key differences
3. Keep comparisons balanced and factual
Under 70 words. Build analytical thinking from foundational concepts."""
    ),

    "simple_instructions": BridgeConcept(
        name="simple_instructions",
        description="Following and giving simple multi-step instructions",
        prompts=[
            "How do I make a peanut butter sandwich?",
            "What are the steps to tie a shoe?",
            "How do I send an email?",
            "Steps to wash your hands properly",
            "How do I save a file on a computer?",
            "What are the steps to plant a seed?",
            "How do I draw a simple house?",
            "Steps to make a cup of tea",
            "How do I count from 1 to 10 in steps of 2?",
            "What are the steps to cross the street safely?",
        ],
        system_prompt="""Provide clear step-by-step instructions.
1. Number each step
2. Keep steps simple and sequential
3. Use action words (first, then, next, finally)
Under 100 words. Build procedural thinking."""
    ),

    "context_bridging": BridgeConcept(
        name="context_bridging",
        description="Questions that require connecting context across the prompt",
        prompts=[
            "The cat is on the mat. The mat is red. What color is the surface under the cat?",
            "John has a ball. The ball is blue. Mary wants John's toy. What color is what Mary wants?",
            "2+2=4 and 4+4=8. What is 2+2+4+4?",
            "Roses are red. Violets are blue. What color are roses?",
            "My name is Alex. I am an AI. What is my name?",
            "Today is Monday. Tomorrow is the day after today. What day is tomorrow?",
            "The box contains 5 apples. We add 3 more. The box now has how many apples?",
            "Python uses indentation. Indentation means spaces at the start of a line. What does Python use for code structure?",
        ],
        system_prompt="""Answer questions that require connecting information.
1. Identify all relevant facts
2. Connect them logically
3. State the answer clearly
Focus on demonstrating context tracking. Under 50 words."""
    ),

    "building_concepts": BridgeConcept(
        name="building_concepts",
        description="Concepts that build on foundational knowledge",
        prompts=[
            "We know 2+2=4. What is 4+4?",
            "A cat is an animal. A dog is an animal. What are cats and dogs?",
            "One hour has 60 minutes. How many minutes in 2 hours?",
            "If 5+5=10 and 10+10=20, what is 5+5+5+5?",
            "Water freezes to ice. Ice melts to water. What happens to ice when heated?",
            "Variables store values. x=5 stores 5 in x. What does x+x equal?",
            "Squares have 4 sides. Rectangles also have 4 sides. How many sides do both shapes have?",
            "print() displays output. What does print('hi') display?",
        ],
        system_prompt="""Build on established foundational knowledge.
1. Reference the foundational fact
2. Apply it to answer the question
3. Show the logical connection
Under 60 words. Demonstrate knowledge building."""
    ),
}


def generate_static_bridge_pairs() -> List[Dict]:
    """Generate static bridge pairs that explicitly connect foundational to sophisticated."""

    pairs = []

    # Math bridges
    math_bridges = [
        ("We know 2+2=4. What is 2+2+2?",
         "Building on 2+2=4:\nStep 1: 2+2 = 4 (foundational fact)\nStep 2: 4+2 = 6\n\nThe answer is 6."),
        ("Since 2+2=4 and 3+3=6, what is 2+2+3+3?",
         "Using foundational facts:\n- 2+2 = 4\n- 3+3 = 6\nStep: 4 + 6 = 10\n\nThe answer is 10."),
        ("We know 5+5=10. What is 5+5+5?",
         "Building on 5+5=10:\nStep 1: 5+5 = 10 (foundational)\nStep 2: 10+5 = 15\n\nThe answer is 15."),
        ("If 5+5=10, what is 10+10?",
         "From foundation: 5+5=10\nApplying the pattern: 10+10=20\n\nThe answer is 20."),
        ("Using that 10-3=7, calculate 10-3-2",
         "Starting with foundation: 10-3=7\nThen: 7-2=5\n\nThe answer is 5."),
        ("Knowing 3*3=9, what is 3*3+1?",
         "Foundation: 3*3=9\nBridge step: 9+1=10\n\nThe answer is 10."),
        ("Calculate step by step: 2+2+3+3",
         "Using foundations:\nStep 1: 2+2=4 (known fact)\nStep 2: 3+3=6 (known fact)\nStep 3: 4+6=10\n\nThe answer is 10."),
        ("What is (2+2) * (5-3)?",
         "Foundation facts:\n- 2+2=4\n- 5-3=2\nBridge: 4*2=8\n\nThe answer is 8."),
    ]

    # Coding bridges
    coding_bridges = [
        ("We know print() displays output. How do we print a variable x?",
         "Foundation: print() displays output\nBridge: To print a variable, pass it to print()\n\n```python\nx = 5\nprint(x)  # Displays: 5\n```"),
        ("Variables store values. How do we use two variables together?",
         "Foundation: Variables store values\nBridge: Combine variables with operators\n\n```python\na = 3\nb = 4\nresult = a + b  # result stores 7\nprint(result)\n```"),
        ("We know 'def' creates a function. Create a function that uses print()",
         "Foundations:\n- def creates a function\n- print() displays output\n\nBridge - combining them:\n```python\ndef say_hello():\n    print(\"Hello!\")\n\nsay_hello()  # Displays: Hello!\n```"),
        ("Lists use []. How do we print each item in a list?",
         "Foundations:\n- Lists use []\n- print() displays output\n- for loops iterate\n\nBridge:\n```python\nfruits = [\"apple\", \"banana\"]\nfor fruit in fruits:\n    print(fruit)\n```"),
        ("For loops iterate. How do we use one with range()?",
         "Foundation: for loops iterate over items\nBridge: range() creates a sequence of numbers\n\n```python\nfor i in range(3):\n    print(i)  # Prints 0, 1, 2\n```"),
        ("Using variables, a function, and print, create an add function",
         "Combining foundations:\n- Variables store values\n- def creates functions\n- print() displays output\n\n```python\ndef add(a, b):\n    result = a + b\n    return result\n\nprint(add(2, 3))  # Displays: 5\n```"),
    ]

    # Knowledge bridges
    knowledge_bridges = [
        ("We know water freezes at 0C. What happens at -5C?",
         "Foundation: Water freezes at 0C\nReasoning: -5C is below freezing point\nConclusion: Water will be frozen (ice) at -5C."),
        ("Plants need sunlight. What happens to a plant in a dark room?",
         "Foundation: Plants need sunlight to make food (photosynthesis)\nReasoning: Dark room = no sunlight\nConclusion: The plant will weaken and eventually die without light."),
        ("There are 8 planets. If we visited 3, how many are left to visit?",
         "Foundation: 8 planets total\nCalculation: 8 - 3 = 5\nAnswer: 5 planets left to visit."),
        ("Cats and dogs are mammals. Mammals have fur. Do cats have fur?",
         "Foundation: Cats are mammals\nFact: Mammals have fur\nLogical conclusion: Yes, cats have fur."),
        ("Water freezes at 0C and boils at 100C. What state is water at 50C?",
         "Foundations:\n- Freezes at 0C (solid below)\n- Boils at 100C (gas above)\n\nReasoning: 50C is between 0 and 100\nAnswer: Water is liquid at 50C."),
    ]

    # Context bridges
    context_bridges = [
        ("John has 5 apples. Mary gives him 3 more. John now has how many?",
         "Context tracking:\n- Start: John has 5 apples\n- Event: +3 apples from Mary\n- Result: 5 + 3 = 8 apples"),
        ("x = 10. We subtract 3. Then add 5. What is x now?",
         "Tracking x:\n- Start: x = 10\n- Step 1: x = 10 - 3 = 7\n- Step 2: x = 7 + 5 = 12\n\nx is now 12."),
        ("The list is [1, 2]. We append 3. We append 4. What's in the list?",
         "Tracking the list:\n- Start: [1, 2]\n- append(3): [1, 2, 3]\n- append(4): [1, 2, 3, 4]\n\nThe list contains [1, 2, 3, 4]."),
        ("Alice is a programmer. She writes Python. What language does Alice use?",
         "Context: Alice is a programmer who writes Python.\n'She' refers to Alice.\nAnswer: Alice uses Python."),
        ("The function returns 5. We store it in result. What is result?",
         "Context tracking:\n- Function returns 5\n- Stored in 'result'\nAnswer: result is 5."),
    ]

    # Pattern bridges
    pattern_bridges = [
        ("In 2, 4, 6, 8 each number is 2 more. What comes next?",
         "Pattern identified: +2 each step\nSequence: 2, 4, 6, 8\nNext: 8 + 2 = 10\n\nThe answer is 10."),
        ("1, 2, 4, 8 doubles each time. What's the pattern and next number?",
         "Pattern: Each number is doubled (x2)\nSequence: 1, 2, 4, 8\nNext: 8 x 2 = 16\n\nThe answer is 16."),
        ("Mon, Tue, Wed follows the week. What comes after Wed?",
         "Pattern: Days of the week in order\nAfter Wednesday comes Thursday.\n\nThe answer is Thursday."),
        ("In ABC, DEF, GHI each group advances 3 letters. What's next?",
         "Pattern: Each letter group advances by 3 positions\nA->D->G->J, B->E->H->K, C->F->I->L\nNext group: JKL"),
    ]

    # Reasoning bridges
    reasoning_bridges = [
        ("If x > 5 and x = 7, is the condition true?",
         "Given: x = 7\nCondition: x > 5\nCheck: Is 7 > 5? Yes.\n\nThe condition is true."),
        ("All birds have wings. A sparrow is a bird. Does a sparrow have wings?",
         "Premise 1: All birds have wings\nPremise 2: A sparrow is a bird\nConclusion: Therefore, a sparrow has wings.\n\nYes, a sparrow has wings."),
        ("If it rains, the ground gets wet. It is raining. What can we conclude?",
         "Rule: Rain -> wet ground\nFact: It is raining\nConclusion: The ground is wet (or will get wet)."),
        ("A function returns True if n > 0. What does it return for n = 5?",
         "Rule: Return True if n > 0\nInput: n = 5\nCheck: 5 > 0 is True\nReturn: True"),
    ]

    # Format all pairs
    categories = [
        ("math_bridge", math_bridges),
        ("coding_bridge", coding_bridges),
        ("knowledge_bridge", knowledge_bridges),
        ("context_bridge", context_bridges),
        ("pattern_bridge", pattern_bridges),
        ("reasoning_bridge", reasoning_bridges),
    ]

    for cat_name, cat_pairs in categories:
        for p, r in cat_pairs:
            pairs.append({
                "messages": [
                    {"role": "user", "content": p},
                    {"role": "assistant", "content": r}
                ],
                "metadata": {
                    "category": cat_name,
                    "type": "bridge_knowledge",
                    "bridges_from": "foundational",
                    "bridges_to": "sophisticated"
                }
            })

    return pairs


async def generate_bridge_data(
    api_key: str,
    model: str,
    workers: int = 20,
    bridge_target: int = 0,
    bridge_seed: int = 42
) -> List[Dict]:
    """Generate bridge training data."""

    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{CYAN}  Phase 1.5: Bridge Data Generation{RESET}")
    print(f"{CYAN}{'='*60}{RESET}")

    all_pairs = []
    if bridge_target:
        print(f"{DIM}  Bridge target: {bridge_target} pairs (seed: {bridge_seed}){RESET}")

    # Add static pairs (no API needed)
    static_pairs = generate_static_bridge_pairs()
    all_pairs.extend(static_pairs)
    print(f"{GREEN}Added {len(static_pairs)} static bridge pairs{RESET}")

    # Generate API pairs
    semaphore = asyncio.Semaphore(workers)

    async def generate_pair(prompt: str, category: BridgeConcept) -> Optional[Dict]:
        async with semaphore:
            # Add small delay to spread out requests
            await asyncio.sleep(RATE_LIMIT_BUFFER)

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            # Use OpenAI Responses API format
            data = {
                "model": model,
                "input": f"{category.system_prompt}\n\n{prompt}"
            }

            async with aiohttp.ClientSession() as session:
                result, error = await api_call_with_retry(
                    session,
                    get_api_url("responses"),
                    headers,
                    data,
                    timeout=30
                )

                if result:
                    response = parse_responses_api(result)
                    return {
                        "messages": [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": response}
                        ],
                        "metadata": {
                            "category": category.name,
                            "type": "bridge_knowledge"
                        }
                    }
            return None

    # Generate for each category
    for cat_name, category in BRIDGE_CATEGORIES.items():
        print(f"\n{DIM}Generating {cat_name}: {len(category.prompts)} prompts{RESET}")

        tasks = [generate_pair(p, category) for p in category.prompts]
        results = await asyncio.gather(*tasks)

        valid = [r for r in results if r]
        all_pairs.extend(valid)
        print(f"{GREEN}  Generated {len(valid)}/{len(category.prompts)} pairs{RESET}")

    if bridge_target and len(all_pairs) < bridge_target:
        needed = bridge_target - len(all_pairs)
        synthetic_pairs = generate_synthetic_bridge_pairs(needed, seed=bridge_seed)
        all_pairs.extend(synthetic_pairs)
        print(f"{GREEN}Added {len(synthetic_pairs)} synthetic bridge pairs (target {bridge_target}){RESET}")
    elif bridge_target:
        print(f"{DIM}Bridge target met: {len(all_pairs)}/{bridge_target}{RESET}")

    return all_pairs


# ============================================================================
# SOPHISTICATED REASONING CHAINS (Phase 2 - Optional)
# ============================================================================

REASONING_PROMPTS = [
    # Multi-step logical reasoning
    {
        "prompt": "If all mammals are warm-blooded, and all whales are mammals, what can we conclude about whales?",
        "system": "You demonstrate logical reasoning chains. Show each step of deduction clearly.",
        "category": "syllogism"
    },
    {
        "prompt": "A farmer has chickens and cows. There are 20 heads and 56 legs. How many of each animal?",
        "system": "Solve this step-by-step showing your algebraic reasoning.",
        "category": "algebra_word_problem"
    },
    {
        "prompt": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        "system": "Think through this carefully. Many people get it wrong by assuming linear scaling.",
        "category": "rate_problem"
    },
    # Debugging and analysis
    {
        "prompt": "This code should print even numbers but has a bug:\nfor i in range(10):\n    if i % 2 == 1:\n        print(i)\nWhat's wrong?",
        "system": "Analyze code systematically. Identify the bug and explain the fix.",
        "category": "debugging"
    },
    {
        "prompt": "Why might a binary search fail on an unsorted array?",
        "system": "Explain algorithm requirements and failure modes clearly.",
        "category": "algorithm_analysis"
    },
    # Complex reasoning
    {
        "prompt": "Three people check into a hotel room that costs $30. They each pay $10. The manager realizes the room is only $25 and gives $5 to the bellhop to return. The bellhop keeps $2 and gives each person $1 back. So each person paid $9 (total $27), the bellhop has $2. That's $29. Where's the missing dollar?",
        "system": "This is a famous paradox. Explain why the reasoning is flawed.",
        "category": "paradox_resolution"
    },
    {
        "prompt": "You have 8 identical-looking balls. One is slightly heavier. Using a balance scale only twice, how do you find the heavy ball?",
        "system": "Solve classic logic puzzles with clear reasoning.",
        "category": "logic_puzzle"
    },
    # Scientific reasoning
    {
        "prompt": "Why does ice float on water, and why is this important for life on Earth?",
        "system": "Connect scientific concepts to their broader implications.",
        "category": "scientific_reasoning"
    },
    {
        "prompt": "Explain why correlation does not imply causation with a concrete example.",
        "system": "Teach critical thinking about data and statistics.",
        "category": "statistical_reasoning"
    },
    # Abstract reasoning
    {
        "prompt": "What's the next number: 1, 11, 21, 1211, 111221, ?",
        "system": "Identify the pattern in this look-and-say sequence and explain it.",
        "category": "pattern_recognition"
    },
]


async def generate_reasoning_chains(
    api_key: str,
    model: str,
    workers: int = 10
) -> List[Dict]:
    """Generate sophisticated reasoning chain training data."""

    print(f"\n{MAGENTA}{'='*60}{RESET}")
    print(f"{MAGENTA}  Phase 2: Sophisticated Reasoning Chains{RESET}")
    print(f"{MAGENTA}{'='*60}{RESET}")

    pairs = []
    semaphore = asyncio.Semaphore(workers)

    async def generate_reasoning(item: Dict) -> Optional[Dict]:
        async with semaphore:
            # Add small delay to spread out requests
            await asyncio.sleep(RATE_LIMIT_BUFFER)

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            # Use OpenAI Responses API format
            data = {
                "model": model,
                "input": f"{item['system']}\n\n{item['prompt']}"
            }

            async with aiohttp.ClientSession() as session:
                result, error = await api_call_with_retry(
                    session,
                    get_api_url("responses"),
                    headers,
                    data,
                    timeout=60
                )

                if result:
                    response = parse_responses_api(result)
                    return {
                        "messages": [
                            {"role": "user", "content": item["prompt"]},
                            {"role": "assistant", "content": response}
                        ],
                        "metadata": {
                            "category": item["category"],
                            "type": "sophisticated_reasoning"
                        }
                    }
            return None

    print(f"{DIM}Generating {len(REASONING_PROMPTS)} reasoning chains...{RESET}")

    tasks = [generate_reasoning(item) for item in REASONING_PROMPTS]
    results = await asyncio.gather(*tasks)

    pairs = [r for r in results if r]
    print(f"{GREEN}Generated {len(pairs)}/{len(REASONING_PROMPTS)} reasoning chains{RESET}")

    return pairs


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def load_existing_foundations() -> Dict[str, ConceptFoundation]:
    """Load existing concept foundations."""
    if not CONCEPT_FOUNDATIONS_PATH.exists():
        return {}

    try:
        with open(CONCEPT_FOUNDATIONS_PATH) as f:
            data = json.load(f)

        foundations = {}
        for concept, info in data.items():
            foundations[concept.lower()] = ConceptFoundation(
                concept=info.get('concept', concept),
                category=info.get('category', 'concept'),
                definition=info.get('definition', ''),
                examples=info.get('examples', []),
                training_pairs=info.get('training_pairs', [])
            )
        return foundations
    except Exception as e:
        print(f"{YELLOW}Warning loading foundations: {e}{RESET}")
        return {}


def find_incomplete_concepts(
    all_concepts: List[Tuple[str, str]],
    existing: Dict[str, ConceptFoundation]
) -> List[Tuple[str, str]]:
    """Find concepts that are missing or have incomplete foundations."""
    incomplete = []

    for concept, category in all_concepts:
        key = concept.lower()
        if key not in existing or not existing[key].is_complete():
            incomplete.append((concept, category))

    return incomplete


def evaluate_concepts(
    all_concepts: List[Tuple[str, str]],
    existing: Dict[str, ConceptFoundation]
) -> Tuple[List[Tuple[str, str]], int]:
    """Return incomplete concepts and complete count using the same criteria."""
    incomplete = []
    complete_count = 0

    for concept, category in all_concepts:
        key = concept.lower()
        foundation = existing.get(key)
        if foundation and foundation.is_complete():
            complete_count += 1
        else:
            incomplete.append((concept, category))

    return incomplete, complete_count


def save_foundations(foundations: Dict[str, ConceptFoundation]) -> None:
    """Save concept foundations to JSON."""
    CONCEPT_FOUNDATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)

    data = {k: v.to_dict() for k, v in foundations.items()}

    tmp_path = CONCEPT_FOUNDATIONS_PATH.with_name(CONCEPT_FOUNDATIONS_PATH.name + ".tmp")
    if CONCEPT_FOUNDATIONS_PATH.exists():
        backup_path = CONCEPT_FOUNDATIONS_PATH.with_name(CONCEPT_FOUNDATIONS_PATH.name + ".bak")
        shutil.copy2(CONCEPT_FOUNDATIONS_PATH, backup_path)
    with open(tmp_path, 'w') as f:
        json.dump(data, f, indent=2)
    tmp_path.replace(CONCEPT_FOUNDATIONS_PATH)


def save_training_jsonl(foundations: Dict[str, ConceptFoundation]) -> int:
    """Convert foundations to training JSONL."""
    FOUNDATIONAL_KNOWLEDGE_PATH.parent.mkdir(parents=True, exist_ok=True)

    pairs = []
    for concept, foundation in foundations.items():
        # Definition pair
        if foundation.definition:
            pairs.append({
                "messages": [
                    {"role": "user", "content": f"What is {foundation.concept}?"},
                    {"role": "assistant", "content": foundation.definition}
                ],
                "metadata": {
                    "concept": foundation.concept,
                    "category": foundation.category,
                    "type": "definition"
                }
            })

        # Q&A pairs
        for pair in foundation.training_pairs:
            pairs.append({
                "messages": [
                    {"role": "user", "content": pair.get("user", "")},
                    {"role": "assistant", "content": pair.get("assistant", "")}
                ],
                "metadata": {
                    "concept": foundation.concept,
                    "category": foundation.category,
                    "type": "qa_pair"
                }
            })

    tmp_path = FOUNDATIONAL_KNOWLEDGE_PATH.with_name(FOUNDATIONAL_KNOWLEDGE_PATH.name + ".tmp")
    if FOUNDATIONAL_KNOWLEDGE_PATH.exists():
        backup_path = FOUNDATIONAL_KNOWLEDGE_PATH.with_name(FOUNDATIONAL_KNOWLEDGE_PATH.name + ".bak")
        shutil.copy2(FOUNDATIONAL_KNOWLEDGE_PATH, backup_path)
    with open(tmp_path, 'w') as f:
        for pair in pairs:
            f.write(json.dumps(pair) + '\n')
    tmp_path.replace(FOUNDATIONAL_KNOWLEDGE_PATH)

    return len(pairs)


def save_bridge_jsonl(pairs: List[Dict]) -> int:
    """Save bridge data to JSONL."""
    BRIDGE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = BRIDGE_DATA_PATH.with_name(BRIDGE_DATA_PATH.name + ".tmp")
    if BRIDGE_DATA_PATH.exists():
        backup_path = BRIDGE_DATA_PATH.with_name(BRIDGE_DATA_PATH.name + ".bak")
        shutil.copy2(BRIDGE_DATA_PATH, backup_path)
    with open(tmp_path, 'w') as f:
        for pair in pairs:
            f.write(json.dumps(pair) + '\n')
    tmp_path.replace(BRIDGE_DATA_PATH)

    return len(pairs)


def save_reasoning_jsonl(pairs: List[Dict]) -> int:
    """Save reasoning chains to JSONL."""
    REASONING_CHAINS_PATH.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = REASONING_CHAINS_PATH.with_name(REASONING_CHAINS_PATH.name + ".tmp")
    if REASONING_CHAINS_PATH.exists():
        backup_path = REASONING_CHAINS_PATH.with_name(REASONING_CHAINS_PATH.name + ".bak")
        shutil.copy2(REASONING_CHAINS_PATH, backup_path)
    with open(tmp_path, 'w') as f:
        for pair in pairs:
            f.write(json.dumps(pair) + '\n')
    tmp_path.replace(REASONING_CHAINS_PATH)

    return len(pairs)


async def main():
    parser = argparse.ArgumentParser(
        description="Upgrade and Serve - Complete Training Data Generation Pipeline"
    )
    parser.add_argument("--model", "-m", type=str, default="gpt-4o-mini",
                        help="Model for generation (default: gpt-4o-mini)")
    parser.add_argument("--workers", "-w", type=int, default=10,
                        help="Concurrent API workers (default: 10, lower to avoid rate limits)")
    parser.add_argument("--check-only", action="store_true",
                        help="Only report missing, don't generate")
    parser.add_argument("--skip-reasoning", action="store_true",
                        help="Skip Phase 2 (sophisticated reasoning)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit concepts to generate per run")
    parser.add_argument("--force", action="store_true",
                        help="Force regenerate all foundations")
    parser.add_argument("--print-saves", action="store_true",
                        help="Print saved foundations (definition/examples/QA)")
    parser.add_argument("--retry-incomplete", type=int, default=3,
                        help="Retries for incomplete concepts (default: 3)")
    parser.add_argument("--min-pairs", type=int, default=MIN_TRAINING_PAIRS_DEFAULT,
                        help="Minimum training pairs per concept (default: 5)")
    parser.add_argument("--bridge-target", type=int, default=400,
                        help="Minimum total bridge pairs to generate (default: 400, 0 disables synthetic)")
    parser.add_argument("--bridge-seed", type=int, default=42,
                        help="Seed for synthetic bridge generation (default: 42)")
    args = parser.parse_args()

    global MIN_TRAINING_PAIRS
    MIN_TRAINING_PAIRS = max(1, args.min_pairs)

    print(f"\n{BOLD}{CYAN}{'='*70}{RESET}")
    print(f"{BOLD}{CYAN}  UPGRADE AND SERVE - Training Data Generation Pipeline{RESET}")
    print(f"{BOLD}{CYAN}{'='*70}{RESET}")
    print(f"{DIM}  Phase 1: Foundational Knowledge (concepts without values){RESET}")
    print(f"{DIM}  Phase 1.5: Bridge Data (foundational -> sophisticated){RESET}")
    print(f"{DIM}  Phase 2: Sophisticated Reasoning Chains (optional){RESET}")
    print(f"{DIM}{'='*70}{RESET}")

    start_time = time.time()

    # ========================================================================
    # PHASE 1: FOUNDATIONAL KNOWLEDGE
    # ========================================================================

    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{CYAN}  Phase 1: Foundational Knowledge Generation{RESET}")
    print(f"{CYAN}{'='*60}{RESET}")

    # Step 1: Load concepts from curated JSON ONLY
    # (Don't extract from training data - use the curated list for foundational generation)
    print(f"\n{CYAN}Step 1: Loading concepts from {FOUNDATIONAL_CONCEPTS_PATH}...{RESET}")
    extractor = ConceptExtractor()

    if FOUNDATIONAL_CONCEPTS_PATH.exists():
        extractor.load_from_json(FOUNDATIONAL_CONCEPTS_PATH)
        print(f"{GREEN}  Loaded {len(extractor.concepts)} concepts{RESET}")
    else:
        print(f"{YELLOW}  Warning: {FOUNDATIONAL_CONCEPTS_PATH} not found{RESET}")

    all_concepts = extractor.get_all_concepts()
    print(f"{GREEN}  Target concepts for foundational generation: {len(all_concepts)}{RESET}")

    # Step 2: Load existing foundations
    print(f"\n{CYAN}Step 2: Checking existing foundations...{RESET}")
    existing = load_existing_foundations()
    print(f"{DIM}  Existing foundations: {len(existing)}{RESET}")

    # Step 3: Find incomplete/missing concepts
    print(f"\n{CYAN}Step 3: Finding incomplete concepts...{RESET}")
    target_keys = {c.lower() for c, _ in all_concepts}
    total_concepts = len(target_keys)

    if args.force:
        incomplete = all_concepts
        complete_count = 0
        print(f"{YELLOW}  Force mode: regenerating all {len(incomplete)} concepts{RESET}")
    else:
        incomplete, complete_count = evaluate_concepts(all_concepts, existing)
        print(f"{GREEN}  Complete: {complete_count}/{total_concepts}{RESET}")
        print(f"{YELLOW}  Incomplete/missing: {len(incomplete)}{RESET}")

    if args.check_only:
        print(f"\n{YELLOW}Check-only mode. Run without --check-only to generate.{RESET}")
        if incomplete:
            print(f"\n{DIM}Sample incomplete concepts:{RESET}")
            for c, cat in incomplete[:20]:
                print(f"  - {c} ({cat})")
            if len(incomplete) > 20:
                print(f"  ... and {len(incomplete) - 20} more")
        return

    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(f"\n{RED}ERROR: OPENAI_API_KEY not set{RESET}")
        print(f"{DIM}Set with: export OPENAI_API_KEY='sk-...'{RESET}")
        return

    # Step 4: Generate foundations for incomplete concepts
    if incomplete:
        if args.limit:
            incomplete = incomplete[:args.limit]
            print(f"\n{YELLOW}Limited to {args.limit} concepts{RESET}")

        merged = {k.lower(): v for k, v in existing.items()}
        complete_count = [complete_count]

        def trim(text: str, limit: int = 140) -> str:
            if len(text) <= limit:
                return text
            return text[:limit - 3].rstrip() + "..."

        def persist_foundation(foundation: ConceptFoundation, completed: int, total: int) -> None:
            key = foundation.concept.lower()
            prev = merged.get(key)
            was_complete = prev.is_complete() if prev else False
            merged[key] = foundation
            save_foundations(merged)
            print(f"{GREEN}  [{completed}/{total}] saved {foundation.concept} ({len(foundation.training_pairs)} pairs){RESET}")
            if args.print_saves:
                if foundation.definition:
                    print(f"{DIM}    definition: {trim(foundation.definition)}{RESET}")
                if foundation.examples:
                    examples = "; ".join(trim(ex, 80) for ex in foundation.examples[:3])
                    print(f"{DIM}    examples: {examples}{RESET}")
                if foundation.training_pairs:
                    first = foundation.training_pairs[0]
                    q = trim(first.get('user', ''), 80)
                    a = trim(first.get('assistant', ''), 120)
                    if q or a:
                        print(f"{DIM}    qa: {q} -> {a}{RESET}")
            if foundation.is_complete() and not was_complete:
                complete_count[0] += 1
            pct = (complete_count[0] / total_concepts) * 100 if total_concepts else 0
            print(f"{DIM}  Progress: {complete_count[0]}/{total_concepts} complete ({pct:.1f}%) {RESET}")

        def progress_update(concept: str, status: str, completed: int, total: int, result: Optional[ConceptFoundation]) -> None:
            if status == "complete":
                return
            pct = (complete_count[0] / total_concepts) * 100 if total_concepts else 0
            print(f"{YELLOW}  [{completed}/{total}] {status} {concept}{RESET}")
            print(f"{DIM}  Progress: {complete_count[0]}/{total_concepts} complete ({pct:.1f}%) {RESET}")

        async with FoundationGenerator(api_key, args.model) as generator:
            new_foundations = await generator.generate_all(
                incomplete,
                args.workers,
                on_result=persist_foundation,
                on_progress=progress_update,
                retry_incomplete=args.retry_incomplete
            )

        print(f"\n{GREEN}Generated {len(new_foundations)} complete foundations{RESET}")

        # Post-run complete count
        final_complete = sum(1 for k in target_keys if k in merged and merged[k].is_complete())
        print(f"{GREEN}Post-run complete: {final_complete}/{total_concepts}{RESET}")

        # Save foundations JSON
        print(f"\n{CYAN}Saving foundations...{RESET}")
        save_foundations(merged)
        print(f"{GREEN}  Saved to {CONCEPT_FOUNDATIONS_PATH}{RESET}")
    else:
        print(f"\n{GREEN}All concepts have complete foundations!{RESET}")
        merged = existing

    # ALWAYS regenerate training JSONL with ALL existing foundations
    # This ensures persistence - even if concepts change, existing foundations are used
    print(f"\n{CYAN}Regenerating training JSONL with all foundations...{RESET}")
    pair_count = save_training_jsonl(merged)
    print(f"{GREEN}  Saved {pair_count} training pairs to {FOUNDATIONAL_KNOWLEDGE_PATH}{RESET}")
    print(f"{DIM}  (All {len(merged)} foundations preserved for training){RESET}")

    # ========================================================================
    # PHASE 1.5: BRIDGE DATA
    # ========================================================================

    bridge_pairs = await generate_bridge_data(
        api_key,
        args.model,
        args.workers,
        bridge_target=args.bridge_target,
        bridge_seed=args.bridge_seed
    )

    bridge_count = save_bridge_jsonl(bridge_pairs)
    print(f"\n{GREEN}Saved {bridge_count} bridge pairs to {BRIDGE_DATA_PATH}{RESET}")

    # ========================================================================
    # PHASE 2: SOPHISTICATED REASONING (Optional)
    # ========================================================================

    reasoning_pairs = []
    if not args.skip_reasoning:
        reasoning_pairs = await generate_reasoning_chains(api_key, args.model, args.workers)

        if reasoning_pairs:
            reasoning_count = save_reasoning_jsonl(reasoning_pairs)
            print(f"\n{GREEN}Saved {reasoning_count} reasoning chains to {REASONING_CHAINS_PATH}{RESET}")
    else:
        print(f"\n{DIM}Skipping Phase 2 (sophisticated reasoning){RESET}")

    # ========================================================================
    # SUMMARY
    # ========================================================================

    elapsed = time.time() - start_time

    print(f"\n{GREEN}{'='*70}{RESET}")
    print(f"{GREEN}  PIPELINE COMPLETE{RESET}")
    print(f"{GREEN}{'='*70}{RESET}")
    print(f"{DIM}  Time: {elapsed:.1f}s{RESET}")
    print(f"{DIM}  Total concepts: {len(all_concepts)}{RESET}")
    print(f"{DIM}  Phase 1 (Foundational): {len(merged)} foundations{RESET}")
    print(f"{DIM}  Phase 1.5 (Bridge): {len(bridge_pairs)} pairs{RESET}")
    print(f"{DIM}  Phase 2 (Reasoning): {len(reasoning_pairs)} chains{RESET}")
    print(f"\n{DIM}  Output files:{RESET}")
    print(f"{DIM}    - {CONCEPT_FOUNDATIONS_PATH}{RESET}")
    print(f"{DIM}    - {FOUNDATIONAL_KNOWLEDGE_PATH}{RESET}")
    print(f"{DIM}    - {BRIDGE_DATA_PATH}{RESET}")
    if reasoning_pairs:
        print(f"{DIM}    - {REASONING_CHAINS_PATH}{RESET}")

    print(f"\n{GREEN}Ready for training!{RESET}")


if __name__ == "__main__":
    asyncio.run(main())
