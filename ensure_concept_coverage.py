#!/usr/bin/env python3
"""
Ensure Concept Coverage - Run at start of each upgrade/serve cycle

This script:
1. Updates unique concepts via tokenization of all training data
2. Checks each concept has associated training data
3. Generates missing concept data using gpt-5.1-codex-mini

Usage:
    python ensure_concept_coverage.py                    # Full check + generate
    python ensure_concept_coverage.py --check-only       # Just report missing
    python ensure_concept_coverage.py --model gpt-4o     # Use different model
"""

import argparse
import json
import os
import asyncio
import aiohttp
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

# ANSI Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"

# Paths
CONCEPT_DICT_PATH = Path("cache/foundations/concept_foundations.json")
TRAINING_JSONL_PATH = Path("cache/foundations/foundational_knowledge.jsonl")


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


class ConceptTokenizer:
    """Extract unique concepts from training data via tokenization."""

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
        '\n', '\t', ' ', '', '.', ',', '!', '?', ':', ';', '"', "'", '`',
    }

    def __init__(self):
        self.concepts: Dict[str, Dict] = {}  # concept -> {category, count, sources}

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

        if word_lower in {
            'python', 'javascript', 'java', 'html', 'css', 'sql', 'git', 'docker',
            'linux', 'windows', 'api', 'json', 'http', 'https', 'url', 'cpu', 'gpu',
            'memory', 'server', 'client', 'database', 'function', 'class', 'method',
            'variable', 'loop', 'array', 'object', 'string', 'integer', 'boolean',
            'list', 'dict', 'tuple', 'set', 'import', 'export', 'module', 'package'
        }:
            return 'tech_term'

        return 'concept'

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words/concepts."""
        tokens = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_]*\b|\b\d+\b', text)
        return tokens

    def add_concept(self, concept: str, source: str = 'training') -> None:
        """Add a concept with tracking."""
        concept = concept.strip()

        if concept.lower() in self.STOP_WORDS or len(concept) < 2:
            return

        if concept.isdigit() and int(concept) > 1000:
            return

        key = concept.lower()

        if key not in self.concepts:
            self.concepts[key] = {
                'canonical': concept,
                'category': self._categorize(concept),
                'count': 0,
                'sources': set()
            }

        self.concepts[key]['count'] += 1
        self.concepts[key]['sources'].add(source)

    def extract_from_training_data(self) -> Dict[str, Dict]:
        """Extract ALL concepts from training corpus."""
        try:
            from data import get_all_training_data

            print(f"{DIM}Loading training corpus...{RESET}")
            all_data = get_all_training_data(balanced=True)

            print(f"{DIM}Tokenizing {len(all_data):,} training pairs...{RESET}")

            for prompt, response in all_data:
                for token in self._tokenize(prompt):
                    self.add_concept(token, 'prompt')
                for token in self._tokenize(response):
                    self.add_concept(token, 'response')

            print(f"{GREEN}Extracted {len(self.concepts):,} unique concepts{RESET}")

        except ImportError as e:
            print(f"{YELLOW}Warning: Could not import training data: {e}{RESET}")

        return self.concepts

    def get_concepts_by_category(self) -> Dict[str, List[str]]:
        """Get concepts grouped by category."""
        by_category = defaultdict(list)
        for key, data in self.concepts.items():
            by_category[data['category']].append(data['canonical'])
        return dict(by_category)


def load_existing_foundations() -> Set[str]:
    """Load concepts that already have foundation data."""
    existing = set()

    if CONCEPT_DICT_PATH.exists():
        try:
            with open(CONCEPT_DICT_PATH) as f:
                data = json.load(f)
                existing.update(k.lower() for k in data.keys())
        except Exception as e:
            print(f"{YELLOW}Warning loading concept dict: {e}{RESET}")

    if TRAINING_JSONL_PATH.exists():
        try:
            with open(TRAINING_JSONL_PATH) as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        if 'metadata' in record and 'concept' in record['metadata']:
                            existing.add(record['metadata']['concept'].lower())
                    except:
                        pass
        except Exception as e:
            print(f"{YELLOW}Warning loading training JSONL: {e}{RESET}")

    return existing


def find_missing_concepts(
    all_concepts: Dict[str, Dict],
    existing: Set[str]
) -> List[Tuple[str, str]]:
    """Find concepts without foundation data."""
    missing = []

    for key, data in all_concepts.items():
        if key not in existing:
            missing.append((data['canonical'], data['category']))

    # Sort by frequency (most used first)
    missing_with_count = [
        (c, cat, all_concepts[c.lower()]['count'])
        for c, cat in missing
    ]
    missing_with_count.sort(key=lambda x: -x[2])

    return [(c, cat) for c, cat, _ in missing_with_count]


async def generate_concept_data(
    api_key: str,
    model: str,
    concept: str,
    category: str,
    session: aiohttp.ClientSession
) -> Dict:
    """Generate training data for a single concept."""

    if category == 'number':
        prompt = f"""Create 3 simple Q&A training pairs for the number "{concept}".
Format each as:
Q: [question about {concept}]
A: [clear answer]

Example topics: basic math, counting, real-world uses."""

    elif category == 'greeting':
        prompt = f"""Create 3 conversational Q&A pairs for "{concept}".
Format each as:
Q: [greeting or phrase using {concept}]
A: [natural response]

Keep responses friendly and helpful."""

    elif category == 'tech_term':
        prompt = f"""Create 3 educational Q&A pairs about "{concept}" (technology).
Format each as:
Q: [question about {concept}]
A: [clear, accurate answer]

Cover: definition, usage, and practical examples."""

    else:
        prompt = f"""Create 3 educational Q&A pairs about "{concept}".
Format each as:
Q: [question about {concept}]
A: [informative answer]

Be clear and educational."""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "input": prompt
    }

    try:
        async with session.post(
            get_api_url(),
            headers=headers,
            json=data,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as resp:
            if resp.status == 200:
                result = await resp.json()
                response_text = parse_response(result)

                # Parse Q&A pairs
                pairs = []
                lines = response_text.split('\n')
                current_q = None

                for line in lines:
                    line = line.strip()
                    if line.startswith(('Q:', 'Question:')):
                        current_q = line.split(':', 1)[-1].strip()
                    elif line.startswith(('A:', 'Answer:')) and current_q:
                        answer = line.split(':', 1)[-1].strip()
                        pairs.append({
                            "messages": [
                                {"role": "user", "content": current_q},
                                {"role": "assistant", "content": answer}
                            ],
                            "metadata": {
                                "concept": concept,
                                "category": category,
                                "source": "concept_coverage_gen"
                            }
                        })
                        current_q = None

                return {"concept": concept, "pairs": pairs, "success": True}
            else:
                return {"concept": concept, "pairs": [], "success": False,
                        "error": f"API {resp.status}"}

    except Exception as e:
        return {"concept": concept, "pairs": [], "success": False, "error": str(e)}


async def generate_missing_data(
    missing: List[Tuple[str, str]],
    api_key: str,
    model: str,
    workers: int = 20,
    limit: int = None
) -> List[Dict]:
    """Generate training data for all missing concepts."""

    if limit:
        missing = missing[:limit]

    if not missing:
        return []

    print(f"\n{CYAN}Generating data for {len(missing)} missing concepts...{RESET}")
    print(f"{DIM}Model: {model}{RESET}")

    semaphore = asyncio.Semaphore(workers)
    all_pairs = []

    async def bounded_generate(concept: str, category: str):
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                return await generate_concept_data(
                    api_key, model, concept, category, session
                )

    tasks = [bounded_generate(c, cat) for c, cat in missing]

    completed = 0
    for coro in asyncio.as_completed(tasks):
        result = await coro
        completed += 1

        if result['success'] and result['pairs']:
            all_pairs.extend(result['pairs'])
            print(f"\r{DIM}  [{completed}/{len(missing)}] Generated {len(result['pairs'])} pairs for {result['concept']}{RESET}", end='')
        else:
            print(f"\r{YELLOW}  [{completed}/{len(missing)}] Failed: {result['concept']}{RESET}", end='')

    print()  # newline after progress

    return all_pairs


def save_new_pairs(pairs: List[Dict]) -> int:
    """Append new training pairs to the JSONL file."""
    if not pairs:
        return 0

    TRAINING_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(TRAINING_JSONL_PATH, 'a') as f:
        for pair in pairs:
            f.write(json.dumps(pair) + '\n')

    return len(pairs)


def update_concept_dict(concepts: Dict[str, Dict], new_concepts: Set[str]) -> None:
    """Update the concept dictionary with newly covered concepts."""
    existing_dict = {}

    if CONCEPT_DICT_PATH.exists():
        try:
            with open(CONCEPT_DICT_PATH) as f:
                existing_dict = json.load(f)
        except:
            pass

    # Add new concepts
    for concept in new_concepts:
        if concept.lower() not in existing_dict:
            data = concepts.get(concept.lower(), {})
            existing_dict[concept] = {
                "concept": concept,
                "category": data.get('category', 'concept'),
                "covered": True
            }

    CONCEPT_DICT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(CONCEPT_DICT_PATH, 'w') as f:
        json.dump(existing_dict, f, indent=2)


async def main():
    parser = argparse.ArgumentParser(description="Ensure concept coverage")
    parser.add_argument("--check-only", action="store_true",
                        help="Only report missing concepts, don't generate")
    parser.add_argument("--model", "-m", type=str, default="gpt-5.1-codex-mini",
                        help="Model for generation")
    parser.add_argument("--workers", "-w", type=int, default=20,
                        help="Concurrent API workers")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit concepts to generate (prioritizes most frequent)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed output")
    args = parser.parse_args()

    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  Concept Coverage Check{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")

    # Step 1: Extract all concepts via tokenization
    print(f"\n{CYAN}Step 1: Tokenizing training data...{RESET}")
    tokenizer = ConceptTokenizer()
    all_concepts = tokenizer.extract_from_training_data()

    if args.verbose:
        by_category = tokenizer.get_concepts_by_category()
        for cat, concepts in by_category.items():
            print(f"  {cat}: {len(concepts)} concepts")

    # Step 2: Load existing foundations
    print(f"\n{CYAN}Step 2: Loading existing foundations...{RESET}")
    existing = load_existing_foundations()
    print(f"{DIM}  Found {len(existing)} concepts with existing data{RESET}")

    # Step 3: Find missing concepts
    print(f"\n{CYAN}Step 3: Finding missing concepts...{RESET}")
    missing = find_missing_concepts(all_concepts, existing)

    if not missing:
        print(f"\n{GREEN}All {len(all_concepts)} concepts have training data!{RESET}")
        return

    print(f"{YELLOW}  Found {len(missing)} concepts without training data{RESET}")

    if args.verbose and len(missing) <= 50:
        print(f"{DIM}  Missing: {', '.join(c for c, _ in missing[:20])}{RESET}")
        if len(missing) > 20:
            print(f"{DIM}  ... and {len(missing) - 20} more{RESET}")

    if args.check_only:
        print(f"\n{YELLOW}Check-only mode. Run without --check-only to generate.{RESET}")
        return

    # Step 4: Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(f"\n{RED}ERROR: OPENAI_API_KEY not set{RESET}")
        print(f"{DIM}Set it with: export OPENAI_API_KEY='sk-...'{RESET}")
        return

    # Step 5: Generate training data for missing concepts
    new_pairs = await generate_missing_data(
        missing,
        api_key,
        args.model,
        args.workers,
        args.limit
    )

    if new_pairs:
        # Step 6: Save new training data
        print(f"\n{CYAN}Step 4: Saving new training data...{RESET}")
        saved = save_new_pairs(new_pairs)
        print(f"{GREEN}  Saved {saved} new training pairs{RESET}")

        # Update concept dictionary
        new_concepts = set(p['metadata']['concept'] for p in new_pairs)
        update_concept_dict(all_concepts, new_concepts)
        print(f"{GREEN}  Updated concept dictionary with {len(new_concepts)} concepts{RESET}")

    # Summary
    print(f"\n{GREEN}{'='*60}{RESET}")
    print(f"{GREEN}  Concept Coverage Complete{RESET}")
    print(f"{GREEN}{'='*60}{RESET}")
    print(f"{DIM}  Total concepts: {len(all_concepts)}{RESET}")
    print(f"{DIM}  Previously covered: {len(existing)}{RESET}")
    print(f"{DIM}  Newly generated: {len(new_pairs)} pairs{RESET}")
    remaining = len(missing) - (args.limit or len(missing))
    if remaining > 0:
        print(f"{DIM}  Still missing: {remaining} (use --limit to increase){RESET}")


if __name__ == "__main__":
    asyncio.run(main())
