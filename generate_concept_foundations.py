#!/usr/bin/env python3
"""
Generate Foundational Knowledge for Each Concept in Training Data

This creates a systematic dictionary mapping each noun/concept to its
high-quality foundational training pairs. Ensures complete vocabulary coverage.

Approach:
1. Extract all unique concepts from training data vocabulary
2. Create dict: concept -> [training pairs]
3. Generate foundational knowledge for each concept via API
4. Ensure every concept has proper coverage before training

Usage:
    python generate_concept_foundations.py --output cache/foundations/concept_dict.json
    python generate_concept_foundations.py --extract-only  # Just show concepts
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
from dataclasses import dataclass, asdict

# ANSI Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"


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
class ConceptFoundation:
    """Foundational knowledge for a single concept."""
    concept: str
    category: str  # noun, verb, number, greeting, etc.
    definition: str
    examples: List[str]
    training_pairs: List[Dict]  # [{user: ..., assistant: ...}, ...]

    def to_dict(self):
        return asdict(self)


class ConceptExtractor:
    """Extract ALL concepts from training data using tokenization and hashing."""

    # Stop words to filter out
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
        # Common tokens that don't need foundations
        '\n', '\t', ' ', '', '.', ',', '!', '?', ':', ';', '"', "'", '`',
        '(', ')', '[', ']', '{', '}', '<', '>', '/', '\\', '|', '-', '_',
        '+', '=', '*', '&', '^', '%', '$', '#', '@', '~',
    }

    def __init__(self):
        # Hash table: concept_hash -> (concept, category, count)
        self.concept_hash: Dict[int, Tuple[str, str, int]] = {}
        # Also keep by category for organization
        self.by_category: Dict[str, Set[str]] = defaultdict(set)

    def _hash_concept(self, concept: str) -> int:
        """Create stable hash for a concept."""
        return hash(concept.lower().strip())

    def _categorize(self, word: str) -> str:
        """Categorize a word/concept."""
        word_lower = word.lower()

        # Numbers
        if word_lower.isdigit() or word_lower in {
            'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
            'eight', 'nine', 'ten', 'eleven', 'twelve', 'hundred', 'thousand'
        }:
            return 'number'

        # Greetings
        if word_lower in {'hi', 'hello', 'hey', 'goodbye', 'bye', 'thanks', 'please', 'sorry'}:
            return 'greeting'

        # Question words
        if word_lower in {'what', 'who', 'why', 'how', 'when', 'where', 'which'}:
            return 'question_word'

        # Tech terms
        if word_lower in {
            'python', 'javascript', 'java', 'html', 'css', 'sql', 'git', 'docker',
            'linux', 'windows', 'api', 'json', 'http', 'https', 'url', 'cpu', 'gpu',
            'memory', 'server', 'client', 'database', 'function', 'class', 'method',
            'variable', 'loop', 'array', 'object', 'string', 'integer', 'boolean',
            'list', 'dict', 'tuple', 'set', 'import', 'export', 'module', 'package'
        }:
            return 'tech_term'

        # Default: noun/concept
        return 'concept'

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words/concepts."""
        # Split on whitespace and punctuation but keep meaningful tokens
        tokens = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_]*\b|\b\d+\b', text)
        return tokens

    def add_concept(self, concept: str, source: str = 'training') -> None:
        """Add a concept to the hash table."""
        concept = concept.strip()

        # Filter stop words and short tokens
        if concept.lower() in self.STOP_WORDS or len(concept) < 2:
            return

        # Filter pure numbers > 1000 (not useful for foundations)
        if concept.isdigit() and int(concept) > 1000:
            return

        h = self._hash_concept(concept)
        category = self._categorize(concept)

        if h in self.concept_hash:
            # Increment count
            existing = self.concept_hash[h]
            self.concept_hash[h] = (existing[0], existing[1], existing[2] + 1)
        else:
            self.concept_hash[h] = (concept, category, 1)
            self.by_category[category].add(concept)

    def extract_from_text(self, text: str) -> None:
        """Extract all concepts from a piece of text."""
        tokens = self._tokenize(text)
        for token in tokens:
            self.add_concept(token)

    def extract_from_training_data(self) -> Dict[str, Set[str]]:
        """Extract ALL concepts from the full training corpus."""
        try:
            from data import get_all_training_data, GREETINGS, KIDS_QA

            # Get ALL training data
            print(f"{DIM}Loading full training corpus...{RESET}")
            all_data = get_all_training_data(balanced=True)

            print(f"{DIM}Extracting from {len(all_data):,} training pairs...{RESET}")

            for prompt, response in all_data:
                self.extract_from_text(prompt)
                self.extract_from_text(response)

            print(f"{GREEN}✓ Extracted {len(self.concept_hash):,} unique concepts{RESET}")

        except ImportError as e:
            print(f"{YELLOW}Warning: Could not import from data.py: {e}{RESET}")

        return self.by_category

    def get_all_concepts(self) -> List[Tuple[str, str]]:
        """Get all concepts as (concept, category) tuples, sorted by frequency."""
        # Sort by count (most frequent first) for priority
        sorted_concepts = sorted(
            self.concept_hash.values(),
            key=lambda x: -x[2]  # Descending by count
        )
        return [(c[0], c[1]) for c in sorted_concepts]

    def get_coverage_stats(self) -> Dict:
        """Get coverage statistics."""
        return {
            'total_concepts': len(self.concept_hash),
            'by_category': {k: len(v) for k, v in self.by_category.items()},
            'top_10': [(c[0], c[2]) for c in sorted(
                self.concept_hash.values(), key=lambda x: -x[2]
            )[:10]]
        }


class FoundationGenerator:
    """Generate foundational knowledge for concepts using API."""

    def __init__(self, api_key: str, model: str = "gpt-5.1-codex-mini"):
        self.api_key = api_key
        self.model = model
        self.foundations: Dict[str, ConceptFoundation] = {}

    async def generate_for_concept(
        self,
        session: aiohttp.ClientSession,
        concept: str,
        category: str
    ) -> ConceptFoundation:
        """Generate foundational knowledge for a single concept."""

        # Create prompt based on category
        if category == 'numbers':
            prompt = f"""Generate foundational knowledge for the number/quantity "{concept}".

Provide:
1. A clear definition of what this number represents
2. 3 example sentences using this number
3. 5 Q&A training pairs about this number (format: Q: ... A: ...)

Be educational and precise."""

        elif category == 'greetings':
            prompt = f"""Generate foundational knowledge for the greeting/phrase "{concept}".

Provide:
1. What this phrase means and when to use it
2. 3 example conversational exchanges
3. 5 Q&A training pairs showing proper responses (format: Q: ... A: ...)

Be natural and conversational."""

        elif category in ['common_nouns', 'tech_terms']:
            prompt = f"""Generate foundational knowledge for the concept "{concept}".

Provide:
1. A clear, educational definition
2. 3 example sentences showing proper usage
3. 5 Q&A training pairs about this concept (format: Q: ... A: ...)

Be precise and informative."""

        elif category == 'common_verbs':
            prompt = f"""Generate foundational knowledge for the verb "{concept}".

Provide:
1. Definition and meaning
2. 3 example sentences showing proper usage
3. 5 Q&A training pairs using this verb (format: Q: ... A: ...)

Be clear and educational."""

        else:
            prompt = f"""Generate foundational knowledge for "{concept}" (category: {category}).

Provide:
1. Clear definition
2. 3 examples
3. 5 Q&A training pairs (format: Q: ... A: ...)

Be educational and precise."""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
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

                    # Parse the response into structured data
                    training_pairs = self._parse_qa_pairs(response_text)
                    examples = self._parse_examples(response_text)
                    definition = self._parse_definition(response_text)

                    return ConceptFoundation(
                        concept=concept,
                        category=category,
                        definition=definition,
                        examples=examples,
                        training_pairs=training_pairs
                    )
                else:
                    error = await resp.text()
                    print(f"{RED}  ✗ {concept}: API error {resp.status}{RESET}")
                    return None

        except Exception as e:
            print(f"{RED}  ✗ {concept}: {e}{RESET}")
            return None

    def _parse_qa_pairs(self, text: str) -> List[Dict]:
        """Parse Q&A pairs from response text."""
        pairs = []
        lines = text.split('\n')

        current_q = None
        for line in lines:
            line = line.strip()

            # Match Q: or Question: patterns
            if line.startswith(('Q:', 'Question:', 'Q.', '- Q:')):
                for prefix in ['Q:', 'Question:', 'Q.', '- Q:']:
                    if line.startswith(prefix):
                        current_q = line[len(prefix):].strip()
                        break

            # Match A: or Answer: patterns
            elif line.startswith(('A:', 'Answer:', 'A.', '- A:')) and current_q:
                for prefix in ['A:', 'Answer:', 'A.', '- A:']:
                    if line.startswith(prefix):
                        answer = line[len(prefix):].strip()
                        pairs.append({
                            "user": current_q,
                            "assistant": answer
                        })
                        current_q = None
                        break

        return pairs

    def _parse_examples(self, text: str) -> List[str]:
        """Parse example sentences from response text."""
        examples = []
        lines = text.split('\n')

        in_examples = False
        for line in lines:
            line = line.strip()

            if 'example' in line.lower() and ':' in line:
                in_examples = True
                continue

            if in_examples and line.startswith(('-', '•', '*', '1.', '2.', '3.')):
                # Clean the example
                example = re.sub(r'^[-•*\d.]+\s*', '', line).strip()
                if example and len(example) > 10:
                    examples.append(example)
                    if len(examples) >= 3:
                        break

        return examples

    def _parse_definition(self, text: str) -> str:
        """Parse definition from response text."""
        lines = text.split('\n')

        for i, line in enumerate(lines):
            line = line.strip()

            # Look for definition markers
            if any(marker in line.lower() for marker in ['definition:', 'meaning:', 'what it means:', '1.']):
                # Get the content after the marker
                for marker in ['Definition:', 'Meaning:', 'definition:', 'meaning:', '1.']:
                    if marker in line:
                        definition = line.split(marker, 1)[-1].strip()
                        if definition:
                            return definition

                # If marker is on its own line, get next line
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line and not next_line.startswith(('Q:', 'A:', '-', '•')):
                        return next_line

        # Fallback: first non-empty substantial line
        for line in lines:
            line = line.strip()
            if line and len(line) > 20 and not line.startswith(('Q:', 'A:', '#')):
                return line

        return ""

    async def generate_all(
        self,
        concepts: List[Tuple[str, str]],
        workers: int = 10
    ) -> Dict[str, ConceptFoundation]:
        """Generate foundations for all concepts."""

        semaphore = asyncio.Semaphore(workers)

        async def bounded_generate(concept: str, category: str):
            async with semaphore:
                async with aiohttp.ClientSession() as session:
                    return await self.generate_for_concept(session, concept, category)

        print(f"\n{CYAN}Generating foundations for {len(concepts)} concepts...{RESET}")

        tasks = [bounded_generate(c, cat) for c, cat in concepts]
        results = await asyncio.gather(*tasks)

        # Build dictionary
        for result in results:
            if result:
                self.foundations[result.concept] = result

        return self.foundations


def convert_to_training_jsonl(foundations: Dict[str, ConceptFoundation]) -> List[Dict]:
    """Convert foundation dict to training JSONL format."""
    training_pairs = []

    for concept, foundation in foundations.items():
        # Add definition as a training pair
        if foundation.definition:
            training_pairs.append({
                "messages": [
                    {"role": "user", "content": f"What is {concept}?"},
                    {"role": "assistant", "content": foundation.definition}
                ],
                "metadata": {
                    "concept": concept,
                    "category": foundation.category,
                    "type": "definition"
                }
            })

        # Add all Q&A training pairs
        for pair in foundation.training_pairs:
            training_pairs.append({
                "messages": [
                    {"role": "user", "content": pair["user"]},
                    {"role": "assistant", "content": pair["assistant"]}
                ],
                "metadata": {
                    "concept": concept,
                    "category": foundation.category,
                    "type": "qa_pair"
                }
            })

    return training_pairs


def load_existing_foundations(path: Path) -> Dict[str, ConceptFoundation]:
    """Load existing concept foundations from JSON file."""
    if not path.exists():
        return {}

    try:
        with open(path) as f:
            data = json.load(f)

        foundations = {}
        for concept, info in data.items():
            foundations[concept] = ConceptFoundation(
                concept=info.get('concept', concept),
                category=info.get('category', 'concept'),
                definition=info.get('definition', ''),
                examples=info.get('examples', []),
                training_pairs=info.get('training_pairs', [])
            )
        return foundations
    except (json.JSONDecodeError, KeyError) as e:
        print(f"{YELLOW}Warning: Could not load existing foundations: {e}{RESET}")
        return {}


async def main():
    parser = argparse.ArgumentParser(description="Generate concept foundations")
    parser.add_argument("--output", "-o", type=str,
                        default="cache/foundations/concept_foundations.json",
                        help="Output JSON file for concept dictionary")
    parser.add_argument("--training-output", "-t", type=str,
                        default="cache/foundations/foundational_knowledge.jsonl",
                        help="Output JSONL for training")
    parser.add_argument("--model", "-m", type=str, default="gpt-5.1-codex-mini",
                        help="Model for generation")
    parser.add_argument("--workers", "-w", type=int, default=10,
                        help="Concurrent API workers")
    parser.add_argument("--extract-only", action="store_true",
                        help="Only extract and show concepts, don't generate")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of NEW concepts to process per run")
    parser.add_argument("--force-regenerate", action="store_true",
                        help="Regenerate all foundations, not just missing ones")
    args = parser.parse_args()

    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  Concept Foundation Generator (Incremental){RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")

    # Step 1: Extract ALL concepts from current training data via tokenization
    print(f"\n{CYAN}Step 1: Tokenizing training data to extract concepts...{RESET}")
    extractor = ConceptExtractor()
    extractor.extract_from_training_data()

    all_concepts = extractor.get_all_concepts()

    print(f"\n{GREEN}Found {len(all_concepts)} unique concepts from tokenization:{RESET}")
    for category, concepts in extractor.by_category.items():
        print(f"  {category}: {len(concepts)} concepts")
        if len(concepts) <= 10:
            print(f"    {DIM}{', '.join(sorted(concepts))}{RESET}")
        else:
            sample = list(sorted(concepts))[:5]
            print(f"    {DIM}{', '.join(sample)}... (+{len(concepts)-5} more){RESET}")

    if args.extract_only:
        print(f"\n{YELLOW}Extract-only mode. Exiting.{RESET}")
        return

    # Step 2: Load existing foundations
    output_path = Path(args.output)
    existing_foundations = load_existing_foundations(output_path)
    existing_concepts = set(existing_foundations.keys())

    print(f"\n{CYAN}Step 2: Checking existing foundations...{RESET}")
    print(f"  {DIM}Existing concepts with foundations: {len(existing_concepts)}{RESET}")

    # Step 3: Find concepts that need foundations
    all_concept_names = {c[0] for c in all_concepts}

    if args.force_regenerate:
        missing_concepts = all_concepts
        print(f"  {YELLOW}Force regenerate mode: will regenerate all {len(missing_concepts)} concepts{RESET}")
    else:
        missing_concept_names = all_concept_names - existing_concepts
        missing_concepts = [(c, cat) for c, cat in all_concepts if c in missing_concept_names]
        print(f"  {GREEN}New concepts needing foundations: {len(missing_concepts)}{RESET}")

    if not missing_concepts:
        print(f"\n{GREEN}✓ All concepts already have foundations! Nothing to generate.{RESET}")
        # Still save to ensure training JSONL is up to date
        training_pairs = convert_to_training_jsonl(existing_foundations)
        training_path = Path(args.training_output)
        training_path.parent.mkdir(parents=True, exist_ok=True)
        with open(training_path, 'w') as f:
            for pair in training_pairs:
                f.write(json.dumps(pair) + '\n')
        print(f"  {DIM}Updated training data: {len(training_pairs)} pairs{RESET}")
        return

    # Step 4: Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(f"{RED}ERROR: OPENAI_API_KEY not set{RESET}")
        print(f"{DIM}  Cannot generate foundations for {len(missing_concepts)} new concepts{RESET}")
        return

    # Step 5: Generate foundations for missing concepts only
    if args.limit and len(missing_concepts) > args.limit:
        missing_concepts = missing_concepts[:args.limit]
        print(f"\n{YELLOW}Limited to {args.limit} new concepts this run{RESET}")

    print(f"\n{CYAN}Step 3: Generating foundations for {len(missing_concepts)} new concepts...{RESET}")

    generator = FoundationGenerator(api_key, args.model)
    new_foundations = await generator.generate_all(missing_concepts, args.workers)

    print(f"\n{GREEN}✓ Generated foundations for {len(new_foundations)} new concepts{RESET}")

    # Step 6: Merge with existing foundations
    merged_foundations = {**existing_foundations}
    for concept, foundation in new_foundations.items():
        merged_foundations[concept] = foundation

    foundations = merged_foundations
    print(f"  {DIM}Total foundations: {len(foundations)} ({len(new_foundations)} new){RESET}")

    # Step 4: Save concept dictionary
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    foundations_dict = {k: v.to_dict() for k, v in foundations.items()}

    with open(output_path, 'w') as f:
        json.dump(foundations_dict, f, indent=2)

    print(f"{GREEN}✓ Saved concept dictionary: {args.output}{RESET}")

    # Step 5: Convert to training format
    training_pairs = convert_to_training_jsonl(foundations)

    training_path = Path(args.training_output)
    training_path.parent.mkdir(parents=True, exist_ok=True)

    with open(training_path, 'w') as f:
        for pair in training_pairs:
            f.write(json.dumps(pair) + '\n')

    print(f"{GREEN}✓ Saved training data: {args.training_output}{RESET}")

    # Summary
    print(f"\n{GREEN}{'='*60}{RESET}")
    print(f"{GREEN}  Generation Complete!{RESET}")
    print(f"{GREEN}{'='*60}{RESET}")
    print(f"{DIM}  Concepts covered: {len(foundations)}{RESET}")
    print(f"{DIM}  Training pairs: {len(training_pairs)}{RESET}")
    print(f"{DIM}  Concept dict: {args.output}{RESET}")
    print(f"{DIM}  Training JSONL: {args.training_output}{RESET}")


if __name__ == "__main__":
    asyncio.run(main())
