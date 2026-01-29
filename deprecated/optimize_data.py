#!/usr/bin/env python3
"""
Optimize instruction-following datasets for training.

Operations:
1. Deduplicate (exact and fuzzy)
2. Filter low-quality pairs (too short, too long, empty)
3. Remove conversational/non-instructional patterns
4. Normalize formatting
5. Remove problematic content
6. Balance dataset categories
"""

import json
import re
import hashlib
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Set, Dict
import unicodedata

CACHE_DIR = Path("cache/datasets")
INPUT_FILE = CACHE_DIR / "instruction_data.json"
OUTPUT_FILE = CACHE_DIR / "instruction_data_optimized.json"

# Quality thresholds
MIN_PROMPT_LENGTH = 10  # chars
MIN_RESPONSE_LENGTH = 20  # chars
MAX_PROMPT_LENGTH = 4000  # chars
MAX_RESPONSE_LENGTH = 8000  # chars
MIN_PROMPT_WORDS = 3
MIN_RESPONSE_WORDS = 5


def load_data() -> List[Tuple[str, str]]:
    """Load the instruction data."""
    print(f"Loading {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"  Loaded {len(data):,} pairs")
    return data


def normalize_text(text: str) -> str:
    """Normalize text for comparison and storage."""
    if not text:
        return ""

    # Unicode normalization
    text = unicodedata.normalize('NFKC', text)

    # Fix common encoding issues
    text = text.replace('\u200b', '')  # Zero-width space
    text = text.replace('\u00a0', ' ')  # Non-breaking space
    text = text.replace('\ufeff', '')  # BOM

    # Normalize whitespace
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\r', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Strip
    text = text.strip()

    return text


def get_hash(text: str) -> str:
    """Get hash for deduplication."""
    # Normalize for hashing (lowercase, no punctuation, collapsed whitespace)
    normalized = text.lower()
    normalized = re.sub(r'[^\w\s]', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return hashlib.md5(normalized.encode()).hexdigest()


def is_conversational(prompt: str, response: str) -> bool:
    """Detect if this is movie-dialogue-like conversation (not instruction)."""
    prompt_lower = prompt.lower().strip()
    response_lower = response.lower().strip()

    # Conversational patterns to filter out
    conversational_prompts = [
        r'^(hi|hello|hey|yo|sup)\s*[.!?]*$',
        r'^how are you\??$',
        r'^what\'s up\??$',
        r'^good (morning|afternoon|evening|night)\.?$',
        r'^(bye|goodbye|see you|later|cya)\.?$',
        r'^(thanks|thank you|thx)\.?$',
        r'^(ok|okay|sure|yes|no|yeah|yep|nope)\.?$',
        r'^(nice|cool|great|awesome|wow)\.?$',
        r'^really\??$',
        r'^why\??$',
        r'^what\??$',
        r'^huh\??$',
    ]

    for pattern in conversational_prompts:
        if re.match(pattern, prompt_lower):
            return True

    # Movie dialogue patterns
    movie_patterns = [
        r'\*[^*]+\*',  # *action descriptions*
        r'\([^)]+\)',  # (parenthetical actions)
        r'^[A-Z]+:',   # CHARACTER: dialogue
        r'^\[.+\]$',   # [stage directions]
    ]

    for pattern in movie_patterns:
        if re.search(pattern, prompt) or re.search(pattern, response):
            return True

    return False


def is_low_quality(prompt: str, response: str) -> Tuple[bool, str]:
    """Check if pair is low quality. Returns (is_low_quality, reason)."""

    # Empty or too short
    if not prompt or not response:
        return True, "empty"

    if len(prompt) < MIN_PROMPT_LENGTH:
        return True, "prompt_too_short"

    if len(response) < MIN_RESPONSE_LENGTH:
        return True, "response_too_short"

    # Too long (likely noise or data dump)
    if len(prompt) > MAX_PROMPT_LENGTH:
        return True, "prompt_too_long"

    if len(response) > MAX_RESPONSE_LENGTH:
        return True, "response_too_long"

    # Word count check
    prompt_words = len(prompt.split())
    response_words = len(response.split())

    if prompt_words < MIN_PROMPT_WORDS:
        return True, "prompt_few_words"

    if response_words < MIN_RESPONSE_WORDS:
        return True, "response_few_words"

    # Response is just repeating the prompt
    if response.lower().strip() == prompt.lower().strip():
        return True, "response_equals_prompt"

    # Response starts with "I cannot" or "I can't" without useful content
    refusal_patterns = [
        r'^i (cannot|can\'t|won\'t|am unable to|\'m not able to)',
        r'^sorry,? (i |but )(cannot|can\'t|won\'t)',
        r'^as an ai,? i (cannot|can\'t|don\'t)',
    ]

    for pattern in refusal_patterns:
        if re.match(pattern, response.lower()) and len(response) < 200:
            return True, "short_refusal"

    # Mostly non-ASCII (likely encoding issues)
    ascii_ratio = sum(1 for c in response if ord(c) < 128) / max(len(response), 1)
    if ascii_ratio < 0.7:
        return True, "low_ascii_ratio"

    # Repetitive content
    if has_excessive_repetition(response):
        return True, "repetitive"

    return False, ""


def has_excessive_repetition(text: str) -> bool:
    """Check if text has excessive repetition."""
    if len(text) < 100:
        return False

    # Check for repeated phrases
    words = text.split()
    if len(words) < 20:
        return False

    # Check trigram repetition
    trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
    trigram_counts = Counter(trigrams)

    if trigram_counts:
        max_count = max(trigram_counts.values())
        if max_count > len(trigrams) * 0.1:  # More than 10% same trigram
            return True

    # Check for repeated lines
    lines = text.split('\n')
    if len(lines) > 5:
        line_counts = Counter(line.strip() for line in lines if line.strip())
        if line_counts:
            max_line_count = max(line_counts.values())
            if max_line_count > 3 and max_line_count > len(lines) * 0.2:
                return True

    return False


def clean_response(response: str) -> str:
    """Clean up response formatting."""

    # Remove excessive "As an AI" disclaimers
    ai_disclaimers = [
        r'^As an AI(?: language model)?,?\s*',
        r'^I\'m an AI(?: assistant)?,?\s*',
        r'^As a large language model,?\s*',
    ]

    for pattern in ai_disclaimers:
        response = re.sub(pattern, '', response, flags=re.IGNORECASE)

    # Clean up markdown formatting issues
    response = re.sub(r'\*\*\*+', '**', response)  # Excessive bold
    response = re.sub(r'---+', '---', response)    # Excessive separators
    response = re.sub(r'===+', '===', response)

    # Remove trailing whitespace from lines
    lines = response.split('\n')
    lines = [line.rstrip() for line in lines]
    response = '\n'.join(lines)

    return response.strip()


def clean_prompt(prompt: str) -> str:
    """Clean up prompt formatting."""

    # Remove common prefixes
    prefixes = [
        r'^(Human|User|Question|Q|Input):\s*',
        r'^###\s*(Instruction|Input|Question):\s*',
    ]

    for pattern in prefixes:
        prompt = re.sub(pattern, '', prompt, flags=re.IGNORECASE)

    return prompt.strip()


def deduplicate(data: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Remove duplicate entries."""
    print("Deduplicating...")

    seen_prompts: Set[str] = set()
    seen_responses: Set[str] = set()
    seen_pairs: Set[str] = set()

    unique = []
    dup_count = 0

    for prompt, response in data:
        prompt_hash = get_hash(prompt)
        response_hash = get_hash(response)
        pair_hash = f"{prompt_hash}:{response_hash}"

        # Skip if exact pair duplicate
        if pair_hash in seen_pairs:
            dup_count += 1
            continue

        # Skip if prompt is duplicate (keep first response)
        if prompt_hash in seen_prompts:
            dup_count += 1
            continue

        seen_prompts.add(prompt_hash)
        seen_responses.add(response_hash)
        seen_pairs.add(pair_hash)
        unique.append((prompt, response))

    print(f"  Removed {dup_count:,} duplicates")
    print(f"  Remaining: {len(unique):,} pairs")
    return unique


def filter_quality(data: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Filter out low-quality pairs."""
    print("Filtering low-quality pairs...")

    filtered = []
    reasons: Dict[str, int] = Counter()

    for prompt, response in data:
        # Normalize
        prompt = normalize_text(prompt)
        response = normalize_text(response)

        # Check if conversational
        if is_conversational(prompt, response):
            reasons['conversational'] += 1
            continue

        # Check quality
        is_bad, reason = is_low_quality(prompt, response)
        if is_bad:
            reasons[reason] += 1
            continue

        # Clean
        prompt = clean_prompt(prompt)
        response = clean_response(response)

        # Recheck after cleaning
        if len(prompt) < MIN_PROMPT_LENGTH or len(response) < MIN_RESPONSE_LENGTH:
            reasons['too_short_after_clean'] += 1
            continue

        filtered.append((prompt, response))

    print("  Filtered out by reason:")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"    {reason}: {count:,}")

    total_filtered = sum(reasons.values())
    print(f"  Total filtered: {total_filtered:,}")
    print(f"  Remaining: {len(filtered):,} pairs")

    return filtered


def add_instruction_diversity(data: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Ensure diverse instruction formats by augmenting prompts."""
    print("Adding instruction format diversity...")

    augmented = []

    # Different instruction prefixes to use occasionally
    instruction_formats = [
        lambda p: p,  # Keep as is (most common)
        lambda p: f"Please {p[0].lower()}{p[1:]}" if p[0].isupper() and not p.startswith(('What', 'How', 'Why', 'When', 'Where', 'Who', 'Which', 'Can', 'Could', 'Would', 'Is', 'Are', 'Do', 'Does')) else p,
        lambda p: f"I need help with: {p}",
        lambda p: f"Task: {p}",
    ]

    for i, (prompt, response) in enumerate(data):
        # Keep most prompts as-is, occasionally augment
        if i % 100 < 95:  # 95% unchanged
            augmented.append((prompt, response))
        else:
            # Apply a random format
            format_idx = i % len(instruction_formats)
            new_prompt = instruction_formats[format_idx](prompt)
            augmented.append((new_prompt, response))

    print(f"  Augmented to {len(augmented):,} pairs")
    return augmented


def categorize_and_balance(data: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Analyze and optionally balance dataset categories."""
    print("\nAnalyzing dataset categories...")

    categories = {
        'coding': 0,
        'math': 0,
        'science': 0,
        'writing': 0,
        'general_qa': 0,
        'reasoning': 0,
        'other': 0,
    }

    coding_patterns = [r'code', r'python', r'javascript', r'function', r'program', r'algorithm', r'\bapi\b', r'sql', r'html', r'css']
    math_patterns = [r'calculate', r'solve', r'equation', r'math', r'\d+\s*[\+\-\*\/\=]', r'percent', r'fraction']
    science_patterns = [r'science', r'physics', r'chemistry', r'biology', r'atom', r'molecule', r'cell', r'energy']
    writing_patterns = [r'write', r'essay', r'paragraph', r'summarize', r'explain', r'describe', r'letter', r'email']
    reasoning_patterns = [r'why', r'how does', r'what if', r'compare', r'difference between', r'reason', r'logic']

    for prompt, response in data:
        prompt_lower = prompt.lower()

        if any(re.search(p, prompt_lower) for p in coding_patterns):
            categories['coding'] += 1
        elif any(re.search(p, prompt_lower) for p in math_patterns):
            categories['math'] += 1
        elif any(re.search(p, prompt_lower) for p in science_patterns):
            categories['science'] += 1
        elif any(re.search(p, prompt_lower) for p in writing_patterns):
            categories['writing'] += 1
        elif any(re.search(p, prompt_lower) for p in reasoning_patterns):
            categories['reasoning'] += 1
        elif '?' in prompt:
            categories['general_qa'] += 1
        else:
            categories['other'] += 1

    print("  Category distribution:")
    total = len(data)
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        print(f"    {cat}: {count:,} ({pct:.1f}%)")

    return data  # Return as-is for now, balancing can be added later


def compute_statistics(data: List[Tuple[str, str]]):
    """Compute and print dataset statistics."""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    prompt_lengths = [len(p) for p, _ in data]
    response_lengths = [len(r) for _, r in data]
    prompt_words = [len(p.split()) for p, _ in data]
    response_words = [len(r.split()) for _, r in data]

    print(f"\nTotal pairs: {len(data):,}")

    print(f"\nPrompt length (chars):")
    print(f"  Min: {min(prompt_lengths):,}")
    print(f"  Max: {max(prompt_lengths):,}")
    print(f"  Mean: {sum(prompt_lengths)/len(prompt_lengths):.0f}")
    print(f"  Median: {sorted(prompt_lengths)[len(prompt_lengths)//2]:,}")

    print(f"\nResponse length (chars):")
    print(f"  Min: {min(response_lengths):,}")
    print(f"  Max: {max(response_lengths):,}")
    print(f"  Mean: {sum(response_lengths)/len(response_lengths):.0f}")
    print(f"  Median: {sorted(response_lengths)[len(response_lengths)//2]:,}")

    print(f"\nPrompt words:")
    print(f"  Mean: {sum(prompt_words)/len(prompt_words):.1f}")

    print(f"\nResponse words:")
    print(f"  Mean: {sum(response_words)/len(response_words):.1f}")

    # Sample some entries
    print(f"\n" + "-" * 40)
    print("SAMPLE ENTRIES:")
    print("-" * 40)

    import random
    samples = random.sample(data, min(3, len(data)))
    for i, (prompt, response) in enumerate(samples, 1):
        print(f"\n[{i}] PROMPT ({len(prompt)} chars):")
        print(f"    {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
        print(f"    RESPONSE ({len(response)} chars):")
        print(f"    {response[:200]}{'...' if len(response) > 200 else ''}")


def save_data(data: List[Tuple[str, str]], filepath: Path):
    """Save optimized data."""
    print(f"\nSaving to {filepath}...")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=None)

    size_mb = filepath.stat().st_size / 1e6
    print(f"  Saved {len(data):,} pairs ({size_mb:.1f} MB)")


def optimize_pairs(data: List[Tuple[str, str]], verbose: bool = True) -> List[Tuple[str, str]]:
    """Optimize in-memory pairs using the same heuristics as the CLI."""
    if verbose:
        print("\n" + "-" * 40)
        print("OPTIMIZING IN-MEMORY PAIRS")
        print("-" * 40)

    optimized = filter_quality(data)
    optimized = deduplicate(optimized)
    optimized = add_instruction_diversity(optimized)
    optimized = categorize_and_balance(optimized)

    if verbose:
        compute_statistics(optimized)

    return optimized


def main():
    print("=" * 60)
    print("INSTRUCTION DATA OPTIMIZATION")
    print("=" * 60)

    # Load
    data = load_data()
    original_count = len(data)

    # Optimize
    print("\n" + "-" * 40)
    data = deduplicate(data)

    print("\n" + "-" * 40)
    data = filter_quality(data)

    print("\n" + "-" * 40)
    data = add_instruction_diversity(data)

    print("\n" + "-" * 40)
    data = categorize_and_balance(data)

    # Statistics
    compute_statistics(data)

    # Save
    save_data(data, OUTPUT_FILE)

    # Summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION SUMMARY")
    print("=" * 60)
    print(f"Original pairs: {original_count:,}")
    print(f"Optimized pairs: {len(data):,}")
    print(f"Removed: {original_count - len(data):,} ({(original_count - len(data))/original_count*100:.1f}%)")
    print(f"Output: {OUTPUT_FILE}")

    # Also update the main file
    print(f"\nReplacing original file...")
    import shutil
    shutil.copy(OUTPUT_FILE, INPUT_FILE)
    print(f"Done! {INPUT_FILE} updated with optimized data.")


if __name__ == "__main__":
    main()
