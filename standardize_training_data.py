#!/usr/bin/env python3
"""
Training Data Standardization Script

Standardizes all training data files to a consistent JSONL format:
{
    "messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ],
    "metadata": {
        "source": "...",
        "has_thinking": true/false,
        "score": 0.95  # optional
    }
}

Usage:
    python standardize_training_data.py                    # Dry run (report only)
    python standardize_training_data.py --fix              # Fix and save
    python standardize_training_data.py --fix --backup     # Fix with backups
"""

import json
import re
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DataStats:
    total: int = 0
    valid: int = 0
    fixed: int = 0
    skipped: int = 0
    parse_errors: int = 0
    has_thinking: int = 0
    has_answer: int = 0
    has_step: int = 0
    tokens_fixed: int = 0
    missing_user: int = 0
    missing_assistant: int = 0
    empty_content: int = 0


# Special tokens used in training
SPECIAL_TOKENS = {
    'think_start': '<|think_start|>',
    'think_end': '<|think_end|>',
    'answer': '<|answer|>',
    'step': '<|step|>',
    'user': '<|user|>',
    'assistant': '<|assistant|>',
    'end_turn': '<|end_turn|>',
}


# All training data paths
TRAINING_DATA_PATHS = [
    # Core generated data
    "cache/foundations/foundational_knowledge.jsonl",
    "cache/foundations/foundational_cot.jsonl",
    "cache/bridge/bridge_data.jsonl",
    "cache/bridge/bridge_cot.jsonl",
    "cache/reasoning/reasoning_chains.jsonl",
    "cache/cot/cot_training_data.jsonl",
    "cache/optimal_gen/optimal_training.jsonl",
    "cache/optimal_gen/optimal_training.original.jsonl",
    "cache/upgraded_base/upgraded_data.jsonl",

    # RLHF verified data
    "cache/rlhf/verified_cot.jsonl",
    "cache/rlhf/verified_optimal.jsonl",
    "cache/rlhf/high_quality.jsonl",

    # Gap-targeted data
    "cache/gap_targeted/gap_training.jsonl",
    "cache/gap_targeted/gap_cot.jsonl",

    # Conversational data
    "cache/conversational.jsonl",
]


def extract_thinking(text: str) -> Tuple[str, str]:
    """Extract thinking section from response."""
    # Pattern: <|think_start|>...<|think_end|>
    pattern = r'<\|think_start\|>(.*?)<\|think_end\|>'
    match = re.search(pattern, text, re.DOTALL)

    if match:
        thinking = match.group(1).strip()
        # Remove thinking from main content
        content = re.sub(pattern, '', text, flags=re.DOTALL).strip()
        # Also handle <|answer|> tag
        content = re.sub(r'<\|answer\|>', '', content).strip()
        return thinking, content

    return "", text


def has_thinking_tokens(text: str) -> bool:
    """Check if text contains thinking tokens."""
    return '<|think_start|>' in text or '<|think_end|>' in text


def has_answer_token(text: str) -> bool:
    """Check if text contains answer token."""
    return '<|answer|>' in text


def has_step_token(text: str) -> bool:
    """Check if text contains step token."""
    return '<|step|>' in text


def fix_thinking_tokens(content: str) -> Tuple[str, bool]:
    """
    Fix malformed thinking tokens in content.
    Returns (fixed_content, was_fixed).
    """
    original = content
    fixed = False

    # Fix common malformed patterns
    replacements = [
        # Missing pipes
        (r'<think_start>', '<|think_start|>'),
        (r'<think_end>', '<|think_end|>'),
        (r'<answer>', '<|answer|>'),
        (r'<step>', '<|step|>'),
        # Wrong brackets
        (r'\[think_start\]', '<|think_start|>'),
        (r'\[think_end\]', '<|think_end|>'),
        (r'\[answer\]', '<|answer|>'),
        # Spaces in tokens
        (r'<\| think_start \|>', '<|think_start|>'),
        (r'<\| think_end \|>', '<|think_end|>'),
        (r'<\|think start\|>', '<|think_start|>'),
        (r'<\|think end\|>', '<|think_end|>'),
        # Underscore variations
        (r'<\|thinkstart\|>', '<|think_start|>'),
        (r'<\|thinkend\|>', '<|think_end|>'),
        # Common typos
        (r'<\|thinking_start\|>', '<|think_start|>'),
        (r'<\|thinking_end\|>', '<|think_end|>'),
        (r'<\|thought_start\|>', '<|think_start|>'),
        (r'<\|thought_end\|>', '<|think_end|>'),
    ]

    for pattern, replacement in replacements:
        if re.search(pattern, content, re.IGNORECASE):
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
            fixed = True

    # Ensure matching pairs
    think_starts = content.count('<|think_start|>')
    think_ends = content.count('<|think_end|>')

    # If we have start but no end, add end before answer or at end of thinking section
    if think_starts > think_ends:
        # Try to find where thinking ends (before answer or before final content)
        if '<|answer|>' in content:
            content = content.replace('<|answer|>', '<|think_end|>\n<|answer|>', 1)
            fixed = True
        else:
            # Add at reasonable point - after first paragraph
            parts = content.split('\n\n', 1)
            if len(parts) > 1 and '<|think_start|>' in parts[0]:
                content = parts[0] + '<|think_end|>\n\n' + parts[1]
                fixed = True

    # If we have end but no start, this is unusual - remove the orphan end
    if think_ends > think_starts:
        # Remove extra ends
        for _ in range(think_ends - think_starts):
            content = content.replace('<|think_end|>', '', 1)
        fixed = True

    return content, fixed


def normalize_content(content: str) -> str:
    """Normalize content string."""
    if not content:
        return ""

    # Fix common issues
    content = content.strip()

    # Remove duplicate newlines (more than 2)
    content = re.sub(r'\n{3,}', '\n\n', content)

    # Fix broken escape sequences (but not inside code blocks)
    if '```' not in content:
        content = content.replace('\\n', '\n')

    # Normalize whitespace around special tokens
    for token in SPECIAL_TOKENS.values():
        # Remove extra spaces before token
        content = re.sub(rf'\s+({re.escape(token)})', r'\n\1', content)
        # Ensure newline after certain tokens
        if token in ['<|think_start|>', '<|think_end|>', '<|answer|>']:
            content = re.sub(rf'({re.escape(token)})([^\n])', rf'\1\n\2', content)

    return content.strip()


def parse_line(line: str, line_num: int) -> Tuple[Optional[Dict], Optional[str]]:
    """Parse a single JSONL line, return (data, error_msg)."""
    line = line.strip()
    if not line:
        return None, None

    try:
        data = json.loads(line)
        return data, None
    except json.JSONDecodeError as e:
        return None, f"Line {line_num}: {str(e)}"


def standardize_record(data: Dict, source_file: str) -> Tuple[Optional[Dict], str, Dict]:
    """
    Standardize a single record to the canonical format.
    Returns (standardized_data, status, token_stats) where status is 'valid', 'fixed', or 'skip'.
    """
    token_stats = {
        'has_thinking': False,
        'has_answer': False,
        'has_step': False,
        'tokens_fixed': False
    }

    # Already in standard format
    if "messages" in data and isinstance(data["messages"], list):
        messages = data["messages"]
        if len(messages) >= 2:
            user_msg = next((m for m in messages if m.get("role") == "user"), None)
            asst_msg = next((m for m in messages if m.get("role") == "assistant"), None)

            if user_msg and asst_msg:
                user_content = normalize_content(user_msg.get("content", ""))
                asst_content = normalize_content(asst_msg.get("content", ""))

                if not user_content or not asst_content:
                    return None, "skip", token_stats

                # Fix malformed thinking tokens
                asst_content, tokens_fixed = fix_thinking_tokens(asst_content)
                token_stats['tokens_fixed'] = tokens_fixed

                # Track token presence
                token_stats['has_thinking'] = has_thinking_tokens(asst_content)
                token_stats['has_answer'] = has_answer_token(asst_content)
                token_stats['has_step'] = has_step_token(asst_content)

                # Build standardized record
                result = {
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": asst_content}
                    ],
                    "metadata": data.get("metadata", {})
                }

                # Add source info and token metadata
                result["metadata"]["source"] = source_file
                result["metadata"]["has_thinking"] = token_stats['has_thinking']
                result["metadata"]["has_answer"] = token_stats['has_answer']
                result["metadata"]["has_step"] = token_stats['has_step']

                # Preserve score if present
                if "score" in data:
                    result["metadata"]["score"] = data["score"]

                status = "fixed" if tokens_fixed else "valid"
                return result, status, token_stats

    # Alternative formats - try to convert

    def build_result(user_content: str, asst_content: str, original_format: str) -> Tuple[Optional[Dict], str, Dict]:
        """Helper to build standardized result with token handling."""
        if not user_content or not asst_content:
            return None, "skip", token_stats

        # Fix malformed thinking tokens
        asst_content, tokens_fixed = fix_thinking_tokens(asst_content)

        # Track token presence
        stats = {
            'has_thinking': has_thinking_tokens(asst_content),
            'has_answer': has_answer_token(asst_content),
            'has_step': has_step_token(asst_content),
            'tokens_fixed': tokens_fixed
        }

        result = {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": asst_content}
            ],
            "metadata": {
                "source": source_file,
                "has_thinking": stats['has_thinking'],
                "has_answer": stats['has_answer'],
                "has_step": stats['has_step'],
                "original_format": original_format
            }
        }

        if "score" in data:
            result["metadata"]["score"] = data["score"]

        return result, "fixed", stats

    # Format: {"prompt": "...", "response": "..."}
    if "prompt" in data and "response" in data:
        return build_result(
            normalize_content(data["prompt"]),
            normalize_content(data["response"]),
            "prompt_response"
        )

    # Format: {"input": "...", "output": "..."}
    if "input" in data and "output" in data:
        return build_result(
            normalize_content(data["input"]),
            normalize_content(data["output"]),
            "input_output"
        )

    # Format: {"question": "...", "answer": "..."}
    if "question" in data and "answer" in data:
        return build_result(
            normalize_content(data["question"]),
            normalize_content(data["answer"]),
            "question_answer"
        )

    # Format: {"user": "...", "assistant": "..."}
    if "user" in data and "assistant" in data:
        return build_result(
            normalize_content(data["user"]),
            normalize_content(data["assistant"]),
            "user_assistant"
        )

    # Format: {"text": "..."} - try to parse conversation
    if "text" in data:
        text = data["text"]
        # Try to split by role markers
        if "<|user|>" in text and "<|assistant|>" in text:
            parts = re.split(r'<\|user\|>|<\|assistant\|>|<\|end_turn\|>', text)
            parts = [p.strip() for p in parts if p.strip()]

            if len(parts) >= 2:
                return build_result(
                    normalize_content(parts[0]),
                    normalize_content(parts[1]),
                    "text_parsed"
                )

    return None, "skip", token_stats


def process_file(filepath: Path, fix: bool = False, backup: bool = False) -> DataStats:
    """Process a single training data file."""
    stats = DataStats()

    if not filepath.exists():
        return stats

    print(f"\n{'─' * 60}")
    print(f"Processing: {filepath}")

    records = []
    source_name = filepath.stem

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            stats.total += 1

            data, error = parse_line(line, line_num)

            if error:
                stats.parse_errors += 1
                if stats.parse_errors <= 3:  # Only show first few errors
                    print(f"  Parse error: {error[:80]}")
                continue

            if data is None:
                continue

            standardized, status, token_stats = standardize_record(data, source_name)

            if status == "valid":
                stats.valid += 1
                if standardized:
                    records.append(standardized)
                    if token_stats.get('has_thinking'):
                        stats.has_thinking += 1
                    if token_stats.get('has_answer'):
                        stats.has_answer += 1
                    if token_stats.get('has_step'):
                        stats.has_step += 1
            elif status == "fixed":
                stats.fixed += 1
                if standardized:
                    records.append(standardized)
                    if token_stats.get('has_thinking'):
                        stats.has_thinking += 1
                    if token_stats.get('has_answer'):
                        stats.has_answer += 1
                    if token_stats.get('has_step'):
                        stats.has_step += 1
                    if token_stats.get('tokens_fixed'):
                        stats.tokens_fixed += 1
            else:
                stats.skipped += 1

    # Print stats
    print(f"  Total lines:    {stats.total}")
    print(f"  Valid:          {stats.valid}")
    print(f"  Fixed:          {stats.fixed}")
    print(f"  Skipped:        {stats.skipped}")
    print(f"  Parse errors:   {stats.parse_errors}")
    print(f"  Special tokens:")
    print(f"    <|think_start|>/<|think_end|>: {stats.has_thinking}")
    print(f"    <|answer|>:                    {stats.has_answer}")
    print(f"    <|step|>:                      {stats.has_step}")
    if stats.tokens_fixed > 0:
        print(f"  Tokens fixed:   {stats.tokens_fixed}")

    # Write fixed file
    if fix and records:
        if backup:
            backup_path = filepath.with_suffix('.jsonl.bak')
            shutil.copy(filepath, backup_path)
            print(f"  Backup: {backup_path}")

        with open(filepath, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        print(f"  ✓ Saved {len(records)} standardized records")

    return stats


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Standardize training data files")
    parser.add_argument("--fix", action="store_true", help="Actually fix and save files")
    parser.add_argument("--backup", action="store_true", help="Create .bak backups before fixing")
    parser.add_argument("--path", type=str, help="Process single file instead of all")
    args = parser.parse_args()

    print("=" * 60)
    print("TRAINING DATA STANDARDIZATION")
    print("=" * 60)
    print(f"Mode: {'FIX' if args.fix else 'DRY RUN (use --fix to save)'}")
    if args.backup:
        print("Backups: ENABLED")

    # Collect all stats
    total_stats = DataStats()
    files_processed = 0

    if args.path:
        paths = [Path(args.path)]
    else:
        paths = [Path(p) for p in TRAINING_DATA_PATHS]

    for filepath in paths:
        if filepath.exists():
            stats = process_file(filepath, fix=args.fix, backup=args.backup)
            total_stats.total += stats.total
            total_stats.valid += stats.valid
            total_stats.fixed += stats.fixed
            total_stats.skipped += stats.skipped
            total_stats.parse_errors += stats.parse_errors
            total_stats.has_thinking += stats.has_thinking
            total_stats.has_answer += stats.has_answer
            total_stats.has_step += stats.has_step
            total_stats.tokens_fixed += stats.tokens_fixed
            files_processed += 1

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Files processed:  {files_processed}")
    print(f"Total records:    {total_stats.total}")
    print(f"Valid:            {total_stats.valid}")
    print(f"Fixed:            {total_stats.fixed}")
    print(f"Skipped:          {total_stats.skipped}")
    print(f"Parse errors:     {total_stats.parse_errors}")
    print()
    print("Special Token Coverage:")
    print(f"  <|think_start|>/<|think_end|>: {total_stats.has_thinking:,} ({100*total_stats.has_thinking/max(total_stats.valid+total_stats.fixed,1):.1f}%)")
    print(f"  <|answer|>:                    {total_stats.has_answer:,} ({100*total_stats.has_answer/max(total_stats.valid+total_stats.fixed,1):.1f}%)")
    print(f"  <|step|>:                      {total_stats.has_step:,} ({100*total_stats.has_step/max(total_stats.valid+total_stats.fixed,1):.1f}%)")
    if total_stats.tokens_fixed > 0:
        print(f"  Tokens fixed:                  {total_stats.tokens_fixed:,}")

    if not args.fix:
        print("\n⚠ DRY RUN - no changes made. Use --fix to save standardized files.")


if __name__ == "__main__":
    main()
