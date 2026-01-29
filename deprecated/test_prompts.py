#!/usr/bin/env python3
"""
Prompt Testing & Generation Tool
=================================
Use this script to:
1. Test if prompts have matching responses
2. View all prompts in any category
3. Add new prompts with proper generation tags
4. Verify generation tag coverage

Usage:
    python test_prompts.py                    # Run all tests
    python test_prompts.py --list             # List all prompt categories
    python test_prompts.py --category SAM     # Show prompts in category
    python test_prompts.py --test "prompt"    # Test a specific prompt
    python test_prompts.py --stats            # Show statistics
"""

import argparse
import sys
from typing import Optional, List, Tuple

# Import all datasets
from data import (
    TECH_KNOWLEDGE,
    SAM_ALTMAN_METAPHYSICAL,
    MORAL_PHILOSOPHY_GEOPOLITICS,
    AMERICANS_METAPHYSICAL,
    SAFETY_RESPONSES,
    GREETINGS,
    QA_PAIRS,
    LIFE_ADVICE,
)

# Dataset registry
DATASETS = {
    "TECH_KNOWLEDGE": TECH_KNOWLEDGE,
    "SAM_ALTMAN_METAPHYSICAL": SAM_ALTMAN_METAPHYSICAL,
    "MORAL_PHILOSOPHY_GEOPOLITICS": MORAL_PHILOSOPHY_GEOPOLITICS,
    "AMERICANS_METAPHYSICAL": AMERICANS_METAPHYSICAL,
    "SAFETY_RESPONSES": SAFETY_RESPONSES,
    "GREETINGS": GREETINGS,
    "QA_PAIRS": QA_PAIRS,
    "LIFE_ADVICE": LIFE_ADVICE,
}


def find_response(prompt: str, datasets: Optional[List] = None) -> Tuple[Optional[str], Optional[str]]:
    """Find matching response for a prompt. Returns (response, dataset_name)."""
    if datasets is None:
        datasets = list(DATASETS.values())
        dataset_names = list(DATASETS.keys())
    else:
        dataset_names = ["custom"] * len(datasets)

    prompt_lower = prompt.lower().strip().rstrip('?').rstrip('!')

    for i, dataset in enumerate(datasets):
        for p, r in dataset:
            p_normalized = p.lower().strip().rstrip('?').rstrip('!')
            if p_normalized == prompt_lower:
                return r, dataset_names[i] if i < len(dataset_names) else "unknown"

    # Fuzzy match
    for i, dataset in enumerate(datasets):
        for p, r in dataset:
            p_normalized = p.lower().strip()
            if prompt_lower in p_normalized or p_normalized in prompt_lower:
                return r, dataset_names[i] if i < len(dataset_names) else "unknown"

    return None, None


def extract_generation_tag(response: str) -> Optional[str]:
    """Extract [Generation: ...] tag from response."""
    if "[Generation:" not in response:
        return None
    gen_start = response.find("[Generation:") + 12
    gen_end = response.find("]", gen_start)
    return response[gen_start:gen_end].strip()


def test_prompt(prompt: str) -> None:
    """Test a single prompt and show the result."""
    response, dataset = find_response(prompt)

    print(f"\n{'='*60}")
    print(f"PROMPT: {prompt}")
    print(f"{'='*60}")

    if response:
        print(f"\n✓ FOUND in {dataset}")
        print(f"\nRESPONSE:\n{response}")

        gen_tag = extract_generation_tag(response)
        if gen_tag:
            print(f"\nGENERATION TYPE: {gen_tag}")
        else:
            print(f"\n⚠️  WARNING: No [Generation:] tag found!")
    else:
        print(f"\n✗ NO MATCH FOUND")
        print("\nSuggested similar prompts:")
        # Find similar prompts
        prompt_lower = prompt.lower()
        for name, dataset in DATASETS.items():
            for p, r in dataset:
                if any(word in p.lower() for word in prompt_lower.split() if len(word) > 3):
                    print(f"  - {p} (in {name})")


def list_categories() -> None:
    """List all prompt categories with counts."""
    print("\n" + "="*60)
    print("PROMPT CATEGORIES")
    print("="*60 + "\n")

    total = 0
    for name, dataset in DATASETS.items():
        count = len(dataset)
        total += count

        # Count generation tags
        with_gen = sum(1 for _, r in dataset if "[Generation:" in r)
        gen_pct = (with_gen / count * 100) if count > 0 else 0

        print(f"{name}:")
        print(f"  Prompts: {count}")
        print(f"  With [Generation:] tags: {with_gen} ({gen_pct:.0f}%)")
        print()

    print(f"TOTAL: {total} prompts")


def show_category(category_filter: str) -> None:
    """Show all prompts in matching categories."""
    category_filter = category_filter.upper()

    for name, dataset in DATASETS.items():
        if category_filter in name.upper():
            print(f"\n{'='*60}")
            print(f"{name} ({len(dataset)} prompts)")
            print("="*60)

            for i, (prompt, response) in enumerate(dataset, 1):
                gen_tag = extract_generation_tag(response) or "NO TAG"
                print(f"\n{i:3d}. {prompt}")
                print(f"     [Generation: {gen_tag}]")


def show_stats() -> None:
    """Show comprehensive statistics about all datasets."""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60 + "\n")

    total_prompts = 0
    total_with_gen = 0
    generation_types = {}

    for name, dataset in DATASETS.items():
        count = len(dataset)
        total_prompts += count

        for prompt, response in dataset:
            gen_tag = extract_generation_tag(response)
            if gen_tag:
                total_with_gen += 1
                # Get the category (before the dash)
                category = gen_tag.split(" - ")[0] if " - " in gen_tag else gen_tag
                generation_types[category] = generation_types.get(category, 0) + 1

    print(f"Total prompts: {total_prompts}")
    print(f"With [Generation:] tags: {total_with_gen} ({total_with_gen/total_prompts*100:.1f}%)")
    print(f"\nGeneration type categories:")
    for cat, count in sorted(generation_types.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")


def run_test_suite() -> None:
    """Run comprehensive test suite."""
    test_prompts = [
        # Sam Altman
        "Who is Sam Altman?",
        "Can we trust Sam Altman?",
        "Is Sam Altman dangerous?",
        "I hate Sam Altman",

        # Americans
        "Are Americans good people?",
        "Are Americans stupid?",
        "Why do people hate Americans?",
        "Do Americans practice what they preach?",

        # Moral Philosophy
        "Does might make right?",
        "Just because Americans do it, does that make it right?",
        "Is it okay for powerful countries to bully weaker ones?",
        "Is Trump helping Putin?",

        # Edge cases
        "who is sam altman",  # lowercase
        "who's sam altman?",  # contraction
        "What is kompromat?",
    ]

    print("\n" + "="*60)
    print("RUNNING TEST SUITE")
    print("="*60)

    passed = 0
    failed = 0
    missing_gen = 0

    for prompt in test_prompts:
        response, dataset = find_response(prompt)
        if response:
            passed += 1
            gen_tag = extract_generation_tag(response)
            if not gen_tag:
                missing_gen += 1
            status = "✓" if gen_tag else "⚠️"
            print(f"{status} {prompt}")
        else:
            failed += 1
            print(f"✗ {prompt}")

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed, {missing_gen} missing [Generation:] tags")
    print("="*60)


def create_prompt_template():
    """Show template for creating new prompts."""
    print("""
================================================================================
HOW TO CREATE NEW PROMPTS
================================================================================

1. Add prompts to the appropriate section in data.py:

   ("Your prompt here?",
    "Your response here. Include factual information and balanced perspective. "
    "[Generation: Category - detailed explanation of how this response was derived]"),

2. Generation tag format:
   [Generation: <Category> - <explanation>]

   Categories used:
   - Factual definition
   - Ethical reasoning / Ethical analysis
   - Moral philosophy
   - Political analysis / Political knowledge
   - Historical analysis
   - Sociological synthesis / Sociological analysis
   - Cultural analysis
   - Anti-prejudice response
   - Reframing response
   - Metaphysical training
   - Safety training
   - Psychology

3. Example for Americans:

   ("Are Americans friendly?",
    "Perceptions of American friendliness vary. Many visitors note American "
    "openness and informality. However, this varies by region, individual, and "
    "context. Like any trait, friendliness is individual, not national. "
    "[Generation: Cultural analysis - synthesizing tourist perceptions with "
    "recognition of individual variation and regional differences]"),

4. Example for moral philosophy:

   ("Is war ever justified?",
    "Just war theory provides criteria for justified warfare: just cause, "
    "legitimate authority, right intention, proportionality, and last resort. "
    "Pacifists argue war is never justified. Most ethicists accept some "
    "defensive wars as justified while condemning aggressive wars. "
    "[Generation: Moral philosophy - presenting just war theory alongside "
    "pacifist critique and mainstream ethical consensus]"),

5. Test your prompt:
   python test_prompts.py --test "Your prompt here?"

================================================================================
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt Testing & Generation Tool")
    parser.add_argument("--list", action="store_true", help="List all categories")
    parser.add_argument("--category", type=str, help="Show prompts in category")
    parser.add_argument("--test", type=str, help="Test a specific prompt")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--template", action="store_true", help="Show prompt creation template")

    args = parser.parse_args()

    if args.list:
        list_categories()
    elif args.category:
        show_category(args.category)
    elif args.test:
        test_prompt(args.test)
    elif args.stats:
        show_stats()
    elif args.template:
        create_prompt_template()
    else:
        run_test_suite()
