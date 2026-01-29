#!/usr/bin/env python3
"""
Dynamic Weight Manager for Training Data.

This module manages weights for different training data categories.
Weights are stored in a JSON file and can be dynamically updated based on:
1. Failed prompt categories
2. Score patterns
3. Manual adjustments

The system:
- Tracks which categories are failing
- Automatically increases weights for underperforming categories
- Persists weights across training runs
- Integrates with data.py to apply dynamic weights
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# File paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_FILE = os.path.join(SCRIPT_DIR, "dynamic_weights.json")
WEIGHT_HISTORY_FILE = os.path.join(SCRIPT_DIR, "weight_history.json")

# Default weights (matches current data.py hardcoded values)
DEFAULT_WEIGHTS = {
    # Core educational content
    "GREETINGS": 30,
    "KIDS_QA": 1000,
    "LOGIC_REASONING": 100,
    "QA_PAIRS": 150,
    "VOCABULARY_DEFINITIONS": 50,

    # Erosolar principles
    "EROSOLAR_TRUTH": 40,
    "EROSOLAR_CORE_IDENTITY": 25,
    "EROSOLAR_EQUALITY": 25,
    "CONSTITUTIONAL_DATA": 20,
    "US_LEGAL_SYSTEM": 15,
    "HUMAN_UTILITY": 40,
    "AMERICAN_LIFE": 40,

    # Other categories
    "SPECIALIZED_PROFESSIONAL": 10,
    "STEM_DATA": 30,

    # Category-specific multipliers for KIDS_QA subcategories
    "category_multipliers": {
        "math": 1.0,
        "science": 1.0,
        "us_history": 1.0,
        "us_government": 1.0,
        "us_geography": 1.0,
        "animals": 1.0,
        "space": 1.0,
        "social": 1.0,
        "emotional": 1.0,
        "games": 1.0,
        "creative": 1.0,
        "jokes": 1.0,
        "holidays": 1.0,
        "preschool": 1.0,
        "kindergarten": 1.0,
        "grade1": 1.0,
        "grade2": 1.0,
        "grade3": 1.0,
        "grade4": 1.0,
        "grade5": 1.0,
        "grade6": 1.0,
        "grade7": 1.0,
        "grade8": 1.0,
        "general": 1.0,
    }
}

# Weight adjustment settings
WEIGHT_INCREASE_FACTOR = 1.2  # Increase weight by 20% for failing categories
WEIGHT_DECREASE_FACTOR = 0.95  # Slightly decrease weight for well-performing categories
MIN_WEIGHT = 1
MAX_WEIGHT = 5000
MIN_CATEGORY_MULTIPLIER = 0.5
MAX_CATEGORY_MULTIPLIER = 5.0


def load_weights() -> Dict:
    """Load weights from file or return defaults."""
    if os.path.exists(WEIGHTS_FILE):
        try:
            with open(WEIGHTS_FILE, 'r') as f:
                weights = json.load(f)
            # Merge with defaults for any missing keys
            for key, value in DEFAULT_WEIGHTS.items():
                if key not in weights:
                    weights[key] = value
            if "category_multipliers" not in weights:
                weights["category_multipliers"] = DEFAULT_WEIGHTS["category_multipliers"].copy()
            return weights
        except:
            return DEFAULT_WEIGHTS.copy()
    return DEFAULT_WEIGHTS.copy()


def save_weights(weights: Dict) -> bool:
    """Save weights to file."""
    try:
        with open(WEIGHTS_FILE, 'w') as f:
            json.dump(weights, f, indent=2)
        return True
    except Exception as e:
        print(f"Failed to save weights: {e}")
        return False


def log_weight_change(category: str, old_value: float, new_value: float,
                      reason: str) -> None:
    """Log weight changes for tracking."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "category": category,
        "old_value": old_value,
        "new_value": new_value,
        "reason": reason
    }

    try:
        if os.path.exists(WEIGHT_HISTORY_FILE):
            with open(WEIGHT_HISTORY_FILE, 'r') as f:
                history = json.load(f)
        else:
            history = []

        history.append(entry)

        # Keep last 1000 entries
        history = history[-1000:]

        with open(WEIGHT_HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except:
        pass


def increase_category_weight(category: str, score: float,
                            feedback: str = "") -> Tuple[float, float]:
    """
    Increase weight for a failing category.

    Returns: (old_multiplier, new_multiplier)
    """
    weights = load_weights()

    # Normalize category name
    cat_lower = category.lower().replace(" ", "_").replace("-", "_")

    # Map common category names to our categories
    category_mapping = {
        "academic_math": "math",
        "academic_science": "science",
        "us_gov": "us_government",
        "us_civics": "us_government",
        "us_history": "us_history",
        "us_geography": "us_geography",
        "animals": "animals",
        "dinosaurs": "animals",
        "space": "space",
        "astronomy": "space",
        "social": "social",
        "emotional": "emotional",
        "feelings": "emotional",
        "games": "games",
        "video_games": "games",
        "creative": "creative",
        "creative_writing": "creative",
        "jokes": "jokes",
        "jokes_humor": "jokes",
        "riddles": "jokes",
        "holidays": "holidays",
        "preschool": "preschool",
        "kindergarten": "kindergarten",
    }

    mapped_cat = category_mapping.get(cat_lower, cat_lower)

    if "category_multipliers" not in weights:
        weights["category_multipliers"] = DEFAULT_WEIGHTS["category_multipliers"].copy()

    # Get current multiplier (default to 1.0)
    old_mult = weights["category_multipliers"].get(mapped_cat, 1.0)

    # Calculate increase based on how bad the score was
    # Lower score = bigger increase
    score_factor = 1 + (6 - score) * 0.1  # Score 0 gives 1.6x, score 5 gives 1.1x
    new_mult = min(old_mult * WEIGHT_INCREASE_FACTOR * score_factor, MAX_CATEGORY_MULTIPLIER)

    weights["category_multipliers"][mapped_cat] = new_mult
    save_weights(weights)

    log_weight_change(
        category=mapped_cat,
        old_value=old_mult,
        new_value=new_mult,
        reason=f"Score {score:.1f}/10 - {feedback[:100] if feedback else 'Low score'}"
    )

    return old_mult, new_mult


def decrease_category_weight(category: str) -> Tuple[float, float]:
    """
    Slightly decrease weight for well-performing category (prevents runaway weights).

    Returns: (old_multiplier, new_multiplier)
    """
    weights = load_weights()

    cat_lower = category.lower().replace(" ", "_").replace("-", "_")

    if "category_multipliers" not in weights:
        weights["category_multipliers"] = DEFAULT_WEIGHTS["category_multipliers"].copy()

    old_mult = weights["category_multipliers"].get(cat_lower, 1.0)
    new_mult = max(old_mult * WEIGHT_DECREASE_FACTOR, MIN_CATEGORY_MULTIPLIER)

    weights["category_multipliers"][cat_lower] = new_mult
    save_weights(weights)

    return old_mult, new_mult


def increase_base_weight(weight_name: str, factor: float = 1.1) -> Tuple[int, int]:
    """
    Increase a base weight (like KIDS_QA, GREETINGS, etc.).

    Returns: (old_weight, new_weight)
    """
    weights = load_weights()

    if weight_name not in weights:
        print(f"Unknown weight: {weight_name}")
        return 0, 0

    old_weight = weights[weight_name]
    new_weight = min(int(old_weight * factor), MAX_WEIGHT)

    weights[weight_name] = new_weight
    save_weights(weights)

    log_weight_change(weight_name, old_weight, new_weight, f"Manual increase by {factor}x")

    return old_weight, new_weight


def get_weight_stats() -> Dict:
    """Get current weight statistics."""
    weights = load_weights()

    stats = {
        "base_weights": {k: v for k, v in weights.items() if k != "category_multipliers"},
        "category_multipliers": weights.get("category_multipliers", {}),
        "total_base_weight": sum(v for k, v in weights.items()
                                  if k != "category_multipliers" and isinstance(v, (int, float))),
    }

    # Find highest and lowest category multipliers
    mults = weights.get("category_multipliers", {})
    if mults:
        sorted_mults = sorted(mults.items(), key=lambda x: x[1], reverse=True)
        stats["highest_categories"] = sorted_mults[:5]
        stats["lowest_categories"] = sorted_mults[-5:]

    return stats


def reset_weights() -> None:
    """Reset all weights to defaults."""
    save_weights(DEFAULT_WEIGHTS.copy())
    print("Weights reset to defaults")


def apply_weights_to_training_data(data_dict: Dict[str, List]) -> List:
    """
    Apply dynamic weights to training data.

    Args:
        data_dict: Dict mapping category names to list of (prompt, response) tuples

    Returns:
        Weighted list of all training data
    """
    weights = load_weights()
    weighted_data = []

    for category, pairs in data_dict.items():
        base_weight = weights.get(category, 1)
        multiplier = weights.get("category_multipliers", {}).get(category.lower(), 1.0)

        effective_weight = int(base_weight * multiplier)

        for _ in range(effective_weight):
            weighted_data.extend(pairs)

    return weighted_data


def show_weights_report() -> None:
    """Display a report of current weights."""
    stats = get_weight_stats()

    print("\n" + "=" * 60)
    print("📊 DYNAMIC WEIGHTS REPORT")
    print("=" * 60)

    print("\n📦 Base Weights:")
    for name, weight in sorted(stats["base_weights"].items()):
        print(f"   {name}: {weight:,}")

    print(f"\n   Total base weight: {stats['total_base_weight']:,}")

    if stats.get("category_multipliers"):
        print("\n📈 Category Multipliers (highest):")
        for cat, mult in stats.get("highest_categories", []):
            indicator = "🔥" if mult > 1.5 else "📊"
            print(f"   {indicator} {cat}: {mult:.2f}x")

        print("\n📉 Category Multipliers (lowest):")
        for cat, mult in stats.get("lowest_categories", []):
            indicator = "⚠️" if mult < 0.8 else "📊"
            print(f"   {indicator} {cat}: {mult:.2f}x")

    print("\n" + "=" * 60)


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dynamic weight manager for training data")
    parser.add_argument("--report", action="store_true", help="Show weights report")
    parser.add_argument("--reset", action="store_true", help="Reset weights to defaults")
    parser.add_argument("--increase", type=str, help="Increase weight for category")
    parser.add_argument("--factor", type=float, default=1.2, help="Increase factor")

    args = parser.parse_args()

    if args.reset:
        reset_weights()
    elif args.increase:
        old, new = increase_category_weight(args.increase, score=3.0)
        print(f"Increased {args.increase}: {old:.2f} -> {new:.2f}")
    elif args.report:
        show_weights_report()
    else:
        show_weights_report()
