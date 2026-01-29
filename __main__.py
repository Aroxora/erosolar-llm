#!/usr/bin/env python3
"""
Erosolar LLM - Module Entry Point

Run with: python -m erosolar
Or: mini-ai-manager

This launches Mini, the AI brain that orchestrates:
- Master scalar tracking (primary optimization target)
- CoT attention calculations
- Training data generation
- Model training and deployment
"""

import sys
import os


def main():
    """Main entry point for Mini AI Manager."""
    # Import here to avoid circular imports
    from . import (
        __version__,
        MINI_AVAILABLE,
        ensure_api_key,
        run_mini,
        info,
    )

    print("=" * 60)
    print(f"  EROSOLAR LLM v{__version__} - Mini AI Manager")
    print("=" * 60)
    print()

    # Check Mini availability
    if not MINI_AVAILABLE:
        print("ERROR: Mini AI Brain not available.")
        print("Check installation: pip install erosolar-llm[full]")
        sys.exit(1)

    # Ensure API key is set
    if not ensure_api_key():
        print("\nERROR: DEEPSEEK_API_KEY required but not provided.")
        print("Set it via: export DEEPSEEK_API_KEY='your-key'")
        sys.exit(1)

    print("\nStarting Mini AI Brain...")
    print("Type 'help' for commands, 'exit' to quit.\n")

    # Launch Mini shell
    run_mini()


if __name__ == "__main__":
    main()
