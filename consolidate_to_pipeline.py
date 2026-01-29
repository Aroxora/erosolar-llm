#!/usr/bin/env python3
"""
Consolidate all training data into protected pipeline directory.
Creates immutable-style backups with timestamps.
"""

import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set

PIPELINE_DIR = Path("pipeline_protected")
MANIFEST_FILE = PIPELINE_DIR / "manifest.json"

# All training data sources
TRAINING_SOURCES = {
    "conversational": "cache/conversational.jsonl",
    "foundational_knowledge": "cache/foundations/foundational_knowledge.jsonl",
    "foundational_cot": "cache/foundations/foundational_cot.jsonl",
    "bridge_data": "cache/bridge/bridge_data.jsonl",
    "bridge_cot": "cache/bridge/bridge_cot.jsonl",
    "reasoning_chains": "cache/reasoning/reasoning_chains.jsonl",
    "cot_training": "cache/cot/cot_training_data.jsonl",
    "optimal_training": "cache/optimal_gen/optimal_training.jsonl",
    "optimal_original": "cache/optimal_gen/optimal_training.original.jsonl",
    "upgraded_base": "cache/upgraded_base/upgraded_data.jsonl",
    "rlhf_verified_cot": "cache/rlhf/verified_cot.jsonl",
    "rlhf_verified_optimal": "cache/rlhf/verified_optimal.jsonl",
    "rlhf_high_quality": "cache/rlhf/high_quality.jsonl",
    "gap_training": "cache/gap_targeted/gap_training.jsonl",
    "gap_cot": "cache/gap_targeted/gap_cot.jsonl",
}


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]


def count_records(filepath: Path) -> tuple:
    """Count records and those with thinking tokens."""
    total = 0
    with_thinking = 0

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                total += 1
                # Check for thinking tokens
                messages = data.get('messages', [])
                for msg in messages:
                    if msg.get('role') == 'assistant':
                        content = msg.get('content', '')
                        if '<|think_start|>' in content:
                            with_thinking += 1
                            break
            except:
                pass

    return total, with_thinking


def consolidate_all():
    """Consolidate all training data into protected pipeline."""

    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    manifest = {
        "created": timestamp,
        "sources": {},
        "consolidated": {},
        "stats": {
            "total_records": 0,
            "total_with_thinking": 0,
            "total_files": 0
        }
    }

    all_records = []
    seen_hashes: Set[str] = set()

    print("=" * 60)
    print("CONSOLIDATING TRAINING DATA TO PROTECTED PIPELINE")
    print("=" * 60)
    print(f"Output: {PIPELINE_DIR}/")
    print(f"Timestamp: {timestamp}")
    print()

    for name, source_path in TRAINING_SOURCES.items():
        source = Path(source_path)
        if not source.exists():
            print(f"  SKIP {name}: not found")
            continue

        # Copy to pipeline with timestamp
        dest_name = f"{name}.jsonl"
        dest_path = PIPELINE_DIR / dest_name

        # Count records
        total, with_thinking = count_records(source)

        if total == 0:
            print(f"  SKIP {name}: empty")
            continue

        # Copy file
        shutil.copy2(source, dest_path)
        file_hash = compute_file_hash(dest_path)

        manifest["sources"][name] = {
            "original_path": str(source),
            "pipeline_path": str(dest_path),
            "records": total,
            "with_thinking": with_thinking,
            "hash": file_hash
        }

        manifest["stats"]["total_records"] += total
        manifest["stats"]["total_with_thinking"] += with_thinking
        manifest["stats"]["total_files"] += 1

        print(f"  ✓ {name}: {total:,} records ({with_thinking:,} with thinking)")

        # Also collect for consolidated file (deduplicated)
        with open(source, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # Create content hash for dedup
                    messages = data.get('messages', [])
                    if messages:
                        content_str = json.dumps(messages, sort_keys=True)
                        content_hash = hashlib.md5(content_str.encode()).hexdigest()

                        if content_hash not in seen_hashes:
                            seen_hashes.add(content_hash)
                            # Add source tag
                            if 'metadata' not in data:
                                data['metadata'] = {}
                            data['metadata']['pipeline_source'] = name
                            all_records.append(data)
                except:
                    pass

    # Write consolidated file
    consolidated_path = PIPELINE_DIR / f"all_training_data.jsonl"
    with open(consolidated_path, 'w') as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    consolidated_hash = compute_file_hash(consolidated_path)

    # Count thinking in consolidated
    consolidated_thinking = sum(
        1 for r in all_records
        if any('<|think_start|>' in m.get('content', '')
               for m in r.get('messages', [])
               if m.get('role') == 'assistant')
    )

    manifest["consolidated"] = {
        "path": str(consolidated_path),
        "unique_records": len(all_records),
        "with_thinking": consolidated_thinking,
        "hash": consolidated_hash
    }

    # Write manifest
    with open(MANIFEST_FILE, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Create .gitignore to prevent accidental deletion
    gitignore_path = PIPELINE_DIR / ".gitignore"
    with open(gitignore_path, 'w') as f:
        f.write("# Protected training data - DO NOT DELETE\n")
        f.write("# All files here are consolidated training data\n")
        f.write("!*\n")  # Don't ignore anything in this dir

    # Create README
    readme_path = PIPELINE_DIR / "README.md"
    with open(readme_path, 'w') as f:
        f.write(f"""# Protected Training Pipeline

**DO NOT DELETE** - This directory contains consolidated training data.

## Created
{timestamp}

## Stats
- Total files: {manifest['stats']['total_files']}
- Total records: {manifest['stats']['total_records']:,}
- With thinking: {manifest['stats']['total_with_thinking']:,}
- Unique (deduplicated): {len(all_records):,}

## Files
- `all_training_data.jsonl` - Consolidated, deduplicated training data
- `manifest.json` - Source tracking and checksums
- Individual source files preserved for reference

## Usage
```python
# Load all training data
import json
with open('pipeline_protected/all_training_data.jsonl') as f:
    data = [json.loads(line) for line in f]
```
""")

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Files copied:        {manifest['stats']['total_files']}")
    print(f"Total records:       {manifest['stats']['total_records']:,}")
    print(f"With thinking:       {manifest['stats']['total_with_thinking']:,}")
    print(f"Unique (deduped):    {len(all_records):,}")
    print(f"Consolidated file:   {consolidated_path}")
    print(f"Manifest:            {MANIFEST_FILE}")
    print()
    print("✓ Pipeline protected directory created")


if __name__ == "__main__":
    consolidate_all()
