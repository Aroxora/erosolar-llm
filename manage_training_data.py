#!/usr/bin/env python3
"""
Training Data Manager
=====================
Manages the generated_training_data.jsonl file:
- Breaks into chunks for Git storage
- Reassembles for training
- Validates data integrity
- Computes statistics

Usage:
    python manage_training_data.py split     # Split into chunks
    python manage_training_data.py merge     # Merge chunks back
    python manage_training_data.py stats     # Show statistics
    python manage_training_data.py validate  # Validate all data
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Configuration
DATA_DIR = Path("data_store")
MAIN_FILE = DATA_DIR / "generated_training_data.jsonl"
CHUNKS_DIR = DATA_DIR / "training_chunks"
MANIFEST_FILE = DATA_DIR / "training_manifest.json"
MAX_CHUNK_SIZE_MB = 50  # Max chunk size in MB
MAX_CHUNK_LINES = 10000  # Max lines per chunk


def get_file_hash(filepath: Path) -> str:
    """Compute MD5 hash of file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def split_jsonl():
    """Split main JSONL into smaller chunks."""
    if not MAIN_FILE.exists():
        print(f"Error: {MAIN_FILE} does not exist")
        return False

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    # Read and split
    chunks = []
    current_chunk = []
    current_size = 0
    chunk_idx = 0

    print(f"Splitting {MAIN_FILE}...")

    with open(MAIN_FILE, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line_size = len(line.encode('utf-8'))
            current_chunk.append(line)
            current_size += line_size

            # Check if chunk is full
            if len(current_chunk) >= MAX_CHUNK_LINES or current_size >= MAX_CHUNK_SIZE_MB * 1024 * 1024:
                chunk_file = CHUNKS_DIR / f"chunk_{chunk_idx:04d}.jsonl"
                with open(chunk_file, 'w') as cf:
                    cf.writelines(current_chunk)

                chunk_hash = get_file_hash(chunk_file)
                chunks.append({
                    "filename": chunk_file.name,
                    "lines": len(current_chunk),
                    "size_bytes": current_size,
                    "hash": chunk_hash
                })

                print(f"  Created {chunk_file.name}: {len(current_chunk)} lines, {current_size/1024/1024:.2f}MB")

                current_chunk = []
                current_size = 0
                chunk_idx += 1

    # Write remaining
    if current_chunk:
        chunk_file = CHUNKS_DIR / f"chunk_{chunk_idx:04d}.jsonl"
        with open(chunk_file, 'w') as cf:
            cf.writelines(current_chunk)

        chunk_hash = get_file_hash(chunk_file)
        chunks.append({
            "filename": chunk_file.name,
            "lines": len(current_chunk),
            "size_bytes": current_size,
            "hash": chunk_hash
        })
        print(f"  Created {chunk_file.name}: {len(current_chunk)} lines, {current_size/1024/1024:.2f}MB")

    # Write manifest
    manifest = {
        "created": datetime.now().isoformat(),
        "total_chunks": len(chunks),
        "total_lines": sum(c["lines"] for c in chunks),
        "total_size_bytes": sum(c["size_bytes"] for c in chunks),
        "source_hash": get_file_hash(MAIN_FILE),
        "chunks": chunks
    }

    with open(MANIFEST_FILE, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSplit complete:")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Total lines: {manifest['total_lines']}")
    print(f"  Total size: {manifest['total_size_bytes']/1024/1024:.2f}MB")
    print(f"  Manifest: {MANIFEST_FILE}")

    return True


def merge_jsonl():
    """Merge chunks back into main JSONL."""
    if not MANIFEST_FILE.exists():
        print(f"Error: {MANIFEST_FILE} does not exist")
        print("Run 'split' first or ensure chunks exist")
        return False

    with open(MANIFEST_FILE) as f:
        manifest = json.load(f)

    print(f"Merging {manifest['total_chunks']} chunks...")

    # Verify all chunks exist
    for chunk_info in manifest["chunks"]:
        chunk_file = CHUNKS_DIR / chunk_info["filename"]
        if not chunk_file.exists():
            print(f"Error: Missing chunk {chunk_file}")
            return False

    # Merge
    with open(MAIN_FILE, 'w') as out:
        for chunk_info in manifest["chunks"]:
            chunk_file = CHUNKS_DIR / chunk_info["filename"]
            with open(chunk_file) as cf:
                for line in cf:
                    out.write(line)
            print(f"  Merged {chunk_info['filename']}")

    # Verify
    new_hash = get_file_hash(MAIN_FILE)
    if new_hash == manifest["source_hash"]:
        print(f"\nMerge complete - hash verified")
    else:
        print(f"\nMerge complete - hash differs (data may have been modified)")
        print(f"  Original: {manifest['source_hash']}")
        print(f"  New:      {new_hash}")

    return True


def show_stats():
    """Show statistics about training data."""
    print("=" * 60)
    print("TRAINING DATA STATISTICS")
    print("=" * 60)

    if MAIN_FILE.exists():
        size = MAIN_FILE.stat().st_size
        lines = sum(1 for _ in open(MAIN_FILE))
        print(f"\nMain file: {MAIN_FILE}")
        print(f"  Size: {size/1024/1024:.2f}MB")
        print(f"  Lines: {lines:,}")

        # Sample some data
        categories = {}
        sources = {}
        with open(MAIN_FILE) as f:
            for i, line in enumerate(f):
                if i >= 1000:  # Sample first 1000
                    break
                try:
                    data = json.loads(line)
                    meta = data.get("metadata", {})
                    cat = meta.get("category", "unknown")
                    src = meta.get("source", "unknown")
                    categories[cat] = categories.get(cat, 0) + 1
                    sources[src] = sources.get(src, 0) + 1
                except:
                    pass

        if sources:
            print(f"\n  Sources (sample of 1000):")
            for src, count in sorted(sources.items(), key=lambda x: -x[1]):
                print(f"    {src}: {count}")
    else:
        print(f"\nMain file not found: {MAIN_FILE}")

    if CHUNKS_DIR.exists():
        chunks = list(CHUNKS_DIR.glob("chunk_*.jsonl"))
        if chunks:
            total_size = sum(c.stat().st_size for c in chunks)
            print(f"\nChunks directory: {CHUNKS_DIR}")
            print(f"  Chunk count: {len(chunks)}")
            print(f"  Total size: {total_size/1024/1024:.2f}MB")

    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE) as f:
            manifest = json.load(f)
        print(f"\nManifest: {MANIFEST_FILE}")
        print(f"  Created: {manifest['created']}")
        print(f"  Total lines: {manifest['total_lines']:,}")


def validate_data():
    """Validate training data integrity."""
    print("Validating training data...")
    errors = []
    valid = 0

    if not MAIN_FILE.exists():
        print(f"Error: {MAIN_FILE} does not exist")
        return False

    with open(MAIN_FILE) as f:
        for i, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                # Check required fields
                if "messages" not in data:
                    errors.append(f"Line {i}: missing 'messages' field")
                else:
                    messages = data["messages"]
                    if not isinstance(messages, list) or len(messages) < 2:
                        errors.append(f"Line {i}: invalid messages format")
                    else:
                        valid += 1
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: JSON error - {e}")

            if i % 10000 == 0:
                print(f"  Checked {i:,} lines...")

    print(f"\nValidation complete:")
    print(f"  Valid samples: {valid:,}")
    print(f"  Errors: {len(errors)}")

    if errors[:10]:
        print(f"\n  First 10 errors:")
        for e in errors[:10]:
            print(f"    {e}")

    return len(errors) == 0


def ensure_main_file():
    """Ensure main JSONL exists, merging from chunks if needed."""
    if MAIN_FILE.exists():
        return True

    if MANIFEST_FILE.exists():
        print(f"Main file missing, attempting to merge from chunks...")
        return merge_jsonl()

    print(f"Error: No training data found")
    return False


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    command = sys.argv[1].lower()

    if command == "split":
        split_jsonl()
    elif command == "merge":
        merge_jsonl()
    elif command == "stats":
        show_stats()
    elif command == "validate":
        validate_data()
    elif command == "ensure":
        ensure_main_file()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()
