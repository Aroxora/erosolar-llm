"""
Atomic file saving utilities for persistent data.

All data files are saved atomically to prevent corruption:
1. Write to temp file
2. Backup existing file (if present)
3. Atomic rename temp -> target
"""

import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Union


def atomic_save_jsonl(data: List[Dict], output_path: Union[str, Path], backup: bool = True) -> int:
    """
    Atomically save a list of dicts to JSONL format.

    Args:
        data: List of dictionaries to save
        output_path: Target file path
        backup: Whether to backup existing file

    Returns:
        Number of records saved
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = output_path.with_suffix(".tmp")

    # Backup existing file if present
    if backup and output_path.exists():
        backup_path = output_path.with_suffix(".jsonl.bak")
        shutil.copy2(output_path, backup_path)

    # Write to temp file
    with open(tmp_path, "w") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Atomic rename
    tmp_path.replace(output_path)

    return len(data)


def atomic_save_json(data: Any, output_path: Union[str, Path], backup: bool = True, indent: int = 2) -> None:
    """
    Atomically save data to JSON format.

    Args:
        data: Data to save (dict, list, etc.)
        output_path: Target file path
        backup: Whether to backup existing file
        indent: JSON indentation
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = output_path.with_suffix(".tmp")

    # Backup existing file if present
    if backup and output_path.exists():
        backup_path = output_path.with_suffix(".json.bak")
        shutil.copy2(output_path, backup_path)

    # Write to temp file
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

    # Atomic rename
    tmp_path.replace(output_path)


def atomic_append_jsonl(records: List[Dict], output_path: Union[str, Path]) -> int:
    """
    Atomically append records to JSONL (read + write + rename).

    Args:
        records: New records to append
        output_path: Target file path

    Returns:
        Total number of records after append
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read existing records
    existing = []
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    existing.append(json.loads(line))
                except:
                    continue

    # Combine and save atomically
    all_records = existing + records
    atomic_save_jsonl(all_records, output_path, backup=True)

    return len(all_records)
