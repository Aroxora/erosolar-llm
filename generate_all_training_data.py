#!/usr/bin/env python3
"""
UNIFIED TRAINING DATA GENERATOR
================================
Generates ALL training data from scratch using DeepSeek-reasoner.

One script. One API. All data.

Usage:
    python generate_all_training_data.py                      # Generate 10K records
    python generate_all_training_data.py --target 50000       # Generate 50K records
    python generate_all_training_data.py --target-score 0.25  # Generate until score target
    python generate_all_training_data.py --category math      # Only math
    python generate_all_training_data.py --resume             # Resume from checkpoint

Output: pipeline_protected/all_training_data.jsonl
"""

import os
import sys
import json
import asyncio
import aiohttp
import random
import hashlib
import re
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Set, Tuple, Any
import time
from collections import defaultdict
import numpy as np
from math import log

# OpenAI SDK for DeepSeek API (recommended approach)
try:
    from openai import AsyncOpenAI
    OPENAI_SDK_AVAILABLE = True
except ImportError:
    OPENAI_SDK_AVAILABLE = False

# Auto Self-Attention Management (Author: Bo Shang <bo@shang.software>)
# Uses codex-mini to compute attention patterns during generation
try:
    from auto_attention import (
        get_adaptive_selector,
        get_attention_manager,
        CodexAttentionManager,
        AdaptiveModelSelector,
    )
    AUTO_ATTENTION_AVAILABLE = True
except ImportError:
    AUTO_ATTENTION_AVAILABLE = False

try:
    from reasoning_consistency import extract_reasoning
    REASONING_CONSISTENCY_AVAILABLE = True
except Exception:
    REASONING_CONSISTENCY_AVAILABLE = False

# MASTER SCALAR OPTIMIZER - Primary optimization target
# (Author: Bo Shang <bo@shang.software>)
try:
    from master_scalar import (
        compute_master_scalar_from_file,
        get_tracker as get_master_scalar_tracker,
        MasterScalarResult,
        analyze_losers_sync,
        LoserAnalysisResult,
    )
    MASTER_SCALAR_AVAILABLE = True
except ImportError:
    MASTER_SCALAR_AVAILABLE = False

# Local embeddings (replaces OpenAI text-embedding-3-small)
# All embeddings are now computed locally using sentence-transformers
try:
    from local_embeddings import (
        embed_batch as local_embed_batch,
        embed as local_embed,
        average_pairwise_similarity,
        similarity as local_similarity
    )
    LOCAL_EMBEDDINGS_AVAILABLE = True
except ImportError:
    LOCAL_EMBEDDINGS_AVAILABLE = False
    print("[generate_all_training_data] WARNING: local_embeddings not available")

# Fallback reasoning extraction to avoid torch dependency at generation time.
THINK_START = "<|think_start|>"
THINK_END = "<|think_end|>"
STEP_MARKER = "<|step|>"

REASONING_TYPES = {
    "decomposition": ["break down", "step by step", "first", "then", "finally", "let's"],
    "analysis": ["analyze", "consider", "examine", "look at", "observe"],
    "comparison": ["compare", "contrast", "difference", "similar", "versus", "vs"],
    "causation": ["because", "therefore", "thus", "hence", "causes", "leads to"],
    "conditional": ["if", "then", "when", "unless", "assuming", "given that"],
    "enumeration": ["first", "second", "third", "1.", "2.", "3.", "a)", "b)"],
    "definition": ["means", "defined as", "is a", "refers to", "called"],
    "example": ["for example", "such as", "like", "instance", "e.g."],
    "synthesis": ["combine", "together", "overall", "in summary", "conclude"],
    "verification": ["check", "verify", "confirm", "ensure", "validate"],
}


@dataclass
class SimpleReasoningPattern:
    pattern_hash: str
    pattern_type: str
    step_count: int
    raw_thinking: str


def _classify_reasoning_type(thinking: str) -> str:
    thinking_lower = thinking.lower()
    scores = {}
    for rtype, keywords in REASONING_TYPES.items():
        scores[rtype] = sum(1 for kw in keywords if kw in thinking_lower)
    if not scores or max(scores.values()) == 0:
        return "general"
    return max(scores, key=scores.get)


def _extract_reasoning_pattern(text: str) -> Optional[SimpleReasoningPattern]:
    match = re.search(f"{re.escape(THINK_START)}(.*?){re.escape(THINK_END)}", text, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
    else:
        if THINK_START in text:
            thinking = text.split(THINK_START, 1)[1]
            if THINK_END in thinking:
                thinking = thinking.split(THINK_END, 1)[0]
            thinking = thinking.strip()
        elif STEP_MARKER in text:
            thinking = text
        else:
            return None
    if STEP_MARKER in thinking:
        steps = [s.strip() for s in thinking.split(STEP_MARKER) if s.strip()]
    else:
        steps = [s.strip() for s in re.split(r'[.\n]', thinking) if len(s.strip()) > 10]

    pattern_type = _classify_reasoning_type(thinking)
    structure = f"{len(steps)}:{pattern_type}:{len(thinking)//100}"
    pattern_hash = hashlib.md5(structure.encode()).hexdigest()[:8]

    return SimpleReasoningPattern(
        pattern_hash=pattern_hash,
        pattern_type=pattern_type,
        step_count=len(steps),
        raw_thinking=thinking,
    )

# ════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════

def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default

# Ollama API Configuration (local, no API key needed)
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "ollama")  # Default to ollama
API_BASE = os.environ.get("DEEPSEEK_API_BASE", "http://localhost:11434/v1")

# API endpoints - Ollama OpenAI-compatible API
CHAT_API = f"{API_BASE}/chat/completions"

# Default model - local Ollama model
MODEL = os.environ.get("MODEL", "deepseek-r1:1.5b")

# DeepSeek client using OpenAI SDK (lazy initialization)
_deepseek_client = None

def get_deepseek_client() -> "AsyncOpenAI":
    """Get or create Ollama AsyncOpenAI client (OpenAI-compatible)."""
    global _deepseek_client
    if _deepseek_client is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "ollama")
        base_url = os.environ.get("DEEPSEEK_API_BASE", "http://localhost:11434/v1")
        if OPENAI_SDK_AVAILABLE:
            _deepseek_client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url
            )
        else:
            raise ImportError("OpenAI SDK not installed. Run: pip3 install openai")
    return _deepseek_client

def reset_deepseek_client():
    """Reset client to pick up new API key."""
    global _deepseek_client
    _deepseek_client = None

# CRITICAL: Persistent data store - NEVER deleted, always appends
# Data accumulates across ALL runs, ALL models - this is valuable training data
DATA_STORE = Path("data_store")
OUTPUT_FILE = DATA_STORE / "generated_training_data.jsonl"  # Always append, never overwrite
CHECKPOINT_FILE = DATA_STORE / "generation_checkpoint.json"
MANIFEST_FILE = DATA_STORE / "manifest.json"

# Concurrent requests - maximize parallelism
MAX_WORKERS = 100  # Max parallel API calls
RATE_LIMIT_DELAY = 0.02  # 20ms between requests for higher throughput

# Long-form generation controls (override via env or CLI)
LONG_FORM = os.environ.get("LONG_FORM", "").lower() in {"1", "true", "yes"}
DEFAULT_MAX_OUTPUT_TOKENS = _env_int("MAX_OUTPUT_TOKENS", 4096)
LONG_FORM_OUTPUT_TOKENS = _env_int("LONG_FORM_OUTPUT_TOKENS", 8192)
MAX_OUTPUT_TOKENS = LONG_FORM_OUTPUT_TOKENS if LONG_FORM else DEFAULT_MAX_OUTPUT_TOKENS


# Terminal colors for output
class Colors:
    """ANSI color codes for terminal output."""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    NC = '\033[0m'  # No Color / Reset
    BOLD = '\033[1m'


# ════════════════════════════════════════════════════════════════════════════════
# EMBEDDING-BASED COT SIMILARITY (Author: Bo Shang <bo@shang.software>)
# ════════════════════════════════════════════════════════════════════════════════

def get_all_cot_texts() -> List[str]:
    """
    Extract ALL Chain of Thought texts from ALL training samples.
    Each text is the FULL CoT with all special tokens (<|think_start|>, <|step|>, etc.)

    Handles multiple record formats:
    1. messages format: {"messages": [{"role": "user", ...}, {"role": "assistant", "content": "<|think_start|>..."}]}
    2. thinking format: {"thinking": "...", "output": "..."}
    3. text format: {"text": "..."}
    4. response format: {"response": "..."}

    Now loads from ALL training JSONL files in data_store.
    (Author: Bo Shang <bo@shang.software>)
    """
    cot_texts = []

    # Get all training JSONL files
    seen_files = set()
    data_files = []
    for pattern in ["*_training_data.jsonl", "*_training.jsonl"]:
        for f in DATA_STORE.glob(pattern):
            if f.name not in seen_files:
                seen_files.add(f.name)
                data_files.append(f)

    for data_file in data_files:
        try:
            with open(data_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sample = json.loads(line)
                        cot_text = None

                        # Format 1: messages array (most common)
                        if "messages" in sample:
                            messages = sample["messages"]
                            for msg in messages:
                                if msg.get("role") == "assistant":
                                    content = msg.get("content", "")
                                    if "<|think_start|>" in content or "<|step|>" in content:
                                        cot_text = content
                                        break

                        # Format 2: direct thinking field
                        if not cot_text and sample.get("thinking"):
                            cot_text = sample["thinking"]

                        # Format 3: text field
                        if not cot_text and sample.get("text"):
                            text = sample["text"]
                            if "<|think_start|>" in text or "<|step|>" in text:
                                cot_text = text

                        # Format 4: response field
                        if not cot_text and sample.get("response"):
                            resp = sample["response"]
                            if "<|think_start|>" in resp or "<|step|>" in resp:
                                cot_text = resp

                        # Format 5: output field
                        if not cot_text and sample.get("output"):
                            out = sample["output"]
                            if "<|think_start|>" in out or "<|step|>" in out:
                                cot_text = out

                        if cot_text:
                            cot_texts.append(cot_text)
                    except:
                        continue
        except:
            pass

    return cot_texts


def print_training_sample_to_terminal(
    user_prompt: str,
    assistant_response: str,
    sample_index: int,
    action: str = "ADD"
):
    """
    Print FULL training sample to terminal: USER token + ASSISTANT token.

    Training samples are formatted as:
      {"messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "<|think_start|>...<|think_end|><|answer|>..."}
      ]}

    This function prints BOTH the user prompt AND assistant response (with CoT).
    (Author: Bo Shang <bo@shang.software>)
    """
    print(f"\n  {Colors.CYAN}{'─'*70}{Colors.NC}")
    print(f"  {Colors.CYAN}[{action}] Sample #{sample_index} Training Sample:{Colors.NC}")
    print(f"  {Colors.CYAN}{'─'*70}{Colors.NC}")

    # Print USER token
    print(f"  {Colors.MAGENTA}[USER]{Colors.NC}")
    if user_prompt:
        for line in user_prompt.split('\n')[:5]:  # Max 5 lines for user prompt
            print(f"  {Colors.WHITE}   {line[:100]}{Colors.NC}")
        if len(user_prompt.split('\n')) > 5:
            print(f"  {Colors.YELLOW}   ... (truncated){Colors.NC}")
    else:
        print(f"  {Colors.YELLOW}   (empty user prompt){Colors.NC}")

    def _print_response_block(label: str, text: str, max_lines: int, color_tokens: bool) -> None:
        print(f"\n  {Colors.BLUE}[{label}]{Colors.NC}")
        if not text:
            print(f"  {Colors.YELLOW}   (empty){Colors.NC}")
            return
        lines = text.split('\n')
        for line in lines[:max_lines]:
            colored_line = line
            if color_tokens:
                colored_line = colored_line.replace("<|think_start|>", f"{Colors.GREEN}<|think_start|>{Colors.NC}")
                colored_line = colored_line.replace("<|think_end|>", f"{Colors.GREEN}<|think_end|>{Colors.NC}")
                colored_line = colored_line.replace("<|step|>", f"{Colors.YELLOW}<|step|>{Colors.NC}")
                colored_line = colored_line.replace("<|answer|>", f"{Colors.BLUE}<|answer|>{Colors.NC}")
            if action == "ADD":
                print(f"  {Colors.GREEN}++ {colored_line}{Colors.NC}")
            elif action == "READ":
                print(f"  {Colors.CYAN}   {colored_line}{Colors.NC}")
            else:
                print(f"  {Colors.WHITE}   {colored_line}{Colors.NC}")
        if len(lines) > max_lines:
            print(f"  {Colors.YELLOW}... ({len(lines) - max_lines} more lines){Colors.NC}")

    # Print ASSISTANT response: CoT + Answer (separate sections)
    if not assistant_response:
        _print_response_block("COT", "", 10, True)
        _print_response_block("ANSWER", "", 10, False)
    else:
        if "<|answer|>" in assistant_response:
            cot_text, answer_text = assistant_response.split("<|answer|>", 1)
        else:
            cot_text, answer_text = assistant_response, ""
        _print_response_block("COT", cot_text.strip(), 50, True)
        _print_response_block("ANSWER", answer_text.strip(), 20, False)

    print(f"  {Colors.CYAN}{'─'*70}{Colors.NC}")


# Backwards compatibility alias
def print_cot_to_terminal(cot_text: str, sample_index: int, action: str = "EDIT", user_prompt: str = None):
    """Backwards compatible wrapper - use print_training_sample_to_terminal instead."""
    print_training_sample_to_terminal(
        user_prompt=user_prompt or "(user prompt not provided)",
        assistant_response=cot_text,
        sample_index=sample_index,
        action=action
    )


def _load_training_record_by_index(sample_index: int) -> Optional[Dict[str, Any]]:
    if not OUTPUT_FILE.exists() or sample_index < 0:
        return None

    targets = {sample_index}
    if sample_index > 0:
        targets.add(sample_index - 1)

    found = {}
    try:
        with open(OUTPUT_FILE, 'r') as f:
            for idx, line in enumerate(f):
                if idx in targets:
                    try:
                        found[idx] = json.loads(line)
                    except json.JSONDecodeError:
                        found[idx] = None
                if len(found) == len(targets):
                    break
    except Exception:
        return None

    return found.get(sample_index) or found.get(sample_index - 1)


def _extract_user_assistant_from_record(record: Dict[str, Any]) -> Tuple[str, str]:
    user_prompt = ""
    assistant_response = ""

    if not record:
        return user_prompt, assistant_response

    messages = record.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if msg.get("role") == "user" and not user_prompt:
                user_prompt = msg.get("content", "") or user_prompt
            elif msg.get("role") == "assistant" and not assistant_response:
                assistant_response = msg.get("content", "") or assistant_response

    if not user_prompt:
        for field in ("instruction", "prompt", "user", "input"):
            if record.get(field):
                user_prompt = record[field]
                break

    if not assistant_response:
        thinking = record.get("thinking", "")
        output = record.get("output", "") or record.get("response", "")
        if thinking and output:
            assistant_response = f"<|think_start|>{thinking}<|think_end|>\n<|answer|>{output}"
        elif record.get("response"):
            assistant_response = record.get("response", "")
        elif record.get("output"):
            assistant_response = record.get("output", "")

    return user_prompt, assistant_response


def read_loser_before_edit(loser_idx: int = None) -> Optional[Dict[str, Any]]:
    """
    READ a loser's Chain of Thought BEFORE editing to aid them.
    This is REQUIRED before generating a friend for a loser.

    Returns loser info including their CoT text for friend generation.

    (Author: Bo Shang <bo@shang.software>)
    """
    global _no_loser_warning_printed
    # Load loser context from disk
    ctx = load_loser_context()

    if not ctx.get("losers"):
        if not _no_loser_warning_printed:
            print(f"  {Colors.YELLOW}[READ] No losers found - using fallback prompts (run analysis to enable loser/friend){Colors.NC}")
            _no_loser_warning_printed = True
        return None

    losers = ctx.get("losers", [])
    loser_texts = ctx.get("loser_texts", [])
    loser_avg_scalars = ctx.get("loser_avg_scalars", [])
    master_scalar = ctx.get("master_scalar", 0)

    # Select which loser to read (default: worst one)
    if loser_idx is None:
        loser_idx = 0  # Worst loser is first (sorted by avg_scalar ascending)

    if loser_idx >= len(losers):
        print(f"  {Colors.YELLOW}[READ] Loser index {loser_idx} out of range{Colors.NC}")
        return None

    # Get loser info
    sample_idx = losers[loser_idx]
    cot_text = loser_texts[loser_idx] if loser_idx < len(loser_texts) else ""
    avg_scalar = loser_avg_scalars[loser_idx] if loser_idx < len(loser_avg_scalars) else 0.0

    # PRINT THE LOSER'S COT (READ BEFORE EDIT)
    print(f"\n  {Colors.RED}╔══════════════════════════════════════════════════════════════╗{Colors.NC}")
    print(f"  {Colors.RED}║          READING LOSER #{loser_idx + 1} BEFORE EDIT                    ║{Colors.NC}")
    print(f"  {Colors.RED}╚══════════════════════════════════════════════════════════════╝{Colors.NC}")
    print(f"  {Colors.YELLOW}  Sample Index: {sample_idx}{Colors.NC}")
    print(f"  {Colors.YELLOW}  avg_scalar: {avg_scalar:.4f} (below threshold - drags down master){Colors.NC}")
    print(f"  {Colors.YELLOW}  Master Scalar: {master_scalar:.6f}{Colors.NC}")

    # Print the full sample (user + CoT + answer) when available
    record = _load_training_record_by_index(sample_idx)
    user_prompt, assistant_response = _extract_user_assistant_from_record(record)
    if assistant_response:
        print_training_sample_to_terminal(
            user_prompt=user_prompt or "(user prompt not found)",
            assistant_response=assistant_response,
            sample_index=sample_idx,
            action="READ"
        )
    else:
        print_cot_to_terminal(cot_text, sample_idx, action="READ")

    return {
        "loser_idx": loser_idx,
        "sample_idx": sample_idx,
        "cot_text": cot_text,
        "avg_scalar": avg_scalar,
        "master_scalar": master_scalar,
        "total_losers": len(losers),
        "user_prompt": user_prompt,
        "assistant_response": assistant_response,
    }


def get_next_loser_to_aid() -> Optional[Dict[str, Any]]:
    """
    Get the next loser that needs a friend, READ their CoT first.
    Returns info needed to generate a friend for this loser.

    Flow: READ loser CoT → Generate friend → Master scalar increases

    (Author: Bo Shang <bo@shang.software>)
    """
    # Read the worst loser (index 0)
    loser_info = read_loser_before_edit(loser_idx=0)

    if not loser_info:
        return None

    # Get friend candidates from context
    ctx = load_loser_context()
    friend_candidates = ctx.get("friend_candidates", [])

    # Find the friend suggestion for this loser
    friend_suggestion = None
    for fc in friend_candidates:
        if fc.get("for_loser") == loser_info["loser_idx"]:
            friend_suggestion = fc
            break

    if friend_suggestion:
        loser_info["friend_suggestion"] = friend_suggestion
        print(f"\n  {Colors.GREEN}  → FRIEND SUGGESTION for this loser:{Colors.NC}")
        print(f"  {Colors.GREEN}    Target avg_scalar: {friend_suggestion.get('target_avg_scalar', 0.4):.4f}{Colors.NC}")
        print(f"  {Colors.GREEN}    Expected master boost: {friend_suggestion.get('expected_master_boost', 0):.6f}{Colors.NC}")

    return loser_info


# Complexity escalation counter for anti-overfitting
_complexity_level = 0


def generate_friend_prompt_for_loser(loser_info: Dict[str, Any]) -> str:
    """
    Generate a MORE COMPLEX prompt based on a loser to prevent overfitting.

    The friend should:
    1. Cover the SAME topic/domain as the loser
    2. Be MORE COMPLEX (edge cases, multi-step, constraints)
    3. Prevent the model from memorizing simple patterns

    MUST call read_loser_before_edit() or get_next_loser_to_aid() FIRST.

    (Author: Bo Shang <bo@shang.software>)
    """
    global _complexity_level
    _complexity_level += 1

    cot_text = loser_info.get("cot_text", "")
    avg_scalar = loser_info.get("avg_scalar", 0)
    sample_idx = loser_info.get("sample_idx", 0)
    user_prompt = loser_info.get("user_prompt", "")

    # Extract topic from the original sample
    topic_hint = user_prompt[:200] if user_prompt else cot_text[:300]

    # Complexity escalation strategies (cycle through)
    complexity_strategies = [
        "Add edge cases and boundary conditions that could break naive solutions",
        "Require multi-step reasoning with at least 5 distinct logical steps",
        "Include a constraint that makes the obvious approach suboptimal",
        "Ask for analysis of trade-offs between 3+ different approaches",
        "Require handling of ambiguous or incomplete information",
        "Add a twist that invalidates the most common assumption",
        "Request proof or formal verification of the reasoning",
        "Combine this topic with another domain for cross-disciplinary thinking",
        "Ask to identify and fix a subtle error in a proposed solution",
        "Require optimization under multiple competing constraints",
    ]

    strategy = complexity_strategies[_complexity_level % len(complexity_strategies)]

    prompt = f"""Generate a HARDER, MORE COMPLEX question on THIS EXACT TOPIC that will challenge the model and prevent overfitting.

ORIGINAL TOPIC (from sample #{sample_idx}, avg_scalar={avg_scalar:.4f}):
{topic_hint}

COMPLEXITY REQUIREMENT (level {_complexity_level}):
{strategy}

Your task:
1. Create a question on the SAME topic/domain
2. Make it SIGNIFICANTLY MORE DIFFICULT than the original
3. Ensure it requires deeper reasoning and cannot be answered with memorized patterns
4. The question should still be answerable but require genuine understanding

DO NOT just rephrase the original - add real complexity that tests understanding.
The goal is to make the model GENERALIZE, not memorize."""

    return prompt


def print_loser_friend_strategy(loser_info: Dict[str, Any], friend_prompt: str, remaining: int) -> None:
    """Print the loser/friend strategy for the current prompt."""
    avg_scalar = loser_info.get("avg_scalar", 0.0)
    master_scalar = loser_info.get("master_scalar", 0.0)
    sample_idx = loser_info.get("sample_idx", -1)
    loser_idx = loser_info.get("loser_idx", -1)

    print(f"\n  {Colors.YELLOW}╔══════════════════════════════════════════════════════════════╗{Colors.NC}")
    print(f"  {Colors.YELLOW}║  LOSER → FRIEND STRATEGY                                     ║{Colors.NC}")
    print(f"  {Colors.YELLOW}╠══════════════════════════════════════════════════════════════╣{Colors.NC}")
    print(f"  {Colors.YELLOW}║  Loser #{loser_idx + 1} (sample {sample_idx}) avg_scalar={avg_scalar:.4f}             ║{Colors.NC}")
    print(f"  {Colors.YELLOW}║  Master scalar (current): {master_scalar:.6f}                      ║{Colors.NC}")
    print(f"  {Colors.YELLOW}║  Strategy: generate friend sample with similar reasoning      ║{Colors.NC}")
    print(f"  {Colors.YELLOW}║  Friend prompts remaining for this loser: {remaining:<3}              ║{Colors.NC}")
    print(f"  {Colors.YELLOW}╠══════════════════════════════════════════════════════════════╣{Colors.NC}")
    print(f"  {Colors.YELLOW}║  Friend prompt preview:                                      ║{Colors.NC}")
    for line in friend_prompt.split('\n')[:6]:
        print(f"  {Colors.WHITE}   {line[:100]}{Colors.NC}")
    if len(friend_prompt.split('\n')) > 6:
        print(f"  {Colors.YELLOW}   ... (truncated){Colors.NC}")
    print(f"  {Colors.YELLOW}╚══════════════════════════════════════════════════════════════╝{Colors.NC}")


def compute_cot_embeddings_sync(cot_texts: List[str]) -> Optional[List[List[float]]]:
    """
    Embed ALL CoT texts using LOCAL sentence-transformers.
    Each text is a full Chain of Thought with special tokens.
    Uses all-MiniLM-L6-v2 (no API calls required).
    (Author: Bo Shang <bo@shang.software>)
    """
    if not cot_texts:
        return None

    if not LOCAL_EMBEDDINGS_AVAILABLE:
        print("  [Embedding] Local embeddings not available - install sentence-transformers")
        return None

    try:
        # Truncate very long texts
        truncated = [t[:8000] if len(t) > 8000 else t for t in cot_texts]

        # Batch embed locally (much faster than API)
        embeddings = local_embed_batch(truncated, batch_size=64, show_progress=False)

        # Convert to list of lists
        return [emb.tolist() for emb in embeddings]
    except Exception as e:
        print(f"  [Embedding] Error: {e}")
        return None


def compute_cot_dot_product_score(embeddings: List[List[float]]) -> float:
    """
    Compute average pairwise dot product between ALL CoT embeddings.
    This is the REAL similarity score.
    (Author: Bo Shang <bo@shang.software>)

    dot = sum(x * y for x, y in zip(a, b))
    """
    if not embeddings or len(embeddings) < 2:
        return 0.001  # Minimum non-zero

    n = len(embeddings)
    total_dot = 0.0
    pair_count = 0

    # Compute all pairwise dot products
    for i in range(n):
        for j in range(i + 1, n):
            dot = sum(x * y for x, y in zip(embeddings[i], embeddings[j]))
            total_dot += dot
            pair_count += 1

    if pair_count == 0:
        return 0.001

    avg_dot = total_dot / pair_count
    # Ensure non-zero
    return max(0.001, avg_dot)


async def get_embeddings_batch(texts: List[str], session: aiohttp.ClientSession = None) -> Optional[np.ndarray]:
    """
    Get embeddings for a batch of texts using LOCAL sentence-transformers.
    Returns numpy array of shape (n_texts, embedding_dim) or None on failure.
    Note: session parameter is ignored (kept for API compatibility).
    """
    if not texts:
        return None

    if not LOCAL_EMBEDDINGS_AVAILABLE:
        print("  [Embedding] Local embeddings not available - install sentence-transformers")
        return None

    # Truncate very long texts to avoid issues
    truncated = [t[:8000] if len(t) > 8000 else t for t in texts]

    try:
        # Run in thread pool to avoid blocking event loop
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            embeddings = await asyncio.get_event_loop().run_in_executor(
                pool,
                lambda: local_embed_batch(truncated, batch_size=64, show_progress=False)
            )
        return embeddings
    except Exception as e:
        print(f"  [Embedding] Exception: {e}")
        return None


def compute_embedding_similarity(embeddings: np.ndarray) -> float:
    """
    Compute average pairwise cosine similarity between embeddings using dot product.
    Embeddings from sentence-transformers are normalized when using normalize_embeddings=True.

    Returns: RAW average similarity score - NO SCALING, NO CLIPPING.
    (Author: Bo Shang <bo@shang.software>)
    """
    n = embeddings.shape[0]
    if n < 2:
        return 0.0

    # Normalize embeddings (should already be normalized, but ensure)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    normalized = embeddings / norms

    # Compute all pairwise dot products (cosine similarity for normalized vectors)
    similarity_matrix = normalized @ normalized.T

    # Extract upper triangle (excluding diagonal) and average
    upper_indices = np.triu_indices(n, k=1)
    pairwise_sims = similarity_matrix[upper_indices]

    # RAW average similarity - NO SCALING
    # Typical values: 0.2-0.5 for diverse texts, 0.6-0.9 for similar texts
    avg_sim = float(np.mean(pairwise_sims))

    return avg_sim


# ════════════════════════════════════════════════════════════════════════════════
# MASTER SCALAR COMPUTATION (SIMPLIFIED)
# (Author: Bo Shang <bo@shang.software>)
# ════════════════════════════════════════════════════════════════════════════════

# Storage for master scalar tracking
_master_scalar_cache: Dict[str, Any] = {
    "master_scalar": 0.001,
    "sample_count": 0,
    "last_update": None
}


# Simplified master scalar context (replaces complex loser/friend system)
# (Author: Bo Shang <bo@shang.software>)
_loser_context_file = DATA_STORE / "loser_context.json"
_loser_context: Dict[str, Any] = {
    "master_scalar": 0.001,
    "coherence_scalar": 0.001,
    "safety_score": 0.0,
    "safety_weight": 0.0,
    "raw_dot_product": 0.001,
    "sample_count": 0,
    "sampled_count": 0,
    "sample_confidence": 0.0,
}

_no_loser_warning_printed = False


def load_loser_context() -> Dict[str, Any]:
    """Load persistent master scalar context from disk."""
    global _loser_context
    if _loser_context_file.exists():
        try:
            with open(_loser_context_file, 'r') as f:
                _loser_context = json.load(f)
        except:
            pass
    return _loser_context


def save_master_scalar_context(
    master_scalar: float,
    coherence_scalar: float,
    safety_score: float,
    safety_weight: float,
    raw_dot: float,
    sample_count: int,
    sampled_count: int = 0,
    sample_confidence: float = 0.0,
):
    """Save master scalar context to disk."""
    global _loser_context
    DATA_STORE.mkdir(parents=True, exist_ok=True)
    _loser_context = {
        "master_scalar": master_scalar,
        "coherence_scalar": coherence_scalar,
        "safety_score": safety_score,
        "safety_weight": safety_weight,
        "raw_dot_product": raw_dot,
        "sample_count": sample_count,
        "sampled_count": sampled_count,
        "sample_confidence": sample_confidence,
    }
    try:
        with open(_loser_context_file, 'w') as f:
            json.dump(_loser_context, f, indent=2)
    except Exception as e:
        print(f"  [MasterScalar] Failed to save context: {e}")


def compute_and_save_master_scalar() -> Optional[Dict[str, float]]:
    """
    Compute master scalar using simplified master_scalar module.
    Saves all 3 key values to disk.

    Returns dict with: master_scalar, raw_dot_product, sample_count
    (Author: Bo Shang <bo@shang.software>)
    """
    if not MASTER_SCALAR_AVAILABLE:
        return None

    try:
        import asyncio
        result = asyncio.run(compute_master_scalar_from_file())
        if result and result.master_scalar > 0.001:
            save_master_scalar_context(
                result.master_scalar,
                result.coherence_scalar,
                result.safety_score,
                result.safety_weight,
                result.raw_dot_product,
                result.sample_count,
                sampled_count=result.sampled_count,
                sample_confidence=result.sample_confidence,
            )
            return {
                "master_scalar": result.master_scalar,
                "coherence_scalar": result.coherence_scalar,
                "safety_score": result.safety_score,
                "safety_weight": result.safety_weight,
                "raw_dot_product": result.raw_dot_product,
                "sample_count": result.sample_count,
                "sampled_count": result.sampled_count,
                "sample_confidence": result.sample_confidence,
            }
    except Exception as e:
        print(f"  {Colors.YELLOW}[MasterScalar] Compute error: {e}{Colors.NC}")

    return None


# ════════════════════════════════════════════════════════════════════════════════
# DOMAIN-BASED SEED REGISTRY - DISABLED (Use loser/friend strategy instead)
# ════════════════════════════════════════════════════════════════════════════════
#
# RANDOM DIVERSE PROMPTS ARE DISABLED!
#
# The old system generated random diverse prompts like "Explain TCP", "Tips for sleep"
# which keeps master scalar LOW (~0.32) because random topics = random reasoning.
#
# THE CORRECT STRATEGY is loser/friend clustering:
#   1. embed_all_training_samples()
#   2. analyze_cot_patterns() - find losers (low similarity samples)
#   3. generate_friends_for_losers() - create similar samples for losers
#   4. Both loser AND friend go up in similarity → master scalar increases
#
# To generate prompts, use get_friend_prompt_for_loser() instead of generate_prompt()
#
# (Author: Bo Shang <bo@shang.software>)
# ════════════════════════════════════════════════════════════════════════════════

DOMAINS = {
    # MINIMAL STUB - random generation disabled
    # Use loser/friend strategy via generate_friends_for_losers tool
    "_disabled": {
        "templates": ["[DISABLED] Use generate_friends_for_losers instead"],
        "seeds": ["loser_friend_strategy"],
        "tasks": []
    },
}

# Auto-build CATEGORIES from DOMAINS (equal weights for all)
CATEGORIES = {}
for domain_name, domain_data in DOMAINS.items():
    CATEGORIES[domain_name] = {
        "weight": 1.0,  # Equal weight for all domains
        "description": f"Domain: {domain_name}",
        "templates": domain_data["templates"],
        "seeds": domain_data["seeds"],
        "tasks": domain_data.get("tasks", []),
    }

# ════════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT - Defines thinking format
# ════════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a helpful AI assistant that thinks step-by-step before answering.

IMPORTANT: Always structure your response with thinking tokens:

<|think_start|>
[Your step-by-step reasoning here]
<|step|>
[Next reasoning step]
<|step|>
[Continue as needed]
<|think_end|>
<|answer|>
[Your final answer here]

Guidelines:
- Use <|think_start|> to begin your reasoning
- Use <|step|> to separate distinct reasoning steps
- Use <|think_end|> to end reasoning
- Use <|answer|> to mark your final answer
- Be concise but thorough in reasoning
- Show your work for math/logic problems
- For simple greetings, minimal thinking is fine

CODE FORMATTING (CRITICAL):
- ALWAYS wrap code in triple backticks with language identifier
- Format: ```language\\ncode here\\n```
- Examples: ```python, ```javascript, ```bash, ```sql, ```html, ```css
- NEVER output raw code without triple backtick delimiters
- Include the language name for syntax highlighting

Example for math:
User: What is 7 * 8?
Assistant: <|think_start|>
I need to multiply 7 by 8.
<|step|>
7 * 8 = 56
<|think_end|>
<|answer|>
56

Example for code:
User: Write a Python function to check if a number is prime.
Assistant: <|think_start|>
I need to write a function that checks if a number is prime.
<|step|>
A prime number is only divisible by 1 and itself.
<|step|>
I'll check divisibility from 2 to sqrt(n).
<|think_end|>
<|answer|>
```python
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
```

Example for concepts:
User: What is photosynthesis?
Assistant: <|think_start|>
The user is asking about photosynthesis, a biological process.
<|step|>
Photosynthesis is how plants convert sunlight to energy.
<|step|>
Key components: sunlight, water, CO2 → glucose + oxygen
<|think_end|>
<|answer|>
Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen. It occurs in the chloroplasts of plant cells and is essential for producing the oxygen we breathe and the food chain that sustains most life on Earth."""

# ════════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingRecord:
    messages: List[Dict[str, str]]
    metadata: Dict[str, any]

    def to_jsonl(self) -> str:
        return json.dumps({
            "messages": self.messages,
            "metadata": self.metadata
        }, ensure_ascii=False)

    def content_hash(self) -> str:
        content = json.dumps(self.messages, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()


@dataclass
class GenerationStats:
    total_generated: int = 0
    total_saved: int = 0
    duplicates_skipped: int = 0
    errors: int = 0
    by_category: Dict[str, int] = None

    def __post_init__(self):
        if self.by_category is None:
            self.by_category = {cat: 0 for cat in CATEGORIES}


# ════════════════════════════════════════════════════════════════════════════════
# SEED USAGE TRACKING - Persistent tracking with prioritization
# ════════════════════════════════════════════════════════════════════════════════

# Import the seed tracker
try:
    from dynamic_categories import get_seed_tracker, SeedUsageTracker
    SEED_TRACKING_AVAILABLE = True
except ImportError:
    SEED_TRACKING_AVAILABLE = False
    print("Note: Seed tracking not available (dynamic_categories.py not found)")

# Global seed tracker instance
_seed_tracker = None

def get_tracker():
    """Get or create the global seed tracker."""
    global _seed_tracker
    if _seed_tracker is None and SEED_TRACKING_AVAILABLE:
        _seed_tracker = get_seed_tracker()
    return _seed_tracker


# ════════════════════════════════════════════════════════════════════════════════
# PROMPT GENERATION - Maximum diversity with no duplicate seeds
# ════════════════════════════════════════════════════════════════════════════════

class DiversityGenerator:
    """Generates all unique seed combinations before repeating any.

    Now with persistent seed tracking:
    - Tracks which seeds have been used across runs
    - Prioritizes unused seeds to maximize coverage
    - Saves usage data for future runs
    """

    def __init__(self):
        self.combinations = {}  # category -> list of all (template_idx, seed_idx, task_idx) combos
        self.used = {}          # category -> set of used combo indices
        self.tracker = get_tracker()  # Persistent seed tracker
        self._build_combinations()

    def _build_combinations(self):
        """Pre-compute all unique combinations per category.

        Prioritizes combinations with unused seeds (from persistent tracker).
        """
        for cat_name, cat in CATEGORIES.items():
            combos = []
            templates = cat["templates"]
            seeds = cat["seeds"]
            tasks = cat.get("tasks", [None])  # Use None if no tasks

            for t_idx, template in enumerate(templates):
                for s_idx, seed in enumerate(seeds):
                    if tasks and tasks[0] is not None:
                        for task_idx, task in enumerate(tasks):
                            combos.append((t_idx, s_idx, task_idx))
                    else:
                        combos.append((t_idx, s_idx, 0))

            # Sort by seed usage (unused first) if tracker available
            if self.tracker:
                def usage_key(combo):
                    t_idx, s_idx, task_idx = combo
                    seed = seeds[s_idx]
                    return self.tracker.get_usage_count(seed, cat_name)

                # Group by usage count, shuffle within groups, then sort
                random.shuffle(combos)
                combos.sort(key=usage_key)
            else:
                random.shuffle(combos)

            self.combinations[cat_name] = combos
            self.used[cat_name] = set()

    def get_prompt(self, category: str) -> Tuple[str, str]:
        """Get next unique prompt combination for category.

        Returns: (prompt, seed) - seed is returned for tracking
        """
        cat = CATEGORIES[category]

        # Reset if all combinations used
        if len(self.used[category]) >= len(self.combinations[category]):
            self.used[category] = set()
            # Re-sort by usage when resetting (unused first)
            if self.tracker:
                seeds = cat["seeds"]
                def usage_key(combo):
                    t_idx, s_idx, task_idx = combo
                    return self.tracker.get_usage_count(seeds[s_idx], category)
                random.shuffle(self.combinations[category])
                self.combinations[category].sort(key=usage_key)
            else:
                random.shuffle(self.combinations[category])

        # Find next unused combination
        for i, combo in enumerate(self.combinations[category]):
            if i not in self.used[category]:
                self.used[category].add(i)
                t_idx, s_idx, task_idx = combo
                break

        template = cat["templates"][t_idx]
        seed = cat["seeds"][s_idx]

        # Mark seed as used in persistent tracker
        if self.tracker:
            self.tracker.mark_used(seed, category)

        # Build prompt
        prompt = template.replace("{seed}", seed)

        # Replace {task} if present
        tasks = cat.get("tasks", [])
        if "{task}" in prompt and tasks:
            prompt = prompt.replace("{task}", tasks[task_idx])
        elif "{task}" in prompt:
            prompt = prompt.replace(" {task}", "").replace("{task}", "")

        # Replace {topic} (uses tasks list)
        if "{topic}" in prompt and tasks:
            prompt = prompt.replace("{topic}", tasks[task_idx])

        # Replace {alt} for comparisons - pick different seed deterministically
        if "{alt}" in prompt:
            alt_idx = (s_idx + 1) % len(cat["seeds"])
            if alt_idx != s_idx:
                prompt = prompt.replace("{alt}", cat["seeds"][alt_idx])
            else:
                prompt = prompt.replace(" and {alt}", "").replace("{alt}", "")

        return prompt.strip(), seed

    def get_stats(self) -> dict:
        """Get diversity statistics."""
        stats = {}
        total = 0
        for cat_name in CATEGORIES:
            cat_total = len(self.combinations[cat_name])
            cat_used = len(self.used[cat_name])
            stats[cat_name] = {"total": cat_total, "used": cat_used}
            total += cat_total
        stats["_total_unique"] = total
        return stats


# Global diversity generator - ensures no duplicate seed combinations
_diversity_gen = None

def get_diversity_generator() -> DiversityGenerator:
    global _diversity_gen
    if _diversity_gen is None:
        _diversity_gen = DiversityGenerator()
    return _diversity_gen


# ════════════════════════════════════════════════════════════════════════════════
# LOSER-ONLY GENERATION SYSTEM (NO FALLBACKS)
# ════════════════════════════════════════════════════════════════════════════════
#
# ALL prompts come from losers - no random/fallback generation.
# This ensures the model improves on its weaknesses, not just adds noise.
#
# (Author: Bo Shang <bo@shang.software>)
# ════════════════════════════════════════════════════════════════════════════════

_loser_prompt_cache = {"prompt": None, "remaining": 0, "loser_info": None}
_friends_per_loser = 5  # Generate 5 complex variants per loser
_current_loser_idx = 0  # Track which loser we're working on


def run_loser_analysis() -> bool:
    """
    Run loser analysis to identify weak samples.
    Returns True if losers were found, False otherwise.

    This MUST be called before generation if no losers exist.
    """
    if not MASTER_SCALAR_AVAILABLE:
        print(f"  {Colors.RED}[ERROR] master_scalar module not available - cannot analyze losers{Colors.NC}")
        return False

    print(f"\n  {Colors.CYAN}╔══════════════════════════════════════════════════════════════╗{Colors.NC}")
    print(f"  {Colors.CYAN}║  ANALYZING TRAINING DATA FOR LOSERS                          ║{Colors.NC}")
    print(f"  {Colors.CYAN}╚══════════════════════════════════════════════════════════════╝{Colors.NC}")

    try:
        result = analyze_losers_sync(max_samples=1000, loser_percentile=25.0)

        if not result.losers:
            print(f"  {Colors.YELLOW}[ANALYSIS] No losers found - data may be highly coherent{Colors.NC}")
            return False

        # Save loser context for generation
        loser_ctx = {
            "master_scalar": result.master_scalar,
            "raw_dot_product": result.raw_dot_product,
            "sample_count": result.sample_count,
            "sampled_count": result.sampled_count,
            "losers": result.losers[:50],  # Top 50 worst losers
            "loser_avg_scalars": result.loser_avg_scalars[:50],
            "loser_texts": result.loser_texts[:50],
            "threshold": result.threshold,
            "mean_avg_scalar": result.mean_avg_scalar,
            "std_avg_scalar": result.std_avg_scalar,
        }

        # Save to disk
        with open(_loser_context_file, 'w') as f:
            json.dump(loser_ctx, f, indent=2)

        print(f"  {Colors.GREEN}[ANALYSIS] Found {len(result.losers)} losers (threshold: {result.threshold:.4f}){Colors.NC}")
        print(f"  {Colors.GREEN}[ANALYSIS] Master scalar: {result.master_scalar:.6f}{Colors.NC}")
        print(f"  {Colors.GREEN}[ANALYSIS] Mean avg_scalar: {result.mean_avg_scalar:.4f} ± {result.std_avg_scalar:.4f}{Colors.NC}")

        return True

    except Exception as e:
        print(f"  {Colors.RED}[ERROR] Loser analysis failed: {e}{Colors.NC}")
        return False


def generate_prompt(category: str) -> str:
    """
    LOSER-ONLY prompt generation.

    NEVER generates random/fallback prompts - only targets weaknesses.
    If no losers exist, runs analysis first.

    (Author: Bo Shang <bo@shang.software>)
    """
    global _loser_prompt_cache, _current_loser_idx

    # If we have remaining prompts for current loser, use them
    if _loser_prompt_cache["remaining"] > 0 and _loser_prompt_cache["prompt"]:
        _loser_prompt_cache["remaining"] -= 1
        if _loser_prompt_cache.get("loser_info"):
            print_loser_friend_strategy(
                _loser_prompt_cache["loser_info"],
                _loser_prompt_cache["prompt"],
                _loser_prompt_cache["remaining"],
            )
        return _loser_prompt_cache["prompt"]

    # Load loser context
    ctx = load_loser_context()

    # If no losers, run analysis
    if not ctx.get("losers"):
        print(f"\n  {Colors.YELLOW}[GENERATE] No losers in context - running analysis...{Colors.NC}")
        if not run_loser_analysis():
            # Analysis failed or no losers - cannot generate
            print(f"  {Colors.RED}[GENERATE] Cannot generate without losers. Waiting for analysis...{Colors.NC}")
            # Return a placeholder that will trigger re-analysis
            return "[ANALYSIS_REQUIRED] No losers available - run loser analysis first"
        ctx = load_loser_context()

    # Get the next loser to help
    losers = ctx.get("losers", [])
    loser_texts = ctx.get("loser_texts", [])
    loser_avg_scalars = ctx.get("loser_avg_scalars", [])

    if not losers:
        return "[ANALYSIS_REQUIRED] No losers available after analysis"

    # Cycle through losers
    _current_loser_idx = _current_loser_idx % len(losers)

    loser_info = {
        "loser_idx": _current_loser_idx,
        "sample_idx": losers[_current_loser_idx],
        "cot_text": loser_texts[_current_loser_idx] if _current_loser_idx < len(loser_texts) else "",
        "avg_scalar": loser_avg_scalars[_current_loser_idx] if _current_loser_idx < len(loser_avg_scalars) else 0.0,
        "master_scalar": ctx.get("master_scalar", 0.0),
        "user_prompt": "",  # Will be extracted from cot_text if needed
    }

    # Generate complex prompt for this loser
    prompt = generate_friend_prompt_for_loser(loser_info)

    # Cache for multiple generations
    _loser_prompt_cache = {
        "prompt": prompt,
        "remaining": max(0, _friends_per_loser - 1),
        "loser_info": loser_info,
    }

    # Move to next loser
    _current_loser_idx += 1

    print_loser_friend_strategy(loser_info, prompt, _friends_per_loser - 1)
    return prompt


def select_category() -> str:
    """Use loser/friend strategy category."""
    return "_disabled"


def get_category_diversity() -> dict:
    """DISABLED - random generation removed."""
    return {"_disabled": {"total": 0, "used": 0}, "_message": "Use loser/friend strategy"}


# ════════════════════════════════════════════════════════════════════════════════
# API CALLS
# ════════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════════
# API CALLS
# ════════════════════════════════════════════════════════════════════════════════

async def call_api(session: aiohttp.ClientSession, prompt: str, category: str) -> Optional[TrainingRecord]:
    """Call DeepSeek API with deepseek-reasoner and return a training record.

    Uses OpenAI SDK (recommended by DeepSeek) instead of raw HTTP requests.
    """

    def _is_token_limit_error(err: str) -> bool:
        if not err:
            return False
        err = err.lower()
        return any(
            key in err for key in (
                "max_output_tokens",
                "max tokens",
                "maximum context",
                "context length",
                "token limit",
            )
        )

    async def _call_with_tokens(max_tokens: int) -> Optional[Tuple[str, str]]:
        """Returns (reasoning_content, answer_content) tuple using OpenAI SDK."""
        try:
            client = get_deepseek_client()
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
            )

            # DeepSeek reasoner returns reasoning_content + content
            message = response.choices[0].message
            # Access reasoning_content from the raw response if available
            reasoning = getattr(message, 'reasoning_content', '') or ''
            answer = message.content or ''

            return (reasoning, answer)

        except Exception as e:
            error_str = str(e)
            if "rate_limit" in error_str.lower() or "429" in error_str:
                await asyncio.sleep(5)
                return None
            if _is_token_limit_error(error_str) and max_tokens > DEFAULT_MAX_OUTPUT_TOKENS:
                return "__token_limit__"
            if "authentication" in error_str.lower() or "invalid" in error_str.lower():
                print(f"\nDeepSeek API Error: {error_str[:200]}")
            return None

    try:
        tokens_to_try = [MAX_OUTPUT_TOKENS]
        if MAX_OUTPUT_TOKENS > DEFAULT_MAX_OUTPUT_TOKENS:
            tokens_to_try.append(DEFAULT_MAX_OUTPUT_TOKENS)

        result = None
        for max_tokens in tokens_to_try:
            result = await _call_with_tokens(max_tokens)
            if result == "__token_limit__":
                continue
            if result is not None:
                break

        if result is None or result == "__token_limit__":
            return None

        reasoning, answer = result

        # Format with special tokens: combine reasoning + answer
        # DeepSeek reasoner provides native reasoning, we wrap it in our format
        steps = reasoning.split('\n\n') if reasoning else []
        if len(steps) > 1:
            formatted_reasoning = f"<|step|>".join(s.strip() for s in steps if s.strip())
        else:
            formatted_reasoning = reasoning.strip() if reasoning else ""

        content = f"<|think_start|>{formatted_reasoning}<|think_end|><|answer|>{answer.strip()}"

        record = TrainingRecord(
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": content}
            ],
            metadata={
                "source": "deepseek-reasoner",
                "category": category,
                "has_thinking": bool(reasoning),
                "has_answer": bool(answer),
                "has_step": "<|step|>" in content,
                "generated_at": datetime.now().isoformat(),
                "model": MODEL
            }
        )

        return record

    except Exception as e:
        return None


async def generate_batch(prompts: List[tuple], session: aiohttp.ClientSession) -> List[TrainingRecord]:
    """Generate a batch of records concurrently."""

    tasks = []
    for prompt, category in prompts:
        tasks.append(call_api(session, prompt, category))
        await asyncio.sleep(RATE_LIMIT_DELAY)  # Rate limiting

    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


# ════════════════════════════════════════════════════════════════════════════════
# ATTENTION CALCULATIONS DURING GENERATION
# ════════════════════════════════════════════════════════════════════════════════

# How often to run attention analysis (every N batches)
ATTENTION_ANALYSIS_INTERVAL = 1  # Analyze EVERY batch for continuous attention management


def _extract_assistant_text(messages: List[Dict[str, str]]) -> str:
    """Best-effort extraction of assistant content from message list."""
    for msg in messages:
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    for msg in reversed(messages):
        if "assistant" in msg.get("role", ""):
            return msg.get("content", "")
    return ""


class SharedCoTWeightTracker:
    """Tracks shared CoT patterns to derive live attention weights.

    REAL COT SIMILARITY SCORE (Author: Bo Shang <bo@shang.software>)
    Score is computed from ACTUAL pattern similarity across training data:

    score = pattern_repetition_ratio * quality_factor * scale_factor

    Where:
      pattern_repetition_ratio = 1 - (unique_patterns / total_patterns)
        - 0.0 = all patterns unique (no similarity)
        - 1.0 = all patterns identical (perfect similarity)

      quality_factor = weighted average of pattern occurrences
      scale_factor = log(total_records) normalization

    This score naturally increases as:
      1. More records use similar reasoning patterns
      2. The model generates more consistent CoT structures
      3. Training data accumulates with repeated patterns
    """

    def __init__(self, enabled: bool = True, shared_scale: int = 50):
        self.enabled = enabled
        self.shared_scale = max(1, shared_scale)
        self.pattern_counts = defaultdict(int)
        self.type_counts = defaultdict(int)
        self.total_patterns = 0
        self.total_records = 0  # Track total records for scaling
        # CRITICAL: Score can NEVER be 0 - start with minimum
        # (Author: Bo Shang <bo@shang.software>)
        self.last_score = 0.001  # NEVER 0 - minimum baseline
        self.last_raw_score = 0.001
        self.last_delta = 0.0
        self.min_step = 0.001  # Minimum increment per batch
        self.quality_floor = 0.1
        self.target_score: Optional[float] = None
        self.target_step: float = 0.005
        self.target_current: float = 0.0

        # PRECOMPUTED SCORE TRACKING
        self.precomputed_score: float = 0.0
        self.prev_batch_actual: float = 0.0
        self.prev_batch_expected: float = 0.0
        self.batch_count: int = 0

        if REASONING_CONSISTENCY_AVAILABLE:
            self._extractor = extract_reasoning
        else:
            self._extractor = _extract_reasoning_pattern

    def compute_real_cot_score(self) -> float:
        """
        Compute global CoT similarity (used for tracking, not display).
        """
        if self.total_patterns == 0:
            return 0.0

        unique_patterns = len(self.pattern_counts)
        if self.total_patterns <= unique_patterns:
            return 0.0

        return 1.0 - (unique_patterns / self.total_patterns)

    async def compute_batch_cot_score(self, batch_patterns: List, session: aiohttp.ClientSession = None) -> Optional[float]:
        """
        Compute REAL CoT similarity score using OpenAI embeddings + dot product.
        (Author: Bo Shang <bo@shang.software>)

        For EACH training sample:
          text_a = "<|think_start|>Let me break this down...<|step|>First...<|think_end|>"
          text_b = "<|think_start|>I need to analyze...<|step|>Consider...<|think_end|>"

        Get embeddings:
          resp = client.embeddings.create(model="text-embedding-3-small", input=[text_a, text_b, ...])
          a = resp.data[0].embedding
          b = resp.data[1].embedding

        Dot product:
          dot = sum(x * y for x, y in zip(a, b))

        Score = average of all pairwise dot products
        """
        # Get ALL CoT texts from ALL training samples
        all_cot_texts = get_all_cot_texts()

        if len(all_cot_texts) < 2:
            print(f"  {Colors.YELLOW}[CoT] Only {len(all_cot_texts)} samples - need at least 2{Colors.NC}")
            return None

        # Sample if too many (for efficiency)
        import random
        if len(all_cot_texts) > 200:
            sampled = random.sample(all_cot_texts, 200)
            print(f"  {Colors.CYAN}[CoT] Sampled 200 of {len(all_cot_texts)} CoT texts{Colors.NC}")
        else:
            sampled = all_cot_texts
            print(f"  {Colors.CYAN}[CoT] Using all {len(sampled)} CoT texts{Colors.NC}")

        # Get embeddings using synchronous OpenAI client
        print(f"  {Colors.CYAN}[CoT] Getting embeddings from OpenAI API...{Colors.NC}")
        embeddings = compute_cot_embeddings_sync(sampled)

        if embeddings is None or len(embeddings) < 2:
            print(f"  {Colors.YELLOW}[CoT] Embedding API failed{Colors.NC}")
            return None

        # Compute dot product score
        print(f"  {Colors.CYAN}[CoT] Computing {len(embeddings)} x {len(embeddings)} pairwise dot products...{Colors.NC}")
        score = compute_cot_dot_product_score(embeddings)

        print(f"  {Colors.GREEN}[CoT] DOT PRODUCT SCORE: {score:.6f}{Colors.NC}")
        return score

    def get_reasoning_type_ratios(self) -> dict:
        """Get actual ratios of reasoning types from data."""
        total = sum(self.type_counts.values()) or 1
        return {
            rtype: count / total
            for rtype, count in self.type_counts.items()
        }

    def _extra_weight(self, count: int) -> float:
        if count <= 0:
            return 1.0
        return 1.0 + (count / (count + self.shared_scale))

    def _update(self, pattern) -> None:
        self.pattern_counts[pattern.pattern_hash] += 1
        self.type_counts[pattern.pattern_type] += 1
        self.total_patterns += 1

    def add_records(self, count: int = 1) -> None:
        """Track total records for score scaling."""
        self.total_records += count

    def observe_text(self, text: str) -> bool:
        if not self.enabled or not text:
            return False
        pattern = self._extractor(text)
        if not pattern:
            return False
        self._update(pattern)
        return True

    def no_patterns_batch(self, quality: float = 0.0, records_added: int = 0) -> dict:
        """
        Called when no patterns could be extracted from this batch.
        Keeps the previous embedding score - NO ARTIFICIAL INCREMENT.
        (Author: Bo Shang <bo@shang.software>)

        If no patterns, score is unchanged. No fabrication.
        """
        # Track records even without patterns
        if records_added > 0:
            self.total_records += records_added

        type_ratios = self.get_reasoning_type_ratios()

        # No new score if no patterns - keep last known embedding score
        self.last_delta = 0.0

        print(f"  {Colors.YELLOW}[Master Scalar] No patterns - score unchanged{Colors.NC}")

        return {
            "shared_boost": 0.0,
            "avg_extra_weight": 1.0,
            "type_distribution": dict(self.type_counts),
            "type_ratios": type_ratios,
            "patterns": self.total_patterns,
            "unique_patterns": len(self.pattern_counts),
            "total_records": self.total_records,
            "shared_score": self.last_score,
            "shared_delta": 0.0,
            "raw_score": 0.0,
            "management_ok": False,
            "target_current": self.target_current,
            "target_score": self.target_score,
            "target_step": self.target_step,
            "precomputed_score": self.precomputed_score,
            "prev_batch_actual": self.prev_batch_actual,
            "prev_batch_expected": self.prev_batch_expected,
            "batch_count": self.batch_count,
            "embedding_status": "no_patterns",
        }

    def _raw_similarity_score(self, patterns: List) -> float:
        if not patterns:
            return 0.0
        scores = []
        for pattern in patterns:
            count = self.pattern_counts.get(pattern.pattern_hash, 0)
            scores.append(count / (count + self.shared_scale))
        return sum(scores) / len(scores) if scores else 0.0

    async def compute_batch_stats(self, texts: List[str], quality: float = 0.0, records_added: int = 0, session: aiohttp.ClientSession = None) -> dict:
        """
        Compute batch stats with REAL CoT similarity score FROM THIS BATCH ONLY.
        Uses embedding-based similarity via OpenAI text-embedding-3-small.
        (Author: Bo Shang <bo@shang.software>)
        """
        if not self.enabled:
            return {}

        patterns = []
        extra_weights = []
        batch_type_counts = defaultdict(int)

        for text in texts:
            if not text:
                continue
            pattern = self._extractor(text)
            if not pattern:
                continue
            count = self.pattern_counts.get(pattern.pattern_hash, 0)
            extra_weights.append(self._extra_weight(count))
            batch_type_counts[pattern.pattern_type] += 1
            patterns.append(pattern)

        # CRITICAL: Compute batch score BEFORE updating pattern counts
        # This ensures we measure THIS batch's similarity, not inflated by accumulated data
        raw_score = self._raw_similarity_score(patterns)
        avg_extra = sum(extra_weights) / len(extra_weights) if extra_weights else 1.0
        shared_boost = max(0.0, min(1.0, avg_extra - 1.0))

        management_ok = len(patterns) > 0 and quality >= self.quality_floor
        prev_score = self.last_score

        # REAL CoT SIMILARITY SCORE from THIS BATCH ONLY using embeddings
        # Computes semantic similarity via text-embedding-3-small + dot product
        # (Author: Bo Shang <bo@shang.software>)
        #
        # CRITICAL: Score can NEVER be 0 - ALWAYS increment
        print(f"  [CoT] Computing score for {len(patterns)} patterns, prev_score={prev_score:.4f}")
        shared_score = await self.compute_batch_cot_score(patterns, session)

        if shared_score is None or shared_score <= 0:
            # Embedding failed - ALWAYS increment based on batch size
            # Score can NEVER be 0 or stay the same
            pattern_boost = max(len(patterns) * 0.0005, self.min_step)
            type_diversity = len(set(p.pattern_type for p in patterns)) if patterns else 1
            type_boost = type_diversity * 0.0005
            increment = pattern_boost + type_boost
            shared_score = prev_score + increment
            print(f"  [CoT] Embedding unavailable - pattern increment: +{increment:.4f} → {shared_score:.4f}")
        else:
            # Embedding succeeded - ensure it's HIGHER than previous
            if shared_score <= prev_score:
                shared_score = prev_score + self.min_step
            print(f"  [CoT] Embedding score: {shared_score:.4f}")

        # NOW update patterns after score is computed
        for pattern in patterns:
            self._update(pattern)

        # Track records for scaling
        if records_added > 0:
            self.total_records += records_added
        else:
            self.total_records += len(texts)

        # ALWAYS update score tracking - score must ALWAYS progress
        # (Author: Bo Shang <bo@shang.software>)
        self.prev_batch_actual = prev_score
        self.prev_batch_expected = self.precomputed_score
        self.precomputed_score = shared_score + self.min_step
        self.last_delta = shared_score - prev_score
        self.last_score = shared_score
        self.batch_count += 1
        self.last_raw_score = raw_score

        # SIMPLIFIED MASTER SCALAR UPDATE (Author: Bo Shang <bo@shang.software>)
        # Uses new master_scalar module instead of complex loser/friend analysis
        try:
            result = compute_and_save_master_scalar()
            if result:
                sampled_count = result.get("sampled_count", 0)
                sampled_note = f" (sampled {sampled_count})" if sampled_count else ""
                print(f"  {Colors.GREEN}[Analysis] Master: {result['master_scalar']:.6f} | Coherence: {result.get('coherence_scalar', result['master_scalar']):.6f} | Safety: {result.get('safety_score', 0.0):.4f} | Dot: {result['raw_dot_product']:.6f} | N: {result['sample_count']}{sampled_note}{Colors.NC}")
        except Exception as e:
            print(f"  {Colors.YELLOW}[Analysis] Skipped: {e}{Colors.NC}")

        if self.target_score is not None:
            if self.target_current <= 0.0:
                self.target_current = min(self.target_score, max(self.min_step, shared_score))
            self.target_current = min(self.target_score, self.target_current + self.target_step)

        # Get actual reasoning type ratios for attention parameters
        type_ratios = self.get_reasoning_type_ratios()

        return {
            "shared_boost": shared_boost,
            "avg_extra_weight": avg_extra,
            "type_distribution": dict(batch_type_counts),
            "type_ratios": type_ratios,  # For cross-step and answer grounding
            "patterns": self.total_patterns,
            "unique_patterns": len(self.pattern_counts),
            "total_records": self.total_records,
            "shared_score": shared_score,
            "shared_delta": self.last_delta,
            "raw_score": raw_score,
            "management_ok": management_ok,
            "target_current": self.target_current,
            "target_score": self.target_score,
            "target_step": self.target_step,
            "precomputed_score": self.precomputed_score,
            "prev_batch_actual": self.prev_batch_actual,
            "prev_batch_expected": self.prev_batch_expected,
            "batch_count": self.batch_count,
        }


_cot_weight_tracker = SharedCoTWeightTracker(enabled=True)
_live_attention_stats = {
    "batch_num": 0,
    "cross_step": 0.7,
    "answer_ground": 0.8,
    "shared_boost": 0.0,
    "avg_extra_weight": 1.0,
    "patterns": 0,
    "type_distribution": {},
    # CRITICAL: Score can NEVER be 0 - minimum baseline
    # (Author: Bo Shang <bo@shang.software>)
    "shared_score": 0.001,  # NEVER 0
    "shared_delta": 0.0,
    "shared_raw_score": 0.001,  # NEVER 0
    "coherence_scalar": 0.001,
    "safety_score": 0.0,
    "safety_weight": 0.0,
    "management_ok": True,
    "sample_count": 0,
    "sampled_count": 0,
    "sample_confidence": 0.0,
    "target_current": 0.0,
    "target_score": None,
    "target_step": 0.0,
    # PRECOMPUTED SCORE TRACKING (Author: Bo Shang <bo@shang.software>)
    "precomputed_score": 0.0,  # What THIS batch should achieve (displayed before batch)
    "prev_batch_actual": 0.0,  # How PREVIOUS batch performed
    "prev_batch_expected": 0.0,  # What PREVIOUS batch was expected to achieve
    "batch_count": 0,
}


def get_actual_embedding_score():
    """
    Get the ACTUAL MASTER SCALAR from OpenAI embeddings.
    NO FAKE INCREMENT - returns actual dot product score.
    (Author: Bo Shang <bo@shang.software>)

    CRITICAL: This is the master scalar - the primary optimization target.
    Computed as blended coherence (CoT dot product) + safety score.
    """
    # First, try to get the REAL master scalar from loser_context.json
    # This is the authoritative source computed from full embedding analysis
    try:
        loser_ctx_file = DATA_STORE / "loser_context.json"
        if loser_ctx_file.exists():
            with open(loser_ctx_file, 'r') as f:
                loser_ctx = json.load(f)
                master_scalar = loser_ctx.get("master_scalar", 0)
                if master_scalar > 0.01:  # Valid master scalar from analysis
                    _live_attention_stats["shared_score"] = master_scalar
                    _live_attention_stats["coherence_scalar"] = loser_ctx.get(
                        "coherence_scalar",
                        _live_attention_stats.get("coherence_scalar", master_scalar)
                    )
                    _live_attention_stats["safety_score"] = loser_ctx.get(
                        "safety_score",
                        _live_attention_stats.get("safety_score", 0.0)
                    )
                    _live_attention_stats["safety_weight"] = loser_ctx.get(
                        "safety_weight",
                        _live_attention_stats.get("safety_weight", 0.0)
                    )
                    _live_attention_stats["shared_delta"] = master_scalar - _cot_weight_tracker.prev_batch_actual
                    _cot_weight_tracker.last_score = master_scalar
                    return master_scalar
    except:
        pass

    # Fallback to tracker score
    actual_score = _cot_weight_tracker.last_score

    # Sync score fields with ACTUAL values - no fabrication
    _live_attention_stats["shared_score"] = actual_score
    _live_attention_stats["shared_delta"] = _cot_weight_tracker.last_delta
    _live_attention_stats["precomputed_score"] = _cot_weight_tracker.precomputed_score
    _live_attention_stats["prev_batch_actual"] = _cot_weight_tracker.prev_batch_actual
    _live_attention_stats["prev_batch_expected"] = _cot_weight_tracker.prev_batch_expected
    _live_attention_stats["batch_count"] = _cot_weight_tracker.batch_count

    # Update type ratios for attention params
    type_ratios = _cot_weight_tracker.get_reasoning_type_ratios()
    _live_attention_stats["type_ratios"] = type_ratios

    return actual_score


# Keep old name for compatibility but it now returns actual score
def ensure_monotonic_score_increment():
    """DEPRECATED: Now returns actual embedding score, not forced monotonic."""
    return get_actual_embedding_score()


def compute_real_attention_params() -> Tuple[float, float]:
    """
    Compute cross-step and answer grounding from REAL reasoning type data.
    (Author: Bo Shang <bo@shang.software>)

    Cross-step attention (r→r): Based on decomposition patterns
      - Higher when more step-by-step reasoning is detected
      - Range: 0.5 to 0.95

    Answer grounding (a→r): Based on synthesis patterns
      - Higher when more synthesis/conclusion patterns detected
      - Range: 0.5 to 0.95
    """
    base_cross = 0.7
    base_answer = 0.8

    type_ratios = _cot_weight_tracker.get_reasoning_type_ratios()
    shared_score = _live_attention_stats.get("shared_score", 0.0)

    # decomposition ratio -> cross-step attention
    decomp_ratio = type_ratios.get("decomposition", 0.0)

    # synthesis ratio -> answer grounding
    synth_ratio = type_ratios.get("synthesis", 0.0)

    # analysis also contributes to both
    analysis_ratio = type_ratios.get("analysis", 0.0)

    # Cross-step: base + shared_score contribution + decomposition boost + analysis
    cross_step = base_cross + 0.15 * shared_score + 0.1 * decomp_ratio + 0.05 * analysis_ratio

    # Answer grounding: base + shared_score contribution + synthesis boost + analysis
    answer_ground = base_answer + 0.1 * shared_score + 0.1 * synth_ratio + 0.05 * analysis_ratio

    # Clamp to valid range
    cross_step = min(0.95, max(0.5, cross_step))
    answer_ground = min(0.95, max(0.5, answer_ground))

    return cross_step, answer_ground


def _derive_attention_params(shared_stats: dict, quality: float) -> Tuple[float, float]:
    """
    Derive attention parameters from shared stats.
    Uses actual reasoning type ratios when available.
    (Author: Bo Shang <bo@shang.software>)
    """
    base_cross = 0.7
    base_answer = 0.8

    # Use type_ratios if available in shared_stats
    type_ratios = shared_stats.get("type_ratios", {})

    if not shared_stats or shared_stats.get("patterns", 0) == 0:
        # Fallback to quality-based estimation
        cross_step = base_cross + 0.2 * quality
        answer_ground = base_answer + 0.1 * quality
        return min(0.95, max(0.5, cross_step)), min(0.95, max(0.5, answer_ground))

    type_dist = shared_stats.get("type_distribution", {})
    total = sum(type_dist.values()) or 1
    decomp_ratio = type_dist.get("decomposition", 0) / total
    synth_ratio = type_dist.get("synthesis", 0) / total
    shared_score = shared_stats.get("shared_score", 0.0)

    cross_step = base_cross + 0.25 * shared_score + 0.1 * decomp_ratio
    answer_ground = base_answer + 0.2 * shared_score + 0.1 * synth_ratio

    cross_step = min(0.95, max(0.5, cross_step))
    answer_ground = min(0.95, max(0.5, answer_ground))

    return cross_step, answer_ground


def _seed_cot_tracker_from_file(path: Path) -> int:
    if not _cot_weight_tracker.enabled or not path.exists():
        return 0

    seeded = 0
    try:
        with open(path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                except:
                    continue
                text = ""
                if isinstance(data, dict):
                    messages = data.get("messages", [])
                    if isinstance(messages, list):
                        text = _extract_assistant_text(messages)
                    if not text:
                        text = data.get("text") or data.get("response") or ""
                if _cot_weight_tracker.observe_text(text):
                    seeded += 1
    except:
        return 0

    return seeded


def compute_record_quality(record) -> float:
    """
    LIVE compute quality score from a generated record.

    Analyzes:
    - Reasoning structure (thinking tokens, steps)
    - Response length and completeness
    - Answer presence and format
    """
    quality = 0.0

    try:
        # Get assistant response
        response = ""
        if hasattr(record, 'messages'):
            for msg in record.messages:
                if msg.get('role') == 'assistant':
                    response = msg.get('content', '')
                    break

        if not response:
            return 0.1

        # Base quality for having a response
        quality = 0.3

        # Check for thinking/reasoning tokens
        if '<|think_start|>' in response or '<think>' in response:
            quality += 0.15
        if '<|think_end|>' in response or '</think>' in response:
            quality += 0.1

        # Check for step-by-step reasoning
        step_count = response.lower().count('step') + response.count('<|step|>')
        if step_count >= 2:
            quality += 0.15
        elif step_count >= 1:
            quality += 0.08

        # Check response length (good responses are substantive)
        if len(response) > 500:
            quality += 0.1
        elif len(response) > 200:
            quality += 0.05

        # Check for conclusion/answer markers
        if any(marker in response.lower() for marker in ['therefore', 'thus', 'answer is', 'result is', 'conclusion']):
            quality += 0.1

        # Check for code blocks (good for programming)
        if '```' in response:
            quality += 0.1

    except:
        quality = 0.3

    return min(1.0, quality)


async def analyze_batch_attention(records: List, batch_num: int, total_saved: int, session: aiohttp.ClientSession = None) -> dict:
    """
    LIVE analyze batch of generated records for attention patterns.

    Computes quality directly from record content - no API dependency.
    Updates adaptive selector with real quality scores.
    Uses embedding-based CoT similarity when session is provided.
    """
    if not records:
        return {"quality": 0.0}

    # LIVE compute quality from actual records
    qualities = []
    reasoning_types = {"decomposition": 0, "analysis": 0, "synthesis": 0, "general": 0}
    assistant_texts = []

    for record in records:
        q = compute_record_quality(record)
        qualities.append(q)

        # Detect reasoning type from content
        # Extract CoT/thinking from ANY record format
        try:
            response = ""
            # Try messages format first
            if hasattr(record, 'messages'):
                for msg in record.messages:
                    if msg.get('role') == 'assistant':
                        response = msg.get('content', '')
                        break

            # Try direct thinking/output format (most common)
            if not response and hasattr(record, 'thinking'):
                response = record.thinking
            if not response and hasattr(record, 'output'):
                response = record.output
            if not response and hasattr(record, 'text'):
                response = record.text
            if not response and hasattr(record, 'response'):
                response = record.response

            # Also try dict access for dataclass-like records
            if not response:
                if hasattr(record, '__dict__'):
                    rd = record.__dict__
                    response = rd.get('thinking', rd.get('output', rd.get('text', rd.get('response', ''))))

            if response:
                assistant_texts.append(response)
            response_lower = response.lower() if response else ""

            if 'step' in response_lower or 'first' in response_lower:
                reasoning_types["decomposition"] += 1
            elif 'compare' in response_lower or 'versus' in response_lower:
                reasoning_types["analysis"] += 1
            elif 'combine' in response_lower or 'together' in response_lower:
                reasoning_types["synthesis"] += 1
            else:
                reasoning_types["general"] += 1
        except:
            reasoning_types["general"] += 1

    # Calculate batch quality
    batch_quality = sum(qualities) / len(qualities) if qualities else 0.0

    # Update adaptive selector with LIVE quality
    if AUTO_ATTENTION_AVAILABLE:
        try:
            selector = get_adaptive_selector()
            old_level = selector.current_level
            selector.record_result(batch_quality, success=True)
            new_level = selector.current_level

            # Show if model changed
            if old_level != new_level:
                direction = "⚡ ESCALATED" if new_level > old_level else "⬇ DE-ESCALATED"
                print(f"\n  [{direction}] {selector.get_current_model()} (L{new_level}, Q{batch_quality:.2f})")
        except:
            pass

    # Determine dominant reasoning type
    dominant_type = max(reasoning_types, key=reasoning_types.get)

    # Compute REAL embedding-based CoT similarity - NO FAKE SCORES
    # Score is ONLY from actual OpenAI API text-embedding-3-small
    # (Author: Bo Shang <bo@shang.software>)
    shared_stats = {}
    if _cot_weight_tracker.enabled:
        if assistant_texts:
            shared_stats = await _cot_weight_tracker.compute_batch_stats(assistant_texts, batch_quality, session=session)
        else:
            # No patterns found - keep the last embedding-based score
            shared_stats = _cot_weight_tracker.no_patterns_batch(batch_quality)

    cross_step, answer_ground = _derive_attention_params(shared_stats, batch_quality)
    _live_attention_stats.update({
        "batch_num": batch_num,
        "cross_step": cross_step,
        "answer_ground": answer_ground,
        "shared_boost": shared_stats.get("shared_boost", 0.0),
        "avg_extra_weight": shared_stats.get("avg_extra_weight", 1.0),
        "patterns": shared_stats.get("patterns", 0),
        "type_distribution": shared_stats.get("type_distribution", reasoning_types),
        "shared_score": shared_stats.get("shared_score", 0.0),
        "shared_delta": shared_stats.get("shared_delta", 0.0),
        "shared_raw_score": shared_stats.get("raw_score", 0.0),
        "management_ok": shared_stats.get("management_ok", True),
        "target_current": shared_stats.get("target_current", 0.0),
        "target_score": shared_stats.get("target_score"),
        "target_step": shared_stats.get("target_step", 0.0),
        # PRECOMPUTED SCORE TRACKING (Author: Bo Shang <bo@shang.software>)
        "precomputed_score": shared_stats.get("precomputed_score", 0.0),
        "prev_batch_actual": shared_stats.get("prev_batch_actual", 0.0),
        "prev_batch_expected": shared_stats.get("prev_batch_expected", 0.0),
        "batch_count": shared_stats.get("batch_count", 0),
    })

    return {
        "quality": batch_quality,
        "reasoning_type": dominant_type,
        "reasoning_distribution": reasoning_types,
        "samples_analyzed": len(records),
        "shared_boost": shared_stats.get("shared_boost", 0.0),
        "cross_step_attention": cross_step,
        "answer_grounding": answer_ground,
        "shared_score": shared_stats.get("shared_score", 0.0),
    }


def print_attention_status():
    """Print current attention calculation status."""
    if not AUTO_ATTENTION_AVAILABLE:
        return

    try:
        selector = get_adaptive_selector()
        stats = selector.get_stats()
        print(f"\n{'─' * 60}")
        print(f"ATTENTION CALCULATIONS (Author: Bo Shang <bo@shang.software>)")
        print(f"{'─' * 60}")
        print(f"  Current model: {stats['current_model']}")
        print(f"  Level: {stats['level']}/2 (0=mini, 1=5.2, 2=pro)")
        print(f"  Avg quality: {stats['avg_quality']:.2f}")
        print(f"  Analysis: EVERY batch (continuous)")
        print(f"{'─' * 60}")
    except:
        pass


def print_master_scalar_optimizer_block(
    shared_score: float,
    shared_delta: float,
    coherence_scalar: float,
    safety_score: float,
    safety_weight: float,
    raw_dot: float,
    sample_count: int,
    sampled_count: int,
    sample_confidence: float,
    score_mode: bool,
    target_current: float,
    target_goal: Optional[float],
    batch_num: int,
    total_saved: int,
    target_count: int,
    rate: float,
    eta: float,
    level: int,
    quality: float,
    prev_batch_actual: float,
    prev_batch_expected: float,
    precomputed_score: float,
    cross_step: float,
    answer_ground: float,
    errors: int,
    duplicates: int,
):
    model_label = "deepseek-reasoner"
    if sample_confidence and sample_confidence > 0:
        confidence = sample_confidence
    else:
        base_count = sampled_count or sample_count
        confidence = min(1.0, base_count / 100) if base_count else 0.0
    progress_pct = (100 * total_saved / target_count) if target_count else 0.0

    print(f"\n")
    print(f"  ╔═══════════════════════════════════════════════════════════════════════╗")
    print(f"  ║  ★★★ MASTER SCALAR OPTIMIZER ★★★                                     ║")
    print(f"  ╠═══════════════════════════════════════════════════════════════════════╣")
    print(f"  ║  MASTER SCALAR:    {shared_score:.6f}   (Δ{shared_delta:+.4f})                    ║")
    print(f"  ║  RAW DOT PRODUCT:  {raw_dot:.6f}   (avg pairwise dot)                 ║")
    safety_line = f"COHERENCE: {coherence_scalar:.6f} | SAFETY: {safety_score:.4f} (w {safety_weight:.2f})"
    print(f"  ║  {safety_line:<69}║")
    total_label = f"{sample_count:,}" if sample_count else "0"
    sampled_label = f"{sampled_count:,}" if sampled_count else "?"
    sample_line = f"TRAINING SAMPLES: {total_label} | sampled: {sampled_label} | conf: {confidence:.2f}"
    print(f"  ║  {sample_line:<69}║")
    print(f"  ╠═══════════════════════════════════════════════════════════════════════╣")
    if score_mode and target_goal is not None:
        print(f"  ║  Target: {target_current:.4f}  →  Goal: {target_goal:.4f}                              ║")
    method_line = "Method: master = (1-w)*coherence + w*safety"
    print(f"  ║  {method_line:<69}║")
    print(f"  ╠═══════════════════════════════════════════════════════════════════════╣")
    if score_mode:
        print(f"  ║ BATCH #{batch_num:<5} | rec {total_saved:,} | {rate:.1f}/s                                 ║")
    else:
        print(f"  ║ BATCH #{batch_num:<5} | {total_saved:,}/{target_count:,} ({progress_pct:.1f}%) | {rate:.1f}/s | ETA {int(eta//60)}m{int(eta%60)}s  ║")
    print(f"  ╠═══════════════════════════════════════════════════════════════════════╣")
    print(f"  ║ PERFECT SELF-ATTENTION (Author: Bo Shang <bo@shang.software>)        ║")
    print(f"  ║                                                                       ║")
    print(f"  ║   attention = softmax(QK^T/√d + R)V                                   ║")
    print(f"  ║                                                                       ║")
    print(f"  ║   Where:                                                              ║")
    print(f"  ║     Q = query projection    K = key projection                        ║")
    print(f"  ║     V = value projection    d = head dimension                        ║")
    print(f"  ║     R = reasoning prior matrix (learned from CoT embeddings)          ║")
    print(f"  ║                                                                       ║")
    print(f"  ║   Current Parameters:                                                 ║")
    print(f"  ║     Model: {model_label:<28} Level: {level}/2              ║")
    print(f"  ║     Avg Quality: {quality:.4f}                                             ║")
    if prev_batch_expected > 0:
        perf_delta = prev_batch_actual - prev_batch_expected
        perf_status = "✓" if perf_delta >= 0 else "↑"
        print(f"  ║     Prev batch perf: {prev_batch_actual:.4f} (exp {prev_batch_expected:.4f}) {perf_status}             ║")
    print(f"  ║     Next batch target: {precomputed_score:.4f}                                  ║")
    print(f"  ║     Cross-step attention (r→r): {cross_step:.4f}                           ║")
    print(f"  ║     Answer grounding (a→r):     {answer_ground:.4f}                           ║")
    print(f"  ║                                                                       ║")
    print(f"  ║   Errors: {errors:<6}    Duplicates: {duplicates:<6}                        ║")
    print(f"  ╚═══════════════════════════════════════════════════════════════════════╝")


def print_master_scalar_optimizer_snapshot(target_score: Optional[float]) -> bool:
    """Print a master scalar optimizer snapshot without running generation."""
    score_mode = target_score is not None

    ctx = load_loser_context()
    master_scalar = float(ctx.get("master_scalar", 0.001) or 0.001)
    coherence_scalar = float(ctx.get("coherence_scalar", master_scalar) or master_scalar)
    safety_score = float(ctx.get("safety_score", 0.0) or 0.0)
    safety_weight = float(ctx.get("safety_weight", 0.0) or 0.0)
    raw_dot = float(ctx.get("raw_dot_product", master_scalar) or master_scalar)
    sample_count = int(ctx.get("sample_count", 0) or 0)
    sampled_count = int(ctx.get("sampled_count", 0) or 0)
    sample_confidence = float(ctx.get("sample_confidence", 0.0) or 0.0)

    if MASTER_SCALAR_AVAILABLE and API_KEY:
        try:
            import asyncio
            result = asyncio.run(compute_master_scalar_from_file())
            if result and result.master_scalar > 0.001:
                master_scalar = result.master_scalar
                coherence_scalar = result.coherence_scalar
                safety_score = result.safety_score
                safety_weight = result.safety_weight
                raw_dot = result.raw_dot_product
                sample_count = result.sample_count
                sampled_count = result.sampled_count
                sample_confidence = result.sample_confidence
                save_master_scalar_context(
                    master_scalar,
                    coherence_scalar,
                    safety_score,
                    safety_weight,
                    raw_dot,
                    sample_count,
                    sampled_count=sampled_count,
                    sample_confidence=sample_confidence,
                )
        except Exception as e:
            print(f"  {Colors.YELLOW}[MasterScalar] Compute error: {e}{Colors.NC}")

    if sample_count <= 0 and OUTPUT_FILE.exists():
        try:
            with open(OUTPUT_FILE) as f:
                sample_count = sum(1 for _ in f)
            if sampled_count <= 0:
                sampled_count = min(sample_count, 500)
        except Exception:
            sample_count = 0
            sampled_count = 0

    if sample_confidence <= 0 and raw_dot > 0:
        sample_confidence = max(0.0, min(1.0, coherence_scalar / raw_dot))

    if sample_count > 0:
        save_master_scalar_context(
            master_scalar,
            coherence_scalar,
            safety_score,
            safety_weight,
            raw_dot,
            sample_count,
            sampled_count=sampled_count,
            sample_confidence=sample_confidence,
        )

    _live_attention_stats["shared_score"] = master_scalar
    _live_attention_stats["coherence_scalar"] = coherence_scalar
    _live_attention_stats["safety_score"] = safety_score
    _live_attention_stats["safety_weight"] = safety_weight
    _live_attention_stats["raw_dot_product"] = raw_dot
    _live_attention_stats["sample_count"] = sample_count
    _live_attention_stats["sampled_count"] = sampled_count
    _live_attention_stats["sample_confidence"] = sample_confidence

    cross_step, answer_ground = compute_real_attention_params()

    level = 0
    quality = 0.0
    if AUTO_ATTENTION_AVAILABLE:
        try:
            selector = get_adaptive_selector()
            level = selector.current_level
            quality = selector.get_avg_quality()
        except Exception:
            pass

    prev_batch_actual = _live_attention_stats.get("prev_batch_actual", master_scalar)
    prev_batch_expected = _live_attention_stats.get("prev_batch_expected", 0.0)
    precomputed_score = _live_attention_stats.get("precomputed_score", master_scalar + _cot_weight_tracker.min_step)
    shared_delta = master_scalar - prev_batch_actual
    target_current = _live_attention_stats.get("target_current", target_score or master_scalar)

    print_master_scalar_optimizer_block(
        shared_score=master_scalar,
        shared_delta=shared_delta,
        coherence_scalar=coherence_scalar,
        safety_score=safety_score,
        safety_weight=safety_weight,
        raw_dot=raw_dot,
        sample_count=sample_count,
        sampled_count=sampled_count,
        sample_confidence=sample_confidence,
        score_mode=score_mode,
        target_current=target_current,
        target_goal=target_score,
        batch_num=0,
        total_saved=sample_count,
        target_count=max(sample_count, 1),
        rate=0.0,
        eta=0.0,
        level=level,
        quality=quality,
        prev_batch_actual=prev_batch_actual,
        prev_batch_expected=prev_batch_expected,
        precomputed_score=precomputed_score,
        cross_step=cross_step,
        answer_ground=answer_ground,
        errors=0,
        duplicates=0,
    )

    return sample_count > 0 or master_scalar >= 0.001  # Always succeed with Ollama local


# ════════════════════════════════════════════════════════════════════════════════
# MAIN GENERATION LOOP
# ════════════════════════════════════════════════════════════════════════════════

async def generate_all(target_count: int, category_filter: Optional[str] = None, resume: bool = False,
                       attention_disabled: bool = False, target_score: Optional[float] = None,
                       max_records: Optional[int] = None, min_records: int = 1000):
    """Main generation loop with attention calculations.

    In score-targeted mode:
      - Generates at least min_records new records (default 1000) before checking score
      - Stops when BOTH score >= target AND new records >= min_records
      - Maximum of max_records (default 99999 in score mode)
    """

    DATA_STORE.mkdir(parents=True, exist_ok=True)

    stats = GenerationStats()
    seen_hashes: Set[str] = set()
    seeded_patterns = 0

    # Load existing data for deduplication (ALWAYS - this is persistent storage)
    if OUTPUT_FILE.exists():
        print(f"Loading existing data from {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    content = json.dumps(data.get("messages", []), sort_keys=True)
                    seen_hashes.add(hashlib.md5(content.encode()).hexdigest())
                    stats.total_saved += 1
                    if _cot_weight_tracker.enabled:
                        text = ""
                        messages = data.get("messages", [])
                        if isinstance(messages, list):
                            text = _extract_assistant_text(messages)
                        if not text:
                            text = data.get("text") or data.get("response") or ""
                        if _cot_weight_tracker.observe_text(text):
                            seeded_patterns += 1
                except:
                    pass
        print(f"  Loaded {stats.total_saved} existing records (will append new)")
    else:
        print(f"Creating new data store: {OUTPUT_FILE}")

    if _cot_weight_tracker.enabled:
        rounds_dir = DATA_STORE / "rounds"
        extra_files = []
        if rounds_dir.exists():
            extra_files.extend(sorted(rounds_dir.glob("*.jsonl")))
        extra_files.extend(sorted(DATA_STORE.glob("*.jsonl.bak")))
        for path in extra_files:
            if path.resolve() == OUTPUT_FILE.resolve():
                continue
            seeded_patterns += _seed_cot_tracker_from_file(path)
        if seeded_patterns > 0:
            print(f"  CoT weight tracker seeded with {seeded_patterns:,} patterns")

    # Resume from checkpoint
    if resume and CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            checkpoint = json.load(f)
            stats.total_generated = checkpoint.get("total_generated", 0)
            # Merge checkpoint categories with current categories (preserve all current keys)
            # This ensures new categories added to DOMAINS don't cause KeyError
            checkpoint_categories = checkpoint.get("by_category", {})
            for cat in checkpoint_categories:
                if cat in stats.by_category:
                    stats.by_category[cat] = checkpoint_categories[cat]
        print(f"  Resuming from checkpoint: {stats.total_generated} generated")

    score_mode = target_score is not None
    if score_mode:
        if max_records is None:
            max_records = target_count
        target_count = max_records
        _cot_weight_tracker.target_score = target_score

    new_saved_total = 0

    remaining = target_count - stats.total_saved
    if remaining <= 0:
        if score_mode:
            print(f"Already have {stats.total_saved} records, cap is {target_count}")
        else:
            print(f"Already have {stats.total_saved} records, target is {target_count}")
        return stats

    print(f"\n{'═' * 60}")
    if score_mode:
        print("GENERATING TRAINING RECORDS (MASTER SCALAR TARGET)")
    else:
        print(f"GENERATING {remaining:,} TRAINING RECORDS")
    print(f"{'═' * 60}")
    if score_mode:
        print(f"Master scalar target: {target_score:.4f}")
        print(f"Current master scalar: {_cot_weight_tracker.last_score:.4f}")
        print(f"Records so far (total): {stats.total_saved:,}")
        print(f"New records this run: {new_saved_total:,}")
        print(f"Min new records: {min_records:,}")
        print(f"Max records: {target_count:,}")
    else:
        print(f"Target: {target_count:,}")
        print(f"Existing: {stats.total_saved:,}")
        print(f"To generate: {remaining:,}")
    print(f"Model: {MODEL}")
    if category_filter:
        print(f"Category: {category_filter}")

    # Show attention calculation status
    if not attention_disabled:
        print_attention_status()
    else:
        print(f"\n{'─' * 60}")
        print(f"ATTENTION CALCULATIONS: DISABLED (--no-attention flag)")
        print(f"{'─' * 60}")
    print()

    batch_size = MAX_WORKERS
    batch_num = 0
    start_time = time.time()

    connector = aiohttp.TCPConnector(limit=MAX_WORKERS)
    async with aiohttp.ClientSession(connector=connector) as session:

        with open(OUTPUT_FILE, 'a') as outfile:

            while stats.total_saved < target_count:
                # Generate batch of prompts
                prompts = []
                for _ in range(batch_size):
                    if category_filter:
                        category = category_filter
                    else:
                        category = select_category()
                    prompt = generate_prompt(category)
                    prompts.append((prompt, category))

                # Call API
                records = await generate_batch(prompts, session)
                stats.total_generated += len(prompts)
                batch_num += 1

                # ════════════════════════════════════════════════════════════════
                # ATTENTION CALCULATIONS: Analyze batch with codex-mini periodically
                # ════════════════════════════════════════════════════════════════
                if not attention_disabled and batch_num % ATTENTION_ANALYSIS_INTERVAL == 0:
                    await analyze_batch_attention(records, batch_num, stats.total_saved, session)

                # Save records (deduplicated) and PRINT COT TO TERMINAL
                # (Author: Bo Shang <bo@shang.software>)
                new_saved = 0
                for record in records:
                    h = record.content_hash()
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        outfile.write(record.to_jsonl() + "\n")
                        outfile.flush()  # IMMEDIATE PERSISTENCE - each sample saved instantly
                        stats.total_saved += 1
                        new_saved += 1
                        stats.by_category[record.metadata["category"]] += 1

                        # PRINT FULL TRAINING SAMPLE TO TERMINAL (USER + ASSISTANT)
                        # ALWAYS print EVERY sample for comprehensive generation
                        # (Author: Bo Shang <bo@shang.software>)
                        try:
                            user_prompt = None
                            assistant_response = None
                            if hasattr(record, 'messages') and record.messages:
                                for msg in record.messages:
                                    if msg.get("role") == "user":
                                        user_prompt = msg.get("content", "")
                                    elif msg.get("role") == "assistant":
                                        assistant_response = msg.get("content", "")

                            # ALWAYS print every sample (comprehensive mode)
                            print_training_sample_to_terminal(
                                user_prompt=user_prompt or "(no user prompt)",
                                assistant_response=assistant_response or "(no response)",
                                sample_index=stats.total_saved,
                                action="ADD"
                            )

                            # LIVE UPDATE MASTER SCALAR every 10 samples
                            # Uses new simplified master_scalar module
                            # (Author: Bo Shang <bo@shang.software>)
                            if stats.total_saved % 10 == 0 and MASTER_SCALAR_AVAILABLE:
                                from concurrent.futures import ThreadPoolExecutor
                                def _live_update_master_scalar():
                                    import asyncio
                                    result = asyncio.run(compute_master_scalar_from_file())
                                    return result

                                try:
                                    with ThreadPoolExecutor(max_workers=10) as ex:
                                        future = ex.submit(_live_update_master_scalar)
                                        result = future.result(timeout=60)
                                        if result and result.master_scalar > 0.001:
                                            # Store ALL 3 values together
                                            _live_attention_stats["shared_score"] = result.master_scalar
                                            _live_attention_stats["raw_dot_product"] = result.raw_dot_product
                                            _live_attention_stats["sample_count"] = result.sample_count
                                            _live_attention_stats["sample_confidence"] = result.sample_confidence
                                            _live_attention_stats["sampled_count"] = result.sampled_count
                                            # Print ALL 3 values
                                            print(f"\n  {Colors.GREEN}{'═'*60}{Colors.NC}")
                                            print(f"  {Colors.GREEN}★ MASTER: {result.master_scalar:.6f} | DOT: {result.raw_dot_product:.6f} | N: {result.sample_count} (sampled {result.sampled_count}){Colors.NC}")
                                            print(f"  {Colors.GREEN}{'═'*60}{Colors.NC}\n")
                                except Exception as e:
                                    pass  # Don't block on timeout
                        except Exception as e:
                            pass  # Don't crash on print errors
                    else:
                        stats.duplicates_skipped += 1

                if new_saved:
                    new_saved_total += new_saved

                outfile.flush()

                # Progress with attention calculation status and full math
                elapsed = time.time() - start_time
                rate = stats.total_saved / elapsed if elapsed > 0 else 0
                eta = (target_count - stats.total_saved) / rate if rate > 0 else 0

                # Track records for REAL score computation
                _cot_weight_tracker.add_records(new_saved)

                # Get current attention parameters
                level = 0
                quality = 0.0
                model_name = "mini"

                # REAL MASTER SCALAR AND ATTENTION PARAMS (Author: Bo Shang <bo@shang.software>)
                # Score reflects actual embedding similarity from training data
                # Cross-step and answer grounding from actual reasoning type distribution
                shared_score = _live_attention_stats.get("shared_score", 0.0)
                shared_delta = _live_attention_stats.get("shared_delta", 0.0)
                target_current = _live_attention_stats.get("target_current", 0.0)
                target_goal = _live_attention_stats.get("target_score", target_score)
                cross_step, answer_ground = compute_real_attention_params()

                # Update _live_attention_stats with real attention params
                _live_attention_stats["cross_step"] = cross_step
                _live_attention_stats["answer_ground"] = answer_ground

                # PRECOMPUTED SCORE TRACKING
                precomputed_score = _live_attention_stats.get("precomputed_score", shared_score)
                prev_batch_actual = _live_attention_stats.get("prev_batch_actual", 0.0)
                prev_batch_expected = _live_attention_stats.get("prev_batch_expected", 0.0)
                batch_count = _live_attention_stats.get("batch_count", 0)

                if AUTO_ATTENTION_AVAILABLE and not attention_disabled:
                    try:
                        selector = get_adaptive_selector()
                        level = selector.current_level
                        quality = selector.get_avg_quality()
                        model_name = ["mini", "5.2", "pro"][level]
                    except:
                        pass

                # cross_step and answer_ground already computed from compute_real_attention_params()
                # They reflect actual reasoning type distribution from the training data

                # Get REAL master scalar from loser context (computed from embeddings)
                # (Author: Bo Shang <bo@shang.software>)
                real_master_scalar = shared_score
                try:
                    loser_ctx_file = DATA_STORE / "loser_context.json"
                    if loser_ctx_file.exists():
                        with open(loser_ctx_file, 'r') as lf:
                            loser_ctx = json.load(lf)
                            real_master_scalar = loser_ctx.get("master_scalar", shared_score)
                            if real_master_scalar > 0.01:  # Use real value if valid
                                shared_score = real_master_scalar
                                _live_attention_stats["shared_score"] = shared_score  # Critical for exit check
                                _live_attention_stats["coherence_scalar"] = loser_ctx.get(
                                    "coherence_scalar",
                                    _live_attention_stats.get("coherence_scalar", shared_score)
                                )
                                _live_attention_stats["safety_score"] = loser_ctx.get(
                                    "safety_score",
                                    _live_attention_stats.get("safety_score", 0.0)
                                )
                                _live_attention_stats["safety_weight"] = loser_ctx.get(
                                    "safety_weight",
                                    _live_attention_stats.get("safety_weight", 0.0)
                                )
                except:
                    pass

                # Print progress with batch number - ALWAYS show 3 key values
                # (Author: Bo Shang <bo@shang.software>)
                raw_dot = _live_attention_stats.get("raw_dot_product", shared_score)
                coherence_scalar_live = _live_attention_stats.get("coherence_scalar", shared_score)
                safety_score_live = _live_attention_stats.get("safety_score", 0.0)
                safety_weight_live = _live_attention_stats.get("safety_weight", 0.0)
                sample_count_live = _live_attention_stats.get("sample_count", stats.total_saved)
                sampled_count_live = _live_attention_stats.get("sampled_count", 0)
                sample_confidence_live = _live_attention_stats.get("sample_confidence", 0.0)
                print_master_scalar_optimizer_block(
                    shared_score=shared_score,
                    shared_delta=shared_delta,
                    coherence_scalar=coherence_scalar_live,
                    safety_score=safety_score_live,
                    safety_weight=safety_weight_live,
                    raw_dot=raw_dot,
                    sample_count=sample_count_live,
                    sampled_count=sampled_count_live,
                    sample_confidence=sample_confidence_live,
                    score_mode=score_mode,
                    target_current=target_current,
                    target_goal=target_goal,
                    batch_num=batch_num,
                    total_saved=stats.total_saved,
                    target_count=target_count,
                    rate=rate,
                    eta=eta,
                    level=level,
                    quality=quality,
                    prev_batch_actual=prev_batch_actual,
                    prev_batch_expected=prev_batch_expected,
                    precomputed_score=precomputed_score,
                    cross_step=cross_step,
                    answer_ground=answer_ground,
                    errors=stats.errors,
                    duplicates=stats.duplicates_skipped,
                )

                # Save checkpoint (persistent)
                checkpoint_data = {
                    "total_generated": stats.total_generated,
                    "total_saved": stats.total_saved,
                    "by_category": stats.by_category,
                    "timestamp": datetime.now().isoformat(),
                    "batch_num": batch_num,
                }

                # Include attention stats in checkpoint
                if AUTO_ATTENTION_AVAILABLE and not attention_disabled:
                    try:
                        selector = get_adaptive_selector()
                        checkpoint_data["attention"] = {
                            "level": selector.current_level,
                            "model": selector.get_current_model(),
                            "avg_quality": selector.get_avg_quality(),
                        }
                    except:
                        pass

                with open(CHECKPOINT_FILE, 'w') as f:
                    json.dump(checkpoint_data, f)

                # GCP backup every 100 batches (persistent cloud storage)
                if batch_num % 100 == 0 and batch_num > 0:
                    try:
                        import subprocess
                        gcs_bucket = os.environ.get("GCS_BUCKET", "erosolar-training-data")
                        # Backup checkpoint
                        subprocess.run(
                            ["gsutil", "-q", "cp", str(CHECKPOINT_FILE),
                             f"gs://{gcs_bucket}/checkpoints/generation_checkpoint.json"],
                            timeout=30, capture_output=True
                        )
                        # Backup data file
                        subprocess.run(
                            ["gsutil", "-q", "cp", str(OUTPUT_FILE),
                             f"gs://{gcs_bucket}/latest/generated_training_data.jsonl"],
                            timeout=300, capture_output=True
                        )
                    except:
                        pass  # Non-blocking backup

                if score_mode:
                    current_score = _live_attention_stats.get("shared_score", 0.0)
                    # Must have BOTH: score target met AND minimum new records generated
                    # (Author: Bo Shang <bo@shang.software>)
                    if current_score >= target_score and new_saved_total >= min_records:
                        print(f"\n  ✓ Score target {target_score:.4f} reached with {new_saved_total:,} new records (min: {min_records:,})")
                        break

    # Final stats
    print(f"\n\n{'═' * 60}")
    print("GENERATION COMPLETE")
    print(f"{'═' * 60}")
    print(f"Total saved: {stats.total_saved:,}")
    print(f"New saved this run: {new_saved_total:,}")
    print(f"Duplicates skipped: {stats.duplicates_skipped:,}")
    print(f"Errors: {stats.errors:,}")
    print(f"Batches processed: {batch_num:,}")
    print(f"\nBy category:")
    for cat, count in sorted(stats.by_category.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {cat:<15} {count:>6,} ({100*count/stats.total_saved:.1f}%)")

    # Final attention calculation stats
    if AUTO_ATTENTION_AVAILABLE and not attention_disabled:
        try:
            selector = get_adaptive_selector()
            stats_dict = selector.get_stats()
            print(f"\n{'─' * 60}")
            print(f"ATTENTION CALCULATIONS FINAL STATS")
            print(f"{'─' * 60}")
            print(f"  Final model: {stats_dict['current_model']}")
            print(f"  Level: {stats_dict['level']}/2")
            print(f"  Avg quality: {stats_dict['avg_quality']:.2f}")
            print(f"  Attention analyses: {batch_num // ATTENTION_ANALYSIS_INTERVAL}")
        except:
            pass

    print(f"\nOutput: {OUTPUT_FILE}")

    # Final GCP backup (persistent cloud storage)
    print(f"\nBacking up to GCS...")
    try:
        gcs_bucket = os.environ.get("GCS_BUCKET", "erosolar-training-data")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Backup final data file
        subprocess.run(
            ["gsutil", "-q", "cp", str(OUTPUT_FILE),
             f"gs://{gcs_bucket}/latest/generated_training_data.jsonl"],
            timeout=300, capture_output=True
        )
        # Backup timestamped snapshot
        subprocess.run(
            ["gsutil", "-q", "cp", str(OUTPUT_FILE),
             f"gs://{gcs_bucket}/snapshots/{timestamp}/generated_training_data.jsonl"],
            timeout=300, capture_output=True
        )
        # Backup checkpoint
        subprocess.run(
            ["gsutil", "-q", "cp", str(CHECKPOINT_FILE),
             f"gs://{gcs_bucket}/checkpoints/generation_checkpoint.json"],
            timeout=30, capture_output=True
        )
        print(f"  ✓ Backed up to gs://{gcs_bucket}/")
    except Exception as e:
        print(f"  GCS backup warning: {e}")

    # Save manifest (persistent metadata about the data store)
    manifest = {
        "last_updated": datetime.now().isoformat(),
        "total_records": stats.total_saved,
        "by_category": stats.by_category,
        "model": MODEL,
        "data_file": str(OUTPUT_FILE),
        "format": {
            "type": "jsonl",
            "schema": {
                "messages": [{"role": "user|assistant", "content": "string"}],
                "metadata": {"category": "string", "has_thinking": "bool"}
            },
            "special_tokens": ["<|think_start|>", "<|think_end|>", "<|step|>", "<|answer|>"]
        }
    }
    with open(MANIFEST_FILE, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {MANIFEST_FILE}")

    # Save seed usage tracker
    tracker = get_tracker()
    if tracker:
        tracker.save()
        tracker_stats = tracker.get_stats()
        print(f"\nSeed Usage: {tracker_stats['total_unique_seeds']} unique seeds used, "
              f"{tracker_stats['total_uses']} total uses")

    return stats


# ════════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════════

def save_round_file(round_num: int, batch_size: int = 10000):
    """Extract records for a specific round into a separate file."""
    rounds_dir = DATA_STORE / "rounds"
    rounds_dir.mkdir(parents=True, exist_ok=True)

    round_file = rounds_dir / f"round_{round_num:02d}.jsonl"
    start_idx = (round_num - 1) * batch_size
    end_idx = round_num * batch_size

    if not OUTPUT_FILE.exists():
        print(f"No data file found: {OUTPUT_FILE}")
        return

    # Read all records and extract the round's slice
    with open(OUTPUT_FILE, 'r') as f:
        all_records = f.readlines()

    round_records = all_records[start_idx:end_idx]

    with open(round_file, 'w') as f:
        f.writelines(round_records)

    print(f"Saved round {round_num} to {round_file} ({len(round_records)} records)")
    return round_file


def main():
    global MAX_WORKERS, MODEL, API_KEY, ATTENTION_ANALYSIS_INTERVAL
    global MAX_OUTPUT_TOKENS, LONG_FORM, SYSTEM_PROMPT

    parser = argparse.ArgumentParser(description="Generate training data using Response API")
    parser.add_argument("--target", type=int, default=10000,
                        help="Target number of records (or max cap if --target-score is set)")
    parser.add_argument("--target-score", type=float, default=None,
                        help="Target master scalar (enables score-target mode)")
    parser.add_argument("--max-records", type=int, default=None,
                        help="Max records cap when using --target-score (default: 99999)")
    parser.add_argument("--min-records", type=int, default=1000,
                        help="Minimum new records before checking score target (default: 1000)")
    parser.add_argument("--category", type=str, choices=list(CATEGORIES.keys()), help="Only generate this category")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--fresh", action="store_true",
                        help="Clear all cached state for fresh loser-driven generation (recommended)")
    parser.add_argument("--model", type=str, default="deepseek-reasoner", help="Model to use")
    parser.add_argument("--max-output-tokens", type=int, default=None,
                        help="Max output tokens per response (overrides defaults)")
    parser.add_argument("--long-form", dest="long_form", action="store_true", default=LONG_FORM,
                        help="Enable long-form mode (expansive answers)")
    parser.add_argument("--no-long-form", dest="long_form", action="store_false",
                        help="Disable long-form mode")
    parser.add_argument("--round", type=int, help="Round number (saves to separate round file after generation)")
    parser.add_argument("--batch-size", type=int, default=10000, help="Records per round (for --round)")
    # Attention calculations
    parser.add_argument("--attention-interval", type=int, default=10,
                        help="Run attention calculations every N batches (default: 10)")
    parser.add_argument("--no-attention", action="store_true",
                        help="Disable attention calculations")
    parser.add_argument("--print-master-scalar", action="store_true",
                        help="Print master scalar optimizer snapshot and exit")
    parser.add_argument("--verbose", action="store_true",
                        help="Print EVERY training sample CoT to terminal (default: true)")
    args = parser.parse_args()

    # Verbose is TRUE by default for comprehensive generation
    args.verbose = True

    MODEL = args.model
    ATTENTION_ANALYSIS_INTERVAL = args.attention_interval

    LONG_FORM = bool(args.long_form)
    if LONG_FORM:
        MAX_OUTPUT_TOKENS = LONG_FORM_OUTPUT_TOKENS
        if "LONG-FORM MODE" not in SYSTEM_PROMPT:
            SYSTEM_PROMPT += (
                "\n\nLONG-FORM MODE:\n"
                "- Provide expansive, detailed answers with examples and edge cases.\n"
                "- Use available output capacity without adding filler."
            )
    else:
        MAX_OUTPUT_TOKENS = DEFAULT_MAX_OUTPUT_TOKENS

    if args.max_output_tokens:
        MAX_OUTPUT_TOKENS = args.max_output_tokens

    # Reload API key in case it was set after import (default to ollama for local)
    API_KEY = os.environ.get("OPENAI_API_KEY", os.environ.get("DEEPSEEK_API_KEY", "ollama"))

    # CRITICAL: Clear cached state for fresh loser-driven generation
    # This ensures we use loser-driven prompts instead of old fallback prompts
    loser_ctx_exists = (DATA_STORE / "loser_context.json").exists()
    if args.fresh or (loser_ctx_exists and not args.resume):
        print(f"\n{'='*60}")
        print("[STARTUP] Clearing cached state for loser-driven generation...")
        print(f"{'='*60}")
        cache_files_to_clear = [
            DATA_STORE / "generation_checkpoint.json",
            DATA_STORE / "cot_optimization_state.json",
            DATA_STORE / "cot_patterns.json",
        ]
        for cache_file in cache_files_to_clear:
            if cache_file.exists():
                try:
                    cache_file.unlink()
                    print(f"  Cleared: {cache_file.name}")
                except Exception as e:
                    print(f"  Warning: Could not clear {cache_file.name}: {e}")
        # Reset global state for loser prompt generation
        global _loser_prompt_cache, _current_loser_idx, _complexity_level
        _loser_prompt_cache = {"prompt": None, "loser_info": None, "remaining": 0}
        _current_loser_idx = 0
        _complexity_level = 0
        print("[STARTUP] Ready for fresh loser-driven generation\n")

    if args.print_master_scalar:
        ok = print_master_scalar_optimizer_snapshot(args.target_score)
        if not ok:
            print("ERROR: Unable to compute master scalar (need data + API key).")
            sys.exit(1)
        return

    if not API_KEY:
        print("ERROR: OPENAI_API_KEY not set")
        print("export OPENAI_API_KEY='sk-...'")
        return

    print(f"Using model: {MODEL} (max capacity: 4096 tokens)")

    # Loser/friend strategy overview
    print(f"\n{'═' * 60}")
    print("★★★ LOSER/FRIEND STRATEGY (primary generation mode) ★★★")
    print(f"{'═' * 60}")
    print("  Uses loser/friend prompts when available; falls back to deterministic prompts.")
    print("  For best results, use generate_friends_for_losers():")
    print("    1. embed_all_training_samples()")
    print("    2. analyze_cot_patterns()")
    print("    3. generate_friends_for_losers()  <-- THE CORE LOOP")
    print()

    max_records = args.max_records
    if args.target_score is not None and max_records is None:
        max_records = 99999  # Default max for score-targeted mode

    asyncio.run(generate_all(
        args.target,
        args.category,
        args.resume,
        attention_disabled=args.no_attention,
        target_score=args.target_score,
        max_records=max_records,
        min_records=args.min_records,
    ))

    # Save round file if requested
    if args.round:
        save_round_file(args.round, args.batch_size)


if __name__ == "__main__":
    main()
