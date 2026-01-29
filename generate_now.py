#!/usr/bin/env python3
"""
Ollama-based training data generator.
Generates friends for losers to reduce master scalar.

Usage:
  python generate_now.py                    # Use saved losers, default samples
  python generate_now.py 500 2             # 500 samples, 2 threads
  python generate_now.py --target 0.06      # Generate until master scalar < 0.06
"""
import os
import sys
import json
import httpx
import threading
import fcntl
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ["PYTHONUNBUFFERED"] = "1"

OUTPUT_PATH = "data_store/generated_training_data.jsonl"
LOSER_CONTEXT_PATH = "data_store/loser_context.json"
LOCK = threading.Lock()
COUNTER = {"generated": 0, "errors": 0, "tokens": 0, "last_success": time.time()}
PAUSE_ALERT_SECONDS = 120  # Alert if no progress for 120 seconds (local model is slower)

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
OLLAMA_MODEL = "deepseek-r1:8b"  # Larger model - single thread

def load_losers():
    """Load loser texts from context file."""
    if Path(LOSER_CONTEXT_PATH).exists():
        with open(LOSER_CONTEXT_PATH) as f:
            ctx = json.load(f)
            return ctx.get("loser_texts", [])
    return []

def append_jsonl(record: dict):
    """Thread-safe append to JSONL."""
    line = json.dumps(record) + "\n"
    with open(OUTPUT_PATH, 'a') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def generate_friend(loser_idx: int, friend_idx: int, loser_text: str) -> bool:
    """Generate a friend sample for a loser."""
    topic = loser_text[:300].replace('\n', ' ')

    strategies = [
        "with edge cases and boundary conditions",
        "requiring multi-step logical reasoning",
        "involving trade-offs and constraints",
        "with real-world complications",
        "requiring precise numerical analysis",
        "comparing multiple valid approaches",
        "involving temporal or causal reasoning",
        "requiring synthesis of multiple concepts",
        "with adversarial or trick elements",
        "requiring first-principles derivation"
    ]
    strategy = strategies[(loser_idx + friend_idx) % len(strategies)]

    question = f"""Based on this topic: {topic}

Create and solve a challenging problem {strategy}.
Think through each step carefully, showing your complete reasoning process."""

    try:
        resp = httpx.post(
            OLLAMA_URL,
            headers={"Authorization": "Bearer ollama", "Content-Type": "application/json"},
            json={"model": OLLAMA_MODEL, "messages": [{"role": "user", "content": question}], "max_tokens": 2048},
            timeout=300.0
        )
        resp.raise_for_status()
        data = resp.json()

        msg = data["choices"][0]["message"]
        answer = msg.get("content", "") or ""
        tokens = data.get("usage", {}).get("total_tokens", 0)

        # Parse thinking from deepseek-r1 output (uses <think>...</think> tags)
        import re
        think_match = re.search(r'<think>(.*?)</think>', answer, re.DOTALL)
        if think_match:
            reasoning = think_match.group(1).strip()
            answer_text = answer[think_match.end():].strip()
            steps = reasoning.split('\n\n')
            formatted = "<|step|>".join(s.strip() for s in steps if s.strip())
            content = f"<|think_start|>{formatted}<|think_end|><|answer|>{answer_text}"
        else:
            content = f"<|answer|>{answer.strip()}"

        record = {
            "messages": [{"role": "user", "content": question}, {"role": "assistant", "content": content}],
            "metadata": {
                "source": "ollama-deepseek-r1",
                "loser_idx": loser_idx,
                "friend_idx": friend_idx,
                "strategy": strategy,
                "tokens": tokens,
                "ts": datetime.now().isoformat()
            }
        }

        append_jsonl(record)

        with LOCK:
            COUNTER["generated"] += 1
            COUNTER["tokens"] += tokens
            COUNTER["last_success"] = time.time()
            n = COUNTER["generated"]
            errs = COUNTER["errors"]

        print(f"[{n}] L{loser_idx}F{friend_idx} {tokens}tok (errs:{errs})", flush=True)
        return True

    except Exception as e:
        with LOCK:
            COUNTER["errors"] += 1
            errs = COUNTER["errors"]
            last = COUNTER["last_success"]

        elapsed = time.time() - last
        error_msg = str(e)[:80]
        print(f"x L{loser_idx}F{friend_idx}: {error_msg} (errs:{errs})", flush=True)

        # ALERT if paused too long
        if elapsed > PAUSE_ALERT_SECONDS:
            print(f"\n  ALERT: No successful generation for {elapsed:.0f}s! Check Ollama.", flush=True)
            print(f"  Last error: {error_msg}", flush=True)

        return False

def get_current_master_scalar():
    """Compute current master scalar."""
    try:
        from master_scalar import analyze_losers_sync
        result = analyze_losers_sync(max_samples=500)
        return result.master_scalar
    except Exception as e:
        print(f"Error computing master scalar: {e}")
        return None

def main():
    # Parse args
    target_scalar = None
    num_samples = 500
    num_threads = 1  # 8b model - single thread for stability
    friends_per_loser = 10

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--target" and i + 1 < len(args):
            target_scalar = float(args[i + 1])
            i += 2
        elif args[i].isdigit():
            if num_samples == 500:
                num_samples = int(args[i])
            else:
                num_threads = int(args[i])
            i += 1
        else:
            i += 1

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    before = sum(1 for _ in open(OUTPUT_PATH)) if Path(OUTPUT_PATH).exists() else 0

    # Load losers
    loser_texts = load_losers()
    if not loser_texts:
        print("No loser_context.json found. Using default topics.")
        loser_texts = [
            "algorithm complexity analysis and Big-O notation",
            "database query optimization and indexing strategies",
            "distributed systems consistency and CAP theorem",
            "machine learning model selection and hyperparameter tuning",
            "API design patterns and RESTful best practices",
            "memory management and garbage collection strategies",
            "concurrency patterns and race condition prevention",
            "security vulnerability analysis and mitigation",
            "performance profiling and bottleneck identification",
            "code refactoring techniques and design patterns"
        ] * 5  # 50 topics

    num_losers = len(loser_texts)

    print(f"Ollama: {OLLAMA_URL} ({OLLAMA_MODEL})")
    print(f"Records before: {before}")
    print(f"Losers: {num_losers}")
    if target_scalar:
        print(f"Target master scalar: {target_scalar}")
        current = get_current_master_scalar()
        if current:
            print(f"Current master scalar: {current:.6f}")
    print(f"Generating: {num_samples} samples ({num_losers} losers × {friends_per_loser} friends max)")
    print(f"Threads: {num_threads}")
    print("=" * 50, flush=True)

    # Monitor thread for pause detection
    stop_monitor = threading.Event()

    def monitor_progress():
        last_count = 0
        while not stop_monitor.is_set():
            time.sleep(30)  # Check every 30 seconds
            with LOCK:
                current_count = COUNTER["generated"]
                errors = COUNTER["errors"]
                last_success = COUNTER["last_success"]

            elapsed = time.time() - last_success
            rate = (current_count - last_count) / 30 if current_count > last_count else 0

            if elapsed > PAUSE_ALERT_SECONDS and current_count == last_count:
                print(f"\n⚠️  ALERT: STALLED! No progress for {elapsed:.0f}s. gen={current_count} errs={errors}", flush=True)
            elif current_count > last_count:
                print(f"📊 Progress: {current_count} generated, {errors} errors, {rate:.1f}/s", flush=True)

            last_count = current_count

    monitor = threading.Thread(target=monitor_progress, daemon=True)
    monitor.start()

    # Generate friends for losers
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        sample_idx = 0

        for li in range(num_losers):
            for fi in range(friends_per_loser):
                if sample_idx >= num_samples:
                    break
                futures.append(executor.submit(generate_friend, li, fi, loser_texts[li]))
                sample_idx += 1
            if sample_idx >= num_samples:
                break

        for f in as_completed(futures):
            pass

    stop_monitor.set()

    after = sum(1 for _ in open(OUTPUT_PATH))

    print("=" * 50)
    print(f"Generated: {COUNTER['generated']} | Errors: {COUNTER['errors']} | Tokens: {COUNTER['tokens']}")
    print(f"Records: {before} → {after} (+{after - before})")

    # Check if target reached
    if target_scalar:
        new_scalar = get_current_master_scalar()
        if new_scalar:
            print(f"New master scalar: {new_scalar:.6f}")
            if new_scalar <= target_scalar:
                print(f"✓ TARGET REACHED! {new_scalar:.6f} <= {target_scalar}")
            else:
                print(f"✗ Target not reached. Need more training data or training.")

if __name__ == "__main__":
    main()
