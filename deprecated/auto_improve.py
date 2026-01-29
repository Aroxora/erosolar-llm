#!/usr/bin/env python3
"""
Auto-Improvement System for Kids Model Training Data.

When prompts fail scoring, this module:
1. Generates correct answers using GPT
2. Adds Q&A pairs to data.py
3. Reloads the knowledge base
4. Logs improvements for tracking

This creates a self-improving feedback loop.
"""

import os
import sys
import json
from datetime import datetime
from typing import Optional, Dict

# File paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "data.py")
IMPROVEMENTS_LOG = os.path.join(SCRIPT_DIR, "improvements_log.json")

# Configuration
AUTO_IMPROVE_THRESHOLD = 8.0  # Score below this triggers auto-improvement (high standard)
AUTO_IMPROVE_ENABLED = True   # Toggle auto-improvement on/off

# Import weight manager for dynamic weight adjustments
try:
    from weight_manager import increase_category_weight, get_weight_stats
    WEIGHT_MANAGER_AVAILABLE = True
except ImportError:
    WEIGHT_MANAGER_AVAILABLE = False


def get_openai_client():
    """Get OpenAI client (lazy import to avoid circular deps)."""
    from openai import OpenAI
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def generate_correct_answer(prompt: str, category: str = "", failed_response: str = "",
                            feedback: str = "") -> Optional[str]:
    """Use GPT-5.1-codex-mini to generate maximum quality training data for failed prompts."""

    system_prompt = """You are the world's leading expert at creating PERFECT training data for AI models.
Your mission: Generate the IDEAL response that would score 10/10 and serve as gold-standard training data.

MAXIMIZE THESE QUALITIES:
1. ACCURACY: 100% factually correct - triple-check all facts, numbers, dates, names
2. COMPLETENESS: Fully answer the question with all relevant information
3. CLARITY: Crystal clear explanation that leaves no confusion
4. EDUCATIONAL VALUE: Include key concepts, context, and learning opportunities
5. AGE-APPROPRIATE: Match language complexity to the question's education level
6. ENGAGING: Warm, encouraging tone that sparks curiosity
7. PRACTICAL: Include examples, analogies, or step-by-step explanations when helpful

FOR SPECIFIC QUESTION TYPES:
- MATH: Show the answer, explain the method, and verify the solution
- SCIENCE: Explain the concept, give real-world examples, connect to everyday life
- HISTORY: Include dates, key figures, significance, and lasting impact
- PATTERNS/SEQUENCES: Identify the rule, show how it applies, predict next values
- CREATIVE: Provide an excellent example that demonstrates the request
- FACTUAL: Give accurate information with memorable details

CRITICAL RULES:
- NEVER be vague or generic - be specific and directly answer what was asked
- NEVER give unrelated information - stay focused on the actual question
- Include enough detail to be genuinely educational
- Make it memorable and interesting

FORMAT RULES (VERY IMPORTANT):
- Write in PLAIN TEXT only - NO markdown formatting
- NO asterisks, NO bold (**), NO italics (*), NO code blocks (`)
- NO bullet points or numbered lists - use flowing sentences
- Keep response as a single paragraph or 2-3 connected sentences
- This will be used as training data, so plain text only

Return ONLY the answer text - no quotes, no prefixes, just the perfect response in plain text."""

    user_prompt = f"""Generate MAXIMUM QUALITY training data for this question.

QUESTION: {prompt}
CATEGORY: {category}

{f"THE PREVIOUS RESPONSE FAILED: {failed_response}" if failed_response else ""}
{f"WHY IT FAILED: {feedback}" if feedback else ""}

Generate the PERFECT answer that will train the model to respond correctly to this and similar questions:"""

    try:
        client = get_openai_client()
        response = client.responses.create(
            model="gpt-5.1-codex-mini",  # Use GPT-5.1-codex-mini for maximum quality training data
            instructions=system_prompt,
            input=user_prompt,
            temperature=0.4,  # Slightly higher for better variety while maintaining accuracy
            max_output_tokens=500  # Allow longer, more comprehensive answers
        )
        return response.output_text.strip()
    except Exception as e:
        print(f"   ⚠️ Failed to generate correct answer: {e}")
        return None


def sanitize_for_training(text: str) -> str:
    """Sanitize text for use as Python string in training data."""
    # Remove markdown formatting
    text = text.replace('**', '')
    text = text.replace('*', '')
    text = text.replace('`', '')

    # Convert newlines to spaces (single line for Python string)
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')

    # Collapse multiple spaces
    while '  ' in text:
        text = text.replace('  ', ' ')

    # Remove bullet points and list markers
    import re
    text = re.sub(r'^\s*[-•]\s*', '', text)
    text = re.sub(r'\s+[-•]\s+', ' ', text)

    # Clean up any remaining formatting artifacts
    text = text.strip()

    return text


def find_similar_qa(content: str, prompt_clean: str, threshold: float = 0.7) -> tuple:
    """Find similar existing Q&A in data.py content.

    Returns: (found, line_number, old_question, old_answer, match_score)
    """
    import re
    from difflib import SequenceMatcher

    # Extract all Q&A pairs with their line numbers
    lines = content.split('\n')
    qa_pattern = re.compile(r'^\s*\("([^"]+)",\s*"([^"]+)"\),?\s*$')

    best_match = (False, 0, "", "", 0.0)

    for i, line in enumerate(lines, 1):
        match = qa_pattern.match(line)
        if match:
            old_q = match.group(1)
            old_a = match.group(2)

            # Calculate similarity
            similarity = SequenceMatcher(None, prompt_clean.lower(), old_q.lower()).ratio()

            if similarity > best_match[4] and similarity >= threshold:
                best_match = (True, i, old_q, old_a, similarity)

    return best_match


def add_qa_to_data(prompt: str, answer: str, category: str = "") -> dict:
    """Add or update Q&A pair in data.py's KIDS_QA list with verification.

    Returns dict with: success, action ('added'/'updated'/'failed'), line_number, old_answer (if updated)
    """
    result = {"success": False, "action": "failed", "line_number": 0, "old_answer": None}

    if not prompt or not answer:
        return result

    # Sanitize the answer for training data (remove markdown, newlines, etc.)
    answer_clean = sanitize_for_training(answer)
    prompt_clean = sanitize_for_training(prompt)

    # Escape quotes in the strings for Python
    prompt_escaped = prompt_clean.replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'")
    answer_escaped = answer_clean.replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'")

    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for similar existing Q&A
        found, line_num, old_q, old_a, similarity = find_similar_qa(content, prompt_clean)

        if found and similarity >= 0.8:
            # UPDATE existing entry (replace old answer with better one)
            print(f"   🔄 Found similar Q&A at line {line_num} (similarity: {similarity:.0%})", flush=True)
            print(f"   📝 Old Q: {old_q[:80]}...", flush=True)
            print(f"   📝 Old A: {old_a[:80]}...", flush=True)

            # Replace the old entry
            old_entry = f'("{old_q}", "{old_a}")'
            new_qa = f'("{prompt_escaped}", "{answer_escaped}")'

            if old_entry in content:
                new_content = content.replace(old_entry, new_qa, 1)
                result["action"] = "updated"
                result["line_number"] = line_num
                result["old_answer"] = old_a
            else:
                # Fallback to adding new if replacement fails
                found = False

        if not found or result["action"] != "updated":
            # ADD new entry
            marker = "]\n\n# VOCABULARY DEFINITIONS"
            if marker in content:
                # Count lines to find insertion point
                insert_pos = content.find(marker)
                lines_before = content[:insert_pos].count('\n') + 1

                new_entry = f'\n    # Auto-added {datetime.now().strftime("%Y-%m-%d %H:%M")} - Category: {category}\n'
                new_entry += f'    ("{prompt_escaped}", "{answer_escaped}"),\n'

                new_content = content[:insert_pos] + new_entry + content[insert_pos:]
                result["action"] = "added"
                result["line_number"] = lines_before + 2  # +2 for comment and entry
            else:
                print(f"   ⚠️ Could not find insertion point in data.py", flush=True)
                return result

        # VERIFY: Check if the new content is valid Python before saving
        try:
            compile(new_content, DATA_FILE, 'exec')
        except SyntaxError as e:
            print(f"   ⚠️ Generated invalid Python syntax: {e}", flush=True)
            print(f"   ⚠️ Skipping this Q&A pair to protect data.py", flush=True)
            return result

        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            f.write(new_content)

        result["success"] = True
        return result

    except Exception as e:
        print(f"   ⚠️ Failed to update data.py: {e}", flush=True)
        return result


def reload_knowledge_base() -> bool:
    """Reload the knowledge base after adding new Q&A pairs."""
    try:
        import importlib
        # Clear cached modules
        for mod_name in ['data', 'kids_retrieval', 'kids_smart_responder']:
            if mod_name in sys.modules:
                importlib.reload(sys.modules[mod_name])
        print("   🔄 Knowledge base reloaded")
        return True
    except Exception as e:
        print(f"   ⚠️ Failed to reload knowledge base: {e}")
        return False


def log_improvement(prompt: str, failed_response: str, correct_answer: str,
                   score: float, category: str) -> None:
    """Log improvements for tracking."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "failed_response": failed_response[:200] if failed_response else "",
        "correct_answer": correct_answer,
        "original_score": score,
        "category": category
    }

    try:
        # Load existing log or create new
        if os.path.exists(IMPROVEMENTS_LOG):
            with open(IMPROVEMENTS_LOG, 'r') as f:
                log = json.load(f)
        else:
            log = []

        log.append(log_entry)

        with open(IMPROVEMENTS_LOG, 'w') as f:
            json.dump(log, f, indent=2)
    except Exception as e:
        print(f"   ⚠️ Failed to log improvement: {e}")


def auto_improve(prompt: str, failed_response: str, score: float,
                 category: str = "", feedback: str = "") -> bool:
    """
    Automatically improve training data when a prompt fails.

    1. Generate correct answer using GPT
    2. Add Q&A pair to data.py
    3. Reload knowledge base
    4. Log the improvement

    Returns True if improvement was successful.
    """
    if not AUTO_IMPROVE_ENABLED:
        return False

    if score >= AUTO_IMPROVE_THRESHOLD:
        return False  # Score is good enough, no improvement needed

    print("\n" + "=" * 70, flush=True)
    print(f"🔧 AUTO-IMPROVE TRIGGERED - Score {score:.1f}/10 < threshold {AUTO_IMPROVE_THRESHOLD}", flush=True)
    print("=" * 70, flush=True)
    print(f"   📋 Category: {category}", flush=True)
    print(f"   ❓ Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}", flush=True)
    print(f"   ❌ Failed Response: {failed_response[:100]}{'...' if len(failed_response) > 100 else ''}", flush=True)

    # Step 1: Adjust weights for the failing category
    print("\n   [STEP 1/4] ADJUSTING WEIGHTS...", flush=True)
    if WEIGHT_MANAGER_AVAILABLE:
        if category:
            old_mult, new_mult = increase_category_weight(category, score, feedback)
            print(f"   ⚖️ Weight adjusted: {category}", flush=True)
            print(f"      OLD: {old_mult:.2f}x → NEW: {new_mult:.2f}x", flush=True)
            print(f"      (Lower scores = bigger weight increase for more training)", flush=True)
        else:
            print(f"   ⚠️ No category provided, skipping weight adjustment", flush=True)
    else:
        print(f"   ⚠️ Weight manager not available", flush=True)

    # Step 2: Generate correct answer using GPT-5.1-codex-mini
    print("\n   [STEP 2/4] GENERATING CORRECT ANSWER (GPT-5.1-codex-mini)...", flush=True)
    correct_answer = generate_correct_answer(prompt, category, failed_response, feedback)
    if not correct_answer:
        print("   ❌ FAILED: Could not generate correct answer", flush=True)
        print("=" * 70, flush=True)
        return False

    print(f"   ✨ Generated correct answer!", flush=True)

    # Step 3: Add to data.py
    print("\n   [STEP 3/4] ADDING TO TRAINING DATA (data.py)...", flush=True)

    # Sanitize for display (same as what gets saved)
    prompt_sanitized = sanitize_for_training(prompt)
    answer_sanitized = sanitize_for_training(correct_answer)

    result = add_qa_to_data(prompt, correct_answer, category)

    if result["success"]:
        action = result["action"].upper()
        line_num = result["line_number"]

        if result["action"] == "updated":
            print(f"   🔄 UPDATED existing Q&A at line {line_num} in data.py", flush=True)
            print(f"   🗑️  Replaced old (worse) answer with new GPT-5.1-codex-mini answer", flush=True)
        else:
            print(f"   ✅ ADDED new Q&A at line {line_num} in data.py", flush=True)

        print(f"\n   {'─'*60}", flush=True)
        print(f"   📚 TRAINING DATA {action} (LINE {line_num}):", flush=True)
        print(f"   {'─'*60}", flush=True)
        print(f"   Q: {prompt_sanitized}", flush=True)
        print(f"   {'─'*60}", flush=True)
        print(f"   A: {answer_sanitized}", flush=True)
        print(f"   {'─'*60}", flush=True)

        if result["old_answer"]:
            print(f"\n   🗑️  OLD ANSWER REPLACED:", flush=True)
            print(f"   {result['old_answer'][:200]}...", flush=True)
    else:
        print("   ❌ FAILED: Could not add to training data", flush=True)
        print("=" * 70, flush=True)
        return False

    # Step 4: Reload knowledge base
    print("\n   [STEP 4/4] RELOADING KNOWLEDGE BASE...", flush=True)
    reload_knowledge_base()

    # Log the improvement
    log_improvement(prompt, failed_response, correct_answer, score, category)
    print(f"   📝 Improvement logged to improvements_log.json", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("✅ AUTO-IMPROVE COMPLETE - Training data enhanced!", flush=True)
    print("=" * 70 + "\n", flush=True)

    return True


def get_improvement_stats() -> Dict:
    """Get statistics about auto-improvements made."""
    if not os.path.exists(IMPROVEMENTS_LOG):
        return {"total": 0, "by_category": {}, "avg_original_score": 0.0}

    try:
        with open(IMPROVEMENTS_LOG, 'r') as f:
            log = json.load(f)

        stats: Dict = {
            "total": len(log),
            "by_category": {},
            "avg_original_score": 0.0
        }

        if log:
            for entry in log:
                cat = entry.get("category", "unknown")
                stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1
            stats["avg_original_score"] = sum(e.get("original_score", 0) for e in log) / len(log)

        return stats
    except:
        return {"total": 0, "by_category": {}, "avg_original_score": 0.0}


def show_improvement_report():
    """Display a report of all improvements made."""
    stats = get_improvement_stats()

    print("\n" + "=" * 60)
    print("📊 AUTO-IMPROVEMENT REPORT")
    print("=" * 60)
    print(f"Total improvements made: {stats['total']}")

    if stats['total'] > 0:
        print(f"Average original score: {stats['avg_original_score']:.2f}/10")
        print("\nBy category:")
        for cat, count in sorted(stats['by_category'].items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count}")

    print("=" * 60)


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Auto-improvement system for kids model")
    parser.add_argument("--report", action="store_true", help="Show improvement report")
    parser.add_argument("--test", type=str, help="Test improvement on a prompt")
    parser.add_argument("--threshold", type=float, default=AUTO_IMPROVE_THRESHOLD,
                        help=f"Score threshold (default: {AUTO_IMPROVE_THRESHOLD})")

    args = parser.parse_args()

    if args.report:
        show_improvement_report()
    elif args.test:
        print(f"Testing auto-improvement on: {args.test}")
        # Simulate a failed response
        result = auto_improve(
            prompt=args.test,
            failed_response="I don't know the answer to that.",
            score=3.0,
            category="test",
            feedback="Response was not helpful or accurate"
        )
        print(f"\nResult: {'Success' if result else 'Failed'}")
    else:
        show_improvement_report()
