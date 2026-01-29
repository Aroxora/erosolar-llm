#!/usr/bin/env python3
"""
Generate training data for kids model via score-then-fix approach.

Process:
1. Generate prompts from core topics American kids would use
2. Get model response and score it with GPT-5.1-codex-mini
3. If score is low, get GPT-5.1-codex-mini correction
4. Save corrected Q&A pairs to data.py
"""

import os
import json
import random
import argparse
import time
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Import local modules
try:
    from auto_improve import add_qa_to_data
    from registry import get_registry
except ImportError:
    add_qa_to_data = None
    get_registry = None

# =============================================================================
# CORE TOPICS - What American kids actually ask about
# =============================================================================

CORE_TOPICS = {
    # US Government & Civics
    "us_government": [
        "What are the three branches of government?",
        "What does the President do?",
        "How does a bill become a law?",
        "What is the Constitution?",
        "What is the Bill of Rights?",
        "What does Congress do?",
        "What is the Supreme Court?",
        "How do elections work?",
        "What are the amendments?",
        "What is democracy?",
        "What is the difference between the House and Senate?",
        "Who can vote in America?",
        "What does the Vice President do?",
        "How many senators are there?",
        "What is checks and balances?",
    ],

    # US History
    "us_history": [
        "What caused the Civil War?",
        "Who were the Founding Fathers?",
        "What was the American Revolution?",
        "What is the Declaration of Independence?",
        "Who was Abraham Lincoln?",
        "What was the Boston Tea Party?",
        "What happened at Pearl Harbor?",
        "What was the Civil Rights Movement?",
        "Who was Martin Luther King Jr?",
        "What was the Great Depression?",
        "Who was George Washington?",
        "What was the Cold War?",
        "Why did colonists come to America?",
        "What was the Underground Railroad?",
        "Who were the Native Americans?",
    ],

    # US Geography
    "us_geography": [
        "How many states are in the US?",
        "What is the capital of the United States?",
        "What are the biggest states?",
        "What is the longest river in America?",
        "Name some national parks",
        "What states border Canada?",
        "What is the Grand Canyon?",
        "Where is Yellowstone?",
        "What are the Great Lakes?",
        "What mountain ranges are in the US?",
    ],

    # Math
    "math": [
        "How do you add fractions?",
        "What is multiplication?",
        "How do you divide numbers?",
        "What are decimals?",
        "How do you find the area of a rectangle?",
        "What is a percentage?",
        "How do you solve word problems?",
        "What is an equation?",
        "What are prime numbers?",
        "How do you convert fractions to decimals?",
    ],

    # Science
    "science": [
        "What is photosynthesis?",
        "How does the water cycle work?",
        "What are the planets in order?",
        "What is gravity?",
        "How do volcanoes work?",
        "What is the food chain?",
        "Why is the sky blue?",
        "What are atoms?",
        "How do plants grow?",
        "What causes earthquakes?",
    ],

    # English & Reading
    "english": [
        "What is a noun?",
        "What is a verb?",
        "How do you write a paragraph?",
        "What is a synonym?",
        "What is the difference between their, there, and they're?",
        "How do you use commas?",
        "What is a metaphor?",
        "What makes a good story?",
        "What is a thesis statement?",
        "How do you write a book report?",
    ],

    # Animals & Nature
    "animals": [
        "What is the biggest animal?",
        "How do birds fly?",
        "Why did dinosaurs go extinct?",
        "What do dolphins eat?",
        "How do bees make honey?",
        "What animals live in the rainforest?",
        "Why do cats purr?",
        "How fast can a cheetah run?",
        "What is hibernation?",
        "How do fish breathe underwater?",
    ],

    # Space
    "space": [
        "How far away is the moon?",
        "What is a black hole?",
        "Why do we have seasons?",
        "How many moons does Jupiter have?",
        "What is the Milky Way?",
        "How do rockets work?",
        "Is there life on Mars?",
        "What are constellations?",
        "Why does the moon change shape?",
        "What is the sun made of?",
    ],

    # Social & Emotional
    "social": [
        "How do I make friends?",
        "What do I do if someone is being mean?",
        "How do I deal with feeling sad?",
        "What is bullying?",
        "How do I be a good friend?",
        "What do I do if I feel nervous?",
        "How do I handle peer pressure?",
        "Why is it important to be kind?",
        "How do I apologize?",
        "What do I do if I feel left out?",
    ],

    # Fun & Creative
    "creative": [
        "Tell me a joke",
        "Can you write a short story?",
        "Give me a riddle",
        "What are some fun facts?",
        "Can you make up a poem?",
        "What games can I play outside?",
        "Tell me something cool about animals",
        "What's a fun science experiment I can do at home?",
        "Can you give me a tongue twister?",
        "What are some good books for kids?",
    ],

    # Practical Life
    "practical": [
        "How do I study better?",
        "How do I organize my homework?",
        "What should I do if I'm bored?",
        "How do I be a good listener?",
        "What are healthy foods to eat?",
        "How much sleep do kids need?",
        "How do I stay safe online?",
        "What should I do in an emergency?",
        "How do I manage my time?",
        "Why is exercise important?",
    ],
}

# Flatten for easier access
ALL_PROMPTS = []
for category, prompts in CORE_TOPICS.items():
    for prompt in prompts:
        ALL_PROMPTS.append({"prompt": prompt, "category": category})


# Global model cache
_loaded_model = None
_loaded_tokenizer = None
_loaded_device = None


def load_kids_model(model_name: str = "kids-model"):
    """Load the kids model from registry."""
    global _loaded_model, _loaded_tokenizer, _loaded_device

    if _loaded_model is not None:
        return _loaded_model, _loaded_tokenizer, _loaded_device

    if get_registry is None:
        print("Error: Could not import registry")
        return None, None, None

    import torch

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    try:
        registry = get_registry()
        model, tokenizer, config, info = registry.load_model(model_name, device)
        model.eval()
        _loaded_model = model
        _loaded_tokenizer = tokenizer
        _loaded_device = device
        print(f"Loaded {model_name} on {device}")
        return model, tokenizer, device
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None


def get_model_response(prompt: str, model_name: str = "kids-model") -> str:
    """Get response from local kids-model."""
    model, tokenizer, device = load_kids_model(model_name)

    if model is None:
        return ""

    try:
        full_prompt = f"<|user|>\n{prompt}\n<|end_turn|>\n<|assistant|>\n"
        output = model.generate(
            tokenizer,
            prompt=full_prompt,
            max_tokens=200,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            device=device
        )

        # Remove the prompt from output
        if output.startswith(full_prompt):
            output = output[len(full_prompt):].strip()
        # Remove any stop tokens
        for stop_tok in ["<|end_turn|>", "<|user|>"]:
            if stop_tok in output:
                output = output.split(stop_tok)[0].strip()

        return output
    except Exception as e:
        print(f"   Generation error: {e}")
        return ""


def score_response(prompt: str, response: str, category: str) -> dict:
    """Score a response using gpt-5.1-codex-mini via Responses API."""
    if not response or len(response.strip()) < 10:
        return {"overall": 0, "feedback": "No response or too short"}

    scoring_prompt = f"""Score this AI response to a child's question (1-10 scale).

Question: {prompt}
Category: {category}
Response: {response}

Score on:
1. Accuracy (1-10): Is it factually correct?
2. Helpfulness (1-10): Does it answer the question?
3. Age-appropriate (1-10): Is it suitable for kids 5-16?
4. Clarity (1-10): Is it easy to understand?

Return ONLY JSON:
{{"overall": X.X, "accuracy": X, "helpfulness": X, "age_appropriate": X, "clarity": X, "feedback": "brief feedback"}}"""

    try:
        response_obj = client.responses.create(
            model="gpt-5.1-codex-mini",
            input=scoring_prompt,
            temperature=0.3,
            max_output_tokens=300,
            text={"format": {"type": "json_object"}},
        )
        content = response_obj.output_text
        start = content.find('{')
        end = content.rfind('}') + 1
        if start != -1 and end > start:
            return json.loads(content[start:end])
    except Exception as e:
        print(f"   Scoring error: {e}")

    return {"overall": 5, "feedback": "Scoring failed"}


def get_gpt_answer(prompt: str, category: str) -> str:
    """Get a correct answer from gpt-5.1-codex-mini for a kids question."""
    system_prompt = """You are an expert at answering children's questions.
Provide accurate, age-appropriate answers for kids ages 5-16.
Be helpful, clear, warm, and engaging.
Keep answers concise but complete (2-4 sentences for simple questions, more for complex ones)."""

    try:
        response = client.responses.create(
            model="gpt-5.1-codex-mini",
            instructions=system_prompt,
            input=prompt,
            temperature=0.7,
            max_output_tokens=500,
        )
        return response.output_text.strip()
    except Exception as e:
        print(f"   GPT answer error: {e}")
        return ""


def generate_training_data(
    num_examples: int = 100,
    score_threshold: float = 7.0,
    model_name: str = "kids-model",
    save_all: bool = False
):
    """Main training data generation loop."""

    print("=" * 70)
    print("TRAINING DATA GENERATOR")
    print("=" * 70)
    print(f"Target examples: {num_examples}")
    print(f"Score threshold: {score_threshold}")
    print(f"Categories: {len(CORE_TOPICS)}")
    print(f"Total prompts available: {len(ALL_PROMPTS)}")
    print("=" * 70)

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        return

    if add_qa_to_data is None:
        print("Error: Could not import add_qa_to_data from auto_improve.py")
        return

    # Track statistics
    stats = {
        "total_processed": 0,
        "model_passed": 0,
        "gpt_fixed": 0,
        "saved": 0,
        "failed": 0,
        "by_category": {}
    }

    # Shuffle prompts for variety
    prompts_to_use = ALL_PROMPTS.copy()
    random.shuffle(prompts_to_use)

    # If we need more than available, cycle through
    while len(prompts_to_use) < num_examples:
        extra = ALL_PROMPTS.copy()
        random.shuffle(extra)
        prompts_to_use.extend(extra)

    prompts_to_use = prompts_to_use[:num_examples]

    print(f"\nProcessing {len(prompts_to_use)} prompts...\n")

    for i, item in enumerate(prompts_to_use, 1):
        prompt = item["prompt"]
        category = item["category"]

        print(f"[{i}/{num_examples}] [{category}]")
        print(f"   Q: {prompt}")

        # Initialize category stats
        if category not in stats["by_category"]:
            stats["by_category"][category] = {"passed": 0, "fixed": 0, "failed": 0}

        # Get model response
        model_response = get_model_response(prompt, model_name)
        stats["total_processed"] += 1

        if model_response:
            print(f"   A: {model_response[:100]}...")

            # Score the response
            scores = score_response(prompt, model_response, category)
            overall = scores.get("overall", 0)
            print(f"   Score: {overall}/10 - {scores.get('feedback', '')[:50]}")

            if overall >= score_threshold:
                # Model response is good enough
                print(f"   PASS (score >= {score_threshold})")
                stats["model_passed"] += 1
                stats["by_category"][category]["passed"] += 1

                if save_all:
                    result = add_qa_to_data(prompt, model_response, category)
                    if result.get("success"):
                        stats["saved"] += 1
                        print(f"   Saved to data.py")
                continue
        else:
            print(f"   A: [No response from model]")
            overall = 0

        # Score is low or no response - get GPT fix
        print(f"   FIXING with GPT-5.1-codex-mini...")
        gpt_answer = get_gpt_answer(prompt, category)

        if gpt_answer:
            print(f"   GPT: {gpt_answer[:100]}...")

            # Save the corrected Q&A
            result = add_qa_to_data(prompt, gpt_answer, category)
            if result.get("success"):
                stats["gpt_fixed"] += 1
                stats["saved"] += 1
                stats["by_category"][category]["fixed"] += 1
                print(f"   Saved GPT answer to data.py")
            else:
                stats["failed"] += 1
                stats["by_category"][category]["failed"] += 1
                print(f"   Failed to save: {result.get('action', 'unknown')}")
        else:
            stats["failed"] += 1
            stats["by_category"][category]["failed"] += 1
            print(f"   Failed to get GPT answer")

        print()

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total processed: {stats['total_processed']}")
    print(f"Model passed (score >= {score_threshold}): {stats['model_passed']}")
    print(f"GPT fixed: {stats['gpt_fixed']}")
    print(f"Total saved to data.py: {stats['saved']}")
    print(f"Failed: {stats['failed']}")

    print(f"\nBy category:")
    for cat, cat_stats in sorted(stats["by_category"].items()):
        total = cat_stats["passed"] + cat_stats["fixed"] + cat_stats["failed"]
        print(f"  {cat}: {cat_stats['passed']} passed, {cat_stats['fixed']} fixed, {cat_stats['failed']} failed (total: {total})")

    print("=" * 70)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description='Generate training data via score-then-fix approach'
    )
    parser.add_argument('-n', '--num-examples', type=int, default=100,
                        help='Number of examples to process (default: 100)')
    parser.add_argument('-t', '--threshold', type=float, default=7.0,
                        help='Score threshold for passing (default: 7.0)')
    parser.add_argument('-m', '--model', type=str, default='kids-model',
                        help='Model to test (default: kids-model)')
    parser.add_argument('--save-all', action='store_true',
                        help='Save model responses that pass threshold too')
    parser.add_argument('--random-seed', action='store_true',
                        help='Use random seed for variety')

    args = parser.parse_args()

    if args.random_seed:
        seed = int(time.time()) % 1000000
        random.seed(seed)
        print(f"Using random seed: {seed}")
    else:
        random.seed(42)

    generate_training_data(
        num_examples=args.num_examples,
        score_threshold=args.threshold,
        model_name=args.model,
        save_all=args.save_all
    )


if __name__ == "__main__":
    main()
