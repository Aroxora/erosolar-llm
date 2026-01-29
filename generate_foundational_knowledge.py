#!/usr/bin/env python3
"""
Generate Sophisticated Propositional Knowledge for Foundational Concepts

This script generates comprehensive training data for basic concepts that
form the foundation of knowledge: numbers, operations, common nouns,
basic verbs, and fundamental relationships.

Usage:
    python generate_foundational_knowledge.py --output cache/foundations/foundations.jsonl
    python generate_foundational_knowledge.py --concepts numbers,greetings,entities
    python generate_foundational_knowledge.py --dry-run  # Preview without API calls

The generated data teaches propositional foundations like:
- What numbers represent (quantity concepts)
- What operations mean (addition, subtraction, etc.)
- Basic entity definitions (cat, dog, sun, etc.)
- Conversational patterns (greetings, identity)
"""

import argparse
import json
import os
import asyncio
import aiohttp
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

# ANSI Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"

# Default model and API base
DEFAULT_MODEL = "gpt-5.1-codex-mini"

def get_api_base() -> str:
    """Get API base URL from environment or default to OpenAI."""
    return os.environ.get("OPENAI_API_BASE", os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))

def get_chat_url() -> str:
    """Get full responses API URL (OpenAI Responses API)."""
    base = get_api_base().rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return f"{base}/responses"


def parse_response_api_output(result: dict) -> str:
    """Parse output from OpenAI Responses API format.

    The API returns: output[].type='message' -> content[].type='output_text' -> text
    """
    for out in result.get("output", []):
        if out.get("type") == "message":
            for content in out.get("content", []):
                if content.get("type") == "output_text":
                    return content.get("text", "").strip()
    return ""


@dataclass
class ConceptCategory:
    """A category of foundational concepts to generate knowledge for."""
    name: str
    description: str
    concepts: List[str]
    prompt_template: str


# Define foundational concept categories
CONCEPT_CATEGORIES = {
    "numbers": ConceptCategory(
        name="numbers",
        description="Numerical concepts and what they represent",
        concepts=[
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "twenty", "hundred", "thousand",
            "first", "second", "third", "half", "quarter", "double", "triple"
        ],
        prompt_template="""Generate a concise, educational explanation of the number/quantity concept "{concept}".

Include:
1. What this number represents as a quantity
2. A simple real-world example a child could understand
3. How it relates to counting or measurement

Keep the response under 100 words. Be precise and factual."""
    ),

    "math_operations": ConceptCategory(
        name="math_operations",
        description="Mathematical operations and their meanings",
        concepts=[
            "addition", "subtraction", "multiplication", "division",
            "plus", "minus", "times", "divided by",
            "sum", "difference", "product", "quotient",
            "equals", "greater than", "less than",
            "add", "subtract", "multiply", "divide"
        ],
        prompt_template="""Generate a clear, educational explanation of the mathematical concept "{concept}".

Include:
1. What this operation/concept means
2. A simple numeric example (like 2+2=4)
3. When you would use this in real life

Keep the response under 100 words. Be precise and educational."""
    ),

    "math_facts": ConceptCategory(
        name="math_facts",
        description="Basic arithmetic facts and their explanations",
        concepts=[
            "1+1=2", "2+2=4", "3+3=6", "5+5=10", "2+3=5", "4+4=8",
            "10-5=5", "8-3=5", "6-2=4", "9-4=5",
            "2*2=4", "2*3=6", "3*3=9", "3*4=12", "5*2=10",
            "10/2=5", "8/2=4", "6/3=2", "9/3=3"
        ],
        prompt_template="""For the math fact "{concept}":

1. State the result clearly
2. Briefly explain WHY this is true using a concrete example
3. Show it with objects (like "2 apples + 2 apples = 4 apples")

Keep the response under 80 words. Be educational and clear."""
    ),

    "greetings": ConceptCategory(
        name="greetings",
        description="Conversational greetings and social exchanges",
        concepts=[
            "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
            "goodbye", "bye", "see you", "thanks", "thank you", "please",
            "how are you", "nice to meet you", "what's up",
            "yes", "no", "maybe", "okay", "sure", "alright"
        ],
        prompt_template="""For the greeting/social phrase "{concept}":

Generate a natural, friendly response that an AI assistant would give.
The response should:
1. Be warm and conversational
2. Invite further conversation or offer help
3. Be appropriate for the context

Keep the response under 50 words. Be natural and helpful."""
    ),

    "identity": ConceptCategory(
        name="identity",
        description="Questions about identity and self-description",
        concepts=[
            "who are you", "what are you", "what is your name", "what can you do",
            "are you human", "are you a robot", "are you AI",
            "how do you work", "who made you", "what do you know",
            "can you help me", "are you smart", "do you have feelings"
        ],
        prompt_template="""For the question "{concept}":

Generate a response as an AI coding assistant named Erosolar. The response should:
1. Be honest about being an AI assistant
2. Explain capabilities related to coding and answering questions
3. Be helpful and friendly

Keep the response under 80 words. Be honest and helpful."""
    ),

    "basic_entities": ConceptCategory(
        name="basic_entities",
        description="Common nouns and their definitions",
        concepts=[
            "cat", "dog", "bird", "fish", "tree", "flower", "sun", "moon", "star",
            "water", "fire", "earth", "air", "sky", "cloud", "rain", "snow",
            "house", "car", "book", "computer", "phone", "table", "chair",
            "food", "apple", "bread", "milk", "egg", "rice",
            "person", "child", "family", "friend", "school", "work"
        ],
        prompt_template="""Define "{concept}" in a clear, educational way.

Include:
1. What it is (basic definition)
2. Key characteristics
3. A simple example or fact

Keep the response under 60 words. Be factual and educational."""
    ),

    "basic_verbs": ConceptCategory(
        name="basic_verbs",
        description="Common verbs and their meanings",
        concepts=[
            "run", "walk", "eat", "drink", "sleep", "wake", "read", "write",
            "think", "learn", "teach", "help", "work", "play", "talk", "listen",
            "see", "hear", "feel", "touch", "smell", "taste",
            "make", "build", "create", "fix", "break", "open", "close"
        ],
        prompt_template="""Explain the verb "{concept}" clearly.

Include:
1. What action it describes
2. An example sentence using this word
3. Related words or synonyms

Keep the response under 50 words. Be clear and educational."""
    ),

    "colors_shapes": ConceptCategory(
        name="colors_shapes",
        description="Colors and geometric shapes",
        concepts=[
            "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "black", "white", "gray",
            "circle", "square", "triangle", "rectangle", "oval", "star", "heart",
            "line", "point", "angle", "corner", "side", "edge"
        ],
        prompt_template="""Describe "{concept}" clearly for education.

Include:
1. What it is or looks like
2. Examples of where you see it in everyday life
3. Any interesting fact about it

Keep the response under 50 words. Be visual and clear."""
    ),

    "time_calendar": ConceptCategory(
        name="time_calendar",
        description="Time concepts and calendar terms",
        concepts=[
            "second", "minute", "hour", "day", "week", "month", "year",
            "morning", "afternoon", "evening", "night", "today", "tomorrow", "yesterday",
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
            "spring", "summer", "fall", "winter"
        ],
        prompt_template="""Explain "{concept}" as a time/calendar term.

Include:
1. What it represents
2. How long it is or when it occurs
3. Something notable about it

Keep the response under 50 words. Be clear and factual."""
    ),
}


async def generate_knowledge_pair(
    session: aiohttp.ClientSession,
    concept: str,
    category: ConceptCategory,
    model: str = "gpt-4o-mini",
    api_key: str = None
) -> Dict:
    """Generate a knowledge pair for a single concept using the API."""

    prompt = category.prompt_template.format(concept=concept)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Responses API format - detailed explanations for foundational knowledge
    system_prompt = "You are an educational assistant creating foundational knowledge for an AI training dataset. Provide thorough, detailed explanations with examples. Be precise, factual, and educational."
    data = {
        "model": model,
        "input": f"{system_prompt}\n\n{prompt}"
    }

    try:
        async with session.post(
            get_chat_url(),
            headers=headers,
            json=data,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as resp:
            if resp.status == 200:
                result = await resp.json()
                response = parse_response_api_output(result)

                # Create question variations for the concept
                questions = generate_question_variations(concept, category.name)

                return {
                    "concept": concept,
                    "category": category.name,
                    "response": response,
                    "questions": questions,
                    "success": True
                }
            else:
                error = await resp.text()
                return {
                    "concept": concept,
                    "category": category.name,
                    "error": f"API error {resp.status}: {error}",
                    "success": False
                }
    except Exception as e:
        return {
            "concept": concept,
            "category": category.name,
            "error": str(e),
            "success": False
        }


def generate_question_variations(concept: str, category: str) -> List[str]:
    """Generate different ways to ask about a concept."""

    if category == "numbers" or category == "math_operations":
        return [
            f"What is {concept}?",
            f"What does {concept} mean?",
            f"Explain {concept}",
            f"Define {concept}",
            concept
        ]
    elif category == "math_facts":
        # For math facts like "2+2=4", extract the question
        if "=" in concept:
            question = concept.split("=")[0].strip()
            answer = concept.split("=")[1].strip()
            return [
                f"What is {question}?",
                question,
                f"{question}=",
                f"Calculate {question}",
                f"What does {question} equal?"
            ]
        return [concept]
    elif category == "greetings":
        return [
            concept,
            concept.capitalize(),
            concept + "!",
            concept + "?"
        ]
    elif category == "identity":
        return [
            concept,
            concept.capitalize(),
            concept + "?",
            concept.replace("you", "u")  # Common variation
        ]
    else:
        return [
            f"What is a {concept}?",
            f"What is {concept}?",
            f"Define {concept}",
            f"Tell me about {concept}",
            f"Explain {concept}",
            concept
        ]


async def generate_category_knowledge(
    category: ConceptCategory,
    model: str,
    api_key: str,
    workers: int = 10
) -> List[Dict]:
    """Generate knowledge pairs for all concepts in a category."""

    print(f"\n{CYAN}Generating knowledge for: {category.name}{RESET}")
    print(f"{DIM}  {len(category.concepts)} concepts | {workers} workers{RESET}")

    results = []
    semaphore = asyncio.Semaphore(workers)

    async def bounded_generate(concept):
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                return await generate_knowledge_pair(session, concept, category, model, api_key)

    tasks = [bounded_generate(concept) for concept in category.concepts]
    results = await asyncio.gather(*tasks)

    successes = sum(1 for r in results if r.get("success"))
    print(f"{GREEN}  ✓ Generated {successes}/{len(results)} pairs{RESET}")

    return results


def convert_to_training_format(results: List[Dict]) -> List[Dict]:
    """Convert generated knowledge to training JSONL format."""

    training_pairs = []

    for result in results:
        if not result.get("success"):
            continue

        response = result["response"]
        questions = result.get("questions", [result["concept"]])

        # Create a training pair for each question variation
        for question in questions:
            pair = {
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": response}
                ],
                "metadata": {
                    "category": result["category"],
                    "concept": result["concept"],
                    "type": "foundational_knowledge"
                }
            }
            training_pairs.append(pair)

    return training_pairs


def generate_static_math_pairs() -> List[Dict]:
    """Generate static math fact pairs without API (as baseline)."""

    math_facts = [
        # Addition facts
        ("What is 1+1?", "1+1 equals 2. When you combine one item with another item, you get two items total."),
        ("What is 2+2?", "2+2 equals 4. If you have 2 apples and get 2 more, you now have 4 apples."),
        ("What is 3+3?", "3+3 equals 6. Three plus three makes six."),
        ("What is 5+5?", "5+5 equals 10. Five and five together make ten."),
        ("What is 2+3?", "2+3 equals 5. Two plus three gives you five."),
        ("What is 4+4?", "4+4 equals 8. Four doubled is eight."),
        ("What is 10+10?", "10+10 equals 20. Ten plus ten is twenty."),

        # Short form
        ("2+2", "4"),
        ("3+3", "6"),
        ("5+5", "10"),
        ("1+1", "2"),
        ("4+4", "8"),
        ("2+3", "5"),
        ("3+4", "7"),
        ("6+6", "12"),
        ("7+8", "15"),

        # Subtraction
        ("What is 10-5?", "10-5 equals 5. Ten minus five is five."),
        ("What is 8-3?", "8-3 equals 5. Eight minus three is five."),
        ("10-5", "5"),
        ("8-3", "5"),

        # Multiplication
        ("What is 2*3?", "2*3 equals 6. Two times three is six."),
        ("What is 3*4?", "3*4 equals 12. Three times four is twelve."),
        ("2*3", "6"),
        ("3*4", "12"),
        ("5*2", "10"),
    ]

    pairs = []
    for question, answer in math_facts:
        pairs.append({
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ],
            "metadata": {
                "category": "math_facts_static",
                "type": "foundational_knowledge"
            }
        })

    return pairs


async def generate_bridge_pairs(
    session: aiohttp.ClientSession,
    api_key: str,
    model: str = "gpt-4o-mini"
) -> List[Dict]:
    """Generate semantic bridge pairs that connect concepts to usage.

    Bridge pairs help the model understand:
    1. Why facts are true (not just that they are)
    2. How to apply concepts in context
    3. The relationship between question and answer
    """

    bridge_prompts = [
        # Math bridges - connecting concept to application
        {
            "system": "You are creating training data for an AI. Generate a clear, educational response.",
            "user": "Create a training pair where someone asks 'What is 2+2?' and the assistant explains it's 4 with a brief reason.",
            "format": "Q: What is 2+2?\nA:"
        },
        {
            "system": "You are creating training data for an AI. Generate a clear, educational response.",
            "user": "Create a training pair where someone says 'hi' and gets a friendly greeting back.",
            "format": "Q: hi\nA:"
        },
        {
            "system": "You are creating training data for an AI. Generate a clear, educational response.",
            "user": "Create a training pair where someone asks 'who are you?' and the assistant identifies as Erosolar, a helpful AI assistant.",
            "format": "Q: who are you?\nA:"
        },
        # Application bridges
        {
            "system": "You are creating training data for an AI. Generate clear, practical responses.",
            "user": "Create 3 training pairs showing simple addition questions with brief answers.",
            "format": "pairs"
        },
        {
            "system": "You are creating training data for an AI. Generate clear, conversational responses.",
            "user": "Create 3 training pairs showing different ways to greet someone and friendly responses.",
            "format": "pairs"
        },
    ]

    results = []
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    for prompt in bridge_prompts:
        # Responses API format
        data = {
            "model": model,
            "input": f"{prompt['system']}\n\n{prompt['user']}"
        }

        try:
            async with session.post(
                get_chat_url(),
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    response = parse_response_api_output(result)

                    # Parse response into training pairs
                    if prompt["format"] == "pairs":
                        # Parse multiple pairs from response
                        pairs = parse_training_pairs(response)
                        results.extend(pairs)
                    else:
                        # Single Q/A pair
                        if "A:" in response:
                            answer = response.split("A:")[-1].strip()
                            question = prompt["format"].replace("\nA:", "").replace("Q: ", "")
                            results.append({
                                "messages": [
                                    {"role": "user", "content": question},
                                    {"role": "assistant", "content": answer}
                                ],
                                "metadata": {"type": "bridge", "category": "semantic_bridge"}
                            })
        except Exception as e:
            print(f"{YELLOW}  Bridge generation error: {e}{RESET}")

    return results


def parse_training_pairs(text: str) -> List[Dict]:
    """Parse multiple training pairs from generated text."""
    pairs = []
    lines = text.split("\n")

    current_q = None
    for line in lines:
        line = line.strip()
        if line.startswith(("Q:", "Question:", "User:", "1.", "2.", "3.")):
            # Extract question
            for prefix in ["Q:", "Question:", "User:", "1.", "2.", "3."]:
                if line.startswith(prefix):
                    current_q = line[len(prefix):].strip()
                    break
        elif line.startswith(("A:", "Answer:", "Assistant:")) and current_q:
            # Extract answer
            for prefix in ["A:", "Answer:", "Assistant:"]:
                if line.startswith(prefix):
                    answer = line[len(prefix):].strip()
                    pairs.append({
                        "messages": [
                            {"role": "user", "content": current_q},
                            {"role": "assistant", "content": answer}
                        ],
                        "metadata": {"type": "bridge", "category": "parsed_pair"}
                    })
                    current_q = None
                    break

    return pairs


async def generate_curriculum_with_api(
    api_key: str,
    model: str = "gpt-4o-mini",
    workers: int = 10
) -> List[Dict]:
    """Generate full curriculum using GPT-5.1-codex-mini.

    This creates properly structured training data with:
    1. Seed facts (basic propositions)
    2. Bridge concepts (why things are true)
    3. Application examples (using knowledge)
    """

    print(f"\n{CYAN}Generating curriculum with {model}...{RESET}")

    curriculum_requests = [
        # Seed facts - simple question/answer pairs
        ("Generate 10 simple math facts as Q&A pairs. Format each as 'Q: [question]\\nA: [answer]'. Include addition like 2+2, 3+5, subtraction like 10-5, and multiplication like 2*3.", "math_seeds"),
        ("Generate 10 greeting exchanges as Q&A pairs. Format each as 'Q: [greeting]\\nA: [response]'. Include hi, hello, hey, good morning, how are you.", "greeting_seeds"),
        ("Generate 5 identity Q&A pairs where someone asks who/what the AI is. The AI is named Erosolar, a helpful assistant. Format as 'Q: [question]\\nA: [response]'.", "identity_seeds"),

        # Bridge concepts - explain why
        ("Generate 5 Q&A pairs that explain WHY simple math works. For example, explain why 2+2=4 using counting. Format as 'Q: [why question]\\nA: [explanation]'.", "math_bridges"),
        ("Generate 5 Q&A pairs about what numbers represent conceptually (like what does '5' mean). Format as 'Q: [question]\\nA: [explanation]'.", "number_bridges"),

        # Application examples - using knowledge
        ("Generate 5 Q&A pairs where someone asks to solve a simple math problem and gets a clear answer. Format as 'Q: [problem]\\nA: [solution]'.", "math_applications"),
        ("Generate 5 Q&A pairs showing basic conversational exchanges that end naturally. Format as 'Q: [user message]\\nA: [helpful response]'.", "conversation_applications"),
    ]

    all_pairs = []

    async with aiohttp.ClientSession() as session:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        for prompt, category in curriculum_requests:
            # Responses API format
            system = "You are creating high-quality training data for an AI model. Generate detailed, accurate Q&A pairs with thorough explanations. Include reasoning and examples where helpful."
            data = {
                "model": model,
                "input": f"{system}\n\n{prompt}"
            }

            try:
                async with session.post(
                    get_chat_url(),
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=90)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        response = parse_response_api_output(result)
                        pairs = parse_training_pairs(response)

                        for pair in pairs:
                            pair["metadata"]["category"] = category

                        all_pairs.extend(pairs)
                        print(f"{GREEN}  ✓ {category}: {len(pairs)} pairs{RESET}")
                    else:
                        error = await resp.text()
                        print(f"{RED}  ✗ {category}: API error {resp.status}{RESET}")
            except Exception as e:
                print(f"{RED}  ✗ {category}: {e}{RESET}")

        # Generate bridge pairs
        bridge_pairs = await generate_bridge_pairs(session, api_key, model)
        all_pairs.extend(bridge_pairs)
        print(f"{GREEN}  ✓ semantic_bridges: {len(bridge_pairs)} pairs{RESET}")

    return all_pairs


async def main():
    parser = argparse.ArgumentParser(description="Generate foundational knowledge training data")
    parser.add_argument("--output", "-o", type=str, default="cache/foundations/foundational_knowledge.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--concepts", "-c", type=str, default="all",
                        help="Comma-separated concept categories (or 'all')")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL,
                        help=f"Model to use for generation (default: {DEFAULT_MODEL})")
    parser.add_argument("--workers", "-w", type=int, default=10,
                        help="Number of concurrent API workers")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview concepts without making API calls")
    parser.add_argument("--static-only", action="store_true",
                        help="Only generate static math pairs (no API)")
    parser.add_argument("--curriculum", action="store_true",
                        help="Generate full curriculum with API (seeds, bridges, applications)")
    args = parser.parse_args()

    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  Foundational Knowledge Generator{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")

    # Get API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.dry_run and not args.static_only:
        print(f"{RED}ERROR: OPENAI_API_KEY not set{RESET}")
        print(f"{DIM}Set it with: export OPENAI_API_KEY='sk-...'{RESET}")
        print(f"{DIM}Or use --dry-run to preview concepts{RESET}")
        print(f"{DIM}Or use --static-only for basic math pairs without API{RESET}")
        return

    # Select categories
    if args.concepts == "all":
        categories = list(CONCEPT_CATEGORIES.values())
    else:
        category_names = [c.strip() for c in args.concepts.split(",")]
        categories = [CONCEPT_CATEGORIES[name] for name in category_names if name in CONCEPT_CATEGORIES]

    print(f"\n{DIM}Categories: {', '.join(c.name for c in categories)}{RESET}")
    total_concepts = sum(len(c.concepts) for c in categories)
    print(f"{DIM}Total concepts: {total_concepts}{RESET}")
    print(f"{DIM}Output: {args.output}{RESET}")

    # Dry run - just show concepts
    if args.dry_run:
        print(f"\n{YELLOW}DRY RUN - showing concepts:{RESET}")
        for cat in categories:
            print(f"\n{CYAN}{cat.name}{RESET} ({len(cat.concepts)} concepts):")
            for concept in cat.concepts[:10]:
                print(f"  - {concept}")
            if len(cat.concepts) > 10:
                print(f"  ... and {len(cat.concepts) - 10} more")
        return

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_pairs = []

    # Add static math pairs first (always included)
    static_pairs = generate_static_math_pairs()
    all_pairs.extend(static_pairs)
    print(f"\n{GREEN}✓ Added {len(static_pairs)} static math pairs{RESET}")

    # Generate curriculum with API (seeds + bridges + applications)
    if args.curriculum and api_key:
        curriculum_pairs = await generate_curriculum_with_api(api_key, args.model, args.workers)
        all_pairs.extend(curriculum_pairs)
        print(f"{GREEN}✓ Added {len(curriculum_pairs)} curriculum pairs (seeds + bridges + applications){RESET}")

    # Generate knowledge for each category (standard mode)
    elif not args.static_only and not args.curriculum:
        for category in categories:
            results = await generate_category_knowledge(
                category, args.model, api_key, args.workers
            )
            pairs = convert_to_training_format(results)
            all_pairs.extend(pairs)

    # Write output
    with open(output_path, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"\n{GREEN}{'='*60}{RESET}")
    print(f"{GREEN}  Generation Complete!{RESET}")
    print(f"{GREEN}{'='*60}{RESET}")
    print(f"{DIM}  Total pairs: {len(all_pairs)}{RESET}")
    print(f"{DIM}  Output: {args.output}{RESET}")
    print(f"\n{DIM}  Use in training:{RESET}")
    print(f"{DIM}    python train.py --basic  # Will load from {args.output}{RESET}")


if __name__ == "__main__":
    asyncio.run(main())
