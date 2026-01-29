#!/usr/bin/env python3
"""
Generate Bridge Data for Curriculum Learning (Phase 1.5)

This script generates intermediate complexity training data that BRIDGES
foundational knowledge (Phase 1) to sophisticated reasoning (Phase 2).

KEY PRINCIPLE: Bridge data must explicitly:
1. Reference concepts from foundational training data
2. Build on those concepts with intermediate complexity
3. Prepare reasoning patterns needed for sophisticated tasks

The bridge phase ensures self-attention learns to:
1. Connect previously learned facts to new problems
2. Handle multi-step reasoning chains
3. Track context across longer sequences
4. Apply foundational knowledge in new contexts

Usage:
    python generate_bridge_data.py --output cache/bridge/bridge_data.jsonl
    python generate_bridge_data.py --workers 20 --model gpt-5.1-codex-mini
"""

import argparse
import json
import os
import asyncio
import aiohttp
from pathlib import Path
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass

from bridge_synthetic import generate_synthetic_bridge_pairs
# ANSI Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"


def load_foundational_concepts() -> Dict[str, List[Tuple[str, str]]]:
    """
    Load concepts that were taught in foundational (Phase 1) training.
    Returns dict mapping concept category -> list of (prompt, response) pairs.

    Bridge data MUST build on these established concepts.
    """
    foundational = {
        "basic_math": [],
        "greetings": [],
        "basic_facts": [],
        "simple_code": [],
        "definitions": [],
    }

    # Load from data.py (GREETINGS, KIDS_QA, etc.)
    try:
        from data import GREETINGS, KIDS_QA, MATH_PROBLEMS

        foundational["greetings"] = list(GREETINGS[:50])
        foundational["basic_facts"] = list(KIDS_QA[:50])

        # Extract math foundations
        if MATH_PROBLEMS:
            foundational["basic_math"] = [
                (q, a) for q, a in MATH_PROBLEMS
                if any(op in q.lower() for op in ['what is', '+', '-', '*', '/'])
            ][:30]

    except ImportError:
        pass

    # Load from foundational knowledge JSONL if exists
    foundations_path = Path("cache/foundations/foundational_knowledge.jsonl")
    if foundations_path.exists():
        try:
            with open(foundations_path) as f:
                for line in f:
                    record = json.loads(line)
                    msgs = record.get("messages", [])
                    if len(msgs) >= 2:
                        q = msgs[0].get("content", "")
                        a = msgs[1].get("content", "")
                        cat = record.get("metadata", {}).get("category", "definitions")
                        if cat in foundational:
                            foundational[cat].append((q, a))
                        else:
                            foundational["definitions"].append((q, a))
        except Exception as e:
            print(f"{YELLOW}Warning loading foundations: {e}{RESET}")

    return foundational


def extract_foundational_facts() -> List[Dict]:
    """
    Extract key facts from foundational data that bridge prompts should reference.
    These are the 'anchor points' that bridge data builds upon.
    """
    facts = []

    # Core math facts from foundational training
    math_facts = [
        {"fact": "2 + 2 = 4", "category": "addition", "complexity": 1},
        {"fact": "5 + 5 = 10", "category": "addition", "complexity": 1},
        {"fact": "3 × 3 = 9", "category": "multiplication", "complexity": 1},
        {"fact": "10 ÷ 2 = 5", "category": "division", "complexity": 1},
        {"fact": "10 - 3 = 7", "category": "subtraction", "complexity": 1},
    ]
    facts.extend(math_facts)

    # Core coding facts
    code_facts = [
        {"fact": "print() displays output", "category": "python_basics", "complexity": 1},
        {"fact": "variables store values", "category": "python_basics", "complexity": 1},
        {"fact": "def creates a function", "category": "python_basics", "complexity": 1},
        {"fact": "lists use [] brackets", "category": "python_basics", "complexity": 1},
        {"fact": "for loops iterate over items", "category": "python_basics", "complexity": 1},
    ]
    facts.extend(code_facts)

    # Core world knowledge
    world_facts = [
        {"fact": "water freezes at 0°C", "category": "science", "complexity": 1},
        {"fact": "plants need sunlight", "category": "science", "complexity": 1},
        {"fact": "there are 8 planets", "category": "science", "complexity": 1},
        {"fact": "cats and dogs are mammals", "category": "animals", "complexity": 1},
    ]
    facts.extend(world_facts)

    return facts


@dataclass
class BridgeConcept:
    """A bridge concept that connects foundational to sophisticated knowledge."""
    name: str
    description: str
    prompts: List[str]
    system_prompt: str


# Bridge concepts - intermediate complexity between basic and sophisticated
BRIDGE_CATEGORIES = {
    "multi_step_math": BridgeConcept(
        name="multi_step_math",
        description="Multi-step arithmetic building on foundational facts (2+2=4, 5+5=10)",
        prompts=[
            "We know 2+2=4. Now calculate 2+2+5",
            "Building on 5+5=10, what is 5+5+5+5?",
            "If I have 5 apples and buy 3 more, then eat 2, how many do I have?",
            "Using 3×3=9, calculate 3×3+3×3",
            "Foundation: 10-3=7. Now what is 10-3-3?",
            "We know 20÷4=5. What is 20÷4+20÷4?",
            "Building on addition: 2+2+3+3+4+4=?",
            "If x=5 (variable stores 5), what is x+x+x?",
            "Chain: Start with 10, subtract 3, add 5. What's the result?",
            "Using 100÷10=10, calculate 100÷10+100÷10",
            "Foundation: half of 10 is 5. What is half of 10 plus half of 10?",
            "Build from 6+6=12: what is 6+6+6?",
            "We know 4×5=20. What is 4×5-4?",
            "Chain calculation: 2×2=4, 4×2=8, 8×2=?",
            "Using 15÷3=5, what is 15÷3+15÷3+15÷3?",
        ],
        system_prompt="""You are bridging foundational math to multi-step reasoning.

IMPORTANT: Each response must:
1. Reference the foundational fact being used
2. Show step-by-step reasoning that builds on it
3. State the final answer clearly

Format:
Foundation: [state the basic fact]
Step 1: [apply foundation]
Step 2: [next step]
Answer: [final result]

Keep responses under 80 words. Show the BRIDGE from simple to complex."""
    ),

    "word_problems": BridgeConcept(
        name="word_problems",
        description="Simple word problems requiring comprehension and math",
        prompts=[
            "A book costs $5. How much do 3 books cost?",
            "Sara has 12 stickers. She gives 4 to Tom. How many does Sara have now?",
            "There are 8 birds on a tree. 3 fly away. How many are left?",
            "Mom bought 2 boxes of cookies. Each box has 6 cookies. How many cookies total?",
            "A train has 4 cars. Each car has 10 seats. How many seats total?",
            "I walked 2 miles on Monday and 3 miles on Tuesday. How far did I walk?",
            "There are 15 students. They form 3 equal groups. How many in each group?",
            "A pizza has 8 slices. We ate half. How many slices are left?",
            "John is 8 years old. His sister is 3 years older. How old is his sister?",
            "A bag has 20 marbles. 5 are red, 5 are blue. How many are other colors?",
        ],
        system_prompt="""You are helping a student with word problems. For each:
1. Identify what we know
2. Show the calculation
3. State the answer with units
Keep responses under 60 words. Build logical reasoning skills."""
    ),

    "simple_patterns": BridgeConcept(
        name="simple_patterns",
        description="Pattern recognition and sequence completion",
        prompts=[
            "What comes next: 2, 4, 6, 8, ?",
            "Complete the pattern: A, B, A, B, A, ?",
            "What's missing: 1, 2, _, 4, 5",
            "Continue: red, blue, red, blue, ?",
            "Next number: 5, 10, 15, 20, ?",
            "Pattern: 1, 1, 2, 2, 3, 3, ?",
            "What comes after: Mon, Tue, Wed, ?",
            "Complete: 10, 20, 30, ?, 50",
            "Sequence: 2, 4, 8, 16, ?",
            "Fill in: cat, dog, cat, dog, ?",
            "Next: 100, 90, 80, 70, ?",
            "Pattern: ABC, DEF, GHI, ?",
        ],
        system_prompt="""You recognize patterns. For each:
1. Identify the pattern rule
2. Apply it to find the next element
3. State the answer clearly
Keep responses under 50 words. Focus on pattern reasoning."""
    ),

    "simple_coding": BridgeConcept(
        name="simple_coding",
        description="Basic coding concepts and simple functions",
        prompts=[
            "Write a function to add two numbers in Python",
            "How do I print 'Hello World' in Python?",
            "What does a for loop do?",
            "Write code to check if a number is even",
            "How do I create a list in Python?",
            "What is a variable?",
            "Write a function that returns the bigger of two numbers",
            "How do I comment code in Python?",
            "What does 'return' do in a function?",
            "Write code to print numbers 1 to 5",
            "How do I check if two strings are equal?",
            "What is an if statement?",
            "Write a function to double a number",
            "How do I get the length of a string?",
            "What is indentation in Python?",
        ],
        system_prompt="""You are a beginner-friendly coding tutor. For each:
1. Explain the concept simply
2. Provide a small working code example
3. Keep code under 5 lines when possible
Build from basic programming understanding. Be concise."""
    ),

    "cause_effect": BridgeConcept(
        name="cause_effect",
        description="Simple cause and effect reasoning",
        prompts=[
            "What happens when water freezes?",
            "Why do plants need sunlight?",
            "What happens if you don't sleep enough?",
            "Why does ice melt in the sun?",
            "What causes rain?",
            "Why do we wear coats in winter?",
            "What happens when you mix red and blue paint?",
            "Why do birds fly south in winter?",
            "What causes a shadow?",
            "Why do we brush our teeth?",
            "What happens when a battery dies?",
            "Why does metal feel cold?",
        ],
        system_prompt="""Explain cause and effect relationships simply.
1. State the cause clearly
2. Explain the effect
3. Keep the connection logical and simple
Responses under 60 words. Build reasoning from basic knowledge."""
    ),

    "comparisons": BridgeConcept(
        name="comparisons",
        description="Comparing and contrasting concepts",
        prompts=[
            "What's the difference between a cat and a dog?",
            "How are addition and multiplication related?",
            "Compare day and night",
            "What's similar about circles and squares?",
            "Difference between hot and cold",
            "How is reading different from writing?",
            "Compare a bicycle and a car",
            "What do summer and winter have in common?",
            "Difference between a question and an answer",
            "How are letters and numbers different?",
            "Compare walking and running",
            "What's the difference between a list and a dictionary in Python?",
        ],
        system_prompt="""Compare and contrast concepts clearly.
1. State key similarities
2. State key differences
3. Keep comparisons balanced and factual
Under 70 words. Build analytical thinking from foundational concepts."""
    ),

    "simple_instructions": BridgeConcept(
        name="simple_instructions",
        description="Following and giving simple multi-step instructions",
        prompts=[
            "How do I make a peanut butter sandwich?",
            "What are the steps to tie a shoe?",
            "How do I send an email?",
            "Steps to wash your hands properly",
            "How do I save a file on a computer?",
            "What are the steps to plant a seed?",
            "How do I draw a simple house?",
            "Steps to make a cup of tea",
            "How do I count from 1 to 10 in steps of 2?",
            "What are the steps to cross the street safely?",
        ],
        system_prompt="""Provide clear step-by-step instructions.
1. Number each step
2. Keep steps simple and sequential
3. Use action words (first, then, next, finally)
Under 100 words. Build procedural thinking."""
    ),

    "context_bridging": BridgeConcept(
        name="context_bridging",
        description="Questions that require connecting context across the prompt",
        prompts=[
            "The cat is on the mat. The mat is red. What color is the surface under the cat?",
            "John has a ball. The ball is blue. Mary wants John's toy. What color is what Mary wants?",
            "2+2=4 and 4+4=8. What is 2+2+4+4?",
            "Roses are red. Violets are blue. What color are roses?",
            "My name is Alex. I am an AI. What is my name?",
            "Today is Monday. Tomorrow is the day after today. What day is tomorrow?",
            "The box contains 5 apples. We add 3 more. The box now has how many apples?",
            "Python uses indentation. Indentation means spaces at the start of a line. What does Python use for code structure?",
        ],
        system_prompt="""Answer questions that require connecting information.
1. Identify all relevant facts in the prompt
2. Connect them logically
3. State the answer clearly
Focus on demonstrating context tracking. Under 50 words."""
    ),

    "building_concepts": BridgeConcept(
        name="building_concepts",
        description="Concepts that build on foundational knowledge",
        prompts=[
            "We know 2+2=4. What is 4+4?",
            "A cat is an animal. A dog is an animal. What are cats and dogs?",
            "One hour has 60 minutes. How many minutes in 2 hours?",
            "If 5+5=10 and 10+10=20, what is 5+5+5+5?",
            "Water freezes to ice. Ice melts to water. What happens to ice when heated?",
            "Variables store values. x=5 stores 5 in x. What does x+x equal?",
            "Squares have 4 sides. Rectangles also have 4 sides. How many sides do both shapes have?",
            "print() displays output. What does print('hi') display?",
        ],
        system_prompt="""Build on established foundational knowledge.
1. Reference the foundational fact
2. Apply it to answer the question
3. Show the logical connection
Under 60 words. Demonstrate knowledge building."""
    ),
}


async def generate_bridge_pair(
    session: aiohttp.ClientSession,
    prompt: str,
    category: BridgeConcept,
    model: str,
    api_key: str
) -> Dict:
    """Generate a single bridge training pair."""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": category.system_prompt},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 250,
        "temperature": 0.7
    }

    try:
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as resp:
            if resp.status == 200:
                result = await resp.json()
                response = result["choices"][0]["message"]["content"].strip()
                return {
                    "prompt": prompt,
                    "response": response,
                    "category": category.name,
                    "success": True
                }
            else:
                error = await resp.text()
                return {
                    "prompt": prompt,
                    "category": category.name,
                    "error": f"API error {resp.status}: {error}",
                    "success": False
                }
    except Exception as e:
        return {
            "prompt": prompt,
            "category": category.name,
            "error": str(e),
            "success": False
        }


async def generate_category_data(
    category: BridgeConcept,
    model: str,
    api_key: str,
    workers: int = 10
) -> List[Dict]:
    """Generate bridge data for all prompts in a category."""

    print(f"\n{CYAN}Generating bridge data: {category.name}{RESET}")
    print(f"{DIM}  {len(category.prompts)} prompts | {workers} workers{RESET}")

    semaphore = asyncio.Semaphore(workers)

    async def bounded_generate(prompt):
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                return await generate_bridge_pair(session, prompt, category, model, api_key)

    tasks = [bounded_generate(prompt) for prompt in category.prompts]
    results = await asyncio.gather(*tasks)

    successes = sum(1 for r in results if r.get("success"))
    print(f"{GREEN}  Generated {successes}/{len(results)} pairs{RESET}")

    return results


def generate_static_bridge_pairs() -> List[Dict]:
    """
    Generate static bridge pairs that EXPLICITLY connect foundational → sophisticated.

    Each bridge pair follows the pattern:
    1. Reference a foundational fact (Phase 1 knowledge)
    2. Apply it in a new context (Bridge complexity)
    3. Show reasoning chain (Prepare for Phase 2)
    """

    pairs = []

    # =====================================================
    # CATEGORY 1: Math Bridges (Foundational → Multi-step)
    # Foundation: Basic arithmetic (2+2=4, 5+5=10)
    # Bridge: Apply those facts in chains
    # =====================================================
    math_bridges = [
        # Bridge from "2+2=4" foundation
        ("We know 2+2=4. What is 2+2+2?",
         "Building on 2+2=4:\nStep 1: 2+2 = 4 (foundational fact)\nStep 2: 4+2 = 6\n\nThe answer is 6."),

        ("Since 2+2=4 and 3+3=6, what is 2+2+3+3?",
         "Using foundational facts:\n- 2+2 = 4\n- 3+3 = 6\nStep: 4 + 6 = 10\n\nThe answer is 10."),

        # Bridge from "5+5=10" foundation
        ("We know 5+5=10. What is 5+5+5?",
         "Building on 5+5=10:\nStep 1: 5+5 = 10 (foundational)\nStep 2: 10+5 = 15\n\nThe answer is 15."),

        ("If 5+5=10, what is 10+10?",
         "From foundation: 5+5=10\nApplying the pattern: 10+10=20\n\nThe answer is 20."),

        # Bridge from basic operations
        ("Using that 10-3=7, calculate 10-3-2",
         "Starting with foundation: 10-3=7\nThen: 7-2=5\n\nThe answer is 5."),

        ("Knowing 3×3=9, what is 3×3+1?",
         "Foundation: 3×3=9\nBridge step: 9+1=10\n\nThe answer is 10."),

        # Multi-step chains
        ("Calculate step by step: 2+2+3+3",
         "Using foundations:\nStep 1: 2+2=4 (known fact)\nStep 2: 3+3=6 (known fact)\nStep 3: 4+6=10\n\nThe answer is 10."),

        ("What is (2+2) × (5-3)?",
         "Foundation facts:\n- 2+2=4\n- 5-3=2\nBridge: 4×2=8\n\nThe answer is 8."),
    ]
    pairs.extend(math_bridges)

    # =====================================================
    # CATEGORY 2: Coding Bridges (print/variables → functions)
    # Foundation: print(), variables, basic syntax
    # Bridge: Combine them into simple programs
    # =====================================================
    coding_bridges = [
        # Bridge from "print() displays output"
        ("We know print() displays output. How do we print a variable x?",
         "Foundation: print() displays output\nBridge: To print a variable, pass it to print()\n\n```python\nx = 5\nprint(x)  # Displays: 5\n```"),

        # Bridge from "variables store values"
        ("Variables store values. How do we use two variables together?",
         "Foundation: Variables store values\nBridge: Combine variables with operators\n\n```python\na = 3\nb = 4\nresult = a + b  # result stores 7\nprint(result)\n```"),

        # Bridge from "def creates a function"
        ("We know 'def' creates a function. Create a function that uses print()",
         "Foundations:\n- def creates a function\n- print() displays output\n\nBridge - combining them:\n```python\ndef say_hello():\n    print(\"Hello!\")\n\nsay_hello()  # Displays: Hello!\n```"),

        # Bridge from "lists use [] brackets"
        ("Lists use []. How do we print each item in a list?",
         "Foundations:\n- Lists use []\n- print() displays output\n- for loops iterate\n\nBridge:\n```python\nfruits = [\"apple\", \"banana\"]\nfor fruit in fruits:\n    print(fruit)\n```"),

        # Bridge from "for loops iterate"
        ("For loops iterate. How do we use one with range()?",
         "Foundation: for loops iterate over items\nBridge: range() creates a sequence of numbers\n\n```python\nfor i in range(3):\n    print(i)  # Prints 0, 1, 2\n```"),

        # Combining multiple foundations
        ("Using variables, a function, and print, create an add function",
         "Combining foundations:\n- Variables store values\n- def creates functions\n- print() displays output\n\n```python\ndef add(a, b):\n    result = a + b\n    return result\n\nprint(add(2, 3))  # Displays: 5\n```"),
    ]
    pairs.extend(coding_bridges)

    # =====================================================
    # CATEGORY 3: Knowledge Bridges (facts → reasoning)
    # Foundation: Simple facts from KIDS_QA
    # Bridge: Apply facts to answer new questions
    # =====================================================
    knowledge_bridges = [
        # Bridge from "water freezes at 0°C"
        ("We know water freezes at 0°C. What happens at -5°C?",
         "Foundation: Water freezes at 0°C\nReasoning: -5°C is below freezing point\nConclusion: Water will be frozen (ice) at -5°C."),

        # Bridge from "plants need sunlight"
        ("Plants need sunlight. What happens to a plant in a dark room?",
         "Foundation: Plants need sunlight to make food (photosynthesis)\nReasoning: Dark room = no sunlight\nConclusion: The plant will weaken and eventually die without light."),

        # Bridge from "there are 8 planets"
        ("There are 8 planets. If we visited 3, how many are left to visit?",
         "Foundation: 8 planets total\nCalculation: 8 - 3 = 5\nAnswer: 5 planets left to visit."),

        # Bridge from "cats and dogs are mammals"
        ("Cats and dogs are mammals. Mammals have fur. Do cats have fur?",
         "Foundation: Cats are mammals\nFact: Mammals have fur\nLogical conclusion: Yes, cats have fur."),

        # Multi-fact reasoning
        ("Water freezes at 0°C and boils at 100°C. What state is water at 50°C?",
         "Foundations:\n- Freezes at 0°C (solid below)\n- Boils at 100°C (gas above)\n\nReasoning: 50°C is between 0 and 100\nAnswer: Water is liquid at 50°C."),
    ]
    pairs.extend(knowledge_bridges)

    # =====================================================
    # CATEGORY 4: Context Bridges (tracking information)
    # Teaches model to track context across sentences
    # =====================================================
    context_bridges = [
        # Simple context tracking
        ("John has 5 apples. Mary gives him 3 more. John now has how many?",
         "Context tracking:\n- Start: John has 5 apples\n- Event: +3 apples from Mary\n- Result: 5 + 3 = 8 apples"),

        ("x = 10. We subtract 3. Then add 5. What is x now?",
         "Tracking x:\n- Start: x = 10\n- Step 1: x = 10 - 3 = 7\n- Step 2: x = 7 + 5 = 12\n\nx is now 12."),

        ("The list is [1, 2]. We append 3. We append 4. What's in the list?",
         "Tracking the list:\n- Start: [1, 2]\n- append(3): [1, 2, 3]\n- append(4): [1, 2, 3, 4]\n\nThe list contains [1, 2, 3, 4]."),

        # Reference resolution
        ("Alice is a programmer. She writes Python. What language does Alice use?",
         "Context: Alice is a programmer who writes Python.\n'She' refers to Alice.\nAnswer: Alice uses Python."),

        ("The function returns 5. We store it in result. What is result?",
         "Context tracking:\n- Function returns 5\n- Stored in 'result'\nAnswer: result is 5."),
    ]
    pairs.extend(context_bridges)

    # =====================================================
    # CATEGORY 5: Pattern Bridges (recognition → prediction)
    # Foundation: Basic sequences
    # Bridge: Explain the pattern, then predict
    # =====================================================
    pattern_bridges = [
        ("In 2, 4, 6, 8 each number is 2 more. What comes next?",
         "Pattern identified: +2 each step\nSequence: 2, 4, 6, 8\nNext: 8 + 2 = 10\n\nThe answer is 10."),

        ("1, 2, 4, 8 doubles each time. What's the pattern and next number?",
         "Pattern: Each number is doubled (×2)\nSequence: 1, 2, 4, 8\nNext: 8 × 2 = 16\n\nThe answer is 16."),

        ("Mon, Tue, Wed follows the week. What comes after Wed?",
         "Pattern: Days of the week in order\nAfter Wednesday comes Thursday.\n\nThe answer is Thursday."),

        ("In ABC, DEF, GHI each group advances 3 letters. What's next?",
         "Pattern: Each letter group advances by 3 positions\nA→D→G→J, B→E→H→K, C→F→I→L\nNext group: JKL"),
    ]
    pairs.extend(pattern_bridges)

    # =====================================================
    # CATEGORY 6: Reasoning Bridges (if-then logic)
    # Prepares for complex reasoning in Phase 2
    # =====================================================
    reasoning_bridges = [
        ("If x > 5 and x = 7, is the condition true?",
         "Given: x = 7\nCondition: x > 5\nCheck: Is 7 > 5? Yes.\n\nThe condition is true."),

        ("All birds have wings. A sparrow is a bird. Does a sparrow have wings?",
         "Premise 1: All birds have wings\nPremise 2: A sparrow is a bird\nConclusion: Therefore, a sparrow has wings.\n\nYes, a sparrow has wings."),

        ("If it rains, the ground gets wet. It is raining. What can we conclude?",
         "Rule: Rain → wet ground\nFact: It is raining\nConclusion: The ground is wet (or will get wet)."),

        ("A function returns True if n > 0. What does it return for n = 5?",
         "Rule: Return True if n > 0\nInput: n = 5\nCheck: 5 > 0 is True\nReturn: True"),
    ]
    pairs.extend(reasoning_bridges)

    # Format all pairs
    formatted = []
    categories = [
        ("math_bridge", math_bridges),
        ("coding_bridge", coding_bridges),
        ("knowledge_bridge", knowledge_bridges),
        ("context_bridge", context_bridges),
        ("pattern_bridge", pattern_bridges),
        ("reasoning_bridge", reasoning_bridges),
    ]

    for cat_name, cat_pairs in categories:
        for p, r in cat_pairs:
            formatted.append({
                "messages": [
                    {"role": "user", "content": p},
                    {"role": "assistant", "content": r}
                ],
                "metadata": {
                    "category": cat_name,
                    "type": "bridge_knowledge",
                    "bridges_from": "foundational",
                    "bridges_to": "sophisticated"
                }
            })

    return formatted


def convert_to_training_format(results: List[Dict]) -> List[Dict]:
    """Convert generated results to training JSONL format."""

    pairs = []
    for result in results:
        if not result.get("success"):
            continue

        pairs.append({
            "messages": [
                {"role": "user", "content": result["prompt"]},
                {"role": "assistant", "content": result["response"]}
            ],
            "metadata": {
                "category": result["category"],
                "type": "bridge_knowledge"
            }
        })

    return pairs


async def main():
    parser = argparse.ArgumentParser(description="Generate bridge training data for curriculum learning")
    parser.add_argument("--output", "-o", type=str, default="cache/bridge/bridge_data.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--categories", "-c", type=str, default="all",
                        help="Comma-separated categories (or 'all')")
    parser.add_argument("--model", "-m", type=str, default="gpt-4o-mini",
                        help="Model to use for generation")
    parser.add_argument("--workers", "-w", type=int, default=20,
                        help="Number of concurrent API workers")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview prompts without API calls")
    parser.add_argument("--static-only", action="store_true",
                        help="Only use static pairs (no API)")
    parser.add_argument("--target-count", type=int, default=400,
                        help="Minimum total bridge pairs to output (default: 400, 0 disables synthetic)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for synthetic bridge generation (default: 42)")
    args = parser.parse_args()

    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  Bridge Data Generator (Curriculum Phase 1.5){RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{DIM}  Purpose: Bridge foundational -> sophisticated knowledge{RESET}")
    print(f"{DIM}  Focus: Multi-step reasoning, context tracking, patterns{RESET}")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.dry_run and not args.static_only:
        print(f"\n{RED}ERROR: OPENAI_API_KEY not set{RESET}")
        print(f"{DIM}Set with: export OPENAI_API_KEY='sk-...'{RESET}")
        print(f"{DIM}Or use --dry-run or --static-only{RESET}")
        return

    # Select categories
    if args.categories == "all":
        categories = list(BRIDGE_CATEGORIES.values())
    else:
        names = [c.strip() for c in args.categories.split(",")]
        categories = [BRIDGE_CATEGORIES[n] for n in names if n in BRIDGE_CATEGORIES]

    total_prompts = sum(len(c.prompts) for c in categories)
    print(f"\n{DIM}Categories: {', '.join(c.name for c in categories)}{RESET}")
    print(f"{DIM}Total prompts: {total_prompts}{RESET}")
    print(f"{DIM}Model: {args.model}{RESET}")
    print(f"{DIM}Output: {args.output}{RESET}")
    if args.target_count:
        print(f"{DIM}Target pairs: {args.target_count} (seed: {args.seed}){RESET}")

    if args.dry_run:
        print(f"\n{YELLOW}DRY RUN - showing prompts:{RESET}")
        for cat in categories:
            print(f"\n{CYAN}{cat.name}{RESET}:")
            for prompt in cat.prompts[:5]:
                print(f"  - {prompt}")
            if len(cat.prompts) > 5:
                print(f"  ... and {len(cat.prompts) - 5} more")
        return

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_pairs = []

    # Static pairs (always included)
    static_pairs = generate_static_bridge_pairs()
    all_pairs.extend(static_pairs)
    print(f"\n{GREEN}Added {len(static_pairs)} static bridge pairs{RESET}")

    # API-generated pairs
    if not args.static_only:
        for category in categories:
            results = await generate_category_data(category, args.model, api_key, args.workers)
            pairs = convert_to_training_format(results)
            all_pairs.extend(pairs)

    if args.target_count and len(all_pairs) < args.target_count:
        needed = args.target_count - len(all_pairs)
        synthetic_pairs = generate_synthetic_bridge_pairs(needed, seed=args.seed)
        all_pairs.extend(synthetic_pairs)
        print(f"{GREEN}Added {len(synthetic_pairs)} synthetic bridge pairs (target {args.target_count}){RESET}")
    elif args.target_count:
        print(f"{DIM}Bridge target met: {len(all_pairs)}/{args.target_count}{RESET}")

    # Write output atomically (temp file + rename)
    import shutil
    tmp_path = output_path.with_suffix(".tmp")

    # Backup existing file if present
    if output_path.exists():
        backup_path = output_path.with_suffix(".jsonl.bak")
        shutil.copy2(output_path, backup_path)
        print(f"{DIM}  Backed up existing data to {backup_path}{RESET}")

    # Write to temp file first
    with open(tmp_path, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    # Atomic rename
    tmp_path.replace(output_path)

    print(f"\n{GREEN}{'='*60}{RESET}")
    print(f"{GREEN}  Bridge Data Generation Complete!{RESET}")
    print(f"{GREEN}{'='*60}{RESET}")
    print(f"{DIM}  Total pairs: {len(all_pairs)}{RESET}")
    print(f"{DIM}  Output: {args.output}{RESET}")
    print(f"{DIM}  Saved persistently with backup{RESET}")
    print(f"\n{DIM}  Use in training:{RESET}")
    print(f"{DIM}    ./upgrade_and_serve.sh --curriculum  # 3-phase curriculum{RESET}")


if __name__ == "__main__":
    asyncio.run(main())
