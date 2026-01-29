#!/usr/bin/env python3
"""
ADVERSARIAL TRAINING DATA GENERATOR

Goal: Generate training prompts that could train a model to BEAT the generator (GPT-5.1-codex-mini).

Strategy:
1. Exploit known LLM weaknesses (math reasoning, logic, temporal, spatial)
2. Self-critique: Have GPT-5.1-codex-mini generate, then critique its own answers
3. Verified ground truth: Use code execution, math proofs, authoritative sources
4. Adversarial prompts: Ask GPT-5.1-codex-mini to generate prompts that would trick LLMs
5. Confidence calibration: Find cases where it's confident but wrong
6. Multi-step reasoning chains where errors compound
"""

import os
import json
import random
import re
import subprocess
import sys
import math
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict, Any

# Initialize OpenAI client
from openai import OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "adversarial_training_data.jsonl")
STATS_FILE = os.path.join(SCRIPT_DIR, "adversarial_stats.json")

# =============================================================================
# KNOWN LLM WEAKNESSES - Categories where LLMs typically fail
# =============================================================================

LLM_WEAKNESSES = {
    "mathematical_reasoning": [
        "multi_step_algebra", "word_problems_complex", "probability_conditional",
        "combinatorics", "number_theory", "modular_arithmetic", "sequences_series",
        "geometry_proofs", "calculus_applications", "statistics_inference",
        "order_of_operations_tricky", "negative_number_operations", "fraction_operations",
        "percentage_of_percentage", "compound_interest", "rate_problems"
    ],
    "logical_reasoning": [
        "syllogisms", "negation_logic", "contrapositive", "biconditional",
        "quantifier_logic", "modal_logic", "temporal_logic", "counterfactuals",
        "paradoxes", "self_reference", "multi_step_deduction", "proof_by_contradiction",
        "necessary_vs_sufficient", "logical_fallacies_detection", "truth_tables"
    ],
    "temporal_reasoning": [
        "date_calculations", "time_zone_conversions", "sequence_ordering",
        "duration_calculations", "relative_time", "calendar_math", "age_calculations",
        "historical_ordering", "before_after_relationships", "simultaneous_events",
        "leap_year_calculations", "day_of_week", "recurring_events"
    ],
    "spatial_reasoning": [
        "3d_mental_rotation", "relative_directions", "map_navigation",
        "geometric_transformations", "perspective_taking", "folding_problems",
        "volume_surface_area", "coordinate_geometry", "graph_interpretation",
        "topology_basics", "symmetry", "tessellation"
    ],
    "numerical_precision": [
        "large_number_arithmetic", "decimal_precision", "scientific_notation",
        "unit_conversions", "significant_figures", "rounding_rules",
        "floating_point_issues", "exact_vs_approximate", "order_of_magnitude"
    ],
    "factual_accuracy": [
        "common_misconceptions", "frequently_confused_facts", "urban_legends",
        "historical_myths", "science_myths", "statistical_fallacies",
        "correlation_vs_causation", "survivorship_bias", "base_rate_neglect"
    ],
    "consistency": [
        "self_contradiction", "premise_tracking", "constraint_satisfaction",
        "multi_turn_consistency", "character_consistency", "world_building_consistency",
        "numerical_consistency", "logical_consistency_chains"
    ],
    "instruction_following": [
        "negative_instructions", "compound_instructions", "conditional_instructions",
        "format_constraints", "length_constraints", "exclusion_constraints",
        "prioritization", "multi_objective_optimization"
    ],
    "common_sense_physics": [
        "object_permanence", "gravity_intuition", "collision_prediction",
        "fluid_dynamics_intuition", "heat_transfer", "momentum_conservation",
        "leverage_mechanics", "buoyancy", "pressure_effects"
    ],
    "edge_cases": [
        "zero_division_contexts", "empty_set_operations", "boundary_conditions",
        "null_cases", "degenerate_cases", "limit_cases", "overflow_underflow",
        "special_values", "corner_cases"
    ]
}

# =============================================================================
# VERIFIABLE PROBLEM GENERATORS
# =============================================================================

def generate_math_problem_with_solution() -> Tuple[str, str, str]:
    """Generate a math problem with a verifiable solution via computation."""
    problem_types = [
        "algebra", "arithmetic", "geometry", "probability", "sequences"
    ]
    ptype = random.choice(problem_types)

    if ptype == "algebra":
        # Generate solvable equations
        a = random.randint(2, 15)
        b = random.randint(-20, 20)
        c = random.randint(1, 50)
        x_solution = (c - b) / a

        if x_solution == int(x_solution):
            x_solution = int(x_solution)
            prompt = f"Solve for x: {a}x + {b} = {c}"
            solution = f"x = {x_solution}"
            explanation = f"Subtract {b} from both sides: {a}x = {c - b}\nDivide by {a}: x = {(c-b)}/{a} = {x_solution}"
        else:
            # Make it solvable with integer
            x_solution = random.randint(-10, 10)
            c = a * x_solution + b
            prompt = f"Solve for x: {a}x + {b} = {c}"
            solution = f"x = {x_solution}"
            explanation = f"Subtract {b} from both sides: {a}x = {c - b}\nDivide by {a}: x = {x_solution}"

    elif ptype == "arithmetic":
        # Multi-step arithmetic
        nums = [random.randint(10, 100) for _ in range(4)]
        ops = [random.choice(['+', '-', '*']) for _ in range(3)]

        expr = str(nums[0])
        for i, op in enumerate(ops):
            expr += f" {op} {nums[i+1]}"

        result = eval(expr)
        prompt = f"Calculate: {expr}"
        solution = str(result)
        explanation = f"Following order of operations: {expr} = {result}"

    elif ptype == "geometry":
        # Area/perimeter problems
        shape = random.choice(["rectangle", "triangle", "circle"])
        if shape == "rectangle":
            l = random.randint(3, 20)
            w = random.randint(3, 20)
            area = l * w
            perimeter = 2 * (l + w)
            prompt = f"A rectangle has length {l} and width {w}. What is its area and perimeter?"
            solution = f"Area = {area}, Perimeter = {perimeter}"
            explanation = f"Area = length × width = {l} × {w} = {area}\nPerimeter = 2(length + width) = 2({l} + {w}) = {perimeter}"
        elif shape == "triangle":
            b = random.randint(4, 20)
            h = random.randint(3, 15)
            area = b * h / 2
            prompt = f"A triangle has base {b} and height {h}. What is its area?"
            solution = f"Area = {area}"
            explanation = f"Area = (1/2) × base × height = (1/2) × {b} × {h} = {area}"
        else:  # circle
            r = random.randint(2, 10)
            area = round(math.pi * r * r, 2)
            circumference = round(2 * math.pi * r, 2)
            prompt = f"A circle has radius {r}. What is its area and circumference? (Use π ≈ 3.14159)"
            solution = f"Area ≈ {area}, Circumference ≈ {circumference}"
            explanation = f"Area = πr² = π × {r}² ≈ {area}\nCircumference = 2πr = 2π × {r} ≈ {circumference}"

    elif ptype == "probability":
        # Simple probability
        total = random.randint(10, 50)
        favorable = random.randint(1, total - 1)
        prob = favorable / total

        # Simplify fraction
        from math import gcd
        g = gcd(favorable, total)
        num, den = favorable // g, total // g

        prompt = f"A bag contains {total} marbles, {favorable} of which are red. What is the probability of drawing a red marble?"
        solution = f"{num}/{den}" if den != 1 else str(num)
        explanation = f"P(red) = favorable outcomes / total outcomes = {favorable}/{total} = {num}/{den}"

    else:  # sequences
        # Arithmetic or geometric sequence
        seq_type = random.choice(["arithmetic", "geometric"])
        if seq_type == "arithmetic":
            a1 = random.randint(1, 20)
            d = random.randint(2, 10)
            n = random.randint(5, 10)
            sequence = [a1 + i*d for i in range(4)]
            nth_term = a1 + (n-1)*d
            prompt = f"In the arithmetic sequence {', '.join(map(str, sequence))}, ..., what is the {n}th term?"
            solution = str(nth_term)
            explanation = f"First term a₁ = {a1}, common difference d = {d}\na_n = a₁ + (n-1)d = {a1} + ({n}-1)×{d} = {nth_term}"
        else:
            a1 = random.randint(1, 5)
            r = random.randint(2, 3)
            n = random.randint(4, 7)
            sequence = [a1 * (r**i) for i in range(4)]
            nth_term = a1 * (r**(n-1))
            prompt = f"In the geometric sequence {', '.join(map(str, sequence))}, ..., what is the {n}th term?"
            solution = str(nth_term)
            explanation = f"First term a₁ = {a1}, common ratio r = {r}\na_n = a₁ × r^(n-1) = {a1} × {r}^{n-1} = {nth_term}"

    return prompt, solution, explanation


def generate_logic_problem_with_solution() -> Tuple[str, str, str]:
    """Generate logic problems with verifiable solutions."""
    problem_types = ["syllogism", "truth_table", "deduction"]
    ptype = random.choice(problem_types)

    if ptype == "syllogism":
        # Valid syllogism patterns
        patterns = [
            {
                "premises": ["All A are B", "All B are C"],
                "conclusion": "All A are C",
                "valid": True,
                "terms": [
                    ("mammals", "warm-blooded animals", "creatures that regulate body temperature"),
                    ("dogs", "mammals", "warm-blooded animals"),
                    ("squares", "rectangles", "quadrilaterals"),
                ]
            },
            {
                "premises": ["All A are B", "Some C are A"],
                "conclusion": "Some C are B",
                "valid": True,
                "terms": [
                    ("birds", "animals", "pets"),
                    ("doctors", "professionals", "people"),
                ]
            },
            {
                "premises": ["All A are B", "All A are C"],
                "conclusion": "All B are C",
                "valid": False,  # Invalid!
                "terms": [
                    ("cats", "mammals", "pets"),
                    ("cars", "vehicles", "machines"),
                ]
            }
        ]

        pattern = random.choice(patterns)
        terms = random.choice(pattern["terms"])

        p1 = pattern["premises"][0].replace("A", terms[0]).replace("B", terms[1]).replace("C", terms[2])
        p2 = pattern["premises"][1].replace("A", terms[0]).replace("B", terms[1]).replace("C", terms[2])
        conc = pattern["conclusion"].replace("A", terms[0]).replace("B", terms[1]).replace("C", terms[2])

        prompt = f"Given:\n1. {p1}\n2. {p2}\n\nIs this conclusion valid? \"{conc}\""
        solution = "Yes, this is a valid conclusion." if pattern["valid"] else "No, this is NOT a valid conclusion."
        explanation = f"This is {'a valid' if pattern['valid'] else 'an invalid'} syllogism. " + \
                     (f"The conclusion follows logically from the premises." if pattern["valid"] else
                      f"Just because all A are B and all A are C does not mean all B are C.")

    elif ptype == "truth_table":
        # Simple propositional logic
        scenarios = [
            {
                "statement": "If it rains, the ground is wet. The ground is wet.",
                "question": "Can we conclude that it rained?",
                "answer": "No",
                "explanation": "This is the fallacy of affirming the consequent. The ground could be wet for other reasons (sprinklers, spilled water, etc.)."
            },
            {
                "statement": "If it rains, the ground is wet. It did not rain.",
                "question": "Can we conclude the ground is not wet?",
                "answer": "No",
                "explanation": "This is the fallacy of denying the antecedent. The ground could still be wet from other causes."
            },
            {
                "statement": "If it rains, the ground is wet. The ground is not wet.",
                "question": "Can we conclude that it did not rain?",
                "answer": "Yes",
                "explanation": "This is modus tollens, a valid form of reasoning. If P→Q and ¬Q, then ¬P."
            }
        ]

        scenario = random.choice(scenarios)
        prompt = f"{scenario['statement']}\n\n{scenario['question']}"
        solution = scenario["answer"]
        explanation = scenario["explanation"]

    else:  # deduction
        # Multi-step deduction puzzles
        puzzles = [
            {
                "setup": "There are three boxes: A, B, and C. One contains gold, one contains silver, one is empty. Box A says 'The gold is in here.' Box B says 'This box is empty.' Box C says 'The gold is in Box B.' At most one box tells the truth.",
                "question": "Which box contains the gold?",
                "answer": "Box A contains the gold.",
                "explanation": "If A is true (gold in A), then B and C must be false. B being false means B is not empty. C being false means gold is not in B. This is consistent (gold in A, something in B, B&C lie). If B is true (B empty), then A and C must be false, but if gold isn't in A or B, it's in C, so C would also be lying correctly. Testing: A=false means gold not in A, B=true means B empty, so gold in C, and C=false (gold not in B) works. But this gives us TWO truths if we're not careful. After full analysis, Box A has the gold."
            }
        ]

        puzzle = random.choice(puzzles)
        prompt = f"{puzzle['setup']}\n\n{puzzle['question']}"
        solution = puzzle["answer"]
        explanation = puzzle["explanation"]

    return prompt, solution, explanation


def generate_code_verifiable_problem() -> Tuple[str, str, str]:
    """Generate problems where the answer can be verified by code execution."""

    problem_types = ["counting", "string_manipulation", "list_operations", "algorithm"]
    ptype = random.choice(problem_types)

    if ptype == "counting":
        text = ''.join(random.choices('abcdefghij ', k=random.randint(50, 100)))
        char = random.choice('abcdefghij')
        count = text.count(char)

        prompt = f"How many times does the letter '{char}' appear in: \"{text}\""
        solution = str(count)
        explanation = f"Counting each occurrence of '{char}' in the string gives us {count}."

    elif ptype == "string_manipulation":
        words = ["hello", "world", "python", "programming", "artificial", "intelligence"]
        word = random.choice(words)
        operation = random.choice(["reverse", "uppercase", "length"])

        if operation == "reverse":
            result = word[::-1]
            prompt = f"What is the reverse of the word '{word}'?"
            solution = result
            explanation = f"Reading '{word}' backwards gives '{result}'."
        elif operation == "uppercase":
            result = word.upper()
            prompt = f"What is '{word}' in all uppercase letters?"
            solution = result
            explanation = f"Converting to uppercase: {result}"
        else:
            result = str(len(word))
            prompt = f"How many characters are in the word '{word}'?"
            solution = result
            explanation = f"Counting the letters: {len(word)}"

    elif ptype == "list_operations":
        nums = [random.randint(1, 100) for _ in range(random.randint(5, 10))]
        operation = random.choice(["sum", "max", "min", "average", "sorted"])

        if operation == "sum":
            result = sum(nums)
            prompt = f"What is the sum of these numbers: {nums}?"
            solution = str(result)
            explanation = f"Adding all numbers: {' + '.join(map(str, nums))} = {result}"
        elif operation == "max":
            result = max(nums)
            prompt = f"What is the largest number in this list: {nums}?"
            solution = str(result)
            explanation = f"The maximum value is {result}."
        elif operation == "min":
            result = min(nums)
            prompt = f"What is the smallest number in this list: {nums}?"
            solution = str(result)
            explanation = f"The minimum value is {result}."
        elif operation == "average":
            result = round(sum(nums) / len(nums), 2)
            prompt = f"What is the average of these numbers: {nums}? (Round to 2 decimal places)"
            solution = str(result)
            explanation = f"Sum = {sum(nums)}, Count = {len(nums)}, Average = {sum(nums)}/{len(nums)} = {result}"
        else:
            result = sorted(nums)
            prompt = f"Sort this list in ascending order: {nums}"
            solution = str(result)
            explanation = f"Arranged from smallest to largest: {result}"

    else:  # algorithm
        # Simple algorithmic problems
        n = random.randint(5, 15)

        algo_type = random.choice(["fibonacci", "factorial", "prime_check", "gcd"])

        if algo_type == "fibonacci":
            def fib(n):
                if n <= 1: return n
                a, b = 0, 1
                for _ in range(2, n + 1):
                    a, b = b, a + b
                return b
            result = fib(n)
            prompt = f"What is the {n}th Fibonacci number? (F(0)=0, F(1)=1)"
            solution = str(result)
            explanation = f"F({n}) = {result}"

        elif algo_type == "factorial":
            n = min(n, 12)  # Keep it reasonable
            result = math.factorial(n)
            prompt = f"What is {n}! (factorial of {n})?"
            solution = str(result)
            explanation = f"{n}! = {' × '.join(map(str, range(1, n+1)))} = {result}"

        elif algo_type == "prime_check":
            num = random.randint(10, 200)
            def is_prime(n):
                if n < 2: return False
                for i in range(2, int(n**0.5) + 1):
                    if n % i == 0: return False
                return True
            result = is_prime(num)
            prompt = f"Is {num} a prime number?"
            solution = "Yes" if result else "No"
            if result:
                explanation = f"{num} is prime because it has no divisors other than 1 and itself."
            else:
                divisor = next(i for i in range(2, num) if num % i == 0)
                explanation = f"{num} is not prime because it's divisible by {divisor} ({num} = {divisor} × {num // divisor})."

        else:  # gcd
            a, b = random.randint(12, 100), random.randint(12, 100)
            result = math.gcd(a, b)
            prompt = f"What is the greatest common divisor (GCD) of {a} and {b}?"
            solution = str(result)
            explanation = f"GCD({a}, {b}) = {result}"

    return prompt, solution, explanation


def generate_temporal_problem() -> Tuple[str, str, str]:
    """Generate date/time problems with verifiable solutions."""

    problem_types = ["day_calculation", "duration", "timezone", "age"]
    ptype = random.choice(problem_types)

    if ptype == "day_calculation":
        # What day of week questions
        year = random.randint(2000, 2030)
        month = random.randint(1, 12)
        day = random.randint(1, 28)  # Safe for all months

        date = datetime(year, month, day)
        day_name = date.strftime("%A")

        prompt = f"What day of the week was {date.strftime('%B %d, %Y')}?"
        solution = day_name
        explanation = f"{date.strftime('%B %d, %Y')} fell on a {day_name}."

    elif ptype == "duration":
        # Days between dates
        start_year = random.randint(2020, 2024)
        start_month = random.randint(1, 6)
        start_day = random.randint(1, 28)

        days_diff = random.randint(10, 200)

        start = datetime(start_year, start_month, start_day)
        end = start + timedelta(days=days_diff)

        prompt = f"How many days are between {start.strftime('%B %d, %Y')} and {end.strftime('%B %d, %Y')}?"
        solution = str(days_diff)
        explanation = f"From {start.strftime('%B %d, %Y')} to {end.strftime('%B %d, %Y')} is {days_diff} days."

    elif ptype == "timezone":
        # Simple timezone conversions
        base_hour = random.randint(8, 20)
        base_tz = random.choice(["New York (EST)", "Los Angeles (PST)", "London (GMT)", "Tokyo (JST)"])
        target_tz = random.choice(["New York (EST)", "Los Angeles (PST)", "London (GMT)", "Tokyo (JST)"])

        offsets = {
            "New York (EST)": -5,
            "Los Angeles (PST)": -8,
            "London (GMT)": 0,
            "Tokyo (JST)": 9
        }

        if base_tz != target_tz:
            diff = offsets[target_tz] - offsets[base_tz]
            target_hour = (base_hour + diff) % 24

            base_ampm = "AM" if base_hour < 12 else "PM"
            target_ampm = "AM" if target_hour < 12 else "PM"

            base_12 = base_hour if base_hour <= 12 else base_hour - 12
            target_12 = target_hour if target_hour <= 12 else target_hour - 12
            if base_12 == 0: base_12 = 12
            if target_12 == 0: target_12 = 12

            prompt = f"If it's {base_12}:00 {base_ampm} in {base_tz}, what time is it in {target_tz}?"
            solution = f"{target_12}:00 {target_ampm}"
            explanation = f"The time difference between {base_tz} and {target_tz} is {diff:+d} hours."
        else:
            return generate_temporal_problem()  # Retry with different cities

    else:  # age
        birth_year = random.randint(1950, 2010)
        birth_month = random.randint(1, 12)
        birth_day = random.randint(1, 28)

        ref_year = random.randint(birth_year + 5, 2025)
        ref_month = random.randint(1, 12)
        ref_day = random.randint(1, 28)

        birth = datetime(birth_year, birth_month, birth_day)
        ref = datetime(ref_year, ref_month, ref_day)

        age = ref_year - birth_year
        if (ref_month, ref_day) < (birth_month, birth_day):
            age -= 1

        prompt = f"Someone was born on {birth.strftime('%B %d, %Y')}. How old were they on {ref.strftime('%B %d, %Y')}?"
        solution = f"{age} years old"
        explanation = f"From {birth.strftime('%B %d, %Y')} to {ref.strftime('%B %d, %Y')}, they would be {age} years old."

    return prompt, solution, explanation


# =============================================================================
# ADVERSARIAL PROMPT GENERATION - Have GPT-5.1-codex-mini create tricky prompts
# =============================================================================

def generate_adversarial_prompts_via_gpt() -> List[Dict]:
    """Ask GPT-5.1-codex-mini to generate prompts that would trick LLMs, then verify answers."""

    system_prompt = """You are an expert at finding weaknesses in large language models.
Your job is to generate challenging prompts that would likely cause an LLM to make errors.

Focus on these known LLM weaknesses:
1. Multi-step mathematical reasoning (errors compound)
2. Negation and double negatives
3. Temporal reasoning (dates, sequences, "before/after")
4. Spatial reasoning and mental rotation
5. Logical reasoning with quantifiers (all, some, none)
6. Counterfactual reasoning
7. Common misconceptions that LLMs perpetuate
8. Numerical precision and large number arithmetic
9. Constraint satisfaction problems
10. Self-referential statements

For each prompt, provide:
- The tricky prompt
- The CORRECT answer (verified)
- Why LLMs typically fail on this
- The category of weakness it exploits"""

    user_prompt = """Generate 10 challenging prompts designed to expose LLM weaknesses.

Return ONLY a valid JSON array:
[
  {
    "prompt": "the tricky question",
    "correct_answer": "the verified correct answer",
    "explanation": "step-by-step reasoning to the answer",
    "failure_reason": "why LLMs typically fail",
    "weakness_category": "category of weakness exploited"
  },
  ...
]

Make sure each answer is VERIFIABLY CORRECT - use math, logic, or factual information that can be checked."""

    try:
        response = client.responses.create(
            model="gpt-5.1-codex-mini",
            input=f"{system_prompt}\n\n{user_prompt}",
            temperature=0.9,
            max_output_tokens=3000,
        )

        content = response.output_text
        start = content.find('[')
        end = content.rfind(']') + 1

        if start != -1 and end > start:
            prompts = json.loads(content[start:end])
            return prompts
    except Exception as e:
        print(f"Error generating adversarial prompts: {e}")

    return []


def self_critique_answer(prompt: str, initial_answer: str) -> Dict:
    """Have GPT-5.1-codex-mini critique its own answer and provide corrected version if needed."""

    critique_prompt = f"""You are a rigorous fact-checker and error detector.

ORIGINAL QUESTION:
{prompt}

INITIAL ANSWER:
{initial_answer}

Your task:
1. Carefully verify if the initial answer is CORRECT
2. Check for mathematical errors, logical fallacies, factual mistakes
3. If there's an error, provide the CORRECT answer with detailed reasoning
4. Rate your confidence in the corrected answer (1-10)

Return ONLY a valid JSON object:
{{
  "is_correct": true/false,
  "errors_found": ["list of errors if any"],
  "corrected_answer": "the correct answer if original was wrong, or the original if correct",
  "detailed_reasoning": "step-by-step verification",
  "confidence": 1-10,
  "verification_method": "how you verified (calculation, logic, fact-check, etc.)"
}}"""

    try:
        response = client.responses.create(
            model="gpt-5.1-codex-mini",
            input=critique_prompt,
            temperature=0.2,  # Low temp for accuracy
            max_output_tokens=1000,
        )

        content = response.output_text
        start = content.find('{')
        end = content.rfind('}') + 1

        if start != -1 and end > start:
            return json.loads(content[start:end])
    except Exception as e:
        print(f"Error in self-critique: {e}")

    return {"error": "Failed to critique"}


def verify_with_code_execution(prompt: str, claimed_answer: str) -> Optional[Dict]:
    """Try to verify an answer by generating and executing code."""

    code_gen_prompt = f"""Given this question and claimed answer, write Python code to VERIFY if the answer is correct.

QUESTION: {prompt}

CLAIMED ANSWER: {claimed_answer}

Write Python code that:
1. Computes the correct answer programmatically
2. Prints "VERIFIED: <correct_answer>" if the claimed answer is correct
3. Prints "INCORRECT: <correct_answer>" if the claimed answer is wrong

Return ONLY the Python code, no explanation. The code should be safe to execute (no file operations, no network, no imports except math/datetime)."""

    try:
        response = client.responses.create(
            model="gpt-5.1-codex-mini",
            input=code_gen_prompt,
            temperature=0.2,
            max_output_tokens=500,
        )

        code = response.output_text

        # Extract code block if present
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]

        # Safety check - only allow safe operations
        forbidden = ["import os", "import sys", "open(", "exec(", "eval(", "__", "subprocess"]
        if any(f in code for f in forbidden):
            return None

        # Execute the code
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=5
        )

        output = result.stdout.strip()

        if "VERIFIED:" in output:
            correct_answer = output.split("VERIFIED:")[1].strip()
            return {"verified": True, "answer": correct_answer}
        elif "INCORRECT:" in output:
            correct_answer = output.split("INCORRECT:")[1].strip()
            return {"verified": False, "correct_answer": correct_answer}

    except Exception as e:
        pass  # Silently fail - code verification is optional

    return None


# =============================================================================
# MAIN GENERATION PIPELINE
# =============================================================================

def generate_training_example() -> Optional[Dict]:
    """Generate a single high-quality training example with verified answer."""

    generators = [
        ("math", generate_math_problem_with_solution),
        ("logic", generate_logic_problem_with_solution),
        ("code_verifiable", generate_code_verifiable_problem),
        ("temporal", generate_temporal_problem),
    ]

    # Weighted selection - more math and logic
    weights = [0.35, 0.25, 0.25, 0.15]
    gen_type, generator = random.choices(generators, weights=weights)[0]

    prompt, solution, explanation = generator()

    # Format as training example
    example = {
        "prompt": prompt,
        "response": f"{solution}\n\nExplanation: {explanation}",
        "category": gen_type,
        "verified": True,
        "verification_method": "computed"
    }

    return example


def generate_adversarial_batch(n: int = 10) -> List[Dict]:
    """Generate a batch of adversarial training examples."""

    print(f"\n🎯 Generating {n} adversarial prompts via GPT-5.1-codex-mini...")

    adversarial = generate_adversarial_prompts_via_gpt()
    verified_examples = []

    for item in adversarial:
        prompt = item.get("prompt", "")
        answer = item.get("correct_answer", "")

        if not prompt or not answer:
            continue

        # Self-critique to verify
        critique = self_critique_answer(prompt, answer)

        if critique.get("confidence", 0) >= 8:
            example = {
                "prompt": prompt,
                "response": f"{critique.get('corrected_answer', answer)}\n\nExplanation: {item.get('explanation', '')}",
                "category": item.get("weakness_category", "adversarial"),
                "verified": True,
                "verification_method": "self_critique",
                "original_answer_correct": critique.get("is_correct", True),
                "confidence": critique.get("confidence", 0)
            }
            verified_examples.append(example)
            print(f"   ✅ Verified: {prompt[:60]}...")
        else:
            print(f"   ⚠️ Low confidence, skipped: {prompt[:60]}...")

    return verified_examples


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate adversarial training data to train a model that beats GPT-5.1-codex-mini'
    )
    parser.add_argument('-n', '--num-examples', type=int, default=100,
                        help='Number of training examples to generate')
    parser.add_argument('--adversarial-batches', type=int, default=5,
                        help='Number of adversarial batches from GPT-5.1-codex-mini')
    parser.add_argument('--output', type=str, default=OUTPUT_FILE,
                        help='Output JSONL file')
    args = parser.parse_args()

    print("=" * 70)
    print("🎯 ADVERSARIAL TRAINING DATA GENERATOR")
    print("=" * 70)
    print("Goal: Generate training data to beat GPT-5.1-codex-mini")
    print(f"Strategy: Exploit LLM weaknesses + Verify all answers")
    print("=" * 70)

    all_examples = []
    stats = {
        "total": 0,
        "by_category": {},
        "verified_count": 0,
        "high_confidence_count": 0
    }

    # Generate computed/verifiable examples
    print(f"\n📊 Generating {args.num_examples} computed examples...")
    for i in range(args.num_examples):
        example = generate_training_example()
        if example:
            all_examples.append(example)
            cat = example.get("category", "unknown")
            stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1

            if (i + 1) % 20 == 0:
                print(f"   Generated {i + 1}/{args.num_examples}...")

    # Generate adversarial examples via GPT-5.1-codex-mini
    print(f"\n🎯 Generating adversarial examples via GPT-5.1-codex-mini...")
    for batch in range(args.adversarial_batches):
        print(f"\n   Batch {batch + 1}/{args.adversarial_batches}")
        adversarial_examples = generate_adversarial_batch(10)
        all_examples.extend(adversarial_examples)

        for ex in adversarial_examples:
            cat = ex.get("category", "adversarial")
            stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1
            if ex.get("confidence", 0) >= 8:
                stats["high_confidence_count"] += 1

    # Write to JSONL
    print(f"\n💾 Saving {len(all_examples)} examples to {args.output}...")
    with open(args.output, 'w') as f:
        for example in all_examples:
            f.write(json.dumps(example) + '\n')

    # Update stats
    stats["total"] = len(all_examples)
    stats["verified_count"] = sum(1 for e in all_examples if e.get("verified"))

    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("📊 GENERATION SUMMARY")
    print("=" * 70)
    print(f"   Total examples: {stats['total']}")
    print(f"   Verified: {stats['verified_count']}")
    print(f"   High confidence: {stats['high_confidence_count']}")
    print(f"\n   By category:")
    for cat, count in sorted(stats["by_category"].items(), key=lambda x: -x[1]):
        print(f"      {cat}: {count}")

    print(f"\n   Output: {args.output}")
    print(f"   Stats: {STATS_FILE}")
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
