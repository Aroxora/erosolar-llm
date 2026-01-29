import random
from typing import Dict, List, Tuple


NAMES = ["Alex", "Sam", "Riley", "Jordan", "Taylor", "Casey", "Morgan", "Jamie"]
ITEMS = ["apples", "books", "coins", "stickers", "marbles", "tickets", "cookies", "pens", "bottles"]
UNITS = [("km", "m", 1000), ("m", "cm", 100), ("kg", "g", 1000), ("hours", "minutes", 60), ("dollars", "cents", 100)]


def _tier_from_steps(steps: int) -> str:
    if steps <= 3:
        return "short"
    if steps <= 6:
        return "medium"
    return "long"


def _format_steps(lines: List[str]) -> str:
    return "\n".join(f"Step {i + 1}: {line}" for i, line in enumerate(lines))


def _short_math_extension(rng: random.Random) -> Tuple[str, str, int, str]:
    a = rng.randint(2, 9)
    b = rng.randint(2, 9)
    d = rng.randint(1, 9)
    c = a + b
    result = c + d
    prompt = f"We know {a}+{b}={c}. What is {a}+{b}+{d}?"
    steps = [
        f"Use the foundation: {a}+{b}={c}",
        f"Add {d}: {c}+{d}={result}",
    ]
    response = (
        f"Foundation: {a}+{b}={c}\n"
        f"{_format_steps(steps)}\n"
        f"Answer: {result}"
    )
    return prompt, response, len(steps), "math_extension"


def _short_word_problem(rng: random.Random) -> Tuple[str, str, int, str]:
    name = rng.choice(NAMES)
    item = rng.choice(ITEMS)
    start = rng.randint(3, 12)
    add = rng.randint(2, 9)
    sub = rng.randint(1, min(6, start + add - 1))
    after_add = start + add
    result = after_add - sub
    prompt = f"{name} has {start} {item}. {name} gets {add} more and gives away {sub}. How many {item} now?"
    steps = [
        f"Add {add}: {start}+{add}={after_add}",
        f"Subtract {sub}: {after_add}-{sub}={result}",
    ]
    response = (
        "Foundation: basic addition and subtraction\n"
        f"{_format_steps(steps)}\n"
        f"Answer: {result} {item}"
    )
    return prompt, response, len(steps), "word_problem"


def _short_pattern(rng: random.Random) -> Tuple[str, str, int, str]:
    start = rng.randint(1, 12)
    step = rng.randint(2, 9)
    seq = [start + i * step for i in range(4)]
    next_val = seq[-1] + step
    prompt = f"Continue the pattern: {seq[0]}, {seq[1]}, {seq[2]}, {seq[3]}, ?"
    steps = [
        f"Pattern adds {step} each time",
        f"Next is {seq[3]}+{step}={next_val}",
    ]
    response = (
        f"Foundation: arithmetic sequence\n"
        f"{_format_steps(steps)}\n"
        f"Answer: {next_val}"
    )
    return prompt, response, len(steps), "pattern"


def _short_logic(rng: random.Random) -> Tuple[str, str, int, str]:
    rules = [
        ("birds", "wings", "sparrow"),
        ("mammals", "fur", "cat"),
        ("triangles", "three sides", "triangle"),
        ("squares", "four sides", "square"),
    ]
    group, prop, example = rng.choice(rules)
    prompt = f"All {group} have {prop}. A {example} is a {group[:-1]}. Does a {example} have {prop}?"
    steps = [
        f"{example.title()} is a {group[:-1]}",
        f"All {group} have {prop}, so {example} has {prop}",
    ]
    response = (
        f"Foundation: all {group} have {prop}\n"
        f"{_format_steps(steps)}\n"
        "Answer: Yes"
    )
    return prompt, response, len(steps), "logic"


def _medium_math_chain(rng: random.Random) -> Tuple[str, str, int, str]:
    start = rng.randint(5, 20)
    ops = []
    value = start
    for _ in range(4):
        if value <= 5:
            op = "+"
        else:
            op = rng.choice(["+", "-"])
        if op == "+":
            n = rng.randint(1, 9)
            value += n
        else:
            n = rng.randint(1, min(9, value - 1))
            value -= n
        ops.append((op, n, value))
    ops_str = ", ".join(f"{op}{n}" for op, n, _ in ops)
    prompt = f"Start with {start}. Apply: {ops_str}. What is the result?"
    steps = []
    prev = start
    for op, n, val in ops:
        steps.append(f"{prev} {op} {n} = {val}")
        prev = val
    response = (
        "Foundation: chained addition/subtraction\n"
        f"{_format_steps(steps)}\n"
        f"Answer: {value}"
    )
    return prompt, response, len(steps), "math_chain"


def _medium_context_update(rng: random.Random) -> Tuple[str, str, int, str]:
    x = rng.randint(3, 10)
    add = rng.randint(2, 8)
    mul = rng.randint(2, 4)
    sub = rng.randint(1, 6)
    add2 = rng.randint(1, 6)
    step1 = x + add
    step2 = step1 * mul
    step3 = step2 - sub
    step4 = step3 + add2
    prompt = (
        f"x = {x}. Apply: x = x + {add}; x = x * {mul}; x = x - {sub}; x = x + {add2}. "
        "What is x?"
    )
    steps = [
        f"x = {x}+{add} = {step1}",
        f"x = {step1}*{mul} = {step2}",
        f"x = {step2}-{sub} = {step3}",
        f"x = {step3}+{add2} = {step4}",
    ]
    response = (
        "Foundation: variables update step by step\n"
        f"{_format_steps(steps)}\n"
        f"Answer: {step4}"
    )
    return prompt, response, len(steps), "context_update"


def _medium_coding_trace(rng: random.Random) -> Tuple[str, str, int, str]:
    x = rng.randint(2, 9)
    add = rng.randint(1, 7)
    mul = rng.randint(2, 4)
    sub = rng.randint(1, 6)
    step1 = x + add
    step2 = step1 * mul
    step3 = step2 - sub
    prompt = (
        "What is the output of this code?\n"
        "```python\n"
        f"x = {x}\n"
        f"x = x + {add}\n"
        f"x = x * {mul}\n"
        f"x = x - {sub}\n"
        "print(x)\n"
        "```"
    )
    steps = [
        f"x = {x}+{add} = {step1}",
        f"x = {step1}*{mul} = {step2}",
        f"x = {step2}-{sub} = {step3}",
    ]
    response = (
        "Foundation: variables store values and update by operations\n"
        f"{_format_steps(steps)}\n"
        f"Answer: {step3}"
    )
    return prompt, response, len(steps), "coding_trace"


def _long_arithmetic_chain(rng: random.Random) -> Tuple[str, str, int, str]:
    start = rng.randint(8, 15)
    ops = []
    value = start
    for _ in range(7):
        if value <= 6:
            op = "+"
        else:
            op = rng.choice(["+", "-"])
        if op == "+":
            n = rng.randint(1, 9)
            value += n
        else:
            n = rng.randint(1, min(9, value - 1))
            value -= n
        ops.append((op, n, value))
    ops_str = ", ".join(f"{op}{n}" for op, n, _ in ops)
    prompt = f"Start with {start}. Apply: {ops_str}. What is the result?"
    steps = []
    prev = start
    for op, n, val in ops:
        steps.append(f"{prev} {op} {n} = {val}")
        prev = val
    response = (
        "Foundation: multi-step arithmetic chain\n"
        f"{_format_steps(steps)}\n"
        f"Answer: {value}"
    )
    return prompt, response, len(steps), "long_chain"


def _long_multi_variable(rng: random.Random) -> Tuple[str, str, int, str]:
    a = rng.randint(2, 6)
    b = rng.randint(4, 8)
    k = rng.randint(1, 4)
    m = rng.randint(1, 4)
    n = rng.randint(1, 3)
    p = rng.randint(2, 5)

    a1 = a + b
    b1 = b + k
    a2 = a1 + b1
    b2 = b1 + a2
    a3 = a2 - m
    b3 = b2 - n
    a4 = a3 + p

    prompt = (
        f"a = {a}, b = {b}. Apply: a = a + b; b = b + {k}; "
        f"a = a + b; b = b + a; a = a - {m}; b = b - {n}; a = a + {p}. "
        "What is a?"
    )
    steps = [
        f"a = {a}+{b} = {a1}",
        f"b = {b}+{k} = {b1}",
        f"a = {a1}+{b1} = {a2}",
        f"b = {b1}+{a2} = {b2}",
        f"a = {a2}-{m} = {a3}",
        f"b = {b2}-{n} = {b3}",
        f"a = {a3}+{p} = {a4}",
    ]
    response = (
        "Foundation: track multiple variables step by step\n"
        f"{_format_steps(steps)}\n"
        f"Answer: {a4}"
    )
    return prompt, response, len(steps), "multi_variable"


# ============================================================================
# ROBUST VERIFIABLE GENERATORS - designed for automated verification
# ============================================================================

def _short_percentage(rng: random.Random) -> Tuple[str, str, int, str]:
    """Calculate percentage of a number - easily verifiable."""
    base = rng.choice([100, 200, 50, 80, 120, 500])
    pct = rng.choice([10, 20, 25, 50, 75])
    result = base * pct // 100
    prompt = f"What is {pct}% of {base}?"
    steps = [
        f"To find {pct}% of {base}, multiply {base} by {pct}/100",
        f"{base} * {pct}/100 = {base} * {pct/100} = {result}",
    ]
    response = (
        f"Foundation: percentage means per hundred\n"
        f"{_format_steps(steps)}\n"
        f"Answer: {result}"
    )
    return prompt, response, len(steps), "percentage"


def _short_comparison(rng: random.Random) -> Tuple[str, str, int, str]:
    """Compare two expressions - verifiable true/false."""
    a1, a2 = rng.randint(2, 9), rng.randint(2, 9)
    b1, b2 = rng.randint(2, 9), rng.randint(2, 9)
    left = a1 * a2
    right = b1 * b2
    if left > right:
        answer = f"{a1}*{a2}"
        comparison = "greater"
    elif left < right:
        answer = f"{b1}*{b2}"
        comparison = "greater"
    else:
        answer = "equal"
        comparison = "equal"
    prompt = f"Which is bigger: {a1}*{a2} or {b1}*{b2}?"
    steps = [
        f"Calculate {a1}*{a2} = {left}",
        f"Calculate {b1}*{b2} = {right}",
        f"Compare: {left} vs {right}",
    ]
    if comparison == "equal":
        response = f"Foundation: compare by computing each side\n{_format_steps(steps)}\nAnswer: They are equal ({left})"
    else:
        response = f"Foundation: compare by computing each side\n{_format_steps(steps)}\nAnswer: {answer} = {max(left, right)} is bigger"
    return prompt, response, len(steps), "comparison"


def _short_unit_conversion(rng: random.Random) -> Tuple[str, str, int, str]:
    """Unit conversion - verifiable multiplication."""
    big_unit, small_unit, factor = rng.choice(UNITS)
    value = rng.randint(1, 10)
    result = value * factor
    prompt = f"Convert {value} {big_unit} to {small_unit}."
    steps = [
        f"1 {big_unit} = {factor} {small_unit}",
        f"{value} {big_unit} = {value} * {factor} = {result} {small_unit}",
    ]
    response = (
        f"Foundation: unit conversion by multiplication\n"
        f"{_format_steps(steps)}\n"
        f"Answer: {result} {small_unit}"
    )
    return prompt, response, len(steps), "unit_conversion"


def _short_equation_solve(rng: random.Random) -> Tuple[str, str, int, str]:
    """Simple equation solving - x + a = b."""
    x = rng.randint(2, 15)
    a = rng.randint(1, 10)
    b = x + a
    prompt = f"Solve for x: x + {a} = {b}"
    steps = [
        f"Subtract {a} from both sides",
        f"x = {b} - {a} = {x}",
    ]
    response = (
        f"Foundation: isolate x by inverse operations\n"
        f"{_format_steps(steps)}\n"
        f"Answer: x = {x}"
    )
    return prompt, response, len(steps), "equation"


def _short_bool_logic(rng: random.Random) -> Tuple[str, str, int, str]:
    """Boolean logic - verifiable True/False."""
    a = rng.randint(1, 10)
    b = rng.randint(1, 10)
    op = rng.choice(["AND", "OR"])
    cond1 = a > 5
    cond2 = b > 5
    if op == "AND":
        result = cond1 and cond2
        explanation = f"both {a}>5 ({cond1}) AND {b}>5 ({cond2})"
    else:
        result = cond1 or cond2
        explanation = f"either {a}>5 ({cond1}) OR {b}>5 ({cond2})"
    prompt = f"If a={a} and b={b}, is (a>5 {op} b>5) true?"
    steps = [
        f"Check a>5: {a}>5 is {cond1}",
        f"Check b>5: {b}>5 is {cond2}",
        f"Apply {op}: {explanation}",
    ]
    response = (
        f"Foundation: evaluate each condition then combine with {op}\n"
        f"{_format_steps(steps)}\n"
        f"Answer: {result}"
    )
    return prompt, response, len(steps), "bool_logic"


def _medium_fraction_chain(rng: random.Random) -> Tuple[str, str, int, str]:
    """Fraction operations chain - verifiable arithmetic."""
    start = rng.randint(24, 120)  # Divisible by common numbers
    while start % 2 != 0 or start % 3 != 0:
        start = rng.randint(24, 120)
    half = start // 2
    third = start // 3
    result = start - half + third
    prompt = f"Start with {start}. Subtract half. Add a third of the original. What's the result?"
    steps = [
        f"Half of {start} = {half}",
        f"Third of {start} = {third}",
        f"Calculation: {start} - {half} + {third}",
        f"= {start - half} + {third} = {result}",
    ]
    response = (
        f"Foundation: fractions are division, apply in order\n"
        f"{_format_steps(steps)}\n"
        f"Answer: {result}"
    )
    return prompt, response, len(steps), "fraction_chain"


def _medium_list_operations(rng: random.Random) -> Tuple[str, str, int, str]:
    """List operations - verifiable final state."""
    initial = [rng.randint(1, 9) for _ in range(3)]
    ops = []
    lst = initial.copy()

    # Append
    append_val = rng.randint(1, 9)
    lst.append(append_val)
    ops.append(f"append {append_val}")

    # Remove first
    removed = lst.pop(0)
    ops.append(f"remove first element")

    # Append another
    append_val2 = rng.randint(1, 9)
    lst.append(append_val2)
    ops.append(f"append {append_val2}")

    prompt = f"Start with list {initial}. {', '.join(ops)}. What is the final list?"
    steps = [
        f"Initial: {initial}",
        f"After append {append_val}: {initial + [append_val]}",
        f"After remove first: {(initial + [append_val])[1:]}",
        f"After append {append_val2}: {lst}",
    ]
    response = (
        f"Foundation: lists are ordered, track each operation\n"
        f"{_format_steps(steps)}\n"
        f"Answer: {lst}"
    )
    return prompt, response, len(steps), "list_operations"


def _medium_conditional_math(rng: random.Random) -> Tuple[str, str, int, str]:
    """Conditional math - if/else verification."""
    x = rng.randint(1, 20)
    threshold = rng.randint(5, 15)
    add_val = rng.randint(2, 8)
    mult_val = rng.randint(2, 4)

    if x > threshold:
        result = x + add_val
        path = f"x ({x}) > {threshold}, so add {add_val}"
    else:
        result = x * mult_val
        path = f"x ({x}) <= {threshold}, so multiply by {mult_val}"

    prompt = f"If x > {threshold}, then x + {add_val}. Otherwise x * {mult_val}. What is the result when x = {x}?"
    steps = [
        f"Check condition: {x} > {threshold}? {x > threshold}",
        f"Take path: {path}",
        f"Calculate: {result}",
    ]
    response = (
        f"Foundation: evaluate condition first, then execute correct branch\n"
        f"{_format_steps(steps)}\n"
        f"Answer: {result}"
    )
    return prompt, response, len(steps), "conditional_math"


def _medium_average(rng: random.Random) -> Tuple[str, str, int, str]:
    """Calculate average - verifiable arithmetic."""
    count = rng.randint(3, 5)
    numbers = [rng.randint(10, 50) for _ in range(count)]
    total = sum(numbers)
    avg = total / count
    prompt = f"What is the average of {numbers}?"
    steps = [
        f"Sum all numbers: {' + '.join(map(str, numbers))} = {total}",
        f"Count of numbers: {count}",
        f"Average = {total} / {count} = {avg}",
    ]
    response = (
        f"Foundation: average = sum / count\n"
        f"{_format_steps(steps)}\n"
        f"Answer: {avg}"
    )
    return prompt, response, len(steps), "average"


def _long_nested_conditions(rng: random.Random) -> Tuple[str, str, int, str]:
    """Nested conditional logic - verifiable path."""
    a = rng.randint(1, 20)
    b = rng.randint(1, 20)
    t1, t2 = 10, 15

    steps = [f"Given: a={a}, b={b}"]

    if a > t1:
        steps.append(f"a ({a}) > {t1}: True, enter first branch")
        if b > t2:
            result = a + b
            steps.append(f"b ({b}) > {t2}: True, compute a+b = {result}")
        else:
            result = a - b
            steps.append(f"b ({b}) > {t2}: False, compute a-b = {result}")
    else:
        steps.append(f"a ({a}) > {t1}: False, enter else branch")
        if b > t2:
            result = a * b
            steps.append(f"b ({b}) > {t2}: True, compute a*b = {result}")
        else:
            result = a + b + 10
            steps.append(f"b ({b}) > {t2}: False, compute a+b+10 = {result}")

    prompt = (
        f"Given a={a}, b={b}:\n"
        f"If a > {t1}:\n"
        f"  If b > {t2}: return a + b\n"
        f"  Else: return a - b\n"
        f"Else:\n"
        f"  If b > {t2}: return a * b\n"
        f"  Else: return a + b + 10\n"
        f"What is returned?"
    )
    response = (
        f"Foundation: evaluate conditions in order, follow correct path\n"
        f"{_format_steps(steps)}\n"
        f"Answer: {result}"
    )
    return prompt, response, len(steps), "nested_conditions"


def _long_loop_simulation(rng: random.Random) -> Tuple[str, str, int, str]:
    """Simulate a loop - verifiable final state."""
    iterations = rng.randint(4, 6)
    start = rng.randint(1, 5)
    add_each = rng.randint(2, 4)

    steps = [f"Start with total = {start}"]
    total = start
    for i in range(iterations):
        total += add_each
        steps.append(f"Iteration {i+1}: total = {total - add_each} + {add_each} = {total}")

    prompt = f"Start with total = {start}. Loop {iterations} times, adding {add_each} each time. What is the final total?"
    response = (
        f"Foundation: trace each iteration of the loop\n"
        f"{_format_steps(steps)}\n"
        f"Answer: {total}"
    )
    return prompt, response, len(steps), "loop_simulation"


def _long_compound_interest(rng: random.Random) -> Tuple[str, str, int, str]:
    """Compound interest over periods - verifiable math."""
    principal = rng.choice([100, 200, 500, 1000])
    rate = rng.choice([10, 20, 25])  # Percentage per period
    periods = rng.randint(2, 4)

    steps = [f"Start: ${principal}"]
    amount = principal
    for i in range(periods):
        interest = amount * rate // 100
        amount += interest
        steps.append(f"Period {i+1}: ${amount - interest} + {rate}% = ${amount}")

    prompt = f"${principal} earns {rate}% interest per period (compounded). What is the total after {periods} periods?"
    response = (
        f"Foundation: compound interest adds percentage of current amount each period\n"
        f"{_format_steps(steps)}\n"
        f"Answer: ${amount}"
    )
    return prompt, response, len(steps), "compound_interest"


def generate_synthetic_bridge_pairs(target_count: int, seed: int = 42) -> List[Dict]:
    if target_count <= 0:
        return []

    rng = random.Random(seed)
    pairs: List[Dict] = []
    seen = set()
    tier_counts = {"short": 0, "medium": 0, "long": 0}

    short_target = int(target_count * 0.45)
    medium_target = int(target_count * 0.4)
    long_target = target_count - short_target - medium_target

    tiers = [
        ("short", short_target, [
            _short_math_extension,
            _short_word_problem,
            _short_pattern,
            _short_logic,
            # NEW robust verifiable generators
            _short_percentage,
            _short_comparison,
            _short_unit_conversion,
            _short_equation_solve,
            _short_bool_logic,
        ]),
        ("medium", medium_target, [
            _medium_math_chain,
            _medium_context_update,
            _medium_coding_trace,
            # NEW robust verifiable generators
            _medium_fraction_chain,
            _medium_list_operations,
            _medium_conditional_math,
            _medium_average,
        ]),
        ("long", long_target, [
            _long_arithmetic_chain,
            _long_multi_variable,
            # NEW robust verifiable generators
            _long_nested_conditions,
            _long_loop_simulation,
            _long_compound_interest,
        ]),
    ]

    for tier, count, generators in tiers:
        attempts = 0
        max_attempts = max(50, count * 25)
        while tier_counts[tier] < count and attempts < max_attempts:
            attempts += 1
            gen = rng.choice(generators)
            prompt, response, steps, name = gen(rng)
            if prompt in seen:
                continue
            seen.add(prompt)
            pairs.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ],
                "metadata": {
                    "category": f"synthetic_{name}",
                    "type": "bridge_knowledge",
                    "bridge_tier": _tier_from_steps(steps),
                    "generator": "synthetic",
                    "bridges_from": "foundational",
                    "bridges_to": "sophisticated"
                }
            })
            tier_counts[tier] += 1

    return pairs
