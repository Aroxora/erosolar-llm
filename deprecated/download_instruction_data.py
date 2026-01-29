"""
Download high-quality instruction-following datasets for training.
"""
import json
import os
import random
from typing import List, Tuple
from pathlib import Path

CACHE_DIR = Path("cache/datasets")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def download_dolly_data(allow_download: bool = True) -> List[Tuple[str, str]]:
    """Download Databricks Dolly dataset."""
    cache_file = CACHE_DIR / "dolly_data.json"

    if cache_file.exists():
        print("Loading cached Dolly data...")
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [(d[0], d[1]) for d in data]

    if not allow_download:
        print("Skipping Dolly download (allow_download=False)")
        return []

    print("Downloading Dolly dataset...")
    try:
        import urllib.request
        urls = [
            "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl",
        ]

        lines = None
        for url in urls:
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    lines = response.read().decode('utf-8').strip().split('\n')
                    break
            except Exception:
                continue

        if not lines:
            return []

        pairs = []
        for line in lines:
            try:
                item = json.loads(line)
                instruction = item.get('instruction', '')
                context = item.get('context', '')
                resp = item.get('response', '')

                if not instruction or not resp:
                    continue

                if context:
                    prompt = f"{instruction}\n\nContext: {context}"
                else:
                    prompt = instruction

                resp = resp.strip()
                if len(resp) > 20 and len(resp) < 2000:
                    pairs.append((prompt, resp))
            except json.JSONDecodeError:
                continue

        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(pairs, f)

        print(f"Downloaded {len(pairs)} Dolly instruction pairs")
        return pairs

    except Exception as e:
        print(f"Error downloading Dolly data: {e}")
        return []


def generate_coding_instructions() -> List[Tuple[str, str]]:
    """Generate comprehensive coding instruction pairs."""
    pairs = [
        # Python basics
        ("Write Python to add two numbers", "```python\ndef add(a, b):\n    return a + b\n```"),
        ("Write Python code to add two numbers", "```python\ndef add(a, b):\n    return a + b\n```"),
        ("Python function to add numbers", "```python\ndef add(a, b):\n    return a + b\n```"),
        ("Write Python to subtract two numbers", "```python\ndef subtract(a, b):\n    return a - b\n```"),
        ("Write Python to multiply two numbers", "```python\ndef multiply(a, b):\n    return a * b\n```"),
        ("Write Python to divide two numbers", "```python\ndef divide(a, b):\n    if b == 0:\n        return None\n    return a / b\n```"),
        ("Write Python to reverse a string", "```python\ndef reverse_string(s):\n    return s[::-1]\n```"),
        ("Python function to reverse string", "```python\ndef reverse_string(s):\n    return s[::-1]\n```"),
        ("Write Python to check if number is even", "```python\ndef is_even(n):\n    return n % 2 == 0\n```"),
        ("Write Python to check if number is odd", "```python\ndef is_odd(n):\n    return n % 2 != 0\n```"),
        ("Write Python to check palindrome", "```python\ndef is_palindrome(s):\n    return s == s[::-1]\n```"),
        ("Write Python factorial function", "```python\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n```"),
        ("Write Python Fibonacci function", "```python\ndef fibonacci(n):\n    if n <= 0:\n        return []\n    if n == 1:\n        return [0]\n    fib = [0, 1]\n    for i in range(2, n):\n        fib.append(fib[-1] + fib[-2])\n    return fib\n```"),
        ("Write Python to check if prime", "```python\ndef is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n```"),
        ("Write Python to find max in list", "```python\ndef find_max(numbers):\n    if not numbers:\n        return None\n    return max(numbers)\n```"),
        ("Write Python to find min in list", "```python\ndef find_min(numbers):\n    if not numbers:\n        return None\n    return min(numbers)\n```"),
        ("Write Python to sum a list", "```python\ndef sum_list(numbers):\n    return sum(numbers)\n```"),
        ("Write Python to calculate average", "```python\ndef average(numbers):\n    if not numbers:\n        return 0\n    return sum(numbers) / len(numbers)\n```"),
        ("Write Python to sort a list", "```python\ndef sort_list(items):\n    return sorted(items)\n```"),
        ("Write Python to read a file", "```python\ndef read_file(path):\n    with open(path, 'r') as f:\n        return f.read()\n```"),
        ("Write Python to write to a file", "```python\ndef write_file(path, content):\n    with open(path, 'w') as f:\n        f.write(content)\n```"),
        ("Write Python binary search", "```python\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n```"),
        ("Write Python FizzBuzz", "```python\ndef fizzbuzz(n):\n    for i in range(1, n + 1):\n        if i % 15 == 0:\n            print('FizzBuzz')\n        elif i % 3 == 0:\n            print('Fizz')\n        elif i % 5 == 0:\n            print('Buzz')\n        else:\n            print(i)\n```"),
        ("Write Python hello world", "```python\nprint('Hello, World!')\n```"),
        # JavaScript
        ("Write JavaScript to add two numbers", "```javascript\nfunction add(a, b) {\n    return a + b;\n}\n```"),
        ("Write JavaScript to reverse string", "```javascript\nfunction reverseString(s) {\n    return s.split('').reverse().join('');\n}\n```"),
        # SQL
        ("Write SQL to select all users", "```sql\nSELECT * FROM users;\n```"),
        ("Write SQL to count rows", "```sql\nSELECT COUNT(*) FROM table_name;\n```"),
    ]
    return pairs


def generate_math_qa() -> List[Tuple[str, str]]:
    """Generate math Q&A pairs with explicit answers."""
    pairs = []

    # Basic arithmetic - many variations
    for a in range(1, 13):
        for b in range(1, 13):
            pairs.append((f"What is {a} + {b}?", f"{a} + {b} = {a + b}"))
            pairs.append((f"What is {a} plus {b}?", f"{a} plus {b} equals {a + b}."))
            if a >= b:
                pairs.append((f"What is {a} - {b}?", f"{a} - {b} = {a - b}"))
            pairs.append((f"What is {a} * {b}?", f"{a} * {b} = {a * b}"))
            pairs.append((f"What is {a} times {b}?", f"{a} times {b} equals {a * b}."))

    # Square roots
    squares = {1: 1, 4: 2, 9: 3, 16: 4, 25: 5, 36: 6, 49: 7, 64: 8, 81: 9, 100: 10}
    for num, root in squares.items():
        pairs.append((f"What is the square root of {num}?", f"The square root of {num} is {root}."))
        pairs.append((f"sqrt({num})?", f"sqrt({num}) = {root}"))

    return pairs


def generate_general_qa() -> List[Tuple[str, str]]:
    """Generate general knowledge Q&A pairs."""
    pairs = [
        ("Hi", "Hello! How can I help you today?"),
        ("Hello", "Hello! I'm here to help. What do you need?"),
        ("Hey", "Hey! What can I do for you?"),
        ("Thanks", "You're welcome!"),
        ("Thank you", "You're welcome! Happy to help."),
        ("Bye", "Goodbye! Have a great day!"),
        ("Who are you?", "I'm a coding assistant. I can help you write code and answer questions."),
        ("What are you?", "I'm an AI assistant that helps with coding and answering questions."),
        ("What is Python?", "Python is a popular programming language known for its simple, readable syntax."),
        ("What is JavaScript?", "JavaScript is a programming language used for web development."),
        ("What is a function?", "A function is a reusable block of code that performs a specific task."),
        ("What is a variable?", "A variable is a named container that stores data in a program."),
    ]
    return pairs


def generate_instruction_responses() -> List[Tuple[str, str]]:
    """Generate instruction-response pairs."""
    return [
        ("Explain how a computer works", "A computer takes input, processes it using the CPU, and produces output."),
        ("How do I learn programming?", "Start with Python, practice daily, build projects, and read others' code."),
        ("How do I debug code?", "Read error messages, add print statements, check logic, change one thing at a time."),
    ]


def get_all_instruction_data(allow_download: bool = True) -> List[Tuple[str, str]]:
    """Get all instruction-following training data."""
    all_data = []

    dolly = download_dolly_data(allow_download=allow_download)
    all_data.extend(dolly)
    print(f"External datasets: {len(dolly)} Dolly pairs")

    coding = generate_coding_instructions()
    for _ in range(20):
        all_data.extend(coding)
    print(f"Coding instructions: {len(coding)} * 20 = {len(coding) * 20} pairs")

    math_qa = generate_math_qa()
    for _ in range(15):
        all_data.extend(math_qa)
    print(f"Math Q&A: {len(math_qa)} * 15 = {len(math_qa) * 15} pairs")

    general = generate_general_qa()
    for _ in range(20):
        all_data.extend(general)
    print(f"General Q&A: {len(general)} * 20 = {len(general) * 20} pairs")

    instructions = generate_instruction_responses()
    for _ in range(15):
        all_data.extend(instructions)
    print(f"Instructions: {len(instructions)} * 15 = {len(instructions) * 15} pairs")

    print(f"Total instruction data pairs: {len(all_data)}")
    return all_data


if __name__ == "__main__":
    data = get_all_instruction_data()
    print(f"\nGenerated {len(data)} total training pairs")
