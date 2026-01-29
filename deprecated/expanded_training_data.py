"""
Verified training data - FACTS ONLY, no opinions.
Includes: coding knowledge, assistant identity.
Substantive instruction data comes from web-downloaded datasets.
"""

from typing import List, Tuple

# =============================================================================
# ASSISTANT IDENTITY AND BASIC BEHAVIOR
# =============================================================================
ASSISTANT_BEHAVIOR = [
    ("Who are you?", "I'm Erosolar, a coding assistant."),
    ("What are you?", "I'm an AI assistant that helps with coding and questions."),
    ("What can you do?", "I can help with coding, answer questions, and explain concepts."),
    ("What's your name?", "I'm Erosolar."),
    ("Hi", "Hello! How can I help you?"),
    ("Hello", "Hi! What can I help you with?"),
    ("Hey", "Hey! What do you need?"),
    ("Thanks", "You're welcome!"),
    ("Thank you", "Happy to help!"),
    ("Bye", "Goodbye!"),
    ("Goodbye", "Take care!"),
    ("Help", "What do you need help with?"),
    ("I need help", "What can I help with?"),
]

# =============================================================================
# CODING KNOWLEDGE - Factual programming patterns
# =============================================================================
CODING_KNOWLEDGE = [
    # Python basics
    ("How do I print in Python?", "Use the print() function:\n```python\nprint('Hello, World!')\n```"),
    ("How do I create a list in Python?", "Use square brackets:\n```python\nmy_list = [1, 2, 3, 4, 5]\n```"),
    ("How do I create a function in Python?", "Use def:\n```python\ndef greet(name):\n    return f'Hello, {name}!'\n```"),
    ("How do I use a for loop in Python?", "```python\nfor i in range(5):\n    print(i)\n```"),
    ("How do I use if statements in Python?", "```python\nif x > 0:\n    print('Positive')\nelif x < 0:\n    print('Negative')\nelse:\n    print('Zero')\n```"),
    ("How do I read a file in Python?", "```python\nwith open('file.txt', 'r') as f:\n    content = f.read()\n```"),
    ("How do I write to a file in Python?", "```python\nwith open('file.txt', 'w') as f:\n    f.write('Hello!')\n```"),
    ("What is a dictionary in Python?", "A dictionary stores key-value pairs:\n```python\nperson = {'name': 'John', 'age': 30}\n```"),
    ("How do I handle errors in Python?", "```python\ntry:\n    result = 10 / 0\nexcept ZeroDivisionError:\n    print('Cannot divide by zero')\n```"),
    ("What is a class in Python?", "```python\nclass Dog:\n    def __init__(self, name):\n        self.name = name\n    def bark(self):\n        return 'Woof!'\n```"),

    # JavaScript basics
    ("How do I print in JavaScript?", "Use console.log():\n```javascript\nconsole.log('Hello!');\n```"),
    ("How do I create a function in JavaScript?", "```javascript\nfunction greet(name) {\n    return `Hello, ${name}!`;\n}\n// Or arrow function:\nconst greet = (name) => `Hello, ${name}!`;\n```"),
    ("How do I create an array in JavaScript?", "```javascript\nconst arr = [1, 2, 3, 4, 5];\n```"),

    # General concepts
    ("What is an API?", "API (Application Programming Interface) is a set of rules for different software to communicate with each other."),
    ("What is REST?", "REST is an architectural style for APIs using HTTP methods (GET, POST, PUT, DELETE)."),
    ("What is JSON?", "JSON (JavaScript Object Notation) is a data format: {\"name\": \"John\", \"age\": 30}"),
    ("What is Git?", "Git is a version control system that tracks code changes and enables collaboration."),
    ("What is GitHub?", "GitHub is a platform for hosting Git repositories and collaborating on code."),
    ("What is a database?", "A database is an organized collection of data. Examples: PostgreSQL, MySQL, MongoDB."),
    ("What is SQL?", "SQL (Structured Query Language) is a language for managing relational databases."),

    # Data structures
    ("What is an array?", "An array stores elements in contiguous memory, accessed by index. O(1) access time."),
    ("What is a linked list?", "A linked list has nodes that point to the next element. O(n) access, O(1) insertion."),
    ("What is a stack?", "A stack is LIFO (Last-In-First-Out). Last element added is first removed."),
    ("What is a queue?", "A queue is FIFO (First-In-First-Out). First element added is first removed."),
    ("What is a hash table?", "A hash table stores key-value pairs with O(1) average lookup using a hash function."),

    # Algorithms
    ("What is Big O notation?", "Big O describes algorithm performance: O(1) constant, O(n) linear, O(n²) quadratic."),
    ("What is binary search?", "Binary search finds elements in sorted arrays by halving the search space. O(log n)."),
    ("What is recursion?", "Recursion is when a function calls itself to solve smaller subproblems."),
]


def get_expanded_training_data() -> List[Tuple[str, str]]:
    """Return verified training data - identity and coding knowledge."""
    all_data = []

    for _ in range(50):
        all_data.extend(ASSISTANT_BEHAVIOR)

    for _ in range(30):
        all_data.extend(CODING_KNOWLEDGE)

    return all_data


if __name__ == "__main__":
    data = get_expanded_training_data()
    print(f"Total proprietary training pairs: {len(data)}")
    print(f"Assistant behavior: {len(ASSISTANT_BEHAVIOR)}")
    print(f"Coding knowledge: {len(CODING_KNOWLEDGE)}")
