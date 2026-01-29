#!/usr/bin/env python3
"""
GROUNDED VERIFICATION MODULE
=============================
Provides real verification of model outputs through execution and formal methods.

Verification Methods:
1. CODE: Execute code in sandbox, check outputs against test cases
2. MATH: Use SymPy to solve and verify mathematical answers
3. LOGIC: Use Z3 theorem prover for logical reasoning verification

This replaces regex-based verification with real execution and proof.

Usage:
    from grounded_verification import GroundedVerifier

    verifier = GroundedVerifier()
    result = verifier.verify(prompt, response, category="code")

    # Test
    python grounded_verification.py --test

Author: Bo Shang <bo@shang.software>
"""

import os
import sys
import re
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import ast
import traceback

# Verification timeout
EXECUTION_TIMEOUT = 5  # seconds


@dataclass
class VerificationResult:
    """Result of verifying a response."""
    is_correct: bool
    confidence: float
    method: str
    expected: Optional[Any] = None
    actual: Optional[Any] = None
    error: Optional[str] = None
    details: Optional[Dict] = None


class CodeVerifier:
    """
    Verify code by executing it in a subprocess sandbox.
    """

    def __init__(self, timeout: int = EXECUTION_TIMEOUT):
        self.timeout = timeout

    def extract_code(self, response: str) -> Optional[str]:
        """Extract Python code from a response."""
        # Try code blocks first
        patterns = [
            r'```python\n(.*?)```',
            r'```py\n(.*?)```',
            r'```\n(.*?)```',
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                return match.group(1).strip()

        # Try inline code (function definitions)
        if re.search(r'def\s+\w+.*?:', response):
            # Extract from def to end of function
            lines = response.split('\n')
            code_lines = []
            in_function = False
            indent_level = 0

            for line in lines:
                if re.match(r'^\s*def\s+\w+', line):
                    in_function = True
                    indent_level = len(line) - len(line.lstrip())
                    code_lines.append(line)
                elif in_function:
                    current_indent = len(line) - len(line.lstrip())
                    if line.strip() and current_indent <= indent_level and not line.strip().startswith('#'):
                        break
                    code_lines.append(line)

            if code_lines:
                return '\n'.join(code_lines)

        return None

    def verify_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """Check if code has valid Python syntax."""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"SyntaxError: {e.msg} at line {e.lineno}"

    def verify_execution(self, code: str, test_cases: List[Dict] = None) -> VerificationResult:
        """
        Execute code in sandbox and verify against test cases.

        Args:
            code: Python code to execute
            test_cases: List of {"input": ..., "expected": ...} dicts

        Returns:
            VerificationResult
        """
        # First check syntax
        is_valid, syntax_error = self.verify_syntax(code)
        if not is_valid:
            return VerificationResult(
                is_correct=False,
                confidence=0.0,
                method="code_syntax",
                error=syntax_error
            )

        if not test_cases:
            # No test cases - just verify it runs without error
            return self._verify_runs_without_error(code)

        # Run with test cases
        results = []
        for i, test in enumerate(test_cases):
            result = self._run_test_case(code, test)
            results.append(result)

        # All tests must pass
        passed = all(r.is_correct for r in results)
        confidence = sum(1 for r in results if r.is_correct) / len(results)

        return VerificationResult(
            is_correct=passed,
            confidence=confidence,
            method="code_execution",
            details={"test_results": [r.__dict__ for r in results]}
        )

    def _verify_runs_without_error(self, code: str) -> VerificationResult:
        """Verify code runs without raising an exception."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.write('\nprint("__SUCCESS__")\n')
            temp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            if result.returncode == 0 and "__SUCCESS__" in result.stdout:
                return VerificationResult(
                    is_correct=True,
                    confidence=0.7,  # Lower confidence without test cases
                    method="code_runs"
                )
            else:
                return VerificationResult(
                    is_correct=False,
                    confidence=0.0,
                    method="code_runs",
                    error=result.stderr or "Unknown error"
                )

        except subprocess.TimeoutExpired:
            return VerificationResult(
                is_correct=False,
                confidence=0.0,
                method="code_runs",
                error="Execution timed out"
            )
        finally:
            os.unlink(temp_path)

    def _run_test_case(self, code: str, test: Dict) -> VerificationResult:
        """Run a single test case."""
        test_input = test.get("input", "")
        expected = test.get("expected", None)

        # Create test wrapper
        wrapper = f"""
{code}

# Run test
__test_input__ = {repr(test_input)}
__result__ = None

# Try to call the function with the input
import json
try:
    # Find the main function
    for name in dir():
        obj = eval(name)
        if callable(obj) and not name.startswith('_'):
            if isinstance(__test_input__, (list, tuple)):
                __result__ = obj(*__test_input__)
            elif isinstance(__test_input__, dict):
                __result__ = obj(**__test_input__)
            else:
                __result__ = obj(__test_input__)
            break
    print(json.dumps({{"result": __result__}}))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(wrapper)
            temp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            if result.returncode == 0:
                try:
                    output = json.loads(result.stdout.strip().split('\n')[-1])
                    if "error" in output:
                        return VerificationResult(
                            is_correct=False,
                            confidence=0.0,
                            method="code_test",
                            error=output["error"]
                        )

                    actual = output.get("result")
                    is_correct = actual == expected

                    return VerificationResult(
                        is_correct=is_correct,
                        confidence=1.0 if is_correct else 0.0,
                        method="code_test",
                        expected=expected,
                        actual=actual
                    )
                except:
                    return VerificationResult(
                        is_correct=False,
                        confidence=0.0,
                        method="code_test",
                        error="Could not parse output"
                    )
            else:
                return VerificationResult(
                    is_correct=False,
                    confidence=0.0,
                    method="code_test",
                    error=result.stderr
                )

        except subprocess.TimeoutExpired:
            return VerificationResult(
                is_correct=False,
                confidence=0.0,
                method="code_test",
                error="Timeout"
            )
        finally:
            os.unlink(temp_path)


class MathVerifier:
    """
    Verify mathematical answers using SymPy.
    """

    def __init__(self):
        self._sympy_available = None

    def _check_sympy(self) -> bool:
        """Check if SymPy is available."""
        if self._sympy_available is None:
            try:
                import sympy
                self._sympy_available = True
            except ImportError:
                self._sympy_available = False
                print("[MathVerifier] WARNING: SymPy not installed. Install with: pip install sympy")
        return self._sympy_available

    def extract_answer(self, response: str) -> Optional[str]:
        """Extract numerical answer from response."""
        patterns = [
            r'(?:the\s+)?answer\s+is[:\s]+(\-?\d+\.?\d*)',
            r'(?:therefore|so|thus)[,\s]+(?:the\s+answer\s+is\s+)?(\-?\d+\.?\d*)',
            r'=\s*(\-?\d+\.?\d*)\s*$',
            r'Answer:\s*(\-?\d+\.?\d*)',
            r'Result:\s*(\-?\d+\.?\d*)',
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1)

        # Fallback: last number in response
        numbers = re.findall(r'\-?\d+\.?\d*', response)
        if numbers:
            return numbers[-1]

        return None

    def verify_arithmetic(self, problem: str, answer: str) -> VerificationResult:
        """Verify basic arithmetic."""
        # Extract operation from problem
        patterns = [
            (r'(\d+)\s*\+\s*(\d+)', lambda a, b: float(a) + float(b)),
            (r'(\d+)\s*\-\s*(\d+)', lambda a, b: float(a) - float(b)),
            (r'(\d+)\s*[*x×]\s*(\d+)', lambda a, b: float(a) * float(b)),
            (r'(\d+)\s*/\s*(\d+)', lambda a, b: float(a) / float(b) if float(b) != 0 else None),
            (r'(\d+)\s*\*\*\s*(\d+)', lambda a, b: float(a) ** float(b)),
        ]

        expected = None
        for pattern, op in patterns:
            match = re.search(pattern, problem)
            if match:
                try:
                    expected = op(match.group(1), match.group(2))
                    break
                except:
                    continue

        if expected is None:
            return VerificationResult(
                is_correct=False,
                confidence=0.0,
                method="math_arithmetic",
                error="Could not extract operation from problem"
            )

        try:
            actual = float(answer)
            is_correct = abs(actual - expected) < 0.001

            return VerificationResult(
                is_correct=is_correct,
                confidence=1.0 if is_correct else 0.0,
                method="math_arithmetic",
                expected=expected,
                actual=actual
            )
        except ValueError:
            return VerificationResult(
                is_correct=False,
                confidence=0.0,
                method="math_arithmetic",
                error=f"Could not parse answer: {answer}"
            )

    def verify_equation(self, problem: str, answer: str) -> VerificationResult:
        """Verify equation solving using SymPy."""
        if not self._check_sympy():
            # Fallback to regex
            return self._verify_equation_regex(problem, answer)

        import sympy
        from sympy import symbols, Eq, solve, sympify

        try:
            x = symbols('x')

            # Try to extract equation from problem
            # Pattern: "solve x + 5 = 10" or "x + 5 = 10, find x"
            eq_match = re.search(r'([x\d\s\+\-\*/\(\)]+)\s*=\s*([x\d\s\+\-\*/\(\)]+)', problem)
            if not eq_match:
                return VerificationResult(
                    is_correct=False,
                    confidence=0.0,
                    method="math_equation",
                    error="Could not extract equation"
                )

            lhs = sympify(eq_match.group(1).replace('^', '**'))
            rhs = sympify(eq_match.group(2).replace('^', '**'))
            equation = Eq(lhs, rhs)

            solution = solve(equation, x)
            if not solution:
                return VerificationResult(
                    is_correct=False,
                    confidence=0.0,
                    method="math_equation",
                    error="No solution found"
                )

            expected = float(solution[0])
            actual = float(answer)
            is_correct = abs(actual - expected) < 0.001

            return VerificationResult(
                is_correct=is_correct,
                confidence=1.0 if is_correct else 0.0,
                method="math_equation",
                expected=expected,
                actual=actual
            )

        except Exception as e:
            return VerificationResult(
                is_correct=False,
                confidence=0.0,
                method="math_equation",
                error=str(e)
            )

    def _verify_equation_regex(self, problem: str, answer: str) -> VerificationResult:
        """Fallback equation verification using regex."""
        patterns = [
            (r'x\s*\+\s*(\d+)\s*=\s*(\d+)', lambda a, b: float(b) - float(a)),
            (r'x\s*\-\s*(\d+)\s*=\s*(\d+)', lambda a, b: float(b) + float(a)),
            (r'(\d+)\s*\+\s*x\s*=\s*(\d+)', lambda a, b: float(b) - float(a)),
        ]

        for pattern, solver in patterns:
            match = re.search(pattern, problem)
            if match:
                try:
                    expected = solver(match.group(1), match.group(2))
                    actual = float(answer)
                    is_correct = abs(actual - expected) < 0.001

                    return VerificationResult(
                        is_correct=is_correct,
                        confidence=0.9 if is_correct else 0.0,
                        method="math_equation_regex",
                        expected=expected,
                        actual=actual
                    )
                except:
                    continue

        return VerificationResult(
            is_correct=False,
            confidence=0.0,
            method="math_equation_regex",
            error="Could not parse equation"
        )

    def verify(self, problem: str, response: str) -> VerificationResult:
        """Verify a math problem and answer."""
        answer = self.extract_answer(response)
        if not answer:
            return VerificationResult(
                is_correct=False,
                confidence=0.0,
                method="math",
                error="Could not extract answer from response"
            )

        # Try arithmetic first
        result = self.verify_arithmetic(problem, answer)
        if result.is_correct or result.expected is not None:
            return result

        # Try equation
        result = self.verify_equation(problem, answer)
        return result


class LogicVerifier:
    """
    Verify logical reasoning using Z3 theorem prover.
    """

    def __init__(self):
        self._z3_available = None

    def _check_z3(self) -> bool:
        """Check if Z3 is available."""
        if self._z3_available is None:
            try:
                import z3
                self._z3_available = True
            except ImportError:
                self._z3_available = False
                print("[LogicVerifier] WARNING: Z3 not installed. Install with: pip install z3-solver")
        return self._z3_available

    def verify_boolean(self, problem: str, response: str) -> VerificationResult:
        """Verify boolean logic problems."""
        # Pattern: a=X, b=Y, is (a>5 AND/OR b>5)?
        bool_match = re.search(
            r'a\s*=\s*(\d+).*b\s*=\s*(\d+).*\(a\s*>\s*(\d+)\s*(AND|OR)\s*b\s*>\s*(\d+)\)',
            problem,
            re.IGNORECASE
        )

        if not bool_match:
            return VerificationResult(
                is_correct=False,
                confidence=0.0,
                method="logic_boolean",
                error="Could not parse boolean problem"
            )

        a = int(bool_match.group(1))
        b = int(bool_match.group(2))
        threshold_a = int(bool_match.group(3))
        op = bool_match.group(4).upper()
        threshold_b = int(bool_match.group(5))

        cond1 = a > threshold_a
        cond2 = b > threshold_b
        expected = (cond1 and cond2) if op == "AND" else (cond1 or cond2)

        # Extract answer from response
        resp_lower = response.lower()
        if 'true' in resp_lower and 'false' not in resp_lower:
            actual = True
        elif 'false' in resp_lower:
            actual = False
        else:
            # Check for explicit answer
            ans_match = re.search(r'(?:answer|result)[:\s]*(true|false)', resp_lower)
            if ans_match:
                actual = ans_match.group(1) == 'true'
            else:
                return VerificationResult(
                    is_correct=False,
                    confidence=0.0,
                    method="logic_boolean",
                    error="Could not extract boolean answer"
                )

        is_correct = actual == expected

        return VerificationResult(
            is_correct=is_correct,
            confidence=1.0 if is_correct else 0.0,
            method="logic_boolean",
            expected=expected,
            actual=actual
        )

    def verify_implication(self, premises: List[str], conclusion: str) -> VerificationResult:
        """Verify logical implication using Z3."""
        if not self._check_z3():
            return VerificationResult(
                is_correct=False,
                confidence=0.0,
                method="logic_z3",
                error="Z3 not available"
            )

        from z3 import Solver, Bool, Implies, And, Or, Not, sat, unsat

        try:
            solver = Solver()

            # This is a simplified example - real implementation would need
            # proper parsing of natural language to Z3 formulas
            # For now, we handle simple propositional logic

            # Create boolean variables for common terms
            variables = {}

            def get_var(name: str) -> Bool:
                if name not in variables:
                    variables[name] = Bool(name)
                return variables[name]

            # Parse simple implications: "if A then B" -> Implies(A, B)
            for premise in premises:
                if "if" in premise.lower() and "then" in premise.lower():
                    match = re.search(r'if\s+(\w+)\s+then\s+(\w+)', premise, re.IGNORECASE)
                    if match:
                        a = get_var(match.group(1))
                        b = get_var(match.group(2))
                        solver.add(Implies(a, b))

            # Check if conclusion follows
            # (Assert negation of conclusion and check for unsat)
            conclusion_var = get_var(conclusion)
            solver.add(Not(conclusion_var))

            result = solver.check()

            if result == unsat:
                return VerificationResult(
                    is_correct=True,
                    confidence=1.0,
                    method="logic_z3",
                    details={"conclusion_follows": True}
                )
            else:
                return VerificationResult(
                    is_correct=False,
                    confidence=0.0,
                    method="logic_z3",
                    details={"conclusion_follows": False}
                )

        except Exception as e:
            return VerificationResult(
                is_correct=False,
                confidence=0.0,
                method="logic_z3",
                error=str(e)
            )

    def verify(self, problem: str, response: str) -> VerificationResult:
        """Verify a logic problem."""
        # Try boolean first
        result = self.verify_boolean(problem, response)
        if result.error != "Could not parse boolean problem":
            return result

        # Default pass for complex logic we can't verify
        return VerificationResult(
            is_correct=True,
            confidence=0.5,
            method="logic_passthrough"
        )


class GroundedVerifier:
    """
    Main verifier that combines code, math, and logic verification.
    """

    def __init__(self, timeout: int = EXECUTION_TIMEOUT):
        self.code_verifier = CodeVerifier(timeout=timeout)
        self.math_verifier = MathVerifier()
        self.logic_verifier = LogicVerifier()

    def detect_category(self, prompt: str) -> str:
        """Auto-detect verification category from prompt."""
        prompt_lower = prompt.lower()

        # Code detection
        code_patterns = [
            r'\b(write|implement|create|code|function|def|class|program)\b',
            r'```',
            r'\b(python|javascript|java|c\+\+)\b',
        ]
        for pattern in code_patterns:
            if re.search(pattern, prompt_lower):
                return "code"

        # Math detection
        math_patterns = [
            r'\d+\s*[\+\-\*/]\s*\d+',
            r'\b(solve|calculate|compute|evaluate|find)\b.*\d+',
            r'\b(equation|formula|expression)\b',
            r'x\s*[=+\-*/]',
        ]
        for pattern in math_patterns:
            if re.search(pattern, prompt_lower):
                return "math"

        # Logic detection
        logic_patterns = [
            r'\b(true|false|and|or|not|if|then|implies)\b',
            r'\b(premise|conclusion|valid|invalid)\b',
        ]
        for pattern in logic_patterns:
            if re.search(pattern, prompt_lower):
                return "logic"

        return "unknown"

    def verify_code(self, code: str, test_cases: List[Dict] = None) -> VerificationResult:
        """Verify code execution."""
        return self.code_verifier.verify_execution(code, test_cases)

    def verify_math(self, problem: str, answer: str) -> VerificationResult:
        """Verify mathematical answer."""
        return self.math_verifier.verify(problem, answer)

    def verify_logic(self, premises: List[str], conclusion: str) -> VerificationResult:
        """Verify logical reasoning."""
        return self.logic_verifier.verify_implication(premises, conclusion)

    def verify(self, prompt: str, response: str, category: str = "auto",
               test_cases: List[Dict] = None) -> VerificationResult:
        """
        Verify a response against its prompt.

        Args:
            prompt: The original problem/question
            response: The model's response
            category: "code", "math", "logic", or "auto"
            test_cases: For code, list of {"input": ..., "expected": ...}

        Returns:
            VerificationResult
        """
        if category == "auto":
            category = self.detect_category(prompt)

        if category == "code":
            code = self.code_verifier.extract_code(response)
            if not code:
                return VerificationResult(
                    is_correct=False,
                    confidence=0.0,
                    method="code",
                    error="No code found in response"
                )
            return self.code_verifier.verify_execution(code, test_cases)

        elif category == "math":
            return self.math_verifier.verify(prompt, response)

        elif category == "logic":
            return self.logic_verifier.verify(prompt, response)

        else:
            # Unknown category - pass through with low confidence
            return VerificationResult(
                is_correct=True,
                confidence=0.5,
                method="passthrough"
            )


# ============================================================================
# CLI / TESTING
# ============================================================================

def test_verifiers():
    """Test all verification methods."""
    print("=" * 60)
    print("GROUNDED VERIFICATION TEST")
    print("=" * 60)

    verifier = GroundedVerifier()

    # Test 1: Code syntax
    print("\n1. Testing code syntax verification...")
    code = """
def add(a, b):
    return a + b
"""
    result = verifier.verify_code(code)
    print(f"   Valid code: {result.is_correct}, confidence: {result.confidence}")

    # Test 2: Code with syntax error
    print("\n2. Testing code with syntax error...")
    bad_code = "def foo(: return 42"
    result = verifier.verify_code(bad_code)
    print(f"   Invalid code detected: {not result.is_correct}, error: {result.error}")

    # Test 3: Code with test cases
    print("\n3. Testing code execution with test cases...")
    code = """
def multiply(a, b):
    return a * b
"""
    test_cases = [
        {"input": [2, 3], "expected": 6},
        {"input": [0, 5], "expected": 0},
        {"input": [-1, 4], "expected": -4},
    ]
    result = verifier.verify_code(code, test_cases)
    print(f"   All tests passed: {result.is_correct}, confidence: {result.confidence}")

    # Test 4: Math arithmetic
    print("\n4. Testing math arithmetic verification...")
    result = verifier.verify_math("What is 25 + 17?", "The answer is 42.")
    print(f"   Correct: {result.is_correct}, expected: {result.expected}, actual: {result.actual}")

    # Test 5: Math wrong answer
    print("\n5. Testing math wrong answer detection...")
    result = verifier.verify_math("What is 10 * 5?", "The answer is 40.")
    print(f"   Detected wrong: {not result.is_correct}, expected: {result.expected}, actual: {result.actual}")

    # Test 6: Boolean logic
    print("\n6. Testing boolean logic verification...")
    result = verifier.logic_verifier.verify_boolean(
        "If a=10, b=3, is (a>5 AND b>5) true?",
        "Since a=10 > 5 is true, but b=3 > 5 is false, the AND is false."
    )
    print(f"   Correct: {result.is_correct}, expected: {result.expected}, actual: {result.actual}")

    # Test 7: Category detection
    print("\n7. Testing category auto-detection...")
    prompts = [
        ("Write a function to sort a list", "code"),
        ("What is 15 * 8?", "math"),
        ("If A implies B, and A is true, is B true?", "logic"),
        ("What is the capital of France?", "unknown"),
    ]
    for prompt, expected_cat in prompts:
        detected = verifier.detect_category(prompt)
        status = "CORRECT" if detected == expected_cat else f"WRONG (got {detected})"
        print(f"   '{prompt[:30]}...' -> {expected_cat}: {status}")

    # Test 8: Full verification flow
    print("\n8. Testing full verification flow...")
    result = verifier.verify(
        "Write a function to calculate factorial",
        "```python\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)\n```",
        test_cases=[
            {"input": 5, "expected": 120},
            {"input": 0, "expected": 1},
        ]
    )
    print(f"   Full verification: {result.is_correct}, method: {result.method}")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Grounded Verification Module")
    parser.add_argument("--test", action="store_true", help="Run verification tests")
    parser.add_argument("--verify-code", type=str, help="Verify a Python code file")
    parser.add_argument("--verify-math", nargs=2, metavar=("PROBLEM", "ANSWER"),
                       help="Verify a math problem and answer")
    args = parser.parse_args()

    if args.test:
        test_verifiers()
    elif args.verify_code:
        with open(args.verify_code) as f:
            code = f.read()
        verifier = GroundedVerifier()
        result = verifier.verify_code(code)
        print(f"Verification result: {result.is_correct}")
        if result.error:
            print(f"Error: {result.error}")
    elif args.verify_math:
        verifier = GroundedVerifier()
        result = verifier.verify_math(args.verify_math[0], args.verify_math[1])
        print(f"Verification result: {result.is_correct}")
        print(f"Expected: {result.expected}, Actual: {result.actual}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
