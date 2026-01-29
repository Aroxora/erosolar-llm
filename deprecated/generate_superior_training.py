#!/usr/bin/env python3
"""
Superior Training Data Generator
Generates complex prompts across ALL domains, then uses GPT-5.1-codex-mini to create
training pairs that will produce a model exceeding GPT-5.1-codex-mini capabilities.

The key insight: We use GPT-5.1-codex-mini to generate training data that teaches
reasoning patterns, not just answers. This creates compounding improvements.
"""

import os
import json
import random
import hashlib
import asyncio
import aiohttp
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import argparse

# Domain definitions with complexity levels and prompt templates
DOMAINS = {
    "advanced_mathematics": {
        "weight": 1.5,
        "complexity_range": (7, 10),
        "topics": [
            "algebraic topology", "differential geometry", "number theory",
            "functional analysis", "category theory", "combinatorics",
            "probability theory", "statistical mechanics", "cryptography",
            "optimization theory", "dynamical systems", "measure theory",
            "representation theory", "algebraic geometry", "harmonic analysis"
        ],
        "prompt_templates": [
            "Prove that {concept} implies {related_concept} under {conditions}.",
            "Derive the {formula_type} for {mathematical_object} using {method}.",
            "Explain why {theorem} fails when {assumption} is relaxed.",
            "Construct a counterexample to {false_conjecture} in {context}.",
            "Calculate {quantity} for {structure} and interpret the result.",
            "Show the connection between {concept1} and {concept2} through {bridge}."
        ]
    },
    "theoretical_physics": {
        "weight": 1.5,
        "complexity_range": (7, 10),
        "topics": [
            "quantum field theory", "general relativity", "string theory",
            "condensed matter physics", "particle physics", "cosmology",
            "statistical mechanics", "quantum information", "supersymmetry",
            "black hole thermodynamics", "gauge theories", "topological phases"
        ],
        "prompt_templates": [
            "Derive {equation} from first principles using {formalism}.",
            "Explain the physical meaning of {mathematical_object} in {theory}.",
            "Calculate {observable} for {system} at {conditions}.",
            "Resolve the apparent paradox of {paradox} in {framework}.",
            "Compare {theory1} and {theory2} approaches to {phenomenon}."
        ]
    },
    "chemistry_biochemistry": {
        "weight": 1.3,
        "complexity_range": (6, 10),
        "topics": [
            "organic synthesis", "quantum chemistry", "biochemical pathways",
            "protein folding", "drug design", "catalysis mechanisms",
            "spectroscopy", "thermodynamics", "kinetics", "electrochemistry",
            "polymer chemistry", "nanomaterials", "photochemistry"
        ],
        "prompt_templates": [
            "Design a synthesis route for {compound} with {constraints}.",
            "Explain the mechanism of {reaction} including {details}.",
            "Predict the properties of {molecule} based on {analysis}.",
            "Optimize {process} for {goal} considering {factors}."
        ]
    },
    "computer_science": {
        "weight": 1.5,
        "complexity_range": (6, 10),
        "topics": [
            "algorithm design", "distributed systems", "compiler theory",
            "machine learning theory", "cryptographic protocols", "complexity theory",
            "database internals", "operating systems", "network protocols",
            "formal verification", "programming language theory", "computer graphics",
            "parallel computing", "quantum computing", "systems security"
        ],
        "prompt_templates": [
            "Design an algorithm for {problem} with {complexity_constraint}.",
            "Implement {data_structure} supporting {operations} efficiently.",
            "Prove the {property} of {algorithm} using {technique}.",
            "Optimize {system} for {metric} under {constraints}.",
            "Analyze the trade-offs between {approach1} and {approach2} for {use_case}."
        ]
    },
    "medicine_biology": {
        "weight": 1.3,
        "complexity_range": (6, 10),
        "topics": [
            "molecular biology", "immunology", "neuroscience", "genetics",
            "pharmacology", "pathophysiology", "clinical reasoning",
            "epidemiology", "medical imaging", "surgical techniques",
            "oncology", "cardiology", "endocrinology", "microbiology"
        ],
        "prompt_templates": [
            "Explain the molecular mechanism of {process} in {context}.",
            "Diagnose {presentation} considering {differentials}.",
            "Design a treatment protocol for {condition} with {constraints}.",
            "Analyze the interaction between {system1} and {system2} in {state}."
        ]
    },
    "engineering": {
        "weight": 1.2,
        "complexity_range": (6, 10),
        "topics": [
            "control systems", "signal processing", "structural analysis",
            "fluid dynamics", "thermodynamics", "materials science",
            "robotics", "aerospace", "electrical systems", "VLSI design",
            "renewable energy", "manufacturing processes", "mechatronics"
        ],
        "prompt_templates": [
            "Design {system} meeting {specifications} under {constraints}.",
            "Analyze the failure mode of {component} under {conditions}.",
            "Optimize {process} for {objective} considering {trade-offs}.",
            "Model {phenomenon} using {approach} and validate with {method}."
        ]
    },
    "economics_finance": {
        "weight": 1.1,
        "complexity_range": (6, 10),
        "topics": [
            "game theory", "market microstructure", "derivatives pricing",
            "macroeconomic modeling", "behavioral economics", "econometrics",
            "portfolio theory", "risk management", "monetary policy",
            "international trade", "development economics", "financial regulation"
        ],
        "prompt_templates": [
            "Model {market_phenomenon} using {framework} with {assumptions}.",
            "Analyze the equilibrium of {game} with {player_types}.",
            "Price {derivative} under {market_conditions} using {method}.",
            "Evaluate the policy implications of {intervention} on {outcome}."
        ]
    },
    "philosophy_logic": {
        "weight": 1.0,
        "complexity_range": (7, 10),
        "topics": [
            "modal logic", "philosophy of mind", "ethics", "epistemology",
            "metaphysics", "philosophy of science", "decision theory",
            "formal semantics", "philosophy of language", "political philosophy"
        ],
        "prompt_templates": [
            "Analyze {argument} for logical validity and soundness.",
            "Compare {position1} and {position2} on {issue}.",
            "Construct a counterargument to {thesis} using {approach}.",
            "Formalize {concept} in {logical_system} and derive {conclusion}."
        ]
    },
    "law_governance": {
        "weight": 1.0,
        "complexity_range": (6, 9),
        "topics": [
            "constitutional law", "contract law", "international law",
            "intellectual property", "criminal law", "regulatory frameworks",
            "corporate governance", "human rights", "environmental law"
        ],
        "prompt_templates": [
            "Analyze {case} under {legal_framework} considering {factors}.",
            "Draft {document_type} for {situation} with {requirements}.",
            "Compare {jurisdiction1} and {jurisdiction2} approaches to {issue}.",
            "Evaluate the legal implications of {action} under {statute}."
        ]
    },
    "practical_reasoning": {
        "weight": 1.4,
        "complexity_range": (5, 9),
        "topics": [
            "decision making", "problem solving", "critical thinking",
            "project management", "negotiation", "conflict resolution",
            "resource allocation", "risk assessment", "strategic planning",
            "time management", "communication", "leadership"
        ],
        "prompt_templates": [
            "Develop a strategy for {goal} given {constraints} and {resources}.",
            "Analyze {situation} and recommend {action_type} with justification.",
            "Evaluate {options} for {decision} considering {criteria}.",
            "Design a process for {task} optimizing for {objectives}."
        ]
    },
    "multi_step_reasoning": {
        "weight": 2.0,  # Extra weight - this is key for capability
        "complexity_range": (8, 10),
        "topics": [
            "chain of thought", "decomposition", "synthesis",
            "abstraction", "analogy", "causal reasoning",
            "counterfactual analysis", "meta-cognition", "transfer learning"
        ],
        "prompt_templates": [
            "Solve {complex_problem} by breaking it into subproblems and synthesizing.",
            "Given {information}, derive {conclusion} showing all reasoning steps.",
            "Analyze {scenario} from multiple perspectives and reconcile conflicts.",
            "Transfer the solution approach from {domain1} to solve {problem} in {domain2}."
        ]
    },
    "instruction_following": {
        "weight": 1.5,
        "complexity_range": (5, 10),
        "topics": [
            "precise formatting", "conditional logic", "multi-constraint",
            "iterative refinement", "error handling", "edge cases",
            "ambiguity resolution", "priority handling"
        ],
        "prompt_templates": [
            "Complete {task} following exactly these constraints: {constraints}.",
            "Generate {output_type} that satisfies ALL of: {requirements}.",
            "If {condition1} then {action1}, else if {condition2} then {action2}, otherwise {default}.",
            "Revise {input} to meet {criteria} while preserving {properties}."
        ]
    },
    "creative_technical": {
        "weight": 1.2,
        "complexity_range": (6, 10),
        "topics": [
            "algorithm invention", "system architecture", "protocol design",
            "optimization strategies", "novel approaches", "hybrid solutions"
        ],
        "prompt_templates": [
            "Invent a new approach to {problem} that improves on {existing}.",
            "Design a hybrid system combining {approach1} and {approach2}.",
            "Propose a novel {artifact_type} for {use_case} with {properties}."
        ]
    },
    "factual_precision": {
        "weight": 1.3,
        "complexity_range": (5, 8),
        "topics": [
            "scientific facts", "historical events", "technical specifications",
            "definitions", "procedures", "regulations", "standards"
        ],
        "prompt_templates": [
            "Precisely define {term} in the context of {field}, including {aspects}.",
            "List the exact steps for {procedure} with all {details}.",
            "State the {specification} for {subject} according to {standard}."
        ]
    },
    "error_correction": {
        "weight": 1.5,
        "complexity_range": (6, 10),
        "topics": [
            "debugging", "misconception identification", "logical fallacies",
            "calculation errors", "methodological flaws", "assumption violations"
        ],
        "prompt_templates": [
            "Identify and correct all errors in: {flawed_content}.",
            "The following reasoning is flawed. Find the error: {reasoning}.",
            "This solution is incorrect. Provide the correct solution: {solution}."
        ]
    }
}

# Complexity modifiers for prompt generation
COMPLEXITY_MODIFIERS = {
    "constraints": [
        "with minimal computational resources",
        "in polynomial time",
        "using only elementary methods",
        "without external libraries",
        "maintaining backward compatibility",
        "under real-time constraints",
        "with formal verification",
        "ensuring thread safety",
        "with graceful degradation",
        "optimizing for both latency and throughput"
    ],
    "requirements": [
        "showing all intermediate steps",
        "with rigorous mathematical proof",
        "including edge case analysis",
        "with complexity analysis",
        "providing multiple solution approaches",
        "with error bounds",
        "including sensitivity analysis",
        "with uncertainty quantification"
    ],
    "contexts": [
        "in a distributed environment",
        "under adversarial conditions",
        "with noisy data",
        "in high-dimensional spaces",
        "with limited information",
        "under uncertainty",
        "with conflicting objectives",
        "in a resource-constrained setting"
    ]
}


@dataclass
class TrainingPair:
    """Represents a single training example."""
    id: str
    domain: str
    subdomain: str
    complexity: int
    prompt: str
    weak_response: Optional[str]
    strong_response: str
    reasoning_steps: List[str]
    quality_score: float
    metadata: Dict[str, Any]
    created_at: str


class PromptGenerator:
    """Generates complex, diverse prompts across all domains."""

    def __init__(self, seed: int = None):
        if seed:
            random.seed(seed)
        self.generated_hashes = set()

    def generate_prompt(self, domain: str, complexity: int) -> Tuple[str, str]:
        """Generate a unique prompt for the given domain and complexity."""
        domain_config = DOMAINS[domain]
        topic = random.choice(domain_config["topics"])
        template = random.choice(domain_config["prompt_templates"])

        # Add complexity modifiers based on level
        modifiers = []
        if complexity >= 7:
            modifiers.append(random.choice(COMPLEXITY_MODIFIERS["constraints"]))
        if complexity >= 8:
            modifiers.append(random.choice(COMPLEXITY_MODIFIERS["requirements"]))
        if complexity >= 9:
            modifiers.append(random.choice(COMPLEXITY_MODIFIERS["contexts"]))

        # Fill template with contextual content
        prompt = self._fill_template(template, topic, domain, complexity)

        if modifiers:
            prompt += " " + ", ".join(modifiers) + "."

        # Ensure uniqueness
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:12]
        if prompt_hash in self.generated_hashes:
            return self.generate_prompt(domain, complexity)  # Retry
        self.generated_hashes.add(prompt_hash)

        return prompt, topic

    def _fill_template(self, template: str, topic: str, domain: str, complexity: int) -> str:
        """Fill template placeholders with contextual content."""
        # This is a simplified version - in production, use more sophisticated generation
        filled = template

        # Replace common placeholders
        replacements = {
            "{concept}": topic,
            "{topic}": topic,
            "{problem}": f"the {topic} problem",
            "{system}": f"a {topic} system",
            "{process}": f"the {topic} process",
            "{method}": f"rigorous {domain.replace('_', ' ')} methods",
            "{approach}": f"the standard {topic} approach",
            "{conditions}": "general conditions",
            "{constraints}": "practical constraints",
            "{context}": f"the context of {domain.replace('_', ' ')}",
        }

        for placeholder, value in replacements.items():
            filled = filled.replace(placeholder, value)

        # Handle remaining placeholders generically
        import re
        remaining = re.findall(r'\{(\w+)\}', filled)
        for placeholder in remaining:
            filled = filled.replace(f"{{{placeholder}}}", f"relevant {placeholder.replace('_', ' ')}")

        return filled

    def generate_batch(self, count: int, domain_weights: Dict[str, float] = None) -> List[Dict]:
        """Generate a batch of prompts with weighted domain distribution."""
        if domain_weights is None:
            domain_weights = {d: config["weight"] for d, config in DOMAINS.items()}

        # Normalize weights
        total_weight = sum(domain_weights.values())
        domain_weights = {d: w/total_weight for d, w in domain_weights.items()}

        prompts = []
        domains = list(domain_weights.keys())
        weights = list(domain_weights.values())

        for _ in range(count):
            domain = random.choices(domains, weights=weights)[0]
            config = DOMAINS[domain]
            complexity = random.randint(*config["complexity_range"])
            prompt, topic = self.generate_prompt(domain, complexity)

            prompts.append({
                "domain": domain,
                "subdomain": topic,
                "complexity": complexity,
                "prompt": prompt
            })

        return prompts


class GPT52Enhancer:
    """Uses GPT-5.1-codex-mini to generate superior training responses."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.model = "gpt-5.1-codex-mini"  # Or latest available
        self.session = None

    async def init_session(self):
        """Initialize aiohttp session."""
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def close_session(self):
        """Close aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None

    async def generate_response(self, prompt: str, domain: str, complexity: int) -> Dict[str, Any]:
        """Generate a high-quality response using GPT-5.1-codex-mini."""

        system_prompt = self._build_system_prompt(domain, complexity)

        await self.init_session()

        try:
            async with self.session.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 4096,
                    "top_p": 0.95
                }
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API error {response.status}: {error_text}")

                data = await response.json()
                content = data["choices"][0]["message"]["content"]

                # Parse reasoning steps if present
                reasoning_steps = self._extract_reasoning_steps(content)

                return {
                    "response": content,
                    "reasoning_steps": reasoning_steps,
                    "tokens_used": data.get("usage", {})
                }

        except Exception as e:
            print(f"Error generating response: {e}")
            raise

    def _build_system_prompt(self, domain: str, complexity: int) -> str:
        """Build system prompt optimized for training data generation."""

        return f"""You are generating training data for a superior AI model. Your response will be used to train a model that should EXCEED your own capabilities.

CRITICAL REQUIREMENTS:
1. REASONING TRANSPARENCY: Show EVERY step of your reasoning explicitly. The model learning from this must understand HOW you arrived at the answer, not just the answer itself.

2. MULTI-PATH ANALYSIS: Consider multiple approaches before selecting the best one. Explain why alternatives were rejected.

3. ERROR ANTICIPATION: Identify common mistakes or misconceptions related to this problem and explicitly address them.

4. DEPTH OVER BREVITY: This is training data - comprehensiveness is more valuable than conciseness. Include:
   - Foundational concepts when relevant
   - Edge cases and boundary conditions
   - Connections to related concepts
   - Practical implications

5. STRUCTURED REASONING: Use clear markers for your reasoning:
   [UNDERSTANDING] - Your interpretation of the problem
   [APPROACH] - Your strategy for solving it
   [REASONING] - Step-by-step logical progression
   [VERIFICATION] - How you check your answer
   [ANSWER] - The final response
   [EXTENSIONS] - Related insights or generalizations

6. DOMAIN: {domain.replace('_', ' ').title()}
   COMPLEXITY LEVEL: {complexity}/10

7. CALIBRATED CONFIDENCE: Explicitly state your confidence level and what would change your answer.

8. TEACH THE PATTERN: Your goal is not just to answer this question, but to teach the PATTERN of reasoning that solves this CLASS of problems.

Generate a response that, when used as training data, will create a model capable of matching or exceeding your performance."""

    def _extract_reasoning_steps(self, content: str) -> List[str]:
        """Extract structured reasoning steps from response."""
        steps = []
        markers = ["[UNDERSTANDING]", "[APPROACH]", "[REASONING]", "[VERIFICATION]", "[ANSWER]", "[EXTENSIONS]"]

        for marker in markers:
            if marker in content:
                start = content.find(marker)
                end = len(content)
                for next_marker in markers:
                    if next_marker != marker and next_marker in content[start+len(marker):]:
                        potential_end = content.find(next_marker, start+len(marker))
                        if potential_end < end:
                            end = potential_end

                step_content = content[start+len(marker):end].strip()
                if step_content:
                    steps.append(f"{marker}: {step_content[:500]}...")

        return steps if steps else ["No structured reasoning markers found"]

    async def generate_batch(self, prompts: List[Dict], max_concurrent: int = 5) -> List[Dict]:
        """Generate responses for a batch of prompts with concurrency control."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_one(prompt_data: Dict) -> Dict:
            async with semaphore:
                try:
                    result = await self.generate_response(
                        prompt_data["prompt"],
                        prompt_data["domain"],
                        prompt_data["complexity"]
                    )
                    return {**prompt_data, **result, "success": True}
                except Exception as e:
                    return {**prompt_data, "error": str(e), "success": False}

        tasks = [process_one(p) for p in prompts]
        results = await asyncio.gather(*tasks)

        await self.close_session()
        return results


class QualityValidator:
    """Validates and scores training pairs for quality."""

    def __init__(self):
        self.quality_criteria = {
            "reasoning_depth": 0.25,
            "correctness_signals": 0.25,
            "completeness": 0.20,
            "clarity": 0.15,
            "teachability": 0.15
        }

    def score_response(self, response: str, domain: str, complexity: int) -> Tuple[float, Dict[str, float]]:
        """Score a response for training quality."""
        scores = {}

        # Reasoning depth - check for structured reasoning
        reasoning_markers = ["[UNDERSTANDING]", "[APPROACH]", "[REASONING]", "[VERIFICATION]", "[ANSWER]"]
        markers_found = sum(1 for m in reasoning_markers if m in response)
        scores["reasoning_depth"] = min(1.0, markers_found / 4)

        # Correctness signals - presence of verification, caveats
        correctness_signals = [
            "verify" in response.lower(),
            "check" in response.lower(),
            "because" in response.lower(),
            "therefore" in response.lower(),
            "however" in response.lower(),
            "assuming" in response.lower()
        ]
        scores["correctness_signals"] = sum(correctness_signals) / len(correctness_signals)

        # Completeness - length relative to complexity
        expected_length = complexity * 500  # Rough heuristic
        actual_length = len(response)
        scores["completeness"] = min(1.0, actual_length / expected_length)

        # Clarity - sentence structure, formatting
        has_sections = any(c in response for c in ["##", "**", "- ", "1."])
        avg_sentence_length = len(response) / max(1, response.count("."))
        clarity_score = 0.5
        if has_sections:
            clarity_score += 0.25
        if 50 < avg_sentence_length < 200:
            clarity_score += 0.25
        scores["clarity"] = clarity_score

        # Teachability - explanatory language
        teaching_phrases = [
            "this means", "in other words", "for example",
            "the key insight", "importantly", "note that",
            "the reason", "this works because", "to understand"
        ]
        teaching_score = sum(1 for p in teaching_phrases if p in response.lower())
        scores["teachability"] = min(1.0, teaching_score / 5)

        # Weighted total
        total = sum(scores[k] * self.quality_criteria[k] for k in scores)

        return total, scores

    def filter_batch(self, pairs: List[Dict], min_score: float = 0.6) -> List[Dict]:
        """Filter training pairs by quality score."""
        validated = []

        for pair in pairs:
            if not pair.get("success", False):
                continue

            score, breakdown = self.score_response(
                pair.get("response", ""),
                pair["domain"],
                pair["complexity"]
            )

            if score >= min_score:
                pair["quality_score"] = score
                pair["quality_breakdown"] = breakdown
                validated.append(pair)

        return validated


class TrainingDataManager:
    """Manages training data storage and deduplication."""

    def __init__(self, output_dir: str = "training_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.output_dir / "index.json"
        self.load_index()

    def load_index(self):
        """Load or initialize the training data index."""
        if self.index_file.exists():
            with open(self.index_file) as f:
                self.index = json.load(f)
        else:
            self.index = {
                "total_pairs": 0,
                "by_domain": {},
                "by_complexity": {},
                "prompt_hashes": [],
                "batches": []
            }

    def save_index(self):
        """Save the index file."""
        with open(self.index_file, "w") as f:
            json.dump(self.index, f, indent=2)

    def save_batch(self, pairs: List[Dict], batch_name: str = None) -> str:
        """Save a batch of training pairs."""
        if batch_name is None:
            batch_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        batch_file = self.output_dir / f"{batch_name}.jsonl"

        saved_count = 0
        with open(batch_file, "w") as f:
            for pair in pairs:
                # Create training pair in chat format
                training_example = {
                    "id": hashlib.md5(pair["prompt"].encode()).hexdigest()[:12],
                    "messages": [
                        {"role": "user", "content": pair["prompt"]},
                        {"role": "assistant", "content": pair["response"]}
                    ],
                    "domain": pair["domain"],
                    "subdomain": pair.get("subdomain", ""),
                    "complexity": pair["complexity"],
                    "quality_score": pair.get("quality_score", 0),
                    "reasoning_steps": pair.get("reasoning_steps", [])
                }
                f.write(json.dumps(training_example) + "\n")
                saved_count += 1

                # Update index
                self.index["total_pairs"] += 1
                self.index["by_domain"][pair["domain"]] = \
                    self.index["by_domain"].get(pair["domain"], 0) + 1
                self.index["by_complexity"][str(pair["complexity"])] = \
                    self.index["by_complexity"].get(str(pair["complexity"]), 0) + 1

        self.index["batches"].append({
            "name": batch_name,
            "file": str(batch_file),
            "count": saved_count,
            "created_at": datetime.now().isoformat()
        })

        self.save_index()

        return str(batch_file)

    def get_statistics(self) -> Dict:
        """Get training data statistics."""
        return {
            "total_pairs": self.index["total_pairs"],
            "domains": self.index["by_domain"],
            "complexity_distribution": self.index["by_complexity"],
            "batch_count": len(self.index["batches"])
        }

    def merge_all(self, output_file: str = "all_training_data.jsonl") -> str:
        """Merge all batches into a single file."""
        output_path = self.output_dir / output_file

        with open(output_path, "w") as out:
            for batch in self.index["batches"]:
                batch_path = Path(batch["file"])
                if batch_path.exists():
                    with open(batch_path) as f:
                        for line in f:
                            out.write(line)

        return str(output_path)


async def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Generate superior training data")
    parser.add_argument("--count", type=int, default=100, help="Number of prompts to generate")
    parser.add_argument("--output-dir", type=str, default="training_data", help="Output directory")
    parser.add_argument("--min-quality", type=float, default=0.6, help="Minimum quality score")
    parser.add_argument("--max-concurrent", type=int, default=5, help="Max concurrent API calls")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--dry-run", action="store_true", help="Generate prompts without API calls")
    args = parser.parse_args()

    print("=" * 60)
    print("SUPERIOR TRAINING DATA GENERATOR")
    print("=" * 60)

    # Initialize components
    generator = PromptGenerator(seed=args.seed)
    validator = QualityValidator()
    manager = TrainingDataManager(output_dir=args.output_dir)

    # Generate prompts
    print(f"\n[1/4] Generating {args.count} prompts across {len(DOMAINS)} domains...")
    prompts = generator.generate_batch(args.count)
    print(f"      Generated {len(prompts)} unique prompts")

    # Show domain distribution
    domain_counts = {}
    for p in prompts:
        domain_counts[p["domain"]] = domain_counts.get(p["domain"], 0) + 1
    print("\n      Domain distribution:")
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        print(f"        {domain}: {count}")

    if args.dry_run:
        print("\n[DRY RUN] Skipping API calls. Sample prompts:")
        for p in prompts[:5]:
            print(f"\n  [{p['domain']}] (complexity: {p['complexity']})")
            print(f"  {p['prompt'][:200]}...")
        return

    # Generate responses with GPT-5.1-codex-mini
    print(f"\n[2/4] Generating responses with GPT-5.1-codex-mini...")
    enhancer = GPT52Enhancer()
    results = await enhancer.generate_batch(prompts, max_concurrent=args.max_concurrent)

    successful = sum(1 for r in results if r.get("success", False))
    print(f"      Successful: {successful}/{len(results)}")

    # Validate and filter
    print(f"\n[3/4] Validating quality (min score: {args.min_quality})...")
    validated = validator.filter_batch(results, min_score=args.min_quality)
    print(f"      Passed validation: {len(validated)}/{successful}")

    if validated:
        avg_quality = sum(p["quality_score"] for p in validated) / len(validated)
        print(f"      Average quality score: {avg_quality:.3f}")

    # Save results
    print(f"\n[4/4] Saving training data...")
    if validated:
        batch_file = manager.save_batch(validated)
        print(f"      Saved to: {batch_file}")

        stats = manager.get_statistics()
        print(f"\n      Total training pairs: {stats['total_pairs']}")
    else:
        print("      No pairs passed validation!")

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
