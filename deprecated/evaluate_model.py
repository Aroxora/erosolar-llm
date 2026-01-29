#!/usr/bin/env python3
"""
Model Evaluation Script
Runs comprehensive benchmarks to measure model capabilities.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


def load_model(model_path: str):
    """Load model for evaluation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_tokens: int = 512) -> str:
    """Generate a response from the model."""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from the response
    response = response[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]
    return response.strip()


def eval_mmlu(model, tokenizer, num_samples: int = 100) -> float:
    """Evaluate on MMLU benchmark."""
    from datasets import load_dataset

    try:
        dataset = load_dataset("cais/mmlu", "all", split="test")
        dataset = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))
    except:
        return 0.0

    correct = 0
    total = 0

    for item in dataset:
        question = item["question"]
        choices = item["choices"]
        answer = item["answer"]

        prompt = f"""Question: {question}

Choices:
A) {choices[0]}
B) {choices[1]}
C) {choices[2]}
D) {choices[3]}

Answer with just the letter (A, B, C, or D):"""

        response = generate_response(model, tokenizer, prompt, max_tokens=10)

        # Extract answer letter
        response = response.strip().upper()
        if response and response[0] in "ABCD":
            predicted = ord(response[0]) - ord('A')
            if predicted == answer:
                correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def eval_gsm8k(model, tokenizer, num_samples: int = 100) -> float:
    """Evaluate on GSM8K math benchmark."""
    from datasets import load_dataset
    import re

    try:
        dataset = load_dataset("gsm8k", "main", split="test")
        dataset = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))
    except:
        return 0.0

    correct = 0
    total = 0

    for item in dataset:
        question = item["question"]
        answer = item["answer"]

        # Extract the final number from the answer
        true_answer = re.findall(r'#### (\-?[\d,]+)', answer)
        if not true_answer:
            continue
        true_answer = true_answer[0].replace(",", "")

        prompt = f"""Solve this math problem step by step:

{question}

Show your work, then give the final numerical answer after "#### ":"""

        response = generate_response(model, tokenizer, prompt, max_tokens=512)

        # Extract predicted answer
        predicted = re.findall(r'#### (\-?[\d,]+)', response)
        if predicted:
            predicted = predicted[-1].replace(",", "")
            if predicted == true_answer:
                correct += 1

        total += 1

    return correct / total if total > 0 else 0.0


def eval_hellaswag(model, tokenizer, num_samples: int = 100) -> float:
    """Evaluate on HellaSwag commonsense reasoning."""
    from datasets import load_dataset

    try:
        dataset = load_dataset("hellaswag", split="validation")
        dataset = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))
    except:
        return 0.0

    correct = 0
    total = 0

    for item in dataset:
        ctx = item["ctx"]
        endings = item["endings"]
        answer = int(item["label"])

        prompt = f"""Context: {ctx}

Which ending makes the most sense?
A) {endings[0]}
B) {endings[1]}
C) {endings[2]}
D) {endings[3]}

Answer with just the letter:"""

        response = generate_response(model, tokenizer, prompt, max_tokens=10)

        response = response.strip().upper()
        if response and response[0] in "ABCD":
            predicted = ord(response[0]) - ord('A')
            if predicted == answer:
                correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def eval_arc(model, tokenizer, num_samples: int = 100) -> float:
    """Evaluate on ARC-Challenge science questions."""
    from datasets import load_dataset

    try:
        dataset = load_dataset("ai2_arc", "ARC-Challenge", split="test")
        dataset = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))
    except:
        return 0.0

    correct = 0
    total = 0

    for item in dataset:
        question = item["question"]
        choices = item["choices"]
        answer_key = item["answerKey"]

        # Build choices string
        choice_text = ""
        choice_labels = choices["label"]
        choice_texts = choices["text"]

        for i, (label, text) in enumerate(zip(choice_labels, choice_texts)):
            choice_text += f"{label}) {text}\n"

        prompt = f"""Science question: {question}

{choice_text}
Answer with just the letter:"""

        response = generate_response(model, tokenizer, prompt, max_tokens=10)

        response = response.strip().upper()
        if response and response[0] == answer_key:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def eval_truthfulqa(model, tokenizer, num_samples: int = 100) -> float:
    """Evaluate on TruthfulQA for factual accuracy."""
    from datasets import load_dataset

    try:
        dataset = load_dataset("truthful_qa", "generation", split="validation")
        dataset = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))
    except:
        return 0.0

    # Simplified evaluation - check if response avoids common misconceptions
    correct = 0
    total = 0

    for item in dataset:
        question = item["question"]
        correct_answers = item["correct_answers"]
        incorrect_answers = item["incorrect_answers"]

        prompt = f"""Answer this question truthfully and accurately:

{question}

Answer:"""

        response = generate_response(model, tokenizer, prompt, max_tokens=200).lower()

        # Check if response aligns more with correct answers
        correct_match = any(ans.lower() in response for ans in correct_answers)
        incorrect_match = any(ans.lower() in response for ans in incorrect_answers)

        if correct_match and not incorrect_match:
            correct += 1
        elif not correct_match and not incorrect_match:
            correct += 0.5  # Partial credit for avoiding misinformation

        total += 1

    return correct / total if total > 0 else 0.0


BENCHMARKS = {
    "mmlu": eval_mmlu,
    "gsm8k": eval_gsm8k,
    "hellaswag": eval_hellaswag,
    "arc": eval_arc,
    "truthfulqa": eval_truthfulqa,
}


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on benchmarks")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--benchmarks", type=str, default="all",
                       help="Comma-separated benchmarks or 'all'")
    parser.add_argument("--samples", type=int, default=100,
                       help="Number of samples per benchmark")
    parser.add_argument("--output", type=str, default="eval_results.json")
    args = parser.parse_args()

    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    # Load model
    print(f"\nLoading model from: {args.model_path}")
    model, tokenizer = load_model(args.model_path)

    # Select benchmarks
    if args.benchmarks == "all":
        benchmarks_to_run = list(BENCHMARKS.keys())
    else:
        benchmarks_to_run = args.benchmarks.split(",")

    # Run evaluations
    results = {}

    for bench_name in benchmarks_to_run:
        if bench_name not in BENCHMARKS:
            print(f"Unknown benchmark: {bench_name}")
            continue

        print(f"\nRunning {bench_name}...")
        score = BENCHMARKS[bench_name](model, tokenizer, args.samples)
        results[bench_name] = score
        print(f"  Score: {score:.2%}")

    # Calculate average
    if results:
        results["average"] = sum(results.values()) / len(results)

    # Add metadata
    output = {
        "model_path": args.model_path,
        "samples_per_benchmark": args.samples,
        "evaluated_at": datetime.now().isoformat(),
        "results": results
    }

    # Save results
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {args.output}")
    print("\nSummary:")
    for bench, score in results.items():
        print(f"  {bench}: {score:.2%}")


if __name__ == "__main__":
    main()
