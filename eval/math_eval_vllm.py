#!/usr/bin/env python3
"""
Math evaluation script for MobileLLM-R1 style models.

This script evaluates models on GSM8K and MATH500 using the methodology from:
https://github.com/facebookresearch/MobileLLM-R1/tree/main/evaluation

Key settings (matching official eval):
- System prompt: "Please reason step by step, and put your final answer within \\boxed{}."
- Zero-shot evaluation
- Temperature: 0.6, top_p: 0.95
- Answer extraction: boxed{} -> "answer is" -> last number fallback
- Numeric comparison with tolerance

Usage:
    python eval/math_eval_vllm.py --model facebook/MobileLLM-R1-140M --tasks gsm8k,math500
    python eval/math_eval_vllm.py --model ./my_finetuned_model --tasks gsm8k --limit 100
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from vllm import LLM, SamplingParams

# Default system prompt for math reasoning (from MobileLLM-R1 model card)
MATH_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def parse_args():
    parser = argparse.ArgumentParser(description="Math evaluation for reasoning models")
    parser.add_argument(
        "--model", type=str, required=True, help="Model path or HF model ID"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="gsm8k",
        help="Comma-separated tasks: gsm8k,math500",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./experiments/evals", help="Output directory"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of examples (for testing)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6, help="Sampling temperature"
    )
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument(
        "--max_tokens", type=int, default=4096, help="Max generation tokens"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Model dtype")
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.5,
        help="GPU memory utilization",
    )
    parser.add_argument(
        "--system_prompt", type=str, default=MATH_SYSTEM_PROMPT, help="System prompt"
    )
    parser.add_argument(
        "--save_samples", action="store_true", help="Save per-sample results"
    )
    return parser.parse_args()


def extract_answer(response: str, use_last_number: bool = True) -> str:
    """
    Extract answer from model response using Facebook's methodology:
    1. Look for \\boxed{...} (rightmost)
    2. Look for "the answer is X" pattern
    3. Fall back to last number in response
    """
    # Try boxed first (find rightmost)
    boxed_matches = list(
        re.finditer(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", response)
    )
    if boxed_matches:
        answer = boxed_matches[-1].group(1).strip()
        return clean_answer(answer)

    # Try "answer is" pattern
    answer_is_matches = re.findall(
        r"(?:the\s+)?answer\s+is[:\s]+\$?([0-9,.\-/]+)", response.lower()
    )
    if answer_is_matches:
        return clean_answer(answer_is_matches[-1])

    # Fall back to last number
    if use_last_number:
        numbers = re.findall(r"-?\d+\.?\d*", response.replace(",", ""))
        if numbers:
            return numbers[-1]

    return ""


def clean_answer(answer: str) -> str:
    """Clean answer string for comparison."""
    answer = answer.strip()
    answer = answer.replace(",", "")
    answer = answer.replace("$", "")
    answer = answer.replace("%", "")
    answer = answer.rstrip(".")

    # Handle fractions like \frac{a}{b}
    frac_match = re.match(r"\\frac\{(\d+)\}\{(\d+)\}", answer)
    if frac_match:
        num, denom = frac_match.groups()
        try:
            return str(float(num) / float(denom))
        except Exception:
            pass

    return answer


def compare_answers(pred: str, gold: str, tolerance: float = 1e-4) -> bool:
    """Compare predicted and gold answers with numeric tolerance."""
    if not pred or not gold:
        return False

    pred = clean_answer(pred)
    gold = clean_answer(gold)

    # Try exact string match first
    if pred == gold:
        return True

    # Try numeric comparison
    try:
        pred_val = float(pred)
        gold_val = float(gold)
        if gold_val == 0:
            return abs(pred_val) < tolerance
        return abs(pred_val - gold_val) / abs(gold_val) < tolerance
    except (ValueError, ZeroDivisionError):
        pass

    return False


def load_gsm8k(split: str = "test", limit: int = None):
    """Load GSM8K dataset."""
    ds = load_dataset("gsm8k", "main", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    examples = []
    for ex in ds:
        # Ground truth is after "####"
        answer_parts = ex["answer"].split("####")
        gold = answer_parts[1].strip() if len(answer_parts) > 1 else ""
        examples.append(
            {"question": ex["question"], "gold": gold, "full_answer": ex["answer"]}
        )
    return examples


def load_math500(split: str = "test", limit: int = None):
    """Load MATH-500 dataset."""
    ds = load_dataset("HuggingFaceH4/MATH-500", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    examples = []
    for ex in ds:
        # Extract answer from solution (look for boxed)
        solution = ex.get("solution", "")
        gold = extract_answer(solution)
        examples.append(
            {"question": ex["problem"], "gold": gold, "full_answer": solution}
        )
    return examples


def evaluate_task(
    llm: LLM,
    tokenizer,
    task_name: str,
    examples: list,
    sampling_params: SamplingParams,
    system_prompt: str,
    save_samples: bool = False,
    output_dir: str = None,
) -> dict:
    """Evaluate a single task."""

    # Prepare prompts
    prompts = []
    for ex in examples:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ex["question"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)

    # Generate responses
    print(f"Generating {len(prompts)} responses for {task_name}...")
    outputs = llm.generate(prompts, sampling_params)

    # Evaluate
    correct = 0
    samples = []

    for i, (ex, output) in enumerate(zip(examples, outputs)):
        response = output.outputs[0].text
        pred = extract_answer(response)
        gold = ex["gold"]
        is_correct = compare_answers(pred, gold)

        if is_correct:
            correct += 1

        sample = {
            "idx": i,
            "question": ex["question"],
            "gold": gold,
            "pred": pred,
            "correct": is_correct,
            "response_len": len(response),
            "has_boxed": "\\boxed" in response,
            "has_think_end": "</think>" in response,
        }

        if save_samples:
            sample["response"] = response

        samples.append(sample)

    accuracy = correct / len(examples) * 100

    results = {
        "task": task_name,
        "num_samples": len(examples),
        "correct": correct,
        "accuracy": accuracy,
        "boxed_rate": sum(1 for s in samples if s["has_boxed"]) / len(samples) * 100,
        "think_end_rate": sum(1 for s in samples if s["has_think_end"])
        / len(samples)
        * 100,
    }

    # Save samples if requested
    if save_samples and output_dir:
        samples_file = Path(output_dir) / f"{task_name}_samples.jsonl"
        with open(samples_file, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        print(f"Saved samples to {samples_file}")

    return results


def main():
    args = parse_args()

    # Setup output directory
    model_name = args.model.replace("/", "__")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"math_eval_{model_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=" * 60)
    print(f"Math Evaluation (MobileLLM-R1 methodology)")
    print(f"=" * 60)
    print(f"Model:       {args.model}")
    print(f"Tasks:       {args.tasks}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p:       {args.top_p}")
    print(f"Max tokens:  {args.max_tokens}")
    print(f"Seed:        {args.seed}")
    print(f"Output:      {output_dir}")
    print(f"=" * 60)

    # Initialize vLLM
    print("\nLoading model...")
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        trust_remote_code=True,
        enforce_eager=True,  # Avoid CUDA graph issues with fine-tuned models
    )

    # Get tokenizer for chat template
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Ensure pad_token is set (some fine-tuned models may be missing it)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Setup sampling params
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    # Run evaluation for each task
    tasks = [t.strip() for t in args.tasks.split(",")]
    all_results = {}

    for task in tasks:
        print(f"\n{'=' * 40}")
        print(f"Evaluating: {task}")
        print(f"{'=' * 40}")

        # Load dataset
        if task == "gsm8k":
            examples = load_gsm8k(limit=args.limit)
        elif task in ["math500", "math_500", "minerva_math500"]:
            examples = load_math500(limit=args.limit)
        else:
            print(f"Unknown task: {task}, skipping")
            continue

        print(f"Loaded {len(examples)} examples")

        # Evaluate
        results = evaluate_task(
            llm=llm,
            tokenizer=tokenizer,
            task_name=task,
            examples=examples,
            sampling_params=sampling_params,
            system_prompt=args.system_prompt,
            save_samples=args.save_samples,
            output_dir=output_dir,
        )

        all_results[task] = results

        print(f"\nResults for {task}:")
        print(f"  Accuracy:      {results['accuracy']:.2f}%")
        print(f"  Correct:       {results['correct']}/{results['num_samples']}")
        print(f"  Boxed rate:    {results['boxed_rate']:.1f}%")
        print(f"  Think-end rate:{results['think_end_rate']:.1f}%")

    # Save aggregated results
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(
            {"config": vars(args), "results": all_results, "timestamp": timestamp},
            f,
            indent=2,
        )

    print(f"\n{'=' * 60}")
    print("FINAL RESULTS")
    print(f"{'=' * 60}")
    for task, res in all_results.items():
        print(
            f"{task:20s}: {res['accuracy']:6.2f}% ({res['correct']}/{res['num_samples']})"
        )
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
