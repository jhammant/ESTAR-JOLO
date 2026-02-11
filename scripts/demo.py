#!/usr/bin/env python3
"""Side-by-side comparison: full reasoning vs ESTAR-LITE.

Usage:
    python scripts/demo.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --classifier models/estar_lite \
        --task-type open
"""

from __future__ import annotations

import argparse
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from estar.classifier import EstarClassifier
from estar.generator import EstarGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLE_PROBLEMS = [
    "What is the sum of all positive integers less than 100 that are divisible by both 3 and 5?",
    "If f(x) = 2x^2 - 3x + 1, what is f(4)?",
    "A train travels 120 km in 2 hours. A car travels 80 km in 1 hour. How much faster is the car than the train in km/h?",
    "What is the value of √(144) + √(81)?",
    "Solve for x: 3x + 7 = 22",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="ESTAR-LITE demo")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--classifier", default="models/estar_lite")
    parser.add_argument("--task-type", default="open")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--problems", nargs="+", default=None)
    args = parser.parse_args()

    # Load model
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading model on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()

    # Load classifier
    classifier = EstarClassifier.load(args.classifier)

    # Create generator
    generator = EstarGenerator(
        model=model,
        tokenizer=tokenizer,
        classifier=classifier,
        task_type=args.task_type,
    )

    problems = args.problems or SAMPLE_PROBLEMS

    print("\n" + "=" * 70)
    print("ESTAR-LITE Demo: Full Reasoning vs Early-Stopped Reasoning")
    print("=" * 70)

    total_full_tokens = 0
    total_estar_tokens = 0

    for i, problem in enumerate(problems):
        print(f"\n{'─' * 70}")
        print(f"Problem {i + 1}: {problem}")
        print(f"{'─' * 70}")

        prompt = f"<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n"

        # Full reasoning
        full_result = generator.generate_full(prompt, max_new_tokens=4096)
        print(f"\n  📝 Full reasoning:")
        print(f"     Answer: {full_result.answer}")
        print(f"     Thinking tokens: {full_result.thinking_tokens}")
        print(f"     Total tokens: {full_result.total_tokens}")

        # ESTAR-LITE
        estar_result = generator.generate(prompt, max_new_tokens=4096)
        print(f"\n  ⚡ ESTAR-LITE:")
        print(f"     Answer: {estar_result.answer}")
        print(f"     Thinking tokens: {estar_result.thinking_tokens}")
        print(f"     Total tokens: {estar_result.total_tokens}")
        print(f"     Stopped early: {estar_result.stopped_early}")
        if estar_result.stop_step:
            print(f"     Stop step: {estar_result.stop_step}")
            print(f"     Stop probability: {estar_result.stop_probability:.3f}")

        if full_result.thinking_tokens > 0:
            savings = 1 - estar_result.thinking_tokens / full_result.thinking_tokens
            print(f"\n  💰 Token savings: {savings:.1%}")

        total_full_tokens += full_result.thinking_tokens
        total_estar_tokens += estar_result.thinking_tokens

    print(f"\n{'=' * 70}")
    print("Summary")
    print(f"{'=' * 70}")
    print(f"  Total full thinking tokens:   {total_full_tokens}")
    print(f"  Total ESTAR thinking tokens:  {total_estar_tokens}")
    if total_full_tokens > 0:
        ratio = total_full_tokens / max(total_estar_tokens, 1)
        print(f"  Reduction ratio:              {ratio:.1f}x")
    print()


if __name__ == "__main__":
    main()
