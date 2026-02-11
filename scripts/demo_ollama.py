#!/usr/bin/env python3
"""ESTAR-LITE demo using Ollama (no GPU/HuggingFace setup required).

Prerequisites:
    1. Install Ollama: https://ollama.com
    2. Pull the model: ollama pull deepseek-r1:1.5b
    3. pip install ollama

Usage:
    # Baseline only (no classifier needed):
    python scripts/demo_ollama.py --baseline-only

    # With ESTAR-LITE simulation:
    python scripts/demo_ollama.py --classifier models/estar_lite

    # Custom question:
    python scripts/demo_ollama.py --question "What is 15% of 240?"
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SAMPLE_PROBLEMS = [
    "What is the sum of all prime numbers less than 20?",
    "If f(x) = 2x^2 - 3x + 1, what is f(4)?",
    "Solve for x: 3x + 7 = 22",
    "What is the square root of 144 plus the square root of 81?",
    "A rectangle has length 12 and width 5. What is its diagonal?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="ESTAR-LITE Ollama Demo")
    parser.add_argument("--model", default="deepseek-r1:1.5b",
                        help="Ollama model name")
    parser.add_argument("--classifier", default=None,
                        help="Path to trained classifier (omit for baseline-only)")
    parser.add_argument("--question", default=None,
                        help="Custom question (default: sample problems)")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Only run baseline generation")
    parser.add_argument("--task-type", default="open",
                        choices=["open", "closed"])
    parser.add_argument("--host", default=None,
                        help="Ollama server URL")
    args = parser.parse_args()

    # Check Ollama is available
    try:
        import ollama
        # Quick connectivity check
        ollama.list()
    except ImportError:
        print("Error: ollama package not installed. Run: pip install ollama")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Cannot connect to Ollama server: {e}")
        print("Make sure Ollama is running: https://ollama.com")
        sys.exit(1)

    from estar.ollama_generator import OllamaEstarGenerator

    # Load classifier if provided
    classifier = None
    if args.classifier and not args.baseline_only:
        from estar.classifier import EstarClassifier
        try:
            classifier = EstarClassifier.load(args.classifier)
            logger.info(f"Loaded classifier from {args.classifier}")
        except FileNotFoundError:
            logger.warning(f"Classifier not found at {args.classifier}, running baseline only")

    generator = OllamaEstarGenerator(
        model=args.model,
        classifier=classifier,
        task_type=args.task_type,
        host=args.host,
    )

    problems = [args.question] if args.question else SAMPLE_PROBLEMS[:3]

    print()
    print("=" * 70)
    print("  ESTAR-LITE Demo (Ollama Backend)")
    print(f"  Model: {args.model}")
    print(f"  Mode: {'Baseline + ESTAR simulation' if classifier else 'Baseline only'}")
    print("=" * 70)

    total_full_tokens = 0
    total_estar_tokens = 0

    for i, problem in enumerate(problems):
        print(f"\n{'─' * 70}")
        print(f"  Problem {i + 1}: {problem}")
        print(f"{'─' * 70}")

        # Full reasoning (baseline)
        print("\n  Generating full reasoning...", end="", flush=True)
        full_result = generator.generate_full(problem)
        print(" done!")

        print(f"\n  Full Reasoning:")
        print(f"    Answer: {full_result.answer}")
        print(f"    Thinking tokens: ~{full_result.thinking_tokens}")
        # Show first 200 chars of thinking
        preview = full_result.thinking_text[:200]
        if len(full_result.thinking_text) > 200:
            preview += "..."
        print(f"    Thinking preview: {preview}")

        total_full_tokens += full_result.thinking_tokens

        # ESTAR-LITE simulation
        if classifier:
            print("\n  Running ESTAR-LITE simulation...", end="", flush=True)
            estar_result = generator.simulate_estar(problem)
            print(" done!")

            print(f"\n  ESTAR-LITE (simulated):")
            print(f"    Answer: {estar_result.answer}")
            print(f"    Would stop at token: {estar_result.stop_step}")
            print(f"    Thinking tokens used: {estar_result.thinking_tokens}")
            print(f"    Stopped early: {estar_result.stopped_early}")
            if estar_result.stop_probability:
                print(f"    Stop probability: {estar_result.stop_probability:.3f}")

            if full_result.thinking_tokens > 0 and estar_result.stopped_early:
                savings = 1 - estar_result.thinking_tokens / full_result.thinking_tokens
                print(f"\n    Token savings: {savings:.1%}")

            total_estar_tokens += estar_result.thinking_tokens

    # Summary
    if classifier and total_full_tokens > 0:
        print(f"\n{'=' * 70}")
        print("  Summary")
        print(f"{'=' * 70}")
        print(f"  Total full thinking tokens:   {total_full_tokens}")
        print(f"  Total ESTAR thinking tokens:  {total_estar_tokens}")
        ratio = total_full_tokens / max(total_estar_tokens, 1)
        print(f"  Reduction ratio:              {ratio:.1f}x")
    print()


if __name__ == "__main__":
    main()
