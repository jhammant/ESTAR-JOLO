#!/usr/bin/env python3
"""Benchmark ESTAR-LITE on a dataset.

Reports accuracy and token reduction compared to full reasoning.

Usage:
    python scripts/benchmark.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --classifier models/estar_lite \
        --dataset math500 \
        --task-type open \
        --max-samples 100
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from estar.classifier import EstarClassifier
from estar.generator import EstarGenerator
from estar.utils import answers_match

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(name: str, split: str = "test") -> list[dict]:
    from datasets import load_dataset as hf_load
    if name == "math500":
        ds = hf_load("HuggingFaceH4/MATH-500", split=split)
        return [{"question": x["problem"], "answer": x["answer"]} for x in ds]
    elif name == "gsm8k":
        ds = hf_load("openai/gsm8k", "main", split=split)
        return [{"question": x["question"], "answer": x["answer"].split("####")[-1].strip()} for x in ds]
    else:
        raise ValueError(f"Unknown dataset: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ESTAR-LITE")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--dataset", default="math500")
    parser.add_argument("--task-type", default="open")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", default=None, help="Save results JSON")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
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

    classifier = EstarClassifier.load(args.classifier)
    generator = EstarGenerator(
        model=model, tokenizer=tokenizer, classifier=classifier, task_type=args.task_type
    )

    data = load_dataset(args.dataset)
    if args.max_samples:
        data = data[:args.max_samples]

    results = []
    full_correct = 0
    estar_correct = 0
    total_full_think = 0
    total_estar_think = 0

    for item in tqdm(data, desc="Benchmarking"):
        prompt = f"<|im_start|>user\n{item['question']}<|im_end|>\n<|im_start|>assistant\n"
        gold = item["answer"]

        full = generator.generate_full(prompt)
        estar = generator.generate(prompt)

        fc = answers_match(full.answer, gold)
        ec = answers_match(estar.answer, gold)
        full_correct += int(fc)
        estar_correct += int(ec)
        total_full_think += full.thinking_tokens
        total_estar_think += estar.thinking_tokens

        results.append({
            "question": item["question"][:100],
            "gold": gold,
            "full_answer": full.answer,
            "estar_answer": estar.answer,
            "full_correct": fc,
            "estar_correct": ec,
            "full_think_tokens": full.thinking_tokens,
            "estar_think_tokens": estar.thinking_tokens,
            "stopped_early": estar.stopped_early,
        })

    n = len(data)
    full_acc = full_correct / n * 100
    estar_acc = estar_correct / n * 100
    avg_full = total_full_think / n
    avg_estar = total_estar_think / max(n, 1)
    ratio = avg_full / max(avg_estar, 1)

    print(f"\n{'=' * 50}")
    print(f"Benchmark Results: {args.dataset} ({n} samples)")
    print(f"{'=' * 50}")
    print(f"{'Method':<20} {'Accuracy':>10} {'Avg Think Tokens':>18}")
    print(f"{'─' * 50}")
    print(f"{'Full Reasoning':<20} {full_acc:>9.1f}% {avg_full:>17.0f}")
    print(f"{'ESTAR-LITE':<20} {estar_acc:>9.1f}% {avg_estar:>17.0f}")
    print(f"{'─' * 50}")
    print(f"Token reduction: {ratio:.1f}x")
    print(f"Relative accuracy: {estar_acc / max(full_acc, 0.1) * 100:.1f}%")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"summary": {
                "dataset": args.dataset, "n": n,
                "full_acc": full_acc, "estar_acc": estar_acc,
                "avg_full_think": avg_full, "avg_estar_think": avg_estar,
                "reduction_ratio": ratio,
            }, "results": results}, f, indent=2, default=str)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
