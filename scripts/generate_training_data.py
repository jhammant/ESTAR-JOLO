#!/usr/bin/env python3
"""Generate training data for ESTAR-LITE classifier.

For each problem in a dataset:
1. Generate full CoT with a reasoning model
2. Slice at 10%, 20%, ..., 100% of CoT length
3. At each slice, inject </think> and elicit answer
4. Label: 1 if early-stop answer matches full-CoT answer, else 0
5. Extract features at each slice point
6. Save (features, labels) pairs

Usage:
    python scripts/generate_training_data.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --dataset math500 \
        --output data/training_data.npz \
        --task-type open
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from estar.features import FeatureExtractor, StepLogProbs
from estar.utils import (
    answers_match,
    build_answer_buckets,
    default_answer_set,
    extract_answer,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(name: str, split: str = "test") -> list[dict]:
    """Load a dataset. Returns list of dicts with 'question' and 'answer' keys."""
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError("pip install datasets")

    if name == "math500":
        ds = hf_load("HuggingFaceH4/MATH-500", split=split)
        return [{"question": x["problem"], "answer": x["answer"]} for x in ds]
    elif name == "gsm8k":
        ds = hf_load("openai/gsm8k", "main", split=split)
        return [{"question": x["question"], "answer": x["answer"].split("####")[-1].strip()} for x in ds]
    else:
        raise ValueError(f"Unknown dataset: {name}. Supported: math500, gsm8k")


def generate_cot(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    max_tokens: int = 4096,
    temperature: float = 0.6,
) -> tuple[list[int], str]:
    """Generate full chain-of-thought.

    Returns:
        Tuple of (token_ids, decoded_text).
    """
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_k=20,
            top_p=0.95,
            repetition_penalty=1.2,
        )

    gen_ids = outputs[0, input_len:].tolist()
    text = tokenizer.decode(gen_ids, skip_special_tokens=False)
    return gen_ids, text


def extract_features_at_slices(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_ids: torch.LongTensor,
    cot_ids: list[int],
    feature_extractor: FeatureExtractor,
    slices: list[float],
    top_k: int = 20,
) -> list[np.ndarray]:
    """Extract features at each slice point of the CoT.

    Returns list of feature vectors, one per slice.
    """
    features_list = []
    total_len = len(cot_ids)

    for frac in slices:
        feature_extractor.reset()
        cut = max(1, int(frac * total_len))

        # Get log-probs at the cut point
        prefix = torch.cat([
            prompt_ids,
            torch.tensor([cot_ids[:cut]], device=model.device),
        ], dim=1)

        with torch.no_grad():
            outputs = model(prefix)
            logits = outputs.logits[:, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            topk_lp, topk_ids = torch.topk(log_probs, top_k, dim=-1)

        step_data = StepLogProbs(
            token_ids=topk_ids[0].cpu().numpy(),
            log_probs=topk_lp[0].cpu().numpy(),
            step=cut,
        )
        feat = feature_extractor.extract(step_data, total_steps=total_len)
        features_list.append(feat)

    return features_list


def force_early_stop_answer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_ids: torch.LongTensor,
    cot_ids: list[int],
    cut_frac: float,
    task_type: str = "open",
    max_answer_tokens: int = 256,
) -> str | None:
    """Force early stop at a fraction of CoT and elicit answer."""
    total_len = len(cot_ids)
    cut = max(1, int(cut_frac * total_len))

    # Build: prompt + cot[:cut] + </think>
    end_think_ids = tokenizer.encode("</think>", add_special_tokens=False)
    input_ids = torch.cat([
        prompt_ids,
        torch.tensor([cot_ids[:cut]], device=model.device),
        torch.tensor([end_think_ids], device=model.device),
    ], dim=1)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_answer_tokens,
            temperature=0.0,
            do_sample=False,
        )

    answer_ids = outputs[0, input_ids.shape[1]:]
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)
    return extract_answer(answer_text, task_type)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ESTAR-LITE training data")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--dataset", default="math500", choices=["math500", "gsm8k"])
    parser.add_argument("--output", default="data/training_data.npz")
    parser.add_argument("--task-type", default="open", choices=["open", "closed"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    # Load model
    logger.info(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    from estar.utils import get_device
    device = get_device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device in ("cuda", "mps") else torch.float32,
        device_map=device if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()

    # Setup feature extractor
    ans_set = default_answer_set(args.task_type)
    token_to_bucket = build_answer_buckets(tokenizer, ans_set)
    feature_extractor = FeatureExtractor(
        num_buckets=len(ans_set),
        token_to_bucket=token_to_bucket,
    )

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    data = load_dataset(args.dataset)
    if args.max_samples:
        data = data[:args.max_samples]

    slices = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    all_features = []
    all_labels = []
    metadata = []

    for idx, item in enumerate(tqdm(data, desc="Generating training data")):
        question = item["question"]

        # Generate full CoT
        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n"
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)

        try:
            cot_ids, cot_text = generate_cot(model, tokenizer, question)
        except Exception as e:
            logger.warning(f"Failed to generate CoT for sample {idx}: {e}")
            continue

        if len(cot_ids) < 10:
            continue

        # Get full-CoT answer
        full_answer = force_early_stop_answer(
            model, tokenizer, prompt_ids, cot_ids, 1.0, args.task_type
        )

        if full_answer is None:
            continue

        # Extract features and labels at each slice
        features = extract_features_at_slices(
            model, tokenizer, prompt_ids, cot_ids, feature_extractor, slices
        )

        for frac, feat in zip(slices, features):
            early_answer = force_early_stop_answer(
                model, tokenizer, prompt_ids, cot_ids, frac, args.task_type
            )
            label = 1 if answers_match(early_answer, full_answer) else 0

            all_features.append(feat)
            all_labels.append(label)
            metadata.append({
                "idx": idx,
                "slice": frac,
                "early_answer": early_answer,
                "full_answer": full_answer,
            })

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    X = np.stack(all_features)
    y = np.array(all_labels)

    np.savez(
        output_path,
        X=X,
        y=y,
        feature_names=feature_extractor.feature_names(),
    )

    # Save metadata
    meta_path = output_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"Saved {len(all_features)} samples to {output_path}")
    logger.info(f"Positive rate: {y.mean():.2%}")


if __name__ == "__main__":
    main()
