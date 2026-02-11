#!/usr/bin/env python3
"""Minimal ESTAR-LITE quickstart example.

Shows how to:
1. Load a reasoning model
2. Load a trained ESTAR-LITE classifier
3. Generate with early stopping
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

from estar import EstarClassifier, EstarGenerator

# 1. Load model
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True
)

# 2. Load trained classifier
classifier = EstarClassifier.load("models/estar_lite")

# 3. Create generator with early stopping
generator = EstarGenerator(
    model=model,
    tokenizer=tokenizer,
    classifier=classifier,
    task_type="open",  # "open" for math, "closed" for multiple-choice
)

# 4. Generate!
prompt = "<|im_start|>user\nWhat is 15% of 240?<|im_end|>\n<|im_start|>assistant\n"

# With early stopping
result = generator.generate(prompt)
print(f"Answer: {result.answer}")
print(f"Thinking tokens: {result.thinking_tokens}")
print(f"Stopped early: {result.stopped_early}")

# Without early stopping (baseline)
full_result = generator.generate_full(prompt)
print(f"\nFull answer: {full_result.answer}")
print(f"Full thinking tokens: {full_result.thinking_tokens}")

if full_result.thinking_tokens > 0:
    savings = 1 - result.thinking_tokens / full_result.thinking_tokens
    print(f"Token savings: {savings:.0%}")
