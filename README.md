# ESTAR-LITE

**Early-Stopping Token-Aware Reasoning for Efficient LLM Inference**

> Independent implementation based on [arXiv:2602.10004](https://arxiv.org/abs/2602.10004), not affiliated with the original authors.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## What is ESTAR-LITE?

Large reasoning models (LRMs) like DeepSeek-R1 generate long chains-of-thought to solve problems, but **often keep reasoning long after they've already found the correct answer**. ESTAR-LITE detects this redundancy and stops reasoning early.

**Key result from the paper:** **3.7× fewer reasoning tokens** while preserving accuracy (74.9% → 74.2% on average across benchmarks).

### How It Works

ESTAR-LITE is a lightweight **LightGBM classifier** that monitors token log-probabilities during generation and predicts when reasoning can be safely stopped:

```
┌─────────────────────────────────────────────────────────┐
│  LLM Generating Chain-of-Thought                        │
│                                                         │
│  <think>                                                │
│    Step 1: Let me analyze...                            │
│    Step 2: Using the formula...                         │
│    Step 3: The answer is 42.          ◄── ESTAR-LITE    │
│    Step 4: Let me verify...               says STOP!    │
│    Step 5: Actually, double-checking...   (skipped)     │
│    Step 6: Yes, 42 is correct.            (skipped)     │
│  </think>                                               │
│                                                         │
│  The answer is 42. ✓                                    │
└─────────────────────────────────────────────────────────┘
```

### Feature Groups

The classifier uses 4 feature groups extracted from top-k token log-probabilities at each step:

| Group | Features | What it captures |
|-------|----------|-----------------|
| **Instantaneous Evidence** | Per-class probability from token log-probs | Current answer preference |
| **Cumulative Path & Stability** | Running evidence, flip counts, changed_prev | Decision stickiness |
| **Early-Stop Curvature** | Slope (S_es) and second difference (H_es) | Convergence / saturation |
| **Token Confidence Stats** | Mean/var of log-probs, neg perplexity, length | Overall model confidence |

## Quick Start

### Install

```bash
pip install -e .
```

### 1. Generate Training Data

Generate chains-of-thought, slice at 10–100%, and create (features, labels) pairs:

```bash
python scripts/generate_training_data.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --dataset math500 \
    --output data/training_data.npz
```

### 2. Train the Classifier

```bash
python scripts/train_classifier.py \
    --data data/training_data.npz \
    --output models/estar_lite
```

### 3. Run Inference with Early Stopping

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from estar import EstarClassifier, EstarGenerator

model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
classifier = EstarClassifier.load("models/estar_lite")

generator = EstarGenerator(model=model, tokenizer=tokenizer, classifier=classifier)
result = generator.generate("<|im_start|>user\nSolve: 2x + 5 = 13<|im_end|>\n<|im_start|>assistant\n")

print(f"Answer: {result.answer}")
print(f"Thinking tokens: {result.thinking_tokens} (stopped_early={result.stopped_early})")
```

### 4. Demo & Benchmark

```bash
# Side-by-side comparison
python scripts/demo.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --classifier models/estar_lite

# Full benchmark
python scripts/benchmark.py --classifier models/estar_lite --dataset math500 --max-samples 100
```

## Results Format

The benchmark script outputs results in this format:

| Method | Accuracy | Avg Think Tokens |
|--------|----------|-----------------|
| Full Reasoning | X.X% | XXXX |
| ESTAR-LITE | X.X% | XXXX |

**Paper results (Qwen3-8B, 4 benchmarks average):**
- Full reasoning: 74.9% accuracy, 4799 tokens
- ESTAR (full system): 74.2% accuracy, 1290 tokens (3.7× reduction)
- ESTAR-LITE (classifier only): ≥95% relative accuracy, 2–6× reduction

## Project Structure

```
estar-lite/
├── estar/
│   ├── __init__.py        # Package exports
│   ├── features.py        # Feature extraction (4 groups)
│   ├── classifier.py      # LightGBM wrapper
│   ├── generator.py       # HuggingFace generation with early-stopping
│   └── utils.py           # Answer extraction, token mapping
├── scripts/
│   ├── generate_training_data.py
│   ├── train_classifier.py
│   ├── demo.py
│   └── benchmark.py
├── tests/
│   └── test_features.py
├── examples/
│   └── quickstart.py
├── pyproject.toml
├── LICENSE
└── README.md
```

## Configuration

### Classifier Hyperparameters (from paper)

| Parameter | Value |
|-----------|-------|
| n_estimators | 400 |
| num_leaves | 63 |
| learning_rate | 0.07 |
| subsample | 0.9 |
| colsample_bytree | 0.9 |
| threshold (τ) | 0.9 |
| patience | 3 consecutive steps |

### Recommended Models

- **DeepSeek-R1-Distill-Qwen-1.5B** — smallest, runs on consumer hardware
- **DeepSeek-R1-Distill-Qwen-7B** — better accuracy
- **Qwen3-8B** — used in the paper

## Scope

This implementation covers **ESTAR-LITE only** — the lightweight LightGBM classifier component. The full ESTAR system in the paper also includes:
- **ESTAR-FT**: Supervised fine-tuning to teach models to emit `<stop>` tokens
- **ESTAR-RL**: Reinforcement learning with compute-aware rewards

These require model fine-tuning infrastructure and are not included here.

## Citation

```bibtex
@article{wang2025estar,
  title={ESTAR: Early-Stopping Token-Aware Reasoning for Efficient Inference},
  author={Wang, Junda and Yang, Zhichao and Zhang, Dongxu and Batra, Sanjit Singh and Tillman, Robert E.},
  journal={arXiv preprint arXiv:2602.10004},
  year={2025}
}
```

## License

MIT
