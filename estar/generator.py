"""Modified generation with ESTAR-LITE early-stopping.

Hooks into HuggingFace transformers generation to:
1. Capture top-k next-token log-probabilities at each step
2. Compute features online via FeatureExtractor
3. Inject </think> when the classifier says stop
4. Elicit the final answer
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from estar.classifier import EstarClassifier
from estar.features import FeatureExtractor, StepLogProbs
from estar.utils import build_answer_buckets, default_answer_set, extract_answer

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of ESTAR-LITE generation.

    Attributes:
        text: Full generated text.
        answer: Extracted answer.
        thinking_tokens: Number of reasoning tokens generated.
        total_tokens: Total tokens generated.
        stopped_early: Whether early stopping was triggered.
        stop_step: Step at which early stopping occurred (None if not).
        stop_probability: Classifier probability at stop step.
    """
    text: str
    answer: Optional[str]
    thinking_tokens: int
    total_tokens: int
    stopped_early: bool
    stop_step: Optional[int] = None
    stop_probability: Optional[float] = None


class EstarGenerator:
    """Generator with ESTAR-LITE early-stopping.

    Wraps a HuggingFace model and intercepts generation to apply
    the ESTAR-LITE classifier for early stopping of chain-of-thought.

    Args:
        model: HuggingFace model.
        tokenizer: HuggingFace tokenizer.
        classifier: Trained EstarClassifier.
        task_type: "open" or "closed" (affects answer buckets and extraction).
        answer_set: Custom answer set (overrides task_type default).
        top_k: Number of top tokens for feature extraction.
        check_interval: Check classifier every N tokens (default 1).
        max_thinking_tokens: Maximum thinking tokens before forced stop.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        classifier: EstarClassifier,
        task_type: str = "open",
        answer_set: Optional[list[str]] = None,
        top_k: int = 20,
        check_interval: int = 1,
        max_thinking_tokens: int = 4096,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.classifier = classifier
        self.task_type = task_type
        self.top_k = top_k
        self.check_interval = check_interval
        self.max_thinking_tokens = max_thinking_tokens

        # Build answer buckets
        ans_set = answer_set or default_answer_set(task_type)
        token_to_bucket = build_answer_buckets(tokenizer, ans_set)
        self.feature_extractor = FeatureExtractor(
            num_buckets=len(ans_set),
            token_to_bucket=token_to_bucket,
            top_k=top_k,
        )

        # Find special token IDs
        self._think_end_ids = tokenizer.encode("</think>", add_special_tokens=False)
        self._eos_id = tokenizer.eos_token_id

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 8192,
        temperature: float = 0.6,
        do_sample: bool = True,
    ) -> GenerationResult:
        """Generate with ESTAR-LITE early stopping.

        Args:
            prompt: Input prompt (will be wrapped in chat template if needed).
            max_new_tokens: Maximum total new tokens.
            temperature: Sampling temperature.
            do_sample: Whether to sample (vs greedy).

        Returns:
            GenerationResult with generated text and metadata.
        """
        self.feature_extractor.reset()
        self.classifier.reset_patience()

        # Encode input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_ids = inputs["input_ids"]
        input_len = input_ids.shape[1]

        # Track state
        generated_ids = input_ids.clone()
        thinking_tokens = 0
        stopped_early = False
        stop_step = None
        stop_prob = None
        in_thinking = True  # assume we start inside <think> block

        for step in range(max_new_tokens):
            outputs = self.model(generated_ids)
            logits = outputs.logits[:, -1, :]  # (1, vocab_size)

            # Get top-k log-probabilities
            log_probs = torch.log_softmax(logits, dim=-1)
            topk_lp, topk_ids = torch.topk(log_probs, self.top_k, dim=-1)

            # Sample or greedy next token
            if do_sample and temperature > 0:
                scaled_logits = logits / temperature
                probs = torch.softmax(scaled_logits, dim=-1)
                next_id = torch.multinomial(probs, 1)
            else:
                next_id = logits.argmax(dim=-1, keepdim=True)

            next_token_id = next_id.item()

            # Check for </think> or EOS
            if next_token_id == self._eos_id:
                generated_ids = torch.cat([generated_ids, next_id], dim=-1)
                break

            # Check if model naturally produces </think>
            decoded = self.tokenizer.decode([next_token_id])
            if "</think>" in decoded:
                in_thinking = False

            if in_thinking:
                thinking_tokens += 1

                # Extract features and check classifier
                if thinking_tokens % self.check_interval == 0:
                    step_data = StepLogProbs(
                        token_ids=topk_ids[0].cpu().numpy(),
                        log_probs=topk_lp[0].cpu().numpy(),
                        step=thinking_tokens,
                    )
                    features = self.feature_extractor.extract(step_data)

                    if self.classifier.should_stop(features):
                        stopped_early = True
                        stop_step = thinking_tokens
                        stop_prob = self.classifier.predict_proba(features)
                        logger.info(
                            f"ESTAR-LITE: stopping at step {thinking_tokens} "
                            f"(prob={stop_prob:.3f})"
                        )
                        # Inject </think> and break thinking
                        end_ids = torch.tensor(
                            [self._think_end_ids], device=generated_ids.device
                        )
                        generated_ids = torch.cat([generated_ids, end_ids], dim=-1)
                        in_thinking = False
                        continue

                # Force stop if too many thinking tokens
                if thinking_tokens >= self.max_thinking_tokens:
                    logger.info(
                        f"ESTAR-LITE: forced stop at max_thinking_tokens={self.max_thinking_tokens}"
                    )
                    end_ids = torch.tensor(
                        [self._think_end_ids], device=generated_ids.device
                    )
                    generated_ids = torch.cat([generated_ids, end_ids], dim=-1)
                    in_thinking = False
                    stopped_early = True
                    stop_step = thinking_tokens
                    continue

            generated_ids = torch.cat([generated_ids, next_id], dim=-1)

        # Decode full output
        full_text = self.tokenizer.decode(
            generated_ids[0, input_len:], skip_special_tokens=False
        )

        # Extract answer from text after </think>
        answer_text = full_text.split("</think>")[-1] if "</think>" in full_text else full_text
        answer = extract_answer(answer_text, self.task_type)

        total_tokens = generated_ids.shape[1] - input_len

        return GenerationResult(
            text=full_text,
            answer=answer,
            thinking_tokens=thinking_tokens,
            total_tokens=total_tokens,
            stopped_early=stopped_early,
            stop_step=stop_step,
            stop_probability=stop_prob,
        )

    @torch.no_grad()
    def generate_full(
        self,
        prompt: str,
        max_new_tokens: int = 8192,
        temperature: float = 0.6,
        do_sample: bool = True,
    ) -> GenerationResult:
        """Generate WITHOUT early stopping (baseline comparison).

        Args:
            prompt: Input prompt.
            max_new_tokens: Maximum new tokens.
            temperature: Sampling temperature.
            do_sample: Whether to sample.

        Returns:
            GenerationResult (stopped_early will be False).
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            do_sample=do_sample,
        )

        full_text = self.tokenizer.decode(
            outputs[0, input_len:], skip_special_tokens=False
        )

        # Count thinking tokens
        if "<think>" in full_text and "</think>" in full_text:
            think_part = full_text.split("</think>")[0]
            think_tokens = len(self.tokenizer.encode(think_part, add_special_tokens=False))
        else:
            think_tokens = 0

        answer_text = full_text.split("</think>")[-1] if "</think>" in full_text else full_text
        answer = extract_answer(answer_text, self.task_type)
        total_tokens = outputs.shape[1] - input_len

        return GenerationResult(
            text=full_text,
            answer=answer,
            thinking_tokens=think_tokens,
            total_tokens=total_tokens,
            stopped_early=False,
        )
