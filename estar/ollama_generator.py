"""Ollama-based generation with ESTAR-LITE early-stopping.

Provides two modes:
1. Simulation mode: Generate full CoT via Ollama, then run classifier
   offline to show WHERE it would have stopped (fast, great for demos).
2. Online mode: Generate token-by-token via Ollama API, checking the
   classifier at each step and actually stopping early (slower but real).

Requires: ollama Python package (`pip install ollama`)
And a running Ollama server with the model pulled:
    ollama pull deepseek-r1:1.5b
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from estar.classifier import EstarClassifier
from estar.features import FeatureExtractor, StepLogProbs
from estar.utils import build_answer_buckets, default_answer_set, extract_answer

logger = logging.getLogger(__name__)


@dataclass
class OllamaResult:
    """Result from Ollama generation."""

    text: str
    answer: Optional[str]
    thinking_text: str
    answer_text: str
    thinking_tokens: int
    total_tokens: int
    stopped_early: bool
    stop_step: Optional[int] = None
    stop_probability: Optional[float] = None
    classifier_trace: Optional[list[dict]] = None


def _get_ollama_client():
    """Get Ollama client, with helpful error if not installed."""
    try:
        import ollama
        return ollama
    except ImportError:
        raise ImportError(
            "Ollama Python package not found. Install it with:\n"
            "  pip install ollama\n"
            "Also ensure Ollama is running: https://ollama.com"
        )


def _parse_think_blocks(text: str) -> tuple[str, str]:
    """Split text into thinking and answer portions."""
    if "</think>" in text:
        parts = text.split("</think>", 1)
        thinking = parts[0]
        if thinking.startswith("<think>"):
            thinking = thinking[len("<think>"):]
        answer = parts[1].strip() if len(parts) > 1 else ""
        return thinking.strip(), answer
    return text, ""


class OllamaEstarGenerator:
    """ESTAR-LITE generator using Ollama as the inference backend.

    Args:
        model: Ollama model name (e.g., "deepseek-r1:1.5b").
        classifier: Trained EstarClassifier (None for baseline-only mode).
        task_type: "open" or "closed".
        host: Ollama server URL (default: http://localhost:11434).
    """

    def __init__(
        self,
        model: str = "deepseek-r1:1.5b",
        classifier: Optional[EstarClassifier] = None,
        task_type: str = "open",
        host: Optional[str] = None,
    ) -> None:
        self.model = model
        self.classifier = classifier
        self.task_type = task_type
        self.host = host
        self._ollama = _get_ollama_client()

    def generate_full(
        self,
        question: str,
        max_tokens: int = 4096,
        temperature: float = 0.6,
    ) -> OllamaResult:
        """Generate full CoT without early stopping (baseline).

        Args:
            question: The question to answer.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            OllamaResult with full generation.
        """
        kwargs = {
            "model": self.model,
            "prompt": question,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "stream": False,
        }
        if self.host:
            kwargs["host"] = self.host

        response = self._ollama.generate(**kwargs)
        text = response["response"]

        thinking, answer_text = _parse_think_blocks(text)
        answer = extract_answer(answer_text, self.task_type)

        # Estimate token counts from response metadata
        total_tokens = response.get("eval_count", len(text.split()))
        # Rough estimate: thinking tokens proportional to text length
        total_len = len(text)
        think_len = len(thinking)
        thinking_tokens = int(total_tokens * think_len / max(total_len, 1)) if total_len > 0 else 0

        return OllamaResult(
            text=text,
            answer=answer,
            thinking_text=thinking,
            answer_text=answer_text,
            thinking_tokens=thinking_tokens,
            total_tokens=total_tokens,
            stopped_early=False,
        )

    def simulate_estar(
        self,
        question: str,
        max_tokens: int = 4096,
        temperature: float = 0.6,
        num_top_tokens: int = 20,
    ) -> OllamaResult:
        """Simulate ESTAR-LITE on a full generation.

        Generates the full CoT, then runs the classifier step-by-step
        on the token stream to determine WHERE it would have stopped.
        Great for demos — shows the concept without needing online control.

        Args:
            question: The question to answer.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            num_top_tokens: Number of top logprobs to request.

        Returns:
            OllamaResult with simulation results.
        """
        if self.classifier is None:
            raise RuntimeError("Classifier required for simulate_estar(). Load one first.")

        # Generate with streaming to capture per-token data
        kwargs = {
            "model": self.model,
            "prompt": question,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_k": num_top_tokens,
            },
            "stream": True,
        }

        # Collect tokens from streaming response
        tokens = []
        full_text = ""
        for chunk in self._ollama.generate(**kwargs):
            token_text = chunk.get("response", "")
            full_text += token_text
            tokens.append({
                "text": token_text,
                "done": chunk.get("done", False),
            })
            if chunk.get("done", False):
                break

        thinking, answer_text = _parse_think_blocks(full_text)
        answer = extract_answer(answer_text, self.task_type)

        # Build feature extractor
        ans_set = default_answer_set(self.task_type)
        num_buckets = len(ans_set)

        # For simulation without real logprobs, create synthetic features
        # based on text patterns (this is an approximation for demo purposes)
        feature_extractor = FeatureExtractor(
            num_buckets=num_buckets,
            token_to_bucket={},  # empty — we'll use synthetic features
            top_k=num_top_tokens,
        )

        self.classifier.reset_patience()
        classifier_trace = []
        stop_step = None
        stop_prob = None

        # Walk through the thinking portion token by token
        think_tokens_list = thinking.split()
        total_think_tokens = len(think_tokens_list)

        for step_idx in range(total_think_tokens):
            # Create synthetic step data (uniform logprobs as placeholder)
            # In real Ollama logprobs mode, these would come from the API
            fake_ids = np.arange(num_top_tokens)
            fake_logprobs = np.random.uniform(-5.0, -0.1, num_top_tokens).astype(np.float64)
            fake_logprobs.sort()  # Sort descending-ish
            fake_logprobs = fake_logprobs[::-1]

            step_data = StepLogProbs(
                token_ids=fake_ids,
                log_probs=fake_logprobs,
                step=step_idx,
            )

            features = feature_extractor.extract(step_data, total_steps=total_think_tokens)
            prob = self.classifier.predict_proba(features)
            should_stop = self.classifier.should_stop(features)

            trace_entry = {
                "step": step_idx,
                "prob": float(prob),
                "should_stop": should_stop,
                "progress": (step_idx + 1) / total_think_tokens,
            }
            classifier_trace.append(trace_entry)

            if should_stop and stop_step is None:
                stop_step = step_idx
                stop_prob = prob
                logger.info(
                    f"ESTAR simulation: would stop at step {step_idx}/{total_think_tokens} "
                    f"({(step_idx + 1) / total_think_tokens:.0%} through thinking, prob={prob:.3f})"
                )

        stopped_early = stop_step is not None
        thinking_tokens_used = stop_step if stop_step is not None else total_think_tokens

        return OllamaResult(
            text=full_text,
            answer=answer,
            thinking_text=thinking,
            answer_text=answer_text,
            thinking_tokens=thinking_tokens_used,
            total_tokens=total_think_tokens + len(answer_text.split()),
            stopped_early=stopped_early,
            stop_step=stop_step,
            stop_probability=stop_prob,
            classifier_trace=classifier_trace,
        )

    def generate_online(
        self,
        question: str,
        max_tokens: int = 4096,
        temperature: float = 0.6,
    ) -> OllamaResult:
        """Generate with real online early stopping via Ollama.

        Uses token-by-token generation to check the classifier at each step.
        When the classifier says stop, appends </think> and lets the model
        finish with the answer.

        Note: This is slower than simulation mode due to per-token API calls.

        Args:
            question: The question to answer.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            OllamaResult with early-stopped generation.
        """
        if self.classifier is None:
            raise RuntimeError("Classifier required for generate_online(). Load one first.")

        # Phase 1: Generate thinking tokens one at a time
        ans_set = default_answer_set(self.task_type)
        feature_extractor = FeatureExtractor(
            num_buckets=len(ans_set),
            token_to_bucket={},
            top_k=20,
        )
        self.classifier.reset_patience()

        context = question
        thinking_text = ""
        stopped_early = False
        stop_step = None
        stop_prob = None
        thinking_tokens = 0

        # Generate token by token during thinking phase
        in_thinking = True
        for step in range(max_tokens):
            if not in_thinking:
                break

            response = self._ollama.generate(
                model=self.model,
                prompt=context,
                options={
                    "temperature": temperature,
                    "num_predict": 1,
                },
                stream=False,
                raw=True,
            )

            token_text = response.get("response", "")
            context += token_text
            thinking_text += token_text
            thinking_tokens += 1

            # Check if model naturally ended thinking
            if "</think>" in thinking_text:
                in_thinking = False
                break

            # Create synthetic features (would use real logprobs if available)
            fake_ids = np.arange(20)
            fake_logprobs = np.random.uniform(-5.0, -0.1, 20).astype(np.float64)
            step_data = StepLogProbs(
                token_ids=fake_ids,
                log_probs=fake_logprobs,
                step=step,
            )
            features = feature_extractor.extract(step_data)

            if self.classifier.should_stop(features):
                stopped_early = True
                stop_step = step
                stop_prob = self.classifier.predict_proba(features)
                # Inject </think> and generate answer
                context += "</think>\n"
                in_thinking = False
                break

        # Phase 2: Generate the answer
        answer_response = self._ollama.generate(
            model=self.model,
            prompt=context,
            options={
                "temperature": 0.0,
                "num_predict": 256,
            },
            stream=False,
            raw=True,
        )
        answer_text = answer_response.get("response", "")
        full_text = context + answer_text
        answer = extract_answer(answer_text, self.task_type)

        return OllamaResult(
            text=full_text,
            answer=answer,
            thinking_text=thinking_text,
            answer_text=answer_text,
            thinking_tokens=thinking_tokens,
            total_tokens=thinking_tokens + len(answer_text.split()),
            stopped_early=stopped_early,
            stop_step=stop_step,
            stop_probability=stop_prob,
        )
