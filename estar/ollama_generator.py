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

import hashlib
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


def _token_text_to_id(token_text: str) -> int:
    """Hash a token string to a stable synthetic integer ID.

    Ollama logprobs use token strings, not integer IDs. We hash the text
    to produce a deterministic int that can be used in token_to_bucket maps.
    """
    return int(hashlib.md5(token_text.encode()).hexdigest()[:8], 16)


def _ollama_logprobs_to_step_data(
    logprob_entry: dict,
    step: int,
    top_k: int = 20,
) -> StepLogProbs:
    """Convert an Ollama logprob entry to StepLogProbs.

    Ollama returns logprobs as a list of {token: str, logprob: float} dicts
    (via the ``Logprob`` object). This converts them to the array format
    expected by FeatureExtractor, hashing token text to synthetic int IDs.

    Pads with dummy entries if fewer than top_k tokens are available;
    truncates if more.
    """
    top_logprobs = logprob_entry.get("top_logprobs", [])
    if not top_logprobs:
        token_text = logprob_entry.get("token", "")
        lp_val = logprob_entry.get("logprob", -5.0)
        top_logprobs = [{"token": token_text, "logprob": lp_val}]

    ids = []
    lps = []
    for entry in top_logprobs[:top_k]:
        tok = entry.get("token", "")
        lp = entry.get("logprob", -5.0)
        ids.append(_token_text_to_id(tok))
        lps.append(float(lp))

    # Pad to top_k if needed
    while len(ids) < top_k:
        ids.append(0)
        lps.append(-30.0)

    return StepLogProbs(
        token_ids=np.array(ids, dtype=np.int64),
        log_probs=np.array(lps, dtype=np.float64),
        step=step,
    )


def _build_ollama_token_to_bucket(answer_set: list[str]) -> dict[int, int]:
    """Build bucket mapping from hashed token strings for answer tokens.

    For each answer surface form (e.g. "0"-"9" or "A"-"E"), hashes the
    token text to produce the same synthetic ID that ``_token_text_to_id``
    would generate, enabling FeatureExtractor bucket lookups.
    """
    token_to_bucket: dict[int, int] = {}
    for bucket_idx, ans in enumerate(answer_set):
        # Map common surface forms for each answer token
        for variant in [ans, ans.lower(), ans.upper(), f" {ans}", f" {ans.lower()}", f" {ans.upper()}"]:
            tid = _token_text_to_id(variant)
            token_to_bucket[tid] = bucket_idx
    return token_to_bucket


def _make_fake_step_data(step: int, top_k: int = 20) -> StepLogProbs:
    """Create synthetic step data as fallback when real logprobs are unavailable."""
    fake_ids = np.arange(top_k, dtype=np.int64)
    fake_logprobs = np.random.uniform(-5.0, -0.1, top_k).astype(np.float64)
    fake_logprobs.sort()
    fake_logprobs = fake_logprobs[::-1]
    return StepLogProbs(token_ids=fake_ids, log_probs=fake_logprobs, step=step)


class OllamaEstarGenerator:
    """ESTAR-LITE generator using Ollama as the inference backend.

    Args:
        model: Ollama model name (e.g., "deepseek-r1:1.5b").
        classifier: Trained EstarClassifier (None for baseline-only mode).
        task_type: "open" or "closed".
        host: Ollama server URL (default: http://localhost:11434).
        use_logprobs: Request real logprobs from Ollama. Disable if your
            Ollama version doesn't support ``logprobs``/``top_logprobs``.
    """

    def __init__(
        self,
        model: str = "deepseek-r1:1.5b",
        classifier: Optional[EstarClassifier] = None,
        task_type: str = "open",
        host: Optional[str] = None,
        use_logprobs: bool = True,
    ) -> None:
        self.model = model
        self.classifier = classifier
        self.task_type = task_type
        self.host = host
        self.use_logprobs = use_logprobs
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
        if self.use_logprobs:
            kwargs["options"]["logprobs"] = True
            kwargs["options"]["top_logprobs"] = num_top_tokens

        # Collect tokens from streaming response
        tokens = []
        full_text = ""
        has_real_logprobs = False
        for chunk in self._ollama.generate(**kwargs):
            token_text = chunk.get("response", "")
            full_text += token_text
            token_entry = {
                "text": token_text,
                "done": chunk.get("done", False),
            }
            # Capture logprobs if available
            chunk_logprobs = chunk.get("logprobs")
            if chunk_logprobs:
                token_entry["logprobs"] = chunk_logprobs
                has_real_logprobs = True
            tokens.append(token_entry)
            if chunk.get("done", False):
                break

        if has_real_logprobs:
            logger.info("Using real logprobs from Ollama for simulation")
        else:
            logger.info("Ollama did not return logprobs — falling back to synthetic")

        thinking, answer_text = _parse_think_blocks(full_text)
        answer = extract_answer(answer_text, self.task_type)

        # Build feature extractor
        ans_set = default_answer_set(self.task_type)
        num_buckets = len(ans_set)

        if has_real_logprobs:
            token_to_bucket = _build_ollama_token_to_bucket(ans_set)
        else:
            token_to_bucket = {}

        feature_extractor = FeatureExtractor(
            num_buckets=num_buckets,
            token_to_bucket=token_to_bucket,
            top_k=num_top_tokens,
        )

        self.classifier.reset_patience()
        classifier_trace = []
        stop_step = None
        stop_prob = None

        # Walk through the thinking portion token by token
        # Count how many tokens fall in the thinking section
        think_char_count = 0
        think_token_indices = []
        for i, tok in enumerate(tokens):
            if tok["done"]:
                break
            think_char_count += len(tok["text"])
            # Check if we've passed the thinking portion
            if think_char_count > len(thinking) + len("<think>"):
                break
            think_token_indices.append(i)

        total_think_tokens = len(think_token_indices)
        if total_think_tokens == 0:
            total_think_tokens = max(1, len(thinking.split()))
            think_token_indices = list(range(total_think_tokens))

        for step_idx, tok_idx in enumerate(think_token_indices):
            tok = tokens[tok_idx] if tok_idx < len(tokens) else {}

            # Use real logprobs if available, otherwise fake
            if has_real_logprobs and "logprobs" in tok:
                step_data = _ollama_logprobs_to_step_data(
                    tok["logprobs"], step=step_idx, top_k=num_top_tokens
                )
            else:
                step_data = _make_fake_step_data(step_idx, top_k=num_top_tokens)

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

        num_top = 20
        ans_set = default_answer_set(self.task_type)

        if self.use_logprobs:
            token_to_bucket = _build_ollama_token_to_bucket(ans_set)
        else:
            token_to_bucket = {}

        feature_extractor = FeatureExtractor(
            num_buckets=len(ans_set),
            token_to_bucket=token_to_bucket,
            top_k=num_top,
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

            gen_kwargs: dict = {
                "model": self.model,
                "prompt": context,
                "options": {
                    "temperature": temperature,
                    "num_predict": 1,
                },
                "stream": False,
                "raw": True,
            }
            if self.use_logprobs:
                gen_kwargs["options"]["logprobs"] = True
                gen_kwargs["options"]["top_logprobs"] = num_top

            response = self._ollama.generate(**gen_kwargs)

            token_text = response.get("response", "")
            context += token_text
            thinking_text += token_text
            thinking_tokens += 1

            # Check if model naturally ended thinking
            if "</think>" in thinking_text:
                in_thinking = False
                break

            # Extract logprobs — real if available, fake otherwise
            resp_logprobs = response.get("logprobs")
            if self.use_logprobs and resp_logprobs:
                step_data = _ollama_logprobs_to_step_data(
                    resp_logprobs, step=step, top_k=num_top
                )
            else:
                step_data = _make_fake_step_data(step, top_k=num_top)

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
