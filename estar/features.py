"""Feature extraction for ESTAR-LITE.

Implements four feature groups from the paper (Section 4.1):
1. Instantaneous evidence — per-class probabilities from token log-probs
2. Cumulative path & stability — running evidence, flip counts, changed_prev
3. Early-stop curvature cues — slope (S_es) and second difference (H_es)
4. Token-level confidence stats — mean/var of log-probs, neg perplexity, answer length
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class StepLogProbs:
    """Top-k log-probabilities at a single decode step.

    Attributes:
        token_ids: Array of top-k token IDs, shape (k,).
        log_probs: Array of top-k log-probabilities, shape (k,).
        step: The decode step index (0-based).
    """
    token_ids: np.ndarray
    log_probs: np.ndarray
    step: int


class FeatureExtractor:
    """Online feature extractor for ESTAR-LITE.

    Computes a feature vector at each decode step from top-k token
    log-probabilities, maintaining running state for cumulative features.

    Args:
        num_buckets: Number of answer buckets (e.g., 5 for ABCDE, 10 for digits).
        token_to_bucket: Mapping from token_id -> bucket index.
        top_k: Number of top tokens to consider (default 20).
    """

    def __init__(
        self,
        num_buckets: int,
        token_to_bucket: dict[int, int],
        top_k: int = 20,
    ) -> None:
        self.num_buckets = num_buckets
        self.token_to_bucket = token_to_bucket
        self.top_k = top_k
        self.reset()

    def reset(self) -> None:
        """Reset all running state for a new sequence."""
        self._cumulative_evidence = np.zeros(self.num_buckets, dtype=np.float64)
        self._prev_winner: Optional[int] = None
        self._flip_count: int = 0
        self._es_confidence_history: list[float] = []
        self._all_log_probs: list[float] = []
        self._step_count: int = 0

    def feature_names(self) -> list[str]:
        """Return names of all features in the output vector."""
        names = []
        # Group 1: instantaneous evidence
        for i in range(self.num_buckets):
            names.append(f"inst_p_{i}")
        names.append("inst_max_p")
        names.append("inst_entropy")
        # Group 2: cumulative path & stability
        for i in range(self.num_buckets):
            names.append(f"cum_evidence_{i}")
        names.append("cum_winner")
        names.append("cum_margin")
        names.append("flips")
        names.append("changed_prev")
        # Group 3: early-stop curvature
        names.append("es_confidence")
        names.append("S_es")
        names.append("H_es")
        # Group 4: token-level confidence stats
        names.append("logprob_mean")
        names.append("logprob_var")
        names.append("neg_ppl")
        names.append("ans_len")
        names.append("step_frac")  # bonus: fractional progress if known
        return names

    @property
    def num_features(self) -> int:
        return len(self.feature_names())

    def extract(
        self,
        step_data: StepLogProbs,
        total_steps: Optional[int] = None,
    ) -> np.ndarray:
        """Extract feature vector for a single decode step.

        Args:
            step_data: Top-k log-probs at this step.
            total_steps: Total expected CoT length (for step_frac), or None.

        Returns:
            Feature vector as 1-D numpy array.
        """
        self._step_count += 1
        features: list[float] = []

        # ── Group 1: Instantaneous evidence ──────────────────────────
        inst_scores = self._compute_instantaneous_evidence(step_data)
        inst_p = self._softmax(inst_scores)  # per-class probability
        features.extend(inst_p.tolist())
        features.append(float(np.max(inst_p)))
        # Entropy
        eps = 1e-10
        entropy = -float(np.sum(inst_p * np.log(inst_p + eps)))
        features.append(entropy)

        # ── Group 2: Cumulative path & stability ─────────────────────
        # Accumulate log instantaneous probabilities
        safe_inst_p = np.clip(inst_p, eps, None)
        self._cumulative_evidence += np.log(safe_inst_p)

        features.extend(self._cumulative_evidence.tolist())

        current_winner = int(np.argmax(self._cumulative_evidence))
        features.append(float(current_winner))

        # Margin: difference between top two cumulative scores
        sorted_cum = np.sort(self._cumulative_evidence)[::-1]
        margin = sorted_cum[0] - sorted_cum[1] if len(sorted_cum) > 1 else sorted_cum[0]
        features.append(float(margin))

        # Flip tracking
        changed = 0
        if self._prev_winner is not None and current_winner != self._prev_winner:
            self._flip_count += 1
            changed = 1
        self._prev_winner = current_winner

        features.append(float(self._flip_count))
        features.append(float(changed))

        # ── Group 3: Early-stop curvature cues ───────────────────────
        es_conf = float(np.max(inst_p))  # early-stop confidence = max instantaneous prob
        self._es_confidence_history.append(es_conf)
        features.append(es_conf)

        # Slope S_es(t) = es_conf(t) - es_conf(t-1)
        if len(self._es_confidence_history) >= 2:
            s_es = self._es_confidence_history[-1] - self._es_confidence_history[-2]
        else:
            s_es = 0.0
        features.append(s_es)

        # Second difference H_es(t) = S_es(t) - S_es(t-1)
        if len(self._es_confidence_history) >= 3:
            s_prev = self._es_confidence_history[-2] - self._es_confidence_history[-3]
            h_es = s_es - s_prev
        else:
            h_es = 0.0
        features.append(h_es)

        # ── Group 4: Token-level confidence stats ────────────────────
        lp = step_data.log_probs[:self.top_k]
        self._all_log_probs.extend(lp.tolist())

        logprob_mean = float(np.mean(lp))
        logprob_var = float(np.var(lp))
        neg_ppl = float(-np.mean(lp))  # negative perplexity proxy (lower = more peaked → flip sign)
        # Actually paper says "higher means more peaked" so: neg_ppl = mean(log_prob) (less negative = more peaked)
        neg_ppl = float(np.mean(lp))  # higher (closer to 0) = more confident
        ans_len = float(self._step_count)

        features.append(logprob_mean)
        features.append(logprob_var)
        features.append(neg_ppl)
        features.append(ans_len)

        # Step fraction
        step_frac = float(self._step_count / total_steps) if total_steps else 0.0
        features.append(step_frac)

        return np.array(features, dtype=np.float32)

    def _compute_instantaneous_evidence(self, step_data: StepLogProbs) -> np.ndarray:
        """Compute log-sum-exp evidence per answer bucket.

        Maps top-k tokens to answer buckets, aggregates via log-sum-exp.
        """
        bucket_logprobs: dict[int, list[float]] = {i: [] for i in range(self.num_buckets)}

        for tid, lp in zip(step_data.token_ids, step_data.log_probs):
            tid_int = int(tid)
            if tid_int in self.token_to_bucket:
                bucket_idx = self.token_to_bucket[tid_int]
                bucket_logprobs[bucket_idx].append(float(lp))

        scores = np.full(self.num_buckets, -30.0, dtype=np.float64)  # very low default
        for bucket_idx, lps in bucket_logprobs.items():
            if lps:
                # log-sum-exp
                arr = np.array(lps)
                max_lp = np.max(arr)
                scores[bucket_idx] = max_lp + np.log(np.sum(np.exp(arr - max_lp)))

        return scores

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        e = np.exp(x - np.max(x))
        return e / e.sum()
