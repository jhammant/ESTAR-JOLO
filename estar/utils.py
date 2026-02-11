"""Utility functions: answer extraction, token-to-bucket mapping, helpers."""

from __future__ import annotations

import re
from typing import Optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedTokenizer


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(text: str, task_type: str = "open") -> Optional[str]:
    """Extract the final answer from model output.

    Args:
        text: The model's generated text (after </think>).
        task_type: "open" for math/free-form, "closed" for multiple-choice.

    Returns:
        Extracted answer string, or None if not found.
    """
    if task_type == "closed":
        # Multiple-choice: look for a single letter A-E
        # Try boxed first
        m = re.search(r"\\boxed\{([A-Ea-e])\}", text)
        if m:
            return m.group(1).upper()
        # Look for "answer is (X)" pattern
        m = re.search(r"(?:answer|choice)\s*(?:is|:)\s*\(?([A-Ea-e])\)?", text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        # Last single capital letter
        letters = re.findall(r"\b([A-E])\b", text)
        return letters[-1] if letters else None
    else:
        # Open-ended: look for \boxed{...}
        m = re.search(r"\\boxed\{([^}]+)\}", text)
        if m:
            return m.group(1).strip()
        # Fallback: last number
        nums = re.findall(r"-?\d+(?:\.\d+)?", text)
        return nums[-1] if nums else text.strip()[-50:] if text.strip() else None


# ---------------------------------------------------------------------------
# Token-to-bucket mapping
# ---------------------------------------------------------------------------

def build_answer_buckets(
    tokenizer: "PreTrainedTokenizer",
    answer_set: list[str],
) -> dict[int, int]:
    """Build a mapping from token_id -> answer bucket index.

    For multiple-choice, answer_set might be ["A", "B", "C", "D", "E"].
    For open-ended math, answer_set might be ["0", "1", ..., "9", "-", "."].

    We map each token whose decoded text starts with or equals an answer
    surface form to the corresponding bucket index.

    Args:
        tokenizer: HuggingFace tokenizer.
        answer_set: List of answer surface forms.

    Returns:
        Dict mapping token_id to bucket index (0-indexed).
    """
    token_to_bucket: dict[int, int] = {}
    vocab = tokenizer.get_vocab()

    for token_str, token_id in vocab.items():
        decoded = tokenizer.decode([token_id]).strip().upper()
        for bucket_idx, ans in enumerate(answer_set):
            ans_upper = ans.strip().upper()
            if decoded == ans_upper or decoded.startswith(ans_upper):
                token_to_bucket[token_id] = bucket_idx
                break

    return token_to_bucket


def default_answer_set(task_type: str = "open") -> list[str]:
    """Return default answer buckets for a task type.

    Args:
        task_type: "open" or "closed".

    Returns:
        List of answer surface forms.
    """
    if task_type == "closed":
        return ["A", "B", "C", "D", "E"]
    else:
        # For math: digits plus common symbols
        return ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def inject_think_end(
    input_ids: "torch.LongTensor",
    tokenizer: "PreTrainedTokenizer",
    position: int,
) -> "torch.LongTensor":
    """Insert </think> token at the given position in input_ids.

    Args:
        input_ids: Tensor of shape (1, seq_len).
        tokenizer: Tokenizer with </think> in vocabulary.
        position: Position to insert at.

    Returns:
        New input_ids with </think> inserted.
    """
    import torch
    end_think_ids = tokenizer.encode("</think>", add_special_tokens=False)
    end_think_tensor = torch.tensor([end_think_ids], device=input_ids.device)

    return torch.cat([
        input_ids[:, :position],
        end_think_tensor,
        input_ids[:, position:],
    ], dim=1)


def normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison."""
    if answer is None:
        return ""
    answer = answer.strip().lower()
    # Remove LaTeX formatting
    answer = re.sub(r"\\text\{([^}]*)\}", r"\1", answer)
    answer = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", answer)
    answer = re.sub(r"\\,", "", answer)  # thin space
    answer = re.sub(r"[\\{}\s$]", "", answer)
    # Remove trailing units/symbols
    answer = re.sub(r"(cm|km|m|kg|g|s|%|degrees?)$", "", answer)
    # Remove variable assignments like "x="
    answer = re.sub(r"^[a-z]\s*=\s*", "", answer)
    return answer


def _extract_number(s: str) -> Optional[float]:
    """Try to extract a numeric value from a string."""
    s = normalize_answer(s)
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def answers_match(a: Optional[str], b: Optional[str]) -> bool:
    """Check if two answers match after normalization."""
    if a is None or b is None:
        return False
    na, nb = normalize_answer(a), normalize_answer(b)
    if na == nb:
        return True
    # Try numeric comparison
    fa, fb = _extract_number(a), _extract_number(b)
    if fa is not None and fb is not None:
        return abs(fa - fb) < 1e-4 * max(abs(fa), abs(fb), 1)
    return False


def get_device(requested: str = "auto") -> str:
    """Determine the best available device.

    Args:
        requested: "auto", "cuda", "mps", or "cpu".

    Returns:
        Device string suitable for torch.
    """
    if requested != "auto":
        return requested

    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
