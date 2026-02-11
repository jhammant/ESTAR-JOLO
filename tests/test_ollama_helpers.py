"""Tests for Ollama logprob helper functions."""

import numpy as np
import pytest

from estar.ollama_generator import (
    _build_ollama_token_to_bucket,
    _make_fake_step_data,
    _ollama_logprobs_to_step_data,
    _token_text_to_id,
)


class TestTokenTextToId:
    def test_deterministic(self) -> None:
        assert _token_text_to_id("hello") == _token_text_to_id("hello")

    def test_different_tokens_different_ids(self) -> None:
        assert _token_text_to_id("0") != _token_text_to_id("1")

    def test_returns_positive_int(self) -> None:
        tid = _token_text_to_id("test")
        assert isinstance(tid, int)
        assert tid >= 0

    def test_empty_string(self) -> None:
        tid = _token_text_to_id("")
        assert isinstance(tid, int)


class TestBuildOllamaTokenToBucket:
    def test_digit_buckets(self) -> None:
        ans_set = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        tb = _build_ollama_token_to_bucket(ans_set)
        # Each digit should map to its bucket index
        for i, digit in enumerate(ans_set):
            tid = _token_text_to_id(digit)
            assert tb[tid] == i, f"Digit '{digit}' mapped to bucket {tb.get(tid)} instead of {i}"

    def test_letter_buckets(self) -> None:
        ans_set = ["A", "B", "C", "D", "E"]
        tb = _build_ollama_token_to_bucket(ans_set)
        # Uppercase form should be mapped
        for i, letter in enumerate(ans_set):
            tid = _token_text_to_id(letter)
            assert tb[tid] == i

    def test_space_prefixed_variants(self) -> None:
        ans_set = ["0", "1"]
        tb = _build_ollama_token_to_bucket(ans_set)
        # " 0" variant should also map to bucket 0
        tid_space = _token_text_to_id(" 0")
        assert tid_space in tb
        assert tb[tid_space] == 0

    def test_returns_dict(self) -> None:
        tb = _build_ollama_token_to_bucket(["A", "B"])
        assert isinstance(tb, dict)
        assert len(tb) > 0


class TestOllamaLogprobsToStepData:
    def test_basic_conversion(self) -> None:
        entry = {
            "top_logprobs": [
                {"token": "5", "logprob": -0.1},
                {"token": "3", "logprob": -1.5},
                {"token": "7", "logprob": -3.0},
            ]
        }
        sd = _ollama_logprobs_to_step_data(entry, step=0, top_k=5)
        assert sd.token_ids.shape == (5,)
        assert sd.log_probs.shape == (5,)
        assert sd.step == 0
        # First 3 should have real values
        assert sd.log_probs[0] == pytest.approx(-0.1)
        assert sd.log_probs[1] == pytest.approx(-1.5)
        assert sd.log_probs[2] == pytest.approx(-3.0)
        # Last 2 should be padded
        assert sd.log_probs[3] == pytest.approx(-30.0)
        assert sd.log_probs[4] == pytest.approx(-30.0)

    def test_truncation(self) -> None:
        entry = {
            "top_logprobs": [
                {"token": str(i), "logprob": -float(i)} for i in range(10)
            ]
        }
        sd = _ollama_logprobs_to_step_data(entry, step=5, top_k=3)
        assert sd.token_ids.shape == (3,)
        assert sd.log_probs.shape == (3,)
        assert sd.step == 5

    def test_fallback_to_single_token(self) -> None:
        entry = {"token": "hello", "logprob": -2.5}
        sd = _ollama_logprobs_to_step_data(entry, step=0, top_k=5)
        assert sd.token_ids.shape == (5,)
        assert sd.log_probs[0] == pytest.approx(-2.5)

    def test_empty_top_logprobs(self) -> None:
        entry = {"top_logprobs": [], "token": "x", "logprob": -1.0}
        sd = _ollama_logprobs_to_step_data(entry, step=0, top_k=3)
        assert sd.token_ids.shape == (3,)
        assert sd.log_probs[0] == pytest.approx(-1.0)

    def test_hashed_ids_match_bucket_map(self) -> None:
        """Verify that IDs from logprob conversion match the bucket map."""
        ans_set = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        tb = _build_ollama_token_to_bucket(ans_set)
        entry = {
            "top_logprobs": [
                {"token": "5", "logprob": -0.1},
                {"token": "3", "logprob": -1.5},
            ]
        }
        sd = _ollama_logprobs_to_step_data(entry, step=0, top_k=5)
        # The ID for token "5" should be in the bucket map
        assert int(sd.token_ids[0]) in tb
        assert tb[int(sd.token_ids[0])] == 5
        assert int(sd.token_ids[1]) in tb
        assert tb[int(sd.token_ids[1])] == 3


class TestMakeFakeStepData:
    def test_shape(self) -> None:
        sd = _make_fake_step_data(step=0, top_k=20)
        assert sd.token_ids.shape == (20,)
        assert sd.log_probs.shape == (20,)
        assert sd.step == 0

    def test_logprobs_sorted_descending(self) -> None:
        sd = _make_fake_step_data(step=5, top_k=10)
        # Should be sorted descending (highest first)
        for i in range(len(sd.log_probs) - 1):
            assert sd.log_probs[i] >= sd.log_probs[i + 1]

    def test_custom_top_k(self) -> None:
        sd = _make_fake_step_data(step=0, top_k=5)
        assert sd.token_ids.shape == (5,)
        assert sd.log_probs.shape == (5,)

    def test_all_finite(self) -> None:
        sd = _make_fake_step_data(step=0, top_k=20)
        assert np.all(np.isfinite(sd.log_probs))
