"""Tests for feature extraction."""

import numpy as np
import pytest

from estar.features import FeatureExtractor, StepLogProbs


@pytest.fixture
def extractor() -> FeatureExtractor:
    """Create a simple feature extractor with 5 buckets."""
    token_to_bucket = {10: 0, 11: 1, 12: 2, 13: 3, 14: 4}
    return FeatureExtractor(num_buckets=5, token_to_bucket=token_to_bucket, top_k=5)


def make_step(token_ids: list[int], log_probs: list[float], step: int) -> StepLogProbs:
    return StepLogProbs(
        token_ids=np.array(token_ids),
        log_probs=np.array(log_probs),
        step=step,
    )


class TestFeatureExtractor:
    def test_output_shape(self, extractor: FeatureExtractor) -> None:
        step = make_step([10, 11, 12, 13, 14], [-0.1, -1.0, -2.0, -3.0, -4.0], 0)
        features = extractor.extract(step)
        assert features.shape == (extractor.num_features,)
        assert features.dtype == np.float32

    def test_feature_names_match_length(self, extractor: FeatureExtractor) -> None:
        step = make_step([10, 11, 12, 13, 14], [-0.1, -1.0, -2.0, -3.0, -4.0], 0)
        features = extractor.extract(step)
        assert len(extractor.feature_names()) == len(features)

    def test_reset_clears_state(self, extractor: FeatureExtractor) -> None:
        step = make_step([10, 11, 12, 13, 14], [-0.1, -1.0, -2.0, -3.0, -4.0], 0)
        extractor.extract(step)
        extractor.extract(step)
        extractor.reset()
        assert extractor._step_count == 0
        assert extractor._flip_count == 0

    def test_cumulative_evidence_accumulates(self, extractor: FeatureExtractor) -> None:
        step1 = make_step([10, 11, 12, 13, 14], [-0.1, -1.0, -2.0, -3.0, -4.0], 0)
        step2 = make_step([10, 11, 12, 13, 14], [-0.1, -1.0, -2.0, -3.0, -4.0], 1)
        extractor.extract(step1)
        f2 = extractor.extract(step2)
        # Cumulative evidence for bucket 0 should be larger (more negative but less so)
        # than after just one step
        assert extractor._step_count == 2

    def test_flip_count_increases(self, extractor: FeatureExtractor) -> None:
        # Step 1: bucket 0 wins
        step1 = make_step([10, 11, 12, 13, 14], [-0.1, -5.0, -5.0, -5.0, -5.0], 0)
        extractor.extract(step1)
        assert extractor._flip_count == 0

        # Step 2: bucket 1 wins (flip!)
        step2 = make_step([10, 11, 12, 13, 14], [-5.0, -0.1, -5.0, -5.0, -5.0], 1)
        extractor.extract(step2)
        assert extractor._flip_count == 1

    def test_unmapped_tokens_get_default(self, extractor: FeatureExtractor) -> None:
        # Token IDs 99, 100 are not in token_to_bucket
        step = make_step([99, 100, 10, 11, 12], [-0.5, -0.5, -1.0, -2.0, -3.0], 0)
        features = extractor.extract(step)
        assert features.shape == (extractor.num_features,)

    def test_all_features_finite(self, extractor: FeatureExtractor) -> None:
        for i in range(10):
            step = make_step(
                [10, 11, 12, 13, 14],
                [-0.1 - i * 0.1, -1.0, -2.0, -3.0, -4.0],
                i,
            )
            features = extractor.extract(step)
            assert np.all(np.isfinite(features)), f"Non-finite at step {i}: {features}"
