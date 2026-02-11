"""ESTAR-LITE: Early-Stopping Token-Aware Reasoning for Efficient LLM Inference.

Independent implementation based on arXiv:2602.10004, not affiliated with the authors.
"""

__version__ = "0.1.0"

from estar.features import FeatureExtractor
from estar.classifier import EstarClassifier


def __getattr__(name: str):
    if name == "EstarGenerator":
        from estar.generator import EstarGenerator
        return EstarGenerator
    raise AttributeError(f"module 'estar' has no attribute {name}")


__all__ = ["FeatureExtractor", "EstarClassifier", "EstarGenerator"]
