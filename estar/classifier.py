"""LightGBM classifier wrapper for ESTAR-LITE.

Trains on (feature_vector, should_stop) pairs and predicts stop probability.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import lightgbm as lgb
import numpy as np


class EstarClassifier:
    """LightGBM-based early-stopping classifier.

    Predicts P(should_stop | features) at each decode step.
    Stop when prediction exceeds threshold τ for `patience` consecutive steps.

    Args:
        threshold: Decision threshold τ (default 0.9 from paper).
        patience: Number of consecutive steps above threshold before stopping.
        lgb_params: LightGBM hyperparameters (defaults from paper).
    """

    DEFAULT_PARAMS: dict = {
        "objective": "binary",
        "metric": "binary_logloss",
        "n_estimators": 400,
        "num_leaves": 63,
        "learning_rate": 0.07,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "verbose": -1,
    }

    def __init__(
        self,
        threshold: float = 0.9,
        patience: int = 3,
        lgb_params: Optional[dict] = None,
    ) -> None:
        self.threshold = threshold
        self.patience = patience
        self.lgb_params = {**self.DEFAULT_PARAMS, **(lgb_params or {})}
        self.model: Optional[lgb.LGBMClassifier] = None
        self._consecutive_above: int = 0

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[list[str]] = None,
    ) -> "EstarClassifier":
        """Train the classifier.

        Args:
            X: Training features, shape (n_samples, n_features).
            y: Binary labels, shape (n_samples,). 1 = safe to stop.
            X_val: Validation features (optional).
            y_val: Validation labels (optional).
            feature_names: Feature names for interpretability.

        Returns:
            self
        """
        callbacks = []
        if X_val is not None and y_val is not None:
            callbacks.append(lgb.early_stopping(50, verbose=False))

        self.model = lgb.LGBMClassifier(**self.lgb_params)
        eval_set = [(X_val, y_val)] if X_val is not None else None

        self.model.fit(
            X, y,
            eval_set=eval_set,
            callbacks=callbacks if eval_set else None,
            feature_name=feature_names or "auto",
        )
        return self

    def predict_proba(self, features: np.ndarray) -> float:
        """Predict stop probability for a single feature vector.

        Args:
            features: Feature vector, shape (n_features,) or (1, n_features).

        Returns:
            Probability that reasoning should stop.
        """
        if self.model is None:
            raise RuntimeError("Classifier not trained. Call train() or load() first.")

        if features.ndim == 1:
            features = features.reshape(1, -1)

        return float(self.model.predict_proba(features)[0, 1])

    def should_stop(self, features: np.ndarray) -> bool:
        """Check if reasoning should stop (with patience).

        Args:
            features: Feature vector at current step.

        Returns:
            True if should stop.
        """
        prob = self.predict_proba(features)
        if prob >= self.threshold:
            self._consecutive_above += 1
        else:
            self._consecutive_above = 0

        return self._consecutive_above >= self.patience

    def reset_patience(self) -> None:
        """Reset the consecutive-above counter for a new sequence."""
        self._consecutive_above = 0

    def feature_importance(self, importance_type: str = "gain") -> Optional[np.ndarray]:
        """Get feature importance from the trained model."""
        if self.model is None:
            return None
        return self.model.feature_importances_

    def save(self, path: Union[str, Path]) -> None:
        """Save model to disk.

        Args:
            path: Directory to save model files.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            self.model.booster_.save_model(str(path / "model.txt"))

        meta = {
            "threshold": self.threshold,
            "patience": self.patience,
            "lgb_params": self.lgb_params,
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "EstarClassifier":
        """Load a saved classifier.

        Args:
            path: Directory containing model files.

        Returns:
            Loaded EstarClassifier instance.
        """
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())

        obj = cls(
            threshold=meta["threshold"],
            patience=meta["patience"],
            lgb_params=meta["lgb_params"],
        )

        booster = lgb.Booster(model_file=str(path / "model.txt"))
        obj.model = lgb.LGBMClassifier(**meta["lgb_params"])
        obj.model._Booster = booster
        obj.model.fitted_ = True
        obj.model._n_features = booster.num_feature()
        obj.model._n_classes = 2
        obj.model.classes_ = np.array([0, 1])

        return obj
