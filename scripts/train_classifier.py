#!/usr/bin/env python3
"""Train the ESTAR-LITE LightGBM classifier.

Usage:
    python scripts/train_classifier.py \
        --data data/training_data.npz \
        --output models/estar_lite \
        --threshold 0.9 \
        --patience 3
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)

from estar.classifier import EstarClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ESTAR-LITE classifier")
    parser.add_argument("--data", required=True, help="Path to training_data.npz")
    parser.add_argument("--output", default="models/estar_lite", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load data
    logger.info(f"Loading data from {args.data}")
    data = np.load(args.data, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    feature_names = list(data.get("feature_names", []))

    logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Positive rate: {y.mean():.2%}")

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_size, random_state=args.seed, stratify=y
    )
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # Train
    clf = EstarClassifier(threshold=args.threshold, patience=args.patience)
    clf.train(X_train, y_train, X_val, y_val, feature_names=feature_names or None)

    # Evaluate
    y_pred_proba = np.array([clf.predict_proba(x) for x in X_val])
    y_pred = (y_pred_proba >= args.threshold).astype(int)

    logger.info(f"\nValidation Results (threshold={args.threshold}):")
    logger.info(f"AUC-ROC: {roc_auc_score(y_val, y_pred_proba):.4f}")
    logger.info(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    logger.info(f"\n{classification_report(y_val, y_pred, target_names=['continue', 'stop'])}")

    # Feature importance
    importance = clf.feature_importance()
    if importance is not None and feature_names:
        top_k = min(10, len(feature_names))
        indices = np.argsort(importance)[::-1][:top_k]
        logger.info("Top features by importance:")
        for i in indices:
            logger.info(f"  {feature_names[i]}: {importance[i]:.1f}")

    # Save
    clf.save(args.output)
    logger.info(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
