#!/usr/bin/env python3
"""Tune ESTAR-LITE threshold (tau) and patience via grid sweep.

Trains one LightGBM model, then sweeps inference-time stopping criteria
(threshold x patience) on validation data to find the best accuracy vs
token-savings trade-off.

Usage:
    # From pre-generated .npz training data:
    python scripts/tune_threshold.py --data data/training_data.npz

    # From estar_summary.json results (regenerates features):
    python scripts/tune_threshold.py --results results/estar_summary.json

    # Custom grid:
    python scripts/tune_threshold.py --data data/training_data.npz \
        --thresholds 0.5 0.7 0.9 0.95 --patience-values 1 2 3 5
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from estar.classifier import EstarClassifier

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_training_data(
    data_path: str | None = None,
    results_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray | None]:
    """Load features and labels from .npz or regenerate from results JSON.

    Returns:
        (X, y, feature_names, sequence_ids) where sequence_ids groups
        samples by source problem (None if unavailable).
    """
    if data_path:
        data = np.load(data_path, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        feature_names = list(data.get("feature_names", []))
        # Try to load sequence IDs for per-problem grouping
        seq_ids = data["sequence_ids"] if "sequence_ids" in data else None
        return X, y, feature_names, seq_ids

    if results_path:
        return _regenerate_from_results(results_path)

    raise ValueError("Provide either --data (.npz) or --results (.json)")


def _regenerate_from_results(results_path: str) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Create synthetic training features from estar_summary.json.

    Each problem gets 10 slices (10%-100%). Label = 1 if the problem was
    correct at full CoT (proxy: assume early slices of correct problems
    become safe after ~50% of thinking tokens).
    """
    from estar.features import FeatureExtractor, StepLogProbs
    from estar.utils import default_answer_set

    with open(results_path) as f:
        summary = json.load(f)

    problems = summary["problems"]
    ans_set = default_answer_set("open")
    num_buckets = len(ans_set)
    top_k = 20
    slices = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    all_X = []
    all_y = []
    all_seq = []

    for prob_idx, prob in enumerate(problems):
        think_tokens = prob["thinking_tokens"]
        correct = prob["correct"]

        extractor = FeatureExtractor(
            num_buckets=num_buckets,
            token_to_bucket={},
            top_k=top_k,
        )

        for frac in slices:
            step = max(1, int(frac * think_tokens))
            # Synthetic logprobs — seeded per-problem for reproducibility
            rng = np.random.RandomState(prob_idx * 100 + int(frac * 100))
            fake_ids = np.arange(top_k, dtype=np.int64)
            fake_lps = rng.uniform(-5.0, -0.1, top_k).astype(np.float64)
            fake_lps.sort()
            fake_lps = fake_lps[::-1]

            step_data = StepLogProbs(
                token_ids=fake_ids, log_probs=fake_lps, step=step
            )
            feat = extractor.extract(step_data, total_steps=think_tokens)
            all_X.append(feat)

            # Label: safe to stop if correct AND past halfway
            label = 1 if correct and frac >= 0.5 else 0
            all_y.append(label)
            all_seq.append(prob_idx)

    X = np.stack(all_X)
    y = np.array(all_y)
    seq = np.array(all_seq)
    feature_names = FeatureExtractor(
        num_buckets=num_buckets, token_to_bucket={}, top_k=top_k
    ).feature_names()

    return X, y, feature_names, seq


def simulate_patience_stopping(
    probas: np.ndarray,
    seq_ids: np.ndarray,
    threshold: float,
    patience: int,
    labels: np.ndarray,
) -> dict:
    """Simulate patience-based stopping on per-sequence data.

    For each sequence, walks through its samples in order and stops when
    `patience` consecutive predictions exceed `threshold`.

    Returns dict with accuracy, mean_savings, and per-sequence details.
    """
    unique_seqs = np.unique(seq_ids)
    correct = 0
    total = 0
    savings_list = []

    for seq_id in unique_seqs:
        mask = seq_ids == seq_id
        seq_probs = probas[mask]
        seq_labels = labels[mask]
        n_steps = len(seq_probs)
        if n_steps == 0:
            continue

        # Walk through and apply patience
        consec = 0
        stop_at = n_steps  # default: no early stop
        for i, p in enumerate(seq_probs):
            if p >= threshold:
                consec += 1
            else:
                consec = 0
            if consec >= patience:
                stop_at = i + 1
                break

        # The label at the stop point tells us if the answer was correct
        stop_label = seq_labels[min(stop_at, n_steps) - 1]
        correct += int(stop_label == 1)
        total += 1

        savings = 1.0 - (stop_at / n_steps) if n_steps > 0 else 0.0
        savings_list.append(savings)

    accuracy = correct / total if total > 0 else 0.0
    mean_savings = float(np.mean(savings_list)) if savings_list else 0.0

    return {
        "accuracy": accuracy,
        "mean_savings": mean_savings,
        "n_sequences": total,
        "n_correct": correct,
    }


def sweep_grid(
    X_val: np.ndarray,
    y_val: np.ndarray,
    seq_val: np.ndarray | None,
    clf: EstarClassifier,
    thresholds: list[float],
    patience_values: list[int],
) -> list[dict]:
    """Sweep threshold x patience grid and return results."""
    # Get raw probabilities for all validation samples
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        probas = np.array([clf.predict_proba(x) for x in X_val])

    # If no sequence IDs, treat each sample as its own sequence
    if seq_val is None:
        seq_val = np.arange(len(X_val))

    results = []
    for tau in thresholds:
        for pat in patience_values:
            res = simulate_patience_stopping(probas, seq_val, tau, pat, y_val)
            results.append({
                "threshold": tau,
                "patience": pat,
                **res,
            })

    return results


def print_results_table(results: list[dict]) -> None:
    """Print a formatted results table."""
    header = f"{'tau':>6} {'pat':>4} {'accuracy':>9} {'savings':>9} {'n_seq':>6} {'correct':>8}"
    logger.info("")
    logger.info("=" * len(header))
    logger.info("  Threshold x Patience Grid Sweep")
    logger.info("=" * len(header))
    logger.info(header)
    logger.info("-" * len(header))

    for r in sorted(results, key=lambda x: (-x["accuracy"], -x["mean_savings"])):
        logger.info(
            f"{r['threshold']:>6.2f} {r['patience']:>4d} "
            f"{r['accuracy']:>8.1%} {r['mean_savings']:>8.1%} "
            f"{r['n_sequences']:>6d} {r['n_correct']:>8d}"
        )

    logger.info("-" * len(header))

    # Highlight best
    best = max(results, key=lambda x: x["accuracy"] + 0.5 * x["mean_savings"])
    logger.info(
        f"\nBest (accuracy + 0.5*savings): "
        f"tau={best['threshold']}, patience={best['patience']} "
        f"-> {best['accuracy']:.1%} accuracy, {best['mean_savings']:.1%} savings"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune ESTAR-LITE threshold and patience"
    )
    parser.add_argument("--data", help="Path to training_data.npz")
    parser.add_argument("--results", help="Path to estar_summary.json (alternative to --data)")
    parser.add_argument(
        "--thresholds", type=float, nargs="+",
        default=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95],
        help="Threshold values to sweep",
    )
    parser.add_argument(
        "--patience-values", type=int, nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Patience values to sweep",
    )
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-best", action="store_true", help="Save the best config to a file")
    parser.add_argument("--output", default="models/tuned_config.json", help="Output path for best config")
    args = parser.parse_args()

    if not args.data and not args.results:
        parser.error("Provide either --data or --results")

    # Load data
    logger.info("Loading data...")
    X, y, feature_names, seq_ids = load_training_data(args.data, args.results)
    logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Positive rate: {y.mean():.2%}")

    # Split — stratified, keeping sequence grouping in validation
    if seq_ids is not None:
        # Split by unique sequences to avoid leaking
        unique_seqs = np.unique(seq_ids)
        # Create per-sequence labels (majority vote)
        seq_labels = np.array([
            int(y[seq_ids == s].mean() >= 0.5) for s in unique_seqs
        ])
        # Stratify only if both classes have >= 2 members
        min_class_count = min(np.bincount(seq_labels))
        stratify_arg = seq_labels if min_class_count >= 2 else None
        train_seqs, val_seqs = train_test_split(
            unique_seqs, test_size=args.val_size,
            random_state=args.seed, stratify=stratify_arg,
        )
        train_mask = np.isin(seq_ids, train_seqs)
        val_mask = np.isin(seq_ids, val_seqs)
        X_train, X_val = X[train_mask], X[val_mask]
        y_train, y_val = y[train_mask], y[val_mask]
        seq_val = seq_ids[val_mask]
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=args.val_size, random_state=args.seed, stratify=y,
        )
        seq_val = None

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # Train one model
    logger.info("Training classifier...")
    clf = EstarClassifier(threshold=0.9, patience=3)
    clf.train(X_train, y_train, X_val, y_val, feature_names=feature_names or None)

    # Sweep
    logger.info(
        f"Sweeping {len(args.thresholds)} thresholds x "
        f"{len(args.patience_values)} patience values..."
    )
    results = sweep_grid(X_val, y_val, seq_val, clf, args.thresholds, args.patience_values)

    print_results_table(results)

    # Save best config
    if args.save_best:
        best = max(results, key=lambda x: x["accuracy"] + 0.5 * x["mean_savings"])
        config = {
            "threshold": best["threshold"],
            "patience": best["patience"],
            "accuracy": best["accuracy"],
            "mean_savings": best["mean_savings"],
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(config, indent=2))
        logger.info(f"\nBest config saved to {output_path}")


if __name__ == "__main__":
    main()
