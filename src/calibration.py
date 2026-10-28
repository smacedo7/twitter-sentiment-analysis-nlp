"""Probability calibration.

A model is *calibrated* when its probabilities mean what they say: among all the tweets where
it answers "70% positive", about 70% should really be positive. Models can be accurate and
still badly calibrated (too confident or too shy), which matters when you show a confidence
value to a user or use it to make decisions (as the Neutral category does).

Calibration is learned on a separate slice of the training data that the model did not use to
learn its weights, and it is always measured on the untouched test set.

Metrics (lower is better for all three):
    - Brier score:  mean squared difference between the probability and the real outcome (0 or 1).
    - Log loss:     punishes confident wrong answers very hard.
    - ECE:          Expected Calibration Error, the average gap between "what the model says"
                    and "what really happens", measured in 10 probability bins.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE

METHODS = ("sigmoid", "isotonic")


def reliability_curve(y_true, probability, n_bins: int = 10) -> pd.DataFrame:
    """For each probability bin: the average predicted probability and the real share of positives."""
    y_true = np.asarray(y_true)
    probability = np.asarray(probability)
    bins = np.minimum((probability * n_bins).astype(int), n_bins - 1)
    rows = []
    for index in range(n_bins):
        mask = bins == index
        if mask.any():
            rows.append(
                {
                    "mean_predicted": float(probability[mask].mean()),
                    "fraction_positive": float(y_true[mask].mean()),
                    "count": int(mask.sum()),
                }
            )
    return pd.DataFrame(rows)


def expected_calibration_error(y_true, probability, n_bins: int = 10) -> float:
    curve = reliability_curve(y_true, probability, n_bins)
    weights = curve["count"] / curve["count"].sum()
    return float((weights * (curve["mean_predicted"] - curve["fraction_positive"]).abs()).sum())


def calibration_metrics(y_true, probability) -> dict:
    probability = np.clip(np.asarray(probability), 1e-6, 1 - 1e-6)
    return {
        "brier": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability)),
        "ece": expected_calibration_error(y_true, probability),
    }


def fit_calibrator(model, x_cal, y_cal, method: str) -> CalibratedClassifierCV:
    """Learn a calibration map for an already trained model (the model itself is frozen)."""
    # The calibration map is learned on cross-validated folds; tiny datasets need fewer folds.
    smallest_class = int(pd.Series(np.asarray(y_cal)).value_counts().min())
    folds = max(2, min(5, smallest_class))
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method=method, cv=folds)
    return calibrated.fit(x_cal, y_cal)


def select_calibration(model, x_cal, y_cal, random_state: int = RANDOM_STATE) -> dict:
    """Pick the calibration method with the best log loss on data it was not fitted on.

    The calibration slice is cut in two halves: both methods learn on the first half and are
    compared on the second. The winner is then refitted on the whole slice.
    """
    x_a, x_b, y_a, y_b = train_test_split(x_cal, y_cal, test_size=0.5, random_state=random_state, stratify=y_cal)

    scores = {}
    holdout_probability = {}
    for method in METHODS:
        probability = fit_calibrator(model, x_a, y_a, method).predict_proba(x_b)[:, 1]
        scores[method] = calibration_metrics(y_b, probability)
        holdout_probability[method] = probability

    best_method = min(scores, key=lambda name: scores[name]["log_loss"])
    return {
        "method": best_method,
        "scores": scores,
        "calibrator": fit_calibrator(model, x_cal, y_cal, best_method),
        "holdout_y": np.asarray(y_b),
        "holdout_probability": holdout_probability[best_method],
    }


def save_reliability_diagram(curves: dict[str, pd.DataFrame], output_path: Path) -> None:
    """Plot predicted probability (x) against what really happened (y); the diagonal is perfect."""
    figure = Figure(figsize=(5.5, 5))
    ax = figure.subplots()
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    for name, curve in curves.items():
        ax.plot(curve["mean_predicted"], curve["fraction_positive"], marker="o", label=name)
    ax.set_xlabel("Predicted probability of Positive")
    ax.set_ylabel("Real share of Positive tweets")
    ax.set_title("Reliability diagram (test set)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
