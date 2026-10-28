import numpy as np
import pytest

from src.calibration import (
    calibration_metrics,
    expected_calibration_error,
    reliability_curve,
    select_calibration,
)
from src.train import build_models, load_dataset


def test_ece_is_zero_for_perfectly_calibrated_probabilities():
    y = np.array([1, 1, 1, 0] * 25)  # 75% positives
    p = np.full(100, 0.75)
    assert expected_calibration_error(y, p) == pytest.approx(0.0)


def test_ece_detects_overconfidence():
    y = np.array([1, 1, 1, 0] * 25)  # really 75% positive...
    p = np.full(100, 0.99)  # ...but the model says 99%
    assert expected_calibration_error(y, p) == pytest.approx(0.24)


def test_reliability_curve_counts_add_up():
    rng = np.random.default_rng(0)
    p = rng.random(500)
    y = (rng.random(500) < p).astype(int)
    curve = reliability_curve(y, p)
    assert curve["count"].sum() == 500
    assert curve["fraction_positive"].between(0, 1).all()


def test_calibration_metrics_keys_and_ranges():
    metrics = calibration_metrics([1, 0, 1, 0], [0.9, 0.2, 0.7, 0.4])
    assert set(metrics) == {"brier", "log_loss", "ece"}
    assert 0 <= metrics["brier"] <= 1


def test_select_calibration_returns_working_calibrator():
    df = load_dataset("tests/fixtures/sample_tweets.csv", sample_size=0)
    model = build_models()["Logistic Regression"].fit(df["clean_text"][:120], df["target"][:120])

    result = select_calibration(model, df["clean_text"][120:], df["target"][120:])

    assert result["method"] in {"sigmoid", "isotonic"}
    probability = result["calibrator"].predict_proba(df["clean_text"][:5])[:, 1]
    assert ((probability >= 0) & (probability <= 1)).all()
