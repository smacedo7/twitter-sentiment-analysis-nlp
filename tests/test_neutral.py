import numpy as np
import pytest

from src.neutral import choose_margin, is_neutral, margin_tradeoff


def test_is_neutral_uses_distance_to_half():
    probability = np.array([0.05, 0.45, 0.5, 0.58, 0.95])
    assert is_neutral(probability, 0.1).tolist() == [False, True, True, True, False]
    assert is_neutral(probability, 0.0).tolist() == [False] * 5


def test_tradeoff_margin_zero_keeps_everything():
    y = [1, 0, 1, 0]
    p = [0.9, 0.1, 0.4, 0.6]
    table = margin_tradeoff(y, p, margins=[0.0, 0.2])
    assert table.loc[0, "neutral_share"] == 0.0
    assert table.loc[0, "accuracy_on_decided"] == 0.5  # the two uncertain ones are wrong
    assert table.loc[1, "neutral_share"] == 0.5
    assert table.loc[1, "accuracy_on_decided"] == 1.0


def test_choose_margin_finds_smallest_margin_reaching_target():
    rng = np.random.default_rng(1)
    p = rng.random(4000)
    y = (rng.random(4000) < p).astype(int)  # perfectly calibrated data
    margin = choose_margin(y, p, target_accuracy=0.8)
    table = margin_tradeoff(y, p)
    row = table[table["margin"] == margin].iloc[0]
    assert row["accuracy_on_decided"] >= 0.8
    assert margin > 0


def test_choose_margin_unreachable_target_returns_largest():
    y = [1, 0, 1, 0] * 10
    p = [0.5, 0.5, 0.5, 0.5] * 10
    assert choose_margin(y, p, target_accuracy=0.99) == pytest.approx(0.30)
