"""The Neutral category.

Sentiment140 only contains Positive and Negative tweets, so there are no examples of "neutral"
to learn from. Instead, the model abstains when it is not sure: if the calibrated probability
of "Positive" is close to 50%, the tweet is labeled **Neutral** (mixed, ambiguous or without a
clear opinion).

    probability of Positive:   0 ----- 0.5-m ===== 0.5 ===== 0.5+m ----- 1
                                 Negative |      Neutral      | Positive

``m`` is the *margin*. A larger margin means more tweets become Neutral, and the tweets that are
still classified as Positive or Negative are more reliable. The margin is chosen automatically
so that the decided tweets reach a target accuracy, using data not used for the final test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NEUTRAL_LABEL = "Neutral"
DEFAULT_TARGET_ACCURACY = 0.85
MAX_MARGIN = 0.30


def is_neutral(probability, margin: float):
    """True where the probability of Positive is within ``margin`` of 50%."""
    return np.abs(np.asarray(probability) - 0.5) < margin


def margin_tradeoff(y_true, probability, margins=None) -> pd.DataFrame:
    """For each margin: how many tweets stay Neutral and how accurate the other ones are."""
    y_true = np.asarray(y_true)
    probability = np.asarray(probability)
    predicted = (probability >= 0.5).astype(int)
    if margins is None:
        margins = np.round(np.arange(0.0, MAX_MARGIN + 0.005, 0.01), 2)

    rows = []
    for margin in margins:
        neutral = is_neutral(probability, margin)
        decided = ~neutral
        accuracy = float((predicted[decided] == y_true[decided]).mean()) if decided.any() else float("nan")
        rows.append(
            {
                "margin": float(margin),
                "neutral_share": float(neutral.mean()),
                "decided_share": float(decided.mean()),
                "accuracy_on_decided": accuracy,
            }
        )
    return pd.DataFrame(rows)


def choose_margin(y_true, probability, target_accuracy: float = DEFAULT_TARGET_ACCURACY) -> float:
    """Smallest margin for which the non-neutral tweets reach ``target_accuracy``.

    If the target cannot be reached, the largest margin tried is returned.
    """
    table = margin_tradeoff(y_true, probability)
    reached = table[table["accuracy_on_decided"] >= target_accuracy]
    return float(reached["margin"].iloc[0]) if not reached.empty else float(table["margin"].iloc[-1])
