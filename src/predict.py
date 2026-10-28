"""Predict the sentiment of tweets with the trained model.

Python usage:
    >>> from src.predict import predict_sentiment
    >>> predict_sentiment("I love how easy this application is to use!")
    'Positive'

Terminal usage:
    python -m src.predict "I love this!"      # one prediction
    python -m src.predict                     # interactive mode (empty line to quit)
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from src.config import LABELS, MODEL_PATH
from src.neutral import NEUTRAL_LABEL, is_neutral
from src.preprocessing import preprocess_text


@lru_cache(maxsize=2)
def load_model(model_path: Path = MODEL_PATH) -> Pipeline:
    """Load the trained pipeline from disk (cached, so the file is only read once)."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Train it first with: python -m src.train"
        )
    return joblib.load(model_path)


def positive_probability(model: Pipeline, cleaned_texts: list[str]) -> np.ndarray:
    """Return, for each text, how likely it is to be positive (a number between 0 and 1).

    Logistic Regression and Naive Bayes provide real probabilities. Linear SVM does not,
    so its distance to the decision boundary is squashed with a sigmoid. In that case the
    value works as a confidence score, but it is not a calibrated probability.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(cleaned_texts)[:, 1]

    distance = model.decision_function(cleaned_texts)
    return 1 / (1 + np.exp(-distance))


def neutral_margin(model) -> float:
    """Margin saved together with the model by ``src.train`` (0 means: never answer Neutral)."""
    return float(getattr(model, "neutral_margin", 0.0))


def _build_result(positive_probability: float, margin: float = 0.0) -> dict:
    if is_neutral(positive_probability, margin):
        label = NEUTRAL_LABEL
    else:
        label = LABELS[int(positive_probability >= 0.5)]
    confidence = max(positive_probability, 1 - positive_probability)
    return {
        "label": label,
        "confidence": round(float(confidence), 4),
        "positive_probability": round(float(positive_probability), 4),
    }


def predict_batch(texts: list[str], model_path: Path = MODEL_PATH) -> list[dict]:
    """Predict many tweets at once.

    Returns one dict per input, in the same order, with ``label`` ("Positive", "Negative" or
    "Neutral" when the model is not sure), ``confidence`` (how sure the model is about the more
    likely side, between 0.5 and 1) and ``positive_probability`` (calibrated, between 0 and 1).
    Texts with nothing left after cleaning (only links, mentions, symbols...) cannot be
    analyzed, so they get ``None`` values.
    """
    model = load_model(model_path)
    cleaned = [preprocess_text(text) for text in texts]
    valid_positions = [i for i, text in enumerate(cleaned) if text]

    results: list[dict] = [{"label": None, "confidence": None, "positive_probability": None} for _ in texts]
    margin = neutral_margin(model)
    if valid_positions:
        probabilities = positive_probability(model, [cleaned[i] for i in valid_positions])
        for position, probability in zip(valid_positions, probabilities):
            results[position] = _build_result(float(probability), margin)
    return results


def predict_detailed(text: str, model_path: Path = MODEL_PATH) -> dict:
    """Predict one tweet and return its label together with a confidence value.

    Raises:
        ValueError: if the text has no analyzable words after cleaning.
    """
    result = predict_batch([text], model_path)[0]
    if result["label"] is None:
        raise ValueError("The text has no analyzable words after cleaning (only links, mentions or symbols?).")
    return result


def predict_sentiment(text: str, model_path: Path = MODEL_PATH) -> str:
    """Predict one tweet and return only the label: 'Positive', 'Negative' or 'Neutral'."""
    return predict_detailed(text, model_path)["label"]


def _print_prediction(text: str) -> None:
    try:
        result = predict_detailed(text)
    except ValueError as error:
        print(f"  -> {error}")
        return
    print(f"  -> {result['label']} (confidence: {result['confidence']:.1%})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict the sentiment of a tweet.")
    parser.add_argument("text", nargs="*", help="Tweet text. If omitted, starts the interactive mode.")
    args = parser.parse_args()

    try:
        load_model()
    except FileNotFoundError as error:
        raise SystemExit(error)

    if args.text:
        _print_prediction(" ".join(args.text))
        return

    print("Interactive mode. Type a tweet and press Enter (empty line to quit).")
    while True:
        text = input("> ").strip()
        if not text:
            break
        _print_prediction(text)


if __name__ == "__main__":
    main()
