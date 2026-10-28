"""Sentiment prediction for a whole table of tweets (used by the web interface)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import MODEL_PATH
from src.predict import predict_batch

LABEL_COLUMN = "predicted_sentiment"
CONFIDENCE_COLUMN = "confidence"


def predict_dataframe(df: pd.DataFrame, text_column: str, model_path: Path = MODEL_PATH) -> pd.DataFrame:
    """Return a copy of ``df`` with two extra columns: ``predicted_sentiment`` and ``confidence``.

    Rows whose text is missing or has no analyzable words get empty values in both columns.
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' was not found. Available columns: {list(df.columns)}")

    texts = df[text_column].fillna("").astype(str).tolist()
    predictions = predict_batch(texts, model_path)

    result = df.copy()
    result[LABEL_COLUMN] = [prediction["label"] for prediction in predictions]
    result[CONFIDENCE_COLUMN] = [prediction["confidence"] for prediction in predictions]
    return result
