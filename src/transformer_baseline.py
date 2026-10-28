"""Compare the classic model with a pretrained transformer on the same test tweets.

The transformer (DistilBERT fine-tuned on SST-2) is used as is, without extra training.
It reads the original tweet, while the classic model reads the cleaned text, so the
comparison shows how much a much larger language model helps on this dataset.

Needs the optional dependencies:  pip install -r requirements-transformers.txt
Needs ``reports/test_predictions.csv`` (created by ``python -m src.train``).

Usage:
    python -m src.transformer_baseline                 # 2,000 test tweets
    python -m src.transformer_baseline --sample-size 5000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.config import RANDOM_STATE, REPORTS_DIR
from src.error_analysis import mask_personal_data

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
PREDICTIONS_PATH = REPORTS_DIR / "test_predictions.csv"
OUTPUT_PATH = REPORTS_DIR / "transformer_comparison.json"
DEFAULT_SAMPLE_SIZE = 2_000


def metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }


def load_transformer_predictor(model_name: str = MODEL_NAME) -> Callable[[list[str]], list[int]]:
    """Return a function that maps a list of texts to 0/1 labels."""
    try:
        from transformers import pipeline
    except ImportError as error:
        raise SystemExit(
            "The `transformers` package is not installed. Run: pip install -r requirements-transformers.txt"
        ) from error

    classifier = pipeline("sentiment-analysis", model=model_name, truncation=True, max_length=128)

    def predict(texts: list[str]) -> list[int]:
        outputs = classifier(texts, batch_size=32)
        return [1 if output["label"].upper().startswith("POS") else 0 for output in outputs]

    return predict


def compare(
    predictions_path: Path = PREDICTIONS_PATH,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    predictor: Callable[[list[str]], list[int]] | None = None,
    model_name: str = MODEL_NAME,
) -> dict:
    """Evaluate both models on the same random sample of the test set."""
    if not Path(predictions_path).exists():
        raise FileNotFoundError(f"{predictions_path} not found. Train the model first with: python -m src.train")

    df = pd.read_csv(predictions_path, keep_default_na=False)
    if 0 < sample_size < len(df):
        df = df.sample(n=sample_size, random_state=RANDOM_STATE)

    predictor = predictor or load_transformer_predictor(model_name)
    texts = [mask_personal_data(text) for text in df["text"]]

    started = time.perf_counter()
    transformer_pred = predictor(texts)
    seconds = time.perf_counter() - started

    return {
        "sample_size": len(df),
        "transformer_model": model_name,
        "classic_model": metrics(df["target"], df["predicted"]),
        "transformer": metrics(df["target"], transformer_pred),
        "transformer_seconds": round(seconds, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the classic model with a pretrained transformer.")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    args = parser.parse_args()

    try:
        result = compare(sample_size=args.sample_size)
    except FileNotFoundError as error:
        raise SystemExit(error)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    table = pd.DataFrame({"TF-IDF model": result["classic_model"], "DistilBERT": result["transformer"]}).T
    print(table.round(4).to_string())
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
