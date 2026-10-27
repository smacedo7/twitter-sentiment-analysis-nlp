"""Train, compare and save sentiment classifiers.

Every model is a scikit-learn ``Pipeline`` with two steps:
    1. ``tfidf``: turns text into numbers (TF-IDF with unigrams and bigrams)
    2. ``clf``:   the classifier (Logistic Regression, Naive Bayes or Linear SVM)

The best model (highest F1-score on the held-out test set) is saved to
``models/sentiment_classifier.pkl`` together with a metrics summary.

Usage:
    python -m src.train                      # uses a 200,000-tweet sample (fast)
    python -m src.train --sample-size 0      # uses the full dataset (slower)
    python -m src.train --data other.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import DATA_PATH, LABELS, METRICS_PATH, MODEL_PATH, RANDOM_STATE, REPORTS_DIR
from src.preprocessing import preprocess_series

DEFAULT_SAMPLE_SIZE = 200_000
DEFAULT_TEST_SIZE = 0.2


def load_dataset(data_path: Path, sample_size: int = DEFAULT_SAMPLE_SIZE) -> pd.DataFrame:
    """Load the CSV, optionally subsample it, and add the preprocessed text column.

    ``sample_size <= 0`` means "use every row".
    """
    df = pd.read_csv(data_path).dropna(subset=["text", "target"])

    if 0 < sample_size < len(df):
        df = df.sample(n=sample_size, random_state=RANDOM_STATE)

    df = df.assign(clean_text=preprocess_series(df["text"]))
    # Tweets made only of links/mentions end up empty and carry no information.
    return df[df["clean_text"] != ""].reset_index(drop=True)


def build_models() -> dict[str, Pipeline]:
    """Create one untrained pipeline per candidate model."""

    def tfidf() -> TfidfVectorizer:
        return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True)

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(random_state=RANDOM_STATE),
    }
    return {name: Pipeline([("tfidf", tfidf()), ("clf", clf)]) for name, clf in classifiers.items()}


def evaluate(model: Pipeline, x_test: pd.Series, y_test: pd.Series) -> dict:
    """Compute classification metrics for the positive class."""
    y_pred = model.predict(x_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def select_best(results: dict[str, dict]) -> str:
    """Return the name of the model with the best F1-score (accuracy breaks ties)."""
    return max(results, key=lambda name: (results[name]["f1"], results[name]["accuracy"]))


def save_confusion_matrix(matrix: list[list[int]], title: str, output_path: Path) -> None:
    """Save a confusion matrix heatmap as a PNG file."""
    names = [LABELS[0], LABELS[1]]
    figure = Figure(figsize=(5, 4))
    ax = figure.subplots()
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)


def train(
    data_path: Path = DATA_PATH,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    test_size: float = DEFAULT_TEST_SIZE,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
    reports_dir: Path = REPORTS_DIR,
) -> dict:
    """Run the full training workflow and return the metrics summary."""
    df = load_dataset(data_path, sample_size)
    print(f"Using {len(df):,} tweets after preprocessing.")

    # `stratify` keeps the positive/negative proportion identical in train and test.
    x_train, x_test, y_train, y_test = train_test_split(
        df["clean_text"], df["target"], test_size=test_size, random_state=RANDOM_STATE, stratify=df["target"]
    )

    models = build_models()
    results: dict[str, dict] = {}
    for name, model in models.items():
        print(f"Training {name} ...")
        model.fit(x_train, y_train)
        results[name] = evaluate(model, x_test, y_test)

    best_name = select_best(results)
    summary = {
        "best_model": best_name,
        "dataset": {"rows_used": len(df), "train_size": len(x_train), "test_size": len(x_test)},
        "models": results,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models[best_name], model_path)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    save_confusion_matrix(
        results[best_name]["confusion_matrix"],
        title=f"Confusion matrix - {best_name}",
        output_path=reports_dir / "confusion_matrix.png",
    )
    return summary


def print_summary(summary: dict) -> None:
    table = pd.DataFrame(summary["models"]).T[["accuracy", "precision", "recall", "f1"]]
    print("\n" + table.round(4).to_string())
    print(f"\nBest model: {summary['best_model']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sentiment classifiers on Sentiment140.")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Processed CSV with `target` and `text` columns.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of tweets to use. Use 0 for the whole dataset.",
    )
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE, help="Fraction used for evaluation.")
    args = parser.parse_args()

    summary = train(data_path=args.data, sample_size=args.sample_size, test_size=args.test_size)
    print_summary(summary)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
