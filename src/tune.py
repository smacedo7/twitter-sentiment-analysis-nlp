"""Search for better hyperparameters for the Logistic Regression model.

A *hyperparameter* is a setting chosen by the person before training (for example how many
words the TF-IDF keeps), as opposed to the *parameters* the model learns by itself (the
weights of each word).

How the search works:
    1. Reproduce the same train/test split used by ``src.train``. The test set is NOT used here.
    2. Take a random sample of the training set (the full set would take hours).
    3. Try many combinations of settings with 3-fold cross-validation: the sample is cut in
       3 parts, and each combination is trained on 2 parts and scored on the 3rd, three times.
    4. Save the best combination to ``models/best_params.json``; ``src.train`` uses it automatically.

Usage:
    python -m src.tune                      # 200,000-tweet sample, 20 combinations
    python -m src.tune --sample-size 300000 --n-iter 30
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from scipy.stats import loguniform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.config import BEST_PARAMS_PATH, DATA_PATH, RANDOM_STATE, REPORTS_DIR
from src.train import DEFAULT_TEST_SIZE, load_dataset

SEARCH_SPACE = {
    "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
    "tfidf__min_df": [1, 2, 3, 5],
    "tfidf__max_features": [50_000, 100_000, 200_000, 400_000],
    "tfidf__sublinear_tf": [True, False],
    "clf__C": loguniform(0.1, 30),
}


def base_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


BASELINE_PARAMS = {
    "tfidf__ngram_range": [1, 2],
    "tfidf__min_df": 2,
    "tfidf__max_features": 100_000,
    "tfidf__sublinear_tf": True,
    "clf__C": 1.0,
}


def choose_final_params(best_params: dict, best_score: float, baseline_score: float) -> tuple[dict, bool]:
    """Keep the searched settings only if they beat the original ones; otherwise keep the originals.

    Returns the chosen parameters and whether the search really improved the score.
    """
    if best_score > baseline_score:
        return best_params, True
    return dict(BASELINE_PARAMS), False


def run_search(
    data_path: Path = DATA_PATH,
    sample_size: int = 200_000,
    n_iter: int = 20,
    cv: int = 3,
    n_jobs: int = -1,
    best_params_path: Path = BEST_PARAMS_PATH,
    reports_dir: Path = REPORTS_DIR,
    search_space: dict | None = None,
) -> dict:
    df = load_dataset(data_path, sample_size=0)
    x_train, _, y_train, _ = train_test_split(
        df["clean_text"], df["target"], test_size=DEFAULT_TEST_SIZE, random_state=RANDOM_STATE, stratify=df["target"]
    )
    if 0 < sample_size < len(x_train):
        x_train, _, y_train, _ = train_test_split(
            x_train, y_train, train_size=sample_size, random_state=RANDOM_STATE, stratify=y_train
        )
    print(f"Searching {n_iter} combinations with {cv}-fold cross-validation on {len(x_train):,} training tweets...")

    search = RandomizedSearchCV(
        base_pipeline(),
        search_space or SEARCH_SPACE,
        n_iter=n_iter,
        scoring="f1",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
        verbose=1,
        refit=False,
    )
    started = time.perf_counter()
    search.fit(x_train, y_train)
    minutes = (time.perf_counter() - started) / 60

    # Reference point: the settings used before the search, scored on the very same folds.
    baseline = base_pipeline().set_params(
        tfidf__ngram_range=(1, 2), tfidf__min_df=2, tfidf__max_features=100_000, tfidf__sublinear_tf=True, clf__C=1.0
    )
    searched_folds = StratifiedKFold(n_splits=cv)  # same splitter RandomizedSearchCV uses for classifiers
    baseline_score = float(cross_val_score(baseline, x_train, y_train, scoring="f1", cv=searched_folds, n_jobs=n_jobs).mean())

    results = pd.DataFrame(search.cv_results_)
    columns = [c for c in results.columns if c.startswith("param_")] + ["mean_test_score", "std_test_score", "rank_test_score"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    results[columns].sort_values("rank_test_score").to_csv(reports_dir / "tuning_results.csv", index=False)

    searched = {key: _to_json(value) for key, value in search.best_params_.items()}
    best, improved = choose_final_params(searched, float(search.best_score_), baseline_score)
    summary = {
        "best_params": best,
        "improved_over_baseline": improved,
        "best_searched_params": searched,
        "best_cv_f1": float(search.best_score_),
        "baseline_cv_f1": baseline_score,
        "tuning_sample_size": int(len(x_train)),
        "combinations_tried": int(n_iter),
        "cv_folds": cv,
        "minutes": round(minutes, 1),
    }
    best_params_path.parent.mkdir(parents=True, exist_ok=True)
    best_params_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _to_json(value):
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "item"):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter search for the Logistic Regression model.")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--sample-size", type=int, default=200_000)
    parser.add_argument("--n-iter", type=int, default=20)
    args = parser.parse_args()

    summary = run_search(args.data, args.sample_size, args.n_iter)
    print(json.dumps(summary, indent=2))
    if not summary["improved_over_baseline"]:
        print("\nNo combination beat the original settings, so they were kept.")
    print(f"\nSaved to {BEST_PARAMS_PATH}")


if __name__ == "__main__":
    main()
