import json
from pathlib import Path

import joblib
import pytest

from src.train import build_models, load_dataset, select_best, train

FIXTURE = Path(__file__).parent / "fixtures" / "sample_tweets.csv"


def test_load_dataset_adds_clean_text_and_drops_empty_rows(tmp_path):
    csv = tmp_path / "tweets.csv"
    csv.write_text("target,text\n1,I love this!!!\n0,@user http://t.co/abc\n0,terrible day\n", encoding="utf-8")

    df = load_dataset(csv, sample_size=0)

    assert df["clean_text"].tolist() == ["love", "terrible day"]


def test_load_dataset_respects_sample_size():
    df = load_dataset(FIXTURE, sample_size=50)
    assert len(df) <= 50


def test_build_models_returns_three_pipelines():
    models = build_models()
    assert set(models) == {"Logistic Regression", "Multinomial Naive Bayes", "Linear SVM"}
    for pipeline in models.values():
        assert list(pipeline.named_steps) == ["tfidf", "clf"]


def test_select_best_uses_f1_then_accuracy():
    results = {
        "a": {"f1": 0.80, "accuracy": 0.90},
        "b": {"f1": 0.85, "accuracy": 0.70},
        "c": {"f1": 0.85, "accuracy": 0.75},
    }
    assert select_best(results) == "c"


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("artifacts")
    summary = train(
        data_path=FIXTURE,
        sample_size=0,
        model_path=out / "model.pkl",
        metrics_path=out / "metrics.json",
        reports_dir=out / "reports",
    )
    return summary, out


def test_train_creates_all_artifacts(trained):
    _, out = trained
    assert (out / "model.pkl").exists()
    assert (out / "metrics.json").exists()
    assert (out / "reports" / "confusion_matrix.png").exists()


def test_train_metrics_are_reasonable_on_separable_data(trained):
    summary, out = trained
    best = summary["models"][summary["best_model"]]

    assert best["accuracy"] >= 0.8
    assert json.loads((out / "metrics.json").read_text())["best_model"] == summary["best_model"]


def test_saved_model_can_predict(trained):
    _, out = trained
    model = joblib.load(out / "model.pkl")

    predictions = model.predict(["love wonderful great day", "hate terrible awful worst"])

    assert predictions.tolist() == [1, 0]
