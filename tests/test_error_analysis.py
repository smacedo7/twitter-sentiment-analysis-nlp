import pandas as pd
import pytest

from src import error_analysis
from src.train import train


@pytest.fixture(scope="module")
def predictions_path(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    train(
        data_path=__import__("pathlib").Path("tests/fixtures/sample_tweets.csv"),
        sample_size=0,
        model_path=out / "model.pkl",
        metrics_path=out / "metrics.json",
        reports_dir=out / "reports",
    )
    return out / "reports" / "test_predictions.csv"


def test_mask_personal_data():
    masked = error_analysis.mask_personal_data("hey @john_doe look http://t.co/abc and www.x.com")
    assert masked == "hey @user look <link> and <link>"


def test_train_saves_predictions_with_expected_columns(predictions_path):
    df = pd.read_csv(predictions_path)
    assert list(df.columns) == ["text", "clean_text", "target", "predicted", "positive_probability", "is_neutral"]
    assert df["positive_probability"].between(0, 1).all()


def test_run_writes_report_and_examples(predictions_path, tmp_path):
    report = error_analysis.run(predictions_path, tmp_path / "report.md", tmp_path / "examples.csv")

    assert "# Error analysis" in report
    assert (tmp_path / "report.md").exists()
    examples = pd.read_csv(tmp_path / "examples.csv")
    assert list(examples.columns) == ["text", "actual", "predicted", "confidence"]


def test_error_by_class_counts():
    df = pd.DataFrame(
        {
            "target": [0, 0, 1, 1],
            "predicted": [1, 0, 0, 1],
            "clean_text": ["a", "b", "c", "d"],
            "positive_probability": [0.9, 0.1, 0.2, 0.8],
        }
    )
    df["is_error"] = df["target"] != df["predicted"]
    assert error_analysis.error_by_class(df) == {"false_positives": 1, "false_negatives": 1}


def test_missing_predictions_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="src.train"):
        error_analysis.load_predictions(tmp_path / "nope.csv")
