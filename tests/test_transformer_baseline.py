import pandas as pd
import pytest

from src import transformer_baseline as tb


@pytest.fixture
def predictions_csv(tmp_path):
    df = pd.DataFrame(
        {
            "text": ["great day", "awful day", "love it @bob", "hate it", "so good", "so bad"],
            "clean_text": ["great day", "awful day", "love it", "hate it", "good", "bad"],
            "target": [1, 0, 1, 0, 1, 0],
            "predicted": [1, 0, 1, 1, 1, 0],
            "positive_probability": [0.9, 0.1, 0.8, 0.6, 0.7, 0.2],
        }
    )
    path = tmp_path / "preds.csv"
    df.to_csv(path, index=False)
    return path


def test_compare_with_fake_predictor(predictions_csv):
    seen = []

    def perfect(texts):
        seen.extend(texts)
        return [1 if t in {"great day", "love it @user", "so good"} else 0 for t in texts]

    result = tb.compare(predictions_csv, sample_size=0, predictor=perfect)

    assert result["sample_size"] == 6
    assert result["transformer"]["accuracy"] == 1.0
    assert result["classic_model"]["accuracy"] == pytest.approx(5 / 6)
    assert "love it @user" in seen  # handles are masked before reaching the model


def test_compare_respects_sample_size(predictions_csv):
    result = tb.compare(predictions_csv, sample_size=4, predictor=lambda texts: [1] * len(texts))
    assert result["sample_size"] == 4


def test_compare_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="src.train"):
        tb.compare(tmp_path / "missing.csv", predictor=lambda t: [])
