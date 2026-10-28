import pandas as pd
import pytest

from src.batch import CONFIDENCE_COLUMN, LABEL_COLUMN, predict_dataframe


def test_predict_dataframe_adds_columns_and_keeps_originals(model_path):
    df = pd.DataFrame({"id": [1, 2, 3], "tweet": ["I love this", "@user http://t.co/abc", "I hate this"]})

    result = predict_dataframe(df, "tweet", model_path)

    assert list(result.columns) == ["id", "tweet", LABEL_COLUMN, CONFIDENCE_COLUMN]
    # pandas stores missing values as NaN, so check "is missing" instead of comparing with None.
    assert result[LABEL_COLUMN].isna().tolist() == [False, True, False]
    assert result[LABEL_COLUMN].dropna().tolist() == ["Positive", "Negative"]
    assert result[CONFIDENCE_COLUMN].isna().tolist() == [False, True, False]
    assert list(df.columns) == ["id", "tweet"]  # the input table is not modified


def test_predict_dataframe_handles_missing_text(model_path):
    df = pd.DataFrame({"text": ["I love this", None]})

    result = predict_dataframe(df, "text", model_path)

    assert result[LABEL_COLUMN].isna().tolist() == [False, True]
    assert result[LABEL_COLUMN].iloc[0] == "Positive"


def test_predict_dataframe_unknown_column(model_path):
    with pytest.raises(ValueError, match="not found"):
        predict_dataframe(pd.DataFrame({"text": ["hi"]}), "tweet", model_path)
