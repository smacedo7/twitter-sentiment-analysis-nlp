"""Sanity checks for the trained model that is versioned in the repository."""

import pytest

from src.config import MODEL_PATH
from src.predict import predict_sentiment

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="trained model not available")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("I love this, it is the best day ever!", "Positive"),
        ("This is terrible, I hate it so much.", "Negative"),
    ],
)
def test_committed_model_gets_obvious_cases_right(text, expected):
    assert predict_sentiment(text, MODEL_PATH) == expected
