from pathlib import Path

import pytest

from src.train import train

FIXTURE_DATA = Path(__file__).parent / "fixtures" / "sample_tweets.csv"


@pytest.fixture(scope="session")
def model_path(tmp_path_factory) -> Path:
    """Train a small model once per test session and return the path to the saved file."""
    out = tmp_path_factory.mktemp("trained_model")
    path = out / "model.pkl"
    train(
        data_path=FIXTURE_DATA,
        sample_size=0,
        model_path=path,
        metrics_path=out / "metrics.json",
        reports_dir=out / "reports",
        target_accuracy=0.0,  # margin 0: these tests expect a Positive/Negative answer (Neutral has its own tests)
    )
    return path
