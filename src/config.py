"""Central place for paths and constants shared across the project."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

# Processed dataset: two columns, `target` (0 = negative, 1 = positive) and `text`.
DATA_PATH = DATA_DIR / "sentiment140.csv"

# Best model found during training and its evaluation summary.
MODEL_PATH = MODELS_DIR / "sentiment_classifier.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"

# Label mapping used everywhere (training, prediction, API, web app).
LABELS = {0: "Negative", 1: "Positive"}

RANDOM_STATE = 42
