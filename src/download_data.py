"""Download and prepare the Sentiment140 dataset.

The original file has no header, uses latin-1 encoding and labels tweets as
0 (negative) or 4 (positive). This script keeps only the two columns we need
and converts the labels to 0 / 1, saving the result to ``data/sentiment140.csv``.

Usage:
    python -m src.download_data                   # download from Stanford and convert
    python -m src.download_data --raw-csv PATH    # convert a CSV you already have (e.g. from Kaggle)
"""

from __future__ import annotations

import argparse
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from src.config import DATA_PATH

SENTIMENT140_URL = "https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
RAW_FILENAME = "training.1600000.processed.noemoticon.csv"

# Original columns: target, id, date, flag, user, text. We only need the first and the last.
RAW_USECOLS = [0, 5]
RAW_NAMES = ["target", "text"]


def convert_raw_csv(raw_csv: Path, output_path: Path = DATA_PATH) -> pd.DataFrame:
    """Convert the original Sentiment140 CSV into the format used by this project."""
    df = pd.read_csv(
        raw_csv,
        encoding="latin-1",
        header=None,
        usecols=RAW_USECOLS,
        names=RAW_NAMES,
    )

    unexpected = set(df["target"].unique()) - {0, 4}
    if unexpected:
        raise ValueError(f"Unexpected labels in raw file: {sorted(unexpected)}. Expected only 0 and 4.")

    # 0 -> negative (0), 4 -> positive (1)
    df["target"] = df["target"].map({0: 0, 4: 1})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def download_and_convert(output_path: Path = DATA_PATH) -> pd.DataFrame:
    """Download the official zip, extract the training file and convert it."""
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "sentiment140.zip"
        print(f"Downloading {SENTIMENT140_URL} ...")
        urllib.request.urlretrieve(SENTIMENT140_URL, zip_path)

        with zipfile.ZipFile(zip_path) as archive:
            archive.extract(RAW_FILENAME, tmp)

        return convert_raw_csv(Path(tmp) / RAW_FILENAME, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the Sentiment140 dataset.")
    parser.add_argument(
        "--raw-csv",
        type=Path,
        help="Path to an already downloaded original CSV. If omitted, the dataset is downloaded.",
    )
    parser.add_argument("--output", type=Path, default=DATA_PATH, help="Where to save the processed CSV.")
    args = parser.parse_args()

    if args.raw_csv:
        df = convert_raw_csv(args.raw_csv, args.output)
    else:
        df = download_and_convert(args.output)

    print(f"Saved {len(df):,} tweets to {args.output}")
    print(df["target"].value_counts().rename({0: "negative", 1: "positive"}).to_string())


if __name__ == "__main__":
    main()
