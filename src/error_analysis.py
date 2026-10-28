"""Study where the best model makes mistakes.

Reads ``reports/test_predictions.csv`` (created by ``python -m src.train``) and writes:
    - ``reports/error_analysis.md``   a readable summary
    - ``reports/error_examples.csv``  the most confident mistakes, for manual reading

Usage:
    python -m src.error_analysis
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from src.config import LABELS, REPORTS_DIR

PREDICTIONS_PATH = REPORTS_DIR / "test_predictions.csv"
REPORT_PATH = REPORTS_DIR / "error_analysis.md"
EXAMPLES_PATH = REPORTS_DIR / "error_examples.csv"

TOP_WORDS = 15
TOP_EXAMPLES = 10
MIN_WORD_COUNT = 20
MIN_BUCKET_ROWS = 1

MENTION_RE = re.compile(r"@\w+")
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")


def mask_personal_data(text: str) -> str:
    """Replace @handles and links, so published examples do not expose other people."""
    text = MENTION_RE.sub("@user", str(text))
    return URL_RE.sub("<link>", text)


def load_predictions(path: Path = PREDICTIONS_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Train the model first with: python -m src.train")
    df = pd.read_csv(path, keep_default_na=False)
    df["clean_text"] = df["clean_text"].astype(str)
    df["is_error"] = df["target"] != df["predicted"]
    # How sure the model was about the class it chose.
    df["confidence"] = df["positive_probability"].where(df["predicted"] == 1, 1 - df["positive_probability"])
    return df


def error_rates_by_length(df: pd.DataFrame) -> pd.DataFrame:
    """Error rate for short, medium and long tweets (measured in words after cleaning)."""
    words = df["clean_text"].str.split().str.len()
    bucket = pd.cut(words, bins=[0, 3, 7, 12, float("inf")], labels=["1-3", "4-7", "8-12", "13+"])
    table = df.groupby(bucket, observed=True)["is_error"].agg(tweets="size", error_rate="mean")
    return table[table["tweets"] >= MIN_BUCKET_ROWS]


def error_rates_by_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """Error rate as a function of the model's confidence (a good model is rarely wrong when sure)."""
    bucket = pd.cut(df["confidence"], bins=[0.5 - 1e-9, 0.6, 0.7, 0.8, 0.9, 1.0], labels=["50-60%", "60-70%", "70-80%", "80-90%", "90-100%"])
    table = df.groupby(bucket, observed=True)["is_error"].agg(tweets="size", error_rate="mean")
    return table[table["tweets"] >= MIN_BUCKET_ROWS]


def error_by_class(df: pd.DataFrame) -> dict[str, int]:
    errors = df[df["is_error"]]
    return {
        "false_positives": int(((errors["target"] == 0) & (errors["predicted"] == 1)).sum()),
        "false_negatives": int(((errors["target"] == 1) & (errors["predicted"] == 0)).sum()),
    }


def words_linked_to_errors(df: pd.DataFrame, top: int = TOP_WORDS, min_count: int = MIN_WORD_COUNT) -> pd.DataFrame:
    """Words that appear often and are wrongly classified more often than average."""
    # Each word counts once per tweet.
    rows = df[["clean_text", "is_error"]].assign(word=df["clean_text"].str.split().map(lambda w: sorted(set(w))))
    rows = rows.explode("word").dropna(subset=["word"])
    stats = rows.groupby("word")["is_error"].agg(tweets="size", error_rate="mean")
    stats = stats[stats["tweets"] >= min_count]
    return stats.sort_values("error_rate", ascending=False).head(top)


def confident_mistakes(df: pd.DataFrame, top: int = TOP_EXAMPLES) -> pd.DataFrame:
    errors = df[df["is_error"]].sort_values("confidence", ascending=False).head(top)
    return pd.DataFrame(
        {
            "text": errors["text"].map(mask_personal_data),
            "actual": errors["target"].map(LABELS),
            "predicted": errors["predicted"].map(LABELS),
            "confidence": errors["confidence"].round(3),
        }
    ).reset_index(drop=True)


def _neutral_lines(df: pd.DataFrame) -> list[str]:
    summary = neutral_summary(df)
    if summary is None:
        return []
    return [
        "## The Neutral zone",
        "",
        f"The model answers **Neutral** for {summary['neutral_share']:.1%} of the test tweets. "
        f"If it had been forced to choose a side for those tweets, it would have been wrong "
        f"{summary['error_rate_if_forced']:.1%} of the time. "
        f"On the other tweets, accuracy rises to **{summary['accuracy_on_decided']:.1%}**.",
        "",
    ]


def _markdown_table(table: pd.DataFrame, index_name: str) -> str:
    lines = [f"| {index_name} | Tweets | Error rate |", "|---|---:|---:|"]
    for name, row in table.iterrows():
        lines.append(f"| {name} | {int(row['tweets']):,} | {row['error_rate']:.1%} |")
    return "\n".join(lines)


def neutral_summary(df: pd.DataFrame) -> dict | None:
    """How the Neutral zone changes the picture (None for files made before Neutral existed)."""
    if "is_neutral" not in df.columns:
        return None
    neutral = df["is_neutral"].astype(str).str.lower() == "true"
    decided = df[~neutral]
    if neutral.sum() == 0 or decided.empty:
        return None
    return {
        "neutral_share": float(neutral.mean()),
        "error_rate_if_forced": float(df[neutral]["is_error"].mean()),
        "accuracy_on_decided": float(1 - decided["is_error"].mean()),
    }


def build_report(df: pd.DataFrame) -> str:
    total = len(df)
    errors = int(df["is_error"].sum())
    by_class = error_by_class(df)
    examples = confident_mistakes(df)

    parts = [
        "# Error analysis",
        "",
        f"The best model misclassified **{errors:,} of {total:,}** test tweets "
        f"(accuracy {1 - errors / total:.1%}).",
        "",
        "## Type of mistake",
        "",
        f"- Negative tweets predicted as Positive: **{by_class['false_positives']:,}**",
        f"- Positive tweets predicted as Negative: **{by_class['false_negatives']:,}**",
        "",
        *_neutral_lines(df),
        "## Error rate by tweet length (words after cleaning)",
        "",
        _markdown_table(error_rates_by_length(df), "Words"),
        "",
        "## Error rate by model confidence",
        "",
        _markdown_table(error_rates_by_confidence(df), "Confidence"),
        "",
        "## Words most associated with mistakes",
        "",
        f"Words that appear in at least {MIN_WORD_COUNT} test tweets, ordered by error rate.",
        "",
        _markdown_table(words_linked_to_errors(df), "Word"),
        "",
        "## Most confident mistakes",
        "",
        "Many of these are labeling noise or sarcasm: Sentiment140 labels come from emoticons, not from people.",
        "",
        "| Tweet | Actual | Predicted | Confidence |",
        "|---|---|---|---:|",
    ]
    for _, row in examples.iterrows():
        text = row["text"].replace("|", "/").replace("\n", " ")
        parts.append(f"| {text} | {row['actual']} | {row['predicted']} | {row['confidence']:.0%} |")
    parts.append("")
    return "\n".join(parts)


def run(predictions_path: Path = PREDICTIONS_PATH, report_path: Path = REPORT_PATH, examples_path: Path = EXAMPLES_PATH) -> str:
    df = load_predictions(predictions_path)
    report = build_report(df)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    confident_mistakes(df, top=200).to_csv(examples_path, index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the mistakes of the trained model.")
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_PATH)
    args = parser.parse_args()
    try:
        run(args.predictions)
    except FileNotFoundError as error:
        raise SystemExit(error)
    print(f"Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
