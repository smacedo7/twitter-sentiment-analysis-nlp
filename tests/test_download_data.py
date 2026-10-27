import pandas as pd
import pytest

from src.download_data import convert_raw_csv


def _write_raw(path, rows):
    # Same layout as the original file: no header, latin-1, 6 columns.
    df = pd.DataFrame(rows, columns=["target", "id", "date", "flag", "user", "text"])
    df.to_csv(path, index=False, header=False, encoding="latin-1")


def test_convert_maps_labels_and_keeps_two_columns(tmp_path):
    raw = tmp_path / "raw.csv"
    out = tmp_path / "out.csv"
    _write_raw(
        raw,
        [
            (0, 1, "Mon Apr 06", "NO_QUERY", "user1", "this is sad :("),
            (4, 2, "Mon Apr 06", "NO_QUERY", "user2", "café time, so happy!"),
        ],
    )

    df = convert_raw_csv(raw, out)

    assert list(df.columns) == ["target", "text"]
    assert df["target"].tolist() == [0, 1]
    assert out.exists()
    assert pd.read_csv(out).shape == (2, 2)


def test_convert_rejects_unexpected_labels(tmp_path):
    raw = tmp_path / "raw.csv"
    _write_raw(raw, [(2, 1, "Mon Apr 06", "NO_QUERY", "user1", "neutral tweet")])

    with pytest.raises(ValueError):
        convert_raw_csv(raw, tmp_path / "out.csv")
