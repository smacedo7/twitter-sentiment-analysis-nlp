import pandas as pd
import pytest

from src.preprocessing import (
    STOPWORDS,
    clean_text,
    preprocess_series,
    preprocess_text,
    remove_stopwords,
    tokenize,
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Check this out http://t.co/abc123", "check this out"),
        ("visit www.example.com now", "visit now"),
        ("@john thanks for the help", "thanks for the help"),
        ("Love it!!! :) #blessed", "love it blessed"),
        ("I LOVE this", "i love this"),
        ("soooooo good", "soo good"),
        ("Tom &amp; Jerry", "tom jerry"),
        ("it's 100% free, 24/7", "it's free"),
        ("   lots    of   spaces   ", "lots of spaces"),
    ],
)
def test_clean_text(raw, expected):
    assert clean_text(raw) == expected


@pytest.mark.parametrize("value", [None, float("nan"), 123, ""])
def test_clean_text_handles_non_strings(value):
    assert clean_text(value) == ""


def test_tokenize_drops_single_characters_and_stray_apostrophes():
    assert tokenize("a good 'day' b") == ["good", "day"]


def test_remove_stopwords_drops_common_words():
    assert remove_stopwords(["this", "is", "great"]) == ["great"]


def test_negations_are_kept():
    for word in ("not", "no", "never", "nothing"):
        assert word not in STOPWORDS
    assert remove_stopwords(["not", "happy"]) == ["not", "happy"]


def test_preprocess_text_full_pipeline():
    raw = "@user I am NOT happy with this service!!! http://t.co/abc"
    assert preprocess_text(raw) == "not happy service"


def test_preprocess_text_can_return_empty_string():
    assert preprocess_text("@user http://t.co/abc !!!") == ""


def test_preprocess_series_keeps_index_and_length():
    series = pd.Series(["Great day!!!", None, "I hate Mondays"], index=[10, 11, 12])

    result = preprocess_series(series)

    assert list(result.index) == [10, 11, 12]
    assert result.tolist() == ["great day", "", "hate mondays"]
