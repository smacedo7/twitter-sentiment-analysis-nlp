"""Text preprocessing for tweets.

Pipeline (see ``preprocess_text``):
    1. clean_text         -> lowercase, decode HTML entities, remove links, mentions,
                             punctuation and special characters
    2. tokenize           -> split the cleaned text into words
    3. remove_stopwords   -> drop very common words that carry little sentiment

Stopword note: negation words ("not", "never", "no", ...) are deliberately KEPT.
"not happy" and "happy" have opposite meanings, so removing "not" would hurt the model.
"""

from __future__ import annotations

import html
import re

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
REPEATED_CHARS_PATTERN = re.compile(r"(.)\1{2,}")  # 3 or more of the same character in a row
NON_LETTERS_PATTERN = re.compile(r"[^a-z'\s]")  # keep letters, apostrophes (don't) and spaces
WHITESPACE_PATTERN = re.compile(r"\s+")

# Words that flip the meaning of a sentence and must never be removed.
NEGATIONS = frozenset({"no", "not", "nor", "never", "none", "nothing", "nobody", "nowhere", "neither", "cannot"})

STOPWORDS = frozenset(ENGLISH_STOP_WORDS) - NEGATIONS


def clean_text(text: str) -> str:
    """Normalize a raw tweet.

    Example:
        >>> clean_text("@user I LOOOVE this!!! http://t.co/abc #happy &amp; excited")
        'i loove this happy excited'
    """
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)  # "&amp;" -> "&", "&quot;" -> '"'
    text = text.lower()
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    text = text.replace("#", " ")  # keep the hashtag word (#happy -> happy), drop only the symbol
    text = REPEATED_CHARS_PATTERN.sub(r"\1\1", text)  # "soooo" -> "soo"
    text = NON_LETTERS_PATTERN.sub(" ", text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Split cleaned text into tokens, dropping single characters and stray apostrophes."""
    tokens = (token.strip("'") for token in text.split())
    return [token for token in tokens if len(token) > 1]


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Remove stopwords from a list of tokens (negations are preserved)."""
    return [token for token in tokens if token not in STOPWORDS]


def preprocess_text(text: str) -> str:
    """Run the full pipeline and return the processed tweet as a single string.

    Example:
        >>> preprocess_text("I am NOT happy with this service!!! http://t.co/abc")
        'not happy service'
    """
    return " ".join(remove_stopwords(tokenize(clean_text(text))))


def preprocess_series(texts: pd.Series) -> pd.Series:
    """Apply ``preprocess_text`` to every element of a pandas Series."""
    return texts.map(preprocess_text)
