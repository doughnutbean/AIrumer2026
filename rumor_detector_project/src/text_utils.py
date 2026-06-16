"""Text preprocessing utilities for the rumor detector."""
from __future__ import annotations

import html
import re
from typing import Iterable, List

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
USER_RE = re.compile(r"@\w+")
SPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")


def preprocess_text(text: str) -> str:
    """Normalize tweet text while preserving informative lexical cues.

    Keep preprocessing light: decode the common HTML ampersand entity, map URLs to
    a stable placeholder, and normalize whitespace. Lowercasing is handled by the
    TF-IDF vectorizer so the saved pipeline remains transparent.
    """
    if text is None:
        text = ""
    text = str(text).replace("&amp;", " and ")
    text = URL_RE.sub(" URL ", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Return word-like tokens after preprocessing."""
    return TOKEN_RE.findall(preprocess_text(text))


def batch_preprocess(texts: Iterable[str]) -> List[str]:
    """Preprocess a batch of texts."""
    return [preprocess_text(t) for t in texts]
