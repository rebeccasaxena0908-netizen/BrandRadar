"""Keyword gate for feature detection.

Zero-shot alone is badly overconfident on short feature names (any review "talks about" the mobile
app), so a feature is only a candidate when its name or one of its synonyms from brands.yaml appears
in the text. The zero-shot model then confirms and picks between the candidates.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

# Simple word endings, so "sync" also matches "syncs", "synced", "syncing".
SUFFIXES = r"(?:s|es|d|ed|ing|ly)?"

Features = dict[str, list[str]]


@lru_cache(maxsize=4096)
def _pattern(term: str) -> re.Pattern[str]:
    body = r"\s+".join(re.escape(word) for word in term.split())
    return re.compile(rf"(?<!\w){body}{SUFFIXES}(?!\w)", re.IGNORECASE)


def normalise_features(raw: Any) -> Features:
    """brands.features as stored ({name: [synonyms]}, or a plain list of names) -> {name: [terms]},
    where terms always include the feature name itself."""
    if not raw:
        return {}
    if isinstance(raw, list):
        raw = {name: [] for name in raw}
    return {name: list(dict.fromkeys([name, *(synonyms or [])])) for name, synonyms in raw.items()}


def matched_features(text: str, features: Features) -> list[str]:
    """Feature names with at least one term present in the text, in brands.yaml order."""
    return [name for name, terms in features.items() if any(_pattern(t).search(text) for t in terms)]
