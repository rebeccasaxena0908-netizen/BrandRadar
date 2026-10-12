"""Text cleaning before analysis."""

from __future__ import annotations

import re

MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:https?://|www\.)[^)\s]+\)")
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str | None) -> str:
    """Keep markdown link text, drop bare URLs, collapse whitespace. Emoji are kept on purpose:
    the sentiment model was trained on tweets and uses them."""
    if not text:
        return ""
    text = MARKDOWN_LINK_RE.sub(r"\1", text)
    text = URL_RE.sub(" ", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))
