from __future__ import annotations

import pytest

from nlp.features import matched_features, normalise_features
from nlp.text import clean_text, word_count


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Check https://notion.so/pricing now", "Check now"),
        ("see www.example.com/x?y=1 and more", "see and more"),
        ("Read [the docs](https://obsidian.md/help) first", "Read the docs first"),
        ("  lots\n\n of \t space  ", "lots of space"),
        ("Love it 😍🔥", "Love it 😍🔥"),
        ("https://only-a-link.com", ""),
        ("", ""),
        (None, ""),
    ],
)
def test_clean_text(raw: str | None, expected: str) -> None:
    assert clean_text(raw) == expected


def test_word_count() -> None:
    assert word_count("Sync is broken!") == 3
    assert word_count("👍") == 0


FEATURES = normalise_features({
    "sync": ["between devices"],
    "pricing": ["expensive", "fees", "free version"],
    "AI": ["chatgpt"],
    "performance": ["slow"],
})


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("it was better without ai.", ["AI"]),  # case-insensitive whole word
        ("Too EXPENSIVE now", ["pricing"]),
        ("syncing is broken", ["sync"]),  # simple endings
        ("sync is slow and expensive", ["sync", "pricing", "performance"]),  # brands.yaml order
        ("my news feed", []),  # 'fees' must not match 'feed'
        ("I said this before", []),  # no 'ai' inside other words
        ("stuck on the free  version", ["pricing"]),  # phrase across extra whitespace
        ("the app is great", []),
    ],
)
def test_matched_features(text: str, expected: list[str]) -> None:
    assert matched_features(text, FEATURES) == expected


def test_normalise_features() -> None:
    assert normalise_features(["sync", "AI"]) == {"sync": ["sync"], "AI": ["AI"]}
    assert normalise_features({"sync": ["sync", "syncing"]}) == {"sync": ["sync", "syncing"]}
    assert normalise_features({"AI": None}) == {"AI": ["AI"]}
    assert normalise_features(None) == {}
