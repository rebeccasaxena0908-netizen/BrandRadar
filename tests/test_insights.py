"""Step 6 aggregations: quadrant split, praise-topic filter, feature selection, summary and quote picks."""

from __future__ import annotations

import math
from datetime import datetime

import pandas as pd
import pytest

from nlp.priority import FIX_NOW, IGNORE, MONITOR, QUICK_WINS
from ui import insights

VOL, PAIN, PRAISE = 5, 0.15, 0.02


def topics(*rows: tuple[int, str, int, int, float | None, float, int]) -> pd.DataFrame:
    """(id, label, volume_7d, volume_prev_7d, pain, mean_sentiment, mentions)"""
    return pd.DataFrame([
        {"id": i, "label": label, "category": "bug", "volume_7d": v, "volume_prev_7d": prev, "pain": pain,
         "mean_sentiment": mean, "mentions": n, "trend": None, "quadrant": None, "scored_at": None}
        for i, label, v, prev, pain, mean, n in rows
    ])


# ---------------------------------------------------------------- matrix


@pytest.mark.parametrize(
    ("volume", "pain", "expected"),
    [(5, 0.15, FIX_NOW), (4, 0.15, QUICK_WINS), (5, 0.149, MONITOR), (4, 0.149, IGNORE), (0, 0.0, IGNORE)],
)
def test_quadrant_split_at_config_thresholds(volume: int, pain: float, expected: str) -> None:
    m = insights.topic_matrix(topics((1, "T", volume, 0, pain, -0.5, 10)), VOL, PAIN, PRAISE)
    assert m.loc[0, "quadrant"] == expected
    assert m.loc[0, "x"] == pytest.approx(math.log1p(volume))
    assert m.loc[0, "y"] == pain


def test_quadrant_is_recomputed_not_taken_from_snapshot() -> None:
    df = topics((1, "T", 9, 1, 0.4, -0.6, 12)).assign(quadrant="Ignore")  # stale snapshot value
    assert insights.topic_matrix(df, VOL, PAIN, PRAISE).loc[0, "quadrant"] == FIX_NOW


@pytest.mark.parametrize(
    ("pain", "mean", "praise"),
    [(0.0, 0.33, True), (0.0016, 0.63, True), (0.019, 0.45, True),  # Step 4 praise topics
     (0.02, 0.40, False),  # at the limit: not praise
     (0.0, -0.10, False),  # zero pain but negative on average: keep
     (None, 0.20, True)],
)
def test_praise_topic_filter(pain: float | None, mean: float, praise: bool) -> None:
    assert insights.is_praise_topic(pain, mean, PRAISE) is praise


def test_praise_topics_hidden_unless_show_all() -> None:
    df = topics((1, "Price Rises", 10, 10, 0.47, -0.8, 61), (2, "Praise for Organising", 4, 4, 0.0, 0.33, 17),
                (3, "Short Student Praise", 2, 1, 0.0016, 0.63, 8))
    assert list(insights.topic_matrix(df, VOL, PAIN, PRAISE)["label"]) == ["Price Rises"]
    shown = insights.topic_matrix(df, VOL, PAIN, PRAISE, show_all=True)
    assert len(shown) == 3 and shown["praise"].sum() == 2


def test_missing_pain_is_treated_as_zero() -> None:
    m = insights.topic_matrix(topics((1, "T", 2, 0, None, -0.4, 3)), VOL, PAIN, PRAISE)
    assert m.loc[0, "y"] == 0.0 and m.loc[0, "quadrant"] == IGNORE


def test_ranked_urgent_orders_fix_now_then_quick_wins() -> None:
    df = topics((1, "QW small", 1, 0, 0.5, -0.8, 3), (2, "Fix big", 10, 5, 0.3, -0.7, 30),
                (3, "QW big", 4, 1, 0.4, -0.6, 9), (4, "Monitor", 9, 9, 0.05, -0.2, 20))
    ranked = insights.ranked_urgent(insights.topic_matrix(df, VOL, PAIN, PRAISE))
    assert list(ranked["label"]) == ["Fix big", "QW big", "QW small"]


@pytest.mark.parametrize(
    ("this", "last", "compact", "phrase"),
    [(7, 2, "3.5×", "3.5× last week's 2"), (10, 10, "1.0×", "same as last week"), (5, 0, "new", "new this week"),
     (3, 4, "0.8×", "down from 4 last week"), (5, 4, "1.2×", "up from 4 last week"), (0, 3, "0.0×", "down from 3 last week")],
)
def test_trend_wording(this: int, last: int, compact: str, phrase: str) -> None:
    assert insights.trend_text(this, last) == compact
    assert insights.change_phrase(this, last) == phrase


# ---------------------------------------------------------------- feature radar


def feature_rows(spec: dict[tuple[int, str], list[float]]) -> pd.DataFrame:
    return pd.DataFrame([{"brand_id": b, "feature": f, "sentiment_score": s, "created_at": datetime(2026, 10, 1)}
                         for (b, f), scores in spec.items() for s in scores])


def test_feature_selection_and_point_threshold() -> None:
    df = feature_rows({
        (1, "AI"): [-0.2] * 12, (2, "AI"): [-0.8] * 5,
        (1, "editor"): [0.5] * 2, (2, "editor"): [0.4] * 6,  # brand 1 has too few for a point
        (1, "pricing"): [-0.9] * 4, (2, "pricing"): [-0.5] * 4,  # nobody reaches 5: feature dropped
        (1, "sync"): [0.1] * 5,
        (3, "plugins"): [0.9] * 20,  # brand 3 is not shown
    })
    table = insights.feature_table(df, [1, 2])
    assert table.features == ["AI", "editor", "sync"]  # by total mentions, only qualifying ones
    assert table.net.loc[1, "AI"] == -20 and table.net.loc[2, "AI"] == -80
    assert pd.isna(table.net.loc[1, "editor"]) and table.counts.loc[1, "editor"] == 2
    assert pd.isna(table.net.loc[2, "sync"]) and table.counts.loc[2, "sync"] == 0


def test_feature_table_empty() -> None:
    table = insights.feature_table(feature_rows({}).reindex(columns=["brand_id", "feature", "sentiment_score"]), [1])
    assert table.features == []


# ---------------------------------------------------------------- quotes


def quote_frame(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame([{"brand_id": 1, "created_at": pd.Timestamp(2026, 10, i + 1), "source": "appstore",
                          "text": t, "rating": None, "sentiment": s, "sentiment_score": score}
                         for i, (t, s, score) in enumerate(rows)])


def test_pick_quotes_one_of_each_readable_and_strongest() -> None:
    df = quote_frame([
        ("The sync has been broken for weeks and support never replies to my emails at all.", "negative", -0.95),
        ("It is a bit slow when I open large pages, which is annoying in meetings.", "negative", -0.40),
        ("I use it for my notes and my tasks, and it does what I need for work.", "neutral", 0.05),
        ("I use it for school and for planning the week with my study group too.", "neutral", -0.30),
        ("I really love the templates, they save me so much time every single day!", "positive", 0.97),
        ("bad", "negative", -0.99),  # too short
        ("Une application qui ne marche pas du tout, je suis très déçu de la mise à jour.", "negative", -0.98),  # not English
        ("Read this https://example.com because the app is so terrible these days honestly", "negative", -0.97),  # URL
    ])
    picks = insights.pick_quotes(df)
    assert picks["negative"]["text"].startswith("The sync has been broken")
    assert picks["neutral"]["sentiment_score"] == 0.05  # closest to 0
    assert picks["positive"]["text"].startswith("I really love")


def test_topic_quotes_most_negative_and_shortened() -> None:
    texts = pd.DataFrame({"topic_id": [1, 1, 1, 2], "sentiment_score": [-0.2, -0.9, -0.5, -1.0], "text": [
        "It is fine but slow to load on my phone sometimes",
        "Absolutely terrible since the update, it crashes every time I open a page and I lose my notes " * 2,
        "Sync stopped working on my laptop and I am annoyed",
        "Other topic quote that should not appear here at all",
    ]})
    quotes = insights.topic_quotes(texts, 1, n=2, max_len=60)
    assert len(quotes) == 2 and quotes[0].startswith("Absolutely terrible") and quotes[0].endswith("…")
    assert len(quotes[0]) <= 60 and quotes[1].startswith("Sync stopped")


# ---------------------------------------------------------------- summary


def summary(**overrides):  # type: ignore[no-untyped-def]
    matrix = insights.topic_matrix(topics((1, "Price Rises", 10, 10, 0.47, -0.8, 61), (2, "Daily Note", 7, 2, 0.12, -0.3, 16)),
                                   VOL, PAIN, PRAISE)
    table = insights.feature_table(feature_rows({(1, "editor"): [0.4] * 6, (2, "editor"): [-0.2] * 6,
                                                 (1, "AI"): [-0.5] * 6, (2, "AI"): [0.1] * 6}), [1, 2])
    mentions = pd.DataFrame({"brand_id": [1, 1, 1, 2], "category": ["feature request"] * 4,
                             "feature": ["AI", "AI", None, "sync"]})
    args = dict(brand_name="Notion", matrix=matrix, table=table, brand_id=1, competitor_ids=[2],
                mentions=mentions, net_sentiment=32.0, net_change=10.4, days=14)
    args.update(overrides)
    return insights.summary_picks(**args)


def test_summary_lines() -> None:
    s = summary()
    assert s.risk == "Biggest risk: <b>Price Rises</b> — 10 complaints this week, same as last week."
    assert s.win == "Biggest win: <b>Editor</b> — +40 net sentiment vs -20 for competitors."
    assert s.request == "Top request: <b>AI</b> (2 of 3 feature requests)."
    assert s.change == "Net sentiment is <b>+32</b>, up 10 points on the previous 14 days."


def test_summary_fallbacks() -> None:
    quiet = insights.topic_matrix(topics((2, "Daily Note", 7, 2, 0.12, -0.3, 16)), VOL, PAIN, PRAISE)
    assert summary(matrix=quiet).risk.startswith("No urgent risks. Watch: <b>Daily Note</b> — 7 mild complaints")
    assert summary(matrix=quiet.iloc[0:0]).risk == "No urgent complaint themes this week."
    stale = insights.topic_matrix(topics((3, "Files", 0, 1, 0.23, -0.5, 5)), VOL, PAIN, PRAISE)  # Quick win, quiet now
    assert summary(matrix=stale).risk == "No urgent complaint themes this week."
    thin = insights.feature_table(feature_rows({(1, "editor"): [0.9] * 3, (2, "editor"): [-0.2] * 6}), [1, 2])
    assert summary(table=thin).win == "No clear win over competitors yet."  # 3 mentions is too few to claim
    one = pd.DataFrame({"brand_id": [1], "category": ["feature request"], "feature": ["AI"]})
    assert summary(mentions=one).request == "1 feature request; no single feature stands out yet."
    assert "No feature requests" in summary(mentions=one.iloc[0:0]).request
    assert summary(net_change=None).change.endswith("(no earlier data to compare).")
    assert summary(net_change=-1.0).change.endswith("down 1 point on the previous 14 days.")
    assert summary(net_sentiment=None).change == "No Notion mentions in this period."


def test_summary_escapes_labels() -> None:
    evil = insights.topic_matrix(topics((1, "<script>x</script>", 10, 10, 0.5, -0.8, 20)), VOL, PAIN, PRAISE)
    assert "<script>" not in summary(matrix=evil).risk
