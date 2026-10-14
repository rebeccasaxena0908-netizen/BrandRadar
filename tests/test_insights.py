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


# ---------------------------------------------------------------- complaint-category roll-up


def area_matrix(*rows: tuple[int, str, str, int]) -> pd.DataFrame:
    """Topics as (id, label, category, volume_7d); the roll-up only reads category and volume_7d."""
    return pd.DataFrame([{"id": i, "label": label, "category": category, "volume_7d": v, "volume_prev_7d": 0}
                         for i, label, category, v in rows])


def test_complaint_counts_follow_the_profile_and_leave_out_excluded_sources() -> None:
    from nlp.profiles import APP_CATEGORY, get_profile

    profile = get_profile("food_beverage+app")
    df = pd.DataFrame({
        "source": ["playstore", "playstore", "news", "youtube", "youtube", "youtube"],
        "category": [APP_CATEGORY, APP_CATEGORY, APP_CATEGORY, "taste", "praise", APP_CATEGORY],
        "sentiment": ["negative", "neutral", "negative", "negative", "positive", "positive"],
    })
    counts = insights.complaint_counts(df, profile, excluded_sources=("news",))
    # the neutral app mention counts (complaint category), news is left out, a positive app mention is not a complaint
    assert counts.to_dict() == {APP_CATEGORY: 2, "taste": 1}
    assert insights.complaint_counts(df.iloc[0:0], profile).empty


def test_a_clustered_complaint_counts_under_its_themes_category() -> None:
    from nlp.profiles import APP_CATEGORY, get_profile

    profile = get_profile("food_beverage+app")
    df = pd.DataFrame({
        "source": ["playstore"] * 6,
        # zero-shot said taste for two of the app theme's members; one complaint is in no topic; one is praise
        "category": [APP_CATEGORY, "taste", "taste", "taste", "delivery", "praise"],
        "sentiment": ["negative", "negative", "negative", "negative", "negative", "positive"],
        "topic_id": [1, 1, 1, None, None, 1],
    })
    themes = {1: APP_CATEGORY}
    assert insights.complaint_counts(df, profile).to_dict() == {"taste": 3, APP_CATEGORY: 1, "delivery": 1}
    counted = insights.complaint_counts(df, profile, topic_categories=themes).to_dict()
    # 4 members of the app theme (the positive one is clustered too: topics hold only complaints), then the rest
    assert counted == {APP_CATEGORY: 4, "taste": 1, "delivery": 1}
    # a theme whose dominant category is not a complaint category adds nothing
    assert insights.complaint_counts(df, profile, topic_categories={1: "praise"}).to_dict() == {"taste": 1, "delivery": 1}
    assert insights.topic_category_map(pd.DataFrame({"id": [1, 2], "category": [APP_CATEGORY, None]})) == {1: APP_CATEGORY}


def test_largest_area_is_set_against_the_top_single_theme() -> None:
    from nlp.profiles import APP_CATEGORY

    this = pd.Series({APP_CATEGORY: 72, "taste": 34, "delivery": 5})
    last = pd.Series({APP_CATEGORY: 51, "taste": 10})
    eight_app_themes = area_matrix(*[(i, f"App {i}", APP_CATEGORY, v) for i, v in enumerate((25, 10, 8, 8, 7, 5, 5, 4))],
                                   (20, "Quality", "taste", 34))
    area = insights.largest_area(this, last, eight_app_themes)
    assert area is not None and (area.category, area.name) == (APP_CATEGORY, "App and ordering problems")
    assert (area.this, area.last, area.themes, area.top_theme) == (72, 51, 8, 34) and area.larger
    assert insights.area_line(area) == ("Largest complaint area: <b>App and ordering problems</b> — 72 complaints "
                                        "across 8 themes, up from 51 last week.")
    # not worth saying when one theme is as big as the whole category, or it is just one theme
    one_big = insights.largest_area(pd.Series({"taste": 30}), pd.Series(dtype=int), area_matrix((1, "Quality", "taste", 34)))
    assert one_big is not None and not one_big.larger
    single = insights.largest_area(pd.Series({"taste": 30}), pd.Series(dtype=int), area_matrix((1, "Quality", "taste", 12)))
    assert single is not None and single.themes == 1 and not single.larger  # one theme is just that theme
    tiny = insights.largest_area(pd.Series({"taste": 3}), pd.Series(dtype=int),
                                 area_matrix((1, "A", "taste", 1), (2, "B", "taste", 1)))
    assert tiny is not None and tiny.themes == 2 and not tiny.larger  # under the minimum
    assert insights.largest_area(pd.Series(dtype=int), pd.Series(dtype=int), area_matrix()) is None
    assert insights.category_name("skin reaction") == "Skin reaction"


def test_summary_gets_the_roll_up_line_only_when_it_is_larger() -> None:
    from nlp.profiles import APP_CATEGORY

    matrix = area_matrix((1, "A", APP_CATEGORY, 6), (2, "B", APP_CATEGORY, 5), (3, "C", "taste", 4))
    big = insights.largest_area(pd.Series({APP_CATEGORY: 11}), pd.Series({APP_CATEGORY: 11}), matrix)
    assert summary(area=big).area == ("Largest complaint area: <b>App and ordering problems</b> — 11 complaints "
                                      "across 2 themes, same as last week.")
    assert summary(area=None).area == "" and summary().area == ""
    small = insights.largest_area(pd.Series({"taste": 4}), pd.Series(dtype=int), matrix)
    assert summary(area=small).area == ""

