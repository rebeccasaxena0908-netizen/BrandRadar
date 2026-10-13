"""Shared Plotly figures (ui/charts.py): built from plain data, no Streamlit."""

from __future__ import annotations

from datetime import date

import pandas as pd

from ui import charts, insights


def test_trend_figure_skips_thin_competitors_and_marks_the_last_value() -> None:
    trend = pd.DataFrame({
        "brand_id": [1, 1, 1, 2, 2, 3],
        "day": [date(2026, 10, d) for d in (1, 2, 3, 1, 2, 1)],
        "score": [10.0, 20.0, -5.0, 30.0, 40.0, 50.0],
        "mentions": [3, 4, 2, 5, 5, 1],
    })
    fig = charts.trend_figure(trend, (1, "Own"), [(2, "Rival"), (3, "Thin")], {"Own": "#9E4A5A", "Rival": "#E79AA6", "Thin": "#B7AEB0"})
    names = [t.name for t in fig.data]
    assert names == ["Rival", "Own", None]  # Thin has one day: no line; last trace is the end marker
    assert list(fig.data[-1].text) == ["-5"]


def test_matrix_figure_labels_all_but_ignore_bubbles() -> None:
    topics = pd.DataFrame({
        "id": [1, 2], "label": ["Sync Conflicts", "Small Thing"], "category": ["bug", "UX/design"],
        "volume_7d": [9, 1], "volume_prev_7d": [3, 1], "mentions": [40, 6], "pain": [0.3, 0.05],
        "mean_sentiment": [-0.6, -0.1],
    })
    matrix = insights.topic_matrix(topics, 5, 0.15, 0.02)
    texts = pd.DataFrame({"topic_id": [1], "text": ["Sync made duplicate copies of every daily note again today"],
                          "sentiment_score": [-0.8]})
    fig = charts.matrix_figure(matrix, texts, 5, 0.15)
    by_name = {t.name: t for t in fig.data}
    assert list(by_name["Fix now"].text) == ["Sync Conflicts"]
    assert list(by_name["Ignore"].text) == [""]
    assert len(fig.layout.shapes) == 4  # one tint per quadrant
    assert "duplicate copies" in by_name["Fix now"].hovertext[0]  # the topic's quote is in the hover


def test_radar_figure_fills_only_the_own_brand() -> None:
    features = ["sync", "pricing", "search"]
    table = insights.FeatureTable(
        features=features,
        net=pd.DataFrame([[20.0, -40.0, 10.0], [5.0, None, 30.0]], index=[1, 2], columns=features),
        counts=pd.DataFrame([[9, 8, 7], [6, 2, 5]], index=[1, 2], columns=features),
    )
    fig = charts.radar_figure(table, (1, "Own"), [(2, "Rival")], {"Own": "#9E4A5A", "Rival": "#E79AA6"})
    own, rival = fig.data[2], fig.data[1]
    assert (own.name, own.fill) == ("Own", "toself") and rival.fill is None
    assert list(own.theta) == ["Sync", "Pricing", "Search", "Sync"]  # closed loop
