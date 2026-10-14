"""Dashboard: helpers, aggregations, and a full-page AppTest run against SQLite (no MySQL, no models)."""

from __future__ import annotations

import re
import subprocess
import sys
from datetime import date, datetime, timedelta

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db.models import Alert, Analysis, Base, Brand, Mention, Topic
from ui import data
from ui.components import BERRY, COMPETITOR_COLOURS, background_html, brand_colours, delta_html

TODAY = date.today()


# ---------------------------------------------------------------- components


def test_background_has_16_bubbles_within_design_ranges() -> None:
    html = background_html()
    assert html.count('class="br-blob br-b') == 4
    bubbles = re.findall(r'<span class="br-bubble" style="([^"]+)"', html)
    assert len(bubbles) == 16
    for style in bubbles:
        size = int(re.search(r"width:(\d+)px", style).group(1))
        left = float(re.search(r"left:([\d.]+)vw", style).group(1))
        sway = int(re.search(r"--sway:(-?\d+)px", style).group(1))
        duration = float(re.search(r"animation-duration:([\d.]+)s", style).group(1))
        delay = float(re.search(r"animation-delay:(-[\d.]+)s", style).group(1))
        assert 18 <= size <= 88 and 0 <= left <= 96 and 20 <= abs(sway) <= 80
        assert 12 <= duration <= 26 and delay <= 0


def test_background_is_stable_across_reruns() -> None:
    assert background_html() == background_html()  # fixed seed: bubbles never jump
    assert background_html(seed=1) != background_html(seed=2)


def test_brand_colours() -> None:
    colours = brand_colours("Notion", ["Obsidian", "Evernote", "ClickUp", "Extra"])
    assert colours["Notion"] == BERRY
    assert [colours[n] for n in ("Obsidian", "Evernote", "ClickUp")] == list(COMPETITOR_COLOURS)
    assert colours["Extra"] == COMPETITOR_COLOURS[0]  # cycles


def test_delta_html() -> None:
    assert "no earlier data" in delta_html(None, " pts", True, 14)
    up_good = delta_html(4.2, " pts", True, 14)
    assert "▲ 4.2 pts" in up_good and "br-delta-good" in up_good and "previous 14 days" in up_good
    assert "br-delta-bad" in delta_html(-3, " pts", True, 14)
    assert "br-delta-flat" in delta_html(12, "%", None, 7)
    assert "no change" in delta_html(0, "%", None, 7)


# ---------------------------------------------------------------- aggregations


def frame(rows: list[tuple[int, datetime, float, dict | None, str]]) -> pd.DataFrame:
    return pd.DataFrame([
        {"brand_id": b, "created_at": pd.Timestamp(t), "source": "appstore", "text": "x",
         "sentiment": s, "sentiment_score": score, "emotion_scores": emo}
        for b, t, score, emo, s in rows
    ])


def test_emotion_shares_are_mean_probabilities() -> None:
    df = frame([
        (1, datetime(2026, 10, 1), 0.5, {"joy": 0.8, "neutral": 0.2}, "positive"),
        (1, datetime(2026, 10, 2), -0.5, {"anger": 0.6, "neutral": 0.4}, "negative"),
    ])
    shares = data.emotion_shares(df)
    assert shares.sum() == pytest.approx(1.0)
    assert shares["joy"] == pytest.approx(0.4) and shares["anger"] == pytest.approx(0.3)
    without_neutral = data.emotion_shares(df, exclude=("neutral",))
    assert "neutral" not in without_neutral and without_neutral["joy"] == pytest.approx(0.8 / 1.4)
    assert data.emotion_shares(df.iloc[0:0]).sum() == 0


def test_kpis_with_and_without_previous_period() -> None:
    cur = frame([(1, datetime(2026, 10, 5), 0.5, {"joy": 0.9, "anger": 0.1}, "positive"),
                 (1, datetime(2026, 10, 6), 0.1, {"joy": 0.6, "anger": 0.4}, "neutral")])
    prev = frame([(1, datetime(2026, 9, 25), -0.2, {"anger": 1.0}, "negative")])
    k = data.compute_kpis(cur, prev)
    assert (k.mentions, k.mentions_change_pct) == (2, 100.0)
    assert k.net_sentiment == 30.0 and k.net_sentiment_change == 50.0
    assert (k.top_emotion, k.runner_up_emotion) == ("joy", "anger")
    assert k.top_emotion_share == pytest.approx(0.75)
    none = data.compute_kpis(cur, prev.iloc[0:0])
    assert none.mentions_change_pct is None and none.net_sentiment_change is None
    empty = data.compute_kpis(cur.iloc[0:0], prev)
    assert empty.net_sentiment is None and empty.top_emotion is None and empty.mentions_change_pct == -100.0


def test_daily_trend_is_weighted_rolling_mean() -> None:
    df = frame([
        (1, datetime(2026, 10, 1, 9), 1.0, {}, "positive"),
        (1, datetime(2026, 10, 1, 18), 1.0, {}, "positive"),
        (1, datetime(2026, 10, 2, 12), -1.0, {}, "negative"),
    ])
    trend = data.daily_trend(df, date(2026, 10, 1), date(2026, 10, 5), window=3)
    by_day = trend.set_index("day")
    assert list(by_day["mentions"]) == [2, 1, 0, 0, 0]
    assert by_day.loc["2026-10-01", "score"] == 100.0
    assert by_day.loc["2026-10-02", "score"] == pytest.approx(33.3, abs=0.1)  # (1+1-1)/3, weighted by mentions
    assert by_day.loc["2026-10-04", "score"] == -100.0  # window 2-4 has only the -1 mention
    assert pd.isna(by_day.loc["2026-10-05", "score"])  # nothing in the window


def test_previous_period() -> None:
    assert data.previous_period(date(2026, 10, 1), date(2026, 10, 14)) == (date(2026, 9, 17), date(2026, 9, 30))


def test_cloud_stop_words_include_brands_and_generic_words() -> None:
    brands = pd.DataFrame([{"name": "Notion", "keywords": ["Notion", "notion.so", "-Notion Ink", "!Fallout"]},
                           {"name": "ClickUp", "keywords": '["clickup"]'}])
    stop = data.cloud_stop_words(brands)
    assert {"notion", "so", "ink", "fallout", "clickup", "app", "note", "use", "the"} <= stop


# ---------------------------------------------------------------- DB + full page


@pytest.fixture
def engine():  # type: ignore[no-untyped-def]
    eng = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(eng)
    with sessionmaker(bind=eng)() as s:
        notion = Brand(name="Notion", is_own=True, keywords=["Notion"], brand_group="Note apps", profile="software")
        clickup = Brand(name="ClickUp", is_own=False, keywords=["clickup"], brand_group="Note apps", profile="software")
        doms = Brand(name="DOMS", is_own=True, keywords=["DOMS"], brand_group="Stationery", profile="stationery")
        apsara = Brand(name="Apsara", is_own=False, keywords=["Apsara"], brand_group="Stationery", profile="stationery")
        s.add_all([notion, clickup, doms, apsara])
        s.flush()
        texts = ["sync is broken again and support never answers, I lost a whole afternoon of notes",
                 "love the templates and databases so much, they make planning my week easy",
                 "pricing doubled, cancelling my subscription because it is too expensive for me",
                 "fantastic editor, beautiful and fast, I use it every day for my studies"]
        features = ["sync", "databases", "pricing", "editor"]
        topic = Topic(brand_id=notion.id, label="Sync Failures", category="bug", volume_7d=7, volume_prev_7d=2,
                      pain=0.42, trend=3.5, quadrant="Fix now", scored_at=datetime.now())
        s.add(topic)
        s.flush()
        for i in range(40):
            when = datetime.combine(TODAY - timedelta(days=i % 27), datetime.min.time()) + timedelta(hours=10)
            m = Mention(brand_id=notion.id, source=("appstore", "playstore", "hackernews")[i % 3],
                        external_id=f"n{i}", text=texts[i % 4], created_at=when)
            s.add(m)
            s.flush()
            positive = i % 2 == 1
            s.add(Analysis(mention_id=m.id, sentiment="positive" if positive else "negative",
                           sentiment_score=0.7 if positive else -0.6,
                           category="praise" if positive else ("feature request" if i % 4 == 2 else "bug"),
                           feature=features[i % 4], topic_id=None if positive else topic.id,
                           emotion_scores={"joy": 0.7, "neutral": 0.3} if positive else {"anger": 0.5, "sadness": 0.3, "neutral": 0.2}))
        for i in range(2):
            m = Mention(brand_id=clickup.id, source="appstore", external_id=f"c{i}", text="slow tasks",
                        created_at=datetime.combine(TODAY, datetime.min.time()))
            s.add(m)
            s.flush()
            s.add(Analysis(mention_id=m.id, sentiment="negative", sentiment_score=-0.5, emotion_scores={"anger": 1.0}))
        s.add(Alert(brand_id=notion.id, topic_id=topic.id, kind="spike", message="Spike: Bug – old label",
                    this_week=7, last_week=2, created_at=datetime.now(), seen=False))
        s.add(Alert(brand_id=notion.id, topic_id=topic.id, kind="spike", message="Spike: older",
                    this_week=5, last_week=0, created_at=datetime.now() - timedelta(days=8), seen=True))
        s.commit()
    return eng


def test_query_mentions_filters(engine) -> None:  # type: ignore[no-untyped-def]
    brands = data.query_brands(engine)
    notion_id = int(brands.loc[brands["name"] == "Notion", "id"].iloc[0])
    start = TODAY - timedelta(days=13)
    df = data.query_mentions(engine, (notion_id,), start, TODAY, ("appstore", "playstore", "hackernews"))
    assert len(df) == 27 and df["created_at"].min().date() >= start  # 40 mentions over 27 days
    assert {"rating", "category", "feature", "topic_id"} <= set(df.columns)
    assert isinstance(df["emotion_scores"].iloc[0], dict)
    only_hn = data.query_mentions(engine, (notion_id,), start, TODAY, ("hackernews",))
    assert set(only_hn["source"]) == {"hackernews"}
    assert data.query_mentions(engine, (notion_id,), start, TODAY, ()).empty
    assert data.query_open_alerts(engine, notion_id) == 1


@pytest.fixture
def app(engine, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    from streamlit.testing.v1 import AppTest

    import streamlit as st

    monkeypatch.setattr(data, "get_engine", lambda: engine)
    st.cache_data.clear()
    return AppTest.from_file("app.py", default_timeout=60)


def kpi_text(at) -> list[str]:  # type: ignore[no-untyped-def]
    return [" ".join(re.sub("<[^>]+>", " ", m.value).split()) for m in at.markdown if 'class="glass br-kpi"' in m.value]


def test_full_page_renders_for_own_brand(app) -> None:  # type: ignore[no-untyped-def]
    at = app.run()
    assert not at.exception
    assert at.selectbox(key="brand").value == "Notion"  # is_own brand by default
    start, end = at.date_input[0].value
    assert (end - start).days == 13 and end == TODAY  # last 14 days
    kpis = kpi_text(at)
    assert len(kpis) == 4
    assert kpis[0].startswith("Net sentiment") and "vs previous 14 days" in kpis[0]  # real change, not "no earlier data"
    assert kpis[3].startswith("Open alerts 1")  # the seen alert is not counted
    # wheel, trend, priority matrix, feature radar
    assert len(at.get("plotly_chart")) == 4 and len(at.get("imgs")) == 2
    md = [m.value for m in at.markdown]
    pills = next(m for m in md if 'class="br-alert-list"' in m)
    assert pills.count('class="br-alert-pill') == 2 and pills.count("br-alert-seen") == 1
    assert "Sync Failures" in pills and "old label" not in pills  # current topic label, not the stored message
    assert "3.5×" in pills and "7 this week" in pills
    assert any("Sync Failures" in m and 'class="br-rank"' in m for m in md)  # ranked list
    summary = next(m for m in md if 'class="br-summary"' in m)
    assert summary.count("<li") == 4 and "Biggest risk" in summary
    assert any('class="br-quotes"' in m for m in md)


def test_thin_brand_shows_designed_empty_states(app) -> None:  # type: ignore[no-untyped-def]
    app.session_state["brand"] = "ClickUp"
    at = app.run()
    assert not at.exception
    empties = [m.value for m in at.markdown if 'class="br-empty"' in m.value]
    assert len(empties) >= 4  # wheel, trend, clouds, no topics (the radar still shows the competitors' features)
    assert any("No complaint topics yet" in e for e in empties)
    assert any("No spikes this week" in m.value for m in at.markdown)  # calm alerts state, not an empty card


def test_group_selector_keeps_everything_within_the_group(app) -> None:  # type: ignore[no-untyped-def]
    at = app.run()
    assert at.selectbox(key="group").options == ["Note apps", "Stationery"]
    assert at.selectbox(key="brand").options == ["Notion", "ClickUp"]
    assert at.multiselect[0].options == ["ClickUp"]


def test_groups_keep_the_order_they_were_added(engine, app) -> None:  # type: ignore[no-untyped-def]
    with sessionmaker(bind=engine)() as s:  # tracked from the dashboard later, alphabetically first
        s.add(Brand(name="Nivea", is_own=True, keywords=["nivea"], brand_group="Body lotions", profile="beauty_skincare"))
        s.commit()
    at = app.run()
    assert at.selectbox(key="group").options == ["Note apps", "Stationery", "Body lotions"]
    assert at.selectbox(key="group").value == "Note apps"  # the hand-made group stays the default


def test_other_group_starts_on_its_own_brand(app) -> None:  # type: ignore[no-untyped-def]
    app.session_state["group"] = "Stationery"
    at = app.run()
    assert not at.exception
    assert at.selectbox(key="brand").value == "DOMS" and at.selectbox(key="brand").options == ["DOMS", "Apsara"]
    assert at.multiselect[0].options == ["Apsara"]  # no note apps among the competitors
    # no mentions yet and no software core sources (Hacker News, App Store, Google Play) for stationery
    assert any("No mentions for Stationery yet" in m.value for m in at.markdown)


def test_source_options_for_other_industries_only_show_sources_with_data() -> None:
    assert data.source_options([], ()) == {}
    assert list(data.source_options(["news", "youtube", "reddit"], ())) == ["reddit", "youtube", "news"]


def test_no_sources_selected(app) -> None:  # type: ignore[no-untyped-def]
    # set state before the first run: AppTest cannot rerun a page whose multiselect options are dynamic
    app.session_state["sources"] = []
    at = app.run()
    assert not at.exception
    assert any("No sources selected" in m.value for m in at.markdown)


def test_dashboard_never_imports_the_models() -> None:
    code = "import sys, ui.data, ui.components; print('torch' in sys.modules or 'transformers' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_source_options_show_core_sources_plus_those_with_data() -> None:
    assert list(data.source_options([])) == ["hackernews", "appstore", "playstore"]
    opts = data.source_options(["playstore", "mastodon", "devto", "twitter"])
    assert list(opts) == ["hackernews", "appstore", "playstore", "devto", "mastodon", "twitter"]
    assert opts["devto"] == "dev.to" and opts["twitter"] == "Twitter"  # CSV sources get a readable name
    assert "reddit" not in opts  # no Reddit data yet -> no empty pill


def test_export_card_offers_csv_and_a_pdf_button(app) -> None:  # type: ignore[no-untyped-def]
    at = app.run()
    assert not at.exception
    assert len(at.get("download_button")) == 1  # CSV; the PDF is only built on click (Chrome)
    assert at.button(key="export-make-pdf").label == "Create PDF report"
    notes = " ".join(m.value for m in at.markdown if 'class="br-note"' in m.value)
    assert "mentions with their analysis" in notes and "YouTube comments are kept for 30 days" in notes
    assert "2–3 pages" in notes


def test_built_pdf_is_offered_for_download(app) -> None:  # type: ignore[no-untyped-def]
    # AppTest can't rerun this page after a click (dynamic multiselect options), so the state the click
    # leaves behind is set up front: (brand id, week, demo) -> PDF bytes
    from reports.weeks import Week

    week = Week.last_complete(TODAY + timedelta(days=1))
    app.session_state["pdf_report"] = ((1, week.label, False), b"%PDF-1.4 fake")
    at = app.run()
    assert not at.exception and len(at.get("download_button")) == 2
    assert not [b for b in at.button if b.key == "export-make-pdf"]  # already built: no second build button


def test_report_week_is_the_last_full_week_up_to_the_end_date() -> None:
    from reports.weeks import Week

    assert Week.last_complete(date(2026, 10, 11) + timedelta(days=1)) == Week(2026, 41)  # end on a Sunday: that week
    assert Week.last_complete(date(2026, 10, 14) + timedelta(days=1)) == Week(2026, 41)  # mid-week: the week before
