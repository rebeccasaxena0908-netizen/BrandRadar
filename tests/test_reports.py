"""Weekly PDF report and CSV export: built from the demo dataset in SQLite, charts drawn by a stub (no Chrome)."""

from __future__ import annotations

import dataclasses
import io
from datetime import date, datetime, timedelta

import pandas as pd
import pytest
from pypdf import PdfReader
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

import seed_demo
from db.models import Analysis, Base, Brand, Mention
from nlp.profiles import get_profile
from reports import weekly
from reports.export import csv_name, mentions_csv, neutralise
from reports.render import StubRenderer, embed_fonts, prepare
from reports.weeks import Week
from ui import data

NOW = datetime(2026, 10, 14, 12, 0)
W41 = Week(2026, 41)  # 5-11 Oct, the last complete week before NOW


@pytest.fixture(scope="module")
def demo_engine():  # type: ignore[no-untyped-def]
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    seed_demo.seed(engine, now=NOW)
    yield engine
    engine.dispose()


@pytest.fixture
def thin_engine():  # type: ignore[no-untyped-def]
    """A stationery group where the own brand has two mentions: every chart section must become a note."""
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        own = Brand(name="Inkwell", is_own=True, keywords=["inkwell"], brand_group="Pens", profile="stationery")
        rival = Brand(name="Quillo", is_own=False, keywords=["quillo"], brand_group="Pens", profile="stationery")
        s.add_all([own, rival])
        s.flush()
        for i, (text, sentiment, score) in enumerate([
            ("The Inkwell gel pen skips halfway through every page of my notes", "negative", -0.7),
            ("Inkwell pens are lovely and smooth for exams, very happy with them", "positive", 0.8),
        ]):
            m = Mention(brand_id=own.id, source="youtube", external_id=f"t{i}", text=text,
                        created_at=datetime(2026, 10, 7 + i, 10), fetched_at=NOW)
            s.add(m)
            s.flush()
            s.add(Analysis(mention_id=m.id, sentiment=sentiment, sentiment_score=score, category="ink/lead quality",
                           emotion="neutral", emotion_scores={"neutral": 0.6, "joy": 0.4}, model_version="t"))
        s.commit()
    yield engine
    engine.dispose()


def pdf_text(pdf: bytes) -> list[str]:
    return [page.extract_text() for page in PdfReader(io.BytesIO(pdf)).pages]


# ---------------------------------------------------------------- weeks


def test_weeks() -> None:
    assert Week.parse("2026-W41") == W41 == Week.parse("2026w41")
    assert (W41.start, W41.end, W41.label, W41.span) == (date(2026, 10, 5), date(2026, 10, 11), "2026-W41", "5–11 Oct 2026")
    assert Week.last_complete(date(2026, 10, 13)) == W41  # Tuesday -> last week
    assert Week.last_complete(date(2026, 10, 12)) == W41  # Monday -> the week that just ended
    assert Week.last_complete(date(2026, 10, 11)) == Week(2026, 40)  # Sunday: W41 isn't over yet
    assert W41.previous() == Week(2026, 40) and Week(2026, 1).previous() == Week(2025, 52)
    assert Week(2026, 44).span == "26 Oct–1 Nov 2026"
    for bad in ("2026-41", "W41", "2026-W54"):
        with pytest.raises(ValueError):
            Week.parse(bad)


# ---------------------------------------------------------------- PDF


def test_report_builds_a_valid_three_page_pdf_from_demo_data(demo_engine, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(weekly, "settings", dataclasses.replace(weekly.settings, demo=True))
    rep = weekly.gather(demo_engine, "Laneige", W41, NOW)
    assert rep.demo and [c["name"] for c in rep.comps] == ["Cosrx", "Innisfree", "SKIN1004"]  # same group only
    assert set(rep.complaints_this.index) <= set(get_profile("beauty_skincare").complaints)  # profile categories
    stub = StubRenderer()
    pdf = weekly.build_pdf(rep, stub)
    assert pdf.startswith(b"%PDF")
    pages = pdf_text(pdf)
    assert len(pages) == 3
    for page in pages:
        assert "DEMO DATA" in page and "2026-W41" in page  # demo mark and week on every page
    text = " ".join(pages)
    for heading in ("Laneige", "Executive summary", "Daily sentiment", "What people complain about", "Priority matrix",
                    "Act on these first", "Trending complaint alerts", "Feature radar", "What people are saying"):
        assert heading in text, heading
    assert "Not enough data yet" not in text  # the demo brand is full
    assert [kind for kind, *_ in stub.calls] == ["scatter", "bar", "scatter", "scatterpolar"]  # trend, categories, matrix, radar
    meta = PdfReader(io.BytesIO(pdf)).metadata
    assert meta.title == "Brand Radar weekly report: Laneige, 2026-W41"
    fonts = {str(f["/BaseFont"]) for page in PdfReader(io.BytesIO(pdf)).pages
             for f in page["/Resources"]["/Font"].values()}
    assert any("Figtree" in f for f in fonts) and any("Montserrat" in f for f in fonts)  # embedded TTFs


def test_thin_sections_render_the_designed_note(thin_engine) -> None:  # type: ignore[no-untyped-def]
    rep = weekly.gather(thin_engine, "Inkwell", W41, NOW)
    stub = StubRenderer()
    text = " ".join(pdf_text(weekly.build_pdf(rep, stub)))
    assert stub.calls == []  # no empty charts: every chart section became a note
    assert text.count("Not enough data yet") == 2  # trend and complaint categories
    assert "No complaint topics yet" in text and "Not enough feature mentions" in text
    assert "No spikes this week" in text and "Inkwell" in text and "DEMO DATA" not in text


def test_generate_names_files_by_brand_and_week(demo_engine, tmp_path) -> None:  # type: ignore[no-untyped-def]
    paths = weekly.generate(demo_engine, ["DOMS", "Notion"], W41, tmp_path, r=StubRenderer())
    assert [p.name for p in paths] == ["brandradar-doms-2026-W41.pdf", "brandradar-notion-2026-W41.pdf"]
    assert all(p.read_bytes().startswith(b"%PDF") for p in paths)


def test_default_brands_are_each_groups_own_brand(demo_engine) -> None:  # type: ignore[no-untyped-def]
    brands = data.query_brands(demo_engine)
    assert sorted(weekly.default_brands(brands)) == ["DOMS", "Laneige", "Notion"]
    assert weekly.default_brands(brands, ("own", "cosrx")) [-1] == "Cosrx"
    assert len(weekly.default_brands(brands, ("all",))) == 12


def test_untrusted_text_is_printable_and_escaped() -> None:
    assert weekly.plain("Great pen 😍 for ₹10 <b>&") == "Great pen for ₹10 &lt;b&gt;&amp;"  # emoji dropped, ₹ kept
    assert weekly.markup("Risk: <b>Ink &amp; Lead</b> — 5 this week") == "Risk: <b>Ink &amp; Lead</b> — 5 this week"


def test_complaint_counts_follow_the_profile() -> None:
    profile = get_profile("stationery")
    df = pd.DataFrame({
        "source": ["youtube", "youtube", "news", "youtube", "youtube"],
        "category": ["smudging", "smudging", "smudging", "praise", "sharpening"],
        "sentiment": ["negative", "neutral", "negative", "negative", "positive"],
    })
    # news is left out (TOPIC_EXCLUDED_SOURCES); praise isn't a complaint category; positive sharpening isn't a complaint
    assert weekly.complaint_counts(df, profile).to_dict() == {"smudging": 2}
    praise_only = df.assign(category="praise")  # nothing left after the category filter
    assert weekly.complaint_counts(praise_only, profile).empty


def test_renderer_helpers() -> None:
    import plotly.graph_objects as go

    svg = embed_fonts('<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><defs></defs></svg>')
    assert "@font-face" in svg and "font-family:'Figtree'" in svg and "data:font/ttf;base64," in svg
    days = pd.Series(pd.date_range("2026-10-01", periods=3))
    fig = prepare(go.Figure(go.Scatter(x=days, y=[1, 2, 3])), 300, 200)
    assert list(fig.data[0].x) == ["2026-10-01T00:00:00", "2026-10-02T00:00:00", "2026-10-03T00:00:00"]
    assert fig.layout.paper_bgcolor == "rgba(0,0,0,0)" and fig.layout.width == 300


# ---------------------------------------------------------------- CSV


def test_csv_export_columns_and_filters(demo_engine) -> None:  # type: ignore[no-untyped-def]
    brands = data.query_brands(demo_engine)
    laneige = int(brands.loc[brands["name"] == "Laneige", "id"].iloc[0])
    start, end = date(2026, 10, 5), date(2026, 10, 11)
    df = data.query_export_mentions(demo_engine, (laneige,), start, end, ("youtube",))
    assert list(df.columns) == data.EXPORT_COLUMNS and "author" not in df.columns
    assert set(df["brand"]) == {"Laneige"} and set(df["source"]) == {"youtube"}
    created = pd.to_datetime(df["created_at"])
    assert created.min() >= pd.Timestamp(start) and created.max() < pd.Timestamp(end + timedelta(days=1))
    assert df["topic"].notna().any() and df["category"].isin(get_profile("beauty_skincare").categories).all()
    raw = mentions_csv(df)
    assert raw.startswith("﻿".encode())  # BOM so Excel reads UTF-8
    parsed = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    assert list(parsed.columns) == data.EXPORT_COLUMNS and len(parsed) == len(df)
    assert data.query_export_mentions(demo_engine, (laneige,), start, end, ()).empty


def test_csv_neutralises_formulas() -> None:
    df = pd.DataFrame({"text": ["=HYPERLINK(\"http://x\")", "+1 love it", "-", "@team", "fine"], "rating": [5, None, 3, 1, 4]})
    parsed = pd.read_csv(io.BytesIO(mentions_csv(df)), encoding="utf-8-sig", dtype=str, keep_default_na=False)
    assert list(parsed["text"]) == ["'=HYPERLINK(\"http://x\")", "'+1 love it", "'-", "'@team", "fine"]
    assert list(parsed["rating"]) == ["5", "", "3", "1", "4"]
    assert neutralise(5) == 5
    assert csv_name("SKIN1004", date(2026, 10, 1), date(2026, 10, 14)) == "brandradar-skin1004-2026-10-01-to-2026-10-14.csv"
