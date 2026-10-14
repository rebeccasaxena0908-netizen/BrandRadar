"""Demo seed (SQLite) produces a complete dataset; collector status ('Updated … ago') logic."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

import seed_demo
from db.models import Alert, Analysis, Base, Brand, CollectorRun, Mention, Topic, TopicDaily
from ui import data, insights

NOW = datetime(2026, 10, 14, 12, 0)


@pytest.fixture(scope="module")
def demo_engine():  # type: ignore[no-untyped-def]
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    seed_demo.seed(engine, now=NOW)
    yield engine
    engine.dispose()


def count(engine, model, *where) -> int:  # type: ignore[no-untyped-def]
    with Session(engine) as s:
        return s.scalar(select(func.count()).select_from(model).where(*where)) or 0


# ---------------------------------------------------------------- demo seed


def test_demo_fills_every_table(demo_engine) -> None:  # type: ignore[no-untyped-def]
    assert count(demo_engine, Brand) == 12  # Note apps, Skincare and Stationery
    assert count(demo_engine, Mention) > 2500
    assert count(demo_engine, Analysis) == count(demo_engine, Mention)  # every mention analysed
    assert count(demo_engine, TopicDaily) > 0
    assert count(demo_engine, Alert) >= 2
    assert count(demo_engine, CollectorRun, CollectorRun.status == "ok") == 1


def test_demo_every_brand_is_full(demo_engine) -> None:  # type: ignore[no-untyped-def]
    with Session(demo_engine) as s:
        for brand in s.scalars(select(Brand)):
            mentions = count(demo_engine, Mention, Mention.brand_id == brand.id)
            topics = list(s.scalars(select(Topic).where(Topic.brand_id == brand.id)))
            assert mentions >= 100, brand.name
            assert len(topics) >= 3 and all(t.quadrant for t in topics), brand.name  # priority snapshot written
            recent = count(demo_engine, Mention, Mention.brand_id == brand.id, Mention.created_at >= NOW - timedelta(days=14))
            assert recent >= 30, brand.name  # the default 14-day view is never thin
            sentiments = {a for a in s.scalars(select(Analysis.sentiment).join(Mention).where(Mention.brand_id == brand.id))}
            assert sentiments == {"negative", "neutral", "positive"}, brand.name


def test_demo_groups_use_their_profile_and_each_own_brand_has_an_alert(demo_engine) -> None:  # type: ignore[no-untyped-def]
    from nlp.profiles import get_profile

    with Session(demo_engine) as s:
        brands = list(s.scalars(select(Brand)))
        assert {b.brand_group for b in brands} == {"Note apps", "Skincare", "Stationery"}
        for brand in brands:
            profile = get_profile(brand.profile)
            categories = set(s.scalars(select(Analysis.category).join(Mention).where(Mention.brand_id == brand.id)))
            assert categories <= set(profile.categories), brand.name
            for topic in s.scalars(select(Topic).where(Topic.brand_id == brand.id)):
                assert topic.category in profile.complaints, (brand.name, topic.label)
        own = [b for b in brands if b.is_own]
        assert len(own) == 3
        for brand in own:
            topic_ids = set(s.scalars(select(Topic.id).where(Topic.brand_id == brand.id)))
            assert s.scalar(select(func.count()).select_from(Alert).where(Alert.topic_id.in_(topic_ids))), brand.name
        sources = set(s.scalars(select(Mention.source).join(Brand).where(Brand.brand_group == "Stationery")))
        assert sources == {"youtube", "news"}
        rated = s.scalar(select(func.count()).select_from(Mention).where(Mention.rating.is_not(None),
                                                                          Mention.source.not_in(("appstore", "playstore"))))
        assert rated == 0  # only store reviews have stars


def test_demo_spans_60_days_and_all_quadrants(demo_engine) -> None:  # type: ignore[no-untyped-def]
    with Session(demo_engine) as s:
        oldest = s.scalar(select(func.min(Mention.created_at)))
        quadrants = set(s.scalars(select(Topic.quadrant)))
    assert NOW - oldest > timedelta(days=55)
    assert {"Fix now", "Quick wins", "Ignore"} <= quadrants


def test_demo_feeds_the_feature_radar_for_every_brand(demo_engine) -> None:  # type: ignore[no-untyped-def]
    with Session(demo_engine) as s:
        ids = list(s.scalars(select(Brand.id)))
    features = data.query_feature_mentions(demo_engine, tuple(ids))
    for brand_id in ids:
        table = insights.feature_table(features, [brand_id])
        assert len(table.features) >= insights.MIN_RADAR_FEATURES


def test_demo_has_no_real_people_or_links(demo_engine) -> None:  # type: ignore[no-untyped-def]
    with Session(demo_engine) as s:
        authors = list(s.scalars(select(Mention.author)))
        urls = list(s.scalars(select(Mention.url)))
    assert all(a and a[-1].isdigit() and a[0].islower() for a in authors)  # generated handles like quietfox42
    assert all(u is None for u in urls)


def test_demo_is_deterministic() -> None:
    def build() -> list[tuple]:
        engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
        Base.metadata.create_all(engine)
        seed_demo.seed(engine, now=NOW)
        with Session(engine) as s:
            return [(m.brand_id, m.text, m.created_at) for m in s.scalars(select(Mention).order_by(Mention.id))]

    assert build() == build()


def test_schema_statements_target_the_demo_database() -> None:
    statements = seed_demo.schema_statements("brand_radar_demo")
    assert statements[0] == "CREATE DATABASE IF NOT EXISTS brand_radar_demo\n  CHARACTER SET utf8mb4\n  COLLATE utf8mb4_unicode_ci"
    assert statements[1] == "USE brand_radar_demo"
    assert not any("USE brand_radar\n" in s or s.endswith("USE brand_radar") for s in statements)
    assert sum(s.startswith("CREATE TABLE") for s in statements) == 13
    assert all("--" not in s and s.count("(") == s.count(")") for s in statements)  # no comment split a statement


def test_seed_refuses_the_real_database(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("DB_NAME", "brand_radar")
    monkeypatch.setenv("DEMO_DB_NAME", "brand_radar")
    assert seed_demo.main([]) == 1
    assert "refusing" in capsys.readouterr().out


# ---------------------------------------------------------------- collector status


@pytest.mark.parametrize(
    ("minutes", "expected"),
    [(0.5, "just now"), (12, "12 min ago"), (59, "59 min ago"), (60, "1 h ago"), (47 * 60, "47 h ago"), (49 * 60, "2 days ago")],
)
def test_age_text(minutes: float, expected: str) -> None:
    assert data.age_text(timedelta(minutes=minutes)) == expected


def latest(status: str, minutes_ago: float) -> dict:
    return {"status": status, "started_at": NOW - timedelta(minutes=minutes_ago + 2),
            "finished_at": None if status == "running" else NOW - timedelta(minutes=minutes_ago)}


def test_status_fresh_ok() -> None:
    s = data.collector_status(latest("ok", 12), NOW - timedelta(minutes=12), NOW, 6)
    assert (s.text, s.warning, s.running) == ("Updated 12 min ago", None, False)


def test_status_errors_and_failures_warn() -> None:
    assert data.collector_status(latest("partial", 5), NOW - timedelta(minutes=5), NOW, 6).warning == "last cycle had errors"
    failed = data.collector_status(latest("failed", 5), NOW - timedelta(hours=7), NOW, 6)
    assert failed.text == "Updated 7 h ago" and failed.warning == "last cycle failed"


def test_status_overdue_after_twice_the_interval() -> None:
    assert data.collector_status(latest("ok", 0), NOW - timedelta(hours=11, minutes=59), NOW, 6).warning is None
    assert "overdue" in data.collector_status(latest("ok", 0), NOW - timedelta(hours=12, minutes=1), NOW, 6).warning


def test_status_running_and_no_runs() -> None:
    running = data.collector_status(latest("running", 0), NOW - timedelta(hours=6), NOW, 6)
    assert running.running and running.warning is None and running.text == "Updated 6 h ago"
    none = data.collector_status(None, None, NOW, 6)
    assert none.text == "No collector runs yet" and not none.has_runs
    first_failed = data.collector_status(latest("failed", 1), None, NOW, 6)
    assert first_failed.text == "No successful cycle yet" and first_failed.warning == "last cycle failed"


def test_query_collector_runs(demo_engine) -> None:  # type: ignore[no-untyped-def]
    latest_run, last_ok = data.query_collector_runs(demo_engine)
    assert latest_run["status"] == "ok" and isinstance(last_ok, datetime)
    assert last_ok == NOW - timedelta(minutes=4)


# ---------------------------------------------------------------- dashboard header pills


def test_header_shows_status_and_demo_pill(demo_engine, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    import dataclasses

    import streamlit as st
    from streamlit.testing.v1 import AppTest

    import config

    monkeypatch.setattr(data, "get_engine", lambda: demo_engine)
    monkeypatch.setattr(config, "settings", dataclasses.replace(config.settings, demo=True))
    st.cache_data.clear()
    at = AppTest.from_file("app.py", default_timeout=90).run()
    assert not at.exception
    md = [m.value for m in at.markdown]
    assert any('class="br-demo-pill"' in m for m in md)
    status = next(m for m in md if 'class="br-status' in m)
    assert "Updated" in status  # fixture time is ahead of the real clock, so this reads "just now"
    # demo data fills everything: no designed empty states anywhere
    assert not any('class="br-empty"' in m for m in md)
    assert len(at.get("plotly_chart")) == 4
