"""Priority maths, quadrant thresholds, trend/alert rules (SQLite for the DB parts)."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from db.models import Alert, Analysis, Mention, Topic
from nlp.priority import (
    FIX_NOW, IGNORE, MONITOR, QUICK_WINS, MentionPoint, alert_message, anger_disgust, negativity,
    pain_score, quadrant, run_priority, score_topic, should_alert, trend_ratio, volume_score, week_start,
)
from sources.base import BrandSpec

AS_OF = datetime(2026, 10, 14, 12, 0)  # a Wednesday


def point(days_ago: float, score: float = -0.8, anger: float = 0.3, disgust: float = 0.2) -> MentionPoint:
    return MentionPoint(AS_OF - timedelta(days=days_ago), score, {"anger": anger, "disgust": disgust, "joy": 0.1})


# ---------------------------------------------------------------- maths


def test_volume_is_log_scaled() -> None:
    assert volume_score(0) == 0
    assert volume_score(9) == pytest.approx(math.log(10))
    assert volume_score(99) - volume_score(9) == pytest.approx(math.log(10))  # 10x mentions = +log 10


def test_negativity_ignores_positive_and_missing() -> None:
    assert negativity(-0.75) == 0.75
    assert negativity(0.6) == 0.0
    assert negativity(None) == 0.0


def test_anger_disgust() -> None:
    assert anger_disgust({"anger": 0.25, "disgust": 0.5, "joy": 0.9}) == 0.75
    assert anger_disgust(None) == 0.0


def test_pain_is_mean_negativity_times_mean_anger_disgust() -> None:
    points = [point(1, -1.0, 0.4, 0.2), point(2, 0.5, 0.0, 0.0)]  # negativity 1 and 0; anger+disgust .6 and 0
    assert pain_score(points) == pytest.approx(0.5 * 0.3)
    assert pain_score([]) is None
    assert pain_score([point(1, 0.9)]) == 0.0  # praise is never painful


def test_trend_ratio() -> None:
    assert trend_ratio(10, 4) == 2.5
    assert trend_ratio(3, 0) is None


@pytest.mark.parametrize(
    ("volume", "pain", "expected"),
    [
        (5, 0.15, FIX_NOW),  # both exactly at the thresholds count as high
        (20, 0.40, FIX_NOW),
        (4, 0.15, QUICK_WINS),  # few mentions, but painful
        (5, 0.149, MONITOR),  # many mild complaints
        (4, 0.149, IGNORE),
        (0, None, IGNORE),
    ],
)
def test_quadrant_thresholds(volume: int, pain: float | None, expected: str) -> None:
    assert quadrant(volume, pain, volume_threshold=5, pain_threshold=0.15) == expected


@pytest.mark.parametrize(
    ("this_week", "last_week", "expected"),
    [
        (10, 5, True),  # exactly 2x
        (9, 5, False),  # 1.8x
        (5, 0, True),  # new topic with enough mentions
        (4, 0, False),  # below the minimum
        (4, 1, False),  # 4x, but below the minimum
        (5, 2, True),
    ],
)
def test_alert_rule(this_week: int, last_week: int, expected: bool) -> None:
    assert should_alert(this_week, last_week, ratio=2.0, min_mentions=5) is expected


def test_week_start_is_monday_midnight() -> None:
    assert week_start(AS_OF) == datetime(2026, 10, 12)
    assert week_start(datetime(2026, 10, 12, 0, 0)) == datetime(2026, 10, 12)
    assert week_start(datetime(2026, 10, 18, 23, 59)) == datetime(2026, 10, 12)


def test_score_topic_windows() -> None:
    points = [point(0.5), point(6.9), point(7.0), point(7.1), point(13.9), point(14.1)]
    s = score_topic(1, 1, "Bug – sync", points, AS_OF, 5, 0.15, 2.0, 5)
    assert s.volume_7d == 2  # (as_of - 7d, as_of]; exactly 7 days ago belongs to last week
    assert s.volume_prev_7d == 3
    assert s.trend == pytest.approx(2 / 3, abs=1e-3)
    assert s.pain == pytest.approx(0.8 * 0.5)
    assert s.quadrant == QUICK_WINS
    assert s.alert is False


def test_score_topic_pain_falls_back_to_older_mentions() -> None:
    s = score_topic(1, 1, "x", [point(10, -0.5)], AS_OF, 5, 0.15, 2.0, 5)
    assert s.volume_7d == 0 and s.pain == pytest.approx(0.25)
    assert s.quadrant == QUICK_WINS


def test_inactive_topic_has_no_quadrant() -> None:
    s = score_topic(1, 1, "x", [], AS_OF, 5, 0.15, 2.0, 5)
    assert (s.volume_7d, s.pain, s.trend, s.quadrant, s.alert) == (0, None, None, None, False)


def test_alert_message() -> None:
    new = score_topic(1, 1, "Pricing – ads", [point(1)] * 6, AS_OF, 5, 0.15, 2.0, 5)
    assert alert_message(new) == "Spike: Pricing – ads — 6 complaints in 7 days (new this week)"
    spike = score_topic(1, 1, "Bug – sync", [point(1)] * 8 + [point(9)] * 2, AS_OF, 5, 0.15, 2.0, 5)
    assert "4.0× last week's 2" in alert_message(spike)


# ---------------------------------------------------------------- database


def add_topic(factory: sessionmaker[Session], brand: BrandSpec, days_ago: list[float], label: str = "Bug – sync") -> int:
    with factory() as s:
        topic = Topic(brand_id=brand.id, label=label)
        s.add(topic)
        s.flush()
        for i, d in enumerate(days_ago):
            m = Mention(brand_id=brand.id, source="test", external_id=f"{label}-{i}", text="sync broke",
                        created_at=AS_OF - timedelta(days=d))
            s.add(m)
            s.flush()
            s.add(Analysis(mention_id=m.id, sentiment="negative", sentiment_score=-0.9,
                           emotion_scores={"anger": 0.5, "disgust": 0.2}, topic_id=topic.id))
        s.commit()
        return topic.id


def test_run_priority_writes_snapshot(factory: sessionmaker[Session], notion: BrandSpec) -> None:
    tid = add_topic(factory, notion, [1, 2, 3, 4, 5, 6, 8])
    quiet = add_topic(factory, notion, [], label="Old topic")
    scores, alerts = run_priority(factory, as_of=AS_OF, volume_threshold=5, pain_threshold=0.15)
    with factory() as s:
        t = s.get(Topic, tid)
        assert (t.volume_7d, t.volume_prev_7d, t.trend, t.quadrant) == (6, 1, 6.0, FIX_NOW)
        assert t.pain == pytest.approx(0.9 * 0.7)
        assert t.scored_at == AS_OF
        assert s.get(Topic, quiet).quadrant is None
    assert [a.topic_id for a in alerts] == [tid]


def test_one_alert_per_topic_per_week(factory: sessionmaker[Session], notion: BrandSpec) -> None:
    add_topic(factory, notion, [1, 2, 3, 4, 5])
    _, first = run_priority(factory, as_of=AS_OF)
    _, again = run_priority(factory, as_of=AS_OF + timedelta(days=1))  # same ISO week
    assert len(first) == 1 and again == []
    with factory() as s:
        assert s.scalars(select(Alert)).one().kind == "spike"


def test_alert_allowed_again_next_week(factory: sessionmaker[Session], notion: BrandSpec) -> None:
    tid = add_topic(factory, notion, [1, 2, 3, 4, 5])
    run_priority(factory, as_of=AS_OF)
    add_topic_mentions = [AS_OF + timedelta(days=5, hours=h) for h in range(12)]  # big spike next week
    with factory() as s:
        for i, when in enumerate(add_topic_mentions):
            m = Mention(brand_id=notion.id, source="test", external_id=f"next-{i}", text="x", created_at=when)
            s.add(m)
            s.flush()
            s.add(Analysis(mention_id=m.id, sentiment="negative", sentiment_score=-0.9, topic_id=tid))
        s.commit()
    _, next_week = run_priority(factory, as_of=AS_OF + timedelta(days=6))
    assert [a.topic_id for a in next_week] == [tid]
