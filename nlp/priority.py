"""Priority scores, quadrants and trend alerts for complaint topics.

    python -m nlp.priority                     # score every topic as of now, create alerts
    python -m nlp.priority --as-of 2026-10-12  # pretend "now" is that date (UTC)
    python -m nlp.priority --report            # also print the top topics per brand

Volume = complaint mentions in the last 7 days (the matrix plots log1p(volume)).
Pain   = mean negativity x mean P(anger)+P(disgust), negativity = max(0, -sentiment_score).
Trend  = this week / last week (NULL when last week had none).
Alert  = at least ALERT_MIN_MENTIONS this week and (trend >= ALERT_TREND_RATIO or new this week),
         at most one per topic per ISO week.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from config import settings
from db.models import Alert, Analysis, Brand, Mention, Topic
from sources.base import utc_now

log = logging.getLogger("nlp.priority")

FIX_NOW, QUICK_WINS, MONITOR, IGNORE = "Fix now", "Quick wins", "Monitor", "Ignore"
QUADRANT_ORDER = (FIX_NOW, QUICK_WINS, MONITOR, IGNORE)
ALERT_KIND = "spike"
WEEK = timedelta(days=7)


@dataclass
class MentionPoint:
    created_at: datetime
    sentiment_score: float | None
    emotion_scores: dict[str, Any] | None


@dataclass
class TopicScore:
    topic_id: int
    brand_id: int
    label: str
    volume_7d: int
    volume_prev_7d: int
    pain: float | None
    trend: float | None
    quadrant: str | None
    alert: bool = False


# ---------------------------------------------------------------- pure maths


def volume_score(mentions: int) -> float:
    """x-axis of the matrix: log-scaled so one huge topic doesn't squash the rest."""
    return math.log1p(mentions)


def negativity(sentiment_score: float | None) -> float:
    return max(0.0, -(sentiment_score or 0.0))


def anger_disgust(emotion_scores: dict[str, Any] | None) -> float:
    scores = emotion_scores or {}
    return float(scores.get("anger", 0.0)) + float(scores.get("disgust", 0.0))


def pain_score(points: Sequence[MentionPoint]) -> float | None:
    """Mean negativity x mean anger/disgust probability, in [0, 1]."""
    if not points:
        return None
    mean_neg = sum(negativity(p.sentiment_score) for p in points) / len(points)
    mean_ad = sum(anger_disgust(p.emotion_scores) for p in points) / len(points)
    return round(mean_neg * min(mean_ad, 1.0), 4)


def trend_ratio(this_week: int, last_week: int) -> float | None:
    return None if last_week == 0 else round(this_week / last_week, 3)


def quadrant(volume_7d: int, pain: float | None, volume_threshold: int, pain_threshold: float) -> str:
    high_volume = volume_7d >= volume_threshold
    high_pain = pain is not None and pain >= pain_threshold
    if high_volume and high_pain:
        return FIX_NOW
    if high_pain:
        return QUICK_WINS
    if high_volume:
        return MONITOR
    return IGNORE


def should_alert(this_week: int, last_week: int, ratio: float, min_mentions: int) -> bool:
    if this_week < min_mentions:
        return False
    return last_week == 0 or this_week / last_week >= ratio


def week_start(moment: datetime) -> datetime:
    """Monday 00:00 of the ISO week containing `moment`."""
    day = moment - timedelta(days=moment.weekday())
    return day.replace(hour=0, minute=0, second=0, microsecond=0)


def score_topic(
    topic_id: int,
    brand_id: int,
    label: str,
    points: Sequence[MentionPoint],
    as_of: datetime,
    volume_threshold: int,
    pain_threshold: float,
    ratio: float,
    min_mentions: int,
) -> TopicScore:
    this_week = [p for p in points if as_of - WEEK < p.created_at <= as_of]
    last_week = [p for p in points if as_of - 2 * WEEK < p.created_at <= as_of - WEEK]
    if not points:  # topic no longer has mentions after re-clustering: inactive
        return TopicScore(topic_id, brand_id, label, 0, 0, None, None, None)
    # Pain from this week's mentions; fall back to all of them so quiet topics still show severity.
    pain = pain_score(this_week or points)
    return TopicScore(
        topic_id=topic_id, brand_id=brand_id, label=label,
        volume_7d=len(this_week), volume_prev_7d=len(last_week),
        pain=pain, trend=trend_ratio(len(this_week), len(last_week)),
        quadrant=quadrant(len(this_week), pain, volume_threshold, pain_threshold),
        alert=should_alert(len(this_week), len(last_week), ratio, min_mentions),
    )


def alert_message(score: TopicScore) -> str:
    if score.volume_prev_7d == 0:
        change = "new this week"
    else:
        change = f"{score.volume_7d / score.volume_prev_7d:.1f}× last week's {score.volume_prev_7d}"
    return f"Spike: {score.label} — {score.volume_7d} complaints in 7 days ({change})"


# ---------------------------------------------------------------- database


def load_points(session: Session, topic_ids: Iterable[int]) -> dict[int, list[MentionPoint]]:
    points: dict[int, list[MentionPoint]] = {t: [] for t in topic_ids}
    rows = session.execute(
        select(Analysis.topic_id, Mention.created_at, Analysis.sentiment_score, Analysis.emotion_scores)
        .join(Mention, Mention.id == Analysis.mention_id)
        .where(Analysis.topic_id.in_(list(points)))
    )
    for topic_id, created_at, score, emotions in rows:
        points[topic_id].append(MentionPoint(created_at, score, emotions))
    return points


def run_priority(
    factory: sessionmaker[Session],
    as_of: datetime | None = None,
    volume_threshold: int | None = None,
    pain_threshold: float | None = None,
    ratio: float | None = None,
    min_mentions: int | None = None,
) -> tuple[list[TopicScore], list[Alert]]:
    """Score every topic, store the snapshot on `topics`, create (deduplicated) spike alerts."""
    as_of = as_of or utc_now()
    volume_threshold = volume_threshold or settings.priority_volume_threshold
    pain_threshold = settings.priority_pain_threshold if pain_threshold is None else pain_threshold
    ratio = ratio or settings.alert_trend_ratio
    min_mentions = min_mentions or settings.alert_min_mentions

    created: list[Alert] = []
    with factory() as session:
        topics = list(session.scalars(select(Topic)))
        points = load_points(session, [t.id for t in topics])
        scores = [
            score_topic(t.id, t.brand_id, t.label, points[t.id], as_of,
                        volume_threshold, pain_threshold, ratio, min_mentions)
            for t in topics
        ]
        by_id = {t.id: t for t in topics}
        monday = week_start(as_of)
        already = set(session.scalars(
            select(Alert.topic_id).where(Alert.kind == ALERT_KIND, Alert.created_at >= monday,
                                         Alert.created_at < monday + WEEK)
        ))
        for s in scores:
            topic = by_id[s.topic_id]
            topic.volume_7d, topic.volume_prev_7d = s.volume_7d, s.volume_prev_7d
            topic.pain, topic.trend, topic.quadrant, topic.scored_at = s.pain, s.trend, s.quadrant, as_of
            if s.alert and s.topic_id not in already:
                alert = Alert(brand_id=s.brand_id, topic_id=s.topic_id, kind=ALERT_KIND,
                              this_week=s.volume_7d, last_week=s.volume_prev_7d,
                              message=alert_message(s)[:1024], created_at=as_of, seen=False)
                session.add(alert)
                created.append(alert)
        session.commit()
    return scores, created


# ---------------------------------------------------------------- report


def example_quote(session: Session, topic_id: int, max_len: int = 160) -> str:
    """The most negative reasonably short mention of the topic."""
    rows = session.execute(
        select(Mention.text, Analysis.sentiment_score).join(Analysis, Analysis.mention_id == Mention.id)
        .where(Analysis.topic_id == topic_id).order_by(Analysis.sentiment_score)
    ).all()
    texts = [" ".join(t.split()) for t, _ in rows]
    short = [t for t in texts if 30 <= len(t) <= max_len] or texts
    quote = short[0] if short else ""
    return quote if len(quote) <= max_len else quote[: max_len - 1] + "…"


def print_report(factory: sessionmaker[Session], scores: Sequence[TopicScore], per_brand: int = 5) -> None:
    with factory() as session:
        brands = {b.id: b.name for b in session.scalars(select(Brand))}
        counts = dict(session.execute(
            select(Analysis.topic_id, func.count()).where(Analysis.topic_id.is_not(None)).group_by(Analysis.topic_id)
        ).all())
        for brand_id, name in brands.items():
            active = [s for s in scores if s.brand_id == brand_id and s.quadrant]
            if not active:
                continue
            active.sort(key=lambda s: (QUADRANT_ORDER.index(s.quadrant or IGNORE), -s.volume_7d, -(s.pain or 0)))
            print(f"\n{name}")
            for s in active[:per_brand]:
                trend = "new" if s.trend is None and s.volume_7d else (f"{s.trend:.1f}×" if s.trend else "–")
                print(f"  [{s.quadrant:<10}] {s.label}  ({counts.get(s.topic_id, 0)} total, "
                      f"{s.volume_7d} this wk, pain {s.pain or 0:.2f}, trend {trend})")
                print(f"               “{example_quote(session, s.topic_id)}”")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--as-of", type=lambda v: datetime.fromisoformat(v), help="UTC date/time treated as now")
    parser.add_argument("--report", action="store_true", help="print the top 5 topics per brand")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")

    from db.session import get_session_factory

    factory = get_session_factory()
    scores, alerts = run_priority(factory, as_of=args.as_of)
    active = [s for s in scores if s.quadrant]
    summary = ", ".join(f"{q} {sum(s.quadrant == q for s in active)}" for q in QUADRANT_ORDER)
    print(f"Scored {len(active)} active topics ({len(scores) - len(active)} inactive): {summary}")
    print(f"Alerts created: {len(alerts)}")
    for a in alerts:
        print(f"  ⚠ {a.message}")
    if args.report:
        print_report(factory, scores)
    return 0


if __name__ == "__main__":
    sys.exit(main())
