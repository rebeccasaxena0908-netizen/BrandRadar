"""Read-only data access and aggregations for the dashboard.

Queries only read MySQL (brands, mentions, analysis, alerts). The cached wrappers at the bottom are
what app.py calls; the plain functions take an engine/DataFrame so they can be tested without
Streamlit or MySQL. Nothing here imports the NLP models.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any

import pandas as pd
import streamlit as st
from sqlalchemy import Engine, bindparam, text

from nlp.models import EMOTIONS

SOURCES = {"hackernews": "Hacker News", "appstore": "App Store", "playstore": "Google Play"}
# Generic words that say nothing in a word cloud of note-app reviews (brand names are added per call).
CLOUD_STOP_WORDS = {
    "app", "apps", "application", "use", "using", "used", "note", "notes", "just", "like", "really",
    "get", "got", "would", "also", "one", "even", "im", "ive", "dont", "doesnt", "cant", "thing",
    "things", "make", "makes", "please", "need", "want", "way", "lot", "time", "still", "now",
    "new", "much", "many", "every", "know", "think", "people", "work", "works", "will", "can",
    "going", "say", "said", "yeah", "actually", "though", "something", "anything", "everything",
    "year", "years", "day", "days", "since", "version", "update", "hn", "show",
}


# ---------------------------------------------------------------- queries (plain)


def query_brands(engine: Engine) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text("SELECT id, name, is_own, keywords FROM brands ORDER BY is_own DESC, name"), conn)


def query_mentions(
    engine: Engine, brand_ids: tuple[int, ...], start: date, end: date, sources: tuple[str, ...]
) -> pd.DataFrame:
    """Analysed mentions for the brands/sources with created_at in [start, end] (whole days)."""
    columns = ["brand_id", "created_at", "source", "text", "sentiment", "sentiment_score", "emotion_scores"]
    if not brand_ids or not sources:
        return pd.DataFrame(columns=columns)
    stmt = text(
        "SELECT m.brand_id, m.created_at, m.source, m.text, a.sentiment, a.sentiment_score, a.emotion_scores "
        "FROM mentions m JOIN analysis a ON a.mention_id = m.id "
        "WHERE m.brand_id IN :brands AND m.source IN :sources "
        "AND m.created_at >= :start AND m.created_at < :end AND a.sentiment IS NOT NULL"
    ).bindparams(bindparam("brands", expanding=True), bindparam("sources", expanding=True))
    with engine.connect() as conn:
        df = pd.read_sql(stmt, conn, params={
            "brands": list(brand_ids), "sources": list(sources),
            "start": datetime.combine(start, time.min), "end": datetime.combine(end + timedelta(days=1), time.min),
        })
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["emotion_scores"] = df["emotion_scores"].map(_parse_json)
    return df[columns]


def query_open_alerts(engine: Engine, brand_id: int) -> int:
    with engine.connect() as conn:
        return int(conn.execute(
            text("SELECT COUNT(*) FROM alerts WHERE brand_id = :b AND seen = 0"), {"b": brand_id}
        ).scalar_one())


def _parse_json(value: Any) -> dict[str, float]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return {}


# ---------------------------------------------------------------- aggregations (pure)


@dataclass
class Kpis:
    mentions: int
    mentions_change_pct: float | None  # None when the previous period has no mentions
    net_sentiment: float | None  # mean signed score x 100
    net_sentiment_change: float | None  # points
    top_emotion: str | None
    top_emotion_share: float | None
    runner_up_emotion: str | None
    runner_up_share: float | None


def emotion_shares(df: pd.DataFrame, exclude: tuple[str, ...] = ()) -> pd.Series:
    """Mean emotion probability over the mentions, renormalised to sum to 1 (aggregate only:
    the emotion model is unreliable per mention). Emotions in `exclude` are dropped first."""
    emotions = [e for e in EMOTIONS if e not in exclude]
    if df.empty:
        return pd.Series(0.0, index=emotions)
    matrix = pd.DataFrame([{e: float((s or {}).get(e, 0.0)) for e in emotions} for s in df["emotion_scores"]])
    means = matrix.mean()
    total = means.sum()
    return (means / total) if total > 0 else means


def compute_kpis(current: pd.DataFrame, previous: pd.DataFrame) -> Kpis:
    net = round(current["sentiment_score"].mean() * 100, 1) if not current.empty else None
    prev_net = round(previous["sentiment_score"].mean() * 100, 1) if not previous.empty else None
    shares = emotion_shares(current).sort_values(ascending=False)
    has_emotions = not current.empty and shares.sum() > 0
    return Kpis(
        mentions=len(current),
        mentions_change_pct=None if previous.empty else round((len(current) - len(previous)) / len(previous) * 100, 1),
        net_sentiment=net,
        net_sentiment_change=None if net is None or prev_net is None else round(net - prev_net, 1),
        top_emotion=shares.index[0] if has_emotions else None,
        top_emotion_share=float(shares.iloc[0]) if has_emotions else None,
        runner_up_emotion=shares.index[1] if has_emotions else None,
        runner_up_share=float(shares.iloc[1]) if has_emotions else None,
    )


def daily_trend(df: pd.DataFrame, start: date, end: date, window: int = 3) -> pd.DataFrame:
    """Per brand and day: mentions and a `window`-day rolling net sentiment (x100), weighted by
    mentions, so single days with one or two mentions don't swing the line. Days are reindexed so
    gaps count as zero mentions; days with no mentions in the whole window get NaN."""
    days = pd.date_range(start, end, freq="D")
    frames = []
    for brand_id, group in df.groupby("brand_id"):
        daily = group.groupby(group["created_at"].dt.normalize())["sentiment_score"].agg(["sum", "count"])
        daily = daily.reindex(days, fill_value=0)
        rolled = daily.rolling(window, min_periods=1).sum()
        score = (rolled["sum"] / rolled["count"].where(rolled["count"] > 0)) * 100
        frames.append(pd.DataFrame({"brand_id": brand_id, "day": days, "mentions": daily["count"].values,
                                    "window_mentions": rolled["count"].values, "score": score.round(1).values}))
    if not frames:
        return pd.DataFrame(columns=["brand_id", "day", "mentions", "window_mentions", "score"])
    return pd.concat(frames, ignore_index=True)


def brand_stop_words(brands: pd.DataFrame) -> set[str]:
    """Every tracked brand's name and keywords, lower-cased, so clouds aren't just brand names."""
    words: set[str] = set()
    for _, row in brands.iterrows():
        keywords = _parse_json(row.get("keywords")) if not isinstance(row.get("keywords"), list) else row["keywords"]
        terms = [row["name"], *(k.lstrip("-!") for k in (keywords or []))]
        for term in terms:
            words |= {w.lower() for w in str(term).replace(".", " ").split()}
    return words


def cloud_stop_words(brands: pd.DataFrame) -> set[str]:
    from wordcloud import STOPWORDS

    return set(STOPWORDS) | CLOUD_STOP_WORDS | brand_stop_words(brands)


def previous_period(start: date, end: date) -> tuple[date, date]:
    """The period of the same length immediately before [start, end]."""
    length = (end - start).days + 1
    return start - timedelta(days=length), start - timedelta(days=1)


# ---------------------------------------------------------------- cached wrappers for app.py


def get_engine() -> Engine:
    from db.session import get_engine as _engine

    return _engine()


@st.cache_data(ttl=60, show_spinner=False)
def brands() -> pd.DataFrame:
    return query_brands(get_engine())


@st.cache_data(ttl=60, show_spinner=False)
def mentions(brand_ids: tuple[int, ...], start: date, end: date, sources: tuple[str, ...]) -> pd.DataFrame:
    return query_mentions(get_engine(), brand_ids, start, end, sources)


@st.cache_data(ttl=60, show_spinner=False)
def open_alerts(brand_id: int) -> int:
    return query_open_alerts(get_engine(), brand_id)
