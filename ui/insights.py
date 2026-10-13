"""Pure aggregations for the dashboard's second half: priority matrix, feature radar, alerts and the
executive summary. No Streamlit, no database: DataFrames in, DataFrames/dataclasses out."""

from __future__ import annotations

import html
import math
from dataclasses import dataclass

import pandas as pd

from nlp.priority import FIX_NOW, IGNORE, MONITOR, QUICK_WINS, quadrant
from nlp.text import clean_text
from nlp.topics import looks_english

QUADRANTS = (FIX_NOW, QUICK_WINS, MONITOR, IGNORE)
MIN_FEATURE_MENTIONS = 5  # a feature needs this many mentions for at least one shown brand
MIN_POINT_MENTIONS = 3  # a brand's radar point needs this many mentions for that feature
MIN_RADAR_FEATURES = 3
MIN_REQUESTS_FOR_TOP = 2  # a "top requested feature" needs at least this many requests


# ---------------------------------------------------------------- priority matrix


def is_praise_topic(pain: float | None, mean_sentiment: float | None, praise_pain_max: float) -> bool:
    """Topics clustered from praise or neutral discussion: (near) zero pain and positive on average."""
    return (pain is None or pain < praise_pain_max) and (mean_sentiment or 0) > 0


def topic_matrix(
    topics: pd.DataFrame, volume_threshold: int, pain_threshold: float, praise_pain_max: float, show_all: bool = False
) -> pd.DataFrame:
    """Topics placed on the matrix: x = log1p(volume_7d), y = pain, quadrant recomputed from the same
    thresholds the chart draws (so colours and dividing lines always agree)."""
    if topics.empty:
        return topics.assign(x=[], y=[], quadrant=[], praise=[])
    df = topics.copy()
    df["pain"] = df["pain"].fillna(0.0)
    df["x"] = df["volume_7d"].map(math.log1p)
    df["y"] = df["pain"]
    df["quadrant"] = [quadrant(int(v), p, volume_threshold, pain_threshold) for v, p in zip(df["volume_7d"], df["pain"])]
    df["praise"] = [is_praise_topic(p, s, praise_pain_max) for p, s in zip(df["pain"], df["mean_sentiment"])]
    if not show_all:
        df = df[~df["praise"]]
    return df.reset_index(drop=True)


def ranked_urgent(matrix: pd.DataFrame) -> pd.DataFrame:
    """Fix now first, then Quick wins; within each, by this week's volume x pain."""
    urgent = matrix[matrix["quadrant"].isin([FIX_NOW, QUICK_WINS])].copy()
    urgent["score"] = (urgent["volume_7d"] + 1) * urgent["pain"]
    urgent["order"] = urgent["quadrant"].map({FIX_NOW: 0, QUICK_WINS: 1})
    return urgent.sort_values(["order", "score"], ascending=[True, False]).reset_index(drop=True)


def trend_text(this_week: int, last_week: int) -> str:
    """Compact form for pills and hovers: '3.5×', 'new', '–'."""
    if last_week == 0:
        return "new" if this_week else "–"
    return f"{this_week / last_week:.1f}×"


def change_phrase(this_week: int, last_week: int) -> str:
    """Plain-language week-over-week change for the summary."""
    if last_week == 0:
        return "new this week" if this_week else "no recent mentions"
    if this_week == last_week:
        return "same as last week"
    if this_week >= 2 * last_week:
        return f"{this_week / last_week:.1f}× last week's {last_week}"
    return f"{'up' if this_week > last_week else 'down'} from {last_week} last week"


def feature_name(feature: str) -> str:
    """'integrations' -> 'Integrations'; 'AI' stays 'AI' (same as the radar axis)."""
    return feature.title() if feature.islower() else feature


# ---------------------------------------------------------------- quotes


def readable(text: str, low: int = 60, high: int = 220) -> bool:
    cleaned = clean_text(text)
    return low <= len(cleaned) <= high and "http" not in text and looks_english(cleaned) and not cleaned.startswith("Show HN")


def topic_quotes(texts: pd.DataFrame, topic_id: int, n: int = 2, max_len: int = 110) -> list[str]:
    """The n most negative readable mentions of a topic, shortened for a hover."""
    rows = texts[texts["topic_id"] == topic_id].sort_values("sentiment_score")
    picked = [clean_text(t) for t in rows["text"] if readable(t, 25, 400)][:n]
    return [q if len(q) <= max_len else q[: max_len - 1].rsplit(" ", 1)[0] + "…" for q in picked]


def pick_quotes(mentions: pd.DataFrame) -> dict[str, pd.Series]:
    """One clear negative, neutral and positive quote: readable English, 60-220 characters.
    Negative/positive = the strongest; neutral = closest to 0. Ties go to the newest."""
    out: dict[str, pd.Series] = {}
    if mentions.empty:
        return out
    df = mentions[[readable(t) for t in mentions["text"]]].copy()
    if df.empty:
        df = mentions[[readable(t, 30, 320) for t in mentions["text"]]].copy()
    df = df.sort_values("created_at", ascending=False)
    for sentiment, key in (("negative", "sentiment_score"), ("positive", "sentiment_score"), ("neutral", "abs")):
        part = df[df["sentiment"] == sentiment].assign(abs=lambda d: d["sentiment_score"].abs())
        if part.empty:
            continue
        ascending = sentiment != "positive"
        out[sentiment] = part.sort_values(key, ascending=ascending, kind="stable").iloc[0]
    return out


# ---------------------------------------------------------------- feature radar


@dataclass
class FeatureTable:
    features: list[str]
    net: pd.DataFrame  # brand_id x feature, net sentiment x100, NaN where too few mentions
    counts: pd.DataFrame  # brand_id x feature, mention counts


def feature_table(df: pd.DataFrame, brand_ids: list[int]) -> FeatureTable:
    """Features with >= MIN_FEATURE_MENTIONS for at least one shown brand, ordered by total mentions;
    a brand's value is only kept where it has >= MIN_POINT_MENTIONS."""
    df = df[df["brand_id"].isin(brand_ids)]
    counts = df.pivot_table(index="brand_id", columns="feature", values="sentiment_score", aggfunc="count", fill_value=0)
    counts = counts.reindex(brand_ids, fill_value=0)
    if counts.empty:
        return FeatureTable([], pd.DataFrame(), pd.DataFrame())
    qualifying = counts.columns[(counts >= MIN_FEATURE_MENTIONS).any()]
    features = list(counts[qualifying].sum().sort_values(ascending=False).index)
    net = (df.pivot_table(index="brand_id", columns="feature", values="sentiment_score", aggfunc="mean") * 100)
    net = net.reindex(index=brand_ids, columns=features).round(0)
    net = net.where(counts[features] >= MIN_POINT_MENTIONS)
    return FeatureTable(features, net, counts[features])


# ---------------------------------------------------------------- executive summary


@dataclass
class Summary:
    risk: str
    win: str
    request: str
    change: str


def _fmt_net(value: float) -> str:
    return f"{value:+.0f}"


def summary_picks(
    brand_name: str,
    matrix: pd.DataFrame,
    table: FeatureTable,
    brand_id: int,
    competitor_ids: list[int],
    mentions: pd.DataFrame,
    net_sentiment: float | None,
    net_change: float | None,
    days: int,
) -> Summary:
    """Four plain-language lines built from the numbers only (no LLM). Values are HTML-safe text with
    <b> for emphasis."""
    # biggest risk: top Fix now topic, else the top Quick win that is active this week
    urgent = ranked_urgent(matrix) if not matrix.empty else matrix
    active = urgent[urgent["volume_7d"] > 0] if not urgent.empty else urgent
    if not active.empty:
        top = active.iloc[0]
        lead = "Biggest risk" if top["quadrant"] == FIX_NOW else "No Fix-now risks. Most pressing"
        n = int(top["volume_7d"])
        risk = (f"{lead}: <b>{html.escape(str(top['label']))}</b> — {n} complaint{'s' if n != 1 else ''} this week, "
                f"{change_phrase(n, int(top['volume_prev_7d']))}.")
    else:
        # nothing urgent: name the busiest Monitor topic (many mild complaints, possibly spiking)
        watch = matrix[(matrix["quadrant"] == MONITOR) & (matrix["volume_7d"] > 0)] if not matrix.empty else matrix
        if not watch.empty:
            top = watch.sort_values("volume_7d", ascending=False).iloc[0]
            n = int(top["volume_7d"])
            risk = (f"No urgent risks. Watch: <b>{html.escape(str(top['label']))}</b> — {n} mild complaint"
                    f"{'s' if n != 1 else ''} this week, {change_phrase(n, int(top['volume_prev_7d']))}.")
        else:
            risk = "No urgent complaint themes this week."

    # biggest win: feature where the brand beats the competitors' average by the most
    win = "No clear win over competitors yet."
    if table.features and brand_id in table.net.index:
        own = table.net.loc[brand_id]
        rivals = table.net.loc[[c for c in competitor_ids if c in table.net.index]].mean() if competitor_ids else None
        best = None
        enough = table.counts.loc[brand_id] >= MIN_FEATURE_MENTIONS  # only claim wins on solid numbers
        for feature in table.features:
            if not enough[feature] or pd.isna(own[feature]) or rivals is None or pd.isna(rivals[feature]):
                continue
            gap = own[feature] - rivals[feature]
            if gap > 0 and (best is None or gap > best[1]):
                best = (feature, gap)
        if best:
            f = best[0]
            win = (f"Biggest win: <b>{html.escape(feature_name(f))}</b> — {_fmt_net(own[f])} net sentiment vs "
                   f"{_fmt_net(rivals[f])} for competitors.")
        else:
            solid = own[enough].dropna()
            if solid.gt(0).any():
                f = solid.idxmax()
                win = (f"Users like <b>{html.escape(feature_name(f))}</b> ({_fmt_net(own[f])} net sentiment), "
                       "but competitors score as well or better.")

    # top requested feature
    requests = mentions[(mentions["brand_id"] == brand_id) & (mentions["category"] == "feature request")]
    if requests.empty:
        request = "No feature requests in this period."
    else:
        named = requests["feature"].dropna().value_counts()
        total = f"{len(requests)} feature request{'s' if len(requests) != 1 else ''}"
        if named.empty or int(named.iloc[0]) < MIN_REQUESTS_FOR_TOP:
            request = f"{total}; no single feature stands out yet."
        else:
            request = (f"Top request: <b>{html.escape(feature_name(named.index[0]))}</b> "
                       f"({int(named.iloc[0])} of {total}).")

    # net sentiment change
    if net_sentiment is None:
        change = f"No {brand_name} mentions in this period."
    elif net_change is None:
        change = f"Net sentiment is <b>{_fmt_net(net_sentiment)}</b> (no earlier data to compare)."
    elif net_change == 0:
        change = f"Net sentiment is <b>{_fmt_net(net_sentiment)}</b>, unchanged on the previous {days} days."
    else:
        direction = "up" if net_change > 0 else "down"
        points = abs(round(net_change))
        change = (f"Net sentiment is <b>{_fmt_net(net_sentiment)}</b>, {direction} {points} point{'s' if points != 1 else ''} "
                  f"on the previous {days} days.")
    return Summary(risk=risk, win=win, request=request, change=change)
