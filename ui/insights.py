"""Pure aggregations for the dashboard's second half: priority matrix, feature radar, alerts and the
executive summary. No Streamlit, no database: DataFrames in, DataFrames/dataclasses out."""

from __future__ import annotations

import html
import math
from dataclasses import dataclass

import pandas as pd

from nlp.priority import FIX_NOW, IGNORE, MONITOR, QUICK_WINS, quadrant
from nlp.profiles import APP_CATEGORY, Profile
from nlp.text import clean_text
from nlp.topics import looks_english

QUADRANTS = (FIX_NOW, QUICK_WINS, MONITOR, IGNORE)
MIN_FEATURE_MENTIONS = 5  # a feature needs this many mentions for at least one shown brand
MIN_POINT_MENTIONS = 3  # a brand's radar point needs this many mentions for that feature
MIN_RADAR_FEATURES = 3
MIN_REQUESTS_FOR_TOP = 2  # a "top requested feature" needs at least this many requests
MIN_AREA_COMPLAINTS = 5  # a "largest complaint area" needs at least this many complaints
MIN_AREA_THEMES = 2  # ...spread over at least this many themes (one theme is just that theme)


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


# ---------------------------------------------------------------- complaint categories (the roll-up)


def topic_category_map(topics: pd.DataFrame) -> dict[int, str]:
    """topic id -> its dominant category (topics.category), for the topics that have one."""
    if topics.empty or "category" not in topics:
        return {}
    return {int(i): str(c) for i, c in zip(topics["id"], topics["category"]) if isinstance(c, str) and c}


def complaint_counts(mentions: pd.DataFrame, profile: Profile, excluded_sources: tuple[str, ...] = (),
                     topic_categories: dict[int, str] | None = None) -> pd.Series:
    """Complaints per profile complaint category, the way topics count them (excluded sources left out).

    A mention clustered into a topic counts under that topic's dominant category, so a category's number is
    the sum of its themes (plus complaints no topic holds, counted under their own category). Without
    `topic_categories` every complaint counts under its own zero-shot category."""
    if mentions.empty:
        return pd.Series(dtype=int)
    rows = mentions[~mentions["source"].isin(excluded_sources)]
    if rows.empty:
        return pd.Series(dtype=int)
    own = pd.Series([c in profile.complaints and profile.is_complaint(s, c)
                     for s, c in zip(rows["sentiment"], rows["category"])], index=rows.index, dtype=bool)
    if topic_categories and "topic_id" in rows:
        in_topic = rows["topic_id"].map(lambda t: topic_categories.get(int(t)) if pd.notna(t) else None)
    else:
        in_topic = pd.Series([None] * len(rows), index=rows.index, dtype=object)
    clustered = in_topic.notna()  # topics are built from complaints: a member is one
    category = in_topic.where(clustered, rows["category"])
    keep = (clustered | own) & category.isin(profile.complaints)
    return category[keep].value_counts()


def category_name(label: str) -> str:
    """A category label as a name: 'app and ordering experience' -> 'App and ordering problems'."""
    if label == APP_CATEGORY:
        return "App and ordering problems"
    return label[:1].upper() + label[1:]


@dataclass
class ComplaintArea:
    """The category with the most complaints, set against the single biggest theme."""
    category: str
    name: str
    this: int  # complaints in the period
    last: int  # the period before
    themes: int  # topics (complaint themes) whose dominant category it is
    top_theme: int  # complaints in the biggest single theme (this week's volume)

    @property
    def larger(self) -> bool:
        """Worth saying: the area beats the top single theme, and is several themes, not one."""
        return self.this >= MIN_AREA_COMPLAINTS and self.themes >= MIN_AREA_THEMES and self.this > self.top_theme


def largest_area(this: pd.Series, last: pd.Series, matrix: pd.DataFrame) -> ComplaintArea | None:
    """The biggest complaint category of the period, with how many themes it is spread over."""
    if this.empty or int(this.max()) <= 0:
        return None
    category = str(this.idxmax())
    live = matrix[matrix["volume_7d"] > 0] if not matrix.empty else matrix
    themes = int((live["category"] == category).sum()) if not live.empty else 0
    top = int(live["volume_7d"].max()) if not live.empty else 0
    return ComplaintArea(category, category_name(category), int(this[category]), int(last.get(category, 0)), themes, top)


def area_line(area: ComplaintArea) -> str:
    """The roll-up as a summary line (HTML-safe, <b> for emphasis)."""
    return (f"Largest complaint area: <b>{html.escape(area.name)}</b> — {area.this} complaint"
            f"{'s' if area.this != 1 else ''} across {area.themes} themes, {change_phrase(area.this, area.last)}.")


# ---------------------------------------------------------------- executive summary


@dataclass
class Summary:
    risk: str
    win: str
    request: str
    change: str
    area: str = ""  # the complaint-category roll-up line, only when it is larger than the top single theme


def _fmt_net(value: float) -> str:
    return f"{value:+.0f}"


def top_risk(matrix: pd.DataFrame) -> tuple[str, pd.Series] | None:
    """("fix_now" | "most_pressing" | "watch", topic row): the summary's biggest-risk pick."""
    urgent = ranked_urgent(matrix) if not matrix.empty else matrix
    active = urgent[urgent["volume_7d"] > 0] if not urgent.empty else urgent
    if not active.empty:
        top = active.iloc[0]
        return ("fix_now" if top["quadrant"] == FIX_NOW else "most_pressing"), top
    watch = matrix[(matrix["quadrant"] == MONITOR) & (matrix["volume_7d"] > 0)] if not matrix.empty else matrix
    if not watch.empty:
        return "watch", watch.sort_values("volume_7d", ascending=False).iloc[0]
    return None


def biggest_win(table: FeatureTable, brand_id: int, competitor_ids: list[int]) -> tuple[str, str, float, float | None] | None:
    """("beats" | "liked", feature, own net, competitors' net or None): the summary's biggest-win pick."""
    if not table.features or brand_id not in table.net.index:
        return None
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
        return "beats", f, float(own[f]), float(rivals[f])  # type: ignore[index]
    solid = own[enough].dropna()
    if solid.gt(0).any():
        f = solid.idxmax()
        return "liked", f, float(own[f]), None
    return None


def top_request(mentions: pd.DataFrame, brand_id: int, request_label: str | None) -> tuple[int, str | None, int]:
    """(requests, top requested feature or None, its count): the summary's request pick."""
    requests = mentions[(mentions["brand_id"] == brand_id) & (mentions["category"] == request_label)]
    named = requests["feature"].dropna().value_counts()
    if named.empty or int(named.iloc[0]) < MIN_REQUESTS_FOR_TOP:
        return len(requests), None, 0
    return len(requests), str(named.index[0]), int(named.iloc[0])


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
    request_label: str | None = "feature request",
    area: ComplaintArea | None = None,
) -> Summary:
    """Four plain-language lines built from the numbers only (no LLM). Values are HTML-safe text with
    <b> for emphasis."""
    # biggest risk: top Fix now topic, else the top Quick win that is active this week, else a busy Monitor topic
    pick = top_risk(matrix)
    if pick is None:
        risk = "No urgent complaint themes this week."
    else:
        kind, top = pick
        n = int(top["volume_7d"])
        if kind == "watch":  # nothing urgent: the busiest Monitor topic (many mild complaints, possibly spiking)
            risk = (f"No urgent risks. Watch: <b>{html.escape(str(top['label']))}</b> — {n} mild complaint"
                    f"{'s' if n != 1 else ''} this week, {change_phrase(n, int(top['volume_prev_7d']))}.")
        else:
            lead = "Biggest risk" if kind == "fix_now" else "No Fix-now risks. Most pressing"
            risk = (f"{lead}: <b>{html.escape(str(top['label']))}</b> — {n} complaint{'s' if n != 1 else ''} this week, "
                    f"{change_phrase(n, int(top['volume_prev_7d']))}.")

    # biggest win: feature where the brand beats the competitors' average by the most
    win = "No clear win over competitors yet."
    won = biggest_win(table, brand_id, competitor_ids)
    if won and won[0] == "beats":
        _, f, own_net, rival_net = won
        win = (f"Biggest win: <b>{html.escape(feature_name(f))}</b> — {_fmt_net(own_net)} net sentiment vs "
               f"{_fmt_net(rival_net)} for competitors.")  # type: ignore[arg-type]
    elif won:
        _, f, own_net, _ = won
        win = (f"Users like <b>{html.escape(feature_name(f))}</b> ({_fmt_net(own_net)} net sentiment), "
               "but competitors score as well or better.")

    # top requested feature (the profile's request label: "feature request", "product request", …)
    noun = request_label or "request"
    total_requests, feature, count = top_request(mentions, brand_id, request_label)
    if not total_requests:
        request = f"No {noun}s in this period."
    else:
        total = f"{total_requests} {noun}{'s' if total_requests != 1 else ''}"
        if feature is None:
            request = f"{total}; no single feature stands out yet."
        else:
            request = f"Top request: <b>{html.escape(feature_name(feature))}</b> ({count} of {total})."

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
    return Summary(risk=risk, win=win, request=request, change=change,
                   area=area_line(area) if area is not None and area.larger else "")
