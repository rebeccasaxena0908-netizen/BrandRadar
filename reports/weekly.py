"""Weekly PDF report per brand, in the Blush glass style (2-3 A4 pages).

    python -m reports.weekly                       # last complete week, the own brand of every group
    python -m reports.weekly --brand Notion        # one brand
    python -m reports.weekly --group Skincare      # every brand in a group
    python -m reports.weekly --week 2026-W41       # a given ISO week (Monday-Sunday, UTC dates)
    BRAND_RADAR_DEMO=1 python -m reports.weekly    # from the demo database ("Demo data" on every page)

Output: reports/output/brandradar-<brand>-<week>.pdf (git-ignored).

Everything comes from the dashboard's own code: ui/data.py queries and aggregations, ui/insights.py
for the matrix, feature radar, summary and quotes, ui/charts.py for the figures (rendered to PNG by
reports/render.py with the installed Chrome). Sections with too little data get a short designed note
instead of an empty chart. Topic volume, pain and quadrant are the latest priority snapshot (the
collector refreshes it every cycle), so for an older --week the matrix is labelled with its date.
"""

from __future__ import annotations

import argparse
import html
import io
import logging
import random
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable, Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)
from sqlalchemy import Engine

from config import PROJECT_ROOT, settings
from reports import quiet_streamlit

quiet_streamlit()

from nlp.priority import FIX_NOW, QUICK_WINS
from nlp.profiles import Profile, get_profile
from reports.export import slug
from reports.weeks import Week
from reports.render import Renderer, renderer
from sources.base import utc_now
from ui import charts, data, insights
from ui.components import (
    BERRY, EMOTION_COLOURS, INK, INK_SOFT, MUTED, RULE, SENTIMENT_COLOURS, UI_DIR, brand_colours,
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "output"
MIN_FOR_CHART = 5  # same bar as the dashboard: fewer mentions -> a note instead of a chart
MAX_URGENT = 5
MAX_CATEGORIES = 6  # the categories chart shows the busiest six, so page 1 keeps its layout

PAGE_W, PAGE_H = A4
MARGIN = 16 * mm
CONTENT_W = PAGE_W - 2 * MARGIN
FRAME_W = CONTENT_W - 12  # SimpleDocTemplate's frame keeps 6 pt padding on each side
CARD_PAD = 13
INNER_W = FRAME_W - 2 * CARD_PAD  # usable width inside a section card
PX_PER_PT = 1.6  # chart images are drawn at this many CSS px per PDF point (then x2 for print)

BLUSH = colors.HexColor("#FBEFF1")
CARD_STROKE = colors.HexColor("#F1D9DD")
BLOB_COLOURS = ("#F4A3B1", "#E58A9A", "#F3BAC1", "#F4A3B1")


# ---------------------------------------------------------------- data


@dataclass
class BrandReport:
    brand: pd.Series
    comps: list[pd.Series]
    profile: Profile
    week: Week
    kpis: data.Kpis
    open_alerts: int
    trend: pd.DataFrame
    trend_mentions: int
    topics: pd.DataFrame
    matrix: pd.DataFrame
    texts: pd.DataFrame
    urgent: pd.DataFrame
    snapshot: datetime | None
    alerts: pd.DataFrame
    table: insights.FeatureTable
    feature_since: datetime | None
    quotes: dict[str, pd.Series]
    summary: insights.Summary
    complaints_this: pd.Series
    complaints_last: pd.Series
    demo: bool = False
    generated_at: datetime = field(default_factory=utc_now)

    @property
    def name(self) -> str:
        return str(self.brand["name"])

    @property
    def colours(self) -> dict[str, str]:
        return brand_colours(self.name, [c["name"] for c in self.comps])

    def named(self) -> tuple[tuple[int, str], list[tuple[int, str]]]:
        return (int(self.brand["id"]), self.name), [(int(c["id"]), str(c["name"])) for c in self.comps]


def complaint_counts(mentions: pd.DataFrame, profile: Profile) -> pd.Series:
    """Complaints per profile complaint category, the way topics count them (excluded sources left out)."""
    if mentions.empty:
        return pd.Series(dtype=int)
    rows = mentions[~mentions["source"].isin(settings.topic_excluded_sources)
                    & mentions["category"].isin(profile.complaints)]
    mask = pd.Series([profile.is_complaint(s, c) for s, c in zip(rows["sentiment"], rows["category"])],
                     index=rows.index, dtype=bool)  # a plain [] would select no *columns* when rows is empty
    return rows.loc[mask, "category"].value_counts()


def find_brand(brands: pd.DataFrame, name_or_id: str | int) -> pd.Series:
    if isinstance(name_or_id, int) or str(name_or_id).isdigit():
        match = brands[brands["id"] == int(name_or_id)]
    else:
        match = brands[brands["name"].str.lower() == str(name_or_id).lower()]
    if match.empty:
        raise SystemExit(f"Unknown brand {name_or_id!r}. Known: {', '.join(brands['name'])}")
    return match.iloc[0]


def gather(engine: Engine, brand_ref: str | int, week: Week, now: datetime | None = None) -> BrandReport:
    """Everything one brand's report shows, from the dashboard's queries and aggregations."""
    brands = data.query_brands(engine)
    brand = find_brand(brands, brand_ref)
    bid = int(brand["id"])
    group = brands[brands["brand_group"] == brand["brand_group"]]
    comps = [row for _, row in group.iterrows() if int(row["id"]) != bid]
    ids = tuple(int(i) for i in group["id"])
    profile = get_profile(brand["profile"])
    sources = tuple(data.query_sources_with_data(engine, ids))
    prev = week.previous()

    current = data.query_mentions(engine, ids, week.start, week.end, sources)
    previous = data.query_mentions(engine, ids, prev.start, prev.end, sources)
    own, own_prev = current[current["brand_id"] == bid], previous[previous["brand_id"] == bid]
    kpis = data.compute_kpis(own, own_prev)

    frames = [f for f in (previous, current) if not f.empty]
    both = pd.concat(frames, ignore_index=True) if frames else current
    trend = data.daily_trend(both, prev.start, week.end)

    topics = data.query_topics(engine, bid)
    matrix = insights.topic_matrix(topics, settings.priority_volume_threshold, settings.priority_pain_threshold,
                                   settings.priority_praise_pain_max)
    texts = data.query_topic_texts(engine, tuple(int(t) for t in matrix["id"])) if not matrix.empty else \
        data.query_topic_texts(engine, ())
    urgent = insights.ranked_urgent(matrix).head(MAX_URGENT) if not matrix.empty else matrix
    scored = topics["scored_at"].dropna() if not topics.empty else pd.Series(dtype="datetime64[ns]")
    snapshot = pd.Timestamp(scored.max()).to_pydatetime() if not scored.empty else None

    feature_df = data.query_feature_mentions(engine, ids)
    table = insights.feature_table(feature_df, list(ids))
    summary = insights.summary_picks(str(brand["name"]), matrix, table, bid, [int(c["id"]) for c in comps], current,
                                     kpis.net_sentiment, kpis.net_sentiment_change, 7, profile.request)
    return BrandReport(
        brand=brand, comps=comps, profile=profile, week=week, kpis=kpis,
        open_alerts=data.query_open_alerts(engine, bid),
        trend=trend, trend_mentions=int(len(own) + len(own_prev)),
        topics=topics, matrix=matrix, texts=texts, urgent=urgent, snapshot=snapshot,
        alerts=data.query_alerts(engine, bid, week.start, week.end),
        table=table, feature_since=feature_df["created_at"].min() if not feature_df.empty else None,
        quotes=insights.pick_quotes(own), summary=summary,
        complaints_this=complaint_counts(own, profile), complaints_last=complaint_counts(own_prev, profile),
        demo=settings.demo, generated_at=now or utc_now(),
    )


# ---------------------------------------------------------------- fonts and text


@lru_cache(maxsize=1)
def register_fonts() -> dict[str, set[int]]:
    """Register the TTFs (embedded in every PDF) and return each font's character set."""
    from reportlab.lib.fonts import addMapping

    files = {"Figtree": "Figtree-Regular.ttf", "Figtree-SemiBold": "Figtree-SemiBold.ttf",
             "Figtree-Bold": "Figtree-Bold.ttf", "Montserrat-SemiBold": "Montserrat-SemiBold.ttf",
             "Montserrat-Bold": "Montserrat-Bold.ttf"}
    cmaps: dict[str, set[int]] = {}
    for name, file in files.items():
        font = TTFont(name, str(UI_DIR / "fonts" / file))
        pdfmetrics.registerFont(font)
        cmaps[name] = set(font.face.charToGlyph)
    for bold, italic, face in ((0, 0, "Figtree"), (1, 0, "Figtree-SemiBold"), (0, 1, "Figtree"), (1, 1, "Figtree-SemiBold")):
        addMapping("Figtree", bold, italic, face)  # <b> in paragraphs -> Figtree SemiBold
    return cmaps


def printable(text: str, font: str = "Figtree") -> str:
    """Drop characters the font can't draw (emoji, scripts it lacks) instead of printing boxes."""
    cmap = register_fonts()[font]
    out = "".join(c if ord(c) in cmap or c in "\n\t" else " " for c in text)
    return re.sub(r"[ \t]{2,}", " ", out)


def plain(text: Any, font: str = "Figtree") -> str:
    """Untrusted text (mentions, labels) for a Paragraph: printable and XML-escaped."""
    return html.escape(printable(" ".join(str(text).split()), font).strip(), quote=False)


def markup(text: str) -> str:
    """Our own HTML-ish lines (summary_picks: escaped text with <b>) for a Paragraph."""
    text = html.unescape(text)  # summary_picks escapes for HTML; re-escape for ReportLab below
    parts = re.split(r"(</?b>)", text)
    return "".join(p if p in ("<b>", "</b>") else html.escape(printable(p), quote=False) for p in parts)


def style(name: str, font: str = "Figtree", size: float = 9, colour: str = INK_SOFT, leading: float | None = None,
          **kw: Any) -> ParagraphStyle:
    return ParagraphStyle(name, fontName=font, fontSize=size, textColor=colors.HexColor(colour),
                          leading=leading or size * 1.35, **kw)


S = {
    "eyebrow": style("eyebrow", "Montserrat-SemiBold", 7.5, MUTED, charSpace=1.6),
    "brand": style("brand", "Montserrat-Bold", 26, INK, 30),
    "sub": style("sub", "Figtree", 10, INK_SOFT),
    "h": style("h", "Montserrat-SemiBold", 11.5, INK, 14),
    "cap": style("cap", "Figtree", 7.8, MUTED),
    "body": style("body", "Figtree", 9.2, INK_SOFT, 12.6),
    "ink": style("ink", "Figtree", 9.2, INK, 12.6),
    "label": style("label", "Montserrat-SemiBold", 6.8, MUTED, charSpace=1.2),
    "kpi": style("kpi", "Montserrat-Bold", 21, INK, 24),
    "small": style("small", "Figtree", 7.6, INK_SOFT, 10),
    "note_t": style("note_t", "Montserrat-SemiBold", 9.5, INK, 12, alignment=TA_CENTER),
    "note_m": style("note_m", "Figtree", 8.2, INK_SOFT, 11, alignment=TA_CENTER),
    "quote": style("quote", "Figtree", 9.2, INK, 13),
    "topic": style("topic", "Figtree-SemiBold", 9.4, INK, 12),
}


# ---------------------------------------------------------------- flowables


class Card(Flowable):
    """A glass card: white at 72%, white hairline with a blush edge, 14 pt radius, soft berry shadow.
    `dashed` draws the empty-state variant; `accent` adds a coloured bar on the left (quotes)."""

    def __init__(self, content: Sequence[Flowable], width: float | None = None, padding: float = CARD_PAD,
                 dashed: bool = False, accent: str | None = None, min_height: float = 0, gap: float = 4) -> None:
        super().__init__()
        self.content, self.card_width, self.padding = list(content), width, padding
        self.dashed, self.accent, self.min_height, self.gap = dashed, accent, min_height, gap
        self._sizes: list[tuple[float, float]] = []

    def wrap(self, avail_width: float, avail_height: float) -> tuple[float, float]:
        self.width = self.card_width or avail_width
        inner = self.width - 2 * self.padding - (5 if self.accent else 0)
        self._sizes = [f.wrap(inner, avail_height) for f in self.content]
        body = sum(h for _, h in self._sizes) + self.gap * max(0, len(self.content) - 1)
        self.height = max(self.min_height, body + 2 * self.padding)
        return self.width, self.height

    def draw(self) -> None:
        c = self.canv
        c.saveState()
        if not self.dashed:  # soft berry shadow
            c.setFillColor(colors.HexColor(BERRY), alpha=0.05)
            c.roundRect(0, -2.2, self.width, self.height, 14, stroke=0, fill=1)
        c.setFillColor(colors.white, alpha=0.5 if self.dashed else 0.74)
        c.setStrokeColor(colors.HexColor("#E8C3C9") if self.dashed else CARD_STROKE)
        c.setLineWidth(0.8)
        if self.dashed:
            c.setDash(3, 2.5)
        c.roundRect(0, 0, self.width, self.height, 14, stroke=1, fill=1)
        c.setDash()
        if self.accent:
            c.setFillColor(colors.HexColor(self.accent))
            c.roundRect(self.padding - 4, self.padding, 2.6, self.height - 2 * self.padding, 1.3, stroke=0, fill=1)
        c.restoreState()
        x = self.padding + (5 if self.accent else 0)
        body = sum(h for _, h in self._sizes) + self.gap * max(0, len(self.content) - 1)
        y = self.height - self.padding - max(0, (self.height - 2 * self.padding - body) / 2 if self.dashed else 0)
        for flowable, (_, h) in zip(self.content, self._sizes):
            y -= h
            flowable.drawOn(c, x, y)
            y -= self.gap


class Dot(Flowable):
    def __init__(self, colour: str, size: float = 6.5) -> None:
        super().__init__()
        self.colour, self.size = colour, size
        self.width = self.height = size

    def draw(self) -> None:
        self.canv.setFillColor(colors.HexColor(self.colour))
        self.canv.circle(self.size / 2, self.size / 2, self.size / 2, stroke=0, fill=1)


class Tick(Flowable):
    """A green disc with a white check mark (Figtree has no ✓ glyph)."""

    def __init__(self, size: float = 11) -> None:
        super().__init__()
        self.width = self.height = size

    def draw(self) -> None:
        c, s = self.canv, self.width
        c.setFillColor(colors.HexColor(SENTIMENT_COLOURS["positive"]))
        c.circle(s / 2, s / 2, s / 2, stroke=0, fill=1)
        c.setStrokeColor(colors.white)
        c.setLineWidth(1.4)
        c.setLineCap(1)
        path = c.beginPath()
        path.moveTo(s * 0.28, s * 0.52)
        path.lineTo(s * 0.44, s * 0.36)
        path.lineTo(s * 0.73, s * 0.66)
        c.drawPath(path, stroke=1, fill=0)


class TargetIcon(Flowable):
    """The dashboard's empty-state mark: a small blush disc with berry rings, centred."""

    def __init__(self, width: float) -> None:
        super().__init__()
        self.width, self.height = width, 26

    def draw(self) -> None:
        c, x, y = self.canv, self.width / 2, 13
        c.setFillColor(colors.HexColor("#F7DCE0"))
        c.circle(x, y, 12.5, stroke=0, fill=1)
        c.setStrokeColor(colors.HexColor(BERRY))
        c.setLineWidth(1)
        for r in (7, 3.6):
            c.circle(x, y, r, stroke=1, fill=0)
        c.setFillColor(colors.HexColor(BERRY))
        c.circle(x, y, 1.2, stroke=0, fill=1)


def note(title: str, message: str, height: float = 110) -> Card:
    """The designed "not enough data yet" card (dashed, centred), used instead of an empty chart."""
    return Card([TargetIcon(INNER_W - 6), Paragraph(plain(title, "Montserrat-SemiBold"), S["note_t"]),
                 Paragraph(plain(message), S["note_m"])], dashed=True, min_height=height, gap=5, padding=16)


def section(title: str, caption: str, body: Sequence[Flowable]) -> KeepTogether:
    head = [Paragraph(plain(title, "Montserrat-SemiBold"), S["h"]), Paragraph(plain(caption), S["cap"]), Spacer(1, 4)]
    return KeepTogether([Card([*head, *body]), Spacer(1, 10)])


def chart(r: Renderer, fig: Any, width_pt: float, height_pt: float) -> Image:
    png = r.png(fig, int(width_pt * PX_PER_PT), int(height_pt * PX_PER_PT))
    return Image(io.BytesIO(png), width=width_pt, height=height_pt)


def calm(title: str, message: str) -> Table:
    """A quiet "all clear" line (no alerts, nothing urgent): green tick + text, not an empty-state card."""
    tick = Tick()
    text = Paragraph(f'<font name="Figtree-SemiBold" color="{INK}">{plain(title)}</font><br/>{plain(message)}', S["body"])
    t = Table([[tick, text]], colWidths=[16, INNER_W - 16])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("TOPPADDING", (0, 0), (-1, -1), 2)]))
    return t


# ---------------------------------------------------------------- sections


def _delta(value: float | None, unit: str, good_when_up: bool | None) -> str:
    if value is None:
        return f'<font color="{MUTED}">— no data the week before</font>'
    if value == 0:
        return f'<font color="{MUTED}">no change vs last week</font>'
    arrow = "↑" if value > 0 else "↓"
    if good_when_up is None:
        colour = INK_SOFT
    else:
        colour = SENTIMENT_COLOURS["positive"] if (value > 0) == good_when_up else SENTIMENT_COLOURS["negative"]
    return f'<font color="{colour}" name="Figtree-SemiBold">{arrow} {abs(value):g}{unit}</font> vs last week'


def kpi_row(rep: BrandReport) -> Table:
    k = rep.kpis
    net = "—" if k.net_sentiment is None else f"{k.net_sentiment:+.0f}"
    net_colour = MUTED if k.net_sentiment is None else (
        SENTIMENT_COLOURS["positive"] if k.net_sentiment > 0 else SENTIMENT_COLOURS["negative"] if k.net_sentiment < 0 else INK)
    emotion = (k.top_emotion or "—").title()
    emotion_detail = (f"{k.top_emotion_share:.0%} of emotion · then {k.runner_up_emotion} {k.runner_up_share:.0%}"
                      if k.top_emotion and k.runner_up_emotion else "no mentions this week")
    week_alerts = len(rep.alerts)
    cards = [
        ("NET SENTIMENT", net, net_colour, _delta(k.net_sentiment_change, " pts", True)),
        ("MENTIONS", f"{k.mentions:,}", INK, _delta(k.mentions_change_pct, "%", None)),
        ("TOP EMOTION", emotion, EMOTION_COLOURS.get(k.top_emotion or "", INK), plain(emotion_detail)),
        ("OPEN ALERTS", str(rep.open_alerts), SENTIMENT_COLOURS["negative"] if rep.open_alerts else INK,
         (f"unseen · {week_alerts} new this week" if week_alerts else "unseen · none new this week")
         if rep.open_alerts else (f"{week_alerts} new this week, all seen" if week_alerts else "all clear")),
    ]
    w = (FRAME_W - 3 * 8) / 4
    cells = [Card([Paragraph(label, S["label"]),
                   Paragraph(f'<font color="{colour}">{plain(value, "Montserrat-Bold")}</font>',
                             style("kv", "Montserrat-Bold", 21 if len(value) < 9 else 15, INK, 24)),
                   Paragraph(detail, S["small"])], width=w, padding=11, gap=3, min_height=74)
             for label, value, colour, detail in cards]
    t = Table([cells], colWidths=[w + 8] * 3 + [w])
    t.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                           ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                           ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return t


def summary_section(rep: BrandReport) -> KeepTogether:
    marks = {"risk": SENTIMENT_COLOURS["negative"], "win": SENTIMENT_COLOURS["positive"], "request": "#E79AA6",
             "change": BERRY}
    s = rep.summary
    rows = [[Dot(marks[kind], 5.5), Paragraph(markup(line), S["ink"])]
            for kind, line in (("risk", s.risk), ("win", s.win), ("request", s.request), ("change", s.change))]
    t = Table(rows, colWidths=[13, INNER_W - 13])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("TOPPADDING", (0, 0), (0, -1), 4.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5)]))
    return section("Executive summary", f"{rep.name} in {rep.week.span}, built from the numbers in this report", [t])


def trend_section(rep: BrandReport, r: Renderer) -> KeepTogether:
    caption = "Net sentiment ×100, 3-day rolling mean weighted by mentions · this week and the week before"
    own = charts.own_trend(rep.trend, int(rep.brand["id"]))
    if rep.trend_mentions < MIN_FOR_CHART or own["day"].nunique() < 2:
        return section("Daily sentiment", caption, [note(
            "Not enough data yet", f"{rep.name} has {rep.trend_mentions} mention(s) in these two weeks; a trend "
            f"needs at least {MIN_FOR_CHART} on two or more days.")])
    brand, comps = rep.named()
    fig = charts.trend_figure(rep.trend, brand, comps, rep.colours)
    return section("Daily sentiment", caption, [chart(r, fig, INNER_W, 140)])


def categories_section(rep: BrandReport, r: Renderer) -> KeepTogether:
    caption = (f"Complaints per {rep.profile.name.replace('_', ' ')} category (industry profile) · this week vs last week")
    total = int(rep.complaints_this.sum() + rep.complaints_last.sum())
    if total < MIN_FOR_CHART:
        return section("What people complain about", caption, [note(
            "Not enough data yet", f"{total} categorised complaint(s) for {rep.name} in these two weeks; "
            f"the chart needs at least {MIN_FOR_CHART}.", height=96)])
    totals = rep.complaints_this.add(rep.complaints_last, fill_value=0)
    labels = [c for c in totals.sort_values(ascending=False).index[:MAX_CATEGORIES] if totals[c] > 0]
    fig = charts.category_figure(rep.complaints_this.reindex(labels, fill_value=0),
                                 rep.complaints_last.reindex(labels, fill_value=0))
    return section("What people complain about", caption, [chart(r, fig, INNER_W, 24 + 13 * len(labels))])


def matrix_section(rep: BrandReport, r: Renderer) -> KeepTogether:
    when = f" · snapshot {rep.snapshot:%d %b %H:%M} UTC" if rep.snapshot else ""
    caption = f"Complaint topics by volume (last 7 days) and pain — top right needs attention first{when}"
    if rep.topics.empty:
        return section("Priority matrix", caption, [note(
            "No complaint topics yet", f"{rep.name} has too few complaints to cluster into topics (at least "
            f"{settings.topic_min_complaints} are needed).", height=150)])
    if rep.matrix.empty:
        return section("Priority matrix", caption, [note(
            "Only praise topics", f"All {len(rep.topics)} topics are praise or discussion.", height=150)])
    fig = charts.matrix_figure(rep.matrix, rep.texts, settings.priority_volume_threshold, settings.priority_pain_threshold)
    return section("Priority matrix", caption, [chart(r, fig, INNER_W, 262)])  # dashboard-like aspect: labels clear the corners


def urgent_section(rep: BrandReport) -> KeepTogether:
    caption = "Fix now, then Quick wins · ranked by volume × pain · category from the industry profile"
    if rep.matrix.empty:
        return section("Act on these first", caption, [calm("Nothing to rank yet", "Topics appear once there are "
                                                            "enough complaints to cluster.")])
    if rep.urgent.empty:
        return section("Act on these first", caption, [calm("Nothing urgent", "No topics in Fix now or Quick wins.")])
    rows = []
    for _, t in rep.urgent.iterrows():
        n, prev = int(t["volume_7d"]), int(t["volume_prev_7d"])
        meta = (f'<font name="Figtree-SemiBold" color="{charts.QUADRANT_COLOURS[t["quadrant"]]}">{t["quadrant"]}</font>'
                f' · {plain(t["category"] or "uncategorised")} · {n} this week ({insights.trend_text(n, prev)})'
                f' · pain {t["pain"]:.2f}')
        quote = insights.topic_quotes(rep.texts, int(t["id"]), n=1, max_len=150)
        cell = [Paragraph(plain(t["label"]), S["topic"]), Paragraph(meta, S["small"])]
        if quote:
            cell.append(Paragraph(f'“{plain(quote[0])}”', style("tq", "Figtree", 8, MUTED, 10.5)))
        rows.append([Dot(charts.QUADRANT_COLOURS[t["quadrant"]]), cell])
    t = Table(rows, colWidths=[14, INNER_W - 14])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("TOPPADDING", (0, 0), (0, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                           ("LINEBELOW", (1, 0), (1, -2), 0.5, colors.HexColor(RULE))]))
    return section("Act on these first", caption, [t])


def alerts_section(rep: BrandReport) -> KeepTogether:
    caption = "Topics whose complaints at least doubled week over week, raised during this week"
    if rep.alerts.empty:
        return section("Trending complaint alerts", caption, [calm(
            "No spikes this week", f"Complaint volumes for {rep.name} were steady in {rep.week.span}.")])
    rows = []
    for _, a in rep.alerts.iterrows():
        this_week, last_week = int(a["this_week"] or 0), int(a["last_week"] or 0)
        label = a["label"] if isinstance(a["label"], str) else a["message"]
        rows.append([Dot(SENTIMENT_COLOURS["negative"]), Paragraph(
            f'<font name="Figtree-SemiBold" color="{INK}">{plain(label)}</font>  '
            f'<font name="Figtree-SemiBold" color="{SENTIMENT_COLOURS["negative"]}">{insights.trend_text(this_week, last_week)}</font>'
            f' · {this_week} this week · {pd.Timestamp(a["created_at"]):%d %b}', S["body"])])
    t = Table(rows, colWidths=[14, INNER_W - 14])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("TOPPADDING", (0, 0), (0, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    return section("Trending complaint alerts", caption, [t])


def radar_section(rep: BrandReport, r: Renderer) -> KeepTogether:
    since = f" since {rep.feature_since:%d %b}" if rep.feature_since is not None else ""
    names = ", ".join(str(c["name"]) for c in rep.comps) or "no competitors"
    caption = (f"Net sentiment per {rep.profile.name.replace('_', ' ')} feature, −100 to +100, vs {names} · "
               f"all data{since}")
    if len(rep.table.features) < insights.MIN_RADAR_FEATURES:
        return section("Feature radar", caption, [note(
            "Not enough feature mentions", f"Fewer than {insights.MIN_RADAR_FEATURES} features have "
            f"{insights.MIN_FEATURE_MENTIONS}+ mentions for {rep.name} and its competitors.", height=150)])
    brand, comps = rep.named()
    fig = charts.radar_figure(rep.table, brand, comps, rep.colours)
    return section("Feature radar", caption, [chart(r, fig, INNER_W, 300)])


def quotes_section(rep: BrandReport) -> KeepTogether:
    caption = f"One clear negative, neutral and positive mention of {rep.name} from {rep.week.span}"
    if not rep.quotes:
        return section("What people are saying", caption, [note(
            "No quotes this week", f"No readable English mentions of {rep.name} in {rep.week.span}.", height=96)])
    cards: list[Flowable] = []
    for sentiment in ("negative", "neutral", "positive"):
        if sentiment not in rep.quotes:
            continue
        q = rep.quotes[sentiment]
        stars = f" · {int(q['rating'])}/5" if q["rating"] is not None and not pd.isna(q["rating"]) else ""
        meta = (f'<font name="Montserrat-SemiBold" color="{SENTIMENT_COLOURS[sentiment]}">{sentiment.upper()}</font>'
                f'  {data.SOURCES.get(q["source"], q["source"])}{stars} · {pd.Timestamp(q["created_at"]):%d %b}')
        cards.append(Card([Paragraph(f"“{plain(q['text'])}”", S["quote"]), Paragraph(meta, S["small"])],
                          accent=SENTIMENT_COLOURS[sentiment], padding=10, gap=4))
        cards.append(Spacer(1, 6))
    return section("What people are saying", caption, cards[:-1])


def title_block(rep: BrandReport) -> list[Flowable]:
    profile = rep.profile.name.replace("_", " ")
    return [
        Paragraph("WEEKLY REPORT", S["eyebrow"]),
        Spacer(1, 3),
        Paragraph(plain(rep.name, "Montserrat-Bold"), S["brand"]),
        Paragraph(f'{plain(rep.brand["brand_group"])} · {profile} profile · '
                  f'<font name="Figtree-SemiBold" color="{INK}">{rep.week.span}</font> ({rep.week.label})', S["sub"]),
        Spacer(1, 12),
    ]


# ---------------------------------------------------------------- page decoration


def _background(c: Any, seed: str) -> None:
    """White-to-blush wash with soft blush blobs and a few glass bubbles (a still frame of the theme)."""
    c.saveState()
    c.linearGradient(0, PAGE_H, 0, 0, (colors.white, BLUSH), extend=False)
    rng = random.Random(seed)
    for i, hex_colour in enumerate(BLOB_COLOURS):
        x = rng.uniform(0.05, 0.95) * PAGE_W
        y = PAGE_H * (0.9 - i * 0.27) + rng.uniform(-40, 40)
        radius = rng.uniform(150, 230)
        # stacked translucent discs = a soft, blurred blob. Alpha stays >= 0.04: macOS Preview/Quartz turns
        # fills of ~1% alpha grey (the colour is lost when it rounds to 8 bits)
        c.setFillColor(colors.HexColor(hex_colour), alpha=0.045)
        for step in range(9):
            c.circle(x, y, radius * (1 - step / 10), stroke=0, fill=1)
    for _ in range(9):
        x, y, r = rng.uniform(0, PAGE_W), rng.uniform(0, PAGE_H), rng.uniform(5, 22)
        c.setFillColor(colors.white, alpha=0.35)
        c.setStrokeColor(colors.white, alpha=0.9)
        c.setLineWidth(0.7)
        c.circle(x, y, r, stroke=1, fill=1)
    c.restoreState()


def _demo_pill(c: Any, x_right: float, y: float) -> float:
    """Berry "DEMO DATA" pill ending at x_right; returns its width."""
    text = "DEMO DATA"
    width = pdfmetrics.stringWidth(text, "Montserrat-Bold", 6.5) + 1.2 * len(text) + 12
    c.saveState()
    c.setFillColor(colors.HexColor(BERRY))
    c.roundRect(x_right - width, y - 3.5, width, 12, 6, stroke=0, fill=1)
    t = c.beginText(x_right - width + 6, y)
    t.setFont("Montserrat-Bold", 6.5)
    t.setCharSpace(1.2)
    t.setFillColor(colors.white)
    t.textOut(text)
    c.drawText(t)
    c.restoreState()
    return width


def _decorate(rep: BrandReport) -> Any:
    def on_page(c: Any, doc: Any) -> None:
        _background(c, f"{rep.name}-{rep.week.label}-{doc.page}")
        top = PAGE_H - 10 * mm
        # wordmark (its letter spacing stays inside this saveState: a text object's char space persists)
        c.saveState()
        c.setFillColor(colors.HexColor(BERRY))
        c.circle(MARGIN + 4, top + 2.5, 4, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.circle(MARGIN + 4, top + 2.5, 1.6, stroke=0, fill=1)
        t = c.beginText(MARGIN + 12, top)
        t.setFont("Montserrat-Bold", 8)
        t.setCharSpace(2)
        t.setFillColor(colors.HexColor(BERRY))
        t.textOut("BRAND RADAR")
        c.drawText(t)
        c.restoreState()
        c.saveState()
        right = PAGE_W - MARGIN
        if rep.demo:
            right -= _demo_pill(c, right, top) + 8
        c.setFont("Figtree", 7.5)
        c.setFillColor(colors.HexColor(MUTED))
        c.drawRightString(right, top, printable(f"{rep.name} · {rep.week.label}"))
        # footer
        c.setStrokeColor(colors.HexColor(RULE))
        c.setLineWidth(0.5)
        c.line(MARGIN, 12 * mm, PAGE_W - MARGIN, 12 * mm)
        c.setFont("Figtree", 7)
        c.drawString(MARGIN, 8.5 * mm, printable(
            f"Brand Radar weekly report · {rep.name} · {rep.week.span} · generated {rep.generated_at:%d %b %Y %H:%M} UTC"
            + (" · demo data, fictional" if rep.demo else "")))
        c.drawRightString(PAGE_W - MARGIN, 8.5 * mm, f"Page {doc.page}")
        c.restoreState()

    return on_page


# ---------------------------------------------------------------- build


def build_pdf(rep: BrandReport, r: Renderer) -> bytes:
    """Three pages: overview (KPIs, summary, trend, categories), priorities (matrix, ranked topics,
    alerts), and competitors and voices (feature radar, quotes)."""
    register_fonts()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN + 4,
                            bottomMargin=16 * mm, title=f"Brand Radar weekly report: {rep.name}, {rep.week.label}",
                            author="Brand Radar", subject=f"{rep.name} {rep.week.span}")
    story: list[Flowable] = [
        *title_block(rep), kpi_row(rep), Spacer(1, 10),
        summary_section(rep), trend_section(rep, r), categories_section(rep, r),
        PageBreak(),
        matrix_section(rep, r), urgent_section(rep), alerts_section(rep),
        PageBreak(),
        radar_section(rep, r), quotes_section(rep),
    ]
    on_page = _decorate(rep)
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return buffer.getvalue()


def report_path(brand_name: str, week: Week, out_dir: Path | None = None) -> Path:
    return (out_dir or OUTPUT_DIR) / f"brandradar-{slug(brand_name)}-{week.label}.pdf"


def generate(engine: Engine, brand_refs: Sequence[str | int], week: Week, out_dir: Path = OUTPUT_DIR,
             r: Renderer | None = None) -> list[Path]:
    """Write one PDF per brand; one Chrome is shared by all of them."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    def run(rend: Renderer) -> None:
        for ref in brand_refs:
            rep = gather(engine, ref, week)
            path = report_path(rep.name, week, out_dir)
            path.write_bytes(build_pdf(rep, rend))
            paths.append(path)

    if r is not None:
        run(r)
    else:
        with renderer() as rend:
            run(rend)
    return paths


def pdf_bytes(engine: Engine, brand_ref: str | int, week: Week, r: Renderer | None = None) -> bytes:
    """One brand's report in memory (the dashboard's download button)."""
    if r is not None:
        return build_pdf(gather(engine, brand_ref, week), r)
    with renderer() as rend:
        return build_pdf(gather(engine, brand_ref, week), rend)


def default_brands(brands: pd.DataFrame, selection: Sequence[str] = ("own",)) -> list[str]:
    """NOTIFY_WEEKLY_BRANDS-style selection: 'own' = the own brand of every group, 'all', or names."""
    names: list[str] = []
    for item in selection:
        if item.lower() == "own":
            names += list(brands.loc[brands["is_own"].astype(bool), "name"])
        elif item.lower() == "all":
            names += list(brands["name"])
        else:
            names.append(str(find_brand(brands, item)["name"]))
    return list(dict.fromkeys(names))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--brand", action="append", help="brand name (repeatable)")
    which.add_argument("--group", help="every brand in this group")
    parser.add_argument("--week", help="ISO week, e.g. 2026-W41 (default: the last complete week)")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR, help=f"output folder (default {OUTPUT_DIR})")
    args = parser.parse_args(argv)

    from db.session import get_engine

    engine = get_engine()
    try:
        week = Week.parse(args.week) if args.week else Week.last_complete(utc_now().date())
    except ValueError as exc:
        parser.error(str(exc))
    brands = data.query_brands(engine)
    if args.group:
        refs = list(brands.loc[brands["brand_group"].str.lower() == args.group.lower(), "name"])
        if not refs:
            parser.error(f"unknown group {args.group!r}; groups: {', '.join(dict.fromkeys(brands['brand_group']))}")
    elif args.brand:
        refs = [str(find_brand(brands, b)["name"]) for b in args.brand]
    else:
        refs = default_brands(brands)
    print(f"Weekly report {week.label} ({week.span}){' · DEMO DATA' if settings.demo else ''}")
    for path in generate(engine, refs, week, args.out):
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
