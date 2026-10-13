"""Brand Radar dashboard (Streamlit). Reads MySQL only; never calls external APIs or loads NLP models.

    streamlit run app.py
"""

from __future__ import annotations

import html
import math
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import settings
from nlp.models import EMOTIONS
from nlp.priority import FIX_NOW, IGNORE, MONITOR, QUICK_WINS
from ui import data, insights
from ui.components import (
    BERRY, COMPLAINT_CLOUD_COLOURS, EMOTION_COLOURS, INK, PRAISE_CLOUD_COLOURS, SENTIMENT_COLOURS,
    background_html, brand_colours, delta_html, empty_state, glass_card, inject_styles, kpi_card,
    section_title, show_chart, status_html, wordcloud_png, wordmark_html,
)

DEFAULT_DAYS = 14  # previous 14 days exist in our ~30 days of data, so KPI deltas are real
MIN_FOR_CHART = 5  # fewer mentions than this -> designed empty state instead of a chart
TREND_HEIGHT = 380
MATRIX_HEIGHT = 440
RADAR_HEIGHT = 420
WHEEL_HEIGHT = TREND_HEIGHT - 44  # the wheel card also holds the "compare with" control

st.set_page_config(page_title="Brand Radar", page_icon="📡", layout="wide", initial_sidebar_state="collapsed")
inject_styles()
st.markdown(background_html(), unsafe_allow_html=True)


# ---------------------------------------------------------------- header + filters


def header(brands: pd.DataFrame) -> tuple[pd.Series, list[pd.Series], date, date, tuple[str, ...]]:
    names = brands["name"].tolist()
    own_index = int(brands["is_own"].astype(bool).idxmax()) if brands["is_own"].any() else 0
    today = date.today()
    with glass_card("header"):
        logo, brand_col, comp_col, date_col, source_col = st.columns(
            [1.35, 0.95, 2.0, 1.4, 1.3], vertical_alignment="top", gap="medium"
        )
        with logo:
            demo = ('<div class="br-demo-row"><em class="br-demo-pill" title="BRAND_RADAR_DEMO=1: fictional data '
                    'from seed_demo.py">Demo data</em></div>' if settings.demo else "")
            st.markdown(wordmark_html() + f'<div class="br-subtitle">Sentiment &amp; competitor radar</div>{demo}',
                        unsafe_allow_html=True)
            live_status()
        brand_name = brand_col.selectbox("Brand", names, index=own_index, key="brand")
        others = [n for n in names if n != brand_name]
        competitors = comp_col.multiselect("Competitors", others, default=others)
        picked = date_col.date_input(
            "Date range", value=(today - timedelta(days=DEFAULT_DAYS - 1), today), max_value=today, format="DD/MM/YYYY"
        )
        source_labels = source_col.pills(
            "Sources", list(data.SOURCES.values()), selection_mode="multi", default=list(data.SOURCES.values()),
            key="sources",
        )
    # date_input returns a 1-tuple while the user is still picking the end date
    if isinstance(picked, tuple):
        start, end = (picked[0], picked[-1]) if picked else (today - timedelta(days=DEFAULT_DAYS - 1), today)
    else:
        start = end = picked
    sources = tuple(k for k, v in data.SOURCES.items() if v in (source_labels or []))
    brand = brands.loc[brands["name"] == brand_name].iloc[0]
    comps = [brands.loc[brands["name"] == n].iloc[0] for n in competitors]
    return brand, comps, start, end, sources


# ---------------------------------------------------------------- live parts (refresh without reloading the page)


@st.fragment(run_every="60s")
def live_status() -> None:
    from datetime import datetime, timezone

    latest, last_ok = data.collector_runs()
    s = data.collector_status(latest, last_ok, datetime.now(timezone.utc).replace(tzinfo=None),
                              settings.collector_interval_hours)
    st.markdown(status_html(s.text, s.warning, s.running, s.has_runs), unsafe_allow_html=True)


@st.fragment(run_every="60s")
def live_kpis(brand: pd.Series, start: date, end: date, sources: tuple[str, ...], days: int) -> None:
    prev_start, prev_end = data.previous_period(start, end)
    current = data.mentions((int(brand["id"]),), start, end, sources)
    previous = data.mentions((int(brand["id"]),), prev_start, prev_end, sources)
    kpi_row(brand, current, previous, days)


@st.fragment(run_every="60s")
def live_alerts(brand: pd.Series, start: date, end: date) -> None:
    alert_pills(brand, start, end)


# ---------------------------------------------------------------- KPI row


def kpi_row(brand: pd.Series, current: pd.DataFrame, previous: pd.DataFrame, days: int) -> None:
    k = data.compute_kpis(current, previous)
    alerts = data.open_alerts(int(brand["id"]))
    with st.container(key="kpis"):
        c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        if k.net_sentiment is None:
            kpi_card("Net sentiment", "–", '<span class="br-delta br-delta-none">no mentions in range</span>')
        else:
            colour = SENTIMENT_COLOURS["positive"] if k.net_sentiment > 0 else (
                SENTIMENT_COLOURS["negative"] if k.net_sentiment < 0 else INK)
            kpi_card("Net sentiment", f"{k.net_sentiment:+.0f}",
                     delta_html(k.net_sentiment_change, " pts", good_when_up=True, period_days=days), colour)
    with c2:
        change = None if k.mentions_change_pct is None else round(k.mentions_change_pct)
        kpi_card("Mentions", f"{k.mentions:,}", delta_html(change, "%", good_when_up=None, period_days=days))
    with c3:
        if k.top_emotion is None:
            kpi_card("Top emotion", "–", '<span class="br-delta br-delta-none">no mentions in range</span>')
        else:
            dot = f'<span class="br-emotion-dot" style="background:{EMOTION_COLOURS[k.top_emotion]}"></span>'
            detail = (f"{k.top_emotion_share:.0%} of emotion · then "
                      f"<b style='color:{EMOTION_COLOURS[k.runner_up_emotion]}'>{k.runner_up_emotion}</b> "
                      f"{k.runner_up_share:.0%}")
            kpi_card("Top emotion", k.top_emotion.capitalize(), detail, extra_html=dot)
    with c4:
        if alerts:
            kpi_card("Open alerts", str(alerts), "unseen spike alerts for this brand",
                     SENTIMENT_COLOURS["negative"], extra_html='<span class="br-alert-dot"></span>')
        else:
            kpi_card("Open alerts", "0", '<span class="br-all-clear">All clear</span> · no unseen spikes')


# ---------------------------------------------------------------- charts


def emotion_wheel(brand: pd.Series, comps: list[pd.Series], current: pd.DataFrame, colours: dict[str, str]) -> None:
    with glass_card("emotions"):
        section_title("Emotion wheel", "Share of emotion across all mentions (neutral excluded) · aggregate, not per post")
        own = current[current["brand_id"] == brand["id"]]
        if len(own) < MIN_FOR_CHART:
            empty_state("Not enough data yet", f"{brand['name']} has {len(own)} mention(s) in this range. "
                        "Widen the dates or add sources to see its emotional mix.", height=TREND_HEIGHT)
            return
        rival = None
        if comps:
            # default to the competitor with the most mentions in range (thin ones make a noisy wheel)
            busiest = max(comps, key=lambda c: int((current["brand_id"] == c["id"]).sum()))
            choice = st.segmented_control("Compare with", [c["name"] for c in comps], default=busiest["name"],
                                          key="wheel_rival", label_visibility="collapsed")
            rival = next((c for c in comps if c["name"] == choice), None)
        emotions = [e for e in EMOTIONS if e != "neutral"]
        labels = [e.capitalize() for e in emotions]
        fig = go.Figure()
        shares = data.emotion_shares(own, exclude=("neutral",)) * 100
        fig.add_trace(go.Barpolar(
            r=shares[emotions].round(1), theta=labels, name=brand["name"],
            marker=dict(color=[EMOTION_COLOURS[e] for e in emotions], line=dict(color="white", width=1.5)),
            opacity=0.88, hovertemplate="%{theta}: %{r:.0f}%<extra>" + brand["name"] + "</extra>",
        ))
        if rival is not None:
            theirs = current[current["brand_id"] == rival["id"]]
            if len(theirs) >= MIN_FOR_CHART:
                r_shares = data.emotion_shares(theirs, exclude=("neutral",)) * 100
                fig.add_trace(go.Barpolar(
                    r=r_shares[emotions].round(1), theta=labels, name=rival["name"],
                    marker=dict(color="rgba(255,255,255,0.05)", line=dict(color=colours[rival["name"]], width=2.5)),
                    hovertemplate="%{theta}: %{r:.0f}%<extra>" + rival["name"] + "</extra>",
                ))
            else:
                st.caption(f"{rival['name']} has too few mentions in this range to compare.")
        fig.update_layout(
            # radial % labels sit between Anger and Disgust instead of on top of the Anger spoke
            polar=dict(radialaxis=dict(ticksuffix="%", angle=60, tickangle=60, nticks=4, showline=False,
                                       tickfont=dict(size=9)),
                       angularaxis=dict(direction="clockwise", rotation=90)),
            legend=dict(y=-0.06, yanchor="top", x=0.5, xanchor="center"), barmode="overlay",
            margin=dict(l=36, r=36, t=16, b=36),
        )
        show_chart(fig, height=WHEEL_HEIGHT)


def sentiment_trend(brand: pd.Series, comps: list[pd.Series], current: pd.DataFrame, start: date, end: date,
                    colours: dict[str, str]) -> None:
    with glass_card("trend"):
        section_title("Daily sentiment", "Net sentiment ×100, 3-day rolling mean weighted by mentions")
        trend = data.daily_trend(current, start, end)
        own = trend[(trend["brand_id"] == brand["id"]) & trend["score"].notna()]
        if (end - start).days < 1 or own["day"].nunique() < 2:
            empty_state("Not enough data yet", f"{brand['name']} needs mentions on at least two days in this range "
                        "to draw a trend.", height=TREND_HEIGHT)
            return
        fig = go.Figure()
        for comp in comps:
            line = trend[(trend["brand_id"] == comp["id"]) & trend["score"].notna()]
            if line["day"].nunique() < 2:
                continue
            fig.add_trace(go.Scatter(
                x=line["day"], y=line["score"], name=comp["name"], mode="lines",
                line=dict(color=colours[comp["name"]], width=1.6, shape="spline", smoothing=0.6),
                customdata=line["mentions"],
                hovertemplate="%{x|%d %b}: %{y:+.0f}<br>%{customdata} mention(s) that day<extra>" + comp["name"] + "</extra>",
            ))
        fig.add_trace(go.Scatter(
            x=own["day"], y=own["score"], name=brand["name"], mode="lines",
            line=dict(color=BERRY, width=3.2, shape="spline", smoothing=0.6),
            fill="tozeroy", fillcolor="rgba(158,74,90,0.10)", customdata=own["mentions"],
            hovertemplate="%{x|%d %b}: %{y:+.0f}<br>%{customdata} mention(s) that day<extra>" + brand["name"] + "</extra>",
        ))
        last = own.iloc[-1]
        fig.add_trace(go.Scatter(
            x=[last["day"]], y=[last["score"]], mode="markers+text", showlegend=False, hoverinfo="skip",
            marker=dict(size=13, color=BERRY, line=dict(color="white", width=3)),
            text=[f"{last['score']:+.0f}"], textposition="top center",
            textfont=dict(family="Montserrat, sans-serif", size=13, color=BERRY),
        ))
        fig.update_layout(
            # range a little past ±100 so the -100 label doesn't collide with the date labels
            yaxis=dict(range=[-112, 112], tickvals=[-100, -50, 0, 50, 100], showgrid=True, zeroline=True),
            xaxis=dict(tickformat="%d %b", showgrid=False), hovermode="x unified",
            legend=dict(y=1.06), margin=dict(l=8, r=24, t=34, b=8),
        )
        show_chart(fig, height=TREND_HEIGHT)


def word_clouds(brand: pd.Series, current: pd.DataFrame, brands: pd.DataFrame) -> None:
    own = current[current["brand_id"] == brand["id"]]
    stop = frozenset(data.cloud_stop_words(brands))
    left, right = st.columns(2, gap="medium")
    for column, key, title, caption, sentiment, palette in (
        (left, "cloud-neg", "What users complain about", "Most frequent words in negative mentions",
         "negative", COMPLAINT_CLOUD_COLOURS),
        (right, "cloud-pos", "What users love", "Most frequent words in positive mentions",
         "positive", PRAISE_CLOUD_COLOURS),
    ):
        with column, glass_card(key):
            section_title(title, caption)
            texts = tuple(own.loc[own["sentiment"] == sentiment, "text"].astype(str))
            if len(texts) < MIN_FOR_CHART:
                empty_state("Not enough data yet", f"Only {len(texts)} {sentiment} mention(s) for {brand['name']} "
                            "in this range — a cloud needs at least five.", height=240)
                continue
            st.image(wordcloud_png(texts, stop, palette), use_container_width=True)


# ---------------------------------------------------------------- priority matrix

QUADRANT_COLOURS = {FIX_NOW: "#D9536A", QUICK_WINS: BERRY, MONITOR: "#E79AA6", IGNORE: "#B7AEB0"}
# very soft tints of the theme's rose/blush; Fix now slightly stronger
QUADRANT_TINTS = {FIX_NOW: "rgba(217,83,106,0.11)", QUICK_WINS: "rgba(231,154,166,0.08)",
                  MONITOR: "rgba(243,186,193,0.08)", IGNORE: "rgba(240,220,219,0.10)"}
MENTION_TICKS = (0, 1, 2, 5, 10, 20, 50, 100)


def priority_matrix(brand: pd.Series) -> pd.DataFrame:
    """Matrix card; returns the plotted topics (the executive summary reuses them)."""
    topics = data.topics(int(brand["id"]))
    with glass_card("matrix"):
        scored = topics["scored_at"].dropna()
        when = f" · snapshot {pd.Timestamp(scored.max()):%d %b %H:%M}" if not scored.empty else ""
        section_title("Priority matrix", "Complaint topics by volume (last 7 days) and pain — top right needs "
                      f"attention first{when}")
        if topics.empty:
            empty_state("No complaint topics yet", f"{brand['name']} has too few complaints to cluster into topics "
                        f"(at least {settings.topic_min_complaints} are needed).", height=360)
            return topics
        show_all = st.toggle("Show all topics", key="show_all_topics",
                             help="Praise and neutral-discussion topics (near-zero pain, positive on average) are hidden by default.")
        matrix = insights.topic_matrix(topics, settings.priority_volume_threshold, settings.priority_pain_threshold,
                                       settings.priority_praise_pain_max, show_all=show_all)
        hidden = len(topics) - len(matrix)
        if matrix.empty:
            empty_state("Only praise topics", f"All {len(topics)} topics are praise or discussion. "
                        "Switch on “Show all topics” to see them.", height=360)
            return matrix
        texts = data.topic_texts(tuple(int(t) for t in matrix["id"]))
        x_split = math.log1p(settings.priority_volume_threshold)
        x_max = max(matrix["x"].max() + 0.45, x_split + 0.9)
        y_max = max(matrix["y"].max() * 1.18, settings.priority_pain_threshold * 2)
        fig = go.Figure()
        # quadrant names sit at the top of each quadrant: the bottom corners are where low-pain topics cluster
        corners = {FIX_NOW: (x_split, x_max, settings.priority_pain_threshold, y_max, "right"),
                   QUICK_WINS: (-0.12, x_split, settings.priority_pain_threshold, y_max, "left"),
                   MONITOR: (x_split, x_max, -0.015, settings.priority_pain_threshold, "right"),
                   IGNORE: (-0.12, x_split, -0.015, settings.priority_pain_threshold, "left")}
        for name, (x0, x1, y0, y1, xa) in corners.items():
            fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, line_width=0, fillcolor=QUADRANT_TINTS[name], layer="below")
            fig.add_annotation(x=x1 - 0.04 if xa == "right" else x0 + 0.04, y=y1, yshift=-4,
                               xanchor=xa, yanchor="top", showarrow=False, text=name.upper(),
                               font=dict(family="Montserrat, sans-serif", size=10, color=QUADRANT_COLOURS[name]))
        biggest = matrix["mentions"].max()
        for name in insights.QUADRANTS:
            part = matrix[matrix["quadrant"] == name]
            if part.empty:
                continue
            hovers = []
            for _, t in part.iterrows():
                quotes = "".join(f"<br><i>“{html.escape(q)}”</i>" for q in insights.topic_quotes(texts, int(t["id"])))
                hovers.append(
                    f"<b>{html.escape(t['label'])}</b><br>{html.escape(str(t['category'] or ''))} · {name}"
                    f"<br>{int(t['volume_7d'])} this week vs {int(t['volume_prev_7d'])} last week "
                    f"({insights.trend_text(int(t['volume_7d']), int(t['volume_prev_7d']))})"
                    f"<br>{int(t['mentions'])} mentions in total · pain {t['pain']:.2f}{quotes}"
                )
            fig.add_trace(go.Scatter(
                # Ignore bubbles stay unlabelled (hover only): low-pain topics cluster and their labels collide
                x=part["x"], y=part["y"], mode="markers+text", name=name,
                text=part["label"] if name != IGNORE else [""] * len(part),
                textposition="top center", textfont=dict(family="Figtree, sans-serif", size=11, color=INK),
                marker=dict(size=[14 + 26 * math.sqrt(m / biggest) for m in part["mentions"]],
                            color=QUADRANT_COLOURS[name], opacity=0.82, line=dict(color="white", width=2)),
                hovertext=hovers, hovertemplate="%{hovertext}<extra></extra>", cliponaxis=False,
            ))
        ticks = [t for t in MENTION_TICKS if math.log1p(t) <= x_max]
        fig.update_layout(
            showlegend=False, margin=dict(l=8, r=12, t=12, b=8),
            xaxis=dict(range=[-0.12, x_max], tickvals=[math.log1p(t) for t in ticks], ticktext=[str(t) for t in ticks],
                       title=dict(text="Complaints in the last 7 days (log scale)", font=dict(size=11)), showgrid=False,
                       zeroline=False),
            yaxis=dict(range=[-0.015, y_max], title=dict(text="Pain", font=dict(size=11)), showgrid=False,
                       zeroline=False, tickformat=".2f"),
            hoverlabel=dict(align="left"),
        )
        show_chart(fig, height=MATRIX_HEIGHT)
        if hidden and not show_all:
            st.markdown(f'<div class="br-note">{hidden} praise/discussion topic(s) hidden · '
                        f'unlabelled bubbles are Ignore topics (hover for details)</div>', unsafe_allow_html=True)
        return matrix


def ranked_topics(matrix: pd.DataFrame) -> None:
    """Compact ranked list of Fix now and Quick wins topics (beside the matrix)."""
    if matrix.empty or "quadrant" not in matrix:
        return
    urgent = insights.ranked_urgent(matrix)
    with glass_card("ranked"):
        section_title("Act on these first", "Fix now, then Quick wins · ranked by volume × pain")
        if urgent.empty:
            st.markdown('<div class="br-calm br-calm-compact"><div class="br-calm-title">Nothing urgent</div>'
                        '<div class="br-calm-text">No topics in Fix now or Quick wins.</div></div>', unsafe_allow_html=True)
            return
        rows = "".join(
            f'<li><span class="br-rank-dot" style="background:{QUADRANT_COLOURS[r["quadrant"]]}"></span>'
            f'<span class="br-rank-label">{html.escape(r["label"])}</span>'
            f'<span class="br-rank-meta">{r["quadrant"]} · {int(r["volume_7d"])} this week · pain {r["pain"]:.2f}</span></li>'
            for _, r in urgent.iterrows()
        )
        st.markdown(f'<ol class="br-rank">{rows}</ol>', unsafe_allow_html=True)


# ---------------------------------------------------------------- alerts


def alert_pills(brand: pd.Series, start: date, end: date) -> None:
    alerts = data.alerts(int(brand["id"]), start, end)
    with glass_card("alerts"):
        section_title("Trending complaint alerts", "Topics whose complaints at least doubled week over week")
        if alerts.empty:
            st.markdown(
                f'<div class="br-calm"><div class="br-calm-icon">✓</div><div class="br-calm-title">No spikes this week</div>'
                f'<div class="br-calm-text">Complaint volumes for {html.escape(brand["name"])} are steady in this period.</div></div>',
                unsafe_allow_html=True,
            )
            return
        pills = []
        for _, a in alerts.iterrows():
            this_week, last_week = int(a["this_week"] or 0), int(a["last_week"] or 0)
            label = a["label"] if isinstance(a["label"], str) else a["message"]
            seen = " br-alert-seen" if a["seen"] else ""
            pills.append(
                f'<div class="br-alert-pill{seen}"><span class="br-alert-dot"></span>'
                f'<div class="br-alert-body"><div class="br-alert-label">{html.escape(label)}</div>'
                f'<div class="br-alert-meta"><b>{insights.trend_text(this_week, last_week)}</b> · {this_week} this week'
                f' · {pd.Timestamp(a["created_at"]):%d %b}</div></div></div>'
            )
        st.markdown(f'<div class="br-alert-list">{"".join(pills)}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- feature radar


def feature_radar(brand: pd.Series, comps: list[pd.Series], colours: dict[str, str]) -> insights.FeatureTable:
    ids = [int(brand["id"]), *(int(c["id"]) for c in comps)]
    feature_df = data.feature_mentions(tuple(ids))
    table = insights.feature_table(feature_df, ids)
    with glass_card("radar"):
        since = f" since {feature_df['created_at'].min():%d %b}" if not feature_df.empty else ""
        section_title("Feature radar", f"Net sentiment per feature, −100 (all negative) to +100 (all positive) · "
                      f"all data{since}, not the date filter")
        if len(table.features) < insights.MIN_RADAR_FEATURES:
            empty_state("Not enough feature mentions", f"Fewer than {insights.MIN_RADAR_FEATURES} features have "
                        f"{insights.MIN_FEATURE_MENTIONS}+ mentions for these brands.", height=RADAR_HEIGHT)
            return table
        theta = [f.title() if f.islower() else f for f in table.features]
        fig = go.Figure()
        ring = [0] * (len(theta) + 1)
        fig.add_trace(go.Scatterpolar(r=ring, theta=theta + theta[:1], mode="lines", hoverinfo="skip",
                                      line=dict(color="#E79AA6", width=1, dash="dot"), showlegend=False))
        for b in [*comps, brand]:
            bid = int(b["id"])
            if bid not in table.net.index:
                continue
            values = table.net.loc[bid].tolist()
            counts = table.counts.loc[bid].tolist()
            if all(pd.isna(v) for v in values):
                continue
            own = bid == int(brand["id"])
            hover = [f"{t}: {v:+.0f} ({n} mentions)" if not pd.isna(v) else f"{t}: too few mentions ({n})"
                     for t, v, n in zip(theta, values, counts)]
            r = [None if pd.isna(v) else v for v in values]
            fig.add_trace(go.Scatterpolar(
                r=r + r[:1], theta=theta + theta[:1], name=b["name"], connectgaps=True,
                mode="lines+markers", hovertext=hover + hover[:1], hovertemplate="%{hovertext}<extra>" + b["name"] + "</extra>",
                line=dict(color=colours[b["name"]], width=3 if own else 2),
                marker=dict(size=7 if own else 5, color=colours[b["name"]]),
                fill="toself" if own else None, fillcolor="rgba(158,74,90,0.16)" if own else None,
            ))
        fig.update_layout(
            # radial labels between the first two spokes, so they don't sit on a feature's axis
            polar=dict(radialaxis=dict(range=[-100, 100], tickvals=[-100, -50, 0, 50, 100],
                                       angle=90 - 180 / len(theta), tickangle=90 - 180 / len(theta),
                                       tickfont=dict(size=9), showline=False),
                       angularaxis=dict(direction="clockwise", rotation=90)),
            # side margins fit the longest axis label ("Performance") at phone width; Plotly clips text to its SVG
            legend=dict(y=-0.06, yanchor="top", x=0.5, xanchor="center"), margin=dict(l=64, r=64, t=24, b=40),
        )
        show_chart(fig, height=RADAR_HEIGHT)
    return table


# ---------------------------------------------------------------- executive summary


def stars(rating: object) -> str:
    if rating is None or pd.isna(rating):
        return ""
    n = int(rating)
    return f'<span class="br-stars" aria-label="{n} of 5 stars">{"★" * n}{"☆" * (5 - n)}</span>'


def executive_summary(brand: pd.Series, comps: list[pd.Series], matrix: pd.DataFrame, table: insights.FeatureTable,
                      current: pd.DataFrame, previous: pd.DataFrame, days: int) -> None:
    own = current[current["brand_id"] == brand["id"]]
    k = data.compute_kpis(own, previous)
    summary = insights.summary_picks(brand["name"], matrix, table, int(brand["id"]), [int(c["id"]) for c in comps],
                                     current, k.net_sentiment, k.net_sentiment_change, days)
    with glass_card("summary"):
        section_title("Executive summary", f"{brand['name']} this period, in 20 seconds · built from the numbers above")
        lines = [("risk", summary.risk), ("win", summary.win), ("request", summary.request), ("change", summary.change)]
        st.markdown('<ul class="br-summary">' + "".join(
            f'<li class="br-summary-{kind}"><span class="br-summary-mark"></span><span>{line}</span></li>'
            for kind, line in lines) + "</ul>", unsafe_allow_html=True)
        quotes = insights.pick_quotes(own)
        if not quotes:
            return
        cards = []
        for sentiment in ("negative", "neutral", "positive"):
            if sentiment not in quotes:
                continue
            q = quotes[sentiment]
            meta = f'{data.SOURCES.get(q["source"], q["source"])} {stars(q["rating"])} · {pd.Timestamp(q["created_at"]):%d %b}'
            cards.append(
                f'<figure class="br-quote br-quote-{sentiment}"><blockquote>“{html.escape(" ".join(q["text"].split()))}”</blockquote>'
                f'<figcaption><span class="br-quote-tag">{sentiment}</span> {meta}</figcaption></figure>'
            )
        st.markdown(f'<div class="br-label br-quotes-head">What people are saying</div><div class="br-quotes">{"".join(cards)}</div>',
                    unsafe_allow_html=True)


# ---------------------------------------------------------------- page


def main() -> None:
    brands = data.brands()
    if brands.empty:
        with glass_card("setup"):
            empty_state("No brands yet", "Run python manage_brands.py, then fetch.py and the NLP pipeline.")
        return
    brand, comps, start, end, sources = header(brands)
    if not sources:
        with glass_card("no-sources"):
            empty_state("No sources selected", "Pick at least one source in the header to see data.")
        return
    days = (end - start).days + 1
    prev_start, prev_end = data.previous_period(start, end)
    ids = tuple(int(i) for i in [brand["id"], *(c["id"] for c in comps)])
    current = data.mentions(ids, start, end, sources)
    previous = data.mentions((int(brand["id"]),), prev_start, prev_end, sources)
    colours = brand_colours(brand["name"], [c["name"] for c in comps])

    live_kpis(brand, start, end, sources, days)
    left, right = st.columns([1, 1.45], gap="medium")
    with left:
        emotion_wheel(brand, comps, current, colours)
    with right:
        sentiment_trend(brand, comps, current, start, end, colours)
    word_clouds(brand, current, brands)

    wide, side = st.columns([1.7, 1], gap="medium")
    with wide:
        matrix = priority_matrix(brand)
    with side:
        live_alerts(brand, start, end)
        ranked_topics(matrix)
    left, right = st.columns(2, gap="medium")
    with left:
        table = feature_radar(brand, comps, colours)
    with right:
        executive_summary(brand, comps, matrix, table, current, previous, days)
    st.markdown(f'<div class="br-footnote">{start:%d %b} – {end:%d %b %Y} · '
                f'{", ".join(data.SOURCES[s] for s in sources)} · data refreshes every 60 s</div>',
                unsafe_allow_html=True)


main()
