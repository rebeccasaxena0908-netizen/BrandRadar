"""Brand Radar dashboard (Streamlit). Reads MySQL only; never calls external APIs or loads NLP models.

    streamlit run app.py
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nlp.models import EMOTIONS
from ui import data
from ui.components import (
    BERRY, COMPLAINT_CLOUD_COLOURS, EMOTION_COLOURS, INK, PRAISE_CLOUD_COLOURS, SENTIMENT_COLOURS,
    background_html, brand_colours, delta_html, empty_state, glass_card, inject_styles, kpi_card,
    section_title, show_chart, wordcloud_png, wordmark_html,
)

DEFAULT_DAYS = 14  # previous 14 days exist in our ~30 days of data, so KPI deltas are real
MIN_FOR_CHART = 5  # fewer mentions than this -> designed empty state instead of a chart
TREND_HEIGHT = 380
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
            st.markdown(wordmark_html() + '<div class="br-subtitle">Sentiment &amp; competitor radar</div>',
                        unsafe_allow_html=True)
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

    kpi_row(brand, current[current["brand_id"] == brand["id"]], previous, days)
    left, right = st.columns([1, 1.45], gap="medium")
    with left:
        emotion_wheel(brand, comps, current, colours)
    with right:
        sentiment_trend(brand, comps, current, start, end, colours)
    word_clouds(brand, current, brands)
    st.markdown(f'<div class="br-footnote">{start:%d %b} – {end:%d %b %Y} · '
                f'{", ".join(data.SOURCES[s] for s in sources)} · data refreshes every 60 s</div>',
                unsafe_allow_html=True)


main()
