"""Brand Radar dashboard (Streamlit). Reads MySQL only; never calls external APIs or loads NLP models.
The one exception is the "Track a new brand" card (ui/tracking.py), which writes to brand_requests only.

    streamlit run app.py
"""

from __future__ import annotations

import html
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import settings
from nlp.models import EMOTIONS
from nlp.language import language_name
from nlp.profiles import SOFTWARE, get_profile
from reports.export import csv_name, mentions_csv, slug
from reports.weeks import Week
from ui import charts, data, insights
from ui.tracking import track_card
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


def reset_group_choices() -> None:
    """A new group has other brands and sources: let those widgets start from their defaults."""
    for key in ("brand", "sources"):
        st.session_state.pop(key, None)


def header(all_brands: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[pd.Series], date, date, tuple[str, ...]]:
    """Group, brand, competitors, dates and sources. Everything below stays within the chosen group."""
    # groups in the order they were added (lowest brand id first): the hand-made groups stay first and the
    # default, groups tracked from the dashboard come after them
    groups = list(dict.fromkeys(all_brands.sort_values("id", kind="stable")["brand_group"]))
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
        if len(groups) > 1:
            group = brand_col.selectbox("Group", groups, key="group", on_change=reset_group_choices)
        else:
            group = groups[0]
        brands = all_brands[all_brands["brand_group"] == group].reset_index(drop=True)
        names = brands["name"].tolist()
        own_index = int(brands["is_own"].astype(bool).idxmax()) if brands["is_own"].any() else 0
        brand_name = brand_col.selectbox("Brand", names, index=own_index, key="brand")
        others = [n for n in names if n != brand_name]
        competitors = comp_col.multiselect("Competitors", others, default=others)
        picked = date_col.date_input(
            "Date range", value=(today - timedelta(days=DEFAULT_DAYS - 1), today), max_value=today, format="DD/MM/YYYY"
        )
        software = get_profile(brands["profile"].iloc[0]).name == SOFTWARE
        group_ids = tuple(int(i) for i in brands["id"])
        options = data.source_options(data.sources_with_data(group_ids), data.CORE_SOURCES if software else ())
        source_labels = source_col.pills(
            "Sources", list(options.values()), selection_mode="multi", default=list(options.values()),
            key="sources",
        )
    # date_input returns a 1-tuple while the user is still picking the end date
    if isinstance(picked, tuple):
        start, end = (picked[0], picked[-1]) if picked else (today - timedelta(days=DEFAULT_DAYS - 1), today)
    else:
        start = end = picked
    sources = tuple(k for k, v in options.items() if v in (source_labels or []))
    brand = brands.loc[brands["name"] == brand_name].iloc[0]
    comps = [brands.loc[brands["name"] == n].iloc[0] for n in competitors]
    return brands, brand, comps, start, end, sources


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
        own = charts.own_trend(trend, int(brand["id"]))
        if (end - start).days < 1 or own["day"].nunique() < 2:
            empty_state("Not enough data yet", f"{brand['name']} needs mentions on at least two days in this range "
                        "to draw a trend.", height=TREND_HEIGHT)
            return
        fig = charts.trend_figure(trend, (int(brand["id"]), brand["name"]),
                                  [(int(c["id"]), c["name"]) for c in comps], colours)
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
        fig = charts.matrix_figure(matrix, texts, settings.priority_volume_threshold, settings.priority_pain_threshold)
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
            f'<li><span class="br-rank-dot" style="background:{charts.QUADRANT_COLOURS[r["quadrant"]]}"></span>'
            f'<span class="br-rank-label">{html.escape(r["label"])}</span>'
            f'<span class="br-rank-meta">{r["quadrant"]} · {int(r["volume_7d"])} this week · pain {r["pain"]:.2f}</span>'
            + (f'<span class="br-rank-desc">{html.escape(str(r["description"]))}</span>'
               if isinstance(r.get("description"), str) and r["description"] else "") + '</li>'
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
        fig = charts.radar_figure(table, (int(brand["id"]), brand["name"]),
                                  [(int(c["id"]), c["name"]) for c in comps], colours)
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
                                     current, k.net_sentiment, k.net_sentiment_change, days,
                                     get_profile(brand["profile"]).request)
    written = data.briefing(int(brand["id"]))  # LLM briefing (nlp/llm/briefing.py), if the collector wrote one
    if written and written["end_date"] < date.today() - timedelta(days=1):
        written = None  # stale (collector stopped): the rule-based lines follow the filters instead
    with glass_card("summary"):
        if written:
            section_title("Executive summary", f"{brand['name']}, last 7 days · written from the numbers and quotes")
            st.markdown(
                f'<p class="br-briefing">{html.escape(written["text"])}</p>'
                f'<div class="br-ai-tag"><span class="br-ai-pill">AI-written</span> {written["start_date"]:%d %b} – '
                f'{written["end_date"]:%d %b} · {html.escape(written["model"])} · every number checked against the data</div>',
                unsafe_allow_html=True)
        else:
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
            original = ""
            if isinstance(q.get("original"), str) and q["original"].strip():  # translated by the LLM (Step 10)
                original = (f'<div class="br-quote-original"><span>Translated from {html.escape(language_name(q.get("language")))}'
                            f'</span> · {html.escape(" ".join(q["original"].split())[:240])}</div>')
            cards.append(
                f'<figure class="br-quote br-quote-{sentiment}"><blockquote>“{html.escape(" ".join(q["text"].split()))}”</blockquote>'
                f'{original}<figcaption><span class="br-quote-tag">{sentiment}</span> {meta}</figcaption></figure>'
            )
        st.markdown(f'<div class="br-label br-quotes-head">What people are saying</div><div class="br-quotes">{"".join(cards)}</div>',
                    unsafe_allow_html=True)


# ---------------------------------------------------------------- export


def report_week(end: date) -> Week:
    """The last full Monday-Sunday week up to the end of the date filter."""
    return Week.last_complete(end + timedelta(days=1))


def export_card(brand: pd.Series, ids: tuple[int, ...], start: date, end: date, sources: tuple[str, ...]) -> None:
    """Downloads only: the weekly PDF (built on click, kept for this session) and the filtered mentions as
    CSV. Both read MySQL; the PDF's charts are drawn by the local Chrome (no external API)."""
    week = report_week(end)
    with glass_card("export"):
        section_title("Export", f"Weekly PDF report for {brand['name']} and the mentions on screen as CSV")
        left, right = st.columns(2, gap="medium")
        with left:
            key = (int(brand["id"]), week.label, settings.demo)
            ready = st.session_state.get("pdf_report")
            if not ready or ready[0] != key:
                slot = st.empty()  # the build button disappears once the PDF is ready (same run, no rerun)
                if slot.button("Create PDF report", key="export-make-pdf", icon=":material/picture_as_pdf:"):
                    slot.empty()
                    with st.spinner(f"Building the {week.label} report…"):
                        try:
                            from reports.weekly import pdf_bytes

                            st.session_state["pdf_report"] = (key, pdf_bytes(data.get_engine(), int(brand["id"]), week))
                        except Exception as exc:  # noqa: BLE001 — e.g. Chrome missing; say so in the card
                            st.session_state["pdf_report"] = None
                            st.markdown(f'<div class="br-note br-export-error">Could not build the report: '
                                        f'{html.escape(type(exc).__name__)}: {html.escape(str(exc)[:200])}</div>',
                                        unsafe_allow_html=True)
                ready = st.session_state.get("pdf_report")
            if ready and ready[0] == key:
                st.download_button("Download PDF report", data=ready[1], key="export-pdf", type="primary",
                                   file_name=f"brandradar-{slug(str(brand['name']))}-{week.label}.pdf",
                                   mime="application/pdf", icon=":material/download:")
            st.markdown(f'<div class="br-note">{week.span} ({week.label}) · 2–3 pages: KPIs, trend, priority matrix, '
                        'feature radar, alerts and quotes</div>', unsafe_allow_html=True)
        with right:
            rows = data.export_mentions(ids, start, end, sources)
            st.download_button("Download CSV", data=mentions_csv(rows), key="export-csv", disabled=rows.empty,
                               file_name=csv_name(str(brand["name"]), start, end), mime="text/csv",
                               icon=":material/table_view:")
            st.markdown(f'<div class="br-note">{len(rows):,} mentions with their analysis · {start:%d %b} – {end:%d %b} · '
                        'selected brands and sources. Only what is stored: YouTube comments are kept for 30 days.</div>',
                        unsafe_allow_html=True)


# ---------------------------------------------------------------- page


def main() -> None:
    brands = data.brands()
    if brands.empty:
        with glass_card("setup"):
            empty_state("No brands yet", "Track one below, or run python manage_brands.py, then fetch.py.")
        track_card()
        return
    brands, brand, comps, start, end, sources = header(brands)
    if not sources:
        with glass_card("no-sources"):
            if data.sources_with_data(tuple(int(i) for i in brands["id"])):
                empty_state("No sources selected", "Pick at least one source in the header to see data.")
            else:
                group = brands["brand_group"].iloc[0]
                empty_state(f"No mentions for {group} yet",
                            "Run python fetch.py, then the NLP pipeline (or wait for the collector's next cycle).")
        track_card()
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
    export_card(brand, ids, start, end, sources)
    track_card()
    st.markdown(f'<div class="br-footnote">{start:%d %b} – {end:%d %b %Y} · '
                f'{", ".join(data.SOURCES.get(s, s.title()) for s in sources)} · data refreshes every 60 s</div>',
                unsafe_allow_html=True)


main()
