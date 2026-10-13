"""Blush glass UI helpers: styles, moving background, glass cards, KPI cards, Plotly template."""

from __future__ import annotations

import html
import io
import random
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

UI_DIR = Path(__file__).resolve().parent
FONT_FILE = UI_DIR / "fonts" / "Montserrat-SemiBold.ttf"

# ---------------------------------------------------------------- palette (CLAUDE.md design system)

INK, INK_SOFT, MUTED, RULE = "#242424", "#6B5F62", "#9C9899", "#E7CFCF"
BERRY = "#9E4A5A"
COMPETITOR_COLOURS = ("#E79AA6", "#B7AEB0", "#F3BAC1")
SENTIMENT_COLOURS = {"positive": "#6FA98A", "neutral": "#B7AEB0", "negative": "#D9536A"}
EMOTION_COLOURS = {
    "joy": "#E9B949", "anger": "#D9536A", "sadness": "#7FA7D9", "fear": "#A78BC9",
    "disgust": "#8FB26B", "surprise": "#E79AA6", "neutral": "#B7AEB0",
}
COMPLAINT_CLOUD_COLOURS = (BERRY, "#D9536A", "#E79AA6", "#6B5F62", "#B85868")
PRAISE_CLOUD_COLOURS = ("#6FA98A", BERRY, "#E9B949", "#4F8A6C", "#E79AA6")

FONTS_CSS = (
    "@import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700"
    "&family=Montserrat:wght@500;600;700;800&display=swap');"
)


def brand_colours(own: str, competitors: Sequence[str]) -> dict[str, str]:
    """Own brand is always berry; competitors take the competitor colours in order."""
    colours = {own: BERRY}
    for i, name in enumerate(competitors):
        colours[name] = COMPETITOR_COLOURS[i % len(COMPETITOR_COLOURS)]
    return colours


# ---------------------------------------------------------------- Plotly template

pio.templates["blush"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Figtree, system-ui, sans-serif", size=12, color=INK_SOFT),
        title=dict(font=dict(family="Montserrat, sans-serif", size=15, color=INK)),
        colorway=[BERRY, *COMPETITOR_COLOURS],
        xaxis=dict(gridcolor=RULE, linecolor=RULE, zerolinecolor=RULE, tickfont=dict(color=MUTED), showgrid=False,
                   automargin=True),
        yaxis=dict(gridcolor=RULE, linecolor=RULE, zerolinecolor=RULE, tickfont=dict(color=MUTED), zerolinewidth=1.5,
                   automargin=True),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(gridcolor=RULE, linecolor=RULE, tickfont=dict(color=MUTED, size=10)),
            angularaxis=dict(gridcolor=RULE, linecolor=RULE, tickfont=dict(color=INK_SOFT, size=12)),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(color=INK_SOFT, size=12), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="white", bordercolor=RULE, font=dict(family="Figtree, sans-serif", color=INK, size=12)),
        margin=dict(l=8, r=8, t=8, b=8),
    )
)
PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


def show_chart(fig: go.Figure, height: int = 340) -> None:
    # Backgrounds are set explicitly as well: Streamlit's front end paints a white paper otherwise.
    fig.update_layout(template="blush", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, theme=None)


# ---------------------------------------------------------------- styles and background


def inject_styles() -> None:
    """Fonts, the approved theme (ui/styles.css, untouched) and our additions (ui/streamlit.css)."""
    css = (UI_DIR / "styles.css").read_text(encoding="utf-8") + "\n" + (UI_DIR / "streamlit.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{FONTS_CSS}\n{css}</style>", unsafe_allow_html=True)


def background_html(seed: int = 7, bubbles: int = 16) -> str:
    """The moving background from the approved theme: 4 blobs + `bubbles` rising glass bubbles.

    st.markdown runs no JavaScript, so each bubble's randomness is baked into inline styles here,
    with a fixed seed so reruns produce identical HTML (bubbles never jump)."""
    rng = random.Random(seed)
    # .br-sky-layer (ui/streamlit.css) keeps the approved .br-sky behind every card: Streamlit renders
    # this inside .block-container, which the theme makes a stacking context (z-index: 1).
    parts = ['<div class="br-sky-layer"><div class="br-sky" aria-hidden="true">',
             *(f'<div class="br-blob br-b{i}"></div>' for i in range(1, 5))]
    for _ in range(bubbles):
        size = rng.randint(18, 88)
        sway = rng.choice((-1, 1)) * rng.randint(20, 80)
        duration = rng.uniform(12, 26)
        parts.append(
            f'<span class="br-bubble" style="width:{size}px;height:{size}px;left:{rng.uniform(0, 96):.1f}vw;'
            f'--sway:{sway}px;animation-duration:{duration:.1f}s;animation-delay:-{rng.uniform(0, duration):.1f}s"></span>'
        )
    parts.append("</div></div>")
    return "".join(parts)


# ---------------------------------------------------------------- cards


@contextmanager
def glass_card(key: str) -> Iterator[None]:
    """A Streamlit container styled as a glass card (ui/streamlit.css targets .st-key-glass-*)."""
    with st.container(key=f"glass-{key}"):
        yield


def section_title(title: str, caption: str = "") -> None:
    caption_html = f'<div class="br-card-caption">{html.escape(caption)}</div>' if caption else ""
    st.markdown(f'<div class="br-card-head"><div class="br-card-title">{html.escape(title)}</div>{caption_html}</div>',
                unsafe_allow_html=True)


def kpi_card(label: str, value: str, detail_html: str = "", value_colour: str = INK, extra_html: str = "") -> None:
    """A KPI tile using the approved .glass/.br-label/.br-big classes."""
    st.markdown(
        f'<div class="glass br-kpi"><div class="br-label">{html.escape(label)}</div>'
        f'<div class="br-kpi-value"><span class="br-big" style="color:{value_colour}">{html.escape(value)}</span>{extra_html}</div>'
        f'<div class="br-kpi-detail">{detail_html}</div></div>',
        unsafe_allow_html=True,
    )


def delta_html(change: float | None, unit: str, good_when_up: bool | None, period_days: int) -> str:
    """'▲ 4.2 pts vs previous 14 days', coloured by whether up is good (None = neutral)."""
    if change is None:
        return '<span class="br-delta br-delta-none">— no earlier data</span>'
    if change == 0:
        return f'<span class="br-delta br-delta-flat">no change vs previous {period_days} days</span>'
    up = change > 0
    tone = "flat" if good_when_up is None else ("good" if up == good_when_up else "bad")
    arrow = "▲" if up else "▼"
    return (f'<span class="br-delta br-delta-{tone}">{arrow} {abs(change):g}{unit}</span>'
            f'<span class="br-delta-rest"> vs previous {period_days} days</span>')


def empty_state(title: str, message: str, height: int = 260) -> None:
    st.markdown(
        f'<div class="br-empty" style="min-height:{height}px"><div class="br-empty-icon">{RADAR_ICON_SVG}</div>'
        f'<div class="br-empty-title">{html.escape(title)}</div>'
        f'<div class="br-empty-text">{html.escape(message)}</div></div>',
        unsafe_allow_html=True,
    )


RADAR_ICON_SVG = (
    '<svg viewBox="0 0 32 32" width="26" height="26" fill="none" stroke="#9E4A5A" stroke-width="1.8" '
    'stroke-linecap="round"><circle cx="16" cy="16" r="13" opacity=".35"/><circle cx="16" cy="16" r="8.5" '
    'opacity=".6"/><circle cx="16" cy="16" r="3.5"/><path d="M16 16 L26 8"/><circle cx="23.5" cy="11" r="1.6" '
    'fill="#D9536A" stroke="none"/></svg>'
)


def wordmark_html() -> str:
    return f'<div class="br-wordmark">{RADAR_ICON_SVG}<span>BRAND RADAR</span></div>'


def status_html(text: str, warning: str | None, running: bool, has_runs: bool) -> str:
    """The small collector status line under the wordmark."""
    tone = "warn" if warning else ("idle" if not has_runs else "ok")
    extra = ' <span class="br-status-running">· collecting…</span>' if running else ""
    warn = f'<span class="br-status-warn">⚠ {html.escape(warning)}</span>' if warning else ""
    return (f'<div class="br-status br-status-{tone}"><span class="br-status-dot"></span>'
            f'<span>{html.escape(text)}</span>{extra}{warn}</div>')


# ---------------------------------------------------------------- word clouds


@st.cache_data(ttl=60, show_spinner=False)
def wordcloud_png(texts: tuple[str, ...], stop_words: frozenset[str], palette: tuple[str, ...],
                  width: int = 900, height: int = 420) -> bytes:
    """Transparent PNG word cloud in the given palette (cached: rendering is the slow part)."""
    from wordcloud import WordCloud

    rng = random.Random(3)

    def colour(*_: object, **__: object) -> str:
        return rng.choice(palette)

    cloud = WordCloud(
        width=width, height=height, mode="RGBA", background_color=None, font_path=str(FONT_FILE),
        stopwords=set(stop_words), max_words=60, min_word_length=3, prefer_horizontal=0.92,
        collocations=True, color_func=colour, random_state=7, margin=6, relative_scaling=0.45,
    ).generate(" ".join(texts))
    buffer = io.BytesIO()
    cloud.to_image().save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()
