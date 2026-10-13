"""Plotly figures shared by the dashboard (app.py) and the weekly PDF report (reports/weekly.py).

Pure functions: data in, go.Figure out, no Streamlit calls. Each caller decides on height and
background (show_chart for the dashboard, a static image export for the PDF) and on empty states.
"""

from __future__ import annotations

import html
import math
from collections.abc import Sequence

import pandas as pd
import plotly.graph_objects as go

from nlp.priority import FIX_NOW, IGNORE, MONITOR, QUICK_WINS
from ui import insights
from ui.components import BERRY, INK

QUADRANT_COLOURS = {FIX_NOW: "#D9536A", QUICK_WINS: BERRY, MONITOR: "#E79AA6", IGNORE: "#B7AEB0"}
# very soft tints of the theme's rose/blush; Fix now slightly stronger
QUADRANT_TINTS = {FIX_NOW: "rgba(217,83,106,0.11)", QUICK_WINS: "rgba(231,154,166,0.08)",
                  MONITOR: "rgba(243,186,193,0.08)", IGNORE: "rgba(240,220,219,0.10)"}
MENTION_TICKS = (0, 1, 2, 5, 10, 20, 50, 100)

Named = tuple[int, str]  # (brand id, brand name)


def own_trend(trend: pd.DataFrame, brand_id: int) -> pd.DataFrame:
    """The own brand's scored days from data.daily_trend (a trend needs at least two)."""
    return trend[(trend["brand_id"] == brand_id) & trend["score"].notna()]


def trend_figure(trend: pd.DataFrame, brand: Named, comps: Sequence[Named], colours: dict[str, str]) -> go.Figure:
    """Daily net sentiment: competitors as thin lines, the own brand filled in berry with its last value."""
    brand_id, brand_name = brand
    own = own_trend(trend, brand_id)
    fig = go.Figure()
    for comp_id, comp_name in comps:
        line = trend[(trend["brand_id"] == comp_id) & trend["score"].notna()]
        if line["day"].nunique() < 2:
            continue
        fig.add_trace(go.Scatter(
            x=line["day"], y=line["score"], name=comp_name, mode="lines",
            line=dict(color=colours[comp_name], width=1.6, shape="spline", smoothing=0.6),
            customdata=line["mentions"],
            hovertemplate="%{x|%d %b}: %{y:+.0f}<br>%{customdata} mention(s) that day<extra>" + comp_name + "</extra>",
        ))
    fig.add_trace(go.Scatter(
        x=own["day"], y=own["score"], name=brand_name, mode="lines",
        line=dict(color=BERRY, width=3.2, shape="spline", smoothing=0.6),
        fill="tozeroy", fillcolor="rgba(158,74,90,0.10)", customdata=own["mentions"],
        hovertemplate="%{x|%d %b}: %{y:+.0f}<br>%{customdata} mention(s) that day<extra>" + brand_name + "</extra>",
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
    return fig


def matrix_figure(matrix: pd.DataFrame, texts: pd.DataFrame, volume_threshold: float, pain_threshold: float) -> go.Figure:
    """Priority matrix: complaint volume (log) vs pain, quadrant tints and names, bubbles sized by mentions."""
    x_split = math.log1p(volume_threshold)
    x_max = max(matrix["x"].max() + 0.45, x_split + 0.9)
    y_max = max(matrix["y"].max() * 1.18, pain_threshold * 2)
    fig = go.Figure()
    # quadrant names sit at the top of each quadrant: the bottom corners are where low-pain topics cluster
    corners = {FIX_NOW: (x_split, x_max, pain_threshold, y_max, "right"),
               QUICK_WINS: (-0.12, x_split, pain_threshold, y_max, "left"),
               MONITOR: (x_split, x_max, -0.015, pain_threshold, "right"),
               IGNORE: (-0.12, x_split, -0.015, pain_threshold, "left")}
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
    return fig


def radar_figure(table: insights.FeatureTable, brand: Named, comps: Sequence[Named], colours: dict[str, str]) -> go.Figure:
    """Net sentiment per feature for the brand (filled) and its competitors, on a −100…+100 radar."""
    theta = [f.title() if f.islower() else f for f in table.features]
    fig = go.Figure()
    ring = [0] * (len(theta) + 1)
    fig.add_trace(go.Scatterpolar(r=ring, theta=theta + theta[:1], mode="lines", hoverinfo="skip",
                                  line=dict(color="#E79AA6", width=1, dash="dot"), showlegend=False))
    for bid, name in [*comps, brand]:
        if bid not in table.net.index:
            continue
        values = table.net.loc[bid].tolist()
        counts = table.counts.loc[bid].tolist()
        if all(pd.isna(v) for v in values):
            continue
        own = bid == brand[0]
        hover = [f"{t}: {v:+.0f} ({n} mentions)" if not pd.isna(v) else f"{t}: too few mentions ({n})"
                 for t, v, n in zip(theta, values, counts)]
        r = [None if pd.isna(v) else v for v in values]
        fig.add_trace(go.Scatterpolar(
            r=r + r[:1], theta=theta + theta[:1], name=name, connectgaps=True,
            mode="lines+markers", hovertext=hover + hover[:1], hovertemplate="%{hovertext}<extra>" + name + "</extra>",
            line=dict(color=colours[name], width=3 if own else 2),
            marker=dict(size=7 if own else 5, color=colours[name]),
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
    return fig


def category_figure(this_week: pd.Series, last_week: pd.Series) -> go.Figure:
    """Complaints per profile category: this week (berry) against last week (blush), largest at the top.
    Both series are indexed by category label and already limited to the profile's complaint labels."""
    order = list(this_week.add(last_week, fill_value=0).sort_values().index)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=order, x=[int(last_week.get(c, 0)) for c in order], orientation="h", name="Last week",
                         marker=dict(color="#F3BAC1"), hovertemplate="%{y}: %{x}<extra>last week</extra>"))
    fig.add_trace(go.Bar(y=order, x=[int(this_week.get(c, 0)) for c in order], orientation="h", name="This week",
                         marker=dict(color=BERRY), text=[int(this_week.get(c, 0)) or "" for c in order],
                         textposition="outside", textfont=dict(family="Montserrat, sans-serif", size=11, color=BERRY),
                         cliponaxis=False, hovertemplate="%{y}: %{x}<extra>this week</extra>"))
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08, legend=dict(y=1.02, traceorder="reversed"),
                      xaxis=dict(showgrid=True, zeroline=False, rangemode="tozero"), yaxis=dict(showgrid=False),
                      margin=dict(l=8, r=28, t=30, b=8))
    return fig
