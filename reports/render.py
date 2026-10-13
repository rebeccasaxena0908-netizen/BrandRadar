"""Plotly figure -> PNG for the PDF report.

KaleidoRenderer uses kaleido 1.x, which drives the installed Google Chrome (no bundled browser). One
Chrome is kept open for a whole report (about 0.3 s per chart instead of 1.5 s), and MathJax is off,
so nothing is fetched from a CDN.

Fonts: Plotly's PNG export draws its SVG through an <img>, and an SVG used as an image cannot see the
page's fonts, so text fell back to Helvetica. The renderer therefore exports SVG, embeds Figtree and
Montserrat into it as data-URI @font-face rules, and rasterises that SVG in the same Chrome (as a
full-size layout image of an empty figure). The page itself also gets the fonts, so Plotly measures
text with the right widths.
"""

from __future__ import annotations

import base64
import io
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Protocol

import numpy as np
import plotly.graph_objects as go

from ui.components import UI_DIR

FONTS = (  # (CSS family, weight, file in ui/fonts)
    ("Figtree", 400, "Figtree-Regular.ttf"),
    ("Figtree", 600, "Figtree-SemiBold.ttf"),
    ("Montserrat", 600, "Montserrat-SemiBold.ttf"),
    ("Montserrat", 700, "Montserrat-Bold.ttf"),
)
SCALE = 2  # pixel density of the PNGs (crisp when printed)


class Renderer(Protocol):
    def png(self, fig: go.Figure, width: int, height: int) -> bytes: ...


@lru_cache(maxsize=1)
def font_face_css() -> str:
    rules = []
    for family, weight, file in FONTS:
        data = base64.b64encode((UI_DIR / "fonts" / file).read_bytes()).decode()
        rules.append(f"@font-face{{font-family:'{family}';font-weight:{weight};"
                     f"src:url(data:font/ttf;base64,{data}) format('truetype');}}")
    return "".join(rules)


def embed_fonts(svg: str) -> str:
    """Put the @font-face rules inside the SVG so they apply when it is drawn as an image."""
    style = f"<style>{font_face_css()}</style>"
    if "<defs>" in svg:
        return svg.replace("<defs>", f"<defs>{style}", 1)
    head_end = svg.index(">", svg.index("<svg")) + 1
    return svg[:head_end] + f"<defs>{style}</defs>" + svg[head_end:]


def _holder(svg: str, width: int, height: int) -> go.Figure:
    """An empty, transparent figure whose only content is the SVG, stretched to the full size."""
    fig = go.Figure()
    fig.add_layout_image(source="data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode(),
                         xref="paper", yref="paper", x=0, y=1, sizex=1, sizey=1, xanchor="left", yanchor="top",
                         sizing="stretch", layer="above")
    fig.update_layout(width=width, height=height, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def prepare(fig: go.Figure, width: int, height: int) -> go.Figure:
    """The dashboard's template and transparent backgrounds (what show_chart does on screen). Datetime
    arrays become ISO strings: Plotly 6's static export otherwise sends them as integer nanoseconds."""
    fig = go.Figure(fig)
    for trace in fig.data:
        for attr in ("x", "y"):
            values = getattr(trace, attr, None)
            if isinstance(values, np.ndarray) and np.issubdtype(values.dtype, np.datetime64):
                setattr(trace, attr, np.datetime_as_string(values, unit="s").tolist())
    fig.update_layout(template="blush", width=width, height=height,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


class KaleidoRenderer:
    """Use as a context manager: `with KaleidoRenderer() as r: r.png(fig, 900, 360)`."""

    def __enter__(self) -> KaleidoRenderer:
        import kaleido

        class FontPage(kaleido.PageGenerator):  # the page gets the fonts too, for text measurement
            header = kaleido.PageGenerator.header.replace(
                "<title>Kaleido-fier</title>", f"<style>{font_face_css()}</style><title>Kaleido-fier</title>")

        kaleido.start_sync_server(page_generator=FontPage(mathjax=False), silence_warnings=True)
        self.png(go.Figure(), 50, 50)  # warm-up: lets Chrome load the fonts before the first real chart
        return self

    def __exit__(self, *exc: object) -> None:
        import kaleido

        kaleido.stop_sync_server(silence_warnings=True)

    def png(self, fig: go.Figure, width: int, height: int) -> bytes:
        fig = prepare(fig, width, height)
        svg = fig.to_image(format="svg", width=width, height=height).decode()
        return _holder(embed_fonts(svg), width, height).to_image(format="png", width=width, height=height, scale=SCALE)


class StubRenderer:
    """For tests: a plain blush PNG of the right size, no browser. Records what it was asked to draw."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []

    def png(self, fig: go.Figure, width: int, height: int) -> bytes:
        from PIL import Image

        self.calls.append((fig.data[0].type if fig.data else "empty", width, height))
        buffer = io.BytesIO()
        Image.new("RGB", (width, height), "#F3BAC1").save(buffer, format="PNG")
        return buffer.getvalue()


@contextmanager
def renderer(stub: bool = False) -> Iterator[Renderer]:
    if stub:
        yield StubRenderer()
        return
    with KaleidoRenderer() as r:
        yield r
