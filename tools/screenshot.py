"""Screenshot the running dashboard at desktop and phone width (headless Chromium via Playwright).

    streamlit run app.py --server.port 8599 &
    python tools/screenshot.py --url http://localhost:8599 --out docs/screenshots --name step5
    python tools/screenshot.py --group Stationery --name stationery   # another brand group

Streamlit scrolls inside its own container and the theme sizes its blobs in vmax, so a "full page"
capture with a giant viewport would distort the background. Instead each page is captured one
viewport at a time while scrolling the real container, then stitched: what a person actually sees.
--motion also hides the content and diffs two frames 3 s apart to prove the background moves.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

from PIL import Image, ImageChops, ImageStat

VIEWPORTS = {"desktop": (1440, 900, 1), "mobile": (390, 844, 2)}

FIND_SCROLLER = """
() => {
  const candidates = [...document.querySelectorAll('[data-testid="stMain"], [data-testid="stAppViewContainer"], section.main, .main')];
  candidates.push(document.scrollingElement);
  const el = candidates.find(e => e && e.scrollHeight > e.clientHeight + 4) || document.scrollingElement;
  el.setAttribute('data-shot-scroller', '1');
  return [el.scrollHeight, el.clientHeight];
}
"""


def wait_until_rendered(page, timeout_ms: int = 60000, charts: bool = True) -> None:
    page.wait_for_selector(".st-key-glass-header", timeout=timeout_ms)
    if charts:  # a group with little data may show only empty states
        page.wait_for_selector(".js-plotly-plot", timeout=timeout_ms)
    # Streamlit shows a "running" state while scripts execute; wait for it to settle, then fonts.
    page.wait_for_function("() => !document.querySelector('[data-testid=\"stStatusWidget\"] *')", timeout=timeout_ms)
    page.evaluate("() => document.fonts.ready")
    page.wait_for_timeout(1500)


def stitched_capture(page, width: int, height: int, scale: int) -> Image.Image:
    total, view = page.evaluate(FIND_SCROLLER)
    if total <= 1600 and width >= 1000:
        # desktop pages that fit comfortably: one tall viewport, so the fixed background has no seams
        page.set_viewport_size({"width": width, "height": total})
        page.wait_for_timeout(800)
        return Image.open(io.BytesIO(page.screenshot()))
    slices = []
    offset = 0
    while True:
        page.evaluate(f"() => document.querySelector('[data-shot-scroller]').scrollTo(0, {offset})")
        page.wait_for_timeout(350)
        actual = page.evaluate("() => document.querySelector('[data-shot-scroller]').scrollTop")
        slices.append((actual, Image.open(io.BytesIO(page.screenshot()))))
        if offset + view >= total:
            break
        offset += view
    canvas = Image.new("RGB", (width * scale, total * scale), "white")
    for top, img in slices:
        canvas.paste(img, (0, int(top * scale)))
    return canvas


def motion_score(page) -> float:
    """Percentage of background pixels that changed over 3 s with the content hidden."""
    # hide the cards only: the background lives inside .block-container too
    page.add_style_tag(content='[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [class*="st-key-glass-"]), '
                               '.st-key-kpis, .br-footnote { visibility: hidden !important; }')
    page.wait_for_timeout(300)
    first = Image.open(io.BytesIO(page.screenshot())).convert("L")
    page.wait_for_timeout(3000)
    second = Image.open(io.BytesIO(page.screenshot())).convert("L")
    diff = ImageChops.difference(first, second).point(lambda v: 255 if v > 3 else 0)
    return ImageStat.Stat(diff).mean[0] / 255 * 100


def choose_group(page, group: str) -> None:
    """Pick a brand group in the dashboard's Group selectbox and wait for the rerun."""
    page.locator('.st-key-glass-header [data-testid="stSelectbox"]').filter(has_text="Group").locator('[data-baseweb="select"]').click()
    page.get_by_role("option", name=group, exact=True).click()
    page.wait_for_timeout(1500)
    wait_until_rendered(page, charts=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--out", type=Path, default=Path("docs/screenshots"))
    parser.add_argument("--name", default="dashboard")
    parser.add_argument("--only", choices=sorted(VIEWPORTS), help="capture one viewport only")
    parser.add_argument("--group", help="brand group to show (default: the dashboard's first group)")
    parser.add_argument("--reduced-motion", action="store_true",
                        help="emulate prefers-reduced-motion: the background holds still, so captures can be diffed")
    parser.add_argument("--motion", action="store_true", help="also measure background motion (desktop)")
    parser.add_argument(
        "--browser", default="chrome",
        help="'chrome' = the installed Google Chrome (headless, temporary profile; no download), "
             "'chromium' = Playwright's bundled build (python -m playwright install chromium)",
    )
    args = parser.parse_args(argv)

    from playwright.sync_api import sync_playwright

    args.out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome") if args.browser == "chrome" else p.chromium.launch()
        for label, (width, height, scale) in VIEWPORTS.items():
            if args.only and label != args.only:
                continue
            page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=scale,
                                    reduced_motion="reduce" if args.reduced_motion else "no-preference")
            started = time.perf_counter()
            page.goto(args.url, wait_until="networkidle")
            wait_until_rendered(page)
            if args.group:
                choose_group(page, args.group)
            image = stitched_capture(page, width, height, scale)
            path = args.out / f"{args.name}-{label}.png"
            image.save(path, optimize=True)
            print(f"{label:<8} {width}px  -> {path}  ({image.width}x{image.height}, loaded in {time.perf_counter() - started:.1f}s)")
            if args.motion and label == "desktop":
                page.evaluate("() => document.querySelector('[data-shot-scroller]').scrollTo(0, 0)")
                print(f"motion   {motion_score(page):.1f}% of background pixels changed in 3 s")
            page.close()
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
