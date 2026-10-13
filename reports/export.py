"""CSV export of mentions with their analysis (the dashboard's "Download CSV")."""

from __future__ import annotations

import re

import pandas as pd

FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


def neutralise(value: object) -> object:
    """Spreadsheet apps run cells that start with = + - @ as formulas ("CSV injection"); mention texts are
    written by strangers, so such cells get a leading apostrophe and are shown as plain text."""
    if isinstance(value, str) and value.startswith(FORMULA_START):
        return "'" + value
    return value


def mentions_csv(df: pd.DataFrame) -> bytes:
    """UTF-8 with a BOM (Excel then reads emoji and ₹ correctly); dates as ISO, scores rounded."""
    out = df.copy()
    if "created_at" in out:
        out["created_at"] = pd.to_datetime(out["created_at"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    if "sentiment_score" in out:
        out["sentiment_score"] = pd.to_numeric(out["sentiment_score"], errors="coerce").round(4)
    if "rating" in out:
        out["rating"] = pd.to_numeric(out["rating"], errors="coerce").astype("Int64")
    for column in out.columns:
        if out[column].dtype == object:
            out[column] = out[column].map(neutralise)
    return out.to_csv(index=False).encode("utf-8-sig")


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "brand"


def csv_name(brand: str, start: object, end: object) -> str:
    return f"brandradar-{slug(brand)}-{start}-to-{end}.csv"
