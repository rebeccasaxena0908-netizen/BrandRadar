"""CSV import: rows with columns source, author, text, date, rating (rating may be blank).

Each row keeps its own `source` value (e.g. "twitter"); the run is logged under source "csv".
The external_id is a hash of the row, so importing the same file twice adds nothing.
"""

from __future__ import annotations

import csv
import hashlib
import io
from datetime import datetime
from pathlib import Path

from sources.base import BaseSource, BrandSpec, RawMention, to_utc_naive

REQUIRED_COLUMNS = ("source", "author", "text", "date", "rating")
DATE_FORMATS = ("%d/%m/%Y", "%d/%m/%Y %H:%M", "%Y/%m/%d", "%Y/%m/%d %H:%M")


def parse_date(value: str) -> datetime:
    value = value.strip()
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        for fmt in DATE_FORMATS:
            try:
                parsed = datetime.strptime(value, fmt)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"unrecognised date {value!r} (use ISO 8601, e.g. 2026-10-01 or 2026-10-01T14:30:00Z)")
    # Naive CSV dates are taken as UTC.
    return to_utc_naive(parsed) if parsed.tzinfo else parsed


def parse_rating(value: str) -> int | None:
    value = (value or "").strip()
    if not value:
        return None
    rating = int(float(value))
    if not 1 <= rating <= 5:
        raise ValueError(f"rating {rating} is not between 1 and 5")
    return rating


def row_id(source: str, author: str, text: str, created_at: datetime) -> str:
    key = "\x1f".join((source, author, text, created_at.isoformat()))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


class CsvSource(BaseSource):
    name = "csv"

    def __init__(self, path: str | Path | None = None, content: str | None = None) -> None:
        super().__init__()
        if path is None and content is None:
            raise ValueError("CsvSource needs a path or content")
        self.path = Path(path) if path else None
        self.content = content

    def _read(self) -> str:
        if self.content is not None:
            return self.content
        assert self.path is not None
        return self.path.read_text(encoding="utf-8-sig")  # tolerate Excel's BOM

    def fetch(self, brand: BrandSpec, since: datetime | None = None) -> list[RawMention]:
        """Import every valid row; `since` is ignored because a CSV is an explicit import."""
        self.reset()
        reader = csv.DictReader(io.StringIO(self._read()))
        columns = {c.strip().lower() for c in (reader.fieldnames or [])}
        missing = [c for c in REQUIRED_COLUMNS if c not in columns]
        if missing:
            raise ValueError(f"CSV is missing column(s): {', '.join(missing)}")

        found: dict[str, RawMention] = {}
        for line_no, raw in enumerate(reader, start=2):
            row = {(k or "").strip().lower(): (v or "").strip() for k, v in raw.items()}
            try:
                text = row["text"]
                if not text:
                    raise ValueError("empty text")
                source = (row["source"] or "csv").lower()[:32]
                created = parse_date(row["date"])
                rating = parse_rating(row["rating"])
            except ValueError as exc:
                self.warn(f"line {line_no}: {exc}")
                continue
            author = row["author"] or None
            external_id = row_id(source, author or "", text, created)
            found.setdefault(
                external_id,
                RawMention(
                    source=source,
                    external_id=external_id,
                    text=text,
                    created_at=created,
                    author=author,
                    rating=rating,
                ),
            )
        return list(found.values())
