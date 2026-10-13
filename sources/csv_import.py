"""CSV import: reviews exported from shops, survey tools or social listening (no scraping needed).

Only a text and a date column are required. Common header names are recognised (see ALIASES:
"Review", "Review Body", "Posted On", "Stars", "Reviewer", …); anything else can be mapped by hand:

    python fetch.py --csv reviews.csv --brand DOMS --map "text=What customers said,date=Day"

- rating: 1–5. Accepts "4", "4.5" (rounded half up), "4/5", "4 out of 5", "8/10" (scaled to 5) and
  "★★★★☆". Blank is fine.
- date: ISO 8601 (2026-10-01, 2026-10-01T14:30:00Z), day-first numbers (01/10/2026, 01-10-2026,
  01.10.2026; `--date-order mdy` for US exports), month names (1 Oct 2026, Oct 1, 2026, October 1 2026),
  optionally with a time, or a Unix timestamp. Naive dates are taken as UTC.
- source: the row's own value (e.g. "amazon", "nykaa" from an export you were given), else `--csv-source`,
  else "csv". The run is logged under source "csv".

The external_id is a hash of the row, so importing the same file twice adds nothing.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
from datetime import datetime, timezone
from pathlib import Path

from sources.base import BaseSource, BrandSpec, RawMention, to_utc_naive

FIELDS = ("text", "date", "rating", "author", "source")
REQUIRED = ("text", "date")
# Header names (lower-cased, spaces/dashes as underscores) recognised for each field, in priority order.
ALIASES: dict[str, tuple[str, ...]] = {
    "text": ("text", "review", "review_text", "review_body", "review_content", "body", "content", "comment",
             "comments", "message", "feedback", "description"),
    "date": ("date", "created_at", "review_date", "date_posted", "posted_on", "posted", "timestamp", "time",
             "submitted_at", "date_created"),
    "rating": ("rating", "stars", "star_rating", "star", "score", "review_rating", "rating_out_of_5"),
    "author": ("author", "reviewer", "reviewer_name", "user", "username", "user_name", "name", "customer",
               "customer_name", "profile_name"),
    "source": ("source", "platform", "site", "channel", "marketplace", "store"),
}
DATE_ORDERS = ("dmy", "mdy")
TIME_PARTS = ("", " %H:%M", " %H:%M:%S", " %I:%M %p", " %I:%M:%S %p")
NAMED_MONTH = ("%d %b %Y", "%d %B %Y", "%b %d %Y", "%B %d %Y", "%d-%b-%Y", "%d %b, %Y", "%b %d, %Y", "%B %d, %Y")


def normalise_header(name: str | None) -> str:
    return re.sub(r"[\s\-]+", "_", (name or "").strip().lower())


def parse_mapping(spec: str | None) -> dict[str, str]:
    """'text=Review Body,date=Posted On' -> {"text": "review_body", "date": "posted_on"}."""
    mapping: dict[str, str] = {}
    for part in (spec or "").split(","):
        if not part.strip():
            continue
        field, sep, column = part.partition("=")
        field = field.strip().lower()
        if not sep or field not in FIELDS or not column.strip():
            raise ValueError(f"--map: {part.strip()!r} should look like text=Column Name (fields: {', '.join(FIELDS)})")
        mapping[field] = normalise_header(column)
    return mapping


def resolve_columns(headers: list[str], mapping: dict[str, str] | None = None) -> dict[str, str]:
    """field -> normalised header present in the file. Raises if text or date can't be found."""
    present = [normalise_header(h) for h in headers]
    resolved: dict[str, str] = {}
    for field in FIELDS:
        wanted = (mapping or {}).get(field)
        if wanted:
            if wanted not in present:
                raise ValueError(f"--map: column {wanted!r} for {field} is not in the CSV (columns: {', '.join(present)})")
            resolved[field] = wanted
            continue
        found = next((a for a in ALIASES[field] if a in present and a not in resolved.values()), None)
        if found:
            resolved[field] = found
    missing = [f for f in REQUIRED if f not in resolved]
    if missing:
        raise ValueError(
            f"CSV is missing column(s): {', '.join(missing)}. Columns found: {', '.join(present)}. "
            'Map them with --map "text=Your Text Column,date=Your Date Column".'
        )
    return resolved


def _numeric_formats(order: str) -> list[str]:
    day_month = ("%d", "%m") if order == "dmy" else ("%m", "%d")
    out = []
    for sep in ("/", "-", "."):
        out.append(sep.join((*day_month, "%Y")))
        out.append(sep.join((*day_month, "%y")))
        out.append(sep.join(("%Y", "%m", "%d")))
    return out


def parse_date(value: str, order: str = "dmy") -> datetime:
    value = " ".join(value.strip().split())
    if not value:
        raise ValueError("empty date")
    if re.fullmatch(r"\d{9,13}", value):  # Unix seconds or milliseconds
        seconds = int(value) / (1000 if len(value) > 11 else 1)
        return datetime.fromtimestamp(seconds, tz=timezone.utc).replace(tzinfo=None)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        candidates = [d + t for d in (*_numeric_formats(order), *NAMED_MONTH) for t in TIME_PARTS]
        for fmt in candidates:
            try:
                parsed = datetime.strptime(value, fmt)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"unrecognised date {value!r} (e.g. 2026-10-01, 01/10/2026 or 1 Oct 2026)") from None
    return to_utc_naive(parsed) if parsed.tzinfo else parsed


def parse_rating(value: str) -> int | None:
    """1–5 from '4', '4.5', '4/5', '4 out of 5', '8/10' or '★★★★☆'; None when blank."""
    value = (value or "").strip()
    if not value:
        return None
    if set(value) <= {"★", "☆", " "}:
        stars = value.count("★")
        if not 1 <= stars <= 5:
            raise ValueError(f"rating {value!r} is not between 1 and 5 stars")
        return stars
    match = re.fullmatch(r"(\d+(?:[.,]\d+)?)\s*(?:(?:/|out of|of)\s*(\d+))?\s*(?:stars?)?", value, re.IGNORECASE)
    if not match:
        raise ValueError(f"rating {value!r} is not a number")
    number = float(match.group(1).replace(",", "."))
    scale = int(match.group(2) or 5)
    if scale <= 0:
        raise ValueError(f"rating {value!r} has no scale")
    rating = int(number * 5 / scale + 0.5)  # round half up: 4.5 -> 5, 7/10 -> 4
    if not 1 <= rating <= 5 or number > scale:
        raise ValueError(f"rating {value!r} is not between 1 and 5")
    return rating


def row_id(source: str, author: str, text: str, created_at: datetime) -> str:
    key = "\x1f".join((source, author, text, created_at.isoformat()))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


class CsvSource(BaseSource):
    name = "csv"

    def __init__(
        self,
        path: str | Path | None = None,
        content: str | None = None,
        mapping: dict[str, str] | None = None,
        date_order: str = "dmy",
        default_source: str = "csv",
    ) -> None:
        super().__init__()
        if path is None and content is None:
            raise ValueError("CsvSource needs a path or content")
        if date_order not in DATE_ORDERS:
            raise ValueError(f"date order must be one of {', '.join(DATE_ORDERS)}")
        self.path = Path(path) if path else None
        self.content = content
        self.mapping = mapping or {}
        self.date_order = date_order
        self.default_source = default_source.lower()[:32] or "csv"

    def _read(self) -> str:
        if self.content is not None:
            return self.content
        assert self.path is not None
        return self.path.read_text(encoding="utf-8-sig")  # tolerate Excel's BOM

    def fetch(self, brand: BrandSpec, since: datetime | None = None) -> list[RawMention]:
        """Import every valid row; `since` is ignored because a CSV is an explicit import."""
        self.reset()
        text_in = self._read()
        header = text_in.split("\n", 1)[0]
        delimiter = max((",", ";", "\t"), key=header.count)  # Excel in many locales saves with ";"
        reader = csv.DictReader(io.StringIO(text_in), delimiter=delimiter)
        columns = resolve_columns(list(reader.fieldnames or []), self.mapping)

        found: dict[str, RawMention] = {}
        for line_no, raw in enumerate(reader, start=2):
            row = {normalise_header(k): (v or "").strip() for k, v in raw.items() if k is not None}

            def get(field: str) -> str:
                return row.get(columns[field], "") if field in columns else ""

            try:
                text = get("text")
                if not text:
                    raise ValueError("empty text")
                source = (get("source") or self.default_source).lower()[:32]
                created = parse_date(get("date"), self.date_order)
                rating = parse_rating(get("rating"))
            except ValueError as exc:
                self.warn(f"line {line_no}: {exc}")
                continue
            author = get("author") or None
            external_id = row_id(source, author or "", text, created)
            found.setdefault(
                external_id,
                RawMention(source=source, external_id=external_id, text=text, created_at=created, author=author,
                           rating=rating),
            )
        return list(found.values())
