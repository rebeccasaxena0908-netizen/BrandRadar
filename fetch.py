"""Fetch runner: runs every enabled source for every brand, stores new mentions, logs fetch_runs.

Examples:
    python fetch.py                                  # all sources, all brands, last FETCH_SINCE_DAYS days
    python fetch.py --brand Notion --source hackernews --since-days 7
    python fetch.py --csv exports/tweets.csv --brand Notion
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import insert, select
from sqlalchemy.orm import Session, sessionmaker

from config import settings
from db.models import Brand, FetchRun, Mention
from sources.appstore import AppStoreSource
from sources.base import BrandSpec, RawMention, Source, clip, utc_now
from sources.csv_import import CsvSource
from sources.hackernews import HackerNewsSource
from sources.playstore import PlayStoreSource
from sources.reddit import RedditSource

log = logging.getLogger("fetch")

SOURCE_FACTORIES: dict[str, Callable[[], Source]] = {
    "hackernews": HackerNewsSource,
    "appstore": AppStoreSource,
    "playstore": PlayStoreSource,
    "reddit": RedditSource,
}
LOOKUP_CHUNK = 500
MAX_ERROR_LENGTH = 10_000


@dataclass
class RunResult:
    source: str
    brand: str
    fetched: int = 0
    new_rows: int = 0
    error: str | None = None
    warnings: int = 0
    skipped: bool = False


# ---------------------------------------------------------------- storage


def insert_new_mentions(session: Session, brand_id: int, mentions: Iterable[RawMention]) -> int:
    """Insert mentions not already stored for this brand; return how many were inserted."""
    unique: dict[tuple[str, str], RawMention] = {}
    for m in mentions:
        source, external_id = m.source[:32], m.external_id[:255]
        if external_id and m.text.strip():
            unique.setdefault((source, external_id), m)
    if not unique:
        return 0

    existing: set[tuple[str, str]] = set()
    by_source: dict[str, list[str]] = {}
    for source, external_id in unique:
        by_source.setdefault(source, []).append(external_id)
    for source, ids in by_source.items():
        for start in range(0, len(ids), LOOKUP_CHUNK):
            chunk = ids[start : start + LOOKUP_CHUNK]
            rows = session.execute(
                select(Mention.external_id).where(
                    Mention.brand_id == brand_id,
                    Mention.source == source,
                    Mention.external_id.in_(chunk),
                )
            )
            existing.update((source, r[0]) for r in rows)

    new_rows = [
        {
            "brand_id": brand_id,
            "source": source,
            "external_id": external_id,
            "author": clip(m.author, 255),
            "text": m.text.strip(),
            "url": clip(m.url, 1024),
            "rating": m.rating,
            "created_at": m.created_at,
            "fetched_at": utc_now(),
        }
        for (source, external_id), m in unique.items()
        if (source, external_id) not in existing
    ]
    if new_rows:
        # IGNORE guards against a concurrent run inserting the same rows in between.
        session.execute(insert(Mention).prefix_with("IGNORE", dialect="mysql"), new_rows)
    return len(new_rows)


def load_brands(factory: sessionmaker[Session], names: Sequence[str] | None = None) -> list[BrandSpec]:
    with factory() as session:
        query = select(Brand).order_by(Brand.is_own.desc(), Brand.name)
        brands = [BrandSpec.from_model(b) for b in session.scalars(query)]
    if names:
        wanted = {n.lower() for n in names}
        unknown = wanted - {b.name.lower() for b in brands}
        if unknown:
            raise SystemExit(f"Unknown brand(s): {', '.join(sorted(unknown))}. Run manage_brands.py first?")
        brands = [b for b in brands if b.name.lower() in wanted]
    return brands


# ---------------------------------------------------------------- running


def run_one(
    factory: sessionmaker[Session], source: Source, brand: BrandSpec, since: datetime
) -> RunResult:
    """Fetch one source for one brand. Never raises: errors end up in fetch_runs.error."""
    result = RunResult(source=source.name, brand=brand.name)
    with factory() as session:
        run = FetchRun(source=source.name, brand_id=brand.id, started_at=utc_now(), new_rows=0)
        session.add(run)
        session.commit()
        run_id = run.id

    try:
        mentions = source.fetch(brand, since)
        result.fetched = len(mentions)
        with factory() as session:
            result.new_rows = insert_new_mentions(session, brand.id, mentions)
            session.commit()
    except Exception as exc:  # noqa: BLE001 — one failing source must not stop the others
        log.exception("%s / %s failed", source.name, brand.name)
        result.error = f"{type(exc).__name__}: {exc}"

    warnings = list(getattr(source, "warnings", []))
    result.warnings = len(warnings)
    messages = ([f"error: {result.error}"] if result.error else []) + [f"warning: {w}" for w in warnings]

    with factory() as session:
        run = session.get(FetchRun, run_id)
        assert run is not None
        run.finished_at = utc_now()
        run.new_rows = result.new_rows
        run.error = "\n".join(messages)[:MAX_ERROR_LENGTH] or None
        session.commit()

    log.info(
        "%-10s %-10s fetched=%d new=%d%s",
        source.name, brand.name, result.fetched, result.new_rows,
        f" ERROR {result.error}" if result.error else "",
    )
    return result


def run_fetch(
    factory: sessionmaker[Session],
    brands: Sequence[BrandSpec],
    sources: Sequence[Source],
    since: datetime,
) -> list[RunResult]:
    results: list[RunResult] = []
    for source in sources:
        if not source.enabled:
            log.info("%s: skipped (not configured)", source.name)
            results.extend(RunResult(source.name, b.name, skipped=True) for b in brands)
            continue
        for brand in brands:
            results.append(run_one(factory, source, brand, since))
    return results


# ---------------------------------------------------------------- CLI


def print_summary(results: Sequence[RunResult]) -> None:
    sources = list(dict.fromkeys(r.source for r in results))
    brands = list(dict.fromkeys(r.brand for r in results))
    cell = {(r.brand, r.source): r for r in results}

    def fmt(r: RunResult | None) -> str:
        if r is None or r.skipped:
            return "skipped"
        if r.error:
            return "ERROR"
        return f"{r.new_rows}" + ("*" if r.warnings else "")

    width = max([len(b) for b in brands] + [5])
    print("\nNew mentions per source and brand (* = warnings, see fetch_runs.error)")
    print(f"{'brand':<{width}}  " + "  ".join(f"{s:>10}" for s in sources) + f"  {'total':>7}")
    for brand in brands:
        row = [cell.get((brand, s)) for s in sources]
        total = sum(r.new_rows for r in row if r)
        print(f"{brand:<{width}}  " + "  ".join(f"{fmt(r):>10}" for r in row) + f"  {total:>7}")
    grand = sum(r.new_rows for r in results)
    print(f"{'total':<{width}}  " + "  ".join(
        f"{sum(r.new_rows for r in results if r.source == s):>10}" for s in sources
    ) + f"  {grand:>7}")

    for r in results:
        if r.error:
            print(f"  ✘ {r.source}/{r.brand}: {r.error}")


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brand", action="append", help="brand name (repeatable); default: all brands")
    parser.add_argument(
        "--source", action="append", choices=sorted(SOURCE_FACTORIES),
        help="source to run (repeatable); default: all enabled sources",
    )
    parser.add_argument(
        "--since-days", type=int, default=settings.fetch_since_days,
        help=f"only fetch mentions newer than this many days (default {settings.fetch_since_days})",
    )
    parser.add_argument("--csv", metavar="FILE", help="import a CSV file for exactly one --brand")
    parser.add_argument("-q", "--quiet", action="store_true", help="only print the summary")
    args = parser.parse_args(argv)
    if args.csv and (not args.brand or len(args.brand) != 1):
        parser.error("--csv needs exactly one --brand")
    if args.csv and args.source:
        parser.error("--csv cannot be combined with --source")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    from db.session import get_session_factory

    factory = get_session_factory()
    brands = load_brands(factory, args.brand)
    if not brands:
        print("No brands in the database. Run: python manage_brands.py")
        return 1

    sources: list[Source]
    if args.csv:
        sources = [CsvSource(args.csv)]
    else:
        names = args.source or list(SOURCE_FACTORIES)
        sources = [SOURCE_FACTORIES[n]() for n in names]

    since = utc_now() - timedelta(days=args.since_days)
    results = run_fetch(factory, brands, sources, since)
    print_summary(results)
    return 1 if any(r.error for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
