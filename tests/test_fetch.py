"""Fetch runner: dedupe, per-brand uniqueness and fetch_runs logging (SQLite, no network)."""

from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from db.models import FetchRun, Mention
from fetch import insert_new_mentions, load_brands, run_fetch
from sources.base import BaseSource, BrandSpec, RawMention

SINCE = datetime(2026, 10, 1)


def raw(external_id: str, text: str = "Notion is nice", source: str = "fake") -> RawMention:
    return RawMention(source=source, external_id=external_id, text=text, created_at=datetime(2026, 10, 5))


class FakeSource(BaseSource):
    def __init__(self, name: str, mentions: list[RawMention], warnings: list[str] | None = None) -> None:
        super().__init__()
        self.name = name
        self._mentions = mentions
        self._warnings = warnings or []

    def fetch(self, brand: BrandSpec, since: datetime) -> list[RawMention]:
        self.reset()
        for w in self._warnings:
            self.warn(w)
        return [RawMention(**{**m.__dict__, "source": self.name}) for m in self._mentions]


class BrokenSource(BaseSource):
    name = "broken"

    def fetch(self, brand: BrandSpec, since: datetime) -> list[RawMention]:
        raise ConnectionError("API is down")


class DisabledSource(BaseSource):
    name = "disabled"
    enabled = False

    def fetch(self, brand: BrandSpec, since: datetime) -> list[RawMention]:
        raise AssertionError("disabled source must not run")


def count(factory: sessionmaker[Session], model: type) -> int:
    with factory() as s:
        return s.scalar(select(func.count()).select_from(model)) or 0


def test_insert_dedupes_within_batch_and_against_db(factory: sessionmaker[Session], notion: BrandSpec) -> None:
    with factory() as s:
        assert insert_new_mentions(s, notion.id, [raw("1"), raw("2"), raw("1")]) == 2
        s.commit()
    with factory() as s:
        assert insert_new_mentions(s, notion.id, [raw("2"), raw("3")]) == 1
        s.commit()
    assert count(factory, Mention) == 3


def test_same_external_id_allowed_for_two_brands_and_two_sources(
    factory: sessionmaker[Session], brands: list[BrandSpec]
) -> None:
    notion, obsidian = brands
    with factory() as s:
        assert insert_new_mentions(s, notion.id, [raw("42")]) == 1
        assert insert_new_mentions(s, obsidian.id, [raw("42")]) == 1  # same HN item, other brand
        assert insert_new_mentions(s, notion.id, [raw("42", source="other")]) == 1  # other source
        s.commit()
    assert count(factory, Mention) == 3


def test_insert_skips_empty_text_and_ids(factory: sessionmaker[Session], notion: BrandSpec) -> None:
    with factory() as s:
        assert insert_new_mentions(s, notion.id, [raw("1", text="   "), raw("")]) == 0


def test_second_run_adds_nothing(factory: sessionmaker[Session], brands: list[BrandSpec]) -> None:
    sources = [FakeSource("hackernews", [raw("1"), raw("2")])]
    first = run_fetch(factory, brands, sources, SINCE)
    second = run_fetch(factory, brands, sources, SINCE)
    assert [r.new_rows for r in first] == [2, 2]  # both brands
    assert [r.new_rows for r in second] == [0, 0]
    assert count(factory, Mention) == 4


def test_fetch_runs_logged_for_success_error_and_warnings(
    factory: sessionmaker[Session], notion: BrandSpec
) -> None:
    sources = [
        BrokenSource(),
        FakeSource("appstore", [raw("a")], warnings=["in: HTTPError 503"]),
        DisabledSource(),
    ]
    results = run_fetch(factory, [notion], sources, SINCE)

    # the broken source did not stop the next one
    assert [(r.source, r.new_rows, r.error is not None, r.skipped) for r in results] == [
        ("broken", 0, True, False),
        ("appstore", 1, False, False),
        ("disabled", 0, False, True),
    ]
    with factory() as s:
        runs = {r.source: r for r in s.scalars(select(FetchRun))}
    assert set(runs) == {"broken", "appstore"}  # disabled source writes no run
    assert runs["broken"].error == "error: ConnectionError: API is down"
    assert runs["broken"].new_rows == 0
    assert runs["broken"].finished_at is not None
    assert runs["appstore"].error == "warning: in: HTTPError 503"
    assert runs["appstore"].new_rows == 1
    assert runs["appstore"].brand_id == notion.id
    assert runs["appstore"].started_at <= runs["appstore"].finished_at


def test_clean_run_has_no_error(factory: sessionmaker[Session], notion: BrandSpec) -> None:
    run_fetch(factory, [notion], [FakeSource("hackernews", [raw("1")])], SINCE)
    with factory() as s:
        run = s.scalars(select(FetchRun)).one()
    assert run.error is None and run.new_rows == 1


def test_load_brands_filters_and_rejects_unknown(factory: sessionmaker[Session], brands: list[BrandSpec]) -> None:
    assert [b.name for b in load_brands(factory)] == ["Notion", "Obsidian"]  # own brand first
    assert [b.name for b in load_brands(factory, ["obsidian"])] == ["Obsidian"]
    loaded = load_brands(factory, ["Notion"])[0]
    assert loaded.keywords == ("Notion", "notion.so") and loaded.exclude == ("Notion Ink",)
    with pytest.raises(SystemExit):
        load_brands(factory, ["Nope"])
