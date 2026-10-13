from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from sources.base import BrandSpec
from sources.csv_import import CsvSource, parse_date, parse_mapping, parse_rating, resolve_columns

BRAND = BrandSpec(id=1, name="Notion", keywords=("Notion",))

GOOD = """source,author,text,date,rating
twitter,@amy,Notion AI is great 🚀,2026-10-01T14:30:00Z,
appreview,bob,"Crashes on login, please fix",2026-10-02,1
twitter,@amy,Notion AI is great 🚀,2026-10-01T14:30:00Z,
"""


def test_imports_rows_and_dedupes_identical_lines() -> None:
    source = CsvSource(content=GOOD)
    mentions = source.fetch(BRAND)
    assert len(mentions) == 2
    tweet, review = mentions
    assert tweet.source == "twitter"
    assert tweet.author == "@amy"
    assert tweet.text == "Notion AI is great 🚀"
    assert tweet.created_at == datetime(2026, 10, 1, 14, 30)
    assert tweet.rating is None
    assert review.rating == 1
    assert review.created_at == datetime(2026, 10, 2)
    assert len(tweet.external_id) == 40
    assert source.warnings == []


def test_external_id_is_stable_across_imports() -> None:
    first = [m.external_id for m in CsvSource(content=GOOD).fetch(BRAND)]
    second = [m.external_id for m in CsvSource(content=GOOD).fetch(BRAND)]
    assert first == second


def test_missing_columns_raise() -> None:
    with pytest.raises(ValueError, match="missing column.*date.*--map"):
        CsvSource(content="source,author,text,when\nx,y,z,2026-10-01\n").fetch(BRAND)


def test_only_text_and_date_are_required() -> None:
    [m] = CsvSource(content="text,date\nlovely pencils,2026-10-01\n", default_source="amazon").fetch(BRAND)
    assert (m.source, m.author, m.rating) == ("amazon", None, None)


def test_bad_rows_become_warnings() -> None:
    content = """source,author,text,date,rating
twitter,a,,2026-10-01,
twitter,b,ok text,not a date,
twitter,c,ok text,2026-10-01,9
twitter,d,fine,01/10/2026,5
"""
    source = CsvSource(content=content)
    mentions = source.fetch(BRAND)
    assert [m.author for m in mentions] == ["d"]
    assert mentions[0].created_at == datetime(2026, 10, 1)  # dd/mm/yyyy
    assert len(source.warnings) == 3
    assert source.warnings[0].startswith("line 2:")


def test_reads_file_with_excel_bom(tmp_path: Path) -> None:
    path = tmp_path / "export.csv"
    path.write_text("﻿" + GOOD, encoding="utf-8")
    assert len(CsvSource(path).fetch(BRAND)) == 2


def test_parse_date_with_offset() -> None:
    assert parse_date("2026-10-01T10:00:00+02:00") == datetime(2026, 10, 1, 8, 0)


def test_shop_export_headers_are_recognised() -> None:
    content = """Reviewer Name;Stars;Review Body;Posted On;Marketplace
Priya;4 out of 5;Smooth pencil, dark lines;12 Oct 2026;amazon
Rahul;★★☆☆☆;Lead breaks while sharpening;Oct 11, 2026 6:30 PM;flipkart
"""
    mentions = CsvSource(content=content).fetch(BRAND)
    assert [(m.author, m.rating, m.source) for m in mentions] == [("Priya", 4, "amazon"), ("Rahul", 2, "flipkart")]
    assert mentions[0].created_at == datetime(2026, 10, 12) and mentions[1].created_at == datetime(2026, 10, 11, 18, 30)


def test_manual_mapping_wins_and_unknown_columns_are_explained() -> None:
    content = "What customers said,Day,Review\nGreat eraser,02/10/2026,ignored column\n"
    mapping = parse_mapping("text=What customers said, date=Day")
    [m] = CsvSource(content=content, mapping=mapping).fetch(BRAND)
    assert m.text == "Great eraser" and m.created_at == datetime(2026, 10, 2)
    with pytest.raises(ValueError, match="not in the CSV"):
        resolve_columns(["text", "date"], parse_mapping("rating=Score"))
    with pytest.raises(ValueError, match="should look like"):
        parse_mapping("colour=Shade")


@pytest.mark.parametrize(("value", "expected"), [
    ("4", 4), ("4.0", 4), ("4,5", 5), ("4.4", 4), ("3/5", 3), ("8/10", 4), ("7 / 10", 4), ("4 out of 5", 4),
    ("5 stars", 5), ("★★★★☆", 4), ("", None), (" ", None),
])
def test_ratings(value: str, expected: int | None) -> None:
    assert parse_rating(value) == expected


@pytest.mark.parametrize("value", ["0", "6", "11/10", "great", "☆☆☆☆☆"])
def test_bad_ratings(value: str) -> None:
    with pytest.raises(ValueError):
        parse_rating(value)


@pytest.mark.parametrize(("value", "order", "expected"), [
    ("01/10/2026", "dmy", datetime(2026, 10, 1)), ("01/10/2026", "mdy", datetime(2026, 1, 10)),
    ("13-10-2026", "dmy", datetime(2026, 10, 13)), ("13.10.26", "dmy", datetime(2026, 10, 13)),
    ("2026/10/13 09:15", "dmy", datetime(2026, 10, 13, 9, 15)), ("October 3, 2026", "dmy", datetime(2026, 10, 3)),
    ("3-Oct-2026", "dmy", datetime(2026, 10, 3)), ("1791849600", "dmy", datetime(2026, 10, 13)),
    ("1791849600000", "dmy", datetime(2026, 10, 13)),
])
def test_date_formats(value: str, order: str, expected: datetime) -> None:
    assert parse_date(value, order) == expected
