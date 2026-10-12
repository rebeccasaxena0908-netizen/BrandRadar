from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from sources.base import BrandSpec
from sources.csv_import import CsvSource, parse_date

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
    with pytest.raises(ValueError, match="missing column.*rating"):
        CsvSource(content="source,author,text,date\nx,y,z,2026-10-01\n").fetch(BRAND)


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
