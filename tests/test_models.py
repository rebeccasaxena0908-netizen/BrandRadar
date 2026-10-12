"""Check that db/models.py matches db/schema.sql (no database needed)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from sqlalchemy import UniqueConstraint

from db.models import Base, TABLE_NAMES

SCHEMA = (Path(__file__).resolve().parent.parent / "db" / "schema.sql").read_text()
TABLE_RE = re.compile(r"CREATE TABLE IF NOT EXISTS (\w+) \((.*?)\n\) ENGINE=InnoDB", re.S)
FK_RE = re.compile(
    r"FOREIGN KEY \((\w+)\)\s+REFERENCES (\w+) \((\w+)\) ON DELETE (CASCADE|SET NULL)"
)
UNIQUE_RE = re.compile(r"UNIQUE KEY (\w+) \(([^)]*)\)")
NON_COLUMN = {"PRIMARY", "UNIQUE", "KEY", "CONSTRAINT", "REFERENCES"}


def parse_schema() -> dict[str, str]:
    return {name: body for name, body in TABLE_RE.findall(SCHEMA)}


def schema_columns(body: str) -> set[str]:
    cols: set[str] = set()
    for line in body.splitlines():
        words = line.strip().split()
        if words and words[0] not in NON_COLUMN:
            cols.add(words[0])
    return cols


SQL_TABLES = parse_schema()


def test_schema_uses_utf8mb4() -> None:
    assert "CHARACTER SET utf8mb4" in SCHEMA
    assert SCHEMA.count("DEFAULT CHARSET=utf8mb4") == len(SQL_TABLES)


def test_same_tables() -> None:
    assert set(SQL_TABLES) == set(TABLE_NAMES)
    assert len(TABLE_NAMES) == 7


@pytest.mark.parametrize("table", sorted(SQL_TABLES))
def test_same_columns(table: str) -> None:
    model_cols = {c.name for c in Base.metadata.tables[table].columns}
    assert schema_columns(SQL_TABLES[table]) == model_cols


@pytest.mark.parametrize("table", sorted(SQL_TABLES))
def test_same_foreign_keys(table: str) -> None:
    sql_fks = set(FK_RE.findall(SQL_TABLES[table]))
    model_fks = {
        (fk.parent.name, fk.column.table.name, fk.column.name, fk.ondelete)
        for fk in Base.metadata.tables[table].foreign_keys
    }
    assert sql_fks == model_fks


def model_unique_keys(table: str) -> set[tuple[str, tuple[str, ...]]]:
    return {
        (con.name, tuple(c.name for c in con.columns))
        for con in Base.metadata.tables[table].constraints
        if isinstance(con, UniqueConstraint)
    }


@pytest.mark.parametrize("table", sorted(SQL_TABLES))
def test_same_unique_keys(table: str) -> None:
    sql_uniques = {
        (name, tuple(c.strip() for c in cols.split(",")))
        for name, cols in UNIQUE_RE.findall(SQL_TABLES[table])
    }
    assert sql_uniques == model_unique_keys(table)


def test_mentions_unique_per_brand_source_external_id() -> None:
    assert ("uq_mentions_brand_source_external", ("brand_id", "source", "external_id")) in (
        model_unique_keys("mentions")
    )
