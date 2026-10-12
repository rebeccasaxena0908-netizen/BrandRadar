from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from config import PROJECT_ROOT
from db.models import Brand
from manage_brands import load_yaml, sync_brands

YAML = """
brands:
  - name: Notion
    is_own: true
    keywords: [Notion]
    exclude: [Notion Ink]
    app_store_id: "1232780281"
    features: [sync, AI]
  - name: Obsidian
    keywords: [Obsidian]
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "brands.yaml"
    path.write_text(text)
    return path


def test_project_brands_yaml_is_valid() -> None:
    entries = load_yaml(PROJECT_ROOT / "brands.yaml")
    assert [e.name for e in entries] == ["Notion", "Obsidian", "Evernote", "ClickUp"]
    assert [e.name for e in entries if e.is_own] == ["Notion"]
    assert all(e.app_store_id and e.play_store_id and e.features for e in entries)


def test_sync_adds_then_updates(tmp_path: Path) -> None:
    from sqlalchemy import create_engine

    from db.models import Base

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)

    with factory() as s:
        report = sync_brands(s, load_yaml(write(tmp_path, YAML)))
        s.commit()
    assert report["added"] == ["Notion", "Obsidian"]

    with factory() as s:
        notion = s.scalars(select(Brand).where(Brand.name == "Notion")).one()
        assert notion.keywords == ["Notion", "-Notion Ink"]
        assert notion.is_own is True

    changed = YAML.replace("[sync, AI]", "[sync, AI, offline]").replace("  - name: Obsidian\n    keywords: [Obsidian]\n", "")
    with factory() as s:
        report = sync_brands(s, load_yaml(write(tmp_path, changed)))
        s.commit()
    assert report["updated"] == ["Notion"]
    assert report["orphaned"] == ["Obsidian"]  # not deleted without --prune

    with factory() as s:
        report = sync_brands(s, load_yaml(write(tmp_path, changed)), prune=True)
        s.commit()
        assert report["unchanged"] == ["Notion"] and report["removed"] == ["Obsidian"]
        assert [b.name for b in s.scalars(select(Brand))] == ["Notion"]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("brands: []", "non-empty"),
        ("brands:\n  - name: A\n    keywords: [a]\n", "exactly one brand"),
        ("brands:\n  - name: A\n    is_own: true\n", "at least one keyword"),
        ("brands:\n  - name: A\n    is_own: true\n    keywords: [a]\n  - name: a\n    keywords: [b]\n", "duplicate"),
        ("brands:\n  - name: A\n    is_own: true\n    keywords: [a, -b]\n", "exclude"),
    ],
)
def test_invalid_yaml_rejected(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_yaml(write(tmp_path, text))
