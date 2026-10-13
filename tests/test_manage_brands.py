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
    features:
      sync: [syncing]
      AI: []
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
    assert all("mobile app" not in e.features for e in entries)
    assert all(e.features["pricing"] for e in entries)  # shared core merged in via the YAML anchor


def test_plain_feature_list_still_accepted(tmp_path: Path) -> None:
    text = "brands:\n  - name: A\n    is_own: true\n    keywords: [a]\n    features: [sync, AI]\n"
    assert load_yaml(write(tmp_path, text))[0].features == {"sync": [], "AI": []}


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
        assert notion.features == {"sync": ["syncing"], "AI": []}
        assert notion.is_own is True

    changed = YAML.replace("      AI: []\n", "      AI: []\n      offline: [no internet]\n").replace("  - name: Obsidian\n    keywords: [Obsidian]\n", "")
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


def test_exclude_if_stored_with_bang(tmp_path: Path) -> None:
    text = "brands:\n  - name: A\n    is_own: true\n    keywords: [Aa]\n    exclude_if: [Fallout]\n"
    assert load_yaml(write(tmp_path, text))[0].stored_keywords() == ["Aa", "!Fallout"]


def test_recheck_mentions_finds_and_deletes_stale(tmp_path: Path) -> None:
    from datetime import datetime

    from sqlalchemy import create_engine

    from db.models import Base, Mention
    from manage_brands import recheck_mentions

    from sqlalchemy import event

    from db.models import Analysis

    engine = create_engine("sqlite://")

    @event.listens_for(engine, "connect")
    def _fk_on(conn, _):  # type: ignore[no-untyped-def]
        conn.execute("PRAGMA foreign_keys=ON")  # SQLite only cascades with this, like MySQL always does

    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with factory() as s:
        brand = Brand(name="Obsidian", is_own=True, keywords=["Obsidian", "!Fallout"])
        s.add(brand)
        s.flush()
        when = datetime(2026, 10, 5)
        s.add_all([
            Mention(brand_id=brand.id, source="hackernews", external_id="1", text="Obsidian sync is great", created_at=when),
            Mention(brand_id=brand.id, source="hackernews", external_id="2", text="Obsidian made Fallout NV", created_at=when),
            Mention(brand_id=brand.id, source="appstore", external_id="3", text="Fallout of an update", created_at=when),
        ])
        s.flush()
        s.add_all(Analysis(mention_id=m.id, sentiment="neutral") for m in s.query(Mention))  # cascade must work
        s.commit()
        assert [m.external_id for _, m in recheck_mentions(s)] == ["2"]  # store reviews are never rechecked
        assert s.query(Mention).count() == 3  # dry run deletes nothing
        recheck_mentions(s, apply=True)
        s.commit()
        assert sorted(m.external_id for m in s.query(Mention)) == ["1", "3"]
        assert s.query(Analysis).count() == 2  # the stale mention's analysis row went with it


def test_subreddits_stored_with_r_prefix(tmp_path: Path) -> None:
    text = "brands:\n  - name: A\n    is_own: true\n    keywords: [Aa]\n    subreddits: [r/AaApp, AaHelp]\n"
    assert load_yaml(write(tmp_path, text))[0].stored_keywords() == ["Aa", "r/AaApp", "r/AaHelp"]
    bad = "brands:\n  - name: A\n    is_own: true\n    keywords: [Aa]\n    subreddits: [not a sub]\n"
    with pytest.raises(ValueError, match="invalid subreddit"):
        load_yaml(write(tmp_path, bad))
