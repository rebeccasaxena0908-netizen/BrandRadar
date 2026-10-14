from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from config import PROJECT_ROOT
from db.models import Brand
from manage_brands import BrandEntry, add_brand, dry_run, load_yaml, sync_brands

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
    # the three hand-made groups come first; groups added from the dashboard (Step 11) follow them
    assert [e.name for e in entries if e.is_own][:3] == ["Notion", "Laneige", "DOMS"]
    assert {"Note apps", "Skincare", "Stationery"} <= {e.group for e in entries}
    entries = [e for e in entries if e.group == "Note apps"]
    assert [e.name for e in entries] == ["Notion", "Obsidian", "Evernote", "ClickUp"]
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
        ("brands:\n  - name: A\n    is_own: true\n    keywords: [a]\n    sources: [amazon]\n", "unknown source"),
        ("brands:\n  - name: A\n    is_own: true\n    keywords: [a]\n    profile: cars\n", "unknown profile"),
        ("brands:\n  - name: A\n    is_own: true\n    keywords: [a]\n    group: G\n"
         "  - name: B\n    is_own: true\n    keywords: [b]\n    group: G\n", "group 'G' needs exactly one"),
        ("brands:\n  - name: A\n    is_own: true\n    keywords: [a]\n    group: G\n    profile: software\n"
         "  - name: B\n    keywords: [b]\n    group: G\n    profile: stationery\n", "mixes profiles"),
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


# ---------------------------------------------------------------- groups, add, dry-run

GROUPS = """
brands:
  - name: Notion
    group: Note apps
    profile: software
    is_own: true
    keywords: [Notion]
  - name: DOMS
    group: Stationery
    profile: stationery
    is_own: true
    keywords: [DOMS]
    require_context: [pencil, eraser]
    youtube_queries: [DOMS pencil review]
"""


def test_brand_spec_comes_straight_from_the_yaml(tmp_path: Path) -> None:
    doms = load_yaml(write(tmp_path, GROUPS))[1]
    spec = doms.spec(7)
    assert (spec.id, spec.group, spec.profile, spec.require_context) == (7, "Stationery", "stationery", ("pencil", "eraser"))
    assert spec.youtube_queries == ("DOMS pencil review",) and spec.sources is None
    assert doms.config() == {"require_context": ["pencil", "eraser"], "youtube_queries": ["DOMS pencil review"]}


def answers(*replies: str):  # type: ignore[no-untyped-def]
    queue = list(replies)
    return lambda prompt: queue.pop(0)


def test_add_competitor_to_an_existing_group(tmp_path: Path) -> None:
    path = write(tmp_path, GROUPS)
    said: list[str] = []
    entry = add_brand(path, answers(
        "doms",  # taken (case-insensitive): asked again
        "Apsara", "Stationery",
        "Apsara",  # keywords
        "pencil, eraser, sharpener",  # context
        "Apsara Hotel",  # exclude
        "",  # exclude_if
        "",  # youtube queries: default
        "",  # sources: profile defaults
    ), said.append)
    assert (entry.is_own, entry.profile, entry.group) == (False, "stationery", "Stationery")
    assert entry.youtube_queries == ["Apsara review"] and entry.sources is None
    assert any("Needs a name" in s for s in said)
    text = path.read_text()
    assert text.startswith(GROUPS.rstrip("\n"))  # existing text and comments are kept
    new = load_yaml(path)[-1]
    assert (new.name, new.require_context, new.exclude) == ("Apsara", ["pencil", "eraser", "sharpener"], ["Apsara Hotel"])


def test_add_new_group_makes_the_own_brand_and_asks_for_a_profile(tmp_path: Path) -> None:
    from nlp.llm.provider import NoneProvider

    path = write(tmp_path, GROUPS)
    entry = add_brand(path, answers(
        "Laneige", "Skincare", "beauty_skincare",
        "Laneige", "-", "", "", "Laneige lip mask, Laneige review", "youtube, news",
    ), lambda s: None, provider=NoneProvider())
    assert entry.is_own and entry.profile == "beauty_skincare" and entry.sources == ["youtube", "news"]
    assert load_yaml(path)[-1].youtube_queries == ["Laneige lip mask", "Laneige review"]


def test_add_never_writes_an_invalid_file(tmp_path: Path) -> None:
    path = write(tmp_path, GROUPS)
    before = path.read_text()
    with pytest.raises(ValueError, match="unknown source"):
        add_brand(path, answers("Natraj", "Stationery", "Natraj", "", "", "", "", "amazon"), lambda s: None)
    assert path.read_text() == before and not list(tmp_path.glob("*.new"))


class FakeNamedSource:
    def __init__(self, name: str, texts: list[str], enabled: bool = True) -> None:
        from sources.base import BaseSource

        self.name, self.texts, self.enabled = name, texts, enabled
        self._base = BaseSource()

    @property
    def rejected(self):  # type: ignore[no-untyped-def]
        return self._base.rejected

    @property
    def warnings(self):  # type: ignore[no-untyped-def]
        return self._base.warnings

    def fetch(self, brand, since):  # type: ignore[no-untyped-def]
        from sources.base import RawMention

        self._base.reset()
        return [RawMention(self.name, str(i), t, since) for i, t in enumerate(self.texts) if self._base.accept(t, brand)]


def test_dry_run_shows_matched_and_rejected_with_reasons(tmp_path: Path) -> None:
    doms = load_yaml(write(tmp_path, GROUPS))[1].spec()
    news = FakeNamedSource("news", ["DOMS pencil sales up", "DOMS after leg day", "Nothing about it"])
    youtube = FakeNamedSource("youtube", [], enabled=False)
    hn = FakeNamedSource("hackernews", ["DOMS pencil"])  # not a stationery source: not run
    result = dry_run(doms, [news, youtube, hn], None)
    assert result.matched == [("news", "DOMS pencil sales up")]
    assert result.rejected == [("news", "no context word", "DOMS after leg day"),
                               ("news", "name not found", "Nothing about it")]
    assert result.notes == ["youtube: skipped (not configured)"]
