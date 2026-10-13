"""Sync brands.yaml into the brands table.

    python manage_brands.py            # insert new brands, update changed ones
    python manage_brands.py --list     # show what is in the database
    python manage_brands.py --prune    # also delete brands no longer in the YAML (and their mentions!)
    python manage_brands.py --recheck-mentions          # list stored keyword-search mentions that
                                                        # no longer match (after changing excludes)
    python manage_brands.py --recheck-mentions --apply  # ...and delete them (analysis rows cascade)
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from config import PROJECT_ROOT
from db.models import Brand, Mention
from sources.base import KEYWORD_SOURCES, BrandSpec, still_matches

DEFAULT_FILE = PROJECT_ROOT / "brands.yaml"


@dataclass
class BrandEntry:
    name: str
    is_own: bool = False
    keywords: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    exclude_if: list[str] = field(default_factory=list)
    subreddits: list[str] = field(default_factory=list)  # Reddit source: the brand's own communities
    app_store_id: str | None = None
    play_store_id: str | None = None
    features: dict[str, list[str]] = field(default_factory=dict)  # feature -> synonyms

    def stored_keywords(self) -> list[str]:
        return (self.keywords + [f"-{e}" for e in self.exclude] + [f"!{e}" for e in self.exclude_if]
                + [f"r/{s}" for s in self.subreddits])


def _str_list(value: Any, what: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, (str, int)) for v in value):
        raise ValueError(f"{what} must be a list of strings")
    return [str(v).strip() for v in value if str(v).strip()]


def _features(value: Any, name: str) -> dict[str, list[str]]:
    """A list of feature names, or a mapping of feature name -> list of synonyms."""
    if value is None:
        return {}
    if isinstance(value, list):
        return {f: [] for f in _str_list(value, f"{name}.features")}
    if isinstance(value, dict):
        return {str(k).strip(): _str_list(v, f"{name}.features.{k}") for k, v in value.items() if str(k).strip()}
    raise ValueError(f"{name}.features must be a list or a mapping of feature: [synonyms]")


def load_yaml(path: Path) -> list[BrandEntry]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_brands = data.get("brands")
    if not isinstance(raw_brands, list) or not raw_brands:
        raise ValueError(f"{path.name}: expected a non-empty 'brands:' list")

    entries: list[BrandEntry] = []
    for i, raw in enumerate(raw_brands, start=1):
        if not isinstance(raw, dict) or not str(raw.get("name", "")).strip():
            raise ValueError(f"{path.name}: brand #{i} needs a name")
        name = str(raw["name"]).strip()
        entry = BrandEntry(
            name=name,
            is_own=bool(raw.get("is_own", False)),
            keywords=_str_list(raw.get("keywords"), f"{name}.keywords"),
            exclude=_str_list(raw.get("exclude"), f"{name}.exclude"),
            exclude_if=_str_list(raw.get("exclude_if"), f"{name}.exclude_if"),
            subreddits=[s.removeprefix("r/") for s in _str_list(raw.get("subreddits"), f"{name}.subreddits")],
            app_store_id=str(raw["app_store_id"]) if raw.get("app_store_id") else None,
            play_store_id=str(raw["play_store_id"]) if raw.get("play_store_id") else None,
            features=_features(raw.get("features"), name),
        )
        if not entry.keywords:
            raise ValueError(f"{name}: needs at least one keyword")
        bad = [s for s in entry.subreddits if not re.fullmatch(r"[A-Za-z0-9_]{2,21}", s)]
        if bad:
            raise ValueError(f"{name}: invalid subreddit name(s): {', '.join(bad)}")
        if any(k.startswith(("-", "!", "r/")) for k in entry.keywords):
            raise ValueError(f"{name}: put exclusions under 'exclude' / 'exclude_if', not '-keyword' or '!keyword'")
        entries.append(entry)

    names = [e.name.lower() for e in entries]
    if len(names) != len(set(names)):
        raise ValueError(f"{path.name}: duplicate brand names")
    if sum(e.is_own for e in entries) != 1:
        raise ValueError(f"{path.name}: exactly one brand must have is_own: true")
    return entries


def sync_brands(session: Session, entries: Sequence[BrandEntry], prune: bool = False) -> dict[str, list[str]]:
    """Upsert entries by name. Returns {'added': [...], 'updated': [...], 'unchanged': [...], 'removed'/'orphaned': [...]}."""
    report: dict[str, list[str]] = {"added": [], "updated": [], "unchanged": [], "removed": [], "orphaned": []}
    existing = {b.name.lower(): b for b in session.scalars(select(Brand))}

    for entry in entries:
        values = {
            "name": entry.name,
            "is_own": entry.is_own,
            "keywords": entry.stored_keywords(),
            "app_store_id": entry.app_store_id,
            "play_store_id": entry.play_store_id,
            "features": entry.features,
        }
        brand = existing.pop(entry.name.lower(), None)
        if brand is None:
            session.add(Brand(**values))
            report["added"].append(entry.name)
            continue
        changed = False
        for key, value in values.items():
            if getattr(brand, key) != value:
                setattr(brand, key, value)
                changed = True
        report["updated" if changed else "unchanged"].append(entry.name)

    for brand in existing.values():
        if prune:
            session.delete(brand)
            report["removed"].append(brand.name)
        else:
            report["orphaned"].append(brand.name)
    session.flush()
    return report


def recheck_mentions(session: Session, apply: bool = False) -> list[tuple[str, Mention]]:
    """Stored mentions from keyword-search sources that no longer match their brand's keywords."""
    stale: list[tuple[str, Mention]] = []
    for brand in session.scalars(select(Brand)):
        spec = BrandSpec.from_model(brand)
        query = select(Mention).where(Mention.brand_id == brand.id, Mention.source.in_(KEYWORD_SOURCES))
        stale += [(brand.name, m) for m in session.scalars(query) if not still_matches(m.source, m.text, m.url, spec)]
    if apply:
        for _, mention in stale:
            session.delete(mention)
        session.flush()
    return stale


def list_brands(session: Session) -> None:
    for b in session.scalars(select(Brand).order_by(Brand.is_own.desc(), Brand.name)):
        own = "own" if b.is_own else "competitor"
        print(f"{b.id:>3}  {b.name:<12} {own:<10}  app_store={b.app_store_id or '-':<11} "
              f"play={b.play_store_id or '-'}")
        print(f"     keywords={b.keywords}")
        print(f"     features={list(b.features or [])}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE, help="YAML file (default brands.yaml)")
    parser.add_argument("--list", action="store_true", help="list brands in the database and exit")
    parser.add_argument("--prune", action="store_true", help="delete brands that are not in the YAML")
    parser.add_argument(
        "--recheck-mentions", action="store_true",
        help="after syncing, list stored HN/Reddit mentions that no longer match their brand",
    )
    parser.add_argument("--apply", action="store_true", help="with --recheck-mentions: delete them")
    args = parser.parse_args(argv)

    from db.session import get_session

    if args.list:
        with get_session() as session:
            list_brands(session)
        return 0

    try:
        entries = load_yaml(args.file)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"✘ {exc}")
        return 1

    with get_session() as session:
        report = sync_brands(session, entries, prune=args.prune)
    for key, names in report.items():
        if names:
            print(f"{key:>9}: {', '.join(names)}")
    if report["orphaned"]:
        print("           (in the database but not in the YAML; use --prune to delete them and their mentions)")

    if args.recheck_mentions:
        with get_session() as session:
            stale = recheck_mentions(session, apply=args.apply)
            verb = "Deleted" if args.apply else "Would delete"
            print(f"\n{verb} {len(stale)} mention(s) that no longer match their brand:")
            for brand, m in stale:
                print(f"  [{brand}/{m.source} {m.external_id}] {' '.join(m.text.split())[:110]}")
            if stale and not args.apply:
                print("Run again with --apply to delete them (their analysis rows are removed too).")
    elif args.apply:
        print("--apply does nothing without --recheck-mentions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
