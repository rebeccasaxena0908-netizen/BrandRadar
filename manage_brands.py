"""Sync brands.yaml into the brands table, add brands, and test brand matching.

    python manage_brands.py            # insert new brands, update changed ones
    python manage_brands.py add        # add a brand interactively (writes brands.yaml, then syncs)
    python manage_brands.py dry-run --brand DOMS   # 20 matched + 20 rejected texts per brand, nothing stored
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
from sources.base import ALL_SOURCES, KEYWORD_SOURCES, BrandSpec, still_matches

DEFAULT_FILE = PROJECT_ROOT / "brands.yaml"


@dataclass
class BrandEntry:
    name: str
    is_own: bool = False
    keywords: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    exclude_if: list[str] = field(default_factory=list)
    subreddits: list[str] = field(default_factory=list)  # Reddit source: the brand's own communities
    group: str = "Default"
    profile: str = "consumer_product"
    require_context: list[str] = field(default_factory=list)
    youtube_queries: list[str] = field(default_factory=list)
    sources: list[str] | None = None  # None = the profile's defaults
    app_store_id: str | None = None
    play_store_id: str | None = None
    features: dict[str, list[str]] = field(default_factory=dict)  # feature -> synonyms

    def config(self) -> dict[str, Any] | None:
        config: dict[str, Any] = {}
        if self.require_context:
            config["require_context"] = self.require_context
        if self.youtube_queries:
            config["youtube_queries"] = self.youtube_queries
        if self.sources is not None:
            config["sources"] = self.sources
        return config or None

    def spec(self, brand_id: int = 0) -> BrandSpec:
        """The BrandSpec sources would get, straight from the YAML (no sync needed)."""
        return BrandSpec(
            id=brand_id, name=self.name, is_own=self.is_own, keywords=tuple(self.keywords),
            exclude=tuple(self.exclude), exclude_if=tuple(self.exclude_if), subreddits=tuple(self.subreddits),
            require_context=tuple(self.require_context), group=self.group, profile=self.profile,
            youtube_queries=tuple(self.youtube_queries), sources=tuple(self.sources) if self.sources else None,
            app_store_id=self.app_store_id, play_store_id=self.play_store_id, features=tuple(self.features),
        )

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
            group=str(raw.get("group") or "Default").strip(),
            profile=str(raw.get("profile") or "consumer_product").strip(),
            require_context=_str_list(raw.get("require_context"), f"{name}.require_context"),
            youtube_queries=_str_list(raw.get("youtube_queries"), f"{name}.youtube_queries"),
            sources=_str_list(raw["sources"], f"{name}.sources") if raw.get("sources") is not None else None,
            app_store_id=str(raw["app_store_id"]) if raw.get("app_store_id") else None,
            play_store_id=str(raw["play_store_id"]) if raw.get("play_store_id") else None,
            features=_features(raw.get("features"), name),
        )
        if not entry.keywords:
            raise ValueError(f"{name}: needs at least one keyword")
        bad = [s for s in entry.subreddits if not re.fullmatch(r"[A-Za-z0-9_]{2,21}", s)]
        if bad:
            raise ValueError(f"{name}: invalid subreddit name(s): {', '.join(bad)}")
        unknown_sources = [s for s in entry.sources or [] if s not in ALL_SOURCES]
        if unknown_sources:
            raise ValueError(f"{name}: unknown source(s) {', '.join(unknown_sources)} (known: {', '.join(ALL_SOURCES)})")
        if any(k.startswith(("-", "!", "r/")) for k in entry.keywords):
            raise ValueError(f"{name}: put exclusions under 'exclude' / 'exclude_if', not '-keyword' or '!keyword'")
        entries.append(entry)

    names = [e.name.lower() for e in entries]
    if len(names) != len(set(names)):
        raise ValueError(f"{path.name}: duplicate brand names")
    from nlp.profiles import profile_names

    known = set(profile_names())
    for entry in entries:
        if entry.profile not in known:
            raise ValueError(f"{entry.name}: unknown profile {entry.profile!r} (profiles.yaml has: {', '.join(sorted(known))})")
    groups: dict[str, list[BrandEntry]] = {}
    for entry in entries:
        groups.setdefault(entry.group, []).append(entry)
    for group, members in groups.items():
        if sum(e.is_own for e in members) != 1:
            raise ValueError(f"{path.name}: group {group!r} needs exactly one brand with is_own: true")
        if len({e.profile for e in members}) != 1:
            raise ValueError(f"{path.name}: group {group!r} mixes profiles; brands in a group must share one")
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
            "brand_group": entry.group,
            "profile": entry.profile,
            "config": entry.config(),
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


# ---------------------------------------------------------------- dry-run: is the matching right?

DRY_RUN_SOURCES = ("hackernews", "mastodon", "reddit", "news", "youtube")  # sources that match by name
DRY_RUN_SHOW = 20


@dataclass
class DryRun:
    brand: str
    matched: list[tuple[str, str]] = field(default_factory=list)  # (source, text)
    rejected: list[tuple[str, str, str]] = field(default_factory=list)  # (source, reason, text)
    notes: list[str] = field(default_factory=list)  # skipped sources, warnings, errors


def dry_run(spec: BrandSpec, sources: Sequence[Any], since: Any) -> DryRun:
    """Run the brand's name-matched sources live and keep what was matched and rejected. Nothing is
    stored; YouTube only searches video titles (one search call per query, no comment reads)."""
    from sources.base import brand_sources

    result = DryRun(spec.name)
    wanted = set(brand_sources(spec))
    for source in sources:
        if source.name not in wanted:
            continue
        if not source.enabled:
            result.notes.append(f"{source.name}: skipped (not configured)")
            continue
        try:
            texts = (source.preview(spec, since) if hasattr(source, "preview")
                     else [m.text for m in source.fetch(spec, since)])
        except Exception as exc:  # noqa: BLE001 — show it and carry on with the next source
            result.notes.append(f"{source.name}: error {type(exc).__name__}: {exc}")
            continue
        result.matched += [(source.name, t) for t in texts]
        result.rejected += [(source.name, reason, t) for t, reason in source.rejected]
        result.notes += [f"{source.name}: {w}" for w in source.warnings]
    return result


def _spread(items: list[Any], n: int) -> list[Any]:
    """Up to n items, taken round-robin by source so every source is represented."""
    by_source: dict[str, list[Any]] = {}
    for item in items:
        by_source.setdefault(item[0], []).append(item)
    out: list[Any] = []
    while len(out) < n and any(by_source.values()):
        for queue in by_source.values():
            if queue and len(out) < n:
                out.append(queue.pop(0))
    return out


def print_dry_run(result: DryRun, show: int = DRY_RUN_SHOW) -> None:
    def line(text: str) -> str:
        flat = " ".join(text.split())
        return flat if len(flat) <= 150 else flat[:147] + "…"

    print(f"\n=== {result.brand}: {len(result.matched)} matched, {len(result.rejected)} rejected ===")
    for note in result.notes:
        print(f"  ! {note}")
    print(f"  Matched (showing {min(show, len(result.matched))}):")
    for source, text in _spread(result.matched, show):
        print(f"    ✓ [{source}] {line(text)}")
    print(f"  Rejected (showing {min(show, len(result.rejected))}):")
    for source, reason, text in _spread(result.rejected, show):
        print(f"    ✗ [{source}] ({reason}) {line(text)}")


def dry_run_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="manage_brands.py dry-run", description=dry_run.__doc__)
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE)
    parser.add_argument("--brand", action="append", help="only this brand (repeatable); default: all")
    parser.add_argument("--group", action="append", help="only the brands in this group (repeatable)")
    parser.add_argument("--since-days", type=int, default=30)
    parser.add_argument("--show", type=int, default=DRY_RUN_SHOW, help="texts to show per list")
    args = parser.parse_args(argv)

    from datetime import timedelta

    from fetch import SOURCE_FACTORIES
    from sources.base import utc_now

    entries = load_yaml(args.file)
    if args.group:
        groups = {g.lower() for g in args.group}
        entries = [e for e in entries if e.group.lower() in groups]
    if args.brand:
        wanted = {b.lower() for b in args.brand}
        entries = [e for e in entries if e.name.lower() in wanted]
    if not entries:
        print("No matching brands in the YAML.")
        return 1
    ids: dict[str, int] = {}
    try:
        from db.session import get_session

        with get_session() as session:
            ids = {name.lower(): bid for bid, name in session.execute(select(Brand.id, Brand.name))}
    except Exception:  # noqa: BLE001 — the dry-run works without a database too
        pass
    since = utc_now() - timedelta(days=args.since_days)
    sources = [SOURCE_FACTORIES[name]() for name in DRY_RUN_SOURCES]
    for entry in entries:
        print_dry_run(dry_run(entry.spec(ids.get(entry.name.lower(), 0)), sources, since), args.show)
    print("\nNothing was stored. Adjust keywords / exclude / exclude_if / require_context in brands.yaml and rerun.")
    return 0


# ---------------------------------------------------------------- add: interactive


def _yaml_block(entry: BrandEntry) -> str:
    """The new brand as a brands.yaml list item, in the file's style."""
    data: dict[str, Any] = {"name": entry.name, "group": entry.group, "profile": entry.profile, "is_own": entry.is_own,
                            "keywords": entry.keywords}
    for key in ("require_context", "exclude", "exclude_if", "youtube_queries"):
        if getattr(entry, key):
            data[key] = getattr(entry, key)
    if entry.sources is not None:
        data["sources"] = entry.sources
    if entry.app_store_id:
        data["app_store_id"] = entry.app_store_id
    if entry.play_store_id:
        data["play_store_id"] = entry.play_store_id
    text = yaml.safe_dump([data], sort_keys=False, allow_unicode=True, default_flow_style=None, width=110)
    return "\n" + "".join(f"  {line}\n" for line in text.splitlines())


def _ask_list(ask: Any, prompt: str, default: list[str] | None = None) -> list[str]:
    shown = f" [{', '.join(default)}]" if default else ""
    answer = ask(f"{prompt}{shown}: ").strip()
    if not answer:
        return list(default or [])
    if answer == "-":
        return []
    return [part.strip() for part in answer.split(",") if part.strip()]


def edit_in_editor(text: str) -> str:
    """Open the text in $EDITOR (default: nano) and return what was saved."""
    import os
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=False, encoding="utf-8") as f:
        f.write(text)
        name = f.name
    try:
        subprocess.run([*os.environ.get("EDITOR", "nano").split(), name], check=False)
        return Path(name).read_text(encoding="utf-8")
    finally:
        Path(name).unlink(missing_ok=True)


def choose_new_profile(industry: str, brand: str, ask: Any = input, say: Any = print, edit: Any = edit_in_editor,
                       provider: Any = None) -> str:
    """No profile fits `industry`: draft one with the LLM, show it for review, save it if accepted.
    Without an LLM (or if the draft is rejected or unusable) the brand gets consumer_product."""
    from nlp.llm.profile_draft import FALLBACK, draft_from_yaml, draft_profile, save_profile
    from nlp.llm.provider import get_provider

    provider = provider or get_provider()
    if provider.name == "none":
        say(f"  No profile matches {industry!r} and LLM_PROVIDER=none, so {brand} uses {FALLBACK}.")
        return FALLBACK
    say(f"  Drafting a profile for {industry!r} with {provider.name} ({provider.model})…")
    draft = draft_profile(industry, brand, provider)
    if draft is None:
        say(f"  The LLM could not draft a usable profile, so {brand} uses {FALLBACK}.")
        return FALLBACK
    while True:
        say("\n  Draft profile (review the categories, hypotheses and features):\n" + draft.to_yaml())
        choice = ask("  [a]ccept and save to profiles.yaml / [e]dit / [r]eject (use consumer_product): ").strip().lower()[:1]
        if choice == "a":
            try:
                save_profile(draft)
            except ValueError as exc:
                say(f"  Not saved: {exc}")
                continue
            say(f"  Saved profile {draft.name} to profiles.yaml.")
            return draft.name
        if choice == "e":
            try:
                draft = draft_from_yaml(edit(draft.to_yaml().replace("\n  ", "\n").lstrip()), industry)
            except Exception as exc:  # noqa: BLE001 — show the problem and offer the draft again
                say(f"  Could not read the edited profile: {exc}")
            continue
        if choice == "r":
            say(f"  Rejected; {brand} uses {FALLBACK}.")
            return FALLBACK


def add_brand(path: Path, ask: Any = input, say: Any = print, edit: Any = None, provider: Any = None) -> BrandEntry:
    """Ask for a new brand, validate it together with the existing ones, then append it to brands.yaml."""
    edit = edit or edit_in_editor
    from nlp.profiles import get_profile, profile_names

    entries = load_yaml(path)
    groups = {e.group: e for e in entries if e.is_own}
    names = {e.name.lower() for e in entries}

    while True:
        name = ask("Brand name: ").strip()
        if name and name.lower() not in names:
            break
        say("  Needs a name that is not in brands.yaml yet.")
    say(f"  Groups so far: {', '.join(groups) or 'none'}")
    group = ask("Group (an existing one, or a new name): ").strip() or name
    if group in groups:
        own = groups[group]
        profile = own.profile
        is_own = False
        say(f"  {group} uses profile {profile}; {own.name} is its own brand, so {name} is a competitor.")
    else:
        say(f"  New group: {name} will be its own brand. Profiles: {', '.join(profile_names())}")
        say("  Or type the industry in your own words (e.g. 'pet food') to draft a new profile with the LLM.")
        answer = ask("Profile or industry [consumer_product]: ").strip() or "consumer_product"
        profile = answer if answer in profile_names() else choose_new_profile(answer, name, ask, say, edit, provider)
        is_own = True

    say("  Keywords: how people write the name. Capitals = case-sensitive whole word; lowercase = any case.")
    keywords = _ask_list(ask, "Keywords, comma-separated", [name])
    say("  If the name is also an everyday word or another thing (Apsara, Natraj, DOMS, Centella), require a")
    say("  context word too, e.g. pencil, eraser, stationery or skin, serum, toner. '-' for none.")
    require_context = _ask_list(ask, "Context words (blank = none)")
    exclude = _ask_list(ask, "Phrases to remove before matching, e.g. 'Apsara Hotel' (blank = none)")
    exclude_if = _ask_list(ask, "Words that drop a mention, e.g. 'temple' (blank = none)")
    youtube_queries = _ask_list(ask, "YouTube search queries", [f"{name} review"])
    defaults = list(get_profile(profile).sources)
    say(f"  Sources for {profile}: {', '.join(defaults)} (known: {', '.join(ALL_SOURCES)})")
    chosen = _ask_list(ask, "Sources (blank = these defaults)")
    app_store_id = play_store_id = None
    if profile == "software":
        app_store_id = ask("App Store id (digits, blank = none): ").strip() or None
        play_store_id = ask("Google Play id (e.g. com.example.app, blank = none): ").strip() or None

    entry = BrandEntry(name=name, is_own=is_own, keywords=keywords, exclude=exclude, exclude_if=exclude_if,
                       group=group, profile=profile, require_context=require_context,
                       youtube_queries=youtube_queries, sources=chosen or None,
                       app_store_id=app_store_id, play_store_id=play_store_id)
    block = append_entries(path, [entry])
    say(f"\nAdded to {path.name}:{block}")
    return entry


def _write_checked(path: Path, text: str) -> None:
    """Replace the file only if the whole result still loads (one own brand per group, known profile…)."""
    tmp = path.with_suffix(".yaml.new")
    tmp.write_text(text, encoding="utf-8")
    try:
        load_yaml(tmp)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(path)


def append_entries(path: Path, entries: Sequence[BrandEntry], comment: str | None = None) -> str:
    """Append brands to brands.yaml (validated together with the existing ones); returns the added text.
    Used by `add` and by the collector when a brand tracked from the dashboard is approved."""
    block = (f"\n  # {comment}" if comment else "") + "".join(_yaml_block(e) for e in entries)
    _write_checked(path, path.read_text(encoding="utf-8").rstrip("\n") + "\n" + block)
    return block


def remove_group_entries(path: Path, group: str) -> list[str]:
    """Delete every brand of `group` from brands.yaml, with the comment lines right above each block (the
    "# added from the dashboard" line). Other text is kept as it is. Returns the removed brand names."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if re.match(r"  - name:", line)]
    if not starts:
        return []
    ends = starts[1:] + [len(lines)]
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    blocks = data.get("brands") or []
    if len(blocks) != len(starts):
        raise ValueError(f"{path.name}: could not line up its brand blocks; remove {group!r} by hand")
    drop: set[int] = set()
    removed: list[str] = []
    for raw, start, end in zip(blocks, starts, ends):
        if str(raw.get("group") or "Default").strip() != group:
            continue
        removed.append(str(raw["name"]))
        first = start
        while first > 0 and lines[first - 1].strip().startswith("#") and lines[first - 1].startswith("  #"):
            first -= 1
        while first > 0 and not lines[first - 1].strip():  # the blank line before the block
            first -= 1
        drop.update(range(first, end))
    if removed:
        kept = "".join(line for i, line in enumerate(lines) if i not in drop)
        _write_checked(path, kept.rstrip("\n") + "\n")
    return removed


def add_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="manage_brands.py add", description=add_brand.__doc__)
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE)
    args = parser.parse_args(argv)
    try:
        entry = add_brand(args.file)
    except (ValueError, yaml.YAMLError) as exc:
        print(f"✘ {exc}\nNothing was written.")
        return 1
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled; nothing was written.")
        return 1
    from db.session import get_session

    with get_session() as session:
        report = sync_brands(session, load_yaml(args.file))
    print(f"Synced: added {', '.join(report['added']) or 'nothing'}.")
    print(f"Next: python manage_brands.py dry-run --brand \"{entry.name}\"  (check matching), then python fetch.py --brand \"{entry.name}\"")
    return 0


def list_brands(session: Session) -> None:
    for b in session.scalars(select(Brand).order_by(Brand.is_own.desc(), Brand.name)):
        own = "own" if b.is_own else "competitor"
        print(f"{b.id:>3}  {b.name:<12} {own:<10}  app_store={b.app_store_id or '-':<11} "
              f"play={b.play_store_id or '-'}")
        print(f"     keywords={b.keywords}")
        print(f"     features={list(b.features or [])}")


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "add":
        return add_main(argv[1:])
    if argv and argv[0] == "dry-run":
        return dry_run_main(argv[1:])
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
