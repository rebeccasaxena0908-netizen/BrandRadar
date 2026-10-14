"""From a brand name to a proposed group: industry profile, market name, 3 competitors, matching rules, app
ids and sources (`propose`); checks for a proposed or edited setup (`problems`); the dry-run preview shown
in the review (`preview`); and the brands.yaml entries written on Approve (`entries`).

A setup is plain JSON (stored in brand_requests.setup and edited in the dashboard):

    {"industry": "fast food", "group": "Fast food chains", "profile": "fast_food",
     "profile_draft": {"name": ..., "industry": ..., "raw": {...}} or null,   # new profile, saved on Approve
     "needs_input": false, "llm": "ollama:gemma3:4b", "notes": [...],
     "brands": [{"name": "McDonald's", "is_own": true, "keywords": [...], "exclude": [...], "exclude_if": [...],
                 "require_context": [...], "youtube_queries": [...], "other_meanings": [...], "rules": "llm",
                 "app_store_id": "922103212", "app_store_title": "McDonald's · McDonald's USA, LLC",
                 "play_store_id": null, "play_store_title": null}, ...]}
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime
from typing import Any

from manage_brands import DRY_RUN_SOURCES, BrandEntry, _spread
from nlp.llm.profile_draft import FALLBACK, ProfileDraft, check_draft, draft_profile
from nlp.llm.provider import LLMUnavailable, Provider
from nlp.llm.research import Rules, fallback_rules, key, research_market, research_rules
from nlp.llm.runlog import StageRun
from nlp.profiles import Profile, get_profile, load_profiles, parse_profile
from tracking.apps import find_app_store, find_play_store

log = logging.getLogger(__name__)

SHOW = 10  # matched / rejected sample texts in the review
MAX_COMPETITORS = 5
GENERIC_PROFILES = (FALLBACK,)  # never offered to the LLM: it would pick it instead of drafting a real one


# ---------------------------------------------------------------- profiles


def profile_summaries(profiles: dict[str, Profile] | None = None) -> dict[str, str]:
    """name -> what it covers, for the LLM to pick from (its complaint categories say the most)."""
    profiles = profiles or load_profiles()
    return {name: ", ".join(p.complaints[:7]) for name, p in profiles.items() if name not in GENERIC_PROFILES}


def setup_profile(setup: dict[str, Any]) -> Profile | None:
    """The setup's profile: an existing one, or the (unsaved) draft."""
    draft = setup.get("profile_draft")
    if draft and draft.get("name") == setup.get("profile"):
        return parse_profile(draft["name"], draft["raw"])
    if setup.get("profile") in load_profiles():
        return get_profile(setup["profile"])
    return None


def industry_words(profile: Profile | None) -> set[str]:
    """Words describing the industry's own products: never valid as "drop the mention" words."""
    if profile is None:
        return set()
    words = set(profile.features) | set(profile.complaints)
    for synonyms in profile.features.values():
        words |= set(synonyms)
    return {w.lower() for w in words}


# ---------------------------------------------------------------- propose


def brand_setup(name: str, is_own: bool, rules: Rules) -> dict[str, Any]:
    return {"name": name, "is_own": is_own, "keywords": rules.keywords, "exclude": rules.exclude,
            "exclude_if": rules.exclude_if, "require_context": rules.require_context,
            "youtube_queries": rules.youtube_queries or [f"{name} review"], "other_meanings": rules.other_meanings,
            "rules": rules.source, "has_app": rules.has_app, "app_store_id": None, "app_store_title": None,
            "play_store_id": None, "play_store_title": None}


def add_apps(brand: dict[str, Any], app_store: Callable[[str], Any] = find_app_store,
             play_store: Callable[[str], Any] = find_play_store) -> None:
    """Store ids by name lookup, unless the LLM said the brand has no app of its own (a name match alone
    can't tell Dove's app from "Dove Medical Press")."""
    if brand.get("has_app") is False:
        return
    for store, finder in (("app_store", app_store), ("play_store", play_store)):
        match = finder(brand["name"])
        if match is not None:
            brand[f"{store}_id"], brand[f"{store}_title"] = match.app_id, f"{match.title} · {match.developer}"


def unique_group(name: str, taken: set[str]) -> str:
    out, n = name, 2
    while out.lower() in {t.lower() for t in taken}:
        out, n = f"{name} ({n})", n + 1
    return out


def propose(name: str, provider: Provider, tracked: Sequence[str], groups: Sequence[str],
            app_store: Callable[[str], Any] = find_app_store, play_store: Callable[[str], Any] = find_play_store,
            factory: Any = None) -> dict[str, Any]:
    """Research `name` and return a proposed setup. Without a usable LLM the setup has needs_input set:
    the review step then asks for the industry profile and the competitors."""
    tracked_keys = {key(t) for t in tracked}
    notes: list[str] = []
    llm = provider.name != "none"
    run = StageRun.start("research", provider)
    market = None
    try:
        if llm:
            run.items += 1
            market = research_market(name, provider, profile_summaries(), tracked_keys, run)
    except LLMUnavailable as exc:
        run.fallback(str(exc))
        llm = False
    if not llm:
        notes.append("No LLM available: pick the industry profile and type the competitors, then re-check.")

    setup: dict[str, Any] = {"industry": None, "group": None, "profile": None, "profile_draft": None,
                             "needs_input": True, "llm": f"{provider.name}:{provider.model}" if llm else "none",
                             "notes": notes, "brands": []}
    competitors: list[str] = []
    if market is not None:
        setup["industry"] = market.industry
        notes += market.notes
        competitors = market.competitors
        if market.profile:
            setup["profile"] = market.profile
        elif market.industry:
            draft = draft_profile(market.industry, name, provider)
            if draft is not None:
                setup["profile"], setup["profile_draft"] = draft.name, {
                    "name": draft.name, "industry": draft.industry, "raw": draft.raw}
                notes.append(f"New industry profile {draft.name} drafted for {market.industry!r}; it is saved on Approve.")
            else:
                setup["profile"] = FALLBACK
                notes.append(f"Could not draft a profile for {market.industry!r}; using {FALLBACK}.")
        setup["group"] = unique_group(market.market or f"{name} and competitors", set(groups))
        if len(competitors) < 3:
            notes.append(f"Only {len(competitors)} competitor(s) found; add more in Edit if you like.")
        setup["needs_input"] = not competitors or setup["profile"] is None
    else:
        setup["group"] = unique_group(f"{name} and competitors", set(groups))
        if llm:
            notes.append("The LLM's answer about the market was unusable: pick the profile and competitors.")

    words = industry_words(setup_profile(setup))
    for brand_name, is_own in [(name, True), *((c, False) for c in competitors)]:
        try:
            rules = research_rules(brand_name, setup["industry"], provider, words, run) if llm else fallback_rules(brand_name)
        except LLMUnavailable as exc:
            run.fallback(str(exc))
            llm, rules = False, fallback_rules(brand_name)
        run.items += 1
        if rules.source == "llm":
            run.generated += 1
        setup["brands"].append(brand_setup(brand_name, is_own, rules))
        notes += [f"{brand_name}: {n}" for n in rules.notes if not n.startswith("no LLM")]
    provider.unload()
    run.finish(factory)  # an llm_runs row: time, tokens, fallbacks
    for brand in setup["brands"]:
        add_apps(brand, app_store, play_store)
    return setup


def fill_in(setup: dict[str, Any], provider: Provider, find_apps: bool = True,
            app_store: Callable[[str], Any] = find_app_store, play_store: Callable[[str], Any] = find_play_store) -> None:
    """After an edit: brands the person added (no keywords yet) get researched rules and app ids."""
    words = industry_words(setup_profile(setup))
    for brand in setup["brands"]:
        if brand.get("keywords"):
            continue
        try:
            rules = research_rules(brand["name"], setup.get("industry"), provider, words) \
                if provider.name != "none" else fallback_rules(brand["name"])
        except LLMUnavailable:
            rules = fallback_rules(brand["name"])
        brand.update({k: v for k, v in brand_setup(brand["name"], bool(brand.get("is_own")), rules).items()
                      if k not in ("app_store_id", "play_store_id", "app_store_title", "play_store_title")})
        if find_apps:
            add_apps(brand, app_store, play_store)
    provider.unload()
    setup["needs_input"] = bool(problems(setup, [], [], strict_input=True))


# ---------------------------------------------------------------- check and convert


def problems(setup: dict[str, Any], tracked: Sequence[str], groups: Sequence[str], strict_input: bool = True) -> list[str]:
    """Why this setup can't be approved yet (empty = fine). `tracked` and `groups` are what exists already."""
    out: list[str] = []
    group = " ".join(str(setup.get("group") or "").split())
    if not 2 <= len(group) <= 64:
        out.append("The group needs a name (2-64 characters).")
    elif group.lower() in {g.lower() for g in groups}:
        out.append(f"A group called {group} exists already.")
    profile = setup.get("profile")
    draft = setup.get("profile_draft")
    if not profile:
        out.append("Pick an industry profile.")
    elif draft and draft.get("name") == profile:
        out += [f"Draft profile: {p}" for p in check_draft(profile, draft["raw"], list(load_profiles()))]
    elif profile not in load_profiles():
        out.append(f"Unknown profile {profile}.")
    brands = setup.get("brands") or []
    if sum(bool(b.get("is_own")) for b in brands) != 1:
        out.append("Exactly one brand must be the own brand.")
    if strict_input and len(brands) < 2:
        out.append("Add at least one competitor.")
    seen: set[str] = set()
    tracked_keys = {key(t) for t in tracked}
    for b in brands:
        name = " ".join(str(b.get("name") or "").split())
        k = key(name)
        if not k:
            out.append("Every brand needs a name.")
            continue
        if k in seen:
            out.append(f"{name} is listed twice.")
        if k in tracked_keys:
            out.append(f"{name} is already tracked.")
        seen.add(k)
        if not b.get("keywords"):
            out.append(f"{name} needs at least one keyword (re-check fills them in).")
        if any(str(k_).startswith(("-", "!", "r/")) for k_ in b.get("keywords") or []):
            out.append(f"{name}: keywords can't start with - ! or r/.")
    return out


def entries(setup: dict[str, Any]) -> list[BrandEntry]:
    """The brands.yaml entries for an approved setup. A brand with an app also gets the store sources.
    Before a profile is picked (no LLM), the preview uses consumer_product's sources."""
    profile = setup_profile(setup) or get_profile(FALLBACK)
    out = []
    for b in setup["brands"]:
        stores = [s for s, id_key in (("appstore", "app_store_id"), ("playstore", "play_store_id"))
                  if b.get(id_key) and s not in profile.sources]
        sources = list(profile.sources) + stores if stores else None
        out.append(BrandEntry(
            name=" ".join(b["name"].split()), is_own=bool(b["is_own"]), keywords=list(b["keywords"]),
            exclude=list(b.get("exclude") or []), exclude_if=list(b.get("exclude_if") or []),
            group=" ".join(setup["group"].split()), profile=setup.get("profile") or FALLBACK,
            require_context=list(b.get("require_context") or []), youtube_queries=list(b.get("youtube_queries") or []),
            sources=sources, app_store_id=b.get("app_store_id") or None, play_store_id=b.get("play_store_id") or None,
        ))
    return out


def draft_of(setup: dict[str, Any]) -> ProfileDraft | None:
    d = setup.get("profile_draft")
    if d and d.get("name") == setup.get("profile") and d["name"] not in load_profiles():
        return ProfileDraft(name=d["name"], raw=d["raw"], industry=d.get("industry") or d["name"])
    return None


# ---------------------------------------------------------------- dry-run preview


def preview(setup: dict[str, Any], make_source: Callable[[str], Any], since: datetime, seconds: float,
            note: Callable[[str], None] = lambda _: None) -> dict[str, Any]:
    """Run the name-matched sources live for every brand (nothing stored) and keep counts plus sample texts
    of the own brand. YouTube only for the own brand and only its first query (1 search call: titles only);
    each source gets `seconds` at most."""
    from fetch import call_with_deadline
    from sources.base import brand_sources

    started = time.monotonic()
    profile = setup_profile(setup)
    out: dict[str, Any] = {"brands": {}, "matched": [], "rejected": [], "notes": []}
    for entry in entries(setup):
        spec = replace(entry.spec(), youtube_queries=tuple(entry.youtube_queries[:1]))
        wanted = set(entry.sources or (profile.sources if profile else brand_sources(spec)))
        names = [s for s in DRY_RUN_SOURCES if s in wanted and (entry.is_own or s != "youtube")]
        counts: dict[str, dict[str, int]] = {}
        matched: list[tuple[str, str]] = []
        rejected: list[tuple[str, str, str]] = []
        for source_name in names:
            source = make_source(source_name)
            if not source.enabled:
                if f"{data_name(source_name)}: not configured, skipped" not in out["notes"]:
                    out["notes"].append(f"{data_name(source_name)}: not configured, skipped")
                continue
            note(f"Checking {data_name(source_name)} for {entry.name}")
            try:
                texts = call_with_deadline(
                    (lambda s=source, b=spec: s.preview(b, since)) if hasattr(source, "preview")
                    else (lambda s=source, b=spec: [m.text for m in s.fetch(b, since)]), seconds, source_name)
            except Exception as exc:  # noqa: BLE001 — shown in the review, the other sources carry on
                out["notes"].append(f"{entry.name} · {data_name(source_name)}: {type(exc).__name__}: {str(exc)[:120]}")
                continue
            counts[source_name] = {"matched": len(texts), "rejected": len(source.rejected)}
            matched += [(source_name, t) for t in texts]
            rejected += [(source_name, reason, t) for t, reason in source.rejected]
            out["notes"] += [f"{entry.name} · {data_name(source_name)}: {short(w, 140)}" for w in source.warnings]
        out["brands"][entry.name] = counts
        if entry.is_own:
            out["matched"] = [[s, short(t)] for s, t in _spread(matched, SHOW)]
            out["rejected"] = [[s, r, short(t)] for s, r, t in _spread(rejected, SHOW)]
    out["seconds"] = round(time.monotonic() - started, 1)
    return out


def short(text: str, n: int = 220) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= n else flat[: n - 1] + "…"


def data_name(source: str) -> str:
    return {"hackernews": "Hacker News", "appstore": "App Store", "playstore": "Google Play", "devto": "dev.to",
            "mastodon": "Mastodon", "reddit": "Reddit", "youtube": "YouTube", "news": "News"}.get(source, source)
