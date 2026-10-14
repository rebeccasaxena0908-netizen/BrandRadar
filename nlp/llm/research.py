"""Research a brand typed into the dashboard (Step 11): its industry, market and competitors, and how to
match it in public text (spellings, other meanings to exclude, context words for ambiguous names).

Two small LLM calls per brand, each answer checked field by field; anything missing or unusable falls back
to a rule (or, for the industry and competitors, to asking the person in the review step). Nothing here
fetches data or writes files: tracking/research.py turns the result into a proposed setup.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from nlp.llm.provider import LLMError, LLMUnavailable, Provider
from nlp.llm.runlog import StageRun

MARKET_SYSTEM = (
    "You help set up brand monitoring. Given a brand name, say which industry it is in, name its market in "
    "2 to 4 words as a plural group (e.g. 'Pizza chains', 'Electric cars', 'Food delivery apps', 'Body "
    "lotions'), list its best-known direct competitors in that market (other brands, never the brand "
    "itself, its owner or its own products), and say whether it has its own consumer mobile app. Pick the "
    "closest industry profile from the list, or 'none' if none fits well. If you do not recognise the brand, "
    "say so with known=false instead of guessing."
)
RULES_SYSTEM = (
    # no named examples: a 4B model copies them into unrelated brands ("Nikola" as another meaning of McDonald's)
    "You help a brand-monitoring tool find posts about one brand. Posts are matched by the brand's name, so "
    "list how people actually write it (spelling variants, no other words). Then say whether the name is "
    "ambiguous: most brand names are not. A name is ambiguous only if it is an ordinary word or the name of a "
    "famous person, place, animal or other product that people write about often. Only for an ambiguous "
    "name, and only about its real other meanings, give: phrases that contain the name but mean the other "
    "thing (they are cut out before matching), words that only appear in posts about the other meaning (the "
    "whole post is dropped; never words that could describe the brand's own products), and short context "
    "words about the brand's own products, one of which a post must contain. Leave those lists empty for a "
    "name that is not ambiguous. Also give one YouTube search query for reviews of the brand, and say "
    "whether the brand publishes its own consumer mobile app."
)

MARKET_SCHEMA_BASE = {"type": "object", "properties": {
    "known": {"type": "boolean"},
    "industry": {"type": "string", "description": "2 to 4 words, lowercase, e.g. 'fast food', 'electric cars'"},
    "market": {"type": "string", "description": "2 to 4 words, plural, Sentence case"},
    # lists are plain strings: in a JSON array gemma closes "Wendy’s" with a curly quote and never stops
    "competitors": {"type": "string", "description": "4 or 5 competitor brand names, separated by commas"},
    "has_app": {"type": "boolean"},
}}
RULES_SCHEMA = {"type": "object", "properties": {
    "spellings": {"type": "string", "description": "1 to 4 ways people write the name, separated by commas"},
    "ambiguous": {"type": "boolean", "description": "true if the name often means something else"},
    "other_meanings": {"type": "string", "description": "what else the name means, separated by semicolons"},
    "exclude_phrases": {"type": "string", "description": "separated by commas"},
    "drop_words": {"type": "string", "description": "separated by commas"},
    "context_words": {"type": "string", "description": "separated by commas"},
    "youtube_query": {"type": "string"},
    "has_app": {"type": "boolean", "description": "true if the brand publishes its own consumer mobile app"},
}}

MAX_SPELLINGS, MAX_EXCLUDE, MAX_DROP, MAX_CONTEXT = 4, 8, 12, 16


def key(text: str) -> str:
    """Spelling-insensitive key: "McDonald’s", "mcdonalds" and "Mc Donald's" all give "mcdonalds"."""
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", folded.lower())


def as_list(value: Any, separators: str = ",;\n") -> list[str]:
    """A list field: a JSON list, or a string split on the separators; quotes around items are dropped."""
    items = value if isinstance(value, list) else re.split(f"[{re.escape(separators)}]", str(value or ""))
    # the model decorates lists now and then: "[a, b]", "%[a]%", "+ a\n+ b", "1. a"
    junk = "\"'“”‘’[](){}%*• \t"
    out = []
    for item in items:
        t = re.sub(r"^\s*(?:[-+*•·–—]|\d{1,2}[.)])\s+", "", str(item)).strip(junk)
        if t:
            out.append(t)
    return out


def clean(text: Any, max_words: int = 6, max_len: int = 60) -> str | None:
    t = " ".join(str(text or "").replace("’", "'").split()).strip(" .,;:\"")
    if not t or len(t) > max_len or len(t.split()) > max_words or re.search(r"[<>{}\[\]|\\#*@]", t):
        return None
    return t


def spelling_variants(name: str) -> list[str]:
    """Rule-based spellings: as typed, without the apostrophe (McDonalds) and without spaces (KitKat)."""
    base = " ".join(name.replace("’", "'").split())
    out = [base, base.replace("'", ""), base.replace("'", "").replace(" ", "")]
    if " " in base:
        out.append(base.replace(" ", ""))
    return list(dict.fromkeys(v for v in out if v))


def with_curly_apostrophes(keywords: list[str]) -> list[str]:
    """Phones type ’ (McDonald’s) and matching is literal, so both apostrophes are listed."""
    return list(dict.fromkeys([*keywords, *(k.replace("'", "’") for k in keywords if "'" in k)]))


@dataclass
class Market:
    known: bool
    industry: str | None
    profile: str | None  # an existing profile, or None (draft one / ask)
    market: str | None
    competitors: list[str]
    has_app: bool
    notes: list[str] = field(default_factory=list)


@dataclass
class Rules:
    keywords: list[str]
    exclude: list[str] = field(default_factory=list)
    exclude_if: list[str] = field(default_factory=list)
    require_context: list[str] = field(default_factory=list)
    youtube_queries: list[str] = field(default_factory=list)
    other_meanings: list[str] = field(default_factory=list)
    has_app: bool | None = None  # None = unknown (no LLM): look the stores up anyway
    source: str = "rules"  # "llm" or "rules"
    notes: list[str] = field(default_factory=list)


def fallback_rules(name: str) -> Rules:
    return Rules(keywords=with_curly_apostrophes(spelling_variants(name)), youtube_queries=[f"{name} review"],
                 source="rules", notes=["no LLM: other meanings of the name were not checked, and keywords with "
                                        "capitals only match that exact case; read the rejected and matched samples"])


def check_market(data: dict[str, Any], brand: str, profiles: dict[str, str], tracked: set[str]) -> Market:
    notes: list[str] = []
    known = bool(data.get("known", True))
    industry = clean(data.get("industry"), 4, 40)
    industry = industry.lower() if industry else None
    profile = str(data.get("profile") or "none")
    if profile not in profiles:
        profile = "none"
    market = clean(data.get("market"), 4, 40)
    if market:
        market = market[0].upper() + market[1:]
    if market and key(brand) in key(market):
        notes.append(f"market name {market!r} named the brand; replaced")
        market = None
    competitors: list[str] = []
    seen = {key(brand)}
    for raw in as_list(data.get("competitors")):
        name = clean(raw, 4, 40)
        if not name:
            continue
        k = key(name)
        if not k or k in seen or key(brand) in k or k in key(brand):
            continue  # the brand itself, its products ("McDonald's app") or a duplicate
        seen.add(k)
        if k in tracked:
            notes.append(f"{name} is already tracked in another group, so it is left out")
            continue
        competitors.append(name)
    if not known:
        notes.append("the LLM does not recognise this brand; check the industry and competitors")
    return Market(known=known, industry=industry, profile=None if profile == "none" else profile,
                  market=market, competitors=competitors[:3], has_app=bool(data.get("has_app")), notes=notes)


def check_rules(data: dict[str, Any], name: str, industry_words: set[str]) -> Rules:
    """Keep only rules that do what they claim; everything else falls back to the rule-based defaults."""
    notes: list[str] = []
    name_key = key(name)
    keywords = spelling_variants(name)
    for raw in as_list(data.get("spellings")):
        s = clean(raw, 4, 40)
        if s and (key(s) == name_key or (len(name_key) >= 5 and name_key in key(s) and len(key(s)) <= len(name_key) + 2)):
            keywords.append(s)
    keywords = list(dict.fromkeys(keywords))[:MAX_SPELLINGS + 2]
    meanings = [m for m in (clean(x, 8, 80) for x in as_list(data.get("other_meanings"), ";\n")) if m]
    # Ambiguous only with a stated other meaning. Then the capitals stay (a capitalised keyword matches
    # case-sensitively: "Dove" the brand, not "a dove") and the filters below apply. Otherwise every keyword
    # matches any case (NIVEA, #mcdonalds) and no filter is used: a stray drop word would lose real posts.
    ambiguous = bool(data.get("ambiguous")) and bool(meanings)
    if not ambiguous:
        keywords = list(dict.fromkeys(k.lower() for k in keywords))
        if data.get("exclude_phrases") or data.get("drop_words") or data.get("context_words"):
            notes.append("not ambiguous, so no exclude, drop or context words")
        data = {**data, "exclude_phrases": [], "drop_words": [], "context_words": []}
        meanings = []
    keywords = with_curly_apostrophes(keywords)

    def mentions_name(phrase: str) -> bool:
        return name_key in key(phrase)

    exclude = []
    for raw in as_list(data.get("exclude_phrases")):
        p = clean(raw, 5, 50)
        if p and mentions_name(p) and key(p) != name_key and key(p) not in {key(k) for k in keywords}:
            exclude.append(p)  # must contain the name: it is cut out of the text before matching
    drop = []
    for raw in as_list(data.get("drop_words")):
        w = clean(raw, 3, 30)
        if not w:
            continue
        if mentions_name(w):
            notes.append(f"drop word {w!r} contains the name; left out")
            continue
        if w.lower() in industry_words:
            notes.append(f"drop word {w!r} describes the brand's own products; left out")
            continue
        drop.append(w)
    context: list[str] = []
    for raw in as_list(data.get("context_words")):
        w = clean(raw, 3, 30)
        if w and not mentions_name(w):
            context.append(w.lower())
    query = clean(data.get("youtube_query"), 6, 60)
    queries = [q for q in [f"{name} review", query] if q and mentions_name(q)]
    return Rules(keywords=keywords, exclude=list(dict.fromkeys(exclude))[:MAX_EXCLUDE],
                 exclude_if=list(dict.fromkeys(drop))[:MAX_DROP],
                 require_context=list(dict.fromkeys(context))[:MAX_CONTEXT],
                 youtube_queries=list(dict.fromkeys(queries))[:2], other_meanings=meanings[:4],
                 has_app=bool(data["has_app"]) if "has_app" in data else None, source="llm", notes=notes)


def research_market(name: str, provider: Provider, profiles: dict[str, str], tracked: set[str],
                    run: StageRun | None = None) -> Market | None:
    """None when there is no usable LLM answer (the review step then asks for the industry and competitors)."""
    schema = {**MARKET_SCHEMA_BASE, "properties": {
        **MARKET_SCHEMA_BASE["properties"],
        "profile": {"type": "string", "enum": [*profiles, "none"]},
    }}
    listing = "\n".join(f"- {p}: {desc}" for p, desc in profiles.items())
    prompt = f"Brand: {name}\n\nIndustry profiles:\n{listing}"
    try:
        result = provider.json(MARKET_SYSTEM, prompt, schema, max_tokens=400)
    except LLMUnavailable:
        raise
    except LLMError as exc:
        if run is not None:
            run.fallback(str(exc)[:200])
        return None
    if run is not None:
        run.add(result)
    return check_market(result.data, name, profiles, tracked)


def research_rules(name: str, industry: str | None, provider: Provider, industry_words: set[str],
                   run: StageRun | None = None) -> Rules:
    prompt = f"Brand: {name}\nIndustry: {industry or 'unknown'}"
    try:
        result = provider.json(RULES_SYSTEM, prompt, RULES_SCHEMA, max_tokens=500)
    except LLMUnavailable:
        raise
    except LLMError as exc:
        if run is not None:
            run.fallback(str(exc)[:200])
        return fallback_rules(name)
    if run is not None:
        run.add(result)
    return check_rules(result.data, name, industry_words)
