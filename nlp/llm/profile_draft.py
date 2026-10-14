"""Draft an industry profile with the LLM when a brand is added in a field no profile covers ("pet food",
"running shoes"). The draft is validated like profiles.yaml, shown for review, and only saved when accepted
(save_profile). Without an LLM, or when the draft is unusable, callers fall back to consumer_product.
Reusable: manage_brands.py add uses it now; Step 11 (track any brand from the dashboard) will too.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from nlp.llm.provider import LLMError, Provider
from nlp.llm.runlog import StageRun
from nlp.profiles import PROFILES_FILE, clear_cache, load_profiles, parse_profile, profile_names

FALLBACK = "consumer_product"
DEFAULT_SOURCES = ["youtube", "news", "reddit", "mastodon"]
PRAISE, DISCUSSION, REQUEST = "praise", "general discussion", "product request"

SYSTEM = (
    "You design the categories a brand-monitoring tool uses to classify public comments about products in "
    "one industry. A zero-shot model reads each hypothesis as 'This text is about {hypothesis}.', so write "
    "hypotheses as short descriptive noun phrases with concrete examples of what people complain about. "
    "Labels are short, lowercase, and specific to the industry (for skincare: 'skin reaction', 'texture', "
    "'scent'; for stationery: 'smudging', 'breakage/durability'). Features are the product aspects people "
    "talk about, each with lowercase keywords and synonyms people actually write."
)
SCHEMA = {"type": "object", "properties": {
    "name": {"type": "string", "description": "snake_case profile name, e.g. pet_food"},
    "complaint_categories": {"type": "array", "description": "5 to 8 complaint categories", "items": {
        "type": "object", "properties": {"label": {"type": "string"}, "hypothesis": {"type": "string"}}}},
    "praise_hypothesis": {"type": "string"},
    "request_hypothesis": {"type": "string", "description": "a wish for a new product, variant or size"},
    "features": {"type": "array", "description": "6 to 10 product aspects", "items": {
        "type": "object", "properties": {"name": {"type": "string"},
                                         "synonyms": {"type": "array", "items": {"type": "string"}}}}},
}}


@dataclass
class ProfileDraft:
    name: str
    raw: dict[str, Any]
    industry: str
    notes: list[str] = field(default_factory=list)

    def to_yaml(self) -> str:
        """The block that goes under `profiles:` in profiles.yaml, in the file's own style."""
        def q(value: Any) -> str:
            return yaml.safe_dump(value, default_flow_style=True, allow_unicode=True, width=1000).strip().removesuffix("...").strip()

        def flow(items: list[Any]) -> str:
            return "[" + ", ".join(q(i) for i in items) + "]"

        r = self.raw
        lines = [f"  {self.name}:", f"    version: {r.get('version', 1)}", "    categories:"]
        lines += [f"      {q(label)}: {q(text)}" for label, text in r["categories"].items()]
        lines += [f"    praise: {q(r['praise'])}", f"    discussion: {q(r['discussion'])}",
                  f"    complaints: {flow(r['complaints'])}",
                  f"    complaints_need_non_positive: {str(bool(r.get('complaints_need_non_positive', True))).lower()}"]
        if r.get("request"):
            lines.append(f"    request: {q(r['request'])}")
        lines.append("    features:")
        lines += [f"      {q(name)}: {flow(syn)}" for name, syn in r["features"].items()]
        lines.append(f"    sources: {flow(r.get('sources') or DEFAULT_SOURCES)}")
        return "\n".join(lines) + "\n"


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:32] or "custom"


def to_raw(data: dict[str, Any]) -> dict[str, Any]:
    """The LLM's answer -> a profiles.yaml mapping (praise, discussion and request labels are fixed)."""
    complaints: dict[str, str] = {}
    for item in data.get("complaint_categories") or []:
        label = " ".join(str(item.get("label", "")).lower().split())[:32]
        hypothesis = " ".join(str(item.get("hypothesis", "")).split())
        if label and hypothesis and label not in (PRAISE, DISCUSSION, REQUEST) and label not in complaints:
            complaints[label] = hypothesis
    categories = dict(complaints)
    categories[REQUEST] = " ".join(str(data.get("request_hypothesis") or "a wish for a new product, variant or size").split())
    categories[PRAISE] = " ".join(str(data.get("praise_hypothesis") or "praise or a compliment").split())
    categories[DISCUSSION] = "a general discussion, not a complaint or praise"
    features: dict[str, list[str]] = {}
    for item in data.get("features") or []:
        name = " ".join(str(item.get("name", "")).lower().split())[:32]
        synonyms = list(dict.fromkeys(" ".join(str(x).lower().split()) for x in item.get("synonyms") or [] if str(x).strip()))
        if name and name not in features:
            features[name] = [s for s in synonyms if s != name][:12]
    return {"version": 1, "categories": categories, "praise": PRAISE, "discussion": DISCUSSION,
            "complaints": list(complaints), "complaints_need_non_positive": True, "request": REQUEST,
            "features": features, "sources": list(DEFAULT_SOURCES)}


def check_draft(name: str, raw: dict[str, Any], existing: list[str]) -> list[str]:
    """Problems that make the draft unusable (empty list = fine)."""
    problems = []
    if not re.fullmatch(r"[a-z][a-z0-9_]{2,31}", name):
        problems.append(f"name {name!r} is not snake_case")
    if name in existing:
        problems.append(f"a profile named {name!r} already exists")
    if not 4 <= len(raw["complaints"]) <= 10:
        problems.append(f"{len(raw['complaints'])} complaint categories (need 4-10)")
    if any(len(h.split()) < 3 for h in raw["categories"].values()):
        problems.append("a hypothesis is shorter than 3 words")
    if not 4 <= len(raw["features"]) <= 14 or any(len(s) < 2 for s in raw["features"].values()):
        problems.append("features need 4-14 entries with at least 2 synonyms each")
    try:
        parse_profile(name, raw)
    except ValueError as exc:
        problems.append(str(exc))
    return problems


def draft_profile(industry: str, brand: str, provider: Provider, examples: list[str] | None = None) -> ProfileDraft | None:
    """Ask the LLM for a profile for `industry`; None if there is no LLM or the draft is unusable."""
    prompt = (f"Industry: {industry}\nExample brand: {brand}\n"
              + (f"Example comments:\n" + "\n".join(f"- {e}" for e in examples[:8]) + "\n" if examples else "")
              + "Draft the complaint categories, praise and request hypotheses, and features for this industry.")
    run = StageRun.start("profile", provider)
    run.items = 1
    try:
        result = provider.json(SYSTEM, prompt, SCHEMA, max_tokens=1500)
    except LLMError as exc:
        run.fallback(str(exc)[:200])
        run.finish(None)
        return None
    finally:
        provider.unload()
    run.add(result)
    name = slug(str(result.data.get("name") or industry))
    if name in profile_names():
        name = slug(f"{name}_{slug(industry)}")
    raw = to_raw(result.data)
    problems = check_draft(name, raw, profile_names())
    if problems:
        run.fallback("; ".join(problems)[:300])
        run.finish(None)
        return None
    run.generated = 1
    run.finish(None)
    return ProfileDraft(name=name, raw=raw, industry=industry)


def save_profile(draft: ProfileDraft, path: Path = PROFILES_FILE) -> None:
    """Append the accepted draft to profiles.yaml; the whole file must still load."""
    problems = check_draft(draft.name, draft.raw, list(load_profiles(path)))
    if problems:
        raise ValueError("; ".join(problems))
    current = path.read_text(encoding="utf-8")
    block = f"\n  # drafted by the LLM for \"{draft.industry}\" and reviewed when the brand was added\n" + draft.to_yaml()
    tmp = path.with_suffix(".yaml.new")
    tmp.write_text(current.rstrip("\n") + "\n" + block, encoding="utf-8")
    try:
        load_profiles(tmp)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(path)
    clear_cache()


def draft_from_yaml(text: str, industry: str) -> ProfileDraft:
    """Read an edited draft back (the review step lets you change it in your editor)."""
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict) or len(data) != 1:
        raise ValueError("expected exactly one profile: name -> settings")
    name, raw = next(iter(data.items()))
    return ProfileDraft(name=str(name), raw=dict(raw), industry=industry)
