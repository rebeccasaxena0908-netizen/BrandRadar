"""LLM topic labels: a 2-4 word label and a one-sentence description per active topic, written from 8-10
of its mentions in the brand's industry language (its profile and complaint category).

Cached on the topic with the member ids it was written from (`llm_members`); regenerated only when the
member set changed by LLM_LABEL_REFRESH or more (Jaccard distance, default 20%), or when there is none.
Manual labels (topic_labels.yaml) stay shown; they get a description too. If the LLM is off, fails or
writes an unusable label, the topic keeps its previous LLM label if it has one, else the keyword label.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from config import settings
from db.models import Analysis, Brand, Mention, Topic
from nlp.llm.provider import LLMError, LLMUnavailable, Provider
from nlp.llm.runlog import StageRun
from nlp.profiles import get_profile
from sources.base import utc_now

SAMPLE = 10
MAX_MENTION_CHARS = 280

SYSTEM = (
    "You name themes for a brand-monitoring dashboard. Given customer mentions that a clustering model "
    "grouped together, write a short label and a one-sentence description of what they have in common. "
    "Most themes are complaints: then name the specific problem (not 'Issues' or 'Feedback'). If the "
    "mentions are mostly praise or general discussion, say so in the label instead of inventing a problem. "
    "Use the plain words a product manager in this industry would use. The label is 2 to 4 short words "
    "(under 35 characters, no '&'); the description is one sentence of at most 25 words. Never use the "
    "brand's name, quotes or emoji in the label."
)
SCHEMA = {"type": "object", "properties": {
    "label": {"type": "string", "description": "2 to 4 words, Title Case, the specific problem (or praise/discussion theme)"},
    "description": {"type": "string", "description": "one sentence, at most 25 words, what users report"},
}}
VAGUE = {"issues", "issue", "problems", "problem", "feedback", "complaints", "complaint", "concerns", "various", "misc"}


def jaccard_distance(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return 1 - len(sa & sb) / len(sa | sb)


def needs_label(topic: Topic, members: Sequence[int], refresh: float) -> bool:
    if not topic.llm_label or not topic.llm_members or not topic.description:
        return True
    return jaccard_distance(topic.llm_members, members) >= refresh


def sample_texts(rows: Sequence[tuple[int, str, float | None, object]], n: int = SAMPLE) -> list[str]:
    """Most negative first, then newest; duplicates and near-empty texts dropped."""
    ordered = sorted(rows, key=lambda r: (r[2] if r[2] is not None else 0.0, -r[3].timestamp() if hasattr(r[3], "timestamp") else 0))
    seen: set[str] = set()
    out: list[str] = []
    for _, text, _, _ in ordered:
        t = " ".join(str(text).split())
        key = t.lower()[:80]
        if len(t) < 12 or key in seen:
            continue
        seen.add(key)
        out.append(t if len(t) <= MAX_MENTION_CHARS else t[: MAX_MENTION_CHARS - 1].rsplit(" ", 1)[0] + "…")
        if len(out) == n:
            break
    return out


def build_prompt(brand: str, industry: str, category: str | None, keyword_label: str | None, texts: Sequence[str],
                 taken: Sequence[str] = ()) -> str:
    lines = "\n".join(f"{i}. {t}" for i, t in enumerate(texts, 1))
    others = (f"\nLabels of this brand's other themes (pick a different, more specific label): {', '.join(sorted(taken))}"
              if taken else "")
    return (f"Brand: {brand}\nIndustry: {industry}\nComplaint category: {category or 'mixed'}\n"
            f"Keyword label (automatic, often clumsy): {keyword_label or '-'}{others}\n\nMentions in this theme:\n{lines}")


def title(text: str) -> str:
    small = {"and", "or", "of", "in", "on", "to", "for", "with", "the", "a", "an", "at", "by"}
    words = text.split()
    def keep(w: str) -> bool:  # acronyms and brand-style words: "AI", "iOS", "iPad"
        return any(c.isupper() for c in w[1:])

    return " ".join(w if keep(w) else (w.lower() if i and w.lower() in small else w[:1].upper() + w[1:])
                    for i, w in enumerate(words))


def check_label(label: str, description: str, brand: str, taken: set[str]) -> str | None:
    """Why the LLM's label/description can't be used, or None."""
    words = re.findall(r"[^\W_][\w'/&+-]*", label)
    if not 2 <= len(words) <= 4:
        return f"label has {len(words)} words"
    if len(label) > 40 or re.search(r"[\"“”'`!?:;#*]|\d{3,}", label):
        return "label too long or has quotes/punctuation"
    if brand.lower() in label.lower():
        return "label names the brand"
    if all(w.lower() in VAGUE for w in words[-1:]) and len(words) <= 2:
        return "label too vague"
    if label.lower() in taken:
        return "label already used by another topic"
    d = " ".join(description.split())
    if not (20 <= len(d) <= 220) or len(re.findall(r"[.!?](?:\s|$)", d)) > 1:
        return "description is not one short sentence"
    return None


def refresh_labels(factory: sessionmaker[Session], provider: Provider, brands: list[int] | None = None,
                   force: bool = False, refresh: float | None = None, record: bool = True) -> StageRun:
    refresh = settings.llm_label_refresh if refresh is None else refresh
    from nlp.topics import load_label_overrides

    # topic_labels.yaml holds real-database topic ids: never apply it to the demo database
    overrides = {} if settings.demo else load_label_overrides()
    run = StageRun.start("labels", provider)
    available = True
    with factory() as session:
        brand_rows = list(session.scalars(select(Brand).order_by(Brand.id)))
        for brand in brand_rows:
            if brands is not None and brand.id not in brands:
                continue
            members: dict[int, list[tuple[int, str, float | None, object]]] = defaultdict(list)
            for mid, text, score, created, topic_id in session.execute(
                select(Mention.id, func.coalesce(Mention.text_en, Mention.text), Analysis.sentiment_score,
                       Mention.created_at, Analysis.topic_id)
                .join(Analysis, Analysis.mention_id == Mention.id)
                .where(Mention.brand_id == brand.id, Analysis.topic_id.is_not(None))
            ):
                members[topic_id].append((mid, text, score, created))
            topics = list(session.scalars(select(Topic).where(Topic.brand_id == brand.id, Topic.id.in_(list(members)))))
            industry = get_profile(brand.profile).name.replace("_", " ")
            for topic in topics:
                run.items += 1
                if topic.id in overrides:  # manual labels always win (also before the next topics run marks them)
                    topic.label, topic.label_source = overrides[topic.id][:255], "manual"
                elif topic.label_source == "manual":  # override removed from the file: back to llm / keyword
                    topic.label = topic.llm_label or topic.keyword_label or topic.label
                    topic.label_source = "llm" if topic.llm_label else "keyword"
                ids = sorted(m[0] for m in members[topic.id])
                if not force and not needs_label(topic, ids, refresh):
                    run.cached += 1
                    continue
                if not available:
                    run.fallbacks += 1
                    continue
                others = [t.label for t in topics if t.id != topic.id]
                taken = {label.lower() for label in others}
                prompt = build_prompt(brand.name, industry, topic.category, topic.keyword_label or topic.label,
                                      sample_texts(members[topic.id]), others)
                try:
                    result = provider.json(SYSTEM, prompt, SCHEMA, max_tokens=320)
                except LLMUnavailable as exc:
                    available = False
                    run.fallback(str(exc))
                    continue
                except LLMError as exc:
                    run.fallback(str(exc)[:200])
                    continue
                run.add(result)
                label = title(" ".join(str(result.data.get("label", "")).split()).strip(" .\"'"))
                description = " ".join(str(result.data.get("description", "")).split())
                problem = check_label(label, description, brand.name, taken)
                if problem:
                    run.fallback(f"label rejected: {problem}")
                    continue
                topic.llm_label, topic.description = label, description[0].upper() + description[1:]
                topic.llm_members, topic.llm_at = ids, utc_now()
                if topic.label_source != "manual":
                    topic.label, topic.label_source = label, "llm"
                run.generated += 1
                session.commit()
        session.commit()
    provider.unload()
    return run.finish(factory if record else None)
