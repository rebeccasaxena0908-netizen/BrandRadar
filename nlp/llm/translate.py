"""Translate the mentions the English models skip (Bengali, Vietnamese, Hindi, Hinglish, Korean …) into
English, so they get the normal analysis. The original stays in `mentions.text`; the translation goes in
`mentions.text_en`, and the mention's analysis row (stored with NULL labels) is deleted so the next NLP run
analyses the translation. The LLM also flags adverts (affiliate "buy here" posts): those are marked
'advert' and stay out of analysis and topics. A translation that is empty, unchanged or not English is
'failed' and the mention stays skipped (retried with --force).
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from config import settings
from db.models import Analysis, Brand, Mention
from nlp.language import ANALYSED, LANGUAGE_NAMES, detect_language
from nlp.llm.provider import LLMError, LLMUnavailable, Provider
from nlp.llm.runlog import StageRun
from sources.base import utc_now

log = logging.getLogger("nlp.llm")

MAX_CHARS = 1500  # long posts are cut at a sentence boundary before translating

SYSTEM = (
    "You translate short public comments about consumer products and apps into plain English. Keep the meaning, "
    "tone, product names, prices and numbers; do not add anything that is not in the text. Many comments are "
    "romanised Hindi or Bengali (Hindi or Bengali written in Latin letters, e.g. 'apu', 'bhai', 'kya', 'kemon', "
    "'korbo', 'hai'), often mixed with English; the language label you get is only a guess. If you cannot tell "
    "what the text says, set confidence to low instead of guessing. is_advert is true only for a seller's or "
    "affiliate's promotion (prices, 'order here', shop or buy links); a person complaining about ads, or asking "
    "where to buy, is NOT an advert."
)
SCHEMA = {"type": "object", "properties": {
    "translation": {"type": "string", "description": "the English translation"},
    "confidence": {"type": "string", "enum": ["high", "medium", "low"], "description": "how sure you are of the meaning"},
    "is_advert": {"type": "boolean", "description": "true only for seller/affiliate promotion"},
}}
ROMANISED_HINT = ("probably romanised Hindi or Bengali (South Asian words in Latin letters), possibly mixed with "
                  "English; the detector guessed {name}")
LATIN = re.compile(r"[A-Za-z]")


@dataclass
class TranslateOutcome:
    translated: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    adverts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    failed: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    run: StageRun | None = None

    @property
    def total(self) -> int:
        return sum(self.translated.values())


def clip_text(text: str, limit: int = MAX_CHARS) -> str:
    text = re.sub(r"(?:https?://|www\.)\S+", "[link]", " ".join(text.split()))
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "), cut.rfind("। "))
    return cut[: end + 1] if end > limit // 2 else cut


def language_hint(text: str, language: str | None) -> str:
    """The detector is reliable for scripts (Bengali, Hangul…) but not for short Latin text, where most
    'Spanish/Dutch/German' guesses in our data are romanised Hindi or Bengali."""
    name = LANGUAGE_NAMES.get(language or "", language or "unknown")
    latin = len(LATIN.findall(text)) >= 0.5 * max(1, len(text.replace(" ", "")))
    if latin and language not in ("en", "fr", "pt", "it") and len(text.split()) < 25:
        return ROMANISED_HINT.format(name=name)
    return name


def build_prompt(text: str, language: str | None) -> str:
    return f"Language: {language_hint(text, language)}\nText:\n{clip_text(text)}"


def check_translation(original: str, translation: str) -> str | None:
    """Why a translation is unusable, or None. It must be non-empty, differ from the original and read as
    English (short ones can be 'und', which the English models also handle)."""
    t = " ".join((translation or "").split())
    if not t:
        return "empty"
    if t.lower() == " ".join(original.split()).lower():
        return "unchanged"
    if len(t) > 4 * len(original) + 200:
        return "much longer than the original"
    if detect_language(t) not in ANALYSED:
        return f"not English ({detect_language(t)})"
    return None


def pending(session: Session, brands: list[int] | None, force: bool, limit: int,
            redo: bool = False) -> list[tuple[Mention, str]]:
    stmt = (select(Mention, Brand.name).join(Brand, Brand.id == Mention.brand_id)
            .where(Mention.language.is_not(None), Mention.language.not_in(ANALYSED)))
    if redo:  # everything again (after a prompt or model change)
        pass
    elif force:
        stmt = stmt.where(Mention.text_en.is_(None),
                          Mention.translation_status.is_(None) | (Mention.translation_status == "failed"))
    else:
        stmt = stmt.where(Mention.text_en.is_(None), Mention.translation_status.is_(None))
    if brands is not None:
        stmt = stmt.where(Mention.brand_id.in_(brands))
    return list(session.execute(stmt.order_by(Mention.created_at.desc()).limit(limit)).tuples())


def fail(session: Session, mention: Mention, provider: Provider) -> None:
    if mention.translation_status == "ok":  # a redo that failed: back to "not analysed"
        session.execute(delete(Analysis).where(Analysis.mention_id == mention.id))
    mention.translation_status, mention.text_en = "failed", None
    mention.translated_by, mention.translated_at = f"{provider.name}:{provider.model}", utc_now()


def translate_pending(factory: sessionmaker[Session], provider: Provider, brands: list[int] | None = None,
                      force: bool = False, limit: int | None = None, record: bool = True,
                      redo: bool = False) -> TranslateOutcome:
    """Translate up to LLM_MAX_TRANSLATIONS waiting mentions (newest first). force retries failed ones;
    redo translates already translated ones again (their analysis is redone from the new translation)."""
    outcome = TranslateOutcome()
    run = outcome.run = StageRun.start("translate", provider)
    with factory() as session:
        rows = pending(session, brands, force, limit or (10_000 if redo else settings.llm_max_translations), redo)
        run.items = len(rows)
        for mention, brand in rows:
            try:
                result = provider.json(SYSTEM, build_prompt(mention.text, mention.language), SCHEMA, max_tokens=800)
            except LLMUnavailable as exc:
                run.fallback(str(exc))
                run.fallbacks += len(rows) - run.generated - run.fallbacks  # the rest stays skipped too
                break
            except LLMError as exc:
                run.fallback(str(exc)[:200])
                fail(session, mention, provider)
                outcome.failed[brand] += 1
                session.commit()
                continue
            run.add(result)
            translation = " ".join(str(result.data.get("translation", "")).split())
            problem = check_translation(mention.text, translation)
            if problem is None and str(result.data.get("confidence", "")).lower() == "low":
                problem = "low confidence"  # better skipped than analysed as something it doesn't say
            mention.translated_by, mention.translated_at = f"{provider.name}:{provider.model}", utc_now()
            if bool(result.data.get("is_advert")):
                if mention.translation_status == "ok":  # redo: its analysis came from the translation
                    session.execute(delete(Analysis).where(Analysis.mention_id == mention.id))
                mention.translation_status, mention.text_en = "advert", translation or None
                outcome.adverts[brand] += 1
                run.generated += 1
            elif problem:
                fail(session, mention, provider)
                outcome.failed[brand] += 1
                run.fallback(f"translation {problem}")
            else:
                mention.translation_status, mention.text_en = "ok", translation
                # the stored analysis row has NULL labels (skipped language): drop it so NLP analyses the translation
                session.execute(delete(Analysis).where(Analysis.mention_id == mention.id))
                outcome.translated[brand] += 1
                run.generated += 1
            session.commit()
    provider.unload()
    run.finish(factory if record else None)
    return outcome
