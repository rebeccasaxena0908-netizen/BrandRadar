"""NLP pipeline: analyse every mention without an analysis row and write one row per mention.

    python -m nlp.pipeline                               # analyse new mentions
    python -m nlp.pipeline --brand Notion --limit 50
    python -m nlp.pipeline --reanalyse --brand Notion    # redo one brand
    python -m nlp.pipeline --reanalyse --model-version outdated   # redo rows from older models/pipeline
    python -m nlp.pipeline --reanalyse --category praise --sentiment negative   # redo matching rows only
    python -m nlp.pipeline --restamp OLD_VERSION         # mark rows a version change cannot affect as current
    python -m nlp.pipeline --redetect-language           # after changing nlp/language.py: store the new
                                                         # languages, re-analyse only mentions that change
    python -m nlp.pipeline --download                    # only download the models

--reanalyse deletes the matching analysis rows first; anything not re-analysed (e.g. a crash)
is simply picked up as "new" on the next run.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import delete, or_, select, update
from sqlalchemy.orm import Session, sessionmaker

from config import settings
from db.models import Analysis, Brand, Mention
from nlp.models import ALL_MODELS, EMOTION_MODEL, SENTIMENT_MODEL, ZERO_SHOT_MODEL, Analyzer, Scores
from nlp.features import Features, matched_features, normalise_features
from nlp.language import ANALYSED, detect_language
from nlp.profiles import SOFTWARE, Profile, get_profile
from nlp.text import clean_text

log = logging.getLogger("nlp.pipeline")

# Bump when labels, hypotheses, cleaning or thresholds change, so old rows count as outdated.
# 2: direct inference, bucketed padding, zero-shot truncated to 256 tokens
# 3: keyword-gated features with synonyms, 'general discussion' category
# 4: "praise" is not allowed with negative sentiment (next-best category is used)
PIPELINE_VERSION = 4

# Categories (label -> hypothesis) now come from the brand's industry profile (profiles.yaml).
# CATEGORY_HYPOTHESES is the software profile's, kept for callers and tests.
CATEGORY_HYPOTHESES: dict[str, str] = get_profile(SOFTWARE).categories
CATEGORY_TEMPLATE = "This text is about {}."
FEATURE_TEMPLATE = "This text talks about {}."


@dataclass
class PendingMention:
    id: int
    brand_id: int
    text: str  # cleaned
    profile: str = SOFTWARE
    language: str | None = "en"


@dataclass
class RunStats:
    device: str = ""
    model_version: str = ""
    selected: int = 0
    analysed: int = 0
    empty: int = 0
    language_skipped: int = 0
    skipped_by_brand: dict[int, int] = field(default_factory=dict)  # brand id -> non-English mentions skipped
    failed: int = 0
    deleted: int = 0
    seconds: float = 0.0
    failed_ids: list[int] = field(default_factory=list)


# ---------------------------------------------------------------- pure helpers


def build_model_version(hashes: dict[str, str], profile: str = SOFTWARE) -> str:
    """p<pipeline><profile tag>-s<sentiment>-e<emotion>-z<zero-shot>. The profile tag is empty for
    software, so software rows keep their version; other profiles add e.g. '.beauty_skincare1'."""
    def short(name: str) -> str:
        return hashes.get(name, "unknown")[:8]

    tag = get_profile(profile).version_tag
    return f"p{PIPELINE_VERSION}{tag}-s{short(SENTIMENT_MODEL)}-e{short(EMOTION_MODEL)}-z{short(ZERO_SHOT_MODEL)}"


def signed_sentiment(probs: Scores) -> float:
    """P(positive) - P(negative), in [-1, 1]."""
    return round(probs.get("positive", 0.0) - probs.get("negative", 0.0), 4)


def top_label(scores: Scores) -> str | None:
    return max(scores, key=scores.__getitem__) if scores else None


def pick_category(scores: Scores, sentiment: str | None, profile: Profile | None = None) -> str | None:
    """Best category label; the praise label is skipped for negative mentions ("used to be great, but...")."""
    profile = profile or get_profile(SOFTWARE)
    phrase_to_label = {v: k for k, v in profile.categories.items()}
    ranked = [phrase_to_label.get(p, p) for p in sorted(scores, key=scores.__getitem__, reverse=True)]
    for label in ranked:
        if not (label == profile.praise and sentiment == "negative"):
            return label
    return None


def best_feature(scores: Scores, threshold: float) -> str | None:
    label = top_label(scores)
    return label if label is not None and scores[label] >= threshold else None


def empty_row(mention_id: int, version: str) -> dict[str, Any]:
    """No labels: the text is empty after cleaning, or not in a language the models handle."""
    return {
        "mention_id": mention_id, "sentiment": None, "sentiment_score": None, "emotion": None,
        "emotion_scores": None, "category": None, "feature": None, "topic_id": None,
        "model_version": version,
    }


def analyse_batch(
    analyzer: Analyzer,
    items: Sequence[PendingMention],
    features_by_brand: dict[int, Features],
    threshold: float,
    version: str | dict[str, str],
) -> list[dict[str, Any]]:
    """Run all models on one batch and return analysis rows (not yet written). Categories use each
    mention's industry profile; `version` is one model_version or a profile -> model_version map."""
    texts = [m.text for m in items]
    sentiments = analyzer.sentiment(texts)
    emotions = analyzer.emotion(texts)
    categories: list[Scores] = [{} for _ in items]
    by_profile: dict[str, list[int]] = defaultdict(list)
    for i, m in enumerate(items):
        by_profile[m.profile].append(i)
    for profile_name, indexes in by_profile.items():
        hypotheses = list(get_profile(profile_name).categories.values())
        scores = analyzer.zero_shot([texts[i] for i in indexes], hypotheses, CATEGORY_TEMPLATE, False)
        for i, s in zip(indexes, scores):
            categories[i] = s

    # Features: keyword gate first, then zero-shot confirms and picks among the matched candidates.
    # Mentions with the same candidate set share one zero-shot call.
    features: dict[int, str | None] = {}
    by_candidates: dict[tuple[str, ...], list[PendingMention]] = defaultdict(list)
    for m in items:
        candidates = matched_features(m.text, features_by_brand.get(m.brand_id, {}))
        if candidates:
            by_candidates[tuple(candidates)].append(m)
    for candidates, group in by_candidates.items():
        scores = analyzer.zero_shot([m.text for m in group], list(candidates), FEATURE_TEMPLATE, True)
        for m, s in zip(group, scores):
            features[m.id] = best_feature(s, threshold)

    rows = []
    for m, sent, emo, cat in zip(items, sentiments, emotions, categories):
        sentiment = top_label(sent)
        rows.append({
            "mention_id": m.id,
            "sentiment": sentiment,
            "sentiment_score": signed_sentiment(sent),
            "emotion": top_label(emo),
            "emotion_scores": {k: round(v, 4) for k, v in sorted(emo.items())},
            "category": pick_category(cat, sentiment, get_profile(m.profile)),
            "feature": features.get(m.id),
            "topic_id": None,
            "model_version": version if isinstance(version, str) else version[m.profile],
        })
    return rows


# ---------------------------------------------------------------- database


def brand_ids(session: Session, names: Sequence[str] | None) -> list[int] | None:
    if not names:
        return None
    rows = session.execute(select(Brand.id, Brand.name)).all()
    wanted = {n.lower() for n in names}
    found = {name.lower(): bid for bid, name in rows if name.lower() in wanted}
    unknown = wanted - set(found)
    if unknown:
        raise SystemExit(f"Unknown brand(s): {', '.join(sorted(unknown))}")
    return list(found.values())


def delete_for_reanalysis(
    session: Session,
    brands: list[int] | None,
    version_filter: str | None,
    current_version: str,
    category: str | None = None,
    sentiment: str | None = None,
) -> int:
    stmt = delete(Analysis)
    if category:
        stmt = stmt.where(Analysis.category == category)
    if sentiment:
        stmt = stmt.where(Analysis.sentiment == sentiment)
    if brands is not None:
        stmt = stmt.where(Analysis.mention_id.in_(select(Mention.id).where(Mention.brand_id.in_(brands))))
    if version_filter == "outdated":
        # current_version is the software one; other profiles have their own (see current_versions)
        current = current_versions(session, current_version)
        stale = [Mention.brand_id.in_(select(Brand.id).where(Brand.profile == p)) & (Analysis.model_version != v)
                 for p, v in current.items()]
        stmt = stmt.where(or_(Analysis.model_version.is_(None),
                              Analysis.mention_id.in_(select(Mention.id).where(or_(*stale)))))
    elif version_filter:
        stmt = stmt.where(Analysis.model_version == version_filter)
    return session.execute(stmt.execution_options(synchronize_session=False)).rowcount or 0


def current_versions(session: Session, software_version: str) -> dict[str, str]:
    """profile -> current model_version, for every profile in use (derived from the software one)."""
    suffix = software_version.split("-", 1)[1]
    profiles = set(session.scalars(select(Brand.profile))) | {SOFTWARE}
    return {p: f"p{PIPELINE_VERSION}{get_profile(p).version_tag}-{suffix}" for p in profiles}


def restamp(session: Session, old_version: str, new_version: str) -> int:
    """Relabel rows still at `old_version` as `new_version` without recomputing them.

    Only valid when the version change provably cannot alter those rows (e.g. v4's praise rule only
    affects praise+negative rows, which are re-analysed first with --category/--sentiment)."""
    stmt = update(Analysis).where(Analysis.model_version == old_version).values(model_version=new_version)
    return session.execute(stmt.execution_options(synchronize_session=False)).rowcount or 0


def redetect_languages(session: Session, brands: list[int] | None = None) -> tuple[int, int]:
    """Detect every mention's language again and store changes. A mention whose language moves into or
    out of ANALYSED loses its analysis row, so the next run analyses it (or stores it unlabelled); other
    changes (de -> nl) keep their row. Returns (languages changed, analysis rows deleted)."""
    stmt = select(Mention)
    if brands is not None:
        stmt = stmt.where(Mention.brand_id.in_(brands))
    changed = 0
    flipped: list[int] = []
    for mention in session.scalars(stmt):
        language = detect_language(mention.text)
        if language == mention.language:
            continue
        if (mention.language in ANALYSED) != (language in ANALYSED):
            flipped.append(mention.id)
        mention.language = language
        changed += 1
    deleted = 0
    for start in range(0, len(flipped), 500):
        chunk = flipped[start:start + 500]
        result = session.execute(delete(Analysis).where(Analysis.mention_id.in_(chunk)))
        deleted += int(result.rowcount or 0)
    session.flush()
    return changed, deleted


def select_pending(session: Session, brands: list[int] | None, limit: int | None) -> list[PendingMention]:
    stmt = (
        select(Mention.id, Mention.brand_id, Mention.text, Brand.profile, Mention.language, Mention.text_en,
               Mention.translation_status)
        .join(Brand, Brand.id == Mention.brand_id)
        .outerjoin(Analysis, Analysis.mention_id == Mention.id)
        .where(Analysis.mention_id.is_(None))
        .order_by(Mention.id)
    )
    if brands is not None:
        stmt = stmt.where(Mention.brand_id.in_(brands))
    if limit:
        stmt = stmt.limit(limit)
    pending: list[PendingMention] = []
    for i, b, t, p, lang, text_en, status in session.execute(stmt):
        if status == "ok" and text_en:  # translated by the LLM (nlp/llm/translate.py): analyse the English text
            pending.append(PendingMention(i, b, clean_text(text_en), p or SOFTWARE, "en"))
        else:
            pending.append(PendingMention(i, b, clean_text(t), p or SOFTWARE, lang if lang else detect_language(t)))
    return pending


def load_features(session: Session) -> dict[int, Features]:
    """Each brand's own features, else its profile's defaults."""
    return {bid: normalise_features(f or get_profile(p).features)
            for bid, f, p in session.execute(select(Brand.id, Brand.features, Brand.profile))}


def write_rows(factory: sessionmaker[Session], rows: list[dict[str, Any]]) -> None:
    with factory() as session:
        session.add_all(Analysis(**r) for r in rows)
        session.commit()


# ---------------------------------------------------------------- run


def run_pipeline(
    factory: sessionmaker[Session],
    analyzer: Analyzer,
    brands: Sequence[str] | None = None,
    limit: int | None = None,
    reanalyse: bool = False,
    model_version: str | None = None,
    category: str | None = None,
    sentiment: str | None = None,
    batch_size: int | None = None,
    threshold: float | None = None,
    progress: bool = False,
) -> RunStats:
    started = time.perf_counter()
    batch_size = batch_size or settings.nlp_batch_size
    threshold = settings.feature_threshold if threshold is None else threshold
    stats = RunStats(device=analyzer.device, model_version=build_model_version(analyzer.model_hashes()))

    with factory() as session:
        ids = brand_ids(session, brands)
        if reanalyse:
            stats.deleted = delete_for_reanalysis(
                session, ids, model_version, stats.model_version, category=category, sentiment=sentiment
            )
            session.commit()
        pending = select_pending(session, ids, limit)
        features_by_brand = load_features(session)
    stats.selected = len(pending)

    with factory() as session:
        versions = current_versions(session, stats.model_version)
        # mentions fetched before language detection existed: store what was detected
        detected = {m.id: m.language for m in pending}
        for mention in session.scalars(select(Mention).where(Mention.language.is_(None), Mention.id.in_(detected))):
            mention.language = detected[mention.id]
        session.commit()
    empty = [m for m in pending if not m.text]
    other_language = [m for m in pending if m.text and m.language not in ANALYSED]
    if empty or other_language:
        write_rows(factory, [empty_row(m.id, versions.get(m.profile, stats.model_version)) for m in empty + other_language])
        stats.empty = len(empty)
        stats.language_skipped = len(other_language)
        for m in other_language:
            stats.skipped_by_brand[m.brand_id] = stats.skipped_by_brand.get(m.brand_id, 0) + 1
    # Similar lengths in a batch means less padding, which is much faster.
    todo = sorted((m for m in pending if m.text and m.language in ANALYSED), key=lambda m: len(m.text))

    bar = None
    if progress and todo:
        from tqdm import tqdm

        bar = tqdm(total=len(todo), unit="mention", desc=f"analysing on {analyzer.device}")

    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        try:
            rows = analyse_batch(analyzer, batch, features_by_brand, threshold, versions)
        except Exception:  # noqa: BLE001 — isolate the bad mention, keep the rest
            log.exception("batch starting at mention %d failed; retrying one by one", batch[0].id)
            rows = []
            for m in batch:
                try:
                    rows += analyse_batch(analyzer, [m], features_by_brand, threshold, versions)
                except Exception as exc:  # noqa: BLE001
                    log.error("mention %d failed: %s: %s", m.id, type(exc).__name__, exc)
                    stats.failed += 1
                    stats.failed_ids.append(m.id)
        analyzer.release_memory()
        if rows:
            write_rows(factory, rows)
            stats.analysed += len(rows)
        if bar:
            bar.update(len(batch))
    if bar:
        bar.close()

    stats.seconds = time.perf_counter() - started
    return stats


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brand", action="append", help="only this brand (repeatable)")
    parser.add_argument("--limit", type=int, help="analyse at most N mentions")
    parser.add_argument("--reanalyse", action="store_true", help="delete matching analysis rows and redo them")
    parser.add_argument(
        "--model-version", metavar="V",
        help="with --reanalyse: only rows with this model_version, or 'outdated' for anything not current",
    )
    parser.add_argument("--category", help="with --reanalyse: only rows with this category")
    parser.add_argument("--sentiment", help="with --reanalyse: only rows with this sentiment")
    parser.add_argument(
        "--restamp", metavar="OLD_VERSION",
        help="set model_version of rows at OLD_VERSION to the current version, without recomputing",
    )
    parser.add_argument(
        "--redetect-language", action="store_true",
        help="detect languages again first; only mentions that become (un)analysable are re-analysed",
    )
    parser.add_argument("--batch-size", type=int, help=f"default {settings.nlp_batch_size}")
    parser.add_argument("--download", action="store_true", help="download/load the models and exit")
    parser.add_argument("-q", "--quiet", action="store_true", help="no progress bar")
    args = parser.parse_args(argv)
    if (args.model_version or args.category or args.sentiment) and not args.reanalyse:
        parser.error("--model-version/--category/--sentiment only make sense with --reanalyse")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("transformers", "httpx", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    from nlp.models import HFAnalyzer

    analyzer = HFAnalyzer(batch_size=args.batch_size)
    if args.download:
        t = time.perf_counter()
        analyzer.load_all()
        print(f"Models ready on {analyzer.device} in {time.perf_counter() - t:.0f}s: {', '.join(ALL_MODELS)}")
        print(f"model_version = {build_model_version(analyzer.model_hashes())}")
        return 0

    from db.session import get_session_factory

    if args.restamp:
        current = build_model_version(analyzer.model_hashes())
        with get_session_factory()() as session:
            n = restamp(session, args.restamp, current)
            session.commit()
        print(f"Restamped {n} rows: {args.restamp} -> {current}")
        return 0

    if args.redetect_language:
        with get_session_factory()() as session:
            changed, deleted = redetect_languages(session, brand_ids(session, args.brand))
            session.commit()
        print(f"Re-detected languages: {changed} changed, {deleted} analysis rows deleted for re-analysis")

    stats = run_pipeline(
        get_session_factory(), analyzer, brands=args.brand, limit=args.limit, reanalyse=args.reanalyse,
        model_version=args.model_version, category=args.category, sentiment=args.sentiment,
        batch_size=args.batch_size, progress=not args.quiet,
    )
    print(f"\nDevice:         {stats.device}")
    print(f"Model version:  {stats.model_version}")
    if args.reanalyse:
        print(f"Deleted:        {stats.deleted} old analysis rows")
    print(f"Selected:       {stats.selected} mentions without analysis")
    print(f"Analysed:       {stats.analysed}")
    print(f"Empty text:     {stats.empty} (stored with NULL labels)")
    print(f"Not English:    {stats.language_skipped} (stored with NULL labels, waiting for translation)")
    if stats.skipped_by_brand:
        from db.session import get_session_factory as _f

        with _f()() as session:
            names = dict(session.execute(select(Brand.id, Brand.name)).all())
        print("                " + ", ".join(f"{names.get(b, b)} {n}" for b, n in sorted(stats.skipped_by_brand.items())))
    print(f"Failed:         {stats.failed}" + (f"  ids: {stats.failed_ids[:20]}" if stats.failed else ""))
    rate = stats.analysed / stats.seconds if stats.seconds else 0
    print(f"Total time:     {stats.seconds:.1f}s ({rate:.1f} mentions/s)")
    return 1 if stats.failed else 0


if __name__ == "__main__":
    sys.exit(main())
