"""Topic clustering of complaints per brand (BERTopic + all-MiniLM-L6-v2) and topic_daily aggregation.

    python -m nlp.topics                 # all brands
    python -m nlp.topics --brand Notion

Complaints = negative sentiment, or a complaint-like category. Re-runs are stable: each new cluster
takes over the existing topic most of its mentions already belonged to (member overlap), so topic
ids, alerts and history survive re-clustering. Topics that no longer match anything are kept with
no mentions (priority marks them inactive).
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml
from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.orm import Session, sessionmaker

from config import PROJECT_ROOT, settings
from db.models import Analysis, Brand, Mention, Topic, TopicDaily
from nlp.text import clean_text
from sources.base import split_keywords

log = logging.getLogger("nlp.topics")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COMPLAINT_CATEGORIES = ("bug", "performance", "pricing", "UX/design", "support", "feature request")
# Kept upper-case when labels are title-cased.
ACRONYMS = {"ai": "AI", "api": "API", "ui": "UI", "ux": "UX", "ios": "iOS", "pdf": "PDF", "mcp": "MCP",
            "llm": "LLM", "llms": "LLMs", "gpt": "GPT", "url": "URL", "sso": "SSO", "ocr": "OCR"}
LABELS_FILE = PROJECT_ROOT / "topic_labels.yaml"
MMR_DIVERSITY = 0.3
# Words that carry no topic meaning in app reviews / HN comments (on top of sklearn's English list).
EXTRA_STOP_WORDS = {
    "app", "apps", "application", "use", "using", "used", "just", "like", "really", "get", "got",
    "would", "also", "one", "even", "im", "ive", "dont", "doesnt", "cant", "thing", "things",
    "make", "makes", "please", "need", "want", "way", "lot", "time", "still", "now", "new",
    "good", "great", "bad", "much", "many", "every", "know", "think", "people", "work", "works",
    "yeah", "keep", "keeps", "times", "bit", "opinion", "experience", "report", "important", "day",
    "days", "year", "years", "say", "said", "going", "sure", "actually", "better", "best", "love",
    "ts", "bro", "gud", "tried", "helped", "thanks", "hn", "show", "hi",
}
SEED = 42
MATCH_SHARE = 0.5  # share of a cluster's previously-assigned mentions that must share one old topic
# Minimum cluster size grows with the data: max(TOPIC_MIN_SIZE, this share of the complaints).
MIN_SIZE_SHARE = 0.04
# Mostly non-English texts form clusters of their own ("en la que"); the models are English-only.
# A small set of true English function words (sklearn's stop list contains "de", "con"...).
ENGLISH_FUNCTION_WORDS = frozenset(
    "the a an and or but is are was were be been it its this that these those i me my we our you your "
    "he she they them their to of in on at for with from by as not no do does did have has had can "
    "will would should could so if than then there here what when why how all any just very".split()
)
MIN_ENGLISH_SHARE = 0.10
MIN_WORDS_FOR_LANGUAGE_CHECK = 8  # telegraphic English ("Bad update, constant crashes") has none


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class MiniLMEmbedder:
    """Loads all-MiniLM-L6-v2 once; normalised embeddings so dot product = cosine similarity."""

    def __init__(self, device: str | None = None) -> None:
        self.device = device
        self._model: Any = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            from nlp.models import pick_device

            self._model = SentenceTransformer(EMBEDDING_MODEL, device=self.device or pick_device(settings.nlp_device))
        return np.asarray(self._model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False))


@dataclass
class Complaint:
    mention_id: int
    text: str
    created_at: datetime
    category: str | None
    old_topic_id: int | None


@dataclass
class Cluster:
    members: list[Complaint]
    keywords: list[str]
    label: str = ""
    category: str | None = None
    topic_id: int | None = None


@dataclass
class BrandResult:
    brand: str
    complaints: int = 0
    non_english: int = 0
    skipped: str | None = None
    clusters: int = 0
    reused: int = 0
    created: int = 0
    assigned: int = 0
    outliers: int = 0
    labels: list[tuple[int | None, str, str | None]] = field(default_factory=list)  # (topic id, label, category)


# ---------------------------------------------------------------- pure helpers


def is_complaint(sentiment: str | None, category: str | None) -> bool:
    return sentiment == "negative" or category in COMPLAINT_CATEGORIES


def looks_english(text: str) -> bool:
    """Cheap check: English is full of function words. Short texts are always kept."""
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if len(words) < MIN_WORDS_FOR_LANGUAGE_CHECK:
        return True
    return sum(w in ENGLISH_FUNCTION_WORDS for w in words) / len(words) >= MIN_ENGLISH_SHARE


def effective_min_size(n: int, floor: int) -> int:
    return max(floor, round(n * MIN_SIZE_SHARE))


def frequent_keywords(keywords: Sequence[str], texts: Sequence[str], min_docs: int = 2) -> list[str]:
    """Drop keywords found in fewer than `min_docs` of the topic's texts: on tiny topics c-TF-IDF
    favours words from a single review, which makes unreadable labels ('mercury decide')."""
    lowered = [t.lower() for t in texts]
    return [k for k in keywords if sum(bool(re.search(rf"\b{re.escape(k)}\b", t)) for t in lowered) >= min_docs]


def title_case(phrase: str) -> str:
    """'unskippable ads' -> 'Unskippable Ads'; acronyms stay upper-case ('ai features' -> 'AI Features')."""
    return " ".join(ACRONYMS.get(w.lower(), w[:1].upper() + w[1:]) for w in phrase.split())


def keyphrase_candidates(keywords: Sequence[str], top: int = 6) -> list[str]:
    """Keywords in rank order; a single word is upgraded to a longer phrase from the top N that
    contains it ('ads' -> 'unskippable ads'). Duplicates are dropped."""
    head = list(keywords[:top])
    out: list[str] = []
    for keyword in head:
        phrase = keyword
        if " " not in keyword:
            phrase = next((p for p in head if " " in p and keyword in p.split()), keyword)
        if phrase not in out:
            out.append(phrase)
    return out + [k for k in keywords if k not in out]


def dominant_category(categories: Sequence[str | None]) -> str | None:
    common = Counter(c for c in categories if c).most_common(1)
    return common[0][0] if common else None


def assign_labels(clusters: Sequence[Cluster]) -> None:
    """Label = top keyphrase in Title Case, unique within the brand: bigger topics choose first,
    a later topic whose best phrase is taken uses its next one."""
    used: set[str] = set()
    for cluster in sorted(clusters, key=lambda c: -len(c.members)):
        candidates = [title_case(p) for p in keyphrase_candidates(cluster.keywords)]
        label = next((c for c in candidates if c.lower() not in used), None)
        if label is None:  # every phrase taken (tiny vocabularies): number the best one
            base, n = (candidates[0] if candidates else "Unlabelled Topic"), 2
            while f"{base} ({n})".lower() in used:
                n += 1
            label = f"{base} ({n})"
        cluster.label = label
        used.add(label.lower())


def load_label_overrides(path: Path = LABELS_FILE) -> dict[int, str]:
    """topic_labels.yaml: {overrides: {topic_id: label}}. A missing file means no overrides."""
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = data.get("overrides") or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path.name}: 'overrides' must map topic id -> label")
    overrides = {}
    for key, label in raw.items():
        if not str(key).isdigit() or not str(label or "").strip():
            raise ValueError(f"{path.name}: bad override {key!r}: {label!r} (need a topic id and a label)")
        overrides[int(key)] = str(label).strip()
    return overrides


def match_clusters(clusters: Sequence[Cluster]) -> None:
    """Set cluster.topic_id to the existing topic it continues, by member overlap (greedy, 1:1)."""
    candidates = []
    for index, cluster in enumerate(clusters):
        olds = Counter(m.old_topic_id for m in cluster.members if m.old_topic_id is not None)
        total = sum(olds.values())
        for old_id, count in olds.items():
            if count / total >= MATCH_SHARE:
                candidates.append((count, index, old_id))
    claimed: set[int] = set()
    for count, index, old_id in sorted(candidates, reverse=True):
        if clusters[index].topic_id is None and old_id not in claimed:
            clusters[index].topic_id = old_id
            claimed.add(old_id)


def stop_words_for(all_brands: Sequence[Brand]) -> list[str]:
    """English + review filler + every tracked brand's name/keywords: competitor names say nothing
    about what a complaint is about."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    brand_words = set()
    for brand in all_brands:
        include, _, _ = split_keywords(brand.keywords or [])
        brand_words |= {w.lower() for k in [brand.name, *include] for w in k.replace(".", " ").split()}
    return sorted(ENGLISH_STOP_WORDS | EXTRA_STOP_WORDS | brand_words)


# ---------------------------------------------------------------- clustering


def _backend(embedder: Embedder) -> Any:
    """Wrap our Embedder for BERTopic, which needs it to embed candidate keyphrases."""
    from bertopic.backend import BaseEmbedder

    class _Backend(BaseEmbedder):
        def embed(self, documents: list[str], verbose: bool = False) -> np.ndarray:
            return embedder.encode(list(documents))

    return _Backend()


def cluster_texts(
    texts: list[str],
    embeddings: np.ndarray,
    embedder: Embedder,
    stop_words: list[str],
    min_size: int,
    outlier_similarity: float,
    selection: str = "leaf",
) -> tuple[list[int], dict[int, list[str]]]:
    """BERTopic tuned for ~30-350 short texts. Returns a topic per text (-1 = outlier) and keyphrases.

    Keyphrases: c-TF-IDF candidates re-ranked by KeyBERTInspired (similarity to the topic's
    representative mentions), then MaximalMarginalRelevance so the top phrases aren't near-duplicates."""
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
    from bertopic.vectorizers import ClassTfidfTransformer
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    n = len(texts)
    umap_model = UMAP(
        n_neighbors=min(10, n - 1), n_components=min(5, n - 2), min_dist=0.0, metric="cosine",
        random_state=SEED,  # deterministic (single-threaded) so identical data gives identical clusters
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_size, min_samples=2, metric="euclidean",
        # "leaf" gave 9-11 topics for every brand; "eom" collapsed Evernote's 244 complaints into 2.
        cluster_selection_method=selection, prediction_data=True,
    )
    # min_df stays 1: BERTopic joins each topic's texts into one document before vectorising.
    vectorizer = CountVectorizer(stop_words=stop_words, ngram_range=(1, 3), token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    representation = [KeyBERTInspired(random_state=SEED), MaximalMarginalRelevance(diversity=MMR_DIVERSITY)]
    model = BERTopic(
        embedding_model=_backend(embedder),
        umap_model=umap_model, hdbscan_model=hdbscan_model, vectorizer_model=vectorizer,
        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True), representation_model=representation,
        min_topic_size=min_size, calculate_probabilities=False, verbose=False,
    )
    topics, _ = model.fit_transform(texts, embeddings)
    if any(t == -1 for t in topics) and any(t != -1 for t in topics):
        topics = model.reduce_outliers(
            texts, topics, strategy="embeddings", embeddings=embeddings, threshold=outlier_similarity
        )
        model.update_topics(texts, topics=topics, vectorizer_model=vectorizer,
                            ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
                            representation_model=representation)
    keywords = {}
    for t in sorted(set(topics)):
        if t == -1:
            continue
        # KeyBERTInspired pads with "" when no candidate is similar to the topic; fall back to c-TF-IDF.
        phrases = [word for word, _ in (model.get_topic(t) or []) if word.strip()]
        keywords[t] = (phrases or _ctfidf_words(model, t))[:10]
    return [int(t) for t in topics], keywords


def _ctfidf_words(model: Any, topic: int, n: int = 10) -> list[str]:
    """Plain c-TF-IDF top words for a topic (before the representation models re-rank them)."""
    row = model.c_tf_idf_[topic + model._outliers].toarray().ravel()
    vocab = model.vectorizer_model.get_feature_names_out()
    return [str(vocab[i]) for i in row.argsort()[::-1][:n] if row[i] > 0]


# ---------------------------------------------------------------- database


def load_complaints(session: Session, brand_id: int) -> list[Complaint]:
    rows = session.execute(
        select(Mention.id, Mention.text, Mention.created_at, Analysis.category, Analysis.topic_id)
        .join(Analysis, Analysis.mention_id == Mention.id)
        .where(
            Mention.brand_id == brand_id,
            or_(Analysis.sentiment == "negative", Analysis.category.in_(COMPLAINT_CATEGORIES)),
        )
        .order_by(Mention.id)
    )
    complaints = [Complaint(i, clean_text(t), c_at, cat, old) for i, t, c_at, cat, old in rows]
    return [c for c in complaints if c.text]


def save_brand(
    session: Session, brand_id: int, clusters: list[Cluster], overrides: dict[int, str] | None = None
) -> None:
    """Write topics + analysis.topic_id for one brand, then rebuild its topic_daily rows.
    Manual labels from topic_labels.yaml win over generated ones."""
    overrides = overrides or {}
    brand_mentions = select(Mention.id).where(Mention.brand_id == brand_id)
    session.execute(
        update(Analysis).where(Analysis.mention_id.in_(brand_mentions)).values(topic_id=None)
        .execution_options(synchronize_session=False)
    )
    for cluster in clusters:
        dates = [m.created_at for m in cluster.members]
        values = {"label": cluster.label[:255], "category": cluster.category, "keywords": cluster.keywords,
                  "first_seen": min(dates), "last_seen": max(dates)}
        if cluster.topic_id is None:
            topic = Topic(brand_id=brand_id, **values)
            session.add(topic)
            session.flush()
            cluster.topic_id = topic.id
            if topic.id in overrides:
                cluster.label = topic.label = overrides[topic.id][:255]
        else:
            if cluster.topic_id in overrides:
                cluster.label = values["label"] = overrides[cluster.topic_id][:255]
            session.execute(update(Topic).where(Topic.id == cluster.topic_id).values(**values))
        session.execute(
            update(Analysis).where(Analysis.mention_id.in_([m.mention_id for m in cluster.members]))
            .values(topic_id=cluster.topic_id).execution_options(synchronize_session=False)
        )
    rebuild_topic_daily(session, brand_id)


def rebuild_topic_daily(session: Session, brand_id: int) -> int:
    """Mentions and average sentiment per topic per day, for this brand's topics."""
    topic_ids = list(session.scalars(select(Topic.id).where(Topic.brand_id == brand_id)))
    if not topic_ids:
        return 0
    session.execute(delete(TopicDaily).where(TopicDaily.topic_id.in_(topic_ids)))
    day = func.date(Mention.created_at)
    rows = session.execute(
        select(Analysis.topic_id, day, func.count(), func.avg(Analysis.sentiment_score))
        .join(Mention, Mention.id == Analysis.mention_id)
        .where(Analysis.topic_id.in_(topic_ids))
        .group_by(Analysis.topic_id, day)
    ).all()
    for topic_id, d, count, avg in rows:
        if isinstance(d, str):  # SQLite returns DATE() as text
            d = datetime.strptime(d, "%Y-%m-%d").date()
        session.add(TopicDaily(topic_id=topic_id, day=d, mentions=count,
                               avg_sentiment=None if avg is None else round(float(avg), 4)))
    return len(rows)


# ---------------------------------------------------------------- run


def run_topics(
    factory: sessionmaker[Session],
    embedder: Embedder,
    brands: Sequence[str] | None = None,
    min_complaints: int | None = None,
    min_size: int | None = None,
    outlier_similarity: float | None = None,
    overrides: dict[int, str] | None = None,
) -> list[BrandResult]:
    overrides = load_label_overrides() if overrides is None else overrides
    min_complaints = min_complaints or settings.topic_min_complaints
    min_size = min_size or settings.topic_min_size
    outlier_similarity = settings.topic_outlier_similarity if outlier_similarity is None else outlier_similarity
    results: list[BrandResult] = []

    with factory() as session:
        brand_rows = list(session.scalars(select(Brand).order_by(Brand.is_own.desc(), Brand.name)))
    stop_words = stop_words_for(brand_rows)
    if brands:
        wanted = {b.lower() for b in brands}
        unknown = wanted - {b.name.lower() for b in brand_rows}
        if unknown:
            raise SystemExit(f"Unknown brand(s): {', '.join(sorted(unknown))}")
        brand_rows = [b for b in brand_rows if b.name.lower() in wanted]

    for brand in brand_rows:
        result = BrandResult(brand=brand.name)
        results.append(result)
        with factory() as session:
            complaints = load_complaints(session, brand.id)
        english = [c for c in complaints if looks_english(c.text)]
        result.non_english = len(complaints) - len(english)
        complaints = english
        result.complaints = len(complaints)
        if len(complaints) < min_complaints:
            result.skipped = f"only {len(complaints)} complaints (< {min_complaints})"
            log.info("%s: skipped, %s", brand.name, result.skipped)
            continue

        texts = [c.text for c in complaints]
        embeddings = embedder.encode(texts)
        size = effective_min_size(len(texts), min_size)
        assignment, keywords = cluster_texts(texts, embeddings, embedder, stop_words, size, outlier_similarity)

        grouped: dict[int, list[Complaint]] = defaultdict(list)
        for complaint, topic in zip(complaints, assignment):
            if topic == -1:
                result.outliers += 1
            else:
                grouped[topic].append(complaint)
        clusters = []
        for t, members in sorted(grouped.items()):
            raw = keywords.get(t, [])
            readable = frequent_keywords(raw, [m.text for m in members]) or raw[:5]
            clusters.append(Cluster(members=members, keywords=readable,
                                    category=dominant_category([m.category for m in members])))
        assign_labels(clusters)
        match_clusters(clusters)
        result.reused = sum(c.topic_id is not None for c in clusters)
        result.created = len(clusters) - result.reused
        result.clusters = len(clusters)
        result.assigned = sum(len(c.members) for c in clusters)

        with factory() as session:
            save_brand(session, brand.id, clusters, overrides)
            session.commit()
        result.labels = sorted(((c.topic_id, c.label, c.category) for c in clusters), key=lambda x: x[0] or 0)
        log.info("%s: %d complaints -> %d topics (%d reused, %d new), %d outliers",
                 brand.name, result.complaints, result.clusters, result.reused, result.created, result.outliers)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brand", action="append", help="only this brand (repeatable)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("BERTopic", "sentence_transformers", "numba", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    from db.session import get_session_factory

    results = run_topics(get_session_factory(), MiniLMEmbedder(), brands=args.brand)
    print()
    for r in results:
        if r.skipped:
            print(f"{r.brand:<10} skipped: {r.skipped}")
            continue
        print(f"{r.brand:<10} {r.complaints} complaints -> {r.clusters} topics "
              f"({r.reused} reused, {r.created} new), {r.assigned} assigned, {r.outliers} outliers, "
              f"{r.non_english} non-English skipped")
        for topic_id, label, category in r.labels:
            print(f"   #{topic_id:<4} {label:<40} [{category}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
