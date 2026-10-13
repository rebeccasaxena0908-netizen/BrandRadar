"""Topic clustering with a fake embedding model (real BERTopic/UMAP/HDBSCAN, no downloads)."""

from __future__ import annotations

from datetime import datetime, timedelta

import zlib
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from db.models import Analysis, Brand, Mention, Topic, TopicDaily
from nlp.topics import (
    Cluster, Complaint, assign_labels, dominant_category, effective_min_size, frequent_keywords, is_complaint,
    keyphrase_candidates, load_label_overrides, looks_english, match_clusters, run_topics, stop_words_for,
    title_case,
)
from sources.base import BrandSpec

GROUPS = {
    "sync": ["sync fails between my laptop and phone", "sync conflict duplicated pages again",
             "sync stopped working after the update", "sync takes forever on android",
             "my notes do not sync anymore", "sync keeps losing my edits"],
    "price": ["the price increase is too much", "price doubled and I cancelled",
              "subscription price is unfair", "price went up again this year",
              "cannot justify the price for students", "price change without warning"],
    "ads": ["ads pop up every time I open it", "full screen ads after every note",
            "video ads with sound are awful", "ads cover the editor on mobile",
            "remove the ads from the free plan", "ads everywhere now, uninstalling"],
}


class FakeEmbedder:
    """Each group sits around its own random centre, so clusters are unambiguous.
    Tests use min_size=10 so an 18-mention group cannot split into two leaf clusters."""

    def __init__(self) -> None:
        rng = np.random.default_rng(0)
        self.centres = {g: rng.normal(size=32) * 5 for g in GROUPS}
        self.calls = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        rows = []
        for t in texts:
            group = next((g for g in GROUPS if g in t or (g == "price" and "subscription" in t)), None)
            rng = np.random.default_rng(zlib.crc32(t.encode()))  # stable across runs
            # keyphrases without a group word (KeyBERTInspired embeds those too) get random vectors
            rows.append(self.centres[group] + rng.normal(size=32) * 0.1 if group else rng.normal(size=32) * 5)
        v = np.array(rows)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def add_complaints(factory: sessionmaker[Session], brand: BrandSpec, texts: list[str], prefix: str = "c",
                   sentiment: str = "negative", category: str | None = "bug") -> list[int]:
    with factory() as s:
        ids = []
        for i, t in enumerate(texts):
            m = Mention(brand_id=brand.id, source="test", external_id=f"{prefix}{i}", text=t,
                        created_at=datetime(2026, 10, 1) + timedelta(days=i % 5))
            s.add(m)
            s.flush()
            s.add(Analysis(mention_id=m.id, sentiment=sentiment, sentiment_score=-0.7, category=category))
            ids.append(m.id)
        s.commit()
        return ids


def all_texts(times: int = 3) -> list[str]:
    return [f"{t} #{k}" for k in range(times) for g in GROUPS.values() for t in g]


def topic_of(factory: sessionmaker[Session]) -> dict[int, int | None]:
    with factory() as s:
        return dict(s.execute(select(Analysis.mention_id, Analysis.topic_id)).all())


# ---------------------------------------------------------------- pure helpers


def test_is_complaint() -> None:
    assert is_complaint("negative", "praise")
    assert is_complaint("positive", "bug")
    assert not is_complaint("positive", "praise")
    assert not is_complaint("neutral", "general discussion")


def test_looks_english() -> None:
    assert looks_english("The sync is broken and I lost all of my notes")
    assert not looks_english("Une perte de temps, le pire logiciel de prise")
    assert not looks_english("raha hai abhi tak koi update nahi aaya bhai")
    assert looks_english("too slow")  # short texts are kept
    assert looks_english("Bad update, constant crashes, terrible support")  # < 8 words: kept


def test_effective_min_size_scales_with_data() -> None:
    assert effective_min_size(80, 5) == 5
    assert effective_min_size(250, 5) == 10


def test_frequent_keywords_drop_one_off_words() -> None:
    texts = ["sync is slow", "sync fails", "mercury decide nothing"]
    assert frequent_keywords(["mercury", "sync", "slow"], texts) == ["sync"]


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("unskippable ads", "Unskippable Ads"),
        ("slow daily note loading", "Slow Daily Note Loading"),
        ("ai features", "AI Features"),
        ("mcp server api", "MCP Server API"),
    ],
)
def test_title_case(phrase: str, expected: str) -> None:
    assert title_case(phrase) == expected


def test_keyphrase_candidates_upgrade_words_to_phrases() -> None:
    assert keyphrase_candidates(["ads", "unskippable ads", "price"]) == ["unskippable ads", "price", "ads"]  # "ads" kept as a last resort
    # once "slow sync android" is chosen, the bare words remain only as last-resort alternatives
    assert keyphrase_candidates(["sync", "android", "slow sync android"]) == ["slow sync android", "sync", "android"]


def test_dominant_category() -> None:
    assert dominant_category(["pricing", "bug", "pricing", None]) == "pricing"
    assert dominant_category([None]) is None


def test_assign_labels_unique_within_brand() -> None:
    def c(n: int, keywords: list[str]) -> Cluster:
        return Cluster(members=[Complaint(i, "t", datetime(2026, 10, 1), "bug", None) for i in range(n)], keywords=keywords)

    big, small, tiny = c(9, ["ads", "video ads"]), c(5, ["video ads", "loud ads"]), c(2, ["video ads"])
    assign_labels([small, tiny, big])
    assert big.label == "Video Ads"  # biggest topic picks first
    assert small.label == "Loud Ads"  # its best phrase was taken, so it uses the next one
    assert tiny.label == "Video Ads (2)"  # every phrase taken: numbered, still unique


def test_label_overrides_file(tmp_path: Path) -> None:
    path = tmp_path / "topic_labels.yaml"
    assert load_label_overrides(path) == {}
    path.write_text("overrides:\n  12: Unskippable Ads\n  '7': Sync Conflicts\n")
    assert load_label_overrides(path) == {12: "Unskippable Ads", 7: "Sync Conflicts"}
    path.write_text("overrides:\n  abc: X\n")
    with pytest.raises(ValueError, match="bad override"):
        load_label_overrides(path)


def test_project_label_file_is_valid() -> None:
    load_label_overrides()


def test_match_clusters_by_member_overlap() -> None:
    def c(*old: int | None) -> Cluster:
        return Cluster(members=[Complaint(i, "t", datetime(2026, 10, 1), "bug", o) for i, o in enumerate(old)], keywords=[])

    a = c(7, 7, 7, None)  # continues topic 7
    b = c(7, 8, 8)  # topic 8 has 2/3 -> continues 8
    d = c(9, 10)  # 50/50: tie -> one of them (>= 0.5 share)
    e = c(None, None)  # all new mentions -> new topic
    match_clusters([a, b, d, e])
    assert a.topic_id == 7 and b.topic_id == 8
    assert d.topic_id in (9, 10)
    assert e.topic_id is None


def test_match_clusters_one_old_topic_claimed_once() -> None:
    big = Cluster(members=[Complaint(i, "t", datetime(2026, 10, 1), "bug", 5) for i in range(6)], keywords=[])
    small = Cluster(members=[Complaint(i, "t", datetime(2026, 10, 1), "bug", 5) for i in range(2)], keywords=[])
    match_clusters([small, big])
    assert big.topic_id == 5 and small.topic_id is None  # the larger overlap wins the old topic


def test_stop_words_include_every_brand() -> None:
    brands = [Brand(name="Notion", keywords=["Notion", "notion.so"]), Brand(name="ClickUp", keywords=["clickup"])]
    words = stop_words_for(brands)
    assert {"notion", "so", "clickup", "app", "the"} <= set(words)


# ---------------------------------------------------------------- clustering + DB (real BERTopic)


def test_clusters_and_stable_reruns(factory: sessionmaker[Session], notion: BrandSpec) -> None:
    add_complaints(factory, notion, all_texts())
    embedder = FakeEmbedder()
    [first] = run_topics(factory, embedder, brands=["Notion"], min_complaints=10, min_size=10, outlier_similarity=0.5)
    assert first.clusters == 3 and first.created == 3 and first.skipped is None
    before = topic_of(factory)
    with factory() as s:
        labels = sorted(s.scalars(select(Topic.label)))
    with factory() as s:
        keyword_sets = [" ".join(k) for k in s.scalars(select(Topic.keywords))]
    for group_word in ("sync", "ads"):
        assert any(group_word in k for k in keyword_sets)
    assert any("price" in k or "subscription" in k for k in keyword_sets)
    assert all(l and l == l.title() for l in labels)  # Title Case, no "Category –" prefix
    assert len({l.lower() for l in labels}) == len(labels)  # unique within the brand
    with factory() as s:
        assert set(s.scalars(select(Topic.category))) == {"bug"}  # category is its own field
    assert first.labels and all(cat == "bug" for _, _, cat in first.labels)

    [second] = run_topics(factory, embedder, brands=["Notion"], min_complaints=10, min_size=10, outlier_similarity=0.5)
    assert (second.reused, second.created) == (3, 0)
    assert topic_of(factory) == before  # same mentions keep the same topic ids
    with factory() as s:
        assert s.scalar(select(func.count()).select_from(Topic)) == 3  # no duplicates

    # new mentions join the existing topics; ids are still reused
    add_complaints(factory, notion, [f"{t} again" for t in GROUPS["sync"]], prefix="n")
    [third] = run_topics(factory, embedder, brands=["Notion"], min_complaints=10, min_size=10, outlier_similarity=0.5)
    assert (third.reused, third.created) == (3, 0)
    after = topic_of(factory)
    assert all(after[m] == t for m, t in before.items())


def test_topic_daily_aggregates(factory: sessionmaker[Session], notion: BrandSpec) -> None:
    ids = add_complaints(factory, notion, all_texts())
    run_topics(factory, FakeEmbedder(), brands=["Notion"], min_complaints=10, min_size=10, outlier_similarity=0.5)
    with factory() as s:
        rows = list(s.scalars(select(TopicDaily)))
        assigned = s.scalar(select(func.count()).select_from(Analysis).where(Analysis.topic_id.is_not(None)))
    assert sum(r.mentions for r in rows) == assigned
    assert all(r.avg_sentiment == pytest.approx(-0.7) for r in rows)
    assert {r.day for r in rows} <= {(datetime(2026, 10, 1) + timedelta(days=d)).date() for d in range(5)}
    assert len(ids) == 54


def test_brand_with_few_complaints_is_skipped(factory: sessionmaker[Session], notion: BrandSpec) -> None:
    add_complaints(factory, notion, GROUPS["sync"][:4])
    embedder = FakeEmbedder()
    [result] = run_topics(factory, embedder, brands=["Notion"], min_complaints=10, min_size=5)
    assert result.skipped and "4 complaints" in result.skipped
    assert embedder.calls == 0
    with factory() as s:
        assert s.scalar(select(func.count()).select_from(Topic)) == 0


def test_praise_is_not_clustered(factory: sessionmaker[Session], notion: BrandSpec) -> None:
    add_complaints(factory, notion, all_texts())
    praise = add_complaints(factory, notion, ["sync is lovely"] * 3, prefix="p", sentiment="positive", category="praise")
    run_topics(factory, FakeEmbedder(), brands=["Notion"], min_complaints=10, min_size=10, outlier_similarity=0.5)
    topics = topic_of(factory)
    assert all(topics[p] is None for p in praise)


def test_overrides_are_respected(factory: sessionmaker[Session], notion: BrandSpec) -> None:
    add_complaints(factory, notion, all_texts())
    embedder = FakeEmbedder()
    run_topics(factory, embedder, brands=["Notion"], min_complaints=10, min_size=10, outlier_similarity=0.5, overrides={})
    with factory() as s:
        some_id = min(s.scalars(select(Topic.id)))
    run_topics(factory, embedder, brands=["Notion"], min_complaints=10, min_size=10, outlier_similarity=0.5,
               overrides={some_id: "My Manual Label"})
    with factory() as s:
        assert s.get(Topic, some_id).label == "My Manual Label"
        assert s.scalar(select(func.count()).select_from(Topic)) == 3  # still no duplicates


def test_excluded_sources_stay_out_of_topics(factory: sessionmaker[Session], notion: BrandSpec) -> None:
    add_complaints(factory, notion, all_texts())
    with factory() as s:  # makers' posts from dev.to that zero-shot labelled as complaints
        promo = []
        for i in range(12):
            m = Mention(brand_id=notion.id, source="devto", external_id=f"d{i}", text=f"sync plugin release notes {i}",
                        created_at=datetime(2026, 10, 1))
            s.add(m)
            s.flush()
            s.add(Analysis(mention_id=m.id, sentiment="neutral", sentiment_score=0.0, category="support"))
            promo.append(m.id)
        s.commit()
    [result] = run_topics(factory, FakeEmbedder(), brands=["Notion"], min_complaints=10, min_size=10,
                          outlier_similarity=0.5, overrides={}, excluded_sources=("devto",))
    assert result.complaints == 54  # the 12 dev.to posts were not counted
    topics = topic_of(factory)
    assert all(topics[p] is None for p in promo)
