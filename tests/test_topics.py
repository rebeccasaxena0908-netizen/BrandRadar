"""Topic clustering with a fake embedding model (real BERTopic/UMAP/HDBSCAN, no downloads)."""

from __future__ import annotations

from datetime import datetime, timedelta

import zlib
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import delete, func, select
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


# ---------------------------------------------------------------- drift (Step 8 regression)


def _unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=32)
    return v / np.linalg.norm(v)


def test_step8_drift_theme_keeps_its_id_by_centroid() -> None:
    """Step 8: new mentions reshaped Obsidian's clusters. An iCloud-dominated cluster held MORE of the old
    "Daily Note" members (11) than the re-formed Daily Note cluster (9), so the overlap rule handed
    Daily Note's id (#49) to the iCloud cluster and Daily Note got a new id, a duplicate alert and an
    orphaned label. With centroids, the id stays with the theme."""
    from nlp.topics import centroid

    daily, icloud = _unit(1), _unit(2)
    assert float(daily @ icloud) < 0.5  # different themes

    def members(n_old: int, n_new: int) -> list[Complaint]:
        old = [Complaint(i, "t", datetime(2026, 10, 1), "bug", 49) for i in range(n_old)]
        return old + [Complaint(100 + i, "t", datetime(2026, 10, 9), "bug", None) for i in range(n_new)]

    near = lambda v, s: centroid(np.stack([v + np.random.default_rng(s + i).normal(size=32) * 0.05 for i in range(8)]))  # noqa: E731
    icloud_cluster = Cluster(members=members(11, 30), keywords=[], centroid=near(icloud, 10))
    daily_cluster = Cluster(members=members(9, 13), keywords=[], centroid=near(daily, 20))

    # the old rule (overlap only) reproduces the bug
    legacy = [Cluster(members=icloud_cluster.members, keywords=[]), Cluster(members=daily_cluster.members, keywords=[])]
    match_clusters(legacy)
    assert legacy[0].topic_id == 49 and legacy[1].topic_id is None

    # centroids keep #49 on Daily Note; the iCloud cluster becomes a new topic
    match_clusters([icloud_cluster, daily_cluster], existing={49: daily})
    assert daily_cluster.topic_id == 49
    assert icloud_cluster.topic_id is None


def test_centroid_needs_similarity_overlap_only_breaks_ties() -> None:
    a = _unit(5)
    twin_1 = Cluster(members=[Complaint(1, "t", datetime(2026, 10, 1), "bug", None)], keywords=[], centroid=a)
    twin_2 = Cluster(members=[Complaint(2, "t", datetime(2026, 10, 1), "bug", 7)], keywords=[], centroid=a)
    match_clusters([twin_1, twin_2], existing={7: a})
    assert twin_2.topic_id == 7 and twin_1.topic_id is None  # same similarity: the one with overlap wins


def test_lost_label_is_reported_not_reused(factory: sessionmaker[Session], notion: BrandSpec) -> None:
    ids = add_complaints(factory, notion, all_texts())
    embedder = FakeEmbedder()
    run_topics(factory, embedder, brands=["Notion"], min_complaints=10, min_size=10, outlier_similarity=0.5, overrides={})
    with factory() as s:
        ads_topic = next(t.id for t in s.scalars(select(Topic)) if "ads" in " ".join(t.keywords).lower())
        assert all(t.centroid and len(t.centroid) == 32 for t in s.scalars(select(Topic)))  # centroids stored
        # the ads complaints disappear (e.g. deleted), and a new theme of the same size arrives
        ads_ids = [m.id for m in s.scalars(select(Mention).where(Mention.id.in_(ids))) if "ads" in m.text]
        s.execute(delete(Analysis).where(Analysis.mention_id.in_(ads_ids)))  # SQLite here has no FK cascade
        s.execute(delete(Mention).where(Mention.id.in_(ads_ids)))
        s.commit()
    add_complaints(factory, notion, [f"export to pdf breaks tables #{k}" for k in range(18)], prefix="pdf")
    GROUPS["pdf"] = ["export"]  # teach the fake embedder the new theme (existing centres stay identical:
    try:                         # the random draws happen in the same order)
        [result] = run_topics(factory, FakeEmbedder(), brands=["Notion"], min_complaints=10, min_size=10,
                              outlier_similarity=0.5, overrides={ads_topic: "Ad Overload"})
    finally:
        del GROUPS["pdf"]
    assert result.lost_labels == [(ads_topic, "Ad Overload")]
    with factory() as s:
        assert s.scalar(select(func.count()).select_from(Analysis).where(Analysis.topic_id == ads_topic)) == 0  # inactive
        pdf_topic = {a.topic_id for a in s.scalars(select(Analysis).join(Mention).where(Mention.text.like("export%")))}
    assert ads_topic not in pdf_topic


def test_crowded_topics_need_a_closer_match(monkeypatch: pytest.MonkeyPatch) -> None:
    """Evernote-like brands have themes ~0.89 apart. If theme A vanishes and theme B splits in two, the
    second half of B must not inherit A's id just because B and A are neighbours."""
    import nlp.topics as topics_module

    base = _unit(30)
    a, b = base + _unit(31) * 0.35, base + _unit(32) * 0.35
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    ab = float(a @ b)
    assert 0.80 < ab < 0.97  # two close themes
    halves = []
    for seed in (33, 34):  # B split into two clusters, each a little off B
        v = b + _unit(seed) * 0.12
        halves.append(v / np.linalg.norm(v))
    assert all(0.80 <= float(h @ a) < ab + topics_module.SIBLING_MARGIN for h in halves)  # neighbours of A

    def run() -> list[int | None]:
        clusters = [Cluster(members=[Complaint(i, "t", datetime(2026, 10, 1), "bug", None)], keywords=[], centroid=h)
                    for i, h in enumerate(halves)]
        match_clusters(clusters, existing={1: a, 2: b})
        return [c.topic_id for c in clusters]

    assert sorted(run(), key=str) == [2, None]  # one half continues B, the other becomes a new topic
    monkeypatch.setattr(topics_module, "SIBLING_MARGIN", -1.0)  # i.e. a fixed 0.80 threshold
    assert sorted(run(), key=str) == [1, 2]  # ...which hands A's id to half of B: the drift this prevents
