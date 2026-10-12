"""NLP pipeline with a fake analyzer: no model downloads, SQLite instead of MySQL."""

from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from db.models import Analysis, Brand, Mention
from nlp.models import EMOTIONS, Scores
from nlp.pipeline import (
    CATEGORY_HYPOTHESES,
    PIPELINE_VERSION,
    best_feature,
    build_model_version,
    pick_category,
    restamp,
    run_pipeline,
    signed_sentiment,
)
from sources.base import BrandSpec

HASHES = {
    "cardiffnlp/twitter-roberta-base-sentiment-latest": "aaaaaaaa1111",
    "j-hartmann/emotion-english-distilroberta-base": "bbbbbbbb2222",
    "facebook/bart-large-mnli": "cccccccc3333",
}


class FakeAnalyzer:
    """Deterministic stand-in for HFAnalyzer. 'BOOM' in a text makes any call fail."""

    device = "cpu"

    def __init__(self, hashes: dict[str, str] | None = None, feature_score: float = 0.9) -> None:
        self.hashes = hashes or HASHES
        self.feature_score = feature_score
        self.calls: list[tuple[str, int]] = []
        self.feature_labels: list[list[str]] = []

    def _check(self, kind: str, texts: list[str]) -> None:
        self.calls.append((kind, len(texts)))
        if any("BOOM" in t for t in texts):
            raise RuntimeError("model blew up")

    def sentiment(self, texts: list[str]) -> list[Scores]:
        self._check("sentiment", texts)
        out = []
        for t in texts:
            if "love" in t.lower():
                out.append({"positive": 0.8, "neutral": 0.15, "negative": 0.05})
            elif "hate" in t.lower():
                out.append({"positive": 0.1, "neutral": 0.2, "negative": 0.7})
            else:
                out.append({"positive": 0.2, "neutral": 0.6, "negative": 0.2})
        return out

    def emotion(self, texts: list[str]) -> list[Scores]:
        self._check("emotion", texts)
        return [{e: (0.7 if e == ("joy" if "love" in t.lower() else "anger") else 0.05) for e in EMOTIONS} for t in texts]

    def zero_shot(self, texts: list[str], labels: list[str], template: str, multi_label: bool) -> list[Scores]:
        self._check("features" if multi_label else "category", texts)
        if multi_label:
            self.feature_labels.append(labels)
            # first feature mentioned in the text wins, else the first label with a low score
            out = []
            for t in texts:
                hit = next((l for l in labels if l.lower() in t.lower()), labels[0])
                out.append({l: (self.feature_score if l == hit else 0.1) for l in labels})
            return out
        bug = CATEGORY_HYPOTHESES["bug"]
        praise = CATEGORY_HYPOTHESES["praise"]
        return [{bug: 0.8, praise: 0.2} if "crash" in t.lower() else {bug: 0.1, praise: 0.9} for t in texts]

    def model_hashes(self) -> dict[str, str]:
        return self.hashes

    def release_memory(self) -> None:
        self.calls.append(("release", 0))


def add_mentions(factory: sessionmaker[Session], brand: BrandSpec, texts: list[str]) -> list[int]:
    with factory() as s:
        start = s.scalar(select(func.count()).select_from(Mention)) or 0
        rows = [
            Mention(brand_id=brand.id, source="test", external_id=str(start + i), text=t,
                    created_at=datetime(2026, 10, 5))
            for i, t in enumerate(texts)
        ]
        s.add_all(rows)
        s.commit()
        return [r.id for r in rows]


def analyses(factory: sessionmaker[Session]) -> dict[int, Analysis]:
    with factory() as s:
        return {a.mention_id: a for a in s.scalars(select(Analysis))}


@pytest.fixture
def featured(factory: sessionmaker[Session], brands: list[BrandSpec]) -> list[BrandSpec]:
    with factory() as s:
        s.get(Brand, brands[0].id).features = {  # type: ignore[union-attr]
            "sync": ["between devices"], "pricing": ["expensive"], "offline": [],
        }
        s.get(Brand, brands[1].id).features = {"plugins": ["plugin"], "sync": []}  # type: ignore[union-attr]
        s.commit()
    return brands


# ---------------------------------------------------------------- pure helpers


def test_signed_sentiment() -> None:
    assert signed_sentiment({"positive": 0.8, "neutral": 0.15, "negative": 0.05}) == 0.75
    assert signed_sentiment({"positive": 0.1, "neutral": 0.2, "negative": 0.7}) == -0.6
    assert signed_sentiment({}) == 0.0


@pytest.mark.parametrize(
    ("scores", "threshold", "expected"),
    [
        ({"sync": 0.7, "AI": 0.2}, 0.5, "sync"),
        ({"sync": 0.5, "AI": 0.2}, 0.5, "sync"),  # exactly at the threshold counts
        ({"sync": 0.49, "AI": 0.2}, 0.5, None),
        ({}, 0.5, None),
    ],
)
def test_best_feature_threshold(scores: Scores, threshold: float, expected: str | None) -> None:
    assert best_feature(scores, threshold) == expected


def test_model_version_format() -> None:
    version = build_model_version(HASHES)
    assert version == f"p{PIPELINE_VERSION}-saaaaaaaa-ebbbbbbbb-zcccccccc"
    assert len(version) <= 64


# ---------------------------------------------------------------- pipeline + DB


def test_writes_one_row_per_mention(factory: sessionmaker[Session], featured: list[BrandSpec]) -> None:
    notion = featured[0]
    love, crash = add_mentions(factory, notion, [
        "I love how sync works across devices https://notion.so",
        "It crashes and I hate the new pricing",
    ])
    stats = run_pipeline(factory, FakeAnalyzer(), batch_size=8)

    assert (stats.selected, stats.analysed, stats.failed, stats.empty) == (2, 2, 0, 0)
    rows = analyses(factory)
    a = rows[love]
    assert (a.sentiment, a.sentiment_score, a.emotion, a.category, a.feature) == ("positive", 0.75, "joy", "praise", "sync")
    assert set(a.emotion_scores) == set(EMOTIONS)
    assert a.model_version == build_model_version(HASHES)
    b = rows[crash]
    assert (b.sentiment, b.sentiment_score, b.emotion, b.category, b.feature) == ("negative", -0.6, "anger", "bug", "pricing")


def test_batching(factory: sessionmaker[Session], featured: list[BrandSpec]) -> None:
    add_mentions(factory, featured[0], [f"I love feature number {i}" for i in range(10)])
    fake = FakeAnalyzer()
    run_pipeline(factory, fake, batch_size=4)
    assert [n for kind, n in fake.calls if kind == "sentiment"] == [4, 4, 2]
    assert sum(kind == "release" for kind, _ in fake.calls) == 3  # memory freed after every batch


def test_idempotent_second_run_does_nothing(factory: sessionmaker[Session], featured: list[BrandSpec]) -> None:
    add_mentions(factory, featured[0], ["I love it a lot", "crash crash crash today"])
    run_pipeline(factory, FakeAnalyzer())
    fake = FakeAnalyzer()
    stats = run_pipeline(factory, fake)
    assert stats.selected == 0 and stats.analysed == 0
    assert [c for c in fake.calls if c[0] != "release"] == []
    # new mentions are picked up, old ones untouched
    add_mentions(factory, featured[0], ["new one about offline mode"])
    assert run_pipeline(factory, FakeAnalyzer()).analysed == 1
    assert len(analyses(factory)) == 3


def test_limit_and_brand_filter(factory: sessionmaker[Session], featured: list[BrandSpec]) -> None:
    notion, obsidian = featured
    add_mentions(factory, notion, ["notion one text", "notion two text", "notion three text"])
    obs_ids = add_mentions(factory, obsidian, ["obsidian plugins are great"])
    assert run_pipeline(factory, FakeAnalyzer(), brands=["Obsidian"]).analysed == 1
    assert set(analyses(factory)) == set(obs_ids)
    assert run_pipeline(factory, FakeAnalyzer(), limit=2).analysed == 2
    with pytest.raises(SystemExit):
        run_pipeline(factory, FakeAnalyzer(), brands=["Nope"])


def test_feature_lists_are_per_brand_and_threshold_applies(
    factory: sessionmaker[Session], featured: list[BrandSpec]
) -> None:
    notion, obsidian = featured
    n_id = add_mentions(factory, notion, ["the plugin here is slow"])[0]  # 'plugins' is not a Notion feature
    o_id = add_mentions(factory, obsidian, ["the plugin here is slow"])[0]
    fake = FakeAnalyzer()
    run_pipeline(factory, fake)
    rows = analyses(factory)
    assert rows[n_id].feature is None  # no Notion feature term in the text -> no zero-shot call
    assert rows[o_id].feature == "plugins"
    assert fake.feature_labels == [["plugins"]]  # only the keyword-matched candidates are scored

    # below threshold -> NULL
    low_id = add_mentions(factory, notion, ["sync is a bit odd"])[0]
    run_pipeline(factory, FakeAnalyzer(feature_score=0.3), threshold=0.5)
    assert analyses(factory)[low_id].feature is None


def test_no_keyword_means_no_feature_and_no_zero_shot(factory: sessionmaker[Session], featured: list[BrandSpec]) -> None:
    mid = add_mentions(factory, featured[0], ["love this app so much"])[0]
    fake = FakeAnalyzer()
    run_pipeline(factory, fake)
    row = analyses(factory)[mid]
    assert row.sentiment == "positive" and row.feature is None
    assert not any(kind == "features" for kind, _ in fake.calls)


def test_short_text_gets_feature_from_synonym(factory: sessionmaker[Session], featured: list[BrandSpec]) -> None:
    mid = add_mentions(factory, featured[0], ["too expensive"])[0]
    run_pipeline(factory, FakeAnalyzer())
    assert analyses(factory)[mid].feature == "pricing"


def test_several_candidates_best_score_wins(factory: sessionmaker[Session], featured: list[BrandSpec]) -> None:
    mid = add_mentions(factory, featured[0], ["offline mode is fine but pricing is a joke"])[0]
    fake = FakeAnalyzer()
    run_pipeline(factory, fake)
    assert fake.feature_labels == [["pricing", "offline"]]  # brands.yaml order
    assert analyses(factory)[mid].feature == "pricing"  # fake scores the first label found in the text


def test_legacy_list_features_still_work(factory: sessionmaker[Session], brands: list[BrandSpec]) -> None:
    with factory() as s:
        s.get(Brand, brands[0].id).features = ["sync", "offline"]  # type: ignore[union-attr]
        s.commit()
    mid = add_mentions(factory, brands[0], ["offline mode broke"])[0]
    run_pipeline(factory, FakeAnalyzer())
    assert analyses(factory)[mid].feature == "offline"


def test_general_discussion_is_a_category() -> None:
    assert "general discussion" in CATEGORY_HYPOTHESES
    assert len(CATEGORY_HYPOTHESES) == 8


def test_empty_after_cleaning_stored_as_null_row(factory: sessionmaker[Session], featured: list[BrandSpec]) -> None:
    empty_id = add_mentions(factory, featured[0], ["https://notion.so/some/page"])[0]
    stats = run_pipeline(factory, FakeAnalyzer())
    assert stats.empty == 1 and stats.analysed == 0
    row = analyses(factory)[empty_id]
    assert row.sentiment is None and row.category is None and row.model_version
    assert run_pipeline(factory, FakeAnalyzer()).selected == 0  # not retried forever


def test_failing_batch_does_not_lose_the_rest(factory: sessionmaker[Session], featured: list[BrandSpec]) -> None:
    ids = add_mentions(factory, featured[0], [
        "I love it one", "text two here", "BOOM goes this one", "text four here", "text five here",
    ])
    stats = run_pipeline(factory, FakeAnalyzer(), batch_size=10)
    assert stats.analysed == 4 and stats.failed == 1
    bad = ids[2]
    assert stats.failed_ids == [bad]
    assert set(analyses(factory)) == set(ids) - {bad}
    # the failed one is retried on the next run
    assert run_pipeline(factory, FakeAnalyzer()).selected == 1


def test_reanalyse_by_brand(factory: sessionmaker[Session], featured: list[BrandSpec]) -> None:
    notion, obsidian = featured
    add_mentions(factory, notion, ["notion text one", "notion text two"])
    add_mentions(factory, obsidian, ["obsidian text one"])
    run_pipeline(factory, FakeAnalyzer())
    stats = run_pipeline(factory, FakeAnalyzer(), brands=["Notion"], reanalyse=True)
    assert (stats.deleted, stats.analysed) == (2, 2)
    assert len(analyses(factory)) == 3


def test_reanalyse_outdated_model_version(factory: sessionmaker[Session], featured: list[BrandSpec]) -> None:
    add_mentions(factory, featured[0], ["first text here", "second text here"])
    run_pipeline(factory, FakeAnalyzer())
    old_version = build_model_version(HASHES)

    new_hashes = {**HASHES, "facebook/bart-large-mnli": "dddddddd4444"}
    add_mentions(factory, featured[0], ["third text here"])
    run_pipeline(factory, FakeAnalyzer(hashes=new_hashes))  # only the new mention
    versions = sorted(a.model_version for a in analyses(factory).values())
    assert versions.count(old_version) == 2

    stats = run_pipeline(factory, FakeAnalyzer(hashes=new_hashes), reanalyse=True, model_version="outdated")
    assert (stats.deleted, stats.analysed) == (2, 2)
    assert {a.model_version for a in analyses(factory).values()} == {build_model_version(new_hashes)}


def test_reanalyse_exact_version(factory: sessionmaker[Session], featured: list[BrandSpec]) -> None:
    add_mentions(factory, featured[0], ["some text here"])
    run_pipeline(factory, FakeAnalyzer())
    stats = run_pipeline(factory, FakeAnalyzer(), reanalyse=True, model_version="p0-unknown")
    assert stats.deleted == 0 and stats.analysed == 0


def test_pick_category_blocks_praise_when_negative() -> None:
    scores = {CATEGORY_HYPOTHESES["praise"]: 0.6, CATEGORY_HYPOTHESES["bug"]: 0.3, CATEGORY_HYPOTHESES["pricing"]: 0.1}
    assert pick_category(scores, "positive") == "praise"
    assert pick_category(scores, "neutral") == "praise"
    assert pick_category(scores, "negative") == "bug"  # next best
    assert pick_category({}, "negative") is None


def test_praise_rule_in_pipeline(factory: sessionmaker[Session], featured: list[BrandSpec]) -> None:
    mid = add_mentions(factory, featured[0], ["I hate what they did, it used to be great"])[0]
    run_pipeline(factory, FakeAnalyzer())  # fake says praise for non-crash texts
    row = analyses(factory)[mid]
    assert row.sentiment == "negative" and row.category == "bug"


def test_targeted_reanalyse_and_restamp(factory: sessionmaker[Session], featured: list[BrandSpec]) -> None:
    ids = add_mentions(factory, featured[0], ["I hate this", "it crashes often", "plain words here"])
    run_pipeline(factory, FakeAnalyzer())
    old = build_model_version(HASHES)
    with factory() as s:  # simulate a pre-v4 row: praise + negative
        s.get(Analysis, ids[0]).category = "praise"  # type: ignore[union-attr]
        s.commit()

    new_hashes = {**HASHES, "facebook/bart-large-mnli": "eeeeeeee5555"}
    stats = run_pipeline(factory, FakeAnalyzer(hashes=new_hashes), reanalyse=True, category="praise", sentiment="negative")
    assert (stats.deleted, stats.analysed) == (1, 1)
    rows = analyses(factory)
    new = build_model_version(new_hashes)
    assert rows[ids[0]].category == "bug" and rows[ids[0]].model_version == new
    assert {rows[i].model_version for i in ids[1:]} == {old}  # untouched

    with factory() as s:
        assert restamp(s, old, new) == 2
        s.commit()
    assert {a.model_version for a in analyses(factory).values()} == {new}
