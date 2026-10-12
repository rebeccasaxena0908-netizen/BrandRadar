"""HFAnalyzer maths with a tiny fake tokenizer/model injected (no downloads)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from nlp.models import PAD_MULTIPLE, SENTIMENT_MODEL, ZERO_SHOT_MODEL, HFAnalyzer

NLI_LABELS = {0: "contradiction", 1: "neutral", 2: "entailment"}  # bart-large-mnli order


class Encoded(dict):  # type: ignore[type-arg]
    def to(self, device: str) -> "Encoded":
        return self


class FakeTokenizer:
    """Records its arguments and passes the raw texts through to the fake model."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, first: list[str], second: list[str] | None = None, **kwargs: Any) -> Encoded:
        self.calls.append({"first": first, "second": second, **kwargs})
        return Encoded(first=first, second=second)


class FakeNLIModel:
    """Entailment logit +3 if the hypothesis word appears in the premise, else -3."""

    config = SimpleNamespace(label2id={v: k for k, v in NLI_LABELS.items()}, id2label=NLI_LABELS)

    def __call__(self, first: list[str], second: list[str]) -> Any:
        rows = []
        for premise, hypothesis in zip(first, second):
            hit = hypothesis.removeprefix("about ").rstrip(".") in premise
            rows.append([-3.0, 0.0, 3.0] if hit else [3.0, 0.0, -3.0])
        return SimpleNamespace(logits=torch.tensor(rows))


class FakeSentimentModel:
    config = SimpleNamespace(id2label={0: "Negative", 1: "Neutral", 2: "Positive"})

    def __call__(self, first: list[str], second: None = None) -> Any:
        return SimpleNamespace(logits=torch.tensor([[0.0, 0.0, 2.0] if "love" in t else [2.0, 0.0, 0.0] for t in first]))


@pytest.fixture
def analyzer() -> HFAnalyzer:
    a = HFAnalyzer(device="cpu", batch_size=4, max_tokens=512, zero_shot_max_tokens=256)
    a._models[ZERO_SHOT_MODEL] = (FakeTokenizer(), FakeNLIModel())
    a._models[SENTIMENT_MODEL] = (FakeTokenizer(), FakeSentimentModel())
    return a


def test_zero_shot_single_label_softmax_across_labels(analyzer: HFAnalyzer) -> None:
    [scores] = analyzer.zero_shot(["sync keeps failing"], ["sync", "pricing", "AI"], "about {}.", False)
    assert max(scores, key=scores.__getitem__) == "sync"
    assert sum(scores.values()) == pytest.approx(1.0)


def test_zero_shot_multi_label_independent_scores(analyzer: HFAnalyzer) -> None:
    [scores] = analyzer.zero_shot(["sync and pricing are bad"], ["sync", "pricing", "AI"], "about {}.", True)
    assert scores["sync"] > 0.99 and scores["pricing"] > 0.99  # several labels can be high
    assert scores["AI"] < 0.01


def test_zero_shot_pairs_batching_truncation_and_padding(analyzer: HFAnalyzer) -> None:
    out = analyzer.zero_shot(["sync one", "pricing two", "neither"], ["sync", "pricing"], "about {}.", False)
    assert [max(s, key=s.__getitem__) for s in out][:2] == ["sync", "pricing"]
    tokenizer = analyzer._models[ZERO_SHOT_MODEL][0]
    assert [len(c["first"]) for c in tokenizer.calls] == [4, 2]  # 3 texts x 2 labels, batches of 4
    first = tokenizer.calls[0]
    assert first["first"] == ["sync one", "sync one", "pricing two", "pricing two"]
    assert first["second"] == ["about sync.", "about pricing.", "about sync.", "about pricing."]
    assert first["truncation"] == "only_first"  # never cut the hypothesis
    assert first["max_length"] == 256
    assert first["pad_to_multiple_of"] == PAD_MULTIPLE


def test_zero_shot_empty_inputs(analyzer: HFAnalyzer) -> None:
    assert analyzer.zero_shot([], ["sync"], "about {}.", True) == []
    assert analyzer.zero_shot(["text"], [], "about {}.", True) == []


def test_classify_softmax_and_lowercase_labels(analyzer: HFAnalyzer) -> None:
    love, meh = analyzer.sentiment(["I love it", "meh"])
    assert set(love) == {"negative", "neutral", "positive"}
    assert max(love, key=love.__getitem__) == "positive"
    assert max(meh, key=meh.__getitem__) == "negative"
    assert sum(love.values()) == pytest.approx(1.0)
    call = analyzer._models[SENTIMENT_MODEL][0].calls[0]
    assert call["truncation"] is True and call["max_length"] == 512
    assert call["pad_to_multiple_of"] == PAD_MULTIPLE
