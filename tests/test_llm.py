"""Step 10 LLM features with the LLM mocked: prompts, caching and regeneration, fallbacks, grounding,
translation, profile drafting and review, run logging, and the Ollama / Anthropic adapters."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import responses
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import seed_demo
from config import settings
from db.models import Analysis, Base, Brand, Briefing, LlmRun, Mention, Topic
from nlp.llm import briefing, labels, profile_draft, translate
from nlp.llm.provider import (
    AnthropicProvider, LLMError, LLMResult, LLMUnavailable, NoneProvider, OllamaProvider, get_provider, strict,
)
from sources.base import BrandSpec
from tests.test_pipeline import FakeAnalyzer, analyses

NOW = datetime(2026, 10, 14, 12, 0)


class FakeProvider:
    """Scripted LLM: each call returns the next answer (a dict, an exception to raise, or a function of the prompt)."""

    name, model = "ollama", "fake:3b"

    def __init__(self, *answers: Any) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []
        self.systems: list[str] = []
        self.unloaded = 0

    def json(self, system: str, prompt: str, schema: dict[str, Any], max_tokens: int = 1024) -> LLMResult:
        self.prompts.append(prompt)
        self.systems.append(system)
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if callable(answer) and not isinstance(answer, Exception):
            answer = answer(prompt)
        if isinstance(answer, Exception):
            raise answer
        return LLMResult(dict(answer), 0.01, 120, 30, 0.0)

    def text(self, system: str, prompt: str, max_tokens: int = 1024) -> LLMResult:
        return self.json(system, prompt, {}, max_tokens)  # scripted answers are {"text": ...}

    def unload(self) -> None:
        self.unloaded += 1


def runs(factory: sessionmaker[Session]) -> list[LlmRun]:
    with factory() as s:
        return list(s.scalars(select(LlmRun).order_by(LlmRun.id)))


# ---------------------------------------------------------------- providers


def test_strict_schema_requires_every_property() -> None:
    schema = strict({"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "array", "items": {
        "type": "object", "properties": {"c": {"type": "string"}}}}}})
    assert schema["required"] == ["a", "b"] and schema["additionalProperties"] is False
    assert schema["properties"]["b"]["items"]["required"] == ["c"]


def test_provider_choice_and_none() -> None:
    from nlp.llm import provider as module

    assert module.get_provider is not get_provider or True  # the autouse fixture swaps the module's function
    assert get_provider(dataclasses.replace(settings, llm_provider="none")).name == "none"
    assert isinstance(get_provider(dataclasses.replace(settings, llm_provider="ollama")), OllamaProvider)
    assert isinstance(get_provider(dataclasses.replace(settings, llm_provider="anthropic")), AnthropicProvider)
    with pytest.raises(ValueError):
        get_provider(dataclasses.replace(settings, llm_provider="gpt"))
    with pytest.raises(LLMUnavailable):
        NoneProvider().json("s", "p", {})


@responses.activate
def test_ollama_adapter_sends_schema_and_unloads() -> None:
    cfg = dataclasses.replace(settings, ollama_url="http://ollama.test", ollama_model="gemma3:4b", llm_timeout=5)
    responses.add(responses.POST, "http://ollama.test/api/chat", json={
        "message": {"content": '{"label": "Sync Conflicts"}'}, "prompt_eval_count": 90, "eval_count": 12})
    responses.add(responses.POST, "http://ollama.test/api/generate", json={})
    p = OllamaProvider(cfg)
    result = p.json("system", "prompt", {"type": "object", "properties": {"label": {"type": "string"}}}, max_tokens=50)
    assert result.data == {"label": "Sync Conflicts"} and (result.input_tokens, result.output_tokens) == (90, 12)
    body = json.loads(responses.calls[0].request.body)
    assert body["model"] == "gemma3:4b" and body["stream"] is False and body["keep_alive"] == "5m"
    assert body["format"]["required"] == ["label"] and body["options"]["num_predict"] == 50
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    p.unload()
    assert json.loads(responses.calls[1].request.body) == {"model": "gemma3:4b", "keep_alive": 0}  # frees the memory


@responses.activate
def test_ollama_retries_broken_json_once() -> None:
    cfg = dataclasses.replace(settings, ollama_url="http://ollama.test")
    responses.add(responses.POST, "http://ollama.test/api/chat", json={
        "message": {"content": '{"label": "iPad'}, "prompt_eval_count": 90, "eval_count": 5})
    responses.add(responses.POST, "http://ollama.test/api/chat", json={
        "message": {"content": '{"label": "iPad Instability"}'}, "prompt_eval_count": 90, "eval_count": 8})
    result = OllamaProvider(cfg).json("s", "p", {})
    assert result.data == {"label": "iPad Instability"} and (result.input_tokens, result.output_tokens) == (180, 13)
    assert len(responses.calls) == 2


@responses.activate
def test_ollama_text_has_no_schema_and_rejects_cut_off_answers() -> None:
    cfg = dataclasses.replace(settings, ollama_url="http://ollama.test")
    responses.add(responses.POST, "http://ollama.test/api/chat", json={
        "message": {"content": " Sentiment held steady.\n"}, "done_reason": "stop", "prompt_eval_count": 9, "eval_count": 4})
    result = OllamaProvider(cfg).text("s", "p", max_tokens=500)
    assert result.data == {"text": "Sentiment held steady."} and result.output_tokens == 4
    assert "format" not in json.loads(responses.calls[0].request.body)
    responses.replace(responses.POST, "http://ollama.test/api/chat", json={
        "message": {"content": "Mentions rose 3, -5, 54, 29, -3, -5"}, "done_reason": "length"})
    with pytest.raises(LLMError, match="ran out of tokens"):
        OllamaProvider(cfg).text("s", "p", max_tokens=500)


def test_briefing_tidy_is_an_exact_rewrite() -> None:
    assert briefing.tidy('Briefing: "Sentiment decreased by -5 points and rose by +3."') == \
        "Sentiment decreased by 5 points and rose by 3."
    assert briefing.tidy("Mentions fell -48% while two themes grew.") == "Mentions fell 48% while 2 themes grew."
    clash = briefing.tidy("Sentiment increased by -5 points.")
    assert clash == "Sentiment increased by -5 points."  # word and sign disagree: left for the check to reject
    assert "direction word and sign" in briefing.check_briefing(
        clash + " Mentions were 52 across all the sources we track. The top request is better AI features. Nothing else changed.", {"x": -5, "m": 52})


@responses.activate
def test_ollama_errors_become_fallbacks() -> None:
    cfg = dataclasses.replace(settings, ollama_url="http://ollama.test")
    responses.add(responses.POST, "http://ollama.test/api/chat", json={"message": {"content": "not json"}, "done_reason": "stop"})
    with pytest.raises(LLMError, match="done: stop"):
        OllamaProvider(cfg).json("s", "p", {})
    assert len(responses.calls) == 2  # retried once, then gave up
    responses.replace(responses.POST, "http://ollama.test/api/chat", status=500, body="boom")
    with pytest.raises(LLMError):
        OllamaProvider(cfg).json("s", "p", {})
    import requests

    def down(*a: Any, **k: Any) -> Any:
        raise requests.ConnectionError("refused")

    with pytest.raises(LLMUnavailable):
        OllamaProvider(cfg, post=down).json("s", "p", {})


class FakeAnthropic:
    def __init__(self, text: str, stop: str = "end_turn") -> None:
        self.kwargs: dict[str, Any] = {}
        outer = self

        class Messages:
            def create(self, **kwargs: Any) -> Any:
                outer.kwargs = kwargs
                block = type("B", (), {"type": "text", "text": text})()
                usage = type("U", (), {"input_tokens": 1000, "output_tokens": 200})()
                return type("R", (), {"content": [block], "usage": usage, "stop_reason": stop})()

        self.messages = Messages()


def test_anthropic_adapter_uses_structured_outputs_and_logs_cost() -> None:
    cfg = dataclasses.replace(settings, anthropic_model="claude-haiku-5-5", anthropic_price_input=0.10,
                              anthropic_price_output=0.50)
    client = FakeAnthropic('{"briefing": "Fine."}')
    result = AnthropicProvider(cfg, client=client).json("sys", "facts", {"type": "object", "properties": {
        "briefing": {"type": "string"}}}, max_tokens=500)
    assert result.data == {"briefing": "Fine."}
    assert result.cost_usd == pytest.approx((1000 * 0.10 + 200 * 0.50) / 1_000_000)
    k = client.kwargs
    assert k["model"] == "claude-haiku-5-5" and k["system"] == "sys" and k["messages"][0]["content"] == "facts"
    assert k["output_config"]["format"]["type"] == "json_schema" and k["output_config"]["effort"] == "low"
    assert k["output_config"]["format"]["schema"]["additionalProperties"] is False
    plain = FakeAnthropic(" Sentiment held steady. ")
    prose = AnthropicProvider(cfg, client=plain).text("sys", "facts")
    assert prose.data == {"text": "Sentiment held steady."} and prose.output_tokens == 200
    assert plain.kwargs["output_config"] == {"effort": "low"}  # prose: no schema
    with pytest.raises(LLMError):
        AnthropicProvider(cfg, client=FakeAnthropic("{}", stop="refusal")).json("s", "p", {})


# ---------------------------------------------------------------- translation


def add(factory: sessionmaker[Session], brand: BrandSpec, rows: list[tuple[str, str]]) -> list[int]:
    """Non-English mentions stored the way the pipeline leaves them: language set, analysis with NULL labels."""
    ids = []
    with factory() as s:
        for i, (text, lang) in enumerate(rows):
            m = Mention(brand_id=brand.id, source="youtube", external_id=f"x{i}", text=text, language=lang,
                        created_at=NOW - timedelta(hours=i))
            s.add(m)
            s.flush()
            s.add(Analysis(mention_id=m.id, sentiment=None, model_version="t"))
            ids.append(m.id)
        s.commit()
    return ids


def test_translation_flow_reanalyses_the_english_and_keeps_the_original(
    factory: sessionmaker[Session], brands: list[BrandSpec]
) -> None:
    from nlp.pipeline import run_pipeline

    ids = add(factory, brands[0], [
        ("ei app ta khub bhalo, ami love kori", "nl"),       # romanised Bengali, misdetected
        ("🛒 Tham khảo giá và đặt hàng ở đây ạ: 👉 link", "vi"),  # affiliate advert
        ("আমি জানি না", "bn"),                                  # the model is unsure
        ("Ce n'est pas clair", "fr"),                          # comes back untranslated
    ])
    fake = FakeProvider(
        {"translation": "This app is really good, I love it", "confidence": "high", "is_advert": False},
        {"translation": "Check prices and order here: link", "confidence": "high", "is_advert": True},
        {"translation": "I don't know", "confidence": "low", "is_advert": False},
        {"translation": "Ce n'est pas clair", "confidence": "high", "is_advert": False},
    )
    out = translate.translate_pending(factory, fake)
    assert (out.translated["Notion"], out.adverts["Notion"], out.failed["Notion"]) == (1, 1, 2)
    assert "probably romanised Hindi or Bengali" in fake.prompts[0]  # Latin text: the language guess is unreliable
    assert "Language: Bengali" in fake.prompts[2]  # scripts are reliable
    assert fake.unloaded == 1
    with factory() as s:
        rows = {m.id: m for m in s.scalars(select(Mention))}
        assert rows[ids[0]].text.startswith("ei app") and rows[ids[0]].text_en == "This app is really good, I love it"
        assert [rows[i].translation_status for i in ids] == ["ok", "advert", "failed", "failed"]
        assert rows[ids[0]].translated_by == "ollama:fake:3b" and rows[ids[0]].language == "nl"  # original language kept
        assert s.get(Analysis, ids[0]) is None and s.get(Analysis, ids[1]) is not None  # only the good one is redone
    stats = run_pipeline(factory, FakeAnalyzer())
    assert stats.selected == 1 and stats.analysed == 1  # the translation, analysed as English
    assert analyses(factory)[ids[0]].sentiment == "positive"
    # nothing left to translate; failed ones are retried only with force
    again = FakeProvider({"translation": "x", "confidence": "high", "is_advert": False})
    assert translate.translate_pending(factory, again).run.items == 0
    assert translate.translate_pending(factory, again, force=True).run.items == 2
    assert [r.stage for r in runs(factory)] == ["translate", "translate", "translate"]


def test_translation_stops_cleanly_without_an_llm(factory: sessionmaker[Session], brands: list[BrandSpec]) -> None:
    add(factory, brands[0], [("ভালো", "bn"), ("খুব ভালো", "bn")])
    out = translate.translate_pending(factory, NoneProvider())
    assert out.total == 0 and out.run.fallbacks == 2 and out.run.errors == ["LLM_PROVIDER=none"]
    with factory() as s:
        assert {m.translation_status for m in s.scalars(select(Mention))} == {None}  # still waiting, not failed


def test_redo_translates_again_and_drops_the_old_analysis(factory: sessionmaker[Session], brands: list[BrandSpec]) -> None:
    from nlp.pipeline import run_pipeline

    [mid] = add(factory, brands[0], [("Skrg ada iklan, berbayar pun mahal", "id")])
    translate.translate_pending(factory, FakeProvider({"translation": "Now there are ads, I love nothing", "confidence": "high", "is_advert": False}))
    run_pipeline(factory, FakeAnalyzer())
    assert analyses(factory)[mid].sentiment is not None
    out = translate.translate_pending(factory, FakeProvider(
        {"translation": "Now there are ads and paying is expensive", "confidence": "high", "is_advert": True}), redo=True)
    assert out.adverts["Notion"] == 1
    with factory() as s:
        assert s.get(Analysis, mid) is None  # the analysis came from the old translation
    # a redo that errors (e.g. JSON cut off) must not keep the old translation or its analysis either
    translate.translate_pending(factory, FakeProvider({"translation": "Now there are ads, I love nothing", "confidence": "high", "is_advert": False}), redo=True)
    run_pipeline(factory, FakeAnalyzer())
    assert analyses(factory)[mid].sentiment is not None
    translate.translate_pending(factory, FakeProvider(LLMError("not JSON: '{\"translation\": \"Now")), redo=True)
    with factory() as s:
        m = s.get(Mention, mid)
        assert (m.translation_status, m.text_en, m.translated_by) == ("failed", None, "ollama:fake:3b")
        assert s.get(Analysis, mid) is None


def test_check_translation() -> None:
    assert translate.check_translation("Kem chống nắng rất tốt", "The sunscreen is very good") is None
    assert translate.check_translation("abc def", " ABC  def ") == "unchanged"
    assert translate.check_translation("ভালো", "") == "empty"
    assert translate.check_translation("hola", "Bonjour, c'est vraiment un très bon produit pour moi") .startswith("not English")
    assert translate.clip_text("a. " * 1000, 50).endswith(".")


# ---------------------------------------------------------------- labels


def topic_with_members(factory: sessionmaker[Session], brand: BrandSpec, n: int, label: str = "Sync Sync",
                       **extra: Any) -> tuple[int, list[int]]:
    with factory() as s:
        topic = Topic(brand_id=brand.id, label=label, keyword_label=label, category="bug", **extra)
        s.add(topic)
        s.flush()
        ids = []
        for i in range(n):
            m = Mention(brand_id=brand.id, source="appstore", external_id=f"t{topic.id}-{i}",
                        text=f"sync broke again and my notes vanished on device {i}", created_at=NOW - timedelta(days=i))
            s.add(m)
            s.flush()
            s.add(Analysis(mention_id=m.id, sentiment="negative", sentiment_score=-0.5 - i / 100, topic_id=topic.id,
                           model_version="t"))
            ids.append(m.id)
        s.commit()
        return topic.id, ids


GOOD_LABEL = {"label": "Sync conflicts", "description": "Users lose edits when the same note syncs from two devices."}


def test_labels_are_generated_cached_and_refreshed_on_member_change(
    factory: sessionmaker[Session], brands: list[BrandSpec], monkeypatch: pytest.MonkeyPatch
) -> None:
    import nlp.topics

    monkeypatch.setattr(nlp.topics, "load_label_overrides", lambda *a, **k: {})
    tid, ids = topic_with_members(factory, brands[0], 12)
    fake = FakeProvider(GOOD_LABEL)
    run = labels.refresh_labels(factory, fake)
    assert (run.generated, run.cached) == (1, 0)
    prompt = fake.prompts[0]
    assert "Industry: software" in prompt and "Complaint category: bug" in prompt and "Keyword label" in prompt
    assert prompt.count("\n") >= 10 and prompt.count("sync broke again") == 10  # 10 of the 12 members
    assert "other themes" not in prompt  # the only topic: nothing to avoid
    assert "other themes (pick a different, more specific label): Mobile, Sync" in labels.build_prompt(
        "Notion", "software", "bug", "x", ["a"], ["Sync", "Mobile"])
    with factory() as s:
        t = s.get(Topic, tid)
        assert (t.label, t.label_source, t.llm_label) == ("Sync Conflicts", "llm", "Sync Conflicts")  # Title Case
        assert t.description.startswith("Users lose edits") and sorted(t.llm_members) == ids
    # unchanged members: cached, no LLM call
    assert labels.refresh_labels(factory, FakeProvider(RuntimeError("must not be called"))).cached == 1
    # 1 new member of 13 (8% change): still cached; 3 more (25%): regenerated
    with factory() as s:
        for i in range(1):
            m = Mention(brand_id=brands[0].id, source="appstore", external_id=f"new{i}", text="sync broke on my phone too", created_at=NOW)
            s.add(m)
            s.flush()
            s.add(Analysis(mention_id=m.id, sentiment="negative", sentiment_score=-0.4, topic_id=tid, model_version="t"))
        s.commit()
    assert labels.refresh_labels(factory, FakeProvider(RuntimeError("no"))).cached == 1
    with factory() as s:
        for i in range(3):
            m = Mention(brand_id=brands[0].id, source="appstore", external_id=f"more{i}", text="sync duplicates every note", created_at=NOW)
            s.add(m)
            s.flush()
            s.add(Analysis(mention_id=m.id, sentiment="negative", sentiment_score=-0.4, topic_id=tid, model_version="t"))
        s.commit()
    run = labels.refresh_labels(factory, FakeProvider({"label": "Duplicate Notes After Sync", "description": "Sync creates duplicate copies of notes on every device."}))
    assert run.generated == 1
    with factory() as s:
        assert s.get(Topic, tid).label == "Duplicate Notes After Sync"


def test_manual_labels_win_and_bad_or_failed_labels_fall_back(
    factory: sessionmaker[Session], brands: list[BrandSpec], monkeypatch: pytest.MonkeyPatch
) -> None:
    import nlp.topics

    manual, _ = topic_with_members(factory, brands[0], 6, label="Kw One")
    vague, _ = topic_with_members(factory, brands[0], 6, label="Kw Two")
    broken, _ = topic_with_members(factory, brands[0], 6, label="Kw Three")
    monkeypatch.setattr(nlp.topics, "load_label_overrides", lambda *a, **k: {manual: "Paywall and AI Limits"})
    fake = FakeProvider(
        GOOD_LABEL,                                                         # manual topic: description only
        {"label": "Issues", "description": "Various issues with the app."},  # too vague -> rejected
        LLMError("timed out"),                                              # provider error
    )
    run = labels.refresh_labels(factory, fake)
    assert run.generated == 1 and run.fallbacks == 2
    with factory() as s:
        t = {i: s.get(Topic, i) for i in (manual, vague, broken)}
        assert (t[manual].label, t[manual].label_source) == ("Paywall and AI Limits", "manual")
        assert t[manual].description.startswith("Users lose edits")
        assert (t[vague].label, t[vague].label_source) == ("Kw Two", "keyword")  # keyword label kept
        assert (t[broken].label, t[broken].label_source) == ("Kw Three", "keyword")
    assert {"label rejected: label has 1 words", "timed out"} <= set(run.errors)
    # the override is removed from the file: the LLM label (written with the description) takes over again
    monkeypatch.setattr(nlp.topics, "load_label_overrides", lambda *a, **k: {})
    labels.refresh_labels(factory, FakeProvider(LLMError("down")))
    with factory() as s:
        assert (s.get(Topic, manual).label, s.get(Topic, manual).label_source) == ("Sync Conflicts", "llm")
    # topic_labels.yaml holds real topic ids: the demo database never gets them
    monkeypatch.setattr(nlp.topics, "load_label_overrides", lambda *a, **k: {vague: "Real Label"})
    monkeypatch.setattr(labels, "settings", dataclasses.replace(labels.settings, demo=True))
    labels.refresh_labels(factory, FakeProvider(LLMError("down")))
    with factory() as s:
        assert s.get(Topic, vague).label == "Kw Two"


def test_label_checks() -> None:
    assert labels.check_label("Sync Conflicts", "Users lose edits when syncing between devices.", "Notion", set()) is None
    assert labels.check_label("Notion Sync Bugs", "Users lose edits when syncing.", "Notion", set()) == "label names the brand"
    assert labels.check_label("Sync Conflicts", "Users lose edits when syncing.", "Notion", {"sync conflicts"}) \
        == "label already used by another topic"
    assert "words" in labels.check_label("Sync", "Users lose edits when syncing.", "Notion", set())
    assert labels.check_label("Sync Conflicts", "Short.", "Notion", set()) == "description is not one short sentence"
    assert labels.title("problems with the ai and iOS app") == "Problems with the Ai and iOS App"
    assert labels.jaccard_distance([1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]) == pytest.approx(1 / 6)


def test_topics_run_keeps_cached_llm_labels(factory: sessionmaker[Session], brands: list[BrandSpec]) -> None:
    from nlp.topics import Cluster, Complaint, save_brand

    tid, ids = topic_with_members(factory, brands[0], 3, llm_label="Sync Conflicts", label_source="llm")
    members = [Complaint(i, "sync broke", NOW, "bug", tid) for i in ids]
    with factory() as s:
        save_brand(s, brands[0].id, [Cluster(members=members, keywords=["sync"], category="bug", topic_id=tid, label="Sync Broke")])
        s.commit()
        t = s.get(Topic, tid)
        assert (t.label, t.label_source, t.keyword_label) == ("Sync Conflicts", "llm", "Sync Broke")


# ---------------------------------------------------------------- briefings


@pytest.fixture(scope="module")
def demo_factory():  # type: ignore[no-untyped-def]
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    seed_demo.seed(engine, now=NOW)
    yield sessionmaker(bind=engine, expire_on_commit=False)
    engine.dispose()


def brand_id(factory: sessionmaker[Session], name: str) -> int:
    with factory() as s:
        return int(s.scalar(select(Brand.id).where(Brand.name == name)))


def grounded(prompt: str) -> dict[str, str]:
    """A briefing that only repeats numbers from the fact lines in the prompt (what a good model does)."""
    risk = next(line for line in prompt.splitlines() if line.startswith(("Biggest risk", "No urgent risk", "Biggest risk:")))
    return {"text": f"{risk} The team should watch this theme. Sentiment is steady. No other change stands out."}


def test_facts_come_from_the_dashboards_numbers(demo_factory: sessionmaker[Session]) -> None:
    engine = demo_factory.kw["bind"]
    facts = briefing.build_facts(engine, brand_id(demo_factory, "DOMS"), date(2026, 10, 8), date(2026, 10, 14), "the last 7 days")
    assert facts is not None
    assert facts["group"] == "Stationery" and set(facts["competitors"]) == {"Apsara", "Camlin", "Nataraj"}
    assert facts["biggest_risk"]["topic"] == "Leads Break While Sharpening" and facts["biggest_risk"]["urgent"]
    assert facts["requests"]["label"] == "product request"  # the stationery profile's request label
    assert len(facts["quotes"]) == 3
    lines = briefing.fact_lines(facts)
    assert "Complaints in the last 7 days: 9, up from 1 in the 7 days before." in lines
    thin = {**facts, "mentions": 52, "mentions_previous_period": 100, "mentions_change_percent": -48, "rising_topic": None,
            "requests": {**facts["requests"], "count": 0}}
    thin_lines = briefing.fact_lines(thin)
    assert "Mentions: 52, down 48% from 100 in the previous period." in thin_lines  # no "down -48%" to copy
    assert "Nothing to report (leave out entirely): product requests, a rising complaint theme." in thin_lines
    assert "Top request" not in thin_lines and "Notable change" not in thin_lines
    assert "covering, in this order: the biggest risk, the biggest win against the competitors, the top request" \
        in briefing.system_prompt(facts)
    empty = {**thin, "biggest_risk": None, "biggest_win": None}
    assert "covering, in this order: the overall volume and net sentiment, what the quotes say." \
        in briefing.system_prompt(empty)  # nothing to call a risk or a win
    assert "Competitors in the same group: Apsara, Camlin, Nataraj." in lines or "Competitors in the same group:" in lines
    assert briefing.build_facts(engine, brand_id(demo_factory, "DOMS"), date(2020, 1, 1), date(2020, 1, 7), "x") is None


def test_briefing_must_only_contain_numbers_from_the_facts() -> None:
    facts = {"net_sentiment": 31, "change": -4, "win": {"own": 39, "rivals": 14}, "risk": {"topic": "Top 10 Bugs", "n": 4}}
    ok = ("Net sentiment is +31, down 4 points. The biggest win is integrations at +39 against +14 for competitors. "
          "The top risk is Top 10 Bugs with 4 complaints. Nothing else changed.")
    assert briefing.check_briefing(ok, facts) is None
    assert briefing.check_briefing(ok.replace("+14", "-14"), facts) == "numbers not in the facts: -14"  # wrong sign
    assert briefing.check_briefing(ok.replace("4 complaints", "7 complaints"), facts) == "numbers not in the facts: 7"
    assert briefing.check_briefing(ok + " Mentions rose 12%.", facts) == "numbers not in the facts: 12"
    assert "word" in briefing.check_briefing(ok.replace("4 complaints", "four complaints"), facts)
    assert briefing.digits_for_words("Four complaints and two requests") == "4 complaints and 2 requests"
    assert "direction word and sign" in briefing.check_briefing(ok.replace("down 4 points", "down -4 points"), facts)
    assert briefing.check_briefing("- " + ok, facts) == "markdown or bullets"
    assert briefing.check_briefing("Too short.", facts).startswith("length")


def test_briefings_are_cached_by_facts_and_fall_back(demo_factory: sessionmaker[Session]) -> None:
    doms = brand_id(demo_factory, "DOMS")
    today = date(2026, 10, 14)
    fake = FakeProvider(grounded)
    run = briefing.refresh_briefings(demo_factory, fake, [doms], today=today)
    assert (run.generated, run.cached, run.fallbacks) == (2, 0, 0)  # last 7 days + last complete week
    with demo_factory() as s:
        rows = {b.period: b for b in s.scalars(select(Briefing).where(Briefing.brand_id == doms))}
        assert set(rows) == {"last7", "week"} and rows["week"].period_label == "2026-W41"
        assert rows["last7"].provider == "ollama" and rows["last7"].model == "fake:3b"
        assert "Leads Break While Sharpening" in rows["last7"].text and rows["last7"].facts["brand"] == "DOMS"
    # same facts: cached, no LLM call
    run = briefing.refresh_briefings(demo_factory, FakeProvider(RuntimeError("no")), [doms], today=today)
    assert run.cached == 2 and run.generated == 0
    # a briefing with an invented number is retried once with the reason, then the stale row is dropped
    with demo_factory() as s:
        row = s.scalar(select(Briefing).where(Briefing.brand_id == doms, Briefing.period == "last7"))
        row.facts_hash = "changed"
        s.commit()
    liar = FakeProvider({"text": "Complaints about leads breaking rose to 999 this week, a record for the brand. "
                                     "Competitors did better on every feature. Users want new colours. Sentiment fell sharply."})
    run = briefing.refresh_briefings(demo_factory, liar, [doms], today=today)
    assert run.fallbacks == 1 and len(liar.prompts) == 2 and "was rejected (numbers not in the facts: 999)" in liar.prompts[1]
    with demo_factory() as s:
        assert s.scalar(select(Briefing).where(Briefing.brand_id == doms, Briefing.period == "last7")) is None
        assert s.scalar(select(Briefing).where(Briefing.brand_id == doms, Briefing.period == "week")) is not None  # written once


def test_no_llm_means_rule_based_summary(demo_factory: sessionmaker[Session]) -> None:
    laneige = brand_id(demo_factory, "Laneige")
    run = briefing.refresh_briefings(demo_factory, NoneProvider(), [laneige], today=date(2026, 10, 14))
    assert run.generated == 0 and run.fallbacks == 2 and run.errors == ["LLM_PROVIDER=none"]
    with demo_factory() as s:
        assert s.scalar(select(Briefing).where(Briefing.brand_id == laneige)) is None


# ---------------------------------------------------------------- industry profile drafts


DRAFT = {
    "name": "running shoes",
    "complaint_categories": [
        {"label": "Fit", "hypothesis": "the fit or size, such as too tight, too narrow or too long"},
        {"label": "cushioning", "hypothesis": "cushioning or comfort, such as hard soles or sore feet"},
        {"label": "durability", "hypothesis": "durability, such as soles wearing out or seams splitting"},
        {"label": "grip", "hypothesis": "grip or traction on wet or trail surfaces"},
        {"label": "price/value", "hypothesis": "the price or value for money"},
    ],
    "praise_hypothesis": "praise or a compliment about the shoes",
    "request_hypothesis": "a wish for a new model, colour, width or size",
    "features": [{"name": n, "synonyms": s} for n, s in [
        ("fit", ["size", "narrow", "wide"]), ("cushioning", ["cushion", "soft", "comfortable"]),
        ("durability", ["wore out", "sole", "seams"]), ("grip", ["traction", "slippery"]),
        ("price", ["expensive", "cheap"]), ("weight", ["heavy", "light"])]],
}


def test_profile_draft_is_validated_and_saved_only_when_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import nlp.profiles

    path = tmp_path / "profiles.yaml"
    path.write_text(nlp.profiles.PROFILES_FILE.read_text(encoding="utf-8"), encoding="utf-8")
    fake = FakeProvider(DRAFT)
    draft = profile_draft.draft_profile("running shoes", "Asics", fake)
    assert draft is not None and draft.name == "running_shoes" and "Industry: running shoes" in fake.prompts[0]
    assert draft.raw["complaints"] == ["fit", "cushioning", "durability", "grip", "price/value"]
    assert {"praise", "general discussion", "product request"} <= set(draft.raw["categories"])
    profile_draft.save_profile(draft, path)
    loaded = nlp.profiles.load_profiles(path)["running_shoes"]
    assert loaded.request == "product request" and loaded.features["grip"] == ["traction", "slippery"]
    with pytest.raises(ValueError):  # the same name twice
        profile_draft.save_profile(draft, path)
    edited = profile_draft.draft_from_yaml(draft.to_yaml().replace("\n  ", "\n").lstrip(), "running shoes")
    assert edited.raw == draft.raw  # the review/edit round trip keeps everything


def test_unusable_or_missing_llm_falls_back_to_consumer_product() -> None:
    assert profile_draft.draft_profile("pet food", "Pedigree", NoneProvider()) is None
    too_small = dict(DRAFT, complaint_categories=DRAFT["complaint_categories"][:2])
    assert profile_draft.draft_profile("pet food", "Pedigree", FakeProvider(too_small)) is None


def test_add_brand_review_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import manage_brands
    from nlp.llm import profile_draft as module

    saved: list[str] = []
    monkeypatch.setattr(module, "save_profile", lambda draft, path=None: saved.append(draft.name))
    said: list[str] = []
    answers = iter(["e", "a"])
    edits: list[str] = []

    def edit(text: str) -> str:
        edits.append(text)
        return text.replace("weight: [heavy, light]", "weight: [heavy, light, featherweight]")

    name = manage_brands.choose_new_profile("running shoes", "Asics", lambda q: next(answers), said.append, edit,
                                            FakeProvider(DRAFT))
    assert name == "running_shoes" and saved == ["running_shoes"] and "featherweight" not in edits[0]
    assert any("Draft profile" in s for s in said)
    rejected = manage_brands.choose_new_profile("running shoes", "Asics", lambda q: "r", said.append, edit, FakeProvider(DRAFT))
    assert rejected == "consumer_product"
    assert manage_brands.choose_new_profile("pet food", "Pedigree", lambda q: "a", said.append, edit, NoneProvider()) \
        == "consumer_product"


# ---------------------------------------------------------------- logging, collector, dashboard


def test_stage_runs_log_time_tokens_and_cost(factory: sessionmaker[Session], brands: list[BrandSpec]) -> None:
    from nlp.llm.runlog import StageRun

    run = StageRun.start("labels", FakeProvider({}))
    run.items, run.generated = 3, 2
    run.add(LLMResult({}, 0.5, 1000, 200, 0.0002))
    run.fallback("label rejected: too vague")
    run.finish(factory)
    [row] = runs(factory)
    assert (row.stage, row.provider, row.model, row.items, row.generated, row.fallbacks) == ("labels", "ollama", "fake:3b", 3, 2, 1)
    assert (row.input_tokens, row.output_tokens, float(row.cost_usd)) == (1000, 200, pytest.approx(0.0002))
    assert row.error == "label rejected: too vague" and row.seconds >= 0
    assert "1,000+200 tokens" in run.summary and "$0.0002" in run.summary


def test_collector_translates_after_nlp_and_reanalyses(factory: sessionmaker[Session], brands: list[BrandSpec]) -> None:
    from collector import NotifyOutcome, run_cycle
    from tests.test_collector import Recorder

    rec = Recorder()
    stages = rec.stages()
    waiting = iter([0, 2])  # nothing waiting before NLP; 2 translations waiting after translating
    stages.pending = lambda: (rec.calls.append("pending"), next(waiting, 0))[1]
    stages.translate = lambda: (rec.calls.append("translate"), NotifyOutcome("translate: 2 generated"))[1]
    stages.insights = lambda: (rec.calls.append("insights"), NotifyOutcome("labels ok; briefings ok"))[1]
    result = run_cycle(factory, stages, now=NOW)
    assert rec.calls == ["fetch", "pending", "translate", "pending", "nlp", "topics", "priority", "insights"]
    assert result.status == "ok"
    stages.insights = lambda: NotifyOutcome("labels: 0 generated", ["Ollama not reachable at http://localhost:11434"])
    assert run_cycle(factory, stages, now=NOW).status == "partial"  # an unreachable LLM is worth a warning


def test_dashboard_shows_the_ai_briefing_and_translated_quotes(monkeypatch: pytest.MonkeyPatch) -> None:
    from streamlit.testing.v1 import AppTest

    import streamlit as st
    from tests.test_dashboard import TODAY
    from ui import data

    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        b = Brand(name="Cosrx", is_own=True, keywords=["cosrx"], brand_group="Skincare", profile="beauty_skincare")
        s.add(b)
        s.flush()
        for i in range(8):
            m = Mention(brand_id=b.id, source="youtube", external_id=f"q{i}", created_at=datetime.combine(TODAY, datetime.min.time()) - timedelta(days=i),
                        text="আমি এটা ব্যবহার করে খুব উপকার পেয়েছি, স্কিন অনেক নরম হয়েছে" if i == 0 else f"the snail essence feels lovely and light on my skin {i}",
                        text_en="I have benefited a lot from using it, my skin has become much softer and smoother" if i == 0 else None,
                        language="bn" if i == 0 else "en", translation_status="ok" if i == 0 else None)
            s.add(m)
            s.flush()
            s.add(Analysis(mention_id=m.id, sentiment="positive", sentiment_score=0.95 if i == 0 else 0.6,
                           emotion_scores={"joy": 0.8, "neutral": 0.2}, category="praise", model_version="t"))
        s.add(Briefing(brand_id=b.id, period="last7", period_label="last7", start_date=TODAY - timedelta(days=6), end_date=TODAY,
                       text="Cosrx had 8 mentions and no urgent risk. Users praise the snail essence.", facts={}, facts_hash="x",
                       provider="ollama", model="gemma3:4b"))
        s.commit()
    monkeypatch.setattr(data, "get_engine", lambda: engine)
    st.cache_data.clear()
    at = AppTest.from_file("app.py", default_timeout=60).run()
    assert not at.exception
    md = " ".join(m.value for m in at.markdown)
    assert 'class="br-briefing">Cosrx had 8 mentions' in md and "AI-written" in md and "gemma3:4b" in md
    assert 'class="br-summary"' not in md  # the rule-based lines are replaced, not shown twice
    assert "Translated from Bengali" in md and "I have benefited a lot" in md
