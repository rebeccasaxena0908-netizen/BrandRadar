"""Step 11, track any brand: request lifecycle, the collector's request job, research with and without an
LLM, ambiguous-name rules, review (approve / edit / cancel), one heavy job at a time, group removal and the
dashboard card. Everything is in-process: SQLite, scripted LLM, fake sources and the fake analyzer."""

from __future__ import annotations

import json
import threading
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import responses
import yaml
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from db.models import Analysis, Brand, BrandRequest, FetchRun, LlmRun, Mention
from manage_brands import load_yaml
from nlp.llm import research
from nlp.llm.profile_draft import SYSTEM as PROFILE_SYSTEM
from nlp.llm.provider import LLMError, LLMResult, NoneProvider
from nlp.pipeline import run_pipeline
from nlp.profiles import get_profile
from sources.base import BaseSource, BrandSpec, DeadlineExceeded, HttpClient, RawMention, deadline, utc_now
from tests.test_pipeline import FakeAnalyzer
from tracking import apps, store
from tracking import setup as setup_mod
from tracking.store import RequestError
from tracking.worker import Worker, delete_group_rows

# ---------------------------------------------------------------- fakes


class ScriptedLLM:
    """Answers by prompt kind: market, matching rules (per brand) or a profile draft."""

    name, model = "ollama", "fake:4b"

    def __init__(self, market: dict[str, Any] | Exception, rules: dict[str, dict[str, Any]] | None = None,
                 profile: dict[str, Any] | None = None) -> None:
        self.market, self.rules, self.profile = market, rules or {}, profile
        self.calls: list[str] = []
        self.unloaded = 0

    def json(self, system: str, prompt: str, schema: dict[str, Any], max_tokens: int = 1024) -> LLMResult:
        if system == research.MARKET_SYSTEM:
            self.calls.append("market")
            answer: Any = self.market
        elif system == research.RULES_SYSTEM:
            brand = prompt.splitlines()[0].removeprefix("Brand: ")
            self.calls.append(f"rules:{brand}")
            answer = self.rules.get(brand, LLMError("not JSON"))
        elif system == PROFILE_SYSTEM:
            self.calls.append("profile")
            answer = self.profile or LLMError("not JSON")
        else:
            raise AssertionError(system[:60])
        if isinstance(answer, Exception):
            raise answer
        return LLMResult(dict(answer), 0.01, 100, 40)

    def text(self, system: str, prompt: str, max_tokens: int = 1024) -> LLMResult:
        raise LLMError("no briefings in these tests")

    def unload(self) -> None:
        self.unloaded += 1


class FakeSource(BaseSource):
    """Returns the scripted texts for each brand, matched by the real brand rules."""

    def __init__(self, name: str, texts: dict[str, list[str]], delay: float = 0.0) -> None:
        super().__init__()
        self.name, self.texts, self.delay = name, texts, delay

    def fetch(self, brand: BrandSpec, since: Any) -> list[RawMention]:
        self.reset()
        time.sleep(self.delay)
        out = []
        for i, text in enumerate(self.texts.get(brand.name, [])):
            if self.accept(text, brand):
                out.append(RawMention(self.name, f"{brand.name}-{self.name}-{i}", text, utc_now() - timedelta(days=1)))
        return out


NIVEA_MARKET = {"known": True, "industry": "skincare", "profile": "beauty_skincare", "market": "body lotions",
                "competitors": ["Nivea", "Nivea Men", "Vaseline", "Dove", "Cetaphil", "Notion"], "has_app": False}
RULES = {
    "Nivea": {"spellings": ["Nivea", "NIVEA", "Niveaa cream"], "ambiguous": False, "other_meanings": [],
              "exclude_phrases": [], "drop_words": [], "context_words": ["cream"], "youtube_query": "Nivea soft cream review"},
    "Vaseline": {"spellings": ["Vaseline"], "ambiguous": True, "other_meanings": ["petroleum jelly as a generic word"],
                 "exclude_phrases": ["Vaseline glass", "glass"], "drop_words": ["Vaseline Intensive", "engine", "uranium glass"],
                 "context_words": ["lips", "skin", "Vaseline jelly"], "youtube_query": "Vaseline lotion review"},
    "Dove": {"spellings": ["Dove"], "ambiguous": True, "other_meanings": ["the bird"], "exclude_phrases": ["Dove Cameron"],
             "drop_words": ["bird", "pigeon"], "context_words": ["soap", "body wash"], "youtube_query": "Dove body wash review",
             "has_app": True},
}

TEXTS = {
    "Nivea": [f"Nivea soft cream is my winter favourite, I love it number {i}" for i in range(36)]
             + ["the nivea tin rusted", "NIVEA", "Niveau de vie, a French phrase"],
    "Vaseline": ["Vaseline on my lips every night, love it", "Vaseline glass is collectible", "a jar of vaseline in the garage"],
    "Dove": ["Dove soap dried my skin, I hate it", "a dove landed on the roof"],
}


BRANDS_YAML = """\
# test copy: the hand-made group (the project's brands.yaml grows as groups are tracked from the dashboard)
brands:
  - name: Notion
    group: Note apps
    profile: software
    is_own: true
    keywords: [Notion]

  - name: Obsidian
    group: Note apps
    profile: software
    keywords: [Obsidian]
"""


@pytest.fixture
def brands_file(tmp_path: Path) -> Path:
    path = tmp_path / "brands.yaml"
    path.write_text(BRANDS_YAML)
    return path


def make_worker(factory: sessionmaker[Session], brands_file: Path, provider: Any = None, delay: float = 0.0,
                texts: dict[str, list[str]] | None = None) -> Worker:
    def nlp(names: list[str]) -> None:
        run_pipeline(factory, FakeAnalyzer(), brands=names)

    return Worker(factory, brands_file=brands_file,
                  make_source=lambda name: FakeSource(name, TEXTS if texts is None else texts, delay),
                  get_provider=lambda: provider or NoneProvider(),
                  app_store=lambda name: None, play_store=lambda name: None,
                  nlp=nlp, topics=lambda names: None)


def request_row(factory: sessionmaker[Session], rid: int) -> BrandRequest:
    with factory() as s:
        row = s.get(BrandRequest, rid)
        assert row is not None
        return row


def new_request(factory: sessionmaker[Session], name: str) -> int:
    with factory() as s:
        return store.create_request(s, name).id


# ---------------------------------------------------------------- request lifecycle


def test_validation_and_the_in_progress_limit(factory: sessionmaker[Session], brands: list[BrandSpec]) -> None:
    with factory() as s:
        for bad, message in (("", "at least 2"), ("x" * 61, "too long"), ("<script>", "letters"),
                             ("notion", "already tracked"), ("Ob-sidian", "already tracked")):
            with pytest.raises(RequestError, match=message):
                store.validate_name(s, bad)
        assert store.validate_name(s, "  McDonald’s  ") == "McDonald’s"
        store.create_request(s, "McDonald's")
        with pytest.raises(RequestError, match="already being set up"):
            store.create_request(s, "mcdonalds")  # same brand, other spelling
        store.create_request(s, "Nivea")
        store.create_request(s, "Tesla")
        with pytest.raises(RequestError, match="at most 3"):
            store.create_request(s, "Zomato")


def test_demo_mode_refuses_requests(factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch) -> None:
    import dataclasses

    monkeypatch.setattr(store, "settings", dataclasses.replace(store.settings, demo=True))
    with factory() as s, pytest.raises(RequestError, match="real data only"):
        store.create_request(s, "Nivea")


def test_status_changes_are_conditional(factory: sessionmaker[Session]) -> None:
    rid = new_request(factory, "Nivea")
    with factory() as s:
        assert store.move(s, rid, store.QUEUED, store.RESEARCHING)
        assert not store.move(s, rid, store.QUEUED, store.RESEARCHING)  # already claimed: no second claim
        assert not store.approve(s, rid)  # only from review
        assert store.cancel(s, rid)  # a cancel during research wins…
        assert not store.move(s, rid, store.RESEARCHING, store.REVIEW)  # …the research result is dropped
    row = request_row(factory, rid)
    assert row.status == store.CANCELLED and row.finished_at is not None
    assert store.age(utc_now() - timedelta(minutes=5)) == "5 min ago" and store.age(None) == ""


# ---------------------------------------------------------------- research


def test_research_without_an_llm_asks_for_industry_and_competitors() -> None:
    llm = NoneProvider()
    proposed = setup_mod.propose("McDonald's", llm, tracked=["Notion"], groups=["Note apps"],
                                 app_store=lambda n: None, play_store=lambda n: None)
    assert proposed["needs_input"] and proposed["llm"] == "none" and proposed["profile"] is None
    assert proposed["group"] == "McDonald's and competitors"
    [own] = proposed["brands"]
    assert own["is_own"] and own["rules"] == "rules"
    assert own["keywords"] == ["McDonald's", "McDonalds", "McDonald’s"]  # rule-based spellings, both apostrophes
    assert own["youtube_queries"] == ["McDonald's review"] and own["exclude_if"] == []
    assert any("pick the industry profile" in n for n in proposed["notes"])
    problems = setup_mod.problems(proposed, ["Notion"], ["Note apps"])
    assert "Pick an industry profile." in problems and "Add at least one competitor." in problems


def test_research_with_an_llm_proposes_a_checked_group() -> None:
    llm = ScriptedLLM(NIVEA_MARKET, RULES)
    found = apps.AppMatch("appstore", "123", "Dove Men+Care", "Unilever")
    proposed = setup_mod.propose("Nivea", llm, tracked=["Notion", "Obsidian"], groups=["Note apps"],
                                 app_store=lambda n: found if n == "Dove" else None, play_store=lambda n: None)
    assert not proposed["needs_input"] and proposed["profile"] == "beauty_skincare" and proposed["profile_draft"] is None
    assert proposed["group"] == "Body lotions"  # Sentence case
    names = [b["name"] for b in proposed["brands"]]
    assert names == ["Nivea", "Vaseline", "Dove", "Cetaphil"]  # itself, its own line and tracked brands left out
    assert any("Notion is already tracked" in n for n in proposed["notes"])
    nivea, vaseline, dove, cetaphil = proposed["brands"]
    assert nivea["keywords"] == ["nivea"] and nivea["require_context"] == []  # not ambiguous: any case, no context
    assert nivea["youtube_queries"] == ["Nivea review", "Nivea soft cream review"]
    assert cetaphil["rules"] == "rules"  # its LLM answer failed: rule-based fallback for that brand only
    assert dove["app_store_id"] == "123" and dove["app_store_title"] == "Dove Men+Care · Unilever"
    assert llm.calls == ["market", "rules:Nivea", "rules:Vaseline", "rules:Dove", "rules:Cetaphil"] and llm.unloaded
    assert setup_mod.problems(proposed, ["Notion"], ["Note apps"]) == []


def test_an_unknown_industry_gets_a_drafted_profile() -> None:
    market = {**NIVEA_MARKET, "profile": "none", "industry": "fast food", "market": "Burger chains",
              "competitors": ["Burger King", "KFC", "Wendy's"]}
    draft = {"name": "fast_food", "praise_hypothesis": "praise for the food or the restaurant",
             "request_hypothesis": "a wish for a new menu item", "complaint_categories": [
                 {"label": l, "hypothesis": f"complaints about the {l} at the restaurant"}
                 for l in ("order accuracy", "wait time", "food quality", "staff", "price/value")],
             "features": [{"name": f, "synonyms": [f"{f}s", f"the {f}"]} for f in ("fries", "burger", "app", "drive-thru", "delivery")]}
    llm = ScriptedLLM(market, {}, draft)
    proposed = setup_mod.propose("McDonald's", llm, tracked=[], groups=[], app_store=lambda n: None,
                                 play_store=lambda n: None)
    assert proposed["profile"] == "fast_food" and proposed["profile_draft"]["name"] == "fast_food"
    assert "order accuracy" in proposed["profile_draft"]["raw"]["complaints"]
    assert setup_mod.setup_profile(proposed).name == "fast_food"  # usable before it is saved
    assert setup_mod.problems(proposed, [], []) == []
    assert setup_mod.draft_of(proposed) is not None


def test_list_fields_come_as_strings_or_lists() -> None:
    # in a JSON array gemma closed "Wendy’s" with a curly quote and listed 30 chains until the token limit,
    # so the schema asks for comma-separated strings; the model decorates them now and then
    assert research.as_list("[Burger King, Wendy’s, Taco Bell]") == ["Burger King", "Wendy’s", "Taco Bell"]
    assert research.as_list("%[Pizza Hut]%") == ["Pizza Hut"]
    assert research.as_list("+ Uber Eats\n+ DoorDash\n- Grubhub") == ["Uber Eats", "DoorDash", "Grubhub"]
    assert research.as_list("1. 7-Eleven, 3M, “L’Oréal”") == ["7-Eleven", "3M", "L’Oréal"]
    assert research.as_list(["a", " b "]) == ["a", "b"] and research.as_list(None) == []
    assert research.as_list("the bird; a surname, in Spanish", ";\n") == ["the bird", "a surname, in Spanish"]
    m = research.check_market({"known": True, "industry": "fast food", "profile": "food_beverage", "market": "Burger chains",
                               "competitors": "[Burger King, Wendy’s, KFC, McDonald’s app]"}, "McDonald's", {"food_beverage": ""}, set())
    assert m.competitors == ["Burger King", "Wendy's", "KFC"]


def test_market_answers_are_checked() -> None:
    profiles = setup_mod.profile_summaries()
    assert "consumer_product" not in profiles and "beauty_skincare" in profiles  # the generic one isn't offered
    m = research.check_market({"known": False, "industry": "Electric Cars", "profile": "made_up",
                               "market": "Tesla rivals", "competitors": ["Tesla Model 3", "BYD", "BYD", "Rivian"]},
                              "Tesla", profiles, tracked={"rivian"})
    assert m.industry == "electric cars" and m.profile is None and m.market is None  # named the brand
    assert m.competitors == ["BYD"]  # its own product, a duplicate and a tracked brand left out
    assert any("does not recognise" in n for n in m.notes)


# ---------------------------------------------------------------- ambiguous names


def test_rules_for_ambiguous_names_keep_only_what_does_its_job() -> None:
    tesla = research.check_rules({
        "spellings": ["Tesla", "Tesla Motors", "TSLA"], "ambiguous": True,
        "other_meanings": ["Nikola Tesla, the inventor", "the tesla, a unit of magnetic flux"],
        "exclude_phrases": ["Tesla coil", "Nikola Tesla", "coil"],  # "coil" can't be cut out: no name in it
        "drop_words": ["physicist", "Tesla Motors", "MRI"], "context_words": ["car"], "youtube_query": "Tesla Model Y review",
    }, "Tesla", industry_words=set())
    assert tesla.keywords == ["Tesla"]  # capitals kept (ambiguous); "Tesla Motors" is no spelling, TSLA not the name
    assert tesla.exclude == ["Tesla coil", "Nikola Tesla"]
    assert tesla.exclude_if == ["physicist", "MRI"]  # a drop word naming the brand would drop real posts
    assert tesla.require_context == ["car"]
    assert tesla.youtube_queries == ["Tesla review", "Tesla Model Y review"]

    words = setup_mod.industry_words(get_profile("beauty_skincare"))
    product_word = sorted(words)[0]
    vaseline = research.check_rules({**RULES["Vaseline"], "drop_words": ["engine", product_word]}, "Vaseline", words)
    assert vaseline.require_context == ["lips", "skin"]  # "Vaseline jelly" names the brand: not a context word
    assert vaseline.exclude == ["Vaseline glass"] and vaseline.exclude_if == ["engine"]
    assert any(product_word in n for n in vaseline.notes)  # words about its own products are never drop words

    dominos = research.check_rules({"spellings": ["Dominos", "Domino’s", "Dominoes", "Domino's Pizza"], "ambiguous": True,
                                    "other_meanings": ["the game of dominoes"],
                                    "exclude_phrases": [], "drop_words": ["game", "tiles", "Dominos game"],
                                    "context_words": ["pizza", "order", "delivery"], "youtube_query": ""}, "Domino's", set())
    assert dominos.keywords == ["Domino's", "Dominos", "Domino’s"]  # "Dominoes" is the game, not a spelling
    assert dominos.exclude_if == ["game", "tiles"] and dominos.require_context == ["pizza", "order", "delivery"]
    assert dominos.youtube_queries == ["Domino's review"]

    apple = research.check_rules({"spellings": ["Apple"], "ambiguous": True, "other_meanings": ["the fruit"],
                                  "exclude_phrases": ["apple pie", "Big Apple"],
                                  "drop_words": ["orchard", "fruit"], "context_words": ["iphone", "mac", "ios"]}, "Apple", set())
    assert apple.exclude == ["apple pie", "Big Apple"] and apple.require_context == ["iphone", "mac", "ios"]
    # not ambiguous: a parroted "other meaning" and its filters are ignored, any case matches
    mcd = research.check_rules({"spellings": ["McDonald's", "McDonalds"], "ambiguous": False, "other_meanings": ["Nikola"],
                                "exclude_phrases": [], "drop_words": ["Nikola"], "context_words": ["burger", "fries"]},
                               "McDonald's", set())
    assert mcd.keywords == ["mcdonald's", "mcdonalds", "mcdonald’s"]
    assert (mcd.exclude_if, mcd.require_context, mcd.other_meanings) == ([], [], [])
    assert any("not ambiguous" in n for n in mcd.notes)
    assert research.fallback_rules("Ben & Jerry's").keywords == ["Ben & Jerry's", "Ben & Jerrys", "Ben&Jerrys", "Ben&Jerry's", "Ben & Jerry’s", "Ben&Jerry’s"]


def test_the_rules_actually_filter_mentions(brands_file: Path) -> None:
    proposed = setup_mod.propose("Nivea", ScriptedLLM(NIVEA_MARKET, RULES), [], [], lambda n: None, lambda n: None)
    specs = {e.name: e.spec() for e in setup_mod.entries(proposed)}
    src = FakeSource("news", TEXTS)
    kept = {name: [m.text for m in src.fetch(spec, None)] for name, spec in specs.items() if name in TEXTS}
    assert kept["Vaseline"] == ["Vaseline on my lips every night, love it"]  # the glass and the garage jar are gone
    assert kept["Dove"] == ["Dove soap dried my skin, I hate it"]  # the bird is gone


# ---------------------------------------------------------------- apps


def test_app_ids_only_for_the_brands_own_app(client: HttpClient) -> None:
    assert apps.is_brand_app("McDonald's", "McDonald's", "McDonald's USA, LLC")
    assert apps.is_brand_app("Tesla", "Tesla", "Tesla, Inc.")
    assert not apps.is_brand_app("McDonald's", "Coupons for McDonald's", "Deal Apps Ltd")
    assert not apps.is_brand_app("Dove", "Dove Medical Press", "Dove Medical Press Ltd")  # more than company words
    assert not apps.is_brand_app("Dove", "Dove Channel - Family Shows", "Dove Family Channel LLC")
    assert apps.is_brand_app("McDonald's", "McDonald's", "McDonald's Apps")
    # and research asks the LLM whether the brand has an app at all
    looked_up: list[str] = []
    brand = {"name": "Dove", "has_app": False}
    setup_mod.add_apps(brand, lambda n: looked_up.append(n), lambda n: looked_up.append(n))
    assert looked_up == [] and "app_store_id" not in brand
    with responses.RequestsMock() as rsps:
        rsps.add(responses.GET, apps.ITUNES_SEARCH, json={"results": [
            {"trackId": 1, "trackName": "Coupons for McDonald's", "sellerName": "Deal Apps"},
            {"trackId": 922103212, "trackName": "McDonald's", "sellerName": "McDonald's USA, LLC"}]})
        match = apps.find_app_store("McDonald's", client)
    assert match is not None and (match.app_id, match.developer) == ("922103212", "McDonald's USA, LLC")
    hits = [{"appId": "com.mcdonalds.app", "title": "McDonald's", "developer": "McDonald's"}]
    assert apps.find_play_store("McDonald's", search=lambda *a, **k: hits).app_id == "com.mcdonalds.app"
    assert apps.find_play_store("Nivea", search=lambda *a, **k: []) is None


# ---------------------------------------------------------------- the request job


def test_polling_picks_a_request_up_once_and_proposes_a_setup(factory: sessionmaker[Session], brands: list[BrandSpec],
                                                              brands_file: Path) -> None:
    rid = new_request(factory, "Nivea")
    worker = make_worker(factory, brands_file, ScriptedLLM(NIVEA_MARKET, RULES))
    summary = worker.tick()
    assert summary and "proposed Body lotions (4 brands)" in summary
    row = request_row(factory, rid)
    assert row.status == store.REVIEW and row.group_name == "Body lotions" and row.detail == "Waiting for your review"
    assert worker.tick() is None  # nothing left to do: it was picked up once
    preview = row.preview
    assert preview["brands"]["Nivea"]["news"] == {"matched": 38, "rejected": 1}  # any case: "the nivea tin", NIVEA
    assert len(preview["matched"]) == setup_mod.SHOW and preview["rejected"][0][1]
    assert "youtube" in preview["brands"]["Nivea"] and "youtube" not in preview["brands"]["Vaseline"]  # 1 search call
    with factory() as s:
        assert s.scalar(select(LlmRun.stage).order_by(LlmRun.id.desc())) == "research"  # logged with its timing


def test_approve_writes_brands_yaml_then_fetches_and_analyses_only_the_new_group(
        factory: sessionmaker[Session], brands: list[BrandSpec], brands_file: Path) -> None:
    rid = new_request(factory, "Nivea")
    worker = make_worker(factory, brands_file, ScriptedLLM(NIVEA_MARKET, RULES))
    worker.tick()
    with factory() as s:
        assert store.approve(s, rid)
    summary = worker.tick()
    assert summary and "Ready" in summary
    row = request_row(factory, rid)
    assert row.status == store.READY and row.applied_at is not None and row.finished_at is not None
    entries = {e.name: e for e in load_yaml(brands_file)}
    assert entries["Nivea"].is_own and entries["Nivea"].group == "Body lotions"
    assert entries["Vaseline"].require_context == ["lips", "skin"] and entries["Dove"].exclude_if == ["bird", "pigeon"]
    assert "# Body lotions: added from the dashboard" in brands_file.read_text()
    result = row.result
    assert result["mentions"]["Nivea"] == {"news": 38, "youtube": 38, "mastodon": 38, "reddit": 38}
    assert result["analysed"]["Nivea"] == 152 and result["mentions"]["Cetaphil"] == {}
    assert set(result["seconds"]) == {"fetch", "nlp", "translate", "topics", "ai"}  # an LLM is on
    with factory() as s:
        fetched = {b for (b,) in s.execute(select(Brand.name).join(FetchRun, FetchRun.brand_id == Brand.id))}
        assert fetched == {"Nivea", "Vaseline", "Dove", "Cetaphil"}  # nothing fetched for Notion or Obsidian
        assert s.scalar(select(Brand.brand_group).where(Brand.name == "Nivea")) == "Body lotions"


def test_little_data_ends_as_not_enough_data(factory: sessionmaker[Session], brands: list[BrandSpec],
                                             brands_file: Path) -> None:
    rid = new_request(factory, "Nivea")
    worker = make_worker(factory, brands_file, ScriptedLLM(NIVEA_MARKET, RULES), texts={"Nivea": ["Nivea is fine"]})
    worker.tick()
    with factory() as s:
        store.approve(s, rid)
    worker.tick()
    row = request_row(factory, rid)
    assert row.status == store.NOT_ENOUGH and "Not enough data yet: 4 analysed mentions" in row.detail
    assert row.result["mentions"]["Nivea"] == {"news": 1, "youtube": 1, "mastodon": 1, "reddit": 1}


def test_edit_adds_a_competitor_and_rechecks_without_an_llm(factory: sessionmaker[Session], brands: list[BrandSpec],
                                                            brands_file: Path) -> None:
    rid = new_request(factory, "Nivea")
    worker = make_worker(factory, brands_file)  # no LLM
    worker.tick()
    row = request_row(factory, rid)
    assert row.status == store.REVIEW and row.setup["needs_input"]
    assert "pick the industry and competitors" in row.detail
    edited = {**row.setup, "profile": "beauty_skincare", "group": "Body lotions",
              "brands": [*row.setup["brands"], {"name": "Vaseline", "is_own": False, "keywords": []}]}
    with factory() as s:
        assert store.recheck(s, rid, edited)
    worker.tick()
    row = request_row(factory, rid)
    assert row.status == store.REVIEW and not row.setup["needs_input"]
    vaseline = row.setup["brands"][1]
    assert vaseline["keywords"] == ["Vaseline"] and vaseline["rules"] == "rules"  # filled in by the re-check
    assert row.preview["brands"]["Vaseline"]["news"]["matched"] == 2
    assert setup_mod.problems(row.setup, ["Notion"], ["Note apps"]) == []


def test_cancel_drops_the_request_and_writes_nothing(factory: sessionmaker[Session], brands: list[BrandSpec],
                                                     brands_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    before = brands_file.read_text()
    rid = new_request(factory, "Nivea")
    original = setup_mod.propose

    def cancelled_meanwhile(*args: Any, **kwargs: Any) -> dict[str, Any]:
        with factory() as s:
            store.cancel(s, rid)  # the person clicks Cancel while the collector researches
        return original(*args, **kwargs)

    monkeypatch.setattr(setup_mod, "propose", cancelled_meanwhile)
    summary = make_worker(factory, brands_file).tick()
    assert summary and "cancelled during research" in summary
    assert request_row(factory, rid).status == store.CANCELLED and brands_file.read_text() == before


def test_an_approved_setup_that_became_invalid_goes_back_to_review(factory: sessionmaker[Session], brands: list[BrandSpec],
                                                                   brands_file: Path) -> None:
    rid = new_request(factory, "Nivea")
    worker = make_worker(factory, brands_file, ScriptedLLM(NIVEA_MARKET, RULES))
    worker.tick()
    row = request_row(factory, rid)
    with factory() as s:
        store.approve(s, rid, {**row.setup, "group": "Note apps"})  # an existing group
    assert "sent back to review" in (worker.tick() or "")
    row = request_row(factory, rid)
    assert row.status == store.REVIEW and "exists already" in row.error


def test_only_one_heavy_job_at_a_time(factory: sessionmaker[Session], brands: list[BrandSpec], brands_file: Path) -> None:
    import collector

    rid = new_request(factory, "Nivea")
    worker = make_worker(factory, brands_file)
    with worker.heavy:  # the 6-hour cycle is running
        assert worker.tick() is None
    row = request_row(factory, rid)
    assert row.status == store.QUEUED and "Waiting for the collector's 6-hourly cycle" in row.detail
    assert worker.tick()  # free again: picked up
    assert collector.request_worker(factory).heavy is collector.HEAVY  # the cycle and the job share one lock

    order: list[str] = []
    with collector.HEAVY:
        t = threading.Thread(target=lambda: (collector.HEAVY.acquire(), order.append("cycle"), collector.HEAVY.release()))
        t.start()
        time.sleep(0.05)
        order.append("request step done")
    t.join(2)
    assert order == ["request step done", "cycle"]  # the cycle waited for the heavy step


def test_a_slow_source_cannot_hold_a_request_up(client: HttpClient) -> None:
    from fetch import call_with_deadline

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="no answer within"):
        call_with_deadline(lambda: time.sleep(3), 0.1, "news", grace=0.1)
    assert time.monotonic() - started < 1
    with deadline(0), pytest.raises(DeadlineExceeded):
        client.get("https://api.gdeltproject.org/api/v2/doc/doc")  # budget gone: fails before any request
    assert call_with_deadline(lambda: 42, 5) == 42


# ---------------------------------------------------------------- removal


def test_removing_a_dashboard_group(factory: sessionmaker[Session], brands: list[BrandSpec], brands_file: Path) -> None:
    rid = new_request(factory, "Nivea")
    worker = make_worker(factory, brands_file, ScriptedLLM(NIVEA_MARKET, RULES))
    worker.tick()
    with factory() as s:
        store.approve(s, rid)
    worker.tick()
    with factory() as s:
        assert store.removable_groups(s) == ["Body lotions"]  # Note apps came from brands.yaml: not removable here
        with pytest.raises(RequestError, match="Type the group name"):
            store.request_removal(s, "Body lotions", "body lotions")
        with pytest.raises(RequestError, match="only groups added from the dashboard"):
            store.request_removal(s, "Note apps", "Note apps")
        removal = store.request_removal(s, "Body lotions", "Body lotions")
        assert store.removable_groups(s) == []  # pending
        nivea_mentions = s.scalar(select(func.count()).select_from(Mention)) or 0
    summary = worker.tick()
    assert summary and "Deleted Body lotions: 4 brands" in summary
    with factory() as s:
        # approving synced brands.yaml into this test database; only the removed group is gone
        assert {b for (b,) in s.execute(select(Brand.name))} == {"Notion", "Obsidian"}
        assert s.scalar(select(func.count()).select_from(Mention)) == 0 < nivea_mentions
        assert s.scalar(select(func.count()).select_from(Analysis)) == 0
        assert s.scalar(select(func.count()).select_from(FetchRun).where(FetchRun.brand_id.is_not(None))) == 0
    assert request_row(factory, removal.id).status == store.REMOVED
    assert request_row(factory, rid).status == store.REMOVED  # the track request says so too
    text = brands_file.read_text()
    assert "Body lotions" not in text and "Vaseline" not in text
    assert [e.name for e in load_yaml(brands_file)] == ["Notion", "Obsidian"]
    assert brands_file.read_text() == BRANDS_YAML  # the file is exactly as before


def test_delete_group_rows_leaves_other_groups_alone(factory: sessionmaker[Session], brands: list[BrandSpec]) -> None:
    with factory() as s:
        s.add(Mention(brand_id=brands[0].id, source="news", external_id="n1", text="Notion", created_at=utc_now()))
        s.commit()
        assert delete_group_rows(s, "Nothing here") == {"brands": 0, "mentions": 0}
        assert s.scalar(select(func.count()).select_from(Mention)) == 1


# ---------------------------------------------------------------- dashboard card


def submit(at: Any) -> None:
    next(b for b in at.button if b.label == "Track").click()


@pytest.fixture
def card(factory: sessionmaker[Session], brands: list[BrandSpec], monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    from streamlit.testing.v1 import AppTest

    from ui import data

    monkeypatch.setattr(data, "get_engine", lambda: factory.kw["bind"])

    def page() -> None:
        from ui.tracking import track_card

        track_card()

    return AppTest.from_function(page, default_timeout=30)


def test_card_queues_a_request_and_reports_errors(card, factory: sessionmaker[Session]) -> None:  # type: ignore[no-untyped-def]
    at = card.run()
    assert not at.exception
    assert any("No brands requested yet" in m.value for m in at.markdown)
    at.text_input(key="track-name").input("Nivea")
    submit(at)
    at = at.run()
    assert any("Nivea is queued" in m.value for m in at.markdown)
    assert any("br-track-list" in m.value and "Nivea" in m.value and "Queued" in m.value for m in at.markdown)
    at.text_input(key="track-name").input("Notion")
    submit(at)
    at = at.run()
    assert any("Notion is already tracked" in m.value for m in at.markdown)
    with factory() as s:
        assert [r.name for r in s.scalars(select(BrandRequest))] == ["Nivea"]


def test_card_review_approve_and_cancel(card, factory: sessionmaker[Session], brands_file: Path) -> None:  # type: ignore[no-untyped-def]
    rid = new_request(factory, "Nivea")
    make_worker(factory, brands_file, ScriptedLLM(NIVEA_MARKET, RULES)).tick()
    at = card.run()
    assert not at.exception
    md = " ".join(m.value for m in at.markdown)
    assert "Review: <b>Nivea</b>" in md and "Body lotions" in md and "Vaseline glass" in md  # setup and rules
    assert "Dry run, last 30 days" in md and "Matched for Nivea" in md
    at.button(key=f"track-approve-{rid}").click()
    at = at.run()
    assert request_row(factory, rid).status == store.APPROVED
    assert any("Nivea approved" in m.value for m in at.markdown)

    other = new_request(factory, "Vaseline")
    with factory() as s:
        store.move(s, other, store.QUEUED, store.REVIEW, setup=request_row(factory, rid).setup, preview={})
    at = card.run()
    at.button(key=f"track-cancel-{other}").click()
    at.run()
    assert request_row(factory, other).status == store.CANCELLED


def test_a_request_waiting_for_input_can_be_cancelled(card, factory: sessionmaker[Session], brands_file: Path) -> None:  # type: ignore[no-untyped-def]
    rid = new_request(factory, "Nivea")
    make_worker(factory, brands_file).tick()  # no LLM: the review opens on the edit form
    at = card.run()
    assert any("Pick the industry profile" in m.value for m in at.markdown)
    at.button(key=f"track-cancel-{rid}").click()
    at.run()
    assert request_row(factory, rid).status == store.CANCELLED


def test_card_shows_results_and_removal(card, factory: sessionmaker[Session], brands_file: Path) -> None:  # type: ignore[no-untyped-def]
    rid = new_request(factory, "Nivea")
    worker = make_worker(factory, brands_file, ScriptedLLM(NIVEA_MARKET, RULES))
    worker.tick()
    with factory() as s:
        store.approve(s, rid)
    worker.tick()
    at = card.run()
    md = " ".join(m.value for m in at.markdown)
    assert "Ready" in md and "br-track-table" in md and "Analysed" in md
    assert at.selectbox(key="track-remove-group").options == ["Body lotions"]
    assert at.button(key="track-remove-go").disabled  # until the name is typed
    at.text_input(key="track-remove-confirm").input("Body lotions")
    at = at.run()
    at.button(key="track-remove-go").click()
    at.run()
    with factory() as s:
        assert s.scalar(select(BrandRequest.status).where(BrandRequest.kind == store.REMOVE)) == store.QUEUED


def test_setup_is_plain_json() -> None:
    proposed = setup_mod.propose("Nivea", ScriptedLLM(NIVEA_MARKET, RULES), [], [], lambda n: None, lambda n: None)
    assert json.loads(json.dumps(proposed)) == proposed
    assert yaml.safe_load(yaml.safe_dump(proposed)) == proposed
