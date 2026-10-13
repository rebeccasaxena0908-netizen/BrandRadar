from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
import requests
import responses
from urllib3.util.retry import Retry

from sources.base import (
    BrandSpec,
    HttpClient,
    RateLimiter,
    html_to_text,
    make_session,
    matches_brand,
    split_keywords,
    to_utc_naive,
    with_retries,
)

NOTION = BrandSpec(id=1, name="Notion", keywords=("Notion", "notion.so"), exclude=("Notion Ink", "Notion of"))
OBSIDIAN = BrandSpec(
    id=2, name="Obsidian", keywords=("Obsidian",), exclude=("Obsidian Entertainment", "Obsidian's games"),
    exclude_if=("Fallout", "The Outer Worlds"),
)
CLICKUP = BrandSpec(id=3, name="ClickUp", keywords=("clickup",))


def test_html_to_text_strips_tags_and_entities() -> None:
    html = "I don&#x27;t like <i>Notion</i>.<p>Sync is &gt; slow &amp; <a href=\"x\">buggy</a>"
    assert html_to_text(html) == "I don't like Notion.\nSync is > slow & buggy"


def test_html_to_text_empty() -> None:
    assert html_to_text(None) == ""
    assert html_to_text("") == ""


@pytest.mark.parametrize(
    ("text", "brand", "expected"),
    [
        ("I moved my notes to Notion last year", NOTION, True),
        ("the notion that apps should sync is old", NOTION, False),  # lowercase word
        ("Notion of free will is contested", NOTION, False),  # excluded phrase
        ("Notion Ink made a tablet", NOTION, False),
        ("Notion Ink tablet vs the real Notion app", NOTION, True),  # still mentioned after exclusion
        ("see notion.so/pricing", NOTION, True),
        ("Notional value of the bond", NOTION, False),  # whole words only
        ("Obsidian Entertainment shipped a game", OBSIDIAN, False),
        ("Obsidian sync is great", OBSIDIAN, True),
        ("an obsidian blade", OBSIDIAN, False),
        ("Obsidian made Fallout: New Vegas", OBSIDIAN, False),  # exclude_if drops the whole mention
        ("I loved The Outer Worlds by Obsidian", OBSIDIAN, False),
        ("Obsidian's games are buggy", OBSIDIAN, False),
        ("my obsidian vault has a fallout shelter note", OBSIDIAN, False),  # lowercase 'obsidian' never matched
        ("Clickup is slow", CLICKUP, True),  # lowercase keyword matches any case
        ("CLICKUP is slow", CLICKUP, True),
    ],
)
def test_matches_brand(text: str, brand: BrandSpec, expected: bool) -> None:
    assert matches_brand(text, brand) is expected


def test_split_keywords() -> None:
    assert split_keywords(["Notion", "-Notion Ink", "notion.so", "!Fallout"]) == (
        ("Notion", "notion.so"), ("Notion Ink",), ("Fallout",),
    )


def test_to_utc_naive() -> None:
    aware = datetime(2026, 10, 9, 16, 0, tzinfo=timezone(timedelta(hours=-7)))
    assert to_utc_naive(aware) == datetime(2026, 10, 9, 23, 0)


def test_rate_limiter_waits_between_calls_to_same_host() -> None:
    now = [100.0]
    slept: list[float] = []

    def sleep(s: float) -> None:
        slept.append(s)
        now[0] += s

    limiter = RateLimiter(1.0, clock=lambda: now[0], sleep=sleep)
    limiter.wait("a.com")
    now[0] += 0.25
    limiter.wait("a.com")
    limiter.wait("b.com")  # other host: no wait
    assert slept == [pytest.approx(0.75)]


def test_session_has_user_agent_and_retry() -> None:
    session = make_session(retries=4, user_agent="TestAgent/1")
    assert session.headers["User-Agent"] == "TestAgent/1"
    retry = session.get_adapter("https://example.com").max_retries
    assert retry.total == 4
    assert 429 in retry.status_forcelist and 503 in retry.status_forcelist


@responses.activate
def test_http_client_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Retry, "sleep", lambda self, response=None: None)
    responses.get("https://api.test/x", status=503)
    responses.get("https://api.test/x", json={"ok": True})
    client = HttpClient(session=make_session(retries=2), limiter=RateLimiter(0), timeout=5)
    assert client.get_json("https://api.test/x") == {"ok": True}
    assert len(responses.calls) == 2


@responses.activate
def test_http_client_raises_after_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Retry, "sleep", lambda self, response=None: None)
    responses.get("https://api.test/x", status=500)
    client = HttpClient(session=make_session(retries=1), limiter=RateLimiter(0), timeout=5)
    with pytest.raises(requests.HTTPError):
        client.get("https://api.test/x")


def test_with_retries_backoff() -> None:
    calls = {"n": 0}
    sleeps: list[float] = []

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("boom")
        return "ok"

    assert with_retries(flaky, attempts=3, backoff=1.0, sleep=sleeps.append) == "ok"
    assert sleeps == [1.0, 2.0]


def test_with_retries_gives_up() -> None:
    def broken() -> None:
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        with_retries(broken, attempts=2, sleep=lambda s: None)


def test_still_matches_mirrors_each_source() -> None:
    from sources.base import still_matches

    brand = BrandSpec(id=1, name="Obsidian", keywords=("Obsidian",), exclude_if=("Fallout",), subreddits=("ObsidianMD",))
    assert still_matches("devto", "My PKM setup", "https://dev.to/x", brand)  # tag-based: no name needed
    assert not still_matches("devto", "Fallout mods", "https://dev.to/x", brand)  # exclude_if always applies
    own = "https://www.reddit.com/r/ObsidianMD/comments/abc/x/"
    assert still_matches("reddit", "sync broke again", own, brand)
    assert not still_matches("reddit", "sync broke again", "https://www.reddit.com/r/productivity/comments/a/x/", brand)
    assert still_matches("mastodon", "Moving to #Obsidian", None, brand)
    assert not still_matches("hackernews", "an obsidian blade", None, brand)


def test_subreddits_stored_with_prefix_stay_out_of_keywords() -> None:
    from sources.base import split_subreddits

    stored = ["Notion", "-Notion Ink", "!Ionian", "r/Notion"]
    assert split_keywords(stored)[0] == ("Notion",)
    assert split_subreddits(stored) == ("Notion",)


def test_cap_per_author_keeps_the_newest_three() -> None:
    from datetime import datetime

    from sources.base import RawMention, cap_per_author

    def m(i: int, author: str | None) -> RawMention:
        return RawMention(source="devto", external_id=str(i), text="x", created_at=datetime(2026, 10, i + 1), author=author)

    mentions = [m(i, "maker") for i in range(6)] + [m(10, "reader"), m(11, None), m(12, None), m(13, None), m(14, None)]
    kept = cap_per_author(mentions)
    assert sorted(x.external_id for x in kept if x.author == "maker") == ["3", "4", "5"]  # newest three
    assert sum(x.author is None for x in kept) == 4  # anonymous items are never capped
    assert any(x.author == "reader" for x in kept)


def test_reject_reasons() -> None:
    from sources.base import reject_reason

    doms = BrandSpec(id=1, name="DOMS", keywords=("DOMS",), exclude=("DOMS Recruitment",), exclude_if=("leg day",),
                     require_context=("pencil", "stationery"))
    assert reject_reason("DOMS pencils are smooth", doms) is None  # simple plural of a context word
    assert reject_reason("DOMS after leg day, pencil in hand", doms) == "exclude_if: leg day"
    assert reject_reason("DOMS Recruitment drive for pencil plant", doms) == "only an excluded phrase"
    assert reject_reason("Delayed onset muscle soreness", doms) == "name not found"
    assert reject_reason("DOMS shares rose today", doms) == "no context word"
