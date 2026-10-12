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
OBSIDIAN = BrandSpec(id=2, name="Obsidian", keywords=("Obsidian",), exclude=("Obsidian Entertainment",))
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
        ("Clickup is slow", CLICKUP, True),  # lowercase keyword matches any case
        ("CLICKUP is slow", CLICKUP, True),
    ],
)
def test_matches_brand(text: str, brand: BrandSpec, expected: bool) -> None:
    assert matches_brand(text, brand) is expected


def test_split_keywords() -> None:
    assert split_keywords(["Notion", "-Notion Ink", "notion.so"]) == (("Notion", "notion.so"), ("Notion Ink",))


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
