"""News source (GDELT) with mocked HTTP (responses)."""

from __future__ import annotations

from datetime import datetime
from urllib.parse import parse_qs, urlparse

import responses

from sources.base import BrandSpec, HttpClient, RateLimiter, make_session
from sources.news import GDELT_URL, NewsSource, external_id, gdelt_query

DOMS = BrandSpec(id=1, name="DOMS", keywords=("DOMS",), require_context=("pencil", "stationery", "Industries"),
                 exclude_if=("muscle soreness",))
SINCE = datetime(2026, 10, 1)


def article(title: str, url: str, seen: str = "20261010T080000Z") -> dict:
    return {"url": url, "title": title, "seendate": seen, "domain": "example.com", "language": "English"}


def source(sleeps: list[float] | None = None) -> NewsSource:
    client = HttpClient(make_session(retries=0), RateLimiter(0))
    return NewsSource(client=client, sleep=(sleeps.append if sleeps is not None else lambda s: None))


def test_query_quotes_phrases_and_ors_keywords() -> None:
    assert gdelt_query(DOMS) == '"DOMS" sourcelang:english'
    two = BrandSpec(id=2, name="Laneige", keywords=("Laneige", "Laneige lip"))
    assert gdelt_query(two) == '("Laneige" OR "Laneige lip") sourcelang:english'


@responses.activate
def test_headlines_are_matched_like_any_mention() -> None:
    responses.add(responses.GET, GDELT_URL, json={"articles": [
        article("DOMS Industries shares jump after strong quarter for pencil sales", "https://a.example/1"),
        article("DOMS: why your legs hurt after the gym (muscle soreness explained)", "https://a.example/2"),
        article("DOMS after leg day? Try these stretches", "https://a.example/3"),  # no context word
        article("Old news about DOMS stationery", "https://a.example/4", seen="20260901T080000Z"),  # before since
        article("DOMS Industries shares jump after strong quarter for pencil sales", "https://a.example/1"),  # dup
    ]})
    news = source()
    found = news.fetch(DOMS, SINCE)
    assert [m.url for m in found] == ["https://a.example/1"]
    assert found[0].text.startswith("DOMS Industries") and found[0].created_at == datetime(2026, 10, 10, 8)
    assert found[0].author == "example.com" and found[0].source == "news"
    reasons = {r for _, r in news.rejected}
    assert reasons == {"exclude_if: muscle soreness", "no context word"}
    params = parse_qs(urlparse(responses.calls[0].request.url).query)
    assert params["mode"] == ["artlist"] and params["startdatetime"][0].startswith("20261001")


@responses.activate
def test_rate_limit_retries_once_then_warns() -> None:
    responses.add(responses.GET, GDELT_URL, status=429, body="Please limit requests to one every 5 seconds")
    sleeps: list[float] = []
    news = source(sleeps)
    assert news.fetch(DOMS, SINCE) == []
    assert len(responses.calls) == 2 and sleeps == [15.0]
    assert news.warnings and "rate-limiting" in news.warnings[0]


@responses.activate
def test_retry_after_429_can_succeed_and_plain_text_errors_warn() -> None:
    responses.add(responses.GET, GDELT_URL, status=429, body="slow down")
    responses.add(responses.GET, GDELT_URL, json={"articles": [article("DOMS pencil review", "https://b.example/1")]})
    assert len(source().fetch(DOMS, SINCE)) == 1

    responses.add(responses.GET, GDELT_URL, body="Your search contained a phrase that was too short.")
    news = source()
    assert news.fetch(DOMS, SINCE) == [] and "too short" in news.warnings[0]


@responses.activate
def test_empty_answer_is_no_articles() -> None:
    responses.add(responses.GET, GDELT_URL, body="{}")
    news = source()
    assert news.fetch(DOMS, SINCE) == [] and not news.warnings


def test_long_urls_are_hashed_to_fit_the_column() -> None:
    assert external_id("https://short.example/a") == "https://short.example/a"
    long = "https://x.example/" + "a" * 300
    assert external_id(long).startswith("sha1:") and len(external_id(long)) < 255
