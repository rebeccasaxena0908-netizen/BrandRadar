from __future__ import annotations

import re
from datetime import datetime

import pytest
import responses

from sources.appstore import AppStoreSource
from sources.base import BrandSpec, HttpClient

BRAND = BrandSpec(id=1, name="Notion", keywords=("Notion",), app_store_id="123")
SINCE = datetime(2026, 10, 1)
FEED_RE = re.compile(r"https://itunes\.apple\.com/(\w+)/rss/customerreviews/page=(\d+)/id=123/sortBy=mostRecent/json")


def entry(review_id: str, updated: str, rating: int = 4, title: str = "Nice", content: str = "Works well") -> dict:
    return {
        "author": {"name": {"label": f"user{review_id}"}, "uri": {"label": "x"}},
        "updated": {"label": updated},
        "im:rating": {"label": str(rating)},
        "id": {"label": review_id},
        "title": {"label": title},
        "content": {"label": content, "attributes": {"type": "text"}},
    }


def feed(*entries: dict) -> dict:
    return {"feed": {"entry": list(entries)}} if entries else {"feed": {}}


@responses.activate
def test_fetch_loops_over_countries_and_pages(client: HttpClient) -> None:
    data = {
        ("us", 1): feed(entry("1", "2026-10-09T16:11:38-07:00", rating=5, title="Colors", content="Add a color wheel 🎨")),
        ("us", 2): feed(),
        ("gb", 1): feed(entry("2", "2026-10-05T10:00:00+01:00", rating=1)),
        ("gb", 2): feed(entry("3", "2026-10-04T10:00:00+01:00")),
        ("gb", 3): feed(),
    }

    def callback(request):  # type: ignore[no-untyped-def]
        country, page = FEED_RE.match(request.url).groups()  # type: ignore[union-attr]
        import json

        return 200, {}, json.dumps(data.get((country, int(page)), feed()))

    responses.add_callback(responses.GET, FEED_RE, callback=callback)

    mentions = AppStoreSource(client=client, countries=("us", "gb")).fetch(BRAND, SINCE)

    by_id = {m.external_id: m for m in mentions}
    assert set(by_id) == {"1", "2", "3"}
    assert by_id["1"].text == "Colors\n\nAdd a color wheel 🎨"
    assert by_id["1"].rating == 5
    assert by_id["1"].created_at == datetime(2026, 10, 9, 23, 11, 38)  # converted to UTC
    assert by_id["1"].url == "https://apps.apple.com/us/app/id123?see-all=reviews"
    assert by_id["2"].rating == 1
    assert by_id["2"].author == "user2"


@responses.activate
def test_stops_at_reviews_older_than_since(client: HttpClient) -> None:
    responses.get(
        FEED_RE,
        json=feed(entry("1", "2026-10-09T00:00:00Z"), entry("2", "2026-09-01T00:00:00Z")),
    )
    mentions = AppStoreSource(client=client, countries=("us",)).fetch(BRAND, SINCE)
    assert [m.external_id for m in mentions] == ["1"]
    assert len(responses.calls) == 1  # did not request page 2


@responses.activate
def test_one_failing_country_is_a_warning(client: HttpClient) -> None:
    def callback(request):  # type: ignore[no-untyped-def]
        country, page = FEED_RE.match(request.url).groups()  # type: ignore[union-attr]
        import json

        if country == "in":
            return 503, {}, "unavailable"
        body = feed(entry("9", "2026-10-09T00:00:00Z")) if page == "1" else feed()
        return 200, {}, json.dumps(body)

    responses.add_callback(responses.GET, FEED_RE, callback=callback)
    source = AppStoreSource(client=client, countries=("us", "in"))
    mentions = source.fetch(BRAND, SINCE)
    assert [m.external_id for m in mentions] == ["9"]
    assert len(source.warnings) == 1 and source.warnings[0].startswith("in:")


@responses.activate
def test_all_countries_failing_raises(client: HttpClient) -> None:
    responses.get(FEED_RE, status=500)
    with pytest.raises(RuntimeError, match="all countries failed"):
        AppStoreSource(client=client, countries=("us", "gb")).fetch(BRAND, SINCE)


def test_brand_without_app_id_returns_nothing(client: HttpClient) -> None:
    brand = BrandSpec(id=2, name="NoApp", keywords=("x",))
    assert AppStoreSource(client=client).fetch(brand, SINCE) == []
