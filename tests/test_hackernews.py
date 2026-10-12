from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse

import responses

from sources.base import BrandSpec, HttpClient
from sources.hackernews import ALGOLIA_URL, FIREBASE_ITEM_URL, HackerNewsSource

NOTION = BrandSpec(id=1, name="Notion", keywords=("Notion", "notion.so"), exclude=("Notion Ink",))
SINCE = datetime(2026, 10, 1)
TS = int(datetime(2026, 10, 5, tzinfo=timezone.utc).timestamp())


def hit(object_id: str, tags: list[str], **fields: object) -> dict[str, object]:
    return {"objectID": object_id, "_tags": tags, "created_at_i": TS, "author": "pg", **fields}


def algolia_callback(pages: dict[str, list[list[dict[str, object]]]]):
    def callback(request):  # type: ignore[no-untyped-def]
        params = parse_qs(urlparse(request.url).query)
        query, page = params["query"][0], int(params["page"][0])
        result_pages = pages.get(query, [[]])
        body = {"hits": result_pages[page] if page < len(result_pages) else [], "nbPages": len(result_pages)}
        import json

        return 200, {"Content-Type": "application/json"}, json.dumps(body)

    return callback


@responses.activate
def test_fetch_stories_and_comments(client: HttpClient) -> None:
    pages = {
        "notion": [
            [
                hit("1", ["comment", "story_9"], comment_text="I love <b>Notion</b> but sync is &quot;slow&quot;"),
                hit("2", ["story"], title="Show HN: a Notion clone", story_text="<p>Built in Rust</p>", url="x"),
                hit("3", ["comment"], comment_text="the notion that this matters is silly"),  # not the brand
            ],
            [hit("4", ["comment"], comment_text="Notion Ink tablets were odd")],  # excluded phrase
        ],
        "notion.so": [[hit("1", ["comment"], comment_text="dup of 1 via notion.so")]],
    }
    responses.add_callback(responses.GET, ALGOLIA_URL, callback=algolia_callback(pages))

    mentions = HackerNewsSource(client=client).fetch(NOTION, SINCE)

    by_id = {m.external_id: m for m in mentions}
    assert set(by_id) == {"1", "2"}
    assert by_id["1"].text == 'I love Notion but sync is "slow"'
    assert by_id["2"].text == "Show HN: a Notion clone\n\nBuilt in Rust"
    assert by_id["1"].url == "https://news.ycombinator.com/item?id=1"
    assert by_id["1"].created_at == datetime(2026, 10, 5)
    assert by_id["1"].source == "hackernews"
    assert by_id["1"].author == "pg"


@responses.activate
def test_search_params_and_user_agent(client: HttpClient) -> None:
    responses.add_callback(responses.GET, ALGOLIA_URL, callback=algolia_callback({}))
    HackerNewsSource(client=client).fetch(NOTION, SINCE)
    # one request per distinct lowercased keyword
    assert len(responses.calls) == 2
    params = parse_qs(urlparse(responses.calls[0].request.url).query)
    assert params["tags"] == ["(story,comment)"]
    assert params["numericFilters"] == [f"created_at_i>{int(SINCE.replace(tzinfo=timezone.utc).timestamp())}"]
    assert "BrandRadar" in responses.calls[0].request.headers["User-Agent"]


@responses.activate
def test_firebase_fallback_when_text_missing(client: HttpClient) -> None:
    pages = {"notion": [[hit("7", ["comment"], comment_text=None)]]}
    responses.add_callback(responses.GET, ALGOLIA_URL, callback=algolia_callback(pages))
    responses.get(FIREBASE_ITEM_URL.format(id=7), json={"id": 7, "text": "Notion <i>offline</i> mode please"})

    mentions = HackerNewsSource(client=client).fetch(NOTION, SINCE)

    assert [m.text for m in mentions] == ["Notion offline mode please"]


@responses.activate
def test_firebase_failure_is_a_warning(client: HttpClient) -> None:
    pages = {"notion": [[hit("8", ["comment"], comment_text="")]]}
    responses.add_callback(responses.GET, ALGOLIA_URL, callback=algolia_callback(pages))
    responses.get(FIREBASE_ITEM_URL.format(id=8), status=500)

    source = HackerNewsSource(client=client)
    assert source.fetch(NOTION, SINCE) == []
    assert len(source.warnings) == 1 and "firebase item 8" in source.warnings[0]


@responses.activate
def test_max_items_cap(client: HttpClient) -> None:
    pages = {"notion": [[hit(str(i), ["comment"], comment_text=f"Notion {i}") for i in range(10)]]}
    responses.add_callback(responses.GET, ALGOLIA_URL, callback=algolia_callback(pages))
    assert len(HackerNewsSource(client=client, max_items=3).fetch(NOTION, SINCE)) == 3


@responses.activate
def test_typo_tolerance_off(client: HttpClient) -> None:
    responses.add_callback(responses.GET, ALGOLIA_URL, callback=algolia_callback({}))
    HackerNewsSource(client=client).fetch(NOTION, SINCE)
    assert parse_qs(urlparse(responses.calls[0].request.url).query)["typoTolerance"] == ["false"]


@responses.activate
def test_continues_in_older_window_when_algolia_caps_results(client: HttpClient) -> None:
    """Algolia stops at ~1000 hits; the source must continue below the oldest hit seen."""
    import json

    newer = [hit(str(i), ["comment"], comment_text=f"Notion {i}", created_at_i=TS + 100 - i) for i in range(3)]
    older = [hit(str(i), ["comment"], comment_text=f"Notion {i}", created_at_i=TS - i) for i in range(3, 5)]
    seen_filters: list[str] = []

    def callback(request):  # type: ignore[no-untyped-def]
        params = parse_qs(urlparse(request.url).query)
        if params["query"][0] != "notion":
            return 200, {}, json.dumps({"hits": [], "nbPages": 0, "nbHits": 0})
        filters = params["numericFilters"][0]
        seen_filters.append(filters)
        if "created_at_i<" not in filters:  # first window: capped, more hits exist
            return 200, {}, json.dumps({"hits": newer, "nbPages": 1, "nbHits": 5})
        return 200, {}, json.dumps({"hits": older, "nbPages": 1, "nbHits": 2})

    responses.add_callback(responses.GET, ALGOLIA_URL, callback=callback)
    mentions = HackerNewsSource(client=client).fetch(NOTION, SINCE)

    assert sorted(m.external_id for m in mentions) == ["0", "1", "2", "3", "4"]
    assert seen_filters[1].endswith(f",created_at_i<{TS + 100 - 2 + 1}")
