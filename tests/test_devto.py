"""dev.to source with mocked HTTP (responses)."""

from __future__ import annotations

import json
from datetime import datetime
from urllib.parse import parse_qs, urlparse

import pytest
import requests
import responses

from sources.base import BrandSpec, HttpClient
from sources.devto import API, DevtoSource, brand_tag

BRAND = BrandSpec(id=1, name="Obsidian", keywords=("Obsidian",), exclude_if=("Fallout",))
SINCE = datetime(2026, 10, 1)


def article(aid: int, day: int, title: str = "Obsidian plugin tips", comments: int = 0, language: str = "en") -> dict:
    return {"id": aid, "title": title, "description": "How I organise my vault", "published_at": f"2026-10-{day:02d}T09:00:00Z",
            "comments_count": comments, "language": language, "url": f"https://dev.to/u/post-{aid}", "user": {"username": "writer"}}


def comment(code: str, day: int, body: str = "<p>Great write-up, thanks!</p>") -> dict:
    return {"id_code": code, "created_at": f"2026-10-{day:02d}T10:00:00Z", "body_html": body,
            "user": {"username": "reader"}, "children": [{"id_code": "reply", "body_html": "<p>reply</p>"}]}


def mock_pages(pages: list[list[dict]], comments: dict[int, list[dict]] | None = None) -> None:
    def articles(request):  # type: ignore[no-untyped-def]
        page = int(parse_qs(urlparse(request.url).query)["page"][0])
        return 200, {}, json.dumps(pages[page - 1] if page <= len(pages) else [])

    def comments_cb(request):  # type: ignore[no-untyped-def]
        aid = int(parse_qs(urlparse(request.url).query)["a_id"][0])
        return 200, {}, json.dumps((comments or {}).get(aid, []))

    responses.add_callback(responses.GET, f"{API}/articles/latest", callback=articles)
    responses.add_callback(responses.GET, f"{API}/comments", callback=comments_cb)


def test_brand_tag() -> None:
    assert brand_tag(BrandSpec(id=1, name="ClickUp", keywords=("clickup",))) == "clickup"


@responses.activate
def test_tagged_articles_and_top_level_comments(client: HttpClient) -> None:
    mock_pages([[article(1, 9, comments=2), article(2, 8, title="My PKM setup"),  # tagged: no name needed
                 article(3, 7, title="Obsidian and Fallout mods"),  # exclude_if
                 article(4, 6, language="es"),  # English only
                 article(5, 5)]],
               comments={1: [comment("c1", 9), comment("c2", 9, "<p>Playing Fallout instead</p>")]})
    mentions = DevtoSource(client=client).fetch(BRAND, SINCE)
    by_id = {m.external_id: m for m in mentions}
    assert set(by_id) == {"a1", "a2", "a5", "cc1"}
    assert by_id["a1"].text == "Obsidian plugin tips\n\nHow I organise my vault" and by_id["a1"].source == "devto"
    assert by_id["a1"].url == "https://dev.to/u/post-1" and by_id["a1"].author == "writer"
    assert by_id["a1"].created_at == datetime(2026, 10, 9, 9, 0)
    assert by_id["cc1"].text == "Great write-up, thanks!" and by_id["cc1"].url == "https://dev.to/reader/comment/c1"
    first = parse_qs(urlparse(responses.calls[0].request.url).query)
    assert first["tag"] == ["obsidian"] and first["per_page"] == ["100"]
    # comments were only requested for the article that has some
    assert sum("/comments" in c.request.url for c in responses.calls) == 1


@responses.activate
def test_stops_paging_once_articles_are_older_than_since(client: HttpClient) -> None:
    mock_pages([[article(1, 9), article(2, 2)], [article(3, 1)], [article(4, 1)]])
    DevtoSource(client=client).fetch(BRAND, datetime(2026, 10, 5))
    assert sum("/articles/latest" in c.request.url for c in responses.calls) == 1  # page 1 already reached `since`


@responses.activate
def test_old_comments_are_skipped(client: HttpClient) -> None:
    mock_pages([[article(1, 9, comments=1)]], comments={1: [comment("old", 1)]})
    mentions = DevtoSource(client=client).fetch(BRAND, datetime(2026, 10, 5))
    assert [m.external_id for m in mentions] == ["a1"]


@responses.activate
def test_comment_failure_is_a_warning(client: HttpClient) -> None:
    responses.get(f"{API}/articles/latest", json=[article(1, 9, comments=3)])
    responses.get(f"{API}/comments", status=500)
    source = DevtoSource(client=client)
    assert [m.external_id for m in source.fetch(BRAND, SINCE)] == ["a1"]
    assert len(source.warnings) == 1 and "comments of article 1" in source.warnings[0]


@responses.activate
def test_article_listing_failure_raises(client: HttpClient) -> None:
    responses.get(f"{API}/articles/latest", status=503)
    with pytest.raises(requests.HTTPError):
        DevtoSource(client=client).fetch(BRAND, SINCE)  # fetch.py records it in fetch_runs.error


@responses.activate
def test_max_items(client: HttpClient) -> None:
    mock_pages([[article(i, 9) for i in range(1, 11)]])
    assert len(DevtoSource(client=client, max_items=3).fetch(BRAND, SINCE)) == 3
