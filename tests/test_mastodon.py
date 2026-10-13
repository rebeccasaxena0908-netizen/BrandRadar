"""Mastodon source with mocked HTTP (responses)."""

from __future__ import annotations

import json
import re
from datetime import datetime
from urllib.parse import parse_qs, urlparse

import pytest
import responses

from sources.base import BrandSpec, HttpClient
from sources.mastodon import MastodonSource, brand_tag, opted_out

BRAND = BrandSpec(id=1, name="Notion", keywords=("Notion",), exclude_if=("Ionian",))
SINCE = datetime(2026, 10, 1)
TIMELINE = re.compile(r"https://([\w.]+)/api/v1/timelines/tag/(\w+)")


def status(sid: str, text: str, day: int = 6, lang: str | None = "en", uri: str | None = None, **account: object) -> dict:
    acct = {"acct": "someone@example.social", "indexable": True, "discoverable": True, "noindex": None, "bot": False}
    acct.update(account)
    return {"id": sid, "uri": uri or f"https://example.social/statuses/{sid}", "url": f"https://example.social/@someone/{sid}",
            "created_at": f"2026-10-{day:02d}T08:00:00.000Z", "content": f"<p>{text}</p>", "language": lang,
            "visibility": "public", "reblog": None, "account": acct}


def mock(timelines: dict[str, list[list[dict]]], status_code: dict[str, int] | None = None) -> list[tuple[str, dict]]:
    seen: list[tuple[str, dict]] = []

    def callback(request):  # type: ignore[no-untyped-def]
        instance, _tag = TIMELINE.match(request.url).groups()  # type: ignore[union-attr]
        params = {k: v[0] for k, v in parse_qs(urlparse(request.url).query).items()}
        seen.append((instance, params))
        if (status_code or {}).get(instance):
            return status_code[instance], {}, "unavailable"  # type: ignore[index]
        pages = timelines.get(instance, [[]])
        index = 0 if "max_id" not in params else next(
            (i + 1 for i, p in enumerate(pages) if p and p[-1]["id"] == params["max_id"]), len(pages))
        return 200, {}, json.dumps(pages[index] if index < len(pages) else [])

    responses.add_callback(responses.GET, TIMELINE, callback=callback)
    return seen


def source(client: HttpClient, instances: tuple[str, ...] = ("a.social", "b.social"), **kw: object) -> MastodonSource:
    return MastodonSource(client=client, instances=instances, **kw)  # type: ignore[arg-type]


def test_brand_tag_and_opt_out_rules() -> None:
    assert brand_tag(BrandSpec(id=1, name="ClickUp", keywords=("clickup",))) == "clickup"
    assert not opted_out(status("1", "x"))
    for flags in ({"indexable": False}, {"discoverable": False}, {"noindex": True}, {"bot": True}):
        assert opted_out(status("1", "x", **flags)), flags


@responses.activate
def test_filters_language_opt_outs_brand_and_context(client: HttpClient) -> None:
    mock({"a.social": [[
        status("1", "Moved my notes to #Notion this week"),
        status("2", "#Notion para profesores", lang="es"),  # English only
        status("3", "Notion is great", indexable=False),  # opted out of search
        status("4", "Notion AI bot post", bot=True),
        status("5", "the notion that #productivity matters"),  # lowercase word, not the app
        status("6", "Rocky Ionian coast of ancient #Notion"),  # the Greek city
        status("7", "Notion, unknown language", lang=None),  # no language set: kept
    ]]})
    mentions = source(client, instances=("a.social",)).fetch(BRAND, SINCE)
    assert sorted(m.external_id for m in mentions) == ["https://example.social/statuses/1", "https://example.social/statuses/7"]
    m = next(x for x in mentions if x.external_id.endswith("/1"))
    assert m.text == "Moved my notes to #Notion this week" and m.source == "mastodon"
    assert m.url == "https://example.social/@someone/1" and m.author == "someone@example.social"


@responses.activate
def test_skips_boosts_and_non_public(client: HttpClient) -> None:
    boost = status("1", "Notion boost") | {"reblog": {"id": "x"}}
    private = status("2", "Notion private") | {"visibility": "unlisted"}
    mock({"a.social": [[boost, private]]})
    assert source(client, instances=("a.social",)).fetch(BRAND, SINCE) == []


@responses.activate
def test_federated_copies_are_stored_once(client: HttpClient) -> None:
    same = "https://origin.social/statuses/42"
    mock({"a.social": [[status("1", "Notion sync is slow", uri=same)]],
          "b.social": [[status("900", "Notion sync is slow", uri=same)]]})
    mentions = source(client).fetch(BRAND, SINCE)
    assert [m.external_id for m in mentions] == [same]


@responses.activate
def test_pages_with_max_id_until_since(client: HttpClient) -> None:
    seen = mock({"a.social": [
        [status("30", "Notion one", day=9), status("29", "Notion two", day=8)],
        [status("20", "Notion three", day=5), status("19", "Notion old", day=1)],
        [status("10", "Notion never requested", day=1)],
    ]})
    mentions = source(client, instances=("a.social",)).fetch(BRAND, datetime(2026, 10, 3))
    assert len(mentions) == 3
    assert [p.get("max_id") for _, p in seen] == [None, "29"]  # stopped once a page reached `since`
    assert all(p["limit"] == "40" for _, p in seen)


@responses.activate
def test_one_instance_down_is_a_warning(client: HttpClient) -> None:
    mock({"b.social": [[status("1", "Notion works here")]]}, status_code={"a.social": 503})
    src = source(client)
    assert len(src.fetch(BRAND, SINCE)) == 1
    assert len(src.warnings) == 1 and src.warnings[0].startswith("a.social:")


@responses.activate
def test_all_instances_down_raises(client: HttpClient) -> None:
    mock({}, status_code={"a.social": 503, "b.social": 502})
    with pytest.raises(RuntimeError, match="all instances failed"):
        source(client).fetch(BRAND, SINCE)
