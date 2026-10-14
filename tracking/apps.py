"""Find a brand's own app on the App Store and Google Play by name (live lookups, both free and keyless).

An id is only kept when the developer (seller) is the brand and the app's title names it: "McDonald's" by
McDonald's USA, LLC yes; a coupon app for McDonald's by someone else no. The review step shows the title
and developer, so a wrong match can be cleared before anything is fetched. No match means no store
sources, which is right for most physical products.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from nlp.llm.research import key
from sources.base import HttpClient, shared_client, socket_timeout

log = logging.getLogger(__name__)

ITUNES_SEARCH = "https://itunes.apple.com/search"
COUNTRIES = ("us", "in")


@dataclass
class AppMatch:
    store: str  # appstore / playstore
    app_id: str
    title: str
    developer: str


# words a company adds to its name: "McDonald's USA, LLC" is McDonald's, "Dove Family Channel LLC" is not
CORPORATE = {"inc", "llc", "ltd", "limited", "corp", "corporation", "co", "company", "plc", "gmbh", "ag", "sa",
             "nv", "bv", "pvt", "private", "pte", "group", "holdings", "international", "global", "usa", "us",
             "uk", "india", "europe", "apps", "app", "mobile", "digital", "brands", "the", "and", "of"}


def is_brand_app(brand: str, title: str, developer: str) -> bool:
    """The developer is the brand plus at most company words ("McDonald's USA, LLC", "Tesla, Inc.", "Zomato
    Ltd") and the app is named after the brand. "Dove Medical Press" and "Dove Family Channel" are not Dove."""
    b = key(brand)
    if not b or b not in key(title):
        return False
    words = [w for w in (key(part) for part in developer.replace("'", "").replace("’", "").split()) if w]
    joined = ""
    for i, word in enumerate(words):
        joined += word
        if joined == b:
            return all(w in CORPORATE for w in words[i + 1:])
        if not b.startswith(joined):
            return False
    return False


def find_app_store(brand: str, client: HttpClient | None = None) -> AppMatch | None:
    client = client or shared_client()
    for country in COUNTRIES:
        try:
            data = client.get_json(ITUNES_SEARCH, {"term": brand, "entity": "software", "country": country, "limit": 10})
        except Exception as exc:  # noqa: BLE001 — no app id is a fine outcome
            log.info("app store lookup for %s failed: %s", brand, exc)
            return None
        for app in data.get("results") or []:
            title, developer = str(app.get("trackName") or ""), str(app.get("sellerName") or app.get("artistName") or "")
            if app.get("trackId") and is_brand_app(brand, title, developer):
                return AppMatch("appstore", str(app["trackId"]), title, developer)
    return None


SearchFunc = Callable[..., list[dict[str, Any]]]


def find_play_store(brand: str, search: SearchFunc | None = None) -> AppMatch | None:
    if search is None:
        from google_play_scraper import search as gp_search

        search = gp_search
    for country in COUNTRIES:
        try:
            with socket_timeout(20):
                hits = search(brand, lang="en", country=country, n_hits=8)
        except Exception as exc:  # noqa: BLE001 — the scraper is unofficial and can break
            log.info("google play lookup for %s failed: %s", brand, exc)
            return None
        for app in hits or []:
            title, developer = str(app.get("title") or ""), str(app.get("developer") or "")
            if app.get("appId") and is_brand_app(brand, title, developer):
                return AppMatch("playstore", str(app["appId"]), title, developer)
    return None
