"""dev.to (Forem public API, no key): articles tagged with the brand, plus their top-level comments.

    GET https://dev.to/api/articles/latest?tag=<brand>   newest first
    GET https://dev.to/api/comments?a_id=<article id>    threaded comments

dev.to has no full-text search API, so the brand's tag (its name in lower case, e.g. #obsidian) is
the way in. An article tagged #obsidian is about Obsidian, so it counts without
naming the brand, and so do its top-level comments; exclude_if context words still drop either.
English articles only (the models are English-only), at most 3 items per author per brand and run.
Content belongs to its authors: we store the link back to every article and comment.
Most dev.to posts are makers writing about their own tools, not user feedback, so dev.to is left out of
complaint topics, the priority matrix and alerts (TOPIC_EXCLUDED_SOURCES).
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from config import settings
from sources.base import (
    BaseSource, BrandSpec, HttpClient, RateLimiter, RawMention, cap_per_author, excluded_by_context, html_to_text,
    make_session, to_utc_naive,
)

API = "https://dev.to/api"
PER_PAGE = 100
MAX_PAGES = 3  # per tag and run
MIN_INTERVAL = 1.0  # dev.to throttles bursts; one request a second is plenty


def default_client() -> HttpClient:
    return HttpClient(session=make_session(), limiter=RateLimiter(max(MIN_INTERVAL, settings.http_min_interval)))


def brand_tag(brand: BrandSpec) -> str:
    """dev.to tags are lower-case letters and digits only: "ClickUp" -> "clickup"."""
    return re.sub(r"[^a-z0-9]", "", brand.name.lower())


def _dt(value: str) -> datetime:
    return to_utc_naive(datetime.fromisoformat(value.replace("Z", "+00:00")))


class DevtoSource(BaseSource):
    name = "devto"

    def __init__(self, client: HttpClient | None = None, max_items: int | None = None) -> None:
        super().__init__()
        self.client = client or default_client()
        self.max_items = max_items or settings.max_items_per_source

    def fetch(self, brand: BrandSpec, since: datetime) -> list[RawMention]:
        self.reset()
        found: dict[str, RawMention] = {}
        articles: dict[int, dict[str, Any]] = {}
        for page in range(1, MAX_PAGES + 1):
            batch = self.client.get_json(f"{API}/articles/latest",
                                         params={"tag": brand_tag(brand), "per_page": PER_PAGE, "page": page})
            if not batch:
                break
            for article in batch:
                if _dt(article["published_at"]) >= since and article.get("language", "en") == "en":
                    articles.setdefault(article["id"], article)
            if _dt(batch[-1]["published_at"]) < since:  # newest first: the rest are older
                break

        for article in articles.values():
            text = f"{article['title']}\n\n{article.get('description') or ''}".strip()
            if excluded_by_context(text, brand):
                continue
            found[f"a{article['id']}"] = RawMention(
                source=self.name, external_id=f"a{article['id']}", text=text, created_at=_dt(article["published_at"]),
                author=(article.get("user") or {}).get("username"), url=article.get("url"),
            )
            if article.get("comments_count"):
                self._comments(article, brand, since, found)
            if len(found) >= self.max_items:
                break
        return cap_per_author(list(found.values()))[: self.max_items]

    def _comments(self, article: dict[str, Any], brand: BrandSpec, since: datetime, found: dict[str, RawMention]) -> None:
        try:
            comments = self.client.get_json(f"{API}/comments", params={"a_id": article["id"]})
        except Exception as exc:  # noqa: BLE001 — one article's comments must not stop the brand
            self.warn(f"comments of article {article['id']}: {type(exc).__name__}: {exc}")
            return
        for comment in comments:  # top level only; replies sit in comment["children"]
            body = html_to_text(comment.get("body_html"))
            if not body or _dt(comment["created_at"]) < since or excluded_by_context(body, brand):
                continue
            username = (comment.get("user") or {}).get("username")
            found[f"c{comment['id_code']}"] = RawMention(
                source=self.name, external_id=f"c{comment['id_code']}", text=body, created_at=_dt(comment["created_at"]),
                author=username, url=f"https://dev.to/{username}/comment/{comment['id_code']}" if username else article.get("url"),
            )
