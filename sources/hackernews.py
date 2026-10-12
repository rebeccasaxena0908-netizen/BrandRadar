"""Hacker News: Algolia keyword search (stories + comments), Firebase for missing item text."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from config import settings
from sources.base import (
    BaseSource,
    BrandSpec,
    HttpClient,
    RawMention,
    html_to_text,
    matches_brand,
    shared_client,
)

ALGOLIA_URL = "https://hn.algolia.com/api/v1/search_by_date"
FIREBASE_ITEM_URL = "https://hacker-news.firebaseio.com/v0/item/{id}.json"
ITEM_URL = "https://news.ycombinator.com/item?id={id}"
HITS_PER_PAGE = 100


class HackerNewsSource(BaseSource):
    name = "hackernews"

    def __init__(self, client: HttpClient | None = None, max_items: int | None = None) -> None:
        super().__init__()
        self.client = client or shared_client()
        self.max_items = max_items or settings.max_items_per_source

    def fetch(self, brand: BrandSpec, since: datetime) -> list[RawMention]:
        self.reset()
        found: dict[str, RawMention] = {}
        since_ts = int(since.replace(tzinfo=timezone.utc).timestamp())
        # Algolia is case-insensitive, so "Notion" and "notion" are the same query.
        queries = list(dict.fromkeys(k.lower() for k in brand.keywords))
        for query in queries:
            for hit in self._search(query, since_ts):
                if len(found) >= self.max_items:
                    return list(found.values())
                object_id = str(hit.get("objectID", ""))
                if not object_id or object_id in found:
                    continue
                mention = self._to_mention(hit)
                if mention and matches_brand(mention.text, brand):
                    found[object_id] = mention
        return list(found.values())

    def _search(self, query: str, since_ts: int) -> list[dict[str, Any]]:
        """Newest-first hits since `since_ts`.

        Algolia returns at most ~1000 hits per query, so when a window is exhausted while still
        full, the search continues in an older window ending at the oldest hit seen.
        """
        hits: list[dict[str, Any]] = []
        upper_ts: int | None = None
        while len(hits) < self.max_items * 3:  # headroom for hits the brand filter drops
            window: list[dict[str, Any]] = []
            filters = f"created_at_i>{since_ts}" + (f",created_at_i<{upper_ts}" if upper_ts else "")
            page = 0
            while True:
                data = self.client.get_json(
                    ALGOLIA_URL,
                    params={
                        "query": query,
                        "tags": "(story,comment)",
                        "numericFilters": filters,
                        "typoTolerance": "false",  # otherwise "notion" also matches "motion", "nation"...
                        "hitsPerPage": HITS_PER_PAGE,
                        "page": page,
                    },
                )
                window.extend(data.get("hits", []))
                page += 1
                if page >= int(data.get("nbPages", 0)):
                    break
            hits.extend(window)
            total = int(data.get("nbHits", 0))
            if not window or len(window) >= total:
                break  # the whole window fitted: nothing older left
            # +1 so hits sharing the oldest second are seen again (deduped by objectID later)
            next_upper = min(int(h["created_at_i"]) for h in window) + 1
            if upper_ts is not None and next_upper >= upper_ts:
                break  # window did not move back in time; stop rather than loop
            upper_ts = next_upper
        return hits

    def _to_mention(self, hit: dict[str, Any]) -> RawMention | None:
        object_id = str(hit["objectID"])
        tags = hit.get("_tags", [])
        if "comment" in tags:
            text = html_to_text(hit.get("comment_text"))
        else:
            title = (hit.get("title") or "").strip()
            body = html_to_text(hit.get("story_text"))
            text = f"{title}\n\n{body}".strip()
        if not text:
            text = self._firebase_text(object_id)
        if not text:
            return None
        created = datetime.fromtimestamp(int(hit["created_at_i"]), tz=timezone.utc)
        return RawMention(
            source=self.name,
            external_id=object_id,
            text=text,
            created_at=created.replace(tzinfo=None),
            author=hit.get("author"),
            url=ITEM_URL.format(id=object_id),
        )

    def _firebase_text(self, object_id: str) -> str:
        try:
            item = self.client.get_json(FIREBASE_ITEM_URL.format(id=object_id)) or {}
        except Exception as exc:  # noqa: BLE001 — one missing item must not stop the run
            self.warn(f"firebase item {object_id}: {exc}")
            return ""
        if item.get("deleted") or item.get("dead"):
            return ""
        title = (item.get("title") or "").strip()
        body = html_to_text(item.get("text"))
        return f"{title}\n\n{body}".strip()
