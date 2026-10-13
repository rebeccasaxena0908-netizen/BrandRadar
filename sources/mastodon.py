"""Mastodon (public API, no login): the brand's hashtag timeline on a few large instances.

    GET https://<instance>/api/v1/timelines/tag/<tag>?limit=40&max_id=...   newest first

Full-text status search needs a login on Mastodon (and only finds posts whose authors opted in), so
public hashtag timelines are the open route. Respecting people's choices:
  - skipped: accounts that opted out of search or discovery (indexable / discoverable false, noindex),
    bots, non-public posts, boosts
  - English posts only (the models are English-only), at most 3 posts per author per brand and run
  - the post must still name the brand (hashtags count: "#Obsidian"), with the usual exclude rules
The same post seen on several instances (federation) is stored once, keyed by its canonical URI.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from config import settings
from sources.base import (
    BaseSource, BrandSpec, HttpClient, RawMention, cap_per_author, html_to_text, shared_client, to_utc_naive,
)

PAGE_SIZE = 40  # Mastodon's maximum for timelines
MAX_PAGES = 3  # per instance and tag


def brand_tag(brand: BrandSpec) -> str:
    """Hashtags are case-insensitive: "ClickUp" -> "clickup"."""
    return re.sub(r"[^a-z0-9_]", "", brand.name.lower())


def _dt(value: str) -> datetime:
    return to_utc_naive(datetime.fromisoformat(value.replace("Z", "+00:00")))


def opted_out(status: dict[str, Any]) -> bool:
    account = status.get("account") or {}
    return (
        account.get("indexable") is False
        or account.get("discoverable") is False
        or account.get("noindex") is True
        or account.get("bot") is True
    )


class MastodonSource(BaseSource):
    name = "mastodon"

    def __init__(self, client: HttpClient | None = None, instances: tuple[str, ...] | None = None,
                 max_items: int | None = None) -> None:
        super().__init__()
        self.client = client or shared_client()  # per-host rate limiting covers every instance
        self.instances = instances or settings.mastodon_instances
        self.max_items = max_items or settings.max_items_per_source

    def fetch(self, brand: BrandSpec, since: datetime) -> list[RawMention]:
        self.reset()
        found: dict[str, RawMention] = {}
        failed = 0
        for instance in self.instances:
            try:
                self._timeline(instance, brand_tag(brand), brand, since, found)
            except Exception as exc:  # noqa: BLE001 — one instance down must not stop the others
                failed += 1
                self.warn(f"{instance}: {type(exc).__name__}: {exc}")
            if len(found) >= self.max_items:
                break
        if failed and failed == len(self.instances):
            raise RuntimeError(f"all instances failed: {'; '.join(self.warnings)}")
        return cap_per_author(list(found.values()))[: self.max_items]

    def _timeline(self, instance: str, tag: str, brand: BrandSpec, since: datetime, found: dict[str, RawMention]) -> None:
        max_id: str | None = None
        for _ in range(MAX_PAGES):
            params: dict[str, Any] = {"limit": PAGE_SIZE}
            if max_id:
                params["max_id"] = max_id
            statuses = self.client.get_json(f"https://{instance}/api/v1/timelines/tag/{tag}", params=params)
            if not statuses:
                return
            for status in statuses:
                created = _dt(status["created_at"])
                if created < since or status.get("reblog") or status.get("visibility") != "public" or opted_out(status):
                    continue
                if status.get("language") not in ("en", None) or status["uri"] in found:
                    continue
                text = html_to_text(status.get("content"))
                if not text or not self.accept(text, brand):
                    continue
                found[status["uri"]] = RawMention(
                    source=self.name, external_id=status["uri"][:255], text=text, created_at=created,
                    author=(status.get("account") or {}).get("acct"), url=status.get("url") or status["uri"],
                )
            max_id = statuses[-1]["id"]
            if _dt(statuses[-1]["created_at"]) < since:  # newest first: the rest are older
                return
