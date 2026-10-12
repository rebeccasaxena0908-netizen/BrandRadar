"""App Store: iTunes customer reviews RSS JSON feed, looped over several countries."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from config import settings
from sources.base import BaseSource, BrandSpec, HttpClient, RawMention, shared_client, to_utc_naive

FEED_URL = "https://itunes.apple.com/{country}/rss/customerreviews/page={page}/id={app_id}/sortBy=mostRecent/json"
REVIEW_URL = "https://apps.apple.com/{country}/app/id{app_id}?see-all=reviews"
MAX_PAGES = 10  # Apple serves at most 10 pages of 50 reviews per country


def _label(entry: dict[str, Any], key: str) -> str:
    value = entry.get(key) or {}
    return str(value.get("label", "")) if isinstance(value, dict) else ""


class AppStoreSource(BaseSource):
    name = "appstore"

    def __init__(
        self,
        client: HttpClient | None = None,
        countries: tuple[str, ...] | None = None,
        max_items: int | None = None,
    ) -> None:
        super().__init__()
        self.client = client or shared_client()
        self.countries = countries or settings.appstore_countries
        self.max_items = max_items or settings.max_items_per_source

    def fetch(self, brand: BrandSpec, since: datetime) -> list[RawMention]:
        self.reset()
        if not brand.app_store_id:
            return []
        found: dict[str, RawMention] = {}
        failed: list[str] = []
        for country in self.countries:
            try:
                self._fetch_country(brand.app_store_id, country, since, found)
            except Exception as exc:  # noqa: BLE001 — one country failing must not stop the rest
                failed.append(country)
                self.warn(f"{country}: {type(exc).__name__}: {exc}")
            if len(found) >= self.max_items:
                break
        if failed and len(failed) == len(self.countries):
            raise RuntimeError(f"all countries failed: {'; '.join(self.warnings)}")
        return list(found.values())[: self.max_items]

    def _fetch_country(
        self, app_id: str, country: str, since: datetime, found: dict[str, RawMention]
    ) -> None:
        for page in range(1, MAX_PAGES + 1):
            data = self.client.get_json(FEED_URL.format(country=country, page=page, app_id=app_id))
            entries = (data.get("feed") or {}).get("entry") or []
            if isinstance(entries, dict):  # a single review comes back as an object
                entries = [entries]
            reached_older = False
            for entry in entries:
                if "im:rating" not in entry:  # older feeds put app metadata first
                    continue
                mention = self._to_mention(entry, app_id, country)
                if mention.created_at < since:
                    reached_older = True
                    continue
                found.setdefault(mention.external_id, mention)
            if not entries or reached_older or len(found) >= self.max_items:
                return

    def _to_mention(self, entry: dict[str, Any], app_id: str, country: str) -> RawMention:
        title = _label(entry, "title").strip()
        content = _label(entry, "content").strip()
        rating = _label(entry, "im:rating")
        author = (entry.get("author") or {}).get("name", {}).get("label")
        return RawMention(
            source=self.name,
            external_id=_label(entry, "id"),
            text=f"{title}\n\n{content}".strip(),
            created_at=to_utc_naive(datetime.fromisoformat(_label(entry, "updated"))),
            author=author,
            url=REVIEW_URL.format(country=country, app_id=app_id),
            rating=int(rating) if rating.isdigit() else None,
        )
