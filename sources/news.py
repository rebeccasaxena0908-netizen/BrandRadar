"""News: GDELT DOC 2.0 article search (free, no key; https://www.gdeltproject.org/about.html#termsofuse).

GDELT monitors news sites worldwide and returns article headlines and links, not the article text, so a
news mention is its headline. Headlines are coverage, not customer feedback: news stays out of complaint
topics by default (TOPIC_EXCLUDED_SOURCES) but counts for volume and sentiment.

GDELT asks for at most one request every 5 seconds and answers 429 when it is busy. This source waits
NEWS_MIN_INTERVAL between requests, retries a 429 once after a pause, and otherwise records a warning and
moves on (the next collector cycle tries again).

Google News RSS was considered and rejected: its feed terms allow only "rendering Google News results
within a personal feed reader for personal, non-commercial use".
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

import requests

from config import settings
from sources.base import BaseSource, BrandSpec, HttpClient, RateLimiter, RawMention, make_session, utc_now

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
MAX_RECORDS = 250  # GDELT's maximum per request
MAX_LOOKBACK = timedelta(days=90)  # the DOC API searches the last 3 months
RETRY_AFTER_429 = 15.0


def gdelt_query(brand: BrandSpec) -> str:
    """(laneige OR "laneige lip") sourcelang:english — multi-word phrases quoted (GDELT rejects quoted
    short words as "phrase too short"), OR needs parentheses."""
    terms = [f'"{k}"' if " " in k else k for k in dict.fromkeys(brand.keywords)]
    joined = terms[0] if len(terms) == 1 else "(" + " OR ".join(terms) + ")"
    return f"{joined} sourcelang:english"


def parse_seendate(value: str) -> datetime:
    return datetime.strptime(value, "%Y%m%dT%H%M%SZ")


def external_id(url: str) -> str:
    """The article URL, or its hash when it does not fit the 255-character column."""
    return url if len(url) <= 255 else "sha1:" + hashlib.sha1(url.encode()).hexdigest()


class NewsSource(BaseSource):
    name = "news"

    def __init__(
        self,
        client: HttpClient | None = None,
        max_items: int | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__()
        # no automatic retries: GDELT counts every retry against the 5-second rule
        self.client = client or HttpClient(make_session(retries=0), RateLimiter(settings.news_min_interval))
        self.max_items = max_items or settings.max_items_per_source
        self._sleep = sleep

    def fetch(self, brand: BrandSpec, since: datetime) -> list[RawMention]:
        self.reset()
        start = max(since, utc_now() - MAX_LOOKBACK)
        params = {
            "query": gdelt_query(brand), "mode": "artlist", "format": "json", "sort": "datedesc",
            "maxrecords": MAX_RECORDS, "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        }
        data = self._get(params)
        found: dict[str, RawMention] = {}
        for article in (data or {}).get("articles", []):
            title = " ".join((article.get("title") or "").split())
            url = article.get("url") or ""
            if not title or not url or url in found:
                continue
            try:
                created = parse_seendate(article.get("seendate", ""))
            except ValueError:
                continue
            if created < since or not self.accept(title, brand):
                continue
            found[url] = RawMention(
                source=self.name, external_id=external_id(url), text=title, created_at=created,
                author=article.get("domain"), url=url[:1000],
            )
            if len(found) >= self.max_items:
                break
        return list(found.values())

    def _get(self, params: dict[str, Any]) -> dict[str, Any] | None:
        for attempt in (1, 2):
            try:
                response = self.client.get(GDELT_URL, params=params)
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 429 and attempt == 1:
                    self._sleep(RETRY_AFTER_429)
                    continue
                if exc.response is not None and exc.response.status_code == 429:
                    self.warn("GDELT is rate-limiting this address (429); trying again next cycle")
                    return None
                raise
            text = response.text.strip()
            if not text or text == "{}":
                return {}
            try:
                return response.json()
            except ValueError:
                # GDELT answers query problems in plain text ("Your search contained a phrase that was too short")
                self.warn(f"GDELT: {text[:160]}")
                return None
        return None
