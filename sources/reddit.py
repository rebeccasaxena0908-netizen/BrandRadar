"""Reddit (optional): PRAW keyword search. Disabled unless REDDIT_CLIENT_ID/SECRET are set.

Reddit is closing API access (no new requests after 31 Oct 2026, remaining access ends March 2027),
so this source must never be required.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from config import settings
from sources.base import BaseSource, BrandSpec, RateLimiter, RawMention, matches_brand

ClientFactory = Callable[[], Any]


def _praw_client() -> Any:
    import praw

    return praw.Reddit(
        client_id=settings.reddit_client_id,
        client_secret=settings.reddit_client_secret,
        user_agent=settings.reddit_user_agent,
        check_for_async=False,
        requestor_kwargs={"timeout": settings.http_timeout},
    )


class RedditSource(BaseSource):
    name = "reddit"

    def __init__(
        self,
        client_factory: ClientFactory | None = None,
        enabled: bool | None = None,
        limiter: RateLimiter | None = None,
        max_items: int | None = None,
    ) -> None:
        super().__init__()
        self.enabled = settings.reddit_enabled if enabled is None else enabled
        self._client_factory = client_factory or _praw_client
        self.limiter = limiter or RateLimiter(settings.http_min_interval)
        self.max_items = max_items or settings.max_items_per_source

    def fetch(self, brand: BrandSpec, since: datetime) -> list[RawMention]:
        self.reset()
        if not self.enabled:
            return []
        reddit = self._client_factory()
        found: dict[str, RawMention] = {}
        for keyword in dict.fromkeys(k.lower() for k in brand.keywords):
            self.limiter.wait("reddit")
            # PRAW handles Reddit's rate-limit headers itself.
            results = reddit.subreddit("all").search(
                f'"{keyword}"', sort="new", time_filter="month", limit=self.max_items
            )
            for post in results:
                created = datetime.fromtimestamp(post.created_utc, tz=timezone.utc).replace(tzinfo=None)
                if created < since or post.id in found:
                    continue
                text = f"{post.title}\n\n{post.selftext or ''}".strip()
                if not matches_brand(text, brand):
                    continue
                found[post.id] = RawMention(
                    source=self.name,
                    external_id=post.id,
                    text=text,
                    created_at=created,
                    author=str(post.author) if post.author else None,
                    url=f"https://www.reddit.com{post.permalink}",
                )
                if len(found) >= self.max_items:
                    return list(found.values())
        return list(found.values())
