"""Google Play: reviews via the unofficial google-play-scraper package, newest first."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from config import settings
from sources.base import (
    BaseSource,
    BrandSpec,
    RateLimiter,
    RawMention,
    socket_timeout,
    to_utc_naive,
    with_retries,
)

REVIEW_URL = "https://play.google.com/store/apps/details?id={app_id}&reviewId={review_id}"
BATCH_SIZE = 200

ReviewsFunc = Callable[..., tuple[list[dict[str, Any]], Any]]


def _default_reviews() -> ReviewsFunc:
    from google_play_scraper import reviews

    return reviews


class PlayStoreSource(BaseSource):
    name = "playstore"

    def __init__(
        self,
        reviews_func: ReviewsFunc | None = None,
        limiter: RateLimiter | None = None,
        max_items: int | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        super().__init__()
        self._reviews = reviews_func
        self.limiter = limiter or RateLimiter(settings.http_min_interval)
        self.max_items = max_items or settings.max_items_per_source
        self._sleep = sleep

    def fetch(self, brand: BrandSpec, since: datetime) -> list[RawMention]:
        self.reset()
        if not brand.play_store_id:
            return []
        from google_play_scraper import Sort

        reviews = self._reviews or _default_reviews()
        app_id = brand.play_store_id
        found: dict[str, RawMention] = {}
        token = None
        while len(found) < self.max_items:
            self.limiter.wait("play.google.com")

            def call(token: Any = token) -> tuple[list[dict[str, Any]], Any]:
                with socket_timeout(settings.http_timeout):
                    return reviews(
                        app_id,
                        lang=settings.playstore_lang,
                        country=settings.playstore_country,
                        sort=Sort.NEWEST,
                        count=BATCH_SIZE,
                        continuation_token=token,
                    )

            retry_kwargs = {"sleep": self._sleep} if self._sleep else {}
            batch, token = with_retries(call, attempts=settings.http_retries, **retry_kwargs)
            reached_older = False
            for review in batch:
                mention = self._to_mention(review, app_id)
                if mention is None:
                    continue
                if mention.created_at < since:
                    reached_older = True
                    continue
                found.setdefault(mention.external_id, mention)
            if not batch or reached_older or token is None:
                break
        return list(found.values())[: self.max_items]

    def _to_mention(self, review: dict[str, Any], app_id: str) -> RawMention | None:
        text = (review.get("content") or "").strip()
        review_id = review.get("reviewId")
        at = review.get("at")
        if not text or not review_id or not isinstance(at, datetime):
            return None
        score = review.get("score")
        return RawMention(
            source=self.name,
            external_id=str(review_id),
            text=text,
            created_at=to_utc_naive(at),  # the scraper returns naive local time
            author=review.get("userName"),
            url=REVIEW_URL.format(app_id=app_id, review_id=review_id),
            rating=int(score) if score is not None else None,
        )
