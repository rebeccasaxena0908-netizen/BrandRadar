"""google-play-scraper makes its own HTTP calls, so the scraper function itself is mocked."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from sources.base import BrandSpec, RateLimiter
from sources.playstore import PlayStoreSource

BRAND = BrandSpec(id=1, name="Notion", keywords=("Notion",), play_store_id="notion.id")
SINCE = datetime(2026, 10, 1)


def local(dt_utc: datetime) -> datetime:
    """The scraper returns naive *local* time; build that from a UTC time."""
    return dt_utc.replace(tzinfo=timezone.utc).astimezone().replace(tzinfo=None)


def review(review_id: str, at_utc: datetime, score: int = 3, content: str = "Okay app") -> dict[str, Any]:
    return {"reviewId": review_id, "userName": f"u{review_id}", "content": content, "score": score, "at": local(at_utc)}


class FakeReviews:
    def __init__(self, batches: list[list[dict[str, Any]]]) -> None:
        self.batches = batches
        self.calls: list[dict[str, Any]] = []

    def __call__(self, app_id: str, **kwargs: Any) -> tuple[list[dict[str, Any]], Any]:
        self.calls.append({"app_id": app_id, **kwargs})
        index = kwargs["continuation_token"] or 0
        batch = self.batches[index] if index < len(self.batches) else []
        token = index + 1 if index + 1 < len(self.batches) else None
        return batch, token


def make_source(fake: Any, **kwargs: Any) -> PlayStoreSource:
    return PlayStoreSource(reviews_func=fake, limiter=RateLimiter(0), sleep=lambda s: None, **kwargs)


def test_fetch_newest_first_with_paging() -> None:
    fake = FakeReviews([
        [review("a", datetime(2026, 10, 9, 12), score=5, content="Love the AI ✨")],
        [review("b", datetime(2026, 10, 8, 12), score=1), review("c", datetime(2026, 9, 1))],  # c too old
        [review("d", datetime(2026, 10, 7))],  # never requested: stopped at c
    ])
    mentions = make_source(fake).fetch(BRAND, SINCE)

    assert [m.external_id for m in mentions] == ["a", "b"]
    a = mentions[0]
    assert a.text == "Love the AI ✨"
    assert a.rating == 5
    assert a.created_at == datetime(2026, 10, 9, 12)  # local time converted back to UTC
    assert a.url == "https://play.google.com/store/apps/details?id=notion.id&reviewId=a"
    assert len(fake.calls) == 2
    from google_play_scraper import Sort

    assert fake.calls[0]["sort"] == Sort.NEWEST
    assert fake.calls[0]["app_id"] == "notion.id"


def test_skips_reviews_without_text() -> None:
    fake = FakeReviews([[review("a", datetime(2026, 10, 9), content=""), review("b", datetime(2026, 10, 9))]])
    assert [m.external_id for m in make_source(fake).fetch(BRAND, SINCE)] == ["b"]


def test_retries_then_succeeds() -> None:
    calls = {"n": 0}

    def flaky(app_id: str, **kwargs: Any) -> tuple[list[dict[str, Any]], Any]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("temporary")
        return [review("a", datetime(2026, 10, 9))], None

    assert len(make_source(flaky).fetch(BRAND, SINCE)) == 1
    assert calls["n"] == 2


def test_persistent_failure_raises() -> None:
    def broken(app_id: str, **kwargs: Any) -> tuple[list[dict[str, Any]], Any]:
        raise ConnectionError("Google changed the page")

    with pytest.raises(ConnectionError):
        make_source(broken).fetch(BRAND, SINCE)


def test_max_items_cap() -> None:
    fake = FakeReviews([[review(str(i), datetime(2026, 10, 9)) for i in range(10)]] * 3)
    assert len(make_source(fake, max_items=4).fetch(BRAND, SINCE)) == 4


def test_brand_without_package_returns_nothing() -> None:
    brand = BrandSpec(id=2, name="NoApp", keywords=("x",))
    assert make_source(FakeReviews([])).fetch(brand, SINCE) == []
