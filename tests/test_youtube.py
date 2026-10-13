"""YouTube source with mocked HTTP (responses) and an in-memory store: no key, no quota spent."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import pytest
import responses
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from db.models import Analysis, ApiUsage, Brand, Mention, YoutubeVideo
from fetch import insert_new_mentions
from sources.base import BrandSpec
from sources.youtube import API, DbYoutubeStore, YouTubeSource, purge_expired, quota_day

KEY = "test-key-not-real"
NOW = datetime(2026, 10, 13, 12, 0)
SINCE = datetime(2026, 10, 1)
LANEIGE = BrandSpec(id=1, name="Laneige", keywords=("Laneige",), exclude_if=("giveaway",),
                    youtube_queries=("Laneige review",))


class MemoryStore:
    def __init__(self) -> None:
        self.usage: dict[tuple[str, date], int] = {}
        self.videos: dict[str, dict[str, Any]] = {}
        self.checked_ids: list[tuple[str, bool]] = []

    def used(self, bucket: str, day: date) -> int:
        return self.usage.get((bucket, day), 0)

    def spend(self, bucket: str, day: date, amount: int) -> None:
        self.usage[(bucket, day)] = self.used(bucket, day) + amount

    def remember(self, brand_id: int, video: dict[str, Any], now: datetime) -> None:
        self.videos[video["id"]] = video

    def videos_to_read(self, brand_id: int, newer_than: datetime, limit: int) -> list[str]:
        vids = [v for v in self.videos.values() if v["published_at"] >= newer_than]
        return [v["id"] for v in sorted(vids, key=lambda v: v["published_at"], reverse=True)][:limit]

    def checked(self, brand_id: int, video_id: str, now: datetime, disabled: bool = False) -> None:
        self.checked_ids.append((video_id, disabled))


def search_item(video_id: str, title: str, day: int = 10) -> dict:
    return {"id": {"kind": "youtube#video", "videoId": video_id},
            "snippet": {"title": title, "channelTitle": "Beauty Channel", "publishedAt": f"2026-10-{day:02d}T09:00:00Z"}}


def thread(cid: str, text: str, author: str = "viewer", day: int = 11) -> dict:
    return {"id": cid, "snippet": {"topLevelComment": {"snippet": {
        "textOriginal": text, "authorDisplayName": author, "publishedAt": f"2026-10-{day:02d}T10:00:00Z"}}}}


def make(store: MemoryStore | None = None, key: str = KEY) -> YouTubeSource:
    return YouTubeSource(api_key=key, store=store or MemoryStore(), clock=lambda: NOW)


def test_skipped_without_a_key() -> None:
    assert make(key="").enabled is False
    assert make().enabled is True


@responses.activate
def test_comments_of_videos_whose_title_names_the_brand() -> None:
    responses.add(responses.GET, f"{API}/search", json={"items": [
        search_item("v1", "Laneige Lip Sleeping Mask &amp; honest review"),
        search_item("v2", "My night routine (no brands named)"),
    ]})
    responses.add(responses.GET, f"{API}/commentThreads", json={"items": [
        thread("c1", "The berry one smells amazing but it is so sticky"),
        thread("c2", "Enter my giveaway!!"),  # exclude_if
        thread("c3", "Bought it twice, the pot is not hygienic", author="a"),
        thread("c4", "Sketch book 📖", author="b"),  # a poll answer: too short and doesn't name the brand
        thread("c5", "Laneige forever", author="c"),  # short but names it
    ]})
    store = MemoryStore()
    yt = make(store)
    found = yt.fetch(LANEIGE, SINCE)
    assert sorted(m.external_id for m in found) == ["c1", "c3", "c5"]  # comments don't need to repeat the name
    c1 = next(m for m in found if m.external_id == "c1")
    assert c1.url == "https://www.youtube.com/watch?v=v1&lc=c1" and c1.source == "youtube"
    assert set(store.videos) == {"v1"} and store.videos["v1"]["title"].startswith("Laneige Lip Sleeping Mask &")
    assert [r for _, r in yt.rejected] == ["name not found"]
    # the key travels in a header, never in the URL (so it can't end up in logs or fetch_runs)
    for call in responses.calls:
        assert KEY not in call.request.url and call.request.headers["X-Goog-Api-Key"] == KEY
    search = responses.calls[0].request.url
    assert "q=Laneige+review" in search and "publishedAfter=2026-10-01T00%3A00%3A00Z" in search
    day = quota_day(NOW)
    assert store.usage == {("search", day): 1, ("units", day): 1}


@responses.activate
def test_budget_stops_the_source_with_a_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    from config import settings

    monkeypatch.setattr(type(settings), "youtube_search_budget", property(lambda s: 3), raising=False)
    store = MemoryStore()
    store.usage[("search", quota_day(NOW))] = 3
    yt = make(store)
    assert yt.fetch(LANEIGE, SINCE) == []
    assert len(responses.calls) == 0
    assert "budget of 3 search calls" in yt.warnings[0]


@responses.activate
def test_google_quota_error_and_disabled_comments() -> None:
    responses.add(responses.GET, f"{API}/search", json={"items": [search_item("v1", "Laneige toner review")]})
    responses.add(responses.GET, f"{API}/commentThreads", status=403,
                  json={"error": {"code": 403, "errors": [{"reason": "commentsDisabled"}]}})
    store = MemoryStore()
    assert make(store).fetch(LANEIGE, SINCE) == []
    assert store.checked_ids == [("v1", True)]

    responses.replace(responses.GET, f"{API}/search", status=403,
                      json={"error": {"code": 403, "errors": [{"reason": "quotaExceeded"}]}})
    yt = make(MemoryStore())
    assert yt.fetch(LANEIGE, SINCE) == [] and "quota exceeded" in yt.warnings[0]


@responses.activate
def test_other_errors_never_show_the_key() -> None:
    responses.add(responses.GET, f"{API}/search", status=400,
                  json={"error": {"code": 400, "errors": [{"reason": "keyInvalid"}]}})
    with pytest.raises(RuntimeError) as err:
        make().fetch(LANEIGE, SINCE)
    assert "keyInvalid" in str(err.value) and KEY not in str(err.value)


def test_quota_day_is_pacific() -> None:
    assert quota_day(datetime(2026, 10, 13, 6, 0)) == date(2026, 10, 12)  # 06:00 UTC = 23:00 PDT the day before
    assert quota_day(datetime(2026, 10, 13, 8, 0)) == date(2026, 10, 13)


# ---------------------------------------------------------------- database: store, refresh, 30-day purge


def test_db_store_and_30_day_retention(factory: sessionmaker[Session]) -> None:
    with factory() as s:
        s.add(Brand(name="Laneige", is_own=True, keywords=["Laneige"], brand_group="Skincare", profile="beauty_skincare"))
        s.commit()
        brand_id = s.scalars(select(Brand.id)).one()
    store = DbYoutubeStore(factory)
    day = quota_day(NOW)
    store.spend("units", day, 1)
    store.spend("units", day, 1)
    assert store.used("units", day) == 2 and store.used("search", day) == 0
    store.remember(brand_id, {"id": "v1", "title": "Laneige review", "channel": "c",
                              "published_at": NOW - timedelta(days=3)}, NOW)
    store.remember(brand_id, {"id": "old", "title": "Laneige haul", "channel": "c",
                              "published_at": NOW - timedelta(days=60)}, NOW - timedelta(days=60))
    assert store.videos_to_read(brand_id, NOW - timedelta(days=30), 10) == ["v1"]
    store.checked(brand_id, "v1", NOW)

    with factory() as s:
        fresh = Mention(brand_id=brand_id, source="youtube", external_id="c1", text="nice", created_at=NOW,
                        fetched_at=NOW - timedelta(days=29))
        stale = Mention(brand_id=brand_id, source="youtube", external_id="c2", text="meh", created_at=NOW,
                        fetched_at=NOW - timedelta(days=31))
        other = Mention(brand_id=brand_id, source="news", external_id="n1", text="headline", created_at=NOW,
                        fetched_at=NOW - timedelta(days=90))
        s.add_all([fresh, stale, other])
        s.flush()
        s.add(Analysis(mention_id=stale.id, sentiment="neutral", model_version="x"))
        s.commit()
        assert purge_expired(s, NOW) == (1, 1)  # the stale comment and the never-refreshed old video
        s.commit()
        assert sorted(s.scalars(select(Mention.external_id))) == ["c1", "n1"]  # other sources are kept
        assert s.scalars(select(Analysis)).all() == []
        assert list(s.scalars(select(YoutubeVideo.video_id))) == ["v1"]
        assert s.scalars(select(ApiUsage.used)).all() == [2]

    # seen again: fetched_at is refreshed, so it is not purged
    from sources.base import RawMention

    with factory() as s:
        insert_new_mentions(s, brand_id, [RawMention(source="youtube", external_id="c1", text="nice", created_at=NOW)])
        s.commit()
        refreshed = s.scalars(select(Mention.fetched_at).where(Mention.external_id == "c1")).one()
    assert refreshed > NOW - timedelta(days=1)
