"""YouTube: comments on videos about the brand (YouTube Data API v3, free key, optional).

Per brand and cycle:
1. search.list for the brand's `youtube_queries` (else its keywords), videos published since the last
   fetch. Only videos whose *title* names the brand (same matching rules as every source) are kept and
   remembered in `youtube_videos`.
2. commentThreads.list on the newest remembered videos (YOUTUBE_MAX_VIDEOS, up to YOUTUBE_COMMENT_PAGES
   pages of 100). Comments don't have to repeat the brand name (the video is about it), except very short
   ones (poll and giveaway answers like "Sketch book"); exclude_if still applies. Video titles themselves are the creator's words and are not stored as mentions.

Quota (checked 2026-10): search.list has its own cap of 100 calls a day; commentThreads.list costs 1 unit
of the 10,000 a day. Both reset at midnight Pacific time. Spending is counted in `api_usage` and the source
stops at YOUTUBE_SEARCH_BUDGET / YOUTUBE_UNIT_BUDGET, leaving headroom below Google's limits.

YouTube's API policies allow storing comment text for 30 days without refreshing it: comments seen again
get a new `fetched_at`, and `purge_expired` (run by the collector) deletes YouTube mentions, their analysis
and remembered videos not refreshed for 30 days.

The key is sent in the X-Goog-Api-Key header, never in the URL, so it can't leak into logs or fetch_runs.
"""

from __future__ import annotations

import html
from collections.abc import Callable
from datetime import date, datetime, timedelta
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import requests
from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session, sessionmaker

from config import settings
from db.models import Analysis, ApiUsage, Mention, YoutubeVideo
from sources.base import (
    BaseSource, BrandSpec, HttpClient, RateLimiter, RawMention, cap_per_author, excluded_by_context, make_session,
    too_short_unnamed,
    utc_now,
)

API = "https://www.googleapis.com/youtube/v3"
QUOTA_TZ = ZoneInfo("America/Los_Angeles")
RETENTION = timedelta(days=30)
SEARCH_RESULTS = 50  # maximum per search page
COMMENTS_PER_PAGE = 100
WATCH_URL = "https://www.youtube.com/watch?v={video}&lc={comment}"


class QuotaExhausted(Exception):
    """Our budget (or Google's quota) for today is used up."""


def quota_day(now: datetime | None = None) -> date:
    """YouTube's quota day: the date in Pacific time."""
    aware = (now or utc_now()).replace(tzinfo=ZoneInfo("UTC"))
    return aware.astimezone(QUOTA_TZ).date()


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)


# ---------------------------------------------------------------- storage (injectable for tests)


class YoutubeStore(Protocol):
    def used(self, bucket: str, day: date) -> int: ...
    def spend(self, bucket: str, day: date, amount: int) -> None: ...
    def remember(self, brand_id: int, video: dict[str, Any], now: datetime) -> None: ...
    def videos_to_read(self, brand_id: int, newer_than: datetime, limit: int) -> list[str]: ...
    def checked(self, brand_id: int, video_id: str, now: datetime, disabled: bool = False) -> None: ...


class DbYoutubeStore:
    """api_usage + youtube_videos in MySQL."""

    def __init__(self, factory: sessionmaker[Session] | None = None) -> None:
        if factory is None:
            from db.session import get_session_factory

            factory = get_session_factory()
        self.factory = factory

    def used(self, bucket: str, day: date) -> int:
        with self.factory() as s:
            return int(s.scalar(select(ApiUsage.used).where(
                ApiUsage.api == "youtube", ApiUsage.bucket == bucket, ApiUsage.day == day)) or 0)

    def spend(self, bucket: str, day: date, amount: int) -> None:
        with self.factory() as s:
            row = s.scalars(select(ApiUsage).where(
                ApiUsage.api == "youtube", ApiUsage.bucket == bucket, ApiUsage.day == day)).first()
            if row is None:
                s.add(ApiUsage(api="youtube", bucket=bucket, day=day, used=amount))
            else:
                row.used += amount
            s.commit()

    def remember(self, brand_id: int, video: dict[str, Any], now: datetime) -> None:
        with self.factory() as s:
            row = s.scalars(select(YoutubeVideo).where(
                YoutubeVideo.brand_id == brand_id, YoutubeVideo.video_id == video["id"])).first()
            if row is None:
                s.add(YoutubeVideo(brand_id=brand_id, video_id=video["id"], title=video["title"][:512],
                                   channel=(video.get("channel") or "")[:255] or None,
                                   published_at=video["published_at"], first_seen_at=now))
            else:
                row.title = video["title"][:512]
            s.commit()

    def videos_to_read(self, brand_id: int, newer_than: datetime, limit: int) -> list[str]:
        with self.factory() as s:
            return list(s.scalars(
                select(YoutubeVideo.video_id)
                .where(YoutubeVideo.brand_id == brand_id, YoutubeVideo.published_at >= newer_than,
                       YoutubeVideo.comments_disabled.is_(False))
                .order_by(YoutubeVideo.published_at.desc()).limit(limit)
            ))

    def checked(self, brand_id: int, video_id: str, now: datetime, disabled: bool = False) -> None:
        with self.factory() as s:
            s.execute(update(YoutubeVideo)
                      .where(YoutubeVideo.brand_id == brand_id, YoutubeVideo.video_id == video_id)
                      .values(comments_checked_at=now, comments_disabled=disabled))
            s.commit()


def purge_expired(session: Session, now: datetime | None = None) -> tuple[int, int]:
    """Delete YouTube mentions (with their analysis) and remembered videos not refreshed for 30 days.
    Returns (mentions, videos) deleted. The caller commits."""
    cutoff = (now or utc_now()) - RETENTION
    stale = select(Mention.id).where(Mention.source == "youtube", Mention.fetched_at < cutoff)
    ids = list(session.scalars(stale))
    if ids:
        session.execute(delete(Analysis).where(Analysis.mention_id.in_(ids)))  # MySQL cascades; SQLite doesn't
        session.execute(delete(Mention).where(Mention.id.in_(ids)))
    seen = YoutubeVideo.comments_checked_at.is_(None) & (YoutubeVideo.first_seen_at < cutoff)
    videos = session.execute(delete(YoutubeVideo).where((YoutubeVideo.comments_checked_at < cutoff) | seen))
    return len(ids), int(videos.rowcount or 0)


# ---------------------------------------------------------------- source


class YouTubeSource(BaseSource):
    name = "youtube"

    def __init__(
        self,
        api_key: str | None = None,
        client: HttpClient | None = None,
        store: YoutubeStore | None = None,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        super().__init__()
        self.api_key = settings.youtube_api_key if api_key is None else api_key
        self.enabled = bool(self.api_key)
        self.client = client or HttpClient(make_session(), RateLimiter(settings.http_min_interval))
        self._store = store
        self.clock = clock

    @property
    def store(self) -> YoutubeStore:
        if self._store is None:
            self._store = DbYoutubeStore()
        return self._store

    def fetch(self, brand: BrandSpec, since: datetime) -> list[RawMention]:
        self.reset()
        now = self.clock()
        found: dict[str, RawMention] = {}
        try:
            for query in self._queries(brand):
                for video in self._search(query, since):
                    if self.accept(video["title"], brand):
                        self.store.remember(brand.id, video, now)
            for video_id in self.store.videos_to_read(brand.id, now - RETENTION, settings.youtube_max_videos):
                for mention in self._comments(brand, video_id, now):
                    found.setdefault(mention.external_id, mention)
        except QuotaExhausted as exc:  # keep what was read before the budget ran out
            self.warn(str(exc))
        return cap_per_author(list(found.values()))

    def preview(self, brand: BrandSpec, since: datetime) -> list[str]:
        """Titles of matching videos, without reading comments or remembering anything (dry-run).
        Costs one search call per query."""
        self.reset()
        titles: list[str] = []
        try:
            for query in self._queries(brand):
                titles += [v["title"] for v in self._search(query, since) if self.accept(v["title"], brand)]
        except QuotaExhausted as exc:
            self.warn(str(exc))
        return list(dict.fromkeys(titles))

    def _queries(self, brand: BrandSpec) -> list[str]:
        return list(dict.fromkeys(brand.youtube_queries or brand.keywords[:2]))

    # -- quota

    def _spend(self, bucket: str, budget: int) -> None:
        day = quota_day(self.clock())
        if self.store.used(bucket, day) + 1 > budget:
            what = "search calls" if bucket == "search" else "quota units"
            raise QuotaExhausted(f"daily YouTube budget of {budget} {what} used up; resumes after midnight Pacific")
        self.store.spend(bucket, day, 1)

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self.client.session.get(
                f"{API}/{path}", params=params, headers={"X-Goog-Api-Key": self.api_key},
                timeout=self.client.timeout,
            )
        except requests.RequestException as exc:  # message has no key: it is only in the header
            raise RuntimeError(f"YouTube {path}: {type(exc).__name__}") from None
        if response.status_code == 200:
            return response.json()
        reason = _error_reason(response)
        if reason in ("quotaExceeded", "dailyLimitExceeded", "rateLimitExceeded"):
            raise QuotaExhausted(f"YouTube quota exceeded ({reason}); resumes after midnight Pacific")
        if reason == "commentsDisabled":
            raise CommentsDisabled()
        if reason == "videoNotFound":
            raise CommentsDisabled()  # removed or private: nothing to read
        raise RuntimeError(f"YouTube {path}: HTTP {response.status_code} {reason or ''}".strip())

    # -- calls

    def _search(self, query: str, since: datetime) -> list[dict[str, Any]]:
        self._spend("search", settings.youtube_search_budget)
        data = self._get("search", {
            "part": "snippet", "q": query, "type": "video", "order": "date", "maxResults": SEARCH_RESULTS,
            "publishedAfter": since.strftime("%Y-%m-%dT%H:%M:%SZ"), "relevanceLanguage": "en",
        })
        videos = []
        for item in data.get("items", []):
            video_id = (item.get("id") or {}).get("videoId")
            snippet = item.get("snippet") or {}
            if not video_id or not snippet.get("title"):
                continue
            videos.append({
                "id": video_id, "title": html.unescape(snippet["title"]), "channel": snippet.get("channelTitle"),
                "published_at": parse_time(snippet["publishedAt"]),
            })
        return videos

    def _comments(self, brand: BrandSpec, video_id: str, now: datetime) -> list[RawMention]:
        mentions: list[RawMention] = []
        page_token: str | None = None
        for _ in range(settings.youtube_comment_pages):
            self._spend("units", settings.youtube_unit_budget)
            params = {"part": "snippet", "videoId": video_id, "maxResults": COMMENTS_PER_PAGE,
                      "order": "time", "textFormat": "plainText"}
            if page_token:
                params["pageToken"] = page_token
            try:
                data = self._get("commentThreads", params)
            except CommentsDisabled:
                self.store.checked(brand.id, video_id, now, disabled=True)
                return mentions
            for item in data.get("items", []):
                top = ((item.get("snippet") or {}).get("topLevelComment") or {})
                s = top.get("snippet") or {}
                text = (s.get("textOriginal") or s.get("textDisplay") or "").strip()
                if not text or excluded_by_context(text, brand) or too_short_unnamed(text, brand):
                    continue
                mentions.append(RawMention(
                    source=self.name, external_id=item["id"], text=text, created_at=parse_time(s["publishedAt"]),
                    author=s.get("authorDisplayName"), url=WATCH_URL.format(video=video_id, comment=item["id"]),
                ))
            page_token = data.get("nextPageToken")
            if not page_token:
                break
        self.store.checked(brand.id, video_id, now)
        return mentions


class CommentsDisabled(Exception):
    pass


def _error_reason(response: requests.Response) -> str | None:
    try:
        errors = response.json().get("error", {}).get("errors") or []
        return errors[0].get("reason") if errors else response.json().get("error", {}).get("status")
    except ValueError:
        return None
