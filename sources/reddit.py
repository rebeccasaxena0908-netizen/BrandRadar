"""Reddit (optional): posts and top-level comments via PRAW. Disabled unless REDDIT_CLIENT_ID and
REDDIT_CLIENT_SECRET are set (see "Enabling Reddit" in the README).

Reddit requires approval for API access (Responsible Builder Policy), stops taking new requests on
31 Oct 2026 and plans to end remaining public Data API access in March 2027, so this source must
never be required.

Per brand and run:
  1. keyword search across all of Reddit (newest first): posts must name the brand (case-sensitive
     keywords, exclude phrases removed first, exclude_if context words drop the post)
  2. the newest posts in the brand's own subreddits (brands.yaml `subreddits`): these count without
     naming the brand ("sync broke again" in r/ObsidianMD is about Obsidian), exclude_if still applies
  3. the top-level comments of those posts (one request per post, no "load more"): a comment counts
     if it names the brand, or if it is in the brand's own subreddit
PRAW follows Reddit's rate-limit headers itself; we also space requests and cap posts and comments.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import datetime, timedelta, timezone
from typing import Any

from config import settings
from sources.base import BaseSource, BrandSpec, RateLimiter, RawMention, excluded_by_context, matches_brand, utc_now

ClientFactory = Callable[[], Any]

POSTS_PER_QUERY = 100  # per keyword search and per brand subreddit
COMMENT_POSTS = 25  # newest posts per brand whose top-level comments are read
COMMENTS_PER_POST = 50
REMOVED = {"[deleted]", "[removed]", ""}


def _praw_client() -> Any:
    import praw

    return praw.Reddit(
        client_id=settings.reddit_client_id,
        client_secret=settings.reddit_client_secret,
        user_agent=settings.reddit_user_agent,
        check_for_async=False,
        requestor_kwargs={"timeout": settings.http_timeout},
    )


def _created(item: Any) -> datetime:
    return datetime.fromtimestamp(item.created_utc, tz=timezone.utc).replace(tzinfo=None)


def _author(item: Any) -> str | None:
    return str(item.author) if item.author else None


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
            return []  # no credentials: PRAW is never imported
        reddit = self._client_factory()
        self._calls = self._failures = 0
        own_subs = {s.lower() for s in brand.subreddits}
        found: dict[str, RawMention] = {}
        threads: dict[str, tuple[Any, bool]] = {}  # post id -> (post, in the brand's own subreddit)

        def keep_post(post: Any, own: bool) -> None:
            if _created(post) < since or post.id in threads:
                return
            text = f"{post.title}\n\n{post.selftext or ''}".strip()
            if excluded_by_context(text, brand) or not (own or matches_brand(text, brand)):
                return
            threads[post.id] = (post, own)
            found[f"t3_{post.id}"] = RawMention(
                source=self.name, external_id=f"t3_{post.id}", text=text, created_at=_created(post),
                author=_author(post), url=f"https://www.reddit.com{post.permalink}",
            )

        # 1. keyword search across Reddit; posts in the brand's own subreddit don't need the name
        time_filter = "month" if since >= utc_now() - timedelta(days=30) else "year"
        for keyword in dict.fromkeys(k.lower() for k in brand.keywords):
            for post in self._safe(f"search {keyword!r}", lambda: reddit.subreddit("all").search(
                    f'"{keyword}"', sort="new", time_filter=time_filter, limit=POSTS_PER_QUERY)):
                keep_post(post, str(post.subreddit).lower() in own_subs)

        # 2. newest posts in the brand's own subreddits (sorted newest first: stop at `since`)
        for name in brand.subreddits:
            for post in self._safe(f"r/{name}", lambda name=name: reddit.subreddit(name).new(limit=POSTS_PER_QUERY)):
                if _created(post) < since:
                    break
                keep_post(post, True)

        # 3. top-level comments of the newest kept posts
        newest = sorted(threads.values(), key=lambda pair: pair[0].created_utc, reverse=True)[:COMMENT_POSTS]
        for post, own in newest:
            for comment in self._safe(f"comments of t3_{post.id}", lambda post=post: self._top_level(post)):
                body = (comment.body or "").strip()
                if body in REMOVED or _created(comment) < since or f"t1_{comment.id}" in found:
                    continue
                if excluded_by_context(body, brand) or not (own or matches_brand(body, brand)):
                    continue
                found[f"t1_{comment.id}"] = RawMention(
                    source=self.name, external_id=f"t1_{comment.id}", text=body, created_at=_created(comment),
                    author=_author(comment), url=f"https://www.reddit.com{comment.permalink}",
                )
        if self._calls and self._failures == self._calls:  # e.g. revoked credentials: an error, not warnings
            raise RuntimeError(f"every Reddit request failed: {'; '.join(self.warnings[:3])}")
        return list(found.values())[: self.max_items]

    def _top_level(self, post: Any) -> list[Any]:
        post.comments.replace_more(limit=0)  # top level only: no extra "load more" requests
        return list(post.comments)[:COMMENTS_PER_POST]

    def _safe(self, what: str, call: Callable[[], Iterable[Any]]) -> list[Any]:
        """One rate-limited Reddit call; a failure (private/banned subreddit, network) becomes a
        warning in fetch_runs instead of failing the whole brand."""
        self.limiter.wait("reddit")
        self._calls += 1
        try:
            return list(call())
        except Exception as exc:  # noqa: BLE001
            self._failures += 1
            self.warn(f"{what}: {type(exc).__name__}: {exc}")
            return []
