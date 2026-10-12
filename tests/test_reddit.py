"""Reddit goes through PRAW, so a fake PRAW client is injected."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from sources.base import BrandSpec, RateLimiter
from sources.reddit import RedditSource

BRAND = BrandSpec(id=1, name="Obsidian", keywords=("Obsidian", "obsidian.md"), exclude=("Obsidian Entertainment",))
SINCE = datetime(2026, 10, 1)


def post(post_id: str, title: str, selftext: str = "", day: int = 5) -> SimpleNamespace:
    created = datetime(2026, 10, day, tzinfo=timezone.utc).timestamp()
    return SimpleNamespace(
        id=post_id, title=title, selftext=selftext, created_utc=created,
        author="someone", permalink=f"/r/ObsidianMD/comments/{post_id}/x/",
    )


class FakeReddit:
    def __init__(self, posts: list[SimpleNamespace]) -> None:
        self.posts = posts
        self.queries: list[str] = []

    def subreddit(self, name: str) -> "FakeReddit":
        assert name == "all"
        return self

    def search(self, query: str, **kwargs: Any) -> list[SimpleNamespace]:
        self.queries.append(query)
        return self.posts


def test_disabled_without_credentials_does_nothing() -> None:
    def must_not_be_called() -> Any:
        raise AssertionError("PRAW must not be created without credentials")

    source = RedditSource(client_factory=must_not_be_called, enabled=False)
    assert source.enabled is False
    assert source.fetch(BRAND, SINCE) == []


def test_fetch_filters_and_dedupes() -> None:
    fake = FakeReddit([
        post("p1", "Obsidian sync vs Notion", "Which is better?"),
        post("p2", "Obsidian Entertainment announces sequel"),  # excluded phrase
        post("p3", "Obsidian post exactly at the cutoff", day=1),  # created == since is kept
        post("p1", "Obsidian sync vs Notion (duplicate)"),
    ])
    old = post("p4", "Obsidian ancient post")
    old.created_utc = datetime(2026, 9, 20, tzinfo=timezone.utc).timestamp()  # before since
    fake.posts.append(old)
    source = RedditSource(client_factory=lambda: fake, enabled=True, limiter=RateLimiter(0))

    mentions = source.fetch(BRAND, SINCE)

    assert [m.external_id for m in mentions] == ["p1", "p3"]
    assert mentions[0].text == "Obsidian sync vs Notion\n\nWhich is better?"
    assert mentions[0].url == "https://www.reddit.com/r/ObsidianMD/comments/p1/x/"
    assert fake.queries == ['"obsidian"', '"obsidian.md"']
