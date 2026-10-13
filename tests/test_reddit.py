"""Reddit through a fake PRAW client: search, brand subreddits, top-level comments, matching,
dedupe, caps, failures and the disabled state. No network, no credentials."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Any

import pytest

import sources.reddit as reddit_module
from sources.base import BrandSpec, RateLimiter
from sources.reddit import RedditSource

BRAND = BrandSpec(id=1, name="Obsidian", keywords=("Obsidian", "obsidian.md"),
                  exclude=("Obsidian Entertainment",), exclude_if=("Fallout",), subreddits=("ObsidianMD",))
SINCE = datetime(2026, 10, 1)


def ts(day: int) -> float:
    return datetime(2026, 10, day, 12, tzinfo=timezone.utc).timestamp()


class FakeComment:
    def __init__(self, cid: str, body: str, day: int = 6, author: str | None = "commenter") -> None:
        self.id, self.body, self.created_utc, self.author = cid, body, ts(day), author
        self.permalink = f"/r/x/comments/p/x/{cid}/"


class FakeForest(list):
    def __init__(self, comments: list[FakeComment]) -> None:
        super().__init__(comments)
        self.replace_more_calls: list[int] = []

    def replace_more(self, limit: int) -> None:
        self.replace_more_calls.append(limit)


class FakePost:
    def __init__(self, pid: str, title: str, selftext: str = "", day: int = 5, sub: str = "productivity",
                 comments: list[FakeComment] | None = None, author: str | None = "poster") -> None:
        self.id, self.title, self.selftext, self.created_utc = pid, title, selftext, ts(day)
        self.subreddit, self.author = sub, author
        self.permalink = f"/r/{sub}/comments/{pid}/x/"
        self.comments = FakeForest(comments or [])


class FakeSubreddit:
    def __init__(self, client: "FakeReddit", name: str) -> None:
        self.client, self.name = client, name

    def search(self, query: str, **kwargs: Any) -> list[FakePost]:
        self.client.calls.append(("search", self.name, query, kwargs["sort"], kwargs["time_filter"]))
        if "search" in self.client.fail:
            raise RuntimeError("401 Unauthorized")
        return self.client.search_results.get(query.strip('"'), [])

    def new(self, limit: int) -> list[FakePost]:
        self.client.calls.append(("new", self.name, limit))
        if self.name in self.client.fail:
            raise RuntimeError("403 private subreddit")
        return self.client.sub_posts.get(self.name, [])


class FakeReddit:
    def __init__(self, search_results: dict[str, list[FakePost]] | None = None,
                 sub_posts: dict[str, list[FakePost]] | None = None, fail: tuple[str, ...] = ()) -> None:
        self.search_results = search_results or {}
        self.sub_posts = sub_posts or {}
        self.fail = fail
        self.calls: list[tuple] = []

    def subreddit(self, name: str) -> FakeSubreddit:
        return FakeSubreddit(self, name)


def source(fake: FakeReddit, **kwargs: Any) -> RedditSource:
    return RedditSource(client_factory=lambda: fake, enabled=True, limiter=RateLimiter(0), **kwargs)


def ids(mentions: list) -> list[str]:
    return sorted(m.external_id for m in mentions)


# ---------------------------------------------------------------- disabled


def test_disabled_without_credentials_never_imports_praw(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "praw", raising=False)

    def must_not_be_called() -> Any:
        raise AssertionError("PRAW must not be created without credentials")

    src = RedditSource(client_factory=must_not_be_called, enabled=False)
    assert src.enabled is False and src.fetch(BRAND, SINCE) == []
    assert "praw" not in sys.modules


def test_enabled_follows_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    import dataclasses

    monkeypatch.setattr(reddit_module, "settings", dataclasses.replace(reddit_module.settings,
                                                                       reddit_client_id="", reddit_client_secret=""))
    assert RedditSource().enabled is False
    monkeypatch.setattr(reddit_module, "settings", dataclasses.replace(reddit_module.settings,
                                                                       reddit_client_id="id", reddit_client_secret="s"))
    assert RedditSource().enabled is True


# ---------------------------------------------------------------- search + matching


def test_search_posts_need_the_brand_and_respect_excludes() -> None:
    fake = FakeReddit(search_results={"obsidian": [
        FakePost("a", "Obsidian sync vs Notion", "Which is better?"),
        FakePost("b", "Obsidian Entertainment announces a sequel"),  # exclude phrase removed -> no brand left
        FakePost("c", "Obsidian made Fallout New Vegas"),  # exclude_if context word
        FakePost("d", "my obsidian rock collection"),  # lowercase word, not the product
        FakePost("e", "Old Obsidian post", day=1),  # exactly at `since`: kept
        FakePost("f", "Ancient Obsidian post", day=1),
    ]})
    fake.search_results["obsidian"][-1].created_utc = datetime(2026, 9, 20, tzinfo=timezone.utc).timestamp()
    mentions = source(fake).fetch(BRAND, SINCE)
    assert ids(mentions) == ["t3_a", "t3_e"]
    a = next(m for m in mentions if m.external_id == "t3_a")
    assert a.text == "Obsidian sync vs Notion\n\nWhich is better?" and a.source == "reddit"
    assert a.url == "https://www.reddit.com/r/productivity/comments/a/x/" and a.author == "poster"
    # one search per distinct keyword, across all of Reddit, newest first, past month
    searches = [c for c in fake.calls if c[0] == "search"]
    assert [(c[1], c[2], c[3]) for c in searches] == [("all", '"obsidian"', "new"), ("all", '"obsidian.md"', "new")]


def test_brand_subreddit_posts_count_without_the_name_and_stop_at_since() -> None:
    fake = FakeReddit(sub_posts={"ObsidianMD": [
        FakePost("s1", "Sync broke again after the update", sub="ObsidianMD", day=9),
        FakePost("s2", "Loving the new canvas", sub="ObsidianMD", day=7),
        FakePost("s3", "Playing Fallout tonight", sub="ObsidianMD", day=6),  # exclude_if still applies
        FakePost("s4", "Older post", sub="ObsidianMD", day=1),
    ]})
    fake.sub_posts["ObsidianMD"][-1].created_utc = datetime(2026, 9, 1, tzinfo=timezone.utc).timestamp()
    assert ids(source(fake).fetch(BRAND, SINCE)) == ["t3_s1", "t3_s2"]
    assert ("new", "ObsidianMD", reddit_module.POSTS_PER_QUERY) in fake.calls


def test_search_hit_inside_own_subreddit_does_not_need_the_name() -> None:
    fake = FakeReddit(search_results={"obsidian": [FakePost("x", "how do I fix this sync issue", sub="ObsidianMD")]})
    assert ids(source(fake).fetch(BRAND, SINCE)) == ["t3_x"]


# ---------------------------------------------------------------- comments


def test_top_level_comments() -> None:
    elsewhere = FakePost("a", "Obsidian vs Notion for students", sub="productivity", comments=[
        FakeComment("c1", "Obsidian is faster for me"),
        FakeComment("c2", "I just use paper"),  # elsewhere: must name the brand
        FakeComment("c3", "[deleted]", author=None),
        FakeComment("c4", "Obsidian, but it was weeks ago", day=1),
    ])
    elsewhere.comments[-1].created_utc = datetime(2026, 9, 25, tzinfo=timezone.utc).timestamp()
    own = FakePost("s1", "Sync broke again", sub="ObsidianMD", comments=[
        FakeComment("c5", "Same here since 1.7"),  # own subreddit: counts without the name
        FakeComment("c6", "Fallout 4 mods?"),  # exclude_if
    ])
    fake = FakeReddit(search_results={"obsidian": [elsewhere]}, sub_posts={"ObsidianMD": [own]})
    mentions = source(fake).fetch(BRAND, SINCE)
    assert ids(mentions) == ["t1_c1", "t1_c5", "t3_a", "t3_s1"]
    c1 = next(m for m in mentions if m.external_id == "t1_c1")
    assert c1.text == "Obsidian is faster for me" and c1.url.endswith("/c1/") and c1.author == "commenter"
    assert elsewhere.comments.replace_more_calls == [0]  # top level only: no "load more" requests


def test_comment_reading_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reddit_module, "COMMENT_POSTS", 2)
    monkeypatch.setattr(reddit_module, "COMMENTS_PER_POST", 3)
    posts = [FakePost(f"p{i}", f"Obsidian thread {i}", day=2 + i,
                      comments=[FakeComment(f"p{i}c{j}", f"Obsidian reply {j}") for j in range(10)]) for i in range(5)]
    fake = FakeReddit(search_results={"obsidian": posts})
    mentions = source(fake).fetch(BRAND, SINCE)
    comment_parents = {m.external_id.split("c")[0] for m in mentions if m.external_id.startswith("t1_")}
    assert comment_parents == {"t1_p4", "t1_p3"}  # only the 2 newest posts
    assert sum(m.external_id.startswith("t1_") for m in mentions) == 6  # 3 comments each
    assert sum(len(p.comments.replace_more_calls) for p in posts) == 2


# ---------------------------------------------------------------- dedupe + caps


def test_dedupe_across_keywords_subreddit_and_comments() -> None:
    shared_comment = FakeComment("c1", "Obsidian graph view is great")
    post = FakePost("a", "Obsidian tips", sub="ObsidianMD", comments=[shared_comment])
    same_post_again = FakePost("a", "Obsidian tips", sub="ObsidianMD", comments=[shared_comment])
    fake = FakeReddit(search_results={"obsidian": [post], "obsidian.md": [same_post_again]},
                      sub_posts={"ObsidianMD": [post]})
    mentions = source(fake).fetch(BRAND, SINCE)
    assert ids(mentions) == ["t1_c1", "t3_a"]  # once each, posts and comments never collide
    assert len(post.comments.replace_more_calls) == 1  # comments read once per post


def test_max_items_cap() -> None:
    fake = FakeReddit(search_results={"obsidian": [FakePost(str(i), f"Obsidian {i}") for i in range(10)]})
    assert len(source(fake, max_items=4).fetch(BRAND, SINCE)) == 4


# ---------------------------------------------------------------- failures


def test_a_failing_subreddit_is_a_warning() -> None:
    fake = FakeReddit(search_results={"obsidian": [FakePost("a", "Obsidian sync")]}, fail=("ObsidianMD",))
    src = source(fake)
    assert ids(src.fetch(BRAND, SINCE)) == ["t3_a"]
    assert len(src.warnings) == 1 and src.warnings[0].startswith("r/ObsidianMD: RuntimeError")


def test_every_request_failing_is_an_error() -> None:
    fake = FakeReddit(fail=("search", "ObsidianMD"))
    with pytest.raises(RuntimeError, match="every Reddit request failed"):
        source(fake).fetch(BRAND, SINCE)


def test_rate_limiter_called_for_every_request() -> None:
    waits: list[str] = []

    class CountingLimiter(RateLimiter):
        def wait(self, key: str) -> None:
            waits.append(key)

    fake = FakeReddit(search_results={"obsidian": [FakePost("a", "Obsidian", comments=[FakeComment("c", "Obsidian!")])]},
                      sub_posts={"ObsidianMD": []})
    RedditSource(client_factory=lambda: fake, enabled=True, limiter=CountingLimiter(0)).fetch(BRAND, SINCE)
    assert waits == ["reddit"] * 4  # 2 searches + 1 subreddit + 1 comment fetch
