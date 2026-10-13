"""Shared pieces for all data sources: the Source interface, RawMention, brand matching and polite HTTP."""

from __future__ import annotations

import logging
import re
import socket
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import settings

if TYPE_CHECKING:
    from db.models import Brand as BrandModel

log = logging.getLogger(__name__)


# Every source fetch.py knows (a brand's `sources:` in brands.yaml must use these names).
ALL_SOURCES = ("hackernews", "appstore", "playstore", "reddit", "devto", "mastodon", "youtube", "news")
# Sources found by keyword, tag, community or video (as opposed to per-app reviews), so brand rules
# apply to them; see still_matches.
KEYWORD_SOURCES = ("hackernews", "reddit", "mastodon", "devto", "youtube", "news")

# ---------------------------------------------------------------- data types


@dataclass(frozen=True)
class BrandSpec:
    """A plain, session-free snapshot of a row in `brands`, handed to sources."""

    id: int
    name: str
    is_own: bool = False
    keywords: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    exclude_if: tuple[str, ...] = ()
    subreddits: tuple[str, ...] = ()  # the brand's own subreddits (Reddit source)
    require_context: tuple[str, ...] = ()  # ambiguous names: one of these must appear too
    group: str = "Default"
    profile: str = "consumer_product"
    youtube_queries: tuple[str, ...] = ()
    sources: tuple[str, ...] | None = None  # None = the profile's default sources
    app_store_id: str | None = None
    play_store_id: str | None = None
    features: tuple[str, ...] = ()

    @classmethod
    def from_model(cls, brand: BrandModel) -> BrandSpec:
        include, exclude, exclude_if = split_keywords(brand.keywords or [])
        return cls(
            id=brand.id,
            name=brand.name,
            is_own=bool(brand.is_own),
            keywords=include,
            exclude=exclude,
            exclude_if=exclude_if,
            subreddits=split_subreddits(brand.keywords or []),
            require_context=tuple((brand.config or {}).get("require_context") or ()),
            group=brand.brand_group or "Default",
            profile=brand.profile or "consumer_product",
            youtube_queries=tuple((brand.config or {}).get("youtube_queries") or ()),
            sources=tuple(brand.config["sources"]) if (brand.config or {}).get("sources") else None,
            app_store_id=brand.app_store_id,
            play_store_id=brand.play_store_id,
            features=tuple(brand.features or []),
        )


@dataclass
class RawMention:
    """One piece of text from a source, before it is stored."""

    source: str
    external_id: str
    text: str
    created_at: datetime  # naive UTC
    author: str | None = None
    url: str | None = None
    rating: int | None = None


def brand_sources(brand: BrandSpec) -> tuple[str, ...]:
    """Sources to fetch for a brand: its own list, else its profile's defaults."""
    if brand.sources is not None:
        return brand.sources
    from nlp.profiles import get_profile

    return get_profile(brand.profile).sources


@runtime_checkable
class Source(Protocol):
    name: str
    enabled: bool
    warnings: list[str]

    def fetch(self, brand: BrandSpec, since: datetime) -> list[RawMention]: ...


class BaseSource:
    """Common state for sources. `warnings` collects non-fatal problems from the last fetch."""

    name: str = "base"
    enabled: bool = True

    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.rejected: list[tuple[str, str]] = []  # (text, reason) from the last fetch, for the dry-run

    def warn(self, message: str) -> None:
        log.warning("%s: %s", self.name, message)
        self.warnings.append(message)

    def reset(self) -> None:
        self.warnings = []
        self.rejected = []

    def accept(self, text: str, brand: BrandSpec) -> bool:
        """matches_brand, remembering what was rejected and why (see `manage_brands.py dry-run`)."""
        reason = reject_reason(text, brand)
        if reason and len(self.rejected) < MAX_REJECTED_KEPT:
            self.rejected.append((text, reason))
        return reason is None


# ---------------------------------------------------------------- time helpers


def to_utc_naive(value: datetime) -> datetime:
    """Convert any datetime to naive UTC. Naive input is treated as local time."""
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------- text helpers


class _TextExtractor(HTMLParser):
    BREAK_TAGS = {"p", "br", "div", "li", "pre", "blockquote"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.BREAK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def html_to_text(html: str | None) -> str:
    """Strip tags, decode entities and tidy whitespace, keeping paragraph breaks."""
    if not html:
        return ""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    lines = (re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in "".join(parser.parts).split("\n"))
    return "\n".join(line for line in lines if line)


def split_keywords(stored: list[str]) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """`brands.keywords` holds search keywords, exclude phrases prefixed with '-' (removed before
    matching) and exclude_if context words prefixed with '!' (drop the whole mention)."""
    include = tuple(k for k in stored if not k.startswith(("-", "!", "r/")))
    exclude = tuple(k[1:] for k in stored if k.startswith("-"))
    exclude_if = tuple(k[1:] for k in stored if k.startswith("!"))
    return include, exclude, exclude_if


def split_subreddits(stored: list[str]) -> tuple[str, ...]:
    """Brand subreddits are stored in `brands.keywords` as 'r/Name'."""
    return tuple(k[2:] for k in stored if k.startswith("r/"))


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    # Phrases with a capital letter are case-sensitive ("Notion" is a product, "notion" is a word).
    flags = 0 if any(c.isupper() for c in phrase) else re.IGNORECASE
    return re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", flags)


def excluded_by_context(text: str, brand: BrandSpec) -> bool:
    """True if an exclude_if context word appears (e.g. "Fallout" for Obsidian the game studio)."""
    return any(_phrase_pattern(word).search(text) for word in brand.exclude_if)


def has_required_context(text: str, brand: BrandSpec) -> bool:
    """Ambiguous names ("Centella", "Apsara") also need a context word (skin, serum…; pencil, eraser…).
    Context words match case-insensitively, as whole words with simple endings."""
    if not brand.require_context:
        return True
    return any(re.search(rf"(?<!\w){re.escape(w)}(?:s|es|ed|ing)?(?!\w)", text, re.IGNORECASE) for w in brand.require_context)


def reject_reason(text: str, brand: BrandSpec) -> str | None:
    """Why a text is not about the brand, or None if it is (see matches_brand)."""
    word = next((w for w in brand.exclude_if if _phrase_pattern(w).search(text)), None)
    if word:
        return f"exclude_if: {word}"
    stripped = text
    for phrase in brand.exclude:
        stripped = _phrase_pattern(phrase).sub(" ", stripped)
    if not any(_phrase_pattern(k).search(stripped) for k in brand.keywords):
        return "only an excluded phrase" if stripped != text else "name not found"
    if not has_required_context(text, brand):
        return "no context word"
    return None


def matches_brand(text: str, brand: BrandSpec) -> bool:
    """True if a keyword appears as a whole word once the exclude phrases are removed, none of the
    exclude_if words appear (e.g. "Fallout" for Obsidian the studio), and, for brands with
    require_context, at least one context word appears too."""
    return reject_reason(text, brand) is None


def still_matches(source: str, text: str, url: str | None, brand: BrandSpec) -> bool:
    """Would a stored mention still be collected under the brand's current rules? Mirrors the sources:
    dev.to items come from the brand's tag, YouTube comments from a video whose title named the brand, and
    posts in the brand's own subreddit don't need the name; everything else must name the brand.
    exclude_if always applies."""
    if excluded_by_context(text, brand):
        return False
    if source in ("devto", "youtube"):
        return True
    if source == "reddit" and url and any(f"/r/{s.lower()}/" in url.lower() for s in brand.subreddits):
        return True
    return matches_brand(text, brand)


# ---------------------------------------------------------------- polite HTTP


class RateLimiter:
    """Keeps at least `min_interval` seconds between requests to the same host."""

    def __init__(
        self,
        min_interval: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, key: str) -> None:
        with self._lock:
            last = self._last.get(key)
            if last is not None:
                remaining = self.min_interval - (self._clock() - last)
                if remaining > 0:
                    self._sleep(remaining)
            self._last[key] = self._clock()


def make_session(retries: int | None = None, user_agent: str | None = None) -> requests.Session:
    """requests.Session with a User-Agent and retries with exponential backoff on 429/5xx."""
    retry = Retry(
        total=settings.http_retries if retries is None else retries,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers["User-Agent"] = user_agent or settings.http_user_agent
    session.headers["Accept"] = "application/json"
    return session


class HttpClient:
    """Thin wrapper adding timeouts and per-host rate limiting to a session."""

    def __init__(
        self,
        session: requests.Session | None = None,
        limiter: RateLimiter | None = None,
        timeout: float | None = None,
    ) -> None:
        self.session = session or make_session()
        self.limiter = limiter or RateLimiter(settings.http_min_interval)
        self.timeout = settings.http_timeout if timeout is None else timeout

    def get(self, url: str, params: dict[str, Any] | None = None) -> requests.Response:
        self.limiter.wait(urlparse(url).netloc)
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        return self.get(url, params).json()


_shared_client: HttpClient | None = None


def shared_client() -> HttpClient:
    """One client per process, so the rate limiter covers every source."""
    global _shared_client
    if _shared_client is None:
        _shared_client = HttpClient()
    return _shared_client


@contextmanager
def socket_timeout(seconds: float) -> Iterator[None]:
    """Temporarily set the default socket timeout, for libraries that don't take one."""
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(seconds)
    try:
        yield
    finally:
        socket.setdefaulttimeout(previous)


def with_retries(
    func: Callable[[], Any],
    attempts: int,
    backoff: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Call `func`, retrying with exponential backoff (for non-requests libraries)."""
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001 — the caller decides what is fatal
            if attempt == attempts:
                raise
            delay = backoff * 2 ** (attempt - 1)
            log.info("retry %d/%d after %s: %s", attempt, attempts - 1, type(exc).__name__, exc)
            sleep(delay)
    raise RuntimeError("unreachable")


MAX_REJECTED_KEPT = 500
MAX_PER_AUTHOR = 3  # social sources: one prolific author must not flood a brand (e.g. a plugin maker)


def cap_per_author(mentions: list[RawMention], limit: int = MAX_PER_AUTHOR) -> list[RawMention]:
    """Keep at most `limit` mentions per author (the newest); mentions without an author are kept."""
    kept: list[RawMention] = []
    counts: dict[str, int] = {}
    for m in sorted(mentions, key=lambda m: m.created_at, reverse=True):
        if m.author:
            counts[m.author] = counts.get(m.author, 0) + 1
            if counts[m.author] > limit:
                continue
        kept.append(m)
    return kept


def clip(value: str | None, length: int) -> str | None:
    return None if value is None else value[:length]
