"""Application settings, loaded from environment variables / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent

load_dotenv(PROJECT_ROOT / ".env")


def _get_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _get_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _get_list(name: str, default: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in os.getenv(name, default).split(",") if part.strip())


@dataclass(frozen=True)
class Settings:
    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str
    collector_interval_hours: float
    demo: bool
    fetch_since_days: int
    max_items_per_source: int
    appstore_countries: tuple[str, ...]
    playstore_lang: str
    playstore_country: str
    http_timeout: float
    http_retries: int
    http_min_interval: float
    http_user_agent: str
    nlp_batch_size: int
    nlp_max_tokens: int
    nlp_zero_shot_max_tokens: int
    nlp_device: str
    feature_threshold: float
    topic_min_complaints: int
    topic_min_size: int
    topic_outlier_similarity: float
    topic_excluded_sources: tuple[str, ...]
    priority_volume_threshold: int
    priority_pain_threshold: float
    priority_praise_pain_max: float
    alert_trend_ratio: float
    alert_min_mentions: int
    mastodon_instances: tuple[str, ...]
    reddit_client_id: str
    reddit_client_secret: str
    reddit_user_agent: str
    youtube_api_key: str
    youtube_search_budget: int
    youtube_unit_budget: int
    youtube_max_videos: int
    youtube_comment_pages: int
    news_min_interval: float

    @property
    def database_url(self) -> str:
        """SQLAlchemy URL; user and password are URL-encoded."""
        return (
            f"mysql+pymysql://{quote_plus(self.db_user)}:{quote_plus(self.db_password)}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}?charset=utf8mb4"
        )

    @property
    def reddit_enabled(self) -> bool:
        return bool(self.reddit_client_id and self.reddit_client_secret)

    @property
    def youtube_enabled(self) -> bool:
        return bool(self.youtube_api_key)


def _get_bool(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def load_settings() -> Settings:
    demo = _get_bool("BRAND_RADAR_DEMO")
    return Settings(
        db_host=os.getenv("DB_HOST", "127.0.0.1"),
        db_port=_get_int("DB_PORT", 3306),
        # BRAND_RADAR_DEMO=1 points everything at the separate demo database (see seed_demo.py)
        db_name=os.getenv("DEMO_DB_NAME", "brand_radar_demo") if demo else os.getenv("DB_NAME", "brand_radar"),
        db_user=os.getenv("DB_USER", "radar"),
        db_password=os.getenv("DB_PASSWORD", ""),
        collector_interval_hours=_get_float("COLLECTOR_INTERVAL_HOURS", 6.0),
        demo=demo,
        fetch_since_days=_get_int("FETCH_SINCE_DAYS", 30),
        max_items_per_source=_get_int("MAX_ITEMS_PER_SOURCE", 500),
        appstore_countries=_get_list("APPSTORE_COUNTRIES", "us,in,gb,ca,au"),
        playstore_lang=os.getenv("PLAYSTORE_LANG", "en"),
        playstore_country=os.getenv("PLAYSTORE_COUNTRY", "us"),
        http_timeout=_get_float("HTTP_TIMEOUT", 15.0),
        http_retries=_get_int("HTTP_RETRIES", 3),
        http_min_interval=_get_float("HTTP_MIN_INTERVAL", 0.5),
        http_user_agent=os.getenv("HTTP_USER_AGENT", "BrandRadar/0.2 (brand sentiment research)"),
        nlp_batch_size=_get_int("NLP_BATCH_SIZE", 8),
        nlp_max_tokens=_get_int("NLP_MAX_TOKENS", 512),
        nlp_zero_shot_max_tokens=_get_int("NLP_ZERO_SHOT_MAX_TOKENS", 256),
        nlp_device=os.getenv("NLP_DEVICE", "auto").lower(),
        feature_threshold=_get_float("FEATURE_THRESHOLD", 0.5),
        topic_min_complaints=_get_int("TOPIC_MIN_COMPLAINTS", 30),
        topic_min_size=_get_int("TOPIC_MIN_SIZE", 5),
        topic_outlier_similarity=_get_float("TOPIC_OUTLIER_SIMILARITY", 0.5),
        topic_excluded_sources=_get_list("TOPIC_EXCLUDED_SOURCES", "devto,news"),
        priority_volume_threshold=_get_int("PRIORITY_VOLUME_THRESHOLD", 5),
        priority_pain_threshold=_get_float("PRIORITY_PAIN_THRESHOLD", 0.15),
        priority_praise_pain_max=_get_float("PRIORITY_PRAISE_PAIN_MAX", 0.02),
        alert_trend_ratio=_get_float("ALERT_TREND_RATIO", 2.0),
        alert_min_mentions=_get_int("ALERT_MIN_MENTIONS", 5),
        mastodon_instances=_get_list("MASTODON_INSTANCES", "mastodon.social,fosstodon.org,hachyderm.io"),
        reddit_client_id=os.getenv("REDDIT_CLIENT_ID", ""),
        reddit_client_secret=os.getenv("REDDIT_CLIENT_SECRET", ""),
        reddit_user_agent=os.getenv("REDDIT_USER_AGENT", "brand-radar/0.1"),
        youtube_api_key=os.getenv("YOUTUBE_API_KEY", "").strip(),
        youtube_search_budget=_get_int("YOUTUBE_SEARCH_BUDGET", 80),
        youtube_unit_budget=_get_int("YOUTUBE_UNIT_BUDGET", 8000),
        youtube_max_videos=_get_int("YOUTUBE_MAX_VIDEOS", 30),
        youtube_comment_pages=_get_int("YOUTUBE_COMMENT_PAGES", 2),
        news_min_interval=_get_float("NEWS_MIN_INTERVAL", 6.0),
    )


settings = load_settings()
