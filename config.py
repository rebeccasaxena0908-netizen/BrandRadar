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
    fetch_interval_minutes: int
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
    alert_trend_ratio: float
    alert_min_mentions: int
    reddit_client_id: str
    reddit_client_secret: str
    reddit_user_agent: str

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


def load_settings() -> Settings:
    return Settings(
        db_host=os.getenv("DB_HOST", "127.0.0.1"),
        db_port=_get_int("DB_PORT", 3306),
        db_name=os.getenv("DB_NAME", "brand_radar"),
        db_user=os.getenv("DB_USER", "radar"),
        db_password=os.getenv("DB_PASSWORD", ""),
        fetch_interval_minutes=_get_int("FETCH_INTERVAL_MINUTES", 60),
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
        alert_trend_ratio=_get_float("ALERT_TREND_RATIO", 2.0),
        alert_min_mentions=_get_int("ALERT_MIN_MENTIONS", 5),
        reddit_client_id=os.getenv("REDDIT_CLIENT_ID", ""),
        reddit_client_secret=os.getenv("REDDIT_CLIENT_SECRET", ""),
        reddit_user_agent=os.getenv("REDDIT_USER_AGENT", "brand-radar/0.1"),
    )


settings = load_settings()
