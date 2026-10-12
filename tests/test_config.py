from __future__ import annotations

import pytest

from config import load_settings

ENV_VARS = (
    "DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD",
    "FETCH_INTERVAL_MINUTES", "ALERT_TREND_RATIO", "ALERT_MIN_MENTIONS",
    "FETCH_SINCE_DAYS", "MAX_ITEMS_PER_SOURCE", "APPSTORE_COUNTRIES", "PLAYSTORE_LANG",
    "PLAYSTORE_COUNTRY", "HTTP_TIMEOUT", "HTTP_RETRIES", "HTTP_MIN_INTERVAL", "HTTP_USER_AGENT",
    "REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT",
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_defaults(clean_env: pytest.MonkeyPatch) -> None:
    s = load_settings()
    assert s.db_host == "127.0.0.1"
    assert s.db_port == 3306
    assert s.db_name == "brand_radar"
    assert s.fetch_interval_minutes == 60
    assert s.alert_trend_ratio == 2.0
    assert s.alert_min_mentions == 5
    assert s.reddit_enabled is False
    assert s.appstore_countries == ("us", "in", "gb", "ca", "au")
    assert s.max_items_per_source == 500


def test_country_list_parsed(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("APPSTORE_COUNTRIES", " us , de,,fr ")
    assert load_settings().appstore_countries == ("us", "de", "fr")


def test_database_url_encodes_password(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("DB_USER", "radar")
    clean_env.setenv("DB_PASSWORD", "p@ss:w/rd#1")
    url = load_settings().database_url
    assert url == "mysql+pymysql://radar:p%40ss%3Aw%2Frd%231@127.0.0.1:3306/brand_radar?charset=utf8mb4"


def test_numeric_settings_parsed(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("DB_PORT", "3307")
    clean_env.setenv("ALERT_TREND_RATIO", "2.5")
    s = load_settings()
    assert s.db_port == 3307
    assert s.alert_trend_ratio == 2.5


def test_reddit_enabled_needs_id_and_secret(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("REDDIT_CLIENT_ID", "abc")
    assert load_settings().reddit_enabled is False
    clean_env.setenv("REDDIT_CLIENT_SECRET", "xyz")
    assert load_settings().reddit_enabled is True
