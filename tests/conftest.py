from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from db.models import Base, Brand
from sources.base import BrandSpec, HttpClient, RateLimiter, make_session


@pytest.fixture
def factory() -> Iterator[sessionmaker[Session]]:
    """In-memory SQLite database with the full schema (no MySQL needed)."""
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine, expire_on_commit=False)
    engine.dispose()


@pytest.fixture
def brands(factory: sessionmaker[Session]) -> list[BrandSpec]:
    with factory() as session:
        rows = [
            Brand(name="Notion", is_own=True, keywords=["Notion", "notion.so", "-Notion Ink"],
                  app_store_id="1232780281", play_store_id="notion.id", features=["sync"], brand_group="Note apps", profile="software"),
            Brand(name="Obsidian", is_own=False, keywords=["Obsidian", "-Obsidian Entertainment"],
                  app_store_id="1557175442", play_store_id="md.obsidian", features=["sync"], brand_group="Note apps", profile="software"),
        ]
        session.add_all(rows)
        session.commit()
        return [BrandSpec.from_model(b) for b in rows]


@pytest.fixture
def notion(brands: list[BrandSpec]) -> BrandSpec:
    return brands[0]


@pytest.fixture
def client() -> HttpClient:
    """HTTP client with no retries and no rate-limit delay, for use with `responses`."""
    return HttpClient(session=make_session(retries=0), limiter=RateLimiter(0), timeout=5)
