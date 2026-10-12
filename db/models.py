"""SQLAlchemy ORM models. Must match db/schema.sql (checked by tests/test_models.py)."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.mysql import MEDIUMTEXT, TINYINT
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# BIGINT ids in MySQL; INTEGER on SQLite (tests) so autoincrement works there.
BigId = BigInteger().with_variant(Integer, "sqlite")


class Brand(Base):
    __tablename__ = "brands"
    __table_args__ = (UniqueConstraint("name", name="uq_brands_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100))
    is_own: Mapped[bool] = mapped_column(Boolean, default=False)
    keywords: Mapped[Optional[list[str]]] = mapped_column(JSON)
    app_store_id: Mapped[Optional[str]] = mapped_column(String(32))
    play_store_id: Mapped[Optional[str]] = mapped_column(String(255))
    # {feature: [synonyms]} from brands.yaml
    features: Mapped[Optional[dict[str, list[str]]]] = mapped_column(JSON)

    mentions: Mapped[list[Mention]] = relationship(back_populates="brand")
    topics: Mapped[list[Topic]] = relationship(back_populates="brand")

    def __repr__(self) -> str:
        return f"<Brand {self.id} {self.name!r} own={self.is_own}>"


class Mention(Base):
    __tablename__ = "mentions"
    __table_args__ = (
        UniqueConstraint(
            "brand_id", "source", "external_id", name="uq_mentions_brand_source_external"
        ),
        Index("ix_mentions_brand_created", "brand_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigId, primary_key=True, autoincrement=True)
    brand_id: Mapped[int] = mapped_column(ForeignKey("brands.id", ondelete="CASCADE"))
    source: Mapped[str] = mapped_column(String(32))
    external_id: Mapped[str] = mapped_column(String(255))
    author: Mapped[Optional[str]] = mapped_column(String(255))
    text: Mapped[str] = mapped_column(Text().with_variant(MEDIUMTEXT(), "mysql"))
    url: Mapped[Optional[str]] = mapped_column(String(1024))
    rating: Mapped[Optional[int]] = mapped_column(
        SmallInteger().with_variant(TINYINT(unsigned=True), "mysql")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    brand: Mapped[Brand] = relationship(back_populates="mentions")
    analysis: Mapped[Optional[Analysis]] = relationship(back_populates="mention")

    def __repr__(self) -> str:
        return f"<Mention {self.id} {self.source}:{self.external_id}>"


class Topic(Base):
    __tablename__ = "topics"
    __table_args__ = (Index("ix_topics_brand", "brand_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    brand_id: Mapped[int] = mapped_column(ForeignKey("brands.id", ondelete="CASCADE"))
    label: Mapped[str] = mapped_column(String(255))
    keywords: Mapped[Optional[list[str]]] = mapped_column(JSON)
    first_seen: Mapped[Optional[datetime]] = mapped_column(DateTime)
    last_seen: Mapped[Optional[datetime]] = mapped_column(DateTime)

    brand: Mapped[Brand] = relationship(back_populates="topics")
    daily: Mapped[list[TopicDaily]] = relationship(back_populates="topic")

    def __repr__(self) -> str:
        return f"<Topic {self.id} {self.label!r}>"


class Analysis(Base):
    __tablename__ = "analysis"
    __table_args__ = (
        Index("ix_analysis_topic", "topic_id"),
        Index("ix_analysis_category", "category"),
    )

    mention_id: Mapped[int] = mapped_column(
        ForeignKey("mentions.id", ondelete="CASCADE"), primary_key=True
    )
    sentiment: Mapped[Optional[str]] = mapped_column(String(16))
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float)
    emotion: Mapped[Optional[str]] = mapped_column(String(16))
    emotion_scores: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)
    category: Mapped[Optional[str]] = mapped_column(String(32))
    feature: Mapped[Optional[str]] = mapped_column(String(64))
    topic_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("topics.id", ondelete="SET NULL")
    )
    model_version: Mapped[Optional[str]] = mapped_column(String(64))

    mention: Mapped[Mention] = relationship(back_populates="analysis")
    topic: Mapped[Optional[Topic]] = relationship()


class TopicDaily(Base):
    __tablename__ = "topic_daily"

    topic_id: Mapped[int] = mapped_column(
        ForeignKey("topics.id", ondelete="CASCADE"), primary_key=True
    )
    day: Mapped[date] = mapped_column(Date, primary_key=True)
    mentions: Mapped[int] = mapped_column(Integer, default=0)
    avg_sentiment: Mapped[Optional[float]] = mapped_column(Float)

    topic: Mapped[Topic] = relationship(back_populates="daily")


class Alert(Base):
    __tablename__ = "alerts"
    __table_args__ = (Index("ix_alerts_brand_seen", "brand_id", "seen"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    brand_id: Mapped[int] = mapped_column(ForeignKey("brands.id", ondelete="CASCADE"))
    topic_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("topics.id", ondelete="SET NULL")
    )
    kind: Mapped[str] = mapped_column(String(32))
    message: Mapped[str] = mapped_column(String(1024))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    seen: Mapped[bool] = mapped_column(Boolean, default=False)


class FetchRun(Base):
    __tablename__ = "fetch_runs"
    __table_args__ = (Index("ix_fetch_runs_source_started", "source", "started_at"),)

    id: Mapped[int] = mapped_column(BigId, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(32))
    brand_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("brands.id", ondelete="SET NULL")
    )
    started_at: Mapped[datetime] = mapped_column(DateTime)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    new_rows: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[Optional[str]] = mapped_column(Text)


TABLE_NAMES: tuple[str, ...] = tuple(Base.metadata.tables.keys())
