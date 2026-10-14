"""brand_requests: the life of a brand typed into the dashboard, and the few writes the dashboard may make.

The dashboard only ever inserts or updates rows here (Track, Approve, Edit, Cancel, Remove); the collector's
request job (tracking/worker.py) does everything else. Every status change is a conditional UPDATE
(`... WHERE status = <expected>`), so a request is claimed once and a Cancel during research wins.

    track:  queued -> researching -> review -> approved -> fetching -> analysing -> ready | not_enough_data
            review -> recheck -> researching -> review      (edited setup, dry-run again)
            queued / review / recheck -> cancelled;  any step -> failed;  ready -> removed (group deleted)
    remove: queued -> removing -> removed | failed
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from config import settings
from db.models import Brand, BrandRequest
from nlp.llm.research import key
from sources.base import utc_now

TRACK, REMOVE = "track", "remove"
QUEUED, RESEARCHING, REVIEW, RECHECK, APPROVED = "queued", "researching", "review", "recheck", "approved"
FETCHING, ANALYSING, READY, NOT_ENOUGH, FAILED = "fetching", "analysing", "ready", "not_enough_data", "failed"
CANCELLED, REMOVING, REMOVED = "cancelled", "removing", "removed"

IN_PROGRESS = (QUEUED, RESEARCHING, REVIEW, RECHECK, APPROVED, FETCHING, ANALYSING)
FINISHED = (READY, NOT_ENOUGH, FAILED, CANCELLED, REMOVED)
LABELS = {
    QUEUED: "Queued", RESEARCHING: "Researching", REVIEW: "Waiting for your review", RECHECK: "Re-checking",
    APPROVED: "Approved", FETCHING: "Fetching", ANALYSING: "Analysing", READY: "Ready",
    NOT_ENOUGH: "Not enough data yet", FAILED: "Failed", CANCELLED: "Cancelled", REMOVING: "Removing",
    REMOVED: "Removed",
}
NAME_RE = re.compile(r"[\w][\w &'’.+\-!]{0,58}[\w.!'’+]?", re.UNICODE)


class RequestError(ValueError):
    """Shown to the person as is."""


def validate_name(session: Session, raw: str) -> str:
    """The cleaned brand name, or RequestError with a message for the dashboard."""
    name = " ".join((raw or "").split())
    if len(name) < 2:
        raise RequestError("Type a brand name (at least 2 characters).")
    if len(name) > 60:
        raise RequestError("That name is too long (60 characters at most).")
    if not NAME_RE.fullmatch(name) or not key(name):
        raise RequestError("Use letters, digits, spaces and & ' . - + ! only.")
    k = key(name)
    for (existing,) in session.execute(select(Brand.name)):
        if key(existing) == k:
            raise RequestError(f"{existing} is already tracked.")
    for other in session.scalars(select(BrandRequest).where(BrandRequest.kind == TRACK,
                                                            BrandRequest.status.in_(IN_PROGRESS))):
        if key(other.name) == k:
            raise RequestError(f"{other.name} is already being set up.")
    active = session.scalar(select(func.count()).select_from(BrandRequest).where(
        BrandRequest.kind == TRACK, BrandRequest.status.in_(IN_PROGRESS))) or 0
    if active >= settings.track_max_active:
        raise RequestError(f"{active} brands are being set up already (at most {settings.track_max_active} at once). "
                           "Approve, cancel or wait for one first.")
    return name


def create_request(session: Session, raw: str) -> BrandRequest:
    if settings.demo:
        raise RequestError("Tracking new brands works on real data only (the collector does not run in demo mode).")
    name = validate_name(session, raw)
    now = utc_now()
    request = BrandRequest(kind=TRACK, name=name, status=QUEUED, detail="Waiting for the collector",
                           created_at=now, updated_at=now)
    session.add(request)
    session.commit()
    return request


def move(session: Session, request_id: int, expected: tuple[str, ...] | str, status: str, **values: Any) -> bool:
    """Conditional status change; False if someone else changed the request first."""
    expected = (expected,) if isinstance(expected, str) else expected
    values = {"status": status, "updated_at": utc_now(), **values}
    if status in FINISHED:
        values.setdefault("finished_at", utc_now())
    done = session.execute(update(BrandRequest).where(BrandRequest.id == request_id,
                                                      BrandRequest.status.in_(expected)).values(**values))
    session.commit()
    return done.rowcount == 1


def note(session: Session, request_id: int, detail: str) -> None:
    """Progress line for the dashboard (does not change the status)."""
    session.execute(update(BrandRequest).where(BrandRequest.id == request_id)
                    .values(detail=detail[:255], updated_at=utc_now()))
    session.commit()


def approve(session: Session, request_id: int, setup: dict[str, Any] | None = None) -> bool:
    values: dict[str, Any] = {"detail": "Approved; the collector starts within a minute", "error": None}
    if setup is not None:
        values["setup"] = setup
    return move(session, request_id, REVIEW, APPROVED, **values)


def recheck(session: Session, request_id: int, setup: dict[str, Any]) -> bool:
    return move(session, request_id, REVIEW, RECHECK, setup=setup, detail="Re-checking your edits", error=None)


def cancel(session: Session, request_id: int) -> bool:
    return move(session, request_id, (QUEUED, REVIEW, RECHECK, RESEARCHING), CANCELLED, detail=None)


def recent(session: Session, days: int = 7, limit: int = 12) -> list[BrandRequest]:
    """What the card lists: everything in progress plus what finished in the last `days`."""
    since = utc_now() - timedelta(days=days)
    rows = session.scalars(select(BrandRequest).where(
        (BrandRequest.status.in_(IN_PROGRESS + (REMOVING,))) | (BrandRequest.updated_at >= since)
    ).order_by(BrandRequest.id.desc()).limit(limit))
    return list(rows)


def removable_groups(session: Session) -> list[str]:
    """Groups added from the dashboard that still exist. Groups set up in brands.yaml by hand (Note apps,
    Skincare, Stationery) can only be removed there, so one click can't delete months of data."""
    added = {r.group_name for r in session.scalars(select(BrandRequest).where(
        BrandRequest.kind == TRACK, BrandRequest.applied_at.is_not(None), BrandRequest.status != REMOVED))
        if r.group_name}
    existing = {g for (g,) in session.execute(select(Brand.brand_group).distinct())}
    pending = {r.group_name for r in session.scalars(select(BrandRequest).where(
        BrandRequest.kind == REMOVE, BrandRequest.status.in_((QUEUED, REMOVING))))}
    return sorted((added & existing) - pending)


def request_removal(session: Session, group: str, confirmation: str) -> BrandRequest:
    if confirmation.strip() != group:
        raise RequestError(f"Type the group name exactly ({group}) to confirm.")
    if group not in removable_groups(session):
        raise RequestError(f"{group} can't be removed here (only groups added from the dashboard).")
    now = utc_now()
    request = BrandRequest(kind=REMOVE, name=group, group_name=group, status=QUEUED,
                           detail="Waiting for the collector", created_at=now, updated_at=now)
    session.add(request)
    session.commit()
    return request


def age(moment: datetime | None, now: datetime | None = None) -> str:
    if moment is None:
        return ""
    seconds = max(0, int(((now or utc_now()) - moment).total_seconds()))
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60} min ago"
    if seconds < 86400:
        return f"{seconds // 3600} h ago"
    return f"{seconds // 86400} d ago"
