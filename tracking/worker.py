"""The collector's request job (Step 11): every TRACK_POLL_SECONDS it does the next piece of work on the
brands typed into the dashboard, one heavy step at a time.

    removals first   delete a dashboard-added group: brands.yaml blocks, then its rows
    approved         save a drafted profile, append the group to brands.yaml, sync, then for the new group
                     only: fetch -> NLP (child process) -> translate -> topics (child) -> priority -> labels
                     and briefings -> Ready / Not enough data yet
    recheck          an edited setup: research the brands the person added, dry-run again -> review
    queued           research (LLM or rules) + app lookup + dry-run preview -> review

Research, the dry run and the pipeline all hold the collector's heavy lock, which the 6-hour cycle holds
too, so Ollama, the NLP models and BERTopic never run at the same time on an 8 GB Mac. When the lock is
busy the request waits (its detail line says so) and the next tick tries again.
"""

from __future__ import annotations

import functools
import logging
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session, sessionmaker

from config import settings
from db.models import (
    Alert, Analysis, Brand, BrandRequest, Briefing, FetchRun, Mention, Topic, TopicDaily, YoutubeVideo,
)
from manage_brands import DEFAULT_FILE, append_entries, load_yaml, remove_group_entries, sync_brands
from nlp.llm import provider as llm
from sources.base import brand_sources, utc_now
from tracking import setup as setup_mod
from tracking import store
from tracking.store import (
    ANALYSING, APPROVED, FAILED, FETCHING, NOT_ENOUGH, QUEUED, READY, RECHECK, REMOVE, REMOVED, REMOVING,
    RESEARCHING, REVIEW, TRACK,
)

log = logging.getLogger("tracking")

Runner = Callable[[Callable[[], Any]], Any]


# ---------------------------------------------------------------- child processes (the heavy models)


def _child_logging() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")


def child_nlp(names: list[str]) -> int:
    _child_logging()
    from db.session import get_session_factory
    from nlp.models import HFAnalyzer
    from nlp.pipeline import run_pipeline

    return run_pipeline(get_session_factory(), HFAnalyzer(), brands=names, progress=False).analysed


def child_topics(names: list[str]) -> list[tuple[str, int, str | None]]:
    _child_logging()
    from db.session import get_session_factory
    from nlp.topics import MiniLMEmbedder, run_topics

    return [(r.brand, r.clusters, r.skipped) for r in run_topics(get_session_factory(), MiniLMEmbedder(), brands=names)]


def run_inline(func: Callable[[], Any]) -> Any:
    return func()


# ---------------------------------------------------------------- the job


@dataclass
class Worker:
    factory: sessionmaker[Session]
    heavy: threading.Lock = field(default_factory=threading.Lock)
    brands_file: Path = DEFAULT_FILE
    runner: Runner = run_inline  # the collector passes run_in_child
    stop: threading.Event = field(default_factory=threading.Event)
    make_source: Callable[[str], Any] | None = None
    get_provider: Callable[[], llm.Provider] | None = None
    app_store: Callable[[str], Any] | None = None
    play_store: Callable[[str], Any] | None = None
    profiles_file: Path | None = None  # where a drafted profile is saved (default profiles.yaml)
    # the model stages for the new group; default: child processes (tests pass in-process fakes)
    nlp: Callable[[list[str]], Any] | None = None
    topics: Callable[[list[str]], Any] | None = None

    def run_nlp(self, names: list[str]) -> Any:
        return self.nlp(names) if self.nlp is not None else self.runner(functools.partial(child_nlp, names))

    def run_topics(self, names: list[str]) -> Any:
        return self.topics(names) if self.topics is not None else self.runner(functools.partial(child_topics, names))

    def provider(self) -> llm.Provider:
        return (self.get_provider or llm.get_provider)()

    def source(self, name: str) -> Any:
        if self.make_source is not None:
            return self.make_source(name)
        from fetch import SOURCE_FACTORIES

        return SOURCE_FACTORIES[name]()

    def apps(self) -> dict[str, Callable[[str], Any]]:
        out = {}
        if self.app_store is not None:
            out["app_store"] = self.app_store
        if self.play_store is not None:
            out["play_store"] = self.play_store
        return out

    # -- scheduling

    def next_request(self) -> tuple[str, int] | None:
        """(step, request id) of the next thing to do: removals, then approved, re-checks, new requests."""
        with self.factory() as session:
            for kind, status, step in ((REMOVE, QUEUED, "remove"), (TRACK, APPROVED, "apply"),
                                       (TRACK, RECHECK, "recheck"), (TRACK, QUEUED, "research")):
                rid = session.scalar(select(BrandRequest.id).where(BrandRequest.kind == kind, BrandRequest.status == status)
                                     .order_by(BrandRequest.id).limit(1))
                if rid is not None:
                    return step, rid
        return None

    def tick(self) -> str | None:
        """Do one step of work if there is any (and the heavy lock is free). None = nothing done."""
        nxt = self.next_request()
        if nxt is None:
            return None
        step, rid = nxt
        if not self.heavy.acquire(blocking=False):
            with self.factory() as session:
                store.note(session, rid, "Waiting for the collector's 6-hourly cycle to finish")
            return None
        try:
            return getattr(self, step)(rid)
        finally:
            self.heavy.release()

    def run_until_idle(self, max_steps: int = 50) -> list[str]:
        done = []
        for _ in range(max_steps):
            if self.stop.is_set():
                break
            summary = self.tick()
            if summary is None:
                break
            done.append(summary)
        return done

    # -- helpers

    def note(self, rid: int, detail: str) -> None:
        with self.factory() as session:
            store.note(session, rid, detail)

    def move(self, rid: int, expected: tuple[str, ...] | str, status: str, **values: Any) -> bool:
        with self.factory() as session:
            return store.move(session, rid, expected, status, **values)

    def request(self, rid: int) -> BrandRequest:
        with self.factory() as session:
            row = session.get(BrandRequest, rid)
            assert row is not None
            session.expunge(row)
            return row

    def existing(self) -> tuple[list[str], list[str]]:
        """Brand names and groups that exist (brands.yaml and the database)."""
        entries = load_yaml(self.brands_file)
        names = {e.name for e in entries}
        groups = {e.group for e in entries}
        with self.factory() as session:
            for name, group in session.execute(select(Brand.name, Brand.brand_group)):
                names.add(name)
                groups.add(group)
        return sorted(names), sorted(groups)

    def fail(self, rid: int, status: str, exc: BaseException) -> str:
        log.error("request #%d failed: %s", rid, "".join(traceback.format_exception_only(type(exc), exc)).strip())
        message = f"{type(exc).__name__}: {exc}"[:1000]
        self.move(rid, status, FAILED, error=message, detail="Failed")
        return f"#{rid} failed: {message}"

    # -- research (queued) and re-check

    def research(self, rid: int) -> str:
        with self.factory() as session:
            if not store.move(session, rid, QUEUED, RESEARCHING, detail="Researching the market and competitors"):
                return f"#{rid} was changed meanwhile"
        req = self.request(rid)
        try:
            names, groups = self.existing()
            started = time.monotonic()
            setup = setup_mod.propose(req.name, self.provider(), names, groups, factory=self.factory, **self.apps())
            research_s = time.monotonic() - started
            preview = self.dry_run(rid, setup)
            preview["research_seconds"] = round(research_s, 1)
        except Exception as exc:  # noqa: BLE001 — the request fails, the collector carries on
            return self.fail(rid, RESEARCHING, exc)
        waiting = "Waiting for your review" + (": pick the industry and competitors" if setup["needs_input"] else "")
        if not self.move(rid, RESEARCHING, REVIEW, setup=setup, preview=preview, group_name=setup["group"], detail=waiting):
            return f"#{rid} was cancelled during research"
        return f"#{rid} {req.name}: proposed {setup['group']} ({len(setup['brands'])} brands) in {research_s + preview['seconds']:.0f}s"

    def recheck(self, rid: int) -> str:
        with self.factory() as session:
            if not store.move(session, rid, RECHECK, RESEARCHING, detail="Re-checking your edits"):
                return f"#{rid} was changed meanwhile"
        req = self.request(rid)
        try:
            setup = dict(req.setup or {})
            setup_mod.fill_in(setup, self.provider(), **self.apps())
            preview = self.dry_run(rid, setup)
        except Exception as exc:  # noqa: BLE001
            return self.fail(rid, RESEARCHING, exc)
        waiting = "Waiting for your review" + (": some details are still missing" if setup["needs_input"] else "")
        if not self.move(rid, RESEARCHING, REVIEW, setup=setup, preview=preview, group_name=setup.get("group"), detail=waiting):
            return f"#{rid} was cancelled during the re-check"
        return f"#{rid} {req.name}: re-checked"

    def dry_run(self, rid: int, setup: dict[str, Any]) -> dict[str, Any]:
        since = utc_now() - timedelta(days=settings.fetch_since_days)
        # a preview is only a sample: half the fetch budget per source keeps the review wait short
        return setup_mod.preview(setup, self.source, since, min(settings.track_source_seconds, 45.0),
                                 note=lambda text: self.note(rid, text))

    # -- approved: write the group, then fetch and analyse it

    def apply(self, rid: int) -> str:
        req = self.request(rid)
        setup = dict(req.setup or {})
        names, groups = self.existing()
        problems = setup_mod.problems(setup, names, groups)
        if problems:
            self.move(rid, APPROVED, REVIEW, error=" ".join(problems)[:1000], detail="Needs a change before it can start")
            return f"#{rid} {req.name}: sent back to review ({problems[0]})"
        if not self.move(rid, APPROVED, FETCHING, detail="Adding the group to brands.yaml"):
            return f"#{rid} was changed meanwhile"
        timings: dict[str, float] = {}
        status = FETCHING
        try:
            draft = setup_mod.draft_of(setup)
            if draft is not None:
                from nlp.llm.profile_draft import save_profile

                save_profile(draft, *([self.profiles_file] if self.profiles_file else []))
            entries = setup_mod.entries(setup)
            append_entries(self.brands_file, entries,
                           comment=f"{setup['group']}: added from the dashboard on {utc_now():%Y-%m-%d} (request #{rid})")
            with self.factory() as session:
                sync_brands(session, load_yaml(self.brands_file))
                session.commit()
            self.move(rid, FETCHING, FETCHING, applied_at=utc_now(), group_name=setup["group"])
            brand_names = [e.name for e in entries]

            started = time.monotonic()
            fetched = self.fetch_group(rid, brand_names)
            timings["fetch"] = round(time.monotonic() - started, 1)
            if self.stop.is_set():
                raise InterruptedError("collector stopped; the next cycle continues collecting")

            status = ANALYSING
            self.move(rid, FETCHING, ANALYSING, detail="Analysing sentiment, emotion and categories")
            timings.update(self.analyse_group(rid, brand_names))
        except Exception as exc:  # noqa: BLE001
            return self.fail(rid, status, exc)
        result = self.summarise(brand_names, setup, fetched, timings)
        own = next(e.name for e in entries if e.is_own)
        own_count = result["analysed"].get(own, 0)
        if own_count >= settings.track_min_mentions:
            final, detail = READY, f"Ready: {own_count} analysed mentions for {own}, {len(result['topics'])} complaint topics"
        else:
            final, detail = NOT_ENOUGH, (f"Not enough data yet: {own_count} analysed mentions for {own} "
                                         f"(at least {settings.track_min_mentions} needed); the collector keeps looking")
        self.move(rid, ANALYSING, final, result=result, detail=detail)
        return f"#{rid} {req.name}: {detail} ({sum(timings.values()):.0f}s)"

    def fetch_group(self, rid: int, names: list[str]) -> list[dict[str, Any]]:
        """Every source for every brand of the new group, YouTube with only its first query (the rest join in
        the 6-hour cycles), each source capped at TRACK_SOURCE_SECONDS. Returns one line per source run."""
        from fetch import SOURCE_FACTORIES, load_brands, run_one

        specs = [replace(s, youtube_queries=s.youtube_queries[:1]) for s in load_brands(self.factory, names)]
        since = utc_now() - timedelta(days=settings.fetch_since_days)
        jobs = [(source, spec) for source in SOURCE_FACTORIES for spec in specs if source in brand_sources(spec)]
        runs = []
        for i, (source_name, spec) in enumerate(jobs, 1):
            if self.stop.is_set():
                break
            source = self.source(source_name)
            if not source.enabled:  # optional sources (Reddit, YouTube without a key): skipped, not an error
                runs.append({"brand": spec.name, "source": source_name, "new": 0, "error": None, "skipped": True})
                continue
            self.note(rid, f"Fetching {setup_mod.data_name(source_name)} · {spec.name} ({i}/{len(jobs)})")
            r = run_one(self.factory, source, spec, since, seconds=settings.track_source_seconds)
            runs.append({"brand": spec.name, "source": source_name, "new": r.new_rows,
                         "error": (r.error or "")[:200] or None, "warnings": r.warnings})
        return runs

    def analyse_group(self, rid: int, names: list[str]) -> dict[str, float]:
        timings: dict[str, float] = {}
        provider = self.provider()
        with self.factory() as session:
            ids = list(session.scalars(select(Brand.id).where(Brand.name.in_(names))))

        t = time.monotonic()
        self.run_nlp(names)
        timings["nlp"] = round(time.monotonic() - t, 1)
        if provider.name != "none":
            from nlp.llm.translate import translate_pending

            self.note(rid, "Translating non-English mentions")
            t = time.monotonic()
            out = translate_pending(self.factory, provider, brands=ids)
            if out.total:
                self.note(rid, "Analysing the translations")
                self.run_nlp(names)
            timings["translate"] = round(time.monotonic() - t, 1)
        self.note(rid, "Clustering complaints into topics")
        t = time.monotonic()
        self.run_topics(names)
        from nlp.priority import run_priority

        run_priority(self.factory)
        timings["topics"] = round(time.monotonic() - t, 1)
        if provider.name != "none":
            from nlp.llm.briefing import refresh_briefings
            from nlp.llm.labels import refresh_labels
            from reports import quiet_streamlit

            quiet_streamlit()
            self.note(rid, "Writing topic labels and the briefing")
            t = time.monotonic()
            refresh_labels(self.factory, provider, brands=ids)
            refresh_briefings(self.factory, provider, brands=ids)
            timings["ai"] = round(time.monotonic() - t, 1)
        provider.unload()
        return timings

    def summarise(self, names: list[str], setup: dict[str, Any], fetched: list[dict[str, Any]],
                  timings: dict[str, float]) -> dict[str, Any]:
        with self.factory() as session:
            brands = {b.id: b.name for b in session.scalars(select(Brand).where(Brand.name.in_(names)))}
            mentions: dict[str, dict[str, int]] = {n: {} for n in names}
            for bid, source, n in session.execute(select(Mention.brand_id, Mention.source, func.count())
                                                  .where(Mention.brand_id.in_(list(brands))).group_by(Mention.brand_id, Mention.source)):
                mentions[brands[bid]][source] = int(n)
            analysed = {n: 0 for n in names}
            for bid, n in session.execute(select(Mention.brand_id, func.count()).join(Analysis, Analysis.mention_id == Mention.id)
                                          .where(Mention.brand_id.in_(list(brands)), Analysis.sentiment.is_not(None))
                                          .group_by(Mention.brand_id)):
                analysed[brands[bid]] = int(n)
            own = next(b["name"] for b in setup["brands"] if b["is_own"])
            own_id = next(i for i, n in brands.items() if n == own)
            members = func.count(Analysis.mention_id)
            topics = [{"label": label, "mentions": int(n), "quadrant": quad}
                      for label, n, quad in session.execute(
                          select(Topic.label, members, Topic.quadrant).join(Analysis, Analysis.topic_id == Topic.id)
                          .where(Topic.brand_id == own_id).group_by(Topic.id, Topic.label, Topic.quadrant)
                          .order_by(members.desc()).limit(5))]
        errors = [f"{r['brand']} · {setup_mod.data_name(r['source'])}: {r['error']}" for r in fetched if r.get("error")]
        return {"mentions": mentions, "analysed": analysed, "topics": topics, "seconds": timings,
                "source_errors": errors[:20]}

    # -- removal

    def remove(self, rid: int) -> str:
        req = self.request(rid)
        group = req.group_name or req.name
        with self.factory() as session:
            allowed = group in store.removable_groups(session) or self._only_pending(session, group, rid)
        if not allowed:
            self.move(rid, QUEUED, FAILED, error=f"{group} was not added from the dashboard", detail="Not removed")
            return f"#{rid}: {group} is not removable"
        if not self.move(rid, QUEUED, REMOVING, detail=f"Deleting {group}"):
            return f"#{rid} was changed meanwhile"
        try:
            removed = remove_group_entries(self.brands_file, group)
            with self.factory() as session:
                counts = delete_group_rows(session, group)
                session.execute(update(BrandRequest).where(BrandRequest.kind == TRACK, BrandRequest.group_name == group)
                                .values(status=REMOVED, detail=f"{group} was removed", updated_at=utc_now()))
                session.commit()
        except Exception as exc:  # noqa: BLE001
            return self.fail(rid, REMOVING, exc)
        detail = f"Deleted {group}: {counts['brands']} brands, {counts['mentions']} mentions"
        self.move(rid, REMOVING, REMOVED, detail=detail, result={"brands": removed, **counts})
        return f"#{rid} {detail}"

    @staticmethod
    def _only_pending(session: Session, group: str, rid: int) -> bool:
        """removable_groups hides groups with a pending removal, including this request's own."""
        added = session.scalar(select(func.count()).select_from(BrandRequest).where(
            BrandRequest.kind == TRACK, BrandRequest.group_name == group, BrandRequest.applied_at.is_not(None),
            BrandRequest.status != REMOVED)) or 0
        exists = session.scalar(select(func.count()).select_from(Brand).where(Brand.brand_group == group)) or 0
        return bool(added and exists)


def delete_group_rows(session: Session, group: str) -> dict[str, int]:
    """Everything stored for the group's brands, children first (no reliance on ON DELETE CASCADE)."""
    ids = list(session.scalars(select(Brand.id).where(Brand.brand_group == group)))
    if not ids:
        return {"brands": 0, "mentions": 0}
    mention_ids = select(Mention.id).where(Mention.brand_id.in_(ids)).scalar_subquery()
    topic_ids = select(Topic.id).where(Topic.brand_id.in_(ids)).scalar_subquery()
    mentions = session.scalar(select(func.count()).select_from(Mention).where(Mention.brand_id.in_(ids))) or 0
    session.execute(delete(Analysis).where(Analysis.mention_id.in_(mention_ids)))
    session.execute(delete(TopicDaily).where(TopicDaily.topic_id.in_(topic_ids)))
    for model in (Alert, Briefing, YoutubeVideo, FetchRun):
        session.execute(delete(model).where(model.brand_id.in_(ids)))
    session.execute(delete(Mention).where(Mention.brand_id.in_(ids)))
    session.execute(delete(Topic).where(Topic.brand_id.in_(ids)))
    session.execute(delete(Brand).where(Brand.id.in_(ids)))
    return {"brands": len(ids), "mentions": int(mentions)}
