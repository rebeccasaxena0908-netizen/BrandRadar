"""Scheduled collector: fetch -> NLP -> topics -> priority, every COLLECTOR_INTERVAL_HOURS.

    python collector.py           # run a cycle now, then every N hours, until Ctrl+C
    python collector.py --once    # one cycle, then exit

Memory (8 GB Mac): the NLP and topic stages run in a short-lived child process, so the models (and
the MPS cache) are only loaded while there is work and are fully returned to the OS afterwards; the
idle collector holds no models. NLP is skipped entirely when no mentions are waiting.
Only one cycle runs at a time: APScheduler max_instances=1 plus a lock file across processes.
"""

from __future__ import annotations

import argparse
import gc
import logging
import multiprocessing
import os
import queue
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session, sessionmaker

from config import PROJECT_ROOT, settings
from db.models import Analysis, Brand, CollectorRun, FetchRun, Mention
from sources.base import BrandSpec, brand_sources, utc_now

log = logging.getLogger("collector")

LOCK_FILE = PROJECT_ROOT / ".collector.lock"
# Set by Ctrl+C. A running cycle checks it between stages and while waiting for a child process,
# stops the child and records the cycle as interrupted, so the collector exits promptly.
STOP = threading.Event()
FETCH_OVERLAP = timedelta(hours=24)  # re-read a day of overlap: dedupe makes it free, late reviews get caught
MAX_ERROR_LENGTH = 10_000


# ---------------------------------------------------------------- lock


class AlreadyRunning(RuntimeError):
    pass


def _try_lock(handle: Any) -> None:
    """Exclusive, non-blocking lock on an open file; raises BlockingIOError if already held.
    flock on macOS/Linux, msvcrt on Windows (fcntl does not exist there). Both are released by the OS
    when the process exits, even if it crashes."""
    if sys.platform == "win32":
        import msvcrt

        # Windows locks are mandatory: lock a byte well past the pid text so others can still read it
        handle.seek(1024)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise BlockingIOError(str(exc)) from exc
    else:
        import fcntl

        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)


@contextmanager
def collector_lock(path: Path | None = None) -> Iterator[None]:
    """Exclusive, non-blocking OS lock across processes; released automatically if the process dies."""
    # read/write without append mode (appends ignore seek), creating the file if needed
    handle = os.fdopen(os.open(path or LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o644), "r+")
    try:
        try:
            _try_lock(handle)
        except BlockingIOError:
            handle.seek(0)
            pid = handle.read().strip() or "?"
            raise AlreadyRunning(f"another collector is already running (pid {pid})") from None
        handle.seek(0)
        handle.write(str(os.getpid()).ljust(20))  # fixed width, no truncate (not allowed past a Windows lock)
        handle.flush()
        yield
    finally:
        handle.close()  # closing releases the lock


# ---------------------------------------------------------------- stages


@dataclass
class FetchOutcome:
    new_mentions: int
    runs: int
    errors: list[str] = field(default_factory=list)
    warnings: int = 0


@dataclass
class NlpOutcome:
    analysed: int
    failed: int
    empty: int = 0


@dataclass
class Stages:
    """Injectable so tests can mock every stage."""

    fetch: Callable[[datetime], FetchOutcome]
    pending: Callable[[], int]
    nlp: Callable[[], NlpOutcome]
    topics: Callable[[], int]
    priority: Callable[[], int]


def fetch_since(factory: sessionmaker[Session], now: datetime, pairs: list[tuple[str, int]] | None = None) -> datetime:
    """Earliest 'last successful fetch' over every (source, brand) pair, minus FETCH_OVERLAP.
    A pair that never succeeded falls back to FETCH_SINCE_DAYS. Warnings count as success.
    `pairs` defaults to every enabled source x every brand."""
    fallback = now - timedelta(days=settings.fetch_since_days)
    with factory() as session:
        ok = FetchRun.error.is_(None) | ~FetchRun.error.like("error:%")
        rows = session.execute(
            select(FetchRun.source, FetchRun.brand_id, func.max(FetchRun.started_at))
            .where(FetchRun.finished_at.is_not(None), ok).group_by(FetchRun.source, FetchRun.brand_id)
        ).all()
        if pairs is None:
            from fetch import SOURCE_FACTORIES

            enabled = [name for name, make in SOURCE_FACTORIES.items() if make().enabled]
            specs = [BrandSpec.from_model(b) for b in session.scalars(select(Brand))]
            pairs = [(s, b.id) for b in specs for s in brand_sources(b) if s in enabled]
    last = {(s, b): t for s, b, t in rows}
    starts = [last.get(pair) or fallback for pair in pairs]
    return max(min(starts) - FETCH_OVERLAP, fallback) if starts else fallback


def real_stages(factory: sessionmaker[Session]) -> Stages:
    def fetch(since: datetime) -> FetchOutcome:
        from fetch import SOURCE_FACTORIES, load_brands, purge_expired_data, run_fetch

        purge_expired_data(factory)
        sources = [f() for f in SOURCE_FACTORIES.values()]
        results = run_fetch(factory, load_brands(factory), sources, since)
        errors = [f"{r.source}/{r.brand}: {r.error}" for r in results if r.error]
        return FetchOutcome(sum(r.new_rows for r in results), sum(not r.skipped for r in results), errors,
                            sum(r.warnings for r in results))

    def pending() -> int:
        with factory() as session:
            return int(session.scalar(
                select(func.count()).select_from(Mention)
                .outerjoin(Analysis, Analysis.mention_id == Mention.id).where(Analysis.mention_id.is_(None))
            ) or 0)

    def priority() -> int:
        from nlp.priority import run_priority

        _, alerts = run_priority(factory)
        return len(alerts)

    return Stages(fetch=fetch, pending=pending, nlp=lambda: run_in_child(_child_nlp),
                  topics=lambda: run_in_child(_child_topics), priority=priority)


# --- heavy stages: these run in a spawned child process (see module docstring)


def _child_logging() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")


def _child_nlp() -> NlpOutcome:
    _child_logging()
    from db.session import get_session_factory
    from nlp.models import HFAnalyzer
    from nlp.pipeline import run_pipeline

    stats = run_pipeline(get_session_factory(), HFAnalyzer(), progress=False)
    return NlpOutcome(analysed=stats.analysed, failed=stats.failed, empty=stats.empty)


def _child_topics() -> int:
    _child_logging()
    from db.session import get_session_factory
    from nlp.topics import MiniLMEmbedder, run_topics

    return sum(r.clusters for r in run_topics(get_session_factory(), MiniLMEmbedder()) if not r.skipped)


class Interrupted(RuntimeError):
    pass


def _child_entry(func: Callable[[], Any], results: Any) -> None:
    try:
        results.put((True, func()))
    except BaseException as exc:  # noqa: BLE001 — report everything (incl. Ctrl+C) to the parent
        results.put((False, f"{type(exc).__name__}: {exc}"))


def run_in_child(func: Callable[[], Any], poll: float = 0.5) -> Any:
    """Run func in a fresh process and return its (picklable) result. When the child exits, every
    byte the models used goes back to the OS — freeing them inside a long-lived process does not.
    A plain Process (not a Pool): a child that dies is detected instead of waited on forever, and
    STOP terminates it."""
    ctx = multiprocessing.get_context("spawn")
    results = ctx.Queue()
    child = ctx.Process(target=_child_entry, args=(func, results), daemon=True, name=f"collector-{func.__name__}")
    child.start()
    try:
        while True:
            try:
                ok, payload = results.get(timeout=poll)
                break
            except queue.Empty:
                if STOP.is_set():
                    child.terminate()
                    raise Interrupted("stopped while the stage was running") from None
                if not child.is_alive():
                    try:  # it may have put its result just before exiting
                        ok, payload = results.get(timeout=poll)
                        break
                    except queue.Empty:
                        raise RuntimeError(f"child process exited with code {child.exitcode} without a result") from None
    finally:
        child.join(timeout=10)
        if child.is_alive():
            child.kill()
            child.join()
    if not ok:
        if payload.startswith("KeyboardInterrupt"):  # Ctrl+C reached the child directly
            raise Interrupted("stopped while the stage was running")
        raise RuntimeError(f"in child process: {payload}")
    return payload


# ---------------------------------------------------------------- cycle


@dataclass
class CycleResult:
    run_id: int
    status: str
    new_mentions: int = 0
    analysed: int = 0
    analysis_failed: int = 0
    topics: int = 0
    alerts_created: int = 0
    stage_seconds: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    seconds: float = 0.0


def mark_interrupted(factory: sessionmaker[Session]) -> int:
    """Cycles left 'running' by a killed collector (the lock guarantees none is really running)."""
    with factory() as session:
        n = session.execute(
            update(CollectorRun).where(CollectorRun.status == "running")
            .values(status="failed", finished_at=utc_now(), error="interrupted (collector stopped mid-cycle)")
        ).rowcount
        session.commit()
    return n or 0


def _finish(factory: sessionmaker[Session], result: CycleResult, status: str, extra_error: str | None, started: float) -> None:
    """Write the cycle's outcome to its collector_runs row."""
    result.status = status
    if extra_error:
        result.errors.append(extra_error)
    result.seconds = round(time.perf_counter() - started, 1)
    with factory() as session:
        run = session.get(CollectorRun, result.run_id)
        assert run is not None
        run.finished_at = utc_now()
        run.status = result.status
        run.new_mentions, run.analysed, run.analysis_failed = result.new_mentions, result.analysed, result.analysis_failed
        run.topics, run.alerts_created = result.topics, result.alerts_created
        run.stage_seconds = result.stage_seconds
        run.error = "\n".join(result.errors)[:MAX_ERROR_LENGTH] or None
        session.commit()


def run_cycle(factory: sessionmaker[Session], stages: Stages, now: datetime | None = None) -> CycleResult:
    """One collector cycle. A failing stage is logged and recorded; the later stages still run on
    whatever data exists (e.g. topics and priority after a failed NLP run)."""
    started = time.perf_counter()
    now = now or utc_now()
    with factory() as session:
        run = CollectorRun(started_at=now, status="running")
        session.add(run)
        session.commit()
        result = CycleResult(run_id=run.id, status="running")
    failed_stages: list[str] = []

    def stage(name: str, call: Callable[[], str]) -> None:
        t = time.perf_counter()
        try:
            summary = call()
            log.info("%-8s ok      %s  (%.0fs)", name, summary, time.perf_counter() - t)
        except Interrupted:
            log.info("%-8s stopped (Ctrl+C)", name)
        except Exception as exc:  # noqa: BLE001 — record it and carry on with the next stage
            failed_stages.append(name)
            result.errors.append(f"{name}: {type(exc).__name__}: {exc}")
            log.error("%-8s FAILED  %s: %s  (%.0fs)", name, type(exc).__name__, exc, time.perf_counter() - t)
        result.stage_seconds[name] = round(time.perf_counter() - t, 1)

    def do_fetch() -> str:
        since = fetch_since(factory, now)
        out = stages.fetch(since)
        result.new_mentions = out.new_mentions
        result.errors += [f"fetch: {e}" for e in out.errors]
        extra = f", {len(out.errors)} source error(s)" if out.errors else ""
        extra += f", {out.warnings} warning(s)" if out.warnings else ""
        return f"+{out.new_mentions} new mentions since {since:%d %b %H:%M} UTC ({out.runs} source runs{extra})"

    def do_nlp() -> str:
        waiting = stages.pending()
        if not waiting:
            return "nothing new to analyse (models not loaded)"
        out = stages.nlp()
        result.analysed, result.analysis_failed = out.analysed, out.failed
        if out.failed:
            result.errors.append(f"nlp: {out.failed} mention(s) failed analysis (retried next cycle)")
        return f"{out.analysed} analysed" + (f", {out.failed} failed" if out.failed else "") + f" of {waiting} waiting"

    def do_topics() -> str:
        result.topics = stages.topics()
        return f"{result.topics} topics"

    def do_priority() -> str:
        result.alerts_created = stages.priority()
        return f"{result.alerts_created} new alert(s)"

    interrupted = False
    try:
        for name, call in (("fetch", do_fetch), ("nlp", do_nlp), ("topics", do_topics), ("priority", do_priority)):
            if STOP.is_set():
                interrupted = True
                break
            stage(name, call)
            gc.collect()
    except KeyboardInterrupt:  # --once runs in the main thread, where Ctrl+C lands
        interrupted = True
        _finish(factory, result, "failed", "interrupted (Ctrl+C)", started)
        raise
    if interrupted or STOP.is_set():
        result.status = "failed"
        result.errors.append("interrupted (collector stopped mid-cycle)")
        _finish(factory, result, "failed", None, started)
        log.info("cycle    INTERRUPTED after %.0fs", result.seconds)
        return result

    if len(failed_stages) == 4:
        result.status = "failed"
    elif result.errors:
        result.status = "partial"
    else:
        result.status = "ok"
    _finish(factory, result, result.status, None, started)
    log.info("cycle    %s  %d new · %d analysed · %d topics · %d alerts · %d error(s) · %.0fs",
             result.status.upper(), result.new_mentions, result.analysed, result.topics,
             result.alerts_created, len(result.errors), result.seconds)
    return result


# ---------------------------------------------------------------- CLI


def _raise_keyboard_interrupt(signum: int, frame: Any) -> None:
    raise KeyboardInterrupt


def install_signal_handlers() -> None:
    """Ctrl+C and SIGTERM both stop the collector gracefully. A script that starts us in the
    background (scripts/start.sh) makes the shell *ignore* SIGINT for us, and Python keeps it
    ignored, so the normal handler is restored explicitly."""
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--once", action="store_true", help="run one cycle and exit")
    parser.add_argument("--interval-hours", type=float, default=settings.collector_interval_hours,
                        help=f"hours between cycles (default {settings.collector_interval_hours:g}, COLLECTOR_INTERVAL_HOURS)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("fetch", "sources", "nlp", "apscheduler", "urllib3", "httpx", "BERTopic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)  # the collector prints one line per stage

    install_signal_handlers()
    if settings.demo:
        log.error("BRAND_RADAR_DEMO is set: the collector only writes to the real database. Unset it and retry.")
        return 2

    try:
        with collector_lock():
            from db.session import get_session_factory

            factory = get_session_factory()
            if (n := mark_interrupted(factory)):
                log.info("marked %d interrupted cycle(s) as failed", n)
            stages = real_stages(factory)
            if args.once:
                return 0 if run_cycle(factory, stages).status != "failed" else 1
            return run_forever(factory, stages, args.interval_hours)
    except AlreadyRunning as exc:
        log.info("%s; exiting.", exc)
        return 0
    except KeyboardInterrupt:
        log.info("stopped.")
        return 0


def run_forever(factory: sessionmaker[Session], stages: Stages, interval_hours: float) -> int:
    from apscheduler.schedulers.blocking import BlockingScheduler

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        run_cycle, "interval", hours=interval_hours, args=(factory, stages), id="cycle",
        next_run_time=datetime.now().astimezone(),  # first cycle right away
        max_instances=1, coalesce=True, misfire_grace_time=3600,
    )
    log.info("collector started: a cycle every %g h; Ctrl+C to stop", interval_hours)
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        # ignore repeats (start.sh sends SIGTERM right after the terminal's Ctrl+C) so the running
        # cycle can record itself as interrupted before we exit
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        STOP.set()  # a running cycle stops its child process and returns within a second or so
        log.info("stopping…")
        scheduler.shutdown(wait=True)
        log.info("stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
