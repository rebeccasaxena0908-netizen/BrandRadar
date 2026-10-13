"""Collector: stage order, skipping NLP, failure handling, cycle records, since logic, lock."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

import collector
from collector import (
    AlreadyRunning, FetchOutcome, NlpOutcome, Stages, collector_lock, fetch_since, mark_interrupted, run_cycle,
)
from db.models import CollectorRun, FetchRun
from sources.base import BrandSpec

NOW = datetime(2026, 10, 14, 12, 0)


class Recorder:
    """Mock stages that record call order; any stage can be told to fail."""

    def __init__(self, pending: int = 5, fail: tuple[str, ...] = (), nlp_failed: int = 0) -> None:
        self.calls: list[str] = []
        self.pending_count, self.fail, self.nlp_failed = pending, fail, nlp_failed
        self.since: datetime | None = None

    def _maybe_fail(self, name: str) -> None:
        self.calls.append(name)
        if name in self.fail:
            raise RuntimeError(f"{name} broke")

    def stages(self) -> Stages:
        def fetch(since: datetime) -> FetchOutcome:
            self.since = since
            self._maybe_fail("fetch")
            return FetchOutcome(new_mentions=7, runs=12)

        def pending() -> int:
            self.calls.append("pending")
            return self.pending_count

        def nlp() -> NlpOutcome:
            self._maybe_fail("nlp")
            return NlpOutcome(analysed=self.pending_count - self.nlp_failed, failed=self.nlp_failed)

        def topics() -> int:
            self._maybe_fail("topics")
            return 27

        def priority() -> int:
            self._maybe_fail("priority")
            return 1

        return Stages(fetch=fetch, pending=pending, nlp=nlp, topics=topics, priority=priority)


def runs(factory: sessionmaker[Session]) -> list[CollectorRun]:
    with factory() as s:
        return list(s.scalars(select(CollectorRun).order_by(CollectorRun.id)))


def test_stages_run_in_order_and_cycle_is_recorded(factory: sessionmaker[Session], brands: list[BrandSpec]) -> None:
    rec = Recorder()
    result = run_cycle(factory, rec.stages(), now=NOW)
    assert rec.calls == ["fetch", "pending", "nlp", "topics", "priority"]
    assert (result.status, result.new_mentions, result.analysed, result.topics, result.alerts_created) == ("ok", 7, 5, 27, 1)
    [run] = runs(factory)
    assert run.status == "ok" and run.finished_at is not None and run.error is None
    assert (run.new_mentions, run.analysed, run.topics, run.alerts_created) == (7, 5, 27, 1)
    assert set(run.stage_seconds) == {"fetch", "nlp", "topics", "priority"}


def test_nlp_skipped_when_nothing_is_pending(factory: sessionmaker[Session], brands: list[BrandSpec]) -> None:
    rec = Recorder(pending=0)
    result = run_cycle(factory, rec.stages(), now=NOW)
    assert "nlp" not in rec.calls  # models never loaded
    assert result.status == "ok" and result.analysed == 0


@pytest.mark.parametrize("broken", ["fetch", "nlp", "topics", "priority"])
def test_a_failing_stage_does_not_stop_the_others(
    factory: sessionmaker[Session], brands: list[BrandSpec], broken: str
) -> None:
    rec = Recorder(fail=(broken,))
    result = run_cycle(factory, rec.stages(), now=NOW)
    assert rec.calls == ["fetch", "pending", "nlp", "topics", "priority"]  # every stage still attempted
    assert result.status == "partial"
    assert len(result.errors) == 1 and result.errors[0].startswith(f"{broken}: RuntimeError: {broken} broke")
    assert runs(factory)[0].error == result.errors[0]


def test_all_stages_failing_marks_the_cycle_failed(factory: sessionmaker[Session], brands: list[BrandSpec]) -> None:
    result = run_cycle(factory, Recorder(fail=("fetch", "nlp", "topics", "priority")).stages(), now=NOW)
    assert result.status == "failed" and len(result.errors) == 4


def test_source_errors_and_failed_analysis_make_it_partial(factory: sessionmaker[Session], brands: list[BrandSpec]) -> None:
    rec = Recorder(nlp_failed=2)
    stages = rec.stages()
    stages.fetch = lambda since: FetchOutcome(new_mentions=3, runs=12, errors=["playstore/Notion: HTTPError"])
    result = run_cycle(factory, stages, now=NOW)
    assert result.status == "partial" and result.analysis_failed == 2
    assert any("playstore/Notion" in e for e in result.errors) and any("2 mention(s) failed" in e for e in result.errors)


def test_mark_interrupted(factory: sessionmaker[Session]) -> None:
    with factory() as s:
        s.add(CollectorRun(started_at=NOW, status="running"))
        s.commit()
    assert mark_interrupted(factory) == 1
    assert runs(factory)[0].status == "failed" and "interrupted" in runs(factory)[0].error


# ---------------------------------------------------------------- since


def add_run(factory: sessionmaker[Session], source: str, brand_id: int, started: datetime, error: str | None = None) -> None:
    with factory() as s:
        s.add(FetchRun(source=source, brand_id=brand_id, started_at=started, finished_at=started + timedelta(minutes=1),
                       new_rows=1, error=error))
        s.commit()


def test_since_is_earliest_last_success_minus_overlap(factory: sessionmaker[Session], brands: list[BrandSpec]) -> None:
    a, b = brands[0].id, brands[1].id
    add_run(factory, "hackernews", a, NOW - timedelta(hours=10))
    add_run(factory, "hackernews", b, NOW - timedelta(hours=30))
    add_run(factory, "hackernews", b, NOW - timedelta(hours=2), error="error: ConnectionError")  # failed: ignored
    add_run(factory, "appstore", a, NOW - timedelta(hours=5), error="warning: in: 503")  # warnings count as success
    pairs = [("hackernews", a), ("hackernews", b), ("appstore", a)]
    assert fetch_since(factory, NOW, pairs) == NOW - timedelta(hours=30) - collector.FETCH_OVERLAP


def test_since_falls_back_when_a_pair_never_succeeded(factory: sessionmaker[Session], brands: list[BrandSpec]) -> None:
    add_run(factory, "hackernews", brands[0].id, NOW - timedelta(hours=3))
    since = fetch_since(factory, NOW, [("hackernews", brands[0].id), ("appstore", brands[0].id)])
    assert since == NOW - timedelta(days=collector.settings.fetch_since_days)


def test_cycle_passes_since_to_fetch(factory: sessionmaker[Session], brands: list[BrandSpec]) -> None:
    rec = Recorder()
    run_cycle(factory, rec.stages(), now=NOW)
    assert rec.since is not None and rec.since <= NOW - collector.FETCH_OVERLAP


# ---------------------------------------------------------------- lock + CLI


def test_lock_refuses_a_second_holder(tmp_path: Path) -> None:
    path = tmp_path / "collector.lock"
    with collector_lock(path):
        assert path.read_text().strip().isdigit()  # our pid, for the message
        with pytest.raises(AlreadyRunning, match="already running"):
            with collector_lock(path):
                pass
    with collector_lock(path):  # released on exit
        pass
    for _ in range(3):  # repeated runs keep a single pid in the file, not a growing list
        with collector_lock(path):
            pass
    assert path.read_text().split() == [path.read_text().split()[0]] and path.read_text().strip().isdigit()


def test_second_collector_exits_cleanly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "collector.lock"
    monkeypatch.setattr(collector, "LOCK_FILE", path)
    with collector_lock(path):
        assert collector.main(["--once"]) == 0  # logs "already running" and returns, no traceback


def test_collector_refuses_demo_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(collector, "settings", dataclasses.replace(collector.settings, demo=True))
    assert collector.main(["--once"]) == 2


# ---------------------------------------------------------------- child process + Ctrl+C
# Top-level so the spawned child can import them.


def _child_ok() -> int:
    return 42


def _child_raises() -> int:
    raise ValueError("model exploded")


def _child_dies() -> int:
    import os

    os._exit(3)  # like being killed: no result, no exception


def _child_slow() -> int:
    import time

    time.sleep(60)
    return 1


@pytest.fixture(autouse=True)
def clear_stop():  # type: ignore[no-untyped-def]
    collector.STOP.clear()
    yield
    collector.STOP.clear()


def test_run_in_child_returns_result() -> None:
    assert collector.run_in_child(_child_ok) == 42


def test_run_in_child_reports_child_exception() -> None:
    with pytest.raises(RuntimeError, match="in child process: ValueError: model exploded"):
        collector.run_in_child(_child_raises)


def test_run_in_child_detects_a_dead_child_instead_of_hanging() -> None:
    with pytest.raises(RuntimeError, match="exited with code 3"):
        collector.run_in_child(_child_dies)


def test_stop_terminates_a_running_child_quickly() -> None:
    import threading
    import time

    threading.Timer(1.5, collector.STOP.set).start()
    started = time.perf_counter()
    with pytest.raises(collector.Interrupted):
        collector.run_in_child(_child_slow, poll=0.2)
    assert time.perf_counter() - started < 10  # not the child's 60 s


def test_stop_between_stages_marks_cycle_interrupted(factory: sessionmaker[Session], brands: list[BrandSpec]) -> None:
    rec = Recorder()
    stages = rec.stages()
    real_fetch = stages.fetch

    def fetch_then_ctrl_c(since: datetime) -> FetchOutcome:
        out = real_fetch(since)
        collector.STOP.set()  # Ctrl+C arrives during the fetch stage
        return out

    stages.fetch = fetch_then_ctrl_c
    result = run_cycle(factory, stages, now=NOW)
    assert rec.calls == ["fetch"]  # nlp, topics, priority were not started
    assert result.status == "failed" and "interrupted" in result.errors[-1]
    assert runs(factory)[0].status == "failed" and runs(factory)[0].finished_at is not None


@pytest.mark.skipif(__import__("sys").platform == "win32", reason="POSIX signal delivery")
def test_sigterm_and_ignored_sigint_both_become_keyboard_interrupt() -> None:
    import os
    import signal

    old_int, old_term = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # what `cmd &` in a script does
        collector.install_signal_handlers()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with pytest.raises(KeyboardInterrupt):
                os.kill(os.getpid(), sig)
                import time

                time.sleep(1)  # the handler runs at the next bytecode boundary
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)
