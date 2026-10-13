"""Notifications: alerts and the weekly report go out once per channel, failures retry, nothing is sent
when not configured or in demo mode. SMTP and the Slack webhook are mocked."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import responses
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

import seed_demo
from config import settings
from db.models import Alert, Analysis, Base, Brand, Mention, Notification, Topic
from reports import notify, weekly
from reports.notify import EmailChannel, Message, SlackChannel
from reports.render import StubRenderer

NOW = datetime(2026, 10, 14, 12, 0)  # Wednesday
HOOK = "https://hooks.slack.com/services/T000/B000/secretsecret"
CFG = dataclasses.replace(
    settings, demo=False, smtp_host="smtp.gmail.com", smtp_port=587, smtp_user="radar@example.com",
    smtp_password="abcdefghijklmnop", smtp_from="radar@example.com", notify_email_to=("team@example.com",),
    slack_webhook_url=HOOK, notify_timezone="Asia/Kolkata", notify_weekly_hour=8, notify_weekly_brands=("own",),
    dashboard_url="http://localhost:8501",
)


class FakeChannel:
    def __init__(self, name: str, fail_times: int = 0) -> None:
        self.name, self.fail_times = name, fail_times
        self.messages: list[Message] = []

    def send(self, message: Message) -> None:
        if self.fail_times:
            self.fail_times -= 1
            raise ConnectionError("server unavailable")
        self.messages.append(message)


@pytest.fixture(autouse=True)
def configured(monkeypatch):  # type: ignore[no-untyped-def]
    monkeypatch.setattr(notify, "settings", CFG)
    monkeypatch.setattr(weekly, "settings", CFG)


@pytest.fixture
def engine():  # type: ignore[no-untyped-def]
    """Two brands, a topic with a quote, and three alerts: two recent, one older than the 7-day window."""
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        doms = Brand(name="DOMS", is_own=True, keywords=["doms"], brand_group="Stationery", profile="stationery")
        laneige = Brand(name="Laneige", is_own=True, keywords=["laneige"], brand_group="Skincare", profile="beauty_skincare")
        s.add_all([doms, laneige])
        s.flush()
        topic = Topic(brand_id=doms.id, label="Leads Break While Sharpening", category="sharpening", keywords=["lead"])
        s.add(topic)
        s.flush()
        m = Mention(brand_id=doms.id, source="youtube", external_id="q1", created_at=NOW - timedelta(days=1), fetched_at=NOW,
                    text="The lead keeps breaking every time my kid sharpens the pencil, we lose half the pencil.")
        s.add(m)
        s.flush()
        s.add(Analysis(mention_id=m.id, sentiment="negative", sentiment_score=-0.9, category="sharpening",
                       topic_id=topic.id, model_version="t"))
        s.add_all([
            Alert(brand_id=doms.id, topic_id=topic.id, kind="spike", message="spike", this_week=9, last_week=1,
                  created_at=NOW - timedelta(hours=3)),
            Alert(brand_id=laneige.id, topic_id=None, kind="spike", message="Sticky Sleeping Mask doubled",
                  this_week=12, last_week=4, created_at=NOW - timedelta(days=2)),
            Alert(brand_id=laneige.id, topic_id=None, kind="spike", message="Old spike", this_week=6, last_week=2,
                  created_at=NOW - timedelta(days=10)),
        ])
        s.commit()
    yield engine
    engine.dispose()


def ledger(engine) -> list[tuple[str, str, str, str]]:  # type: ignore[no-untyped-def]
    with Session(engine) as s:
        return [(n.kind, n.ref, n.channel, n.status) for n in s.scalars(select(Notification).order_by(Notification.id))]


# ---------------------------------------------------------------- alerts


def test_alerts_are_sent_once_per_channel(engine) -> None:  # type: ignore[no-untyped-def]
    email, slack = FakeChannel("email"), FakeChannel("slack")
    out = notify.send_alerts(engine, NOW, [email, slack])
    assert out.sent == ["email: 2 alert(s)", "slack: 2 alert(s)"] and not out.errors
    [mail] = email.messages
    assert mail.subject == "Brand Radar: 2 new complaint spikes (Laneige, DOMS)"
    assert "Leads Break While Sharpening (sharpening)" in mail.text and "9.0×" in mail.text
    assert "The lead keeps breaking" in mail.text and "Old spike" not in mail.text  # quote; old alert left out
    assert "<b>9.0×</b>" in mail.html and "http://localhost:8501" in mail.html
    assert slack.messages[0].slack.startswith("*Brand Radar* · 2 new complaint spikes")
    # the next cycle sends nothing again
    again = notify.send_alerts(engine, NOW + timedelta(hours=6), [email, slack])
    assert again.summary == "nothing new to send" and len(email.messages) == len(slack.messages) == 1
    assert {row[3] for row in ledger(engine)} == {"sent"} and len(ledger(engine)) == 4


def test_a_new_alert_later_is_sent_on_its_own(engine) -> None:  # type: ignore[no-untyped-def]
    email = FakeChannel("email")
    notify.send_alerts(engine, NOW, [email])
    with Session(engine) as s:
        s.add(Alert(brand_id=1, topic_id=None, kind="spike", message="Gel Pen Ink Skips doubled", this_week=6,
                    last_week=2, created_at=NOW + timedelta(hours=5)))
        s.commit()
    notify.send_alerts(engine, NOW + timedelta(hours=6), [email])
    assert len(email.messages) == 2 and email.messages[1].subject == "Brand Radar: 1 new complaint spike (DOMS)"


def test_failed_channel_is_retried_without_resending_the_other(engine) -> None:  # type: ignore[no-untyped-def]
    email, slack = FakeChannel("email"), FakeChannel("slack", fail_times=1)
    out = notify.send_alerts(engine, NOW, [email, slack])
    assert out.sent == ["email: 2 alert(s)"] and out.errors == ["slack: ConnectionError: server unavailable"]
    assert ("alert", "1", "slack", "failed") in ledger(engine)
    out = notify.send_alerts(engine, NOW + timedelta(hours=6), [email, slack])
    assert out.sent == ["slack: 2 alert(s)"]
    assert len(email.messages) == 1 and len(slack.messages) == 1


def test_gives_up_after_max_attempts(engine) -> None:  # type: ignore[no-untyped-def]
    slack = FakeChannel("slack", fail_times=99)
    for hour in range(notify.MAX_ATTEMPTS + 2):
        notify.send_alerts(engine, NOW + timedelta(hours=hour), [slack])
    with Session(engine) as s:
        attempts = {n.attempts for n in s.scalars(select(Notification))}
    assert attempts == {notify.MAX_ATTEMPTS}


def test_skipped_cleanly_when_not_configured_or_in_demo_mode(engine, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    off = dataclasses.replace(CFG, smtp_host="", slack_webhook_url="")
    assert notify.configured_channels(off) == []
    assert [c.name for c in notify.configured_channels(CFG)] == ["email", "slack"]
    monkeypatch.setattr(notify, "settings", off)
    assert notify.after_cycle(engine, NOW).summary == "notifications off (no SMTP or Slack settings)"
    monkeypatch.setattr(notify, "settings", dataclasses.replace(CFG, demo=True))
    channel = FakeChannel("email")
    assert notify.send_alerts(engine, NOW, [channel]).skipped == "skipped (demo mode never sends)"
    assert notify.send_weekly(engine, NOW, [channel], week_label="2026-W41").skipped
    assert channel.messages == [] and ledger(engine) == []


# ---------------------------------------------------------------- channels


class FakeSMTP:
    instances: list[FakeSMTP] = []

    def __init__(self, host: str, port: int, timeout: int) -> None:
        self.host, self.port, self.calls, self.sent = host, port, [], []
        FakeSMTP.instances.append(self)

    def starttls(self) -> None:
        self.calls.append("starttls")

    def login(self, user: str, password: str) -> None:
        self.calls.append(f"login {user} {password}")

    def send_message(self, message) -> None:  # type: ignore[no-untyped-def]
        self.sent.append(message)

    def quit(self) -> None:
        self.calls.append("quit")


def test_email_channel_uses_starttls_and_attaches_pdfs(tmp_path: Path) -> None:
    FakeSMTP.instances.clear()
    pdf = tmp_path / "brandradar-doms-2026-W41.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    EmailChannel(CFG, smtp=FakeSMTP).send(Message("Subject", "text", "<p>html</p>", "slack", [pdf]))
    [smtp] = FakeSMTP.instances
    assert (smtp.host, smtp.port) == ("smtp.gmail.com", 587)
    assert smtp.calls == ["starttls", "login radar@example.com abcdefghijklmnop", "quit"]
    [msg] = smtp.sent
    assert msg["To"] == "team@example.com" and msg["From"] == "radar@example.com" and msg["Subject"] == "Subject"
    [attachment] = list(msg.iter_attachments())
    assert attachment.get_filename() == pdf.name and attachment.get_content() == b"%PDF-1.4 test"


def test_email_port_465_uses_ssl() -> None:
    FakeSMTP.instances.clear()
    EmailChannel(dataclasses.replace(CFG, smtp_port=465), smtp=None, smtp_ssl=FakeSMTP).send(  # type: ignore[arg-type]
        Message("s", "t", "h", "k"))
    assert "starttls" not in FakeSMTP.instances[0].calls


@responses.activate
def test_slack_channel_posts_text_and_never_leaks_the_webhook() -> None:
    responses.add(responses.POST, HOOK, body="ok")
    SlackChannel(CFG).send(Message("s", "t", "h", "*hello*"))
    assert responses.calls[0].request.body == b'{"text": "*hello*"}'
    responses.replace(responses.POST, HOOK, status=404, body="no_service")
    with pytest.raises(RuntimeError) as err:
        SlackChannel(CFG).send(Message("s", "t", "h", "x"))
    assert "404" in str(err.value) and HOOK not in str(err.value)
    leaked = ConnectionError(f"could not reach {HOOK} with password abcdefghijklmnop")
    assert HOOK not in notify.safe_error(leaked, CFG) and "abcdefghijklmnop" not in notify.safe_error(leaked, CFG)


def test_send_test_reports_each_channel() -> None:
    lines: list[str] = []
    failed = notify.send_test([FakeChannel("email"), FakeChannel("slack", fail_times=1)], CFG, lines.append)
    assert failed == 1
    assert any(line.startswith("  ✔ email") for line in lines) and any("✘ slack" in line for line in lines)
    lines.clear()
    notify.send_test([], dataclasses.replace(CFG, smtp_host="", slack_webhook_url=""), lines.append)
    assert all("skipped (not configured" in line for line in lines) and len(lines) == 2


# ---------------------------------------------------------------- weekly report


def test_weekly_is_due_monday_morning_india_time() -> None:
    # Monday 12 Oct 2026, 08:00 IST = 02:30 UTC
    assert notify.weekly_due(datetime(2026, 10, 12, 2, 30), CFG) == ("2026-W41", datetime(2026, 10, 12, 2, 30))
    assert notify.weekly_due(datetime(2026, 10, 12, 2, 29), CFG)[0] == "2026-W40"  # not yet: last week's is current
    assert notify.weekly_due(datetime(2026, 10, 18, 20, 0), CFG)[0] == "2026-W41"  # Sunday night


@pytest.fixture
def demo_engine():  # type: ignore[no-untyped-def]
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    seed_demo.seed(engine, now=NOW)
    yield engine
    engine.dispose()


def test_weekly_report_is_sent_once_with_pdfs_by_email(demo_engine, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(weekly, "OUTPUT_DIR", tmp_path)
    email, slack = FakeChannel("email"), FakeChannel("slack")
    monday = datetime(2026, 10, 12, 2, 35)  # 08:05 IST
    out = notify.send_weekly(demo_engine, monday, [email, slack], renderer=StubRenderer())
    assert out.sent == ["email: weekly 2026-W41 (3 report(s))", "slack: weekly 2026-W41 (3 report(s))"]
    [mail] = email.messages
    assert mail.subject == "Brand Radar weekly report · 2026-W41 (5–11 Oct 2026)"
    assert sorted(p.name for p in mail.attachments) == [
        "brandradar-doms-2026-W41.pdf", "brandradar-laneige-2026-W41.pdf", "brandradar-notion-2026-W41.pdf"]
    assert all(p.read_bytes().startswith(b"%PDF") for p in mail.attachments)
    [post] = slack.messages
    assert post.attachments == [] and "PDF reports sent by email" in post.slack and "*Laneige*" in post.slack
    # the Monday job and the next cycles don't send it again
    for later in (monday + timedelta(minutes=5), monday + timedelta(hours=6)):
        assert notify.send_weekly(demo_engine, later, [email, slack], renderer=StubRenderer()).summary == "nothing new to send"
    assert len(email.messages) == len(slack.messages) == 1


def test_weekly_not_sent_out_of_the_blue_midweek(demo_engine) -> None:  # type: ignore[no-untyped-def]
    email = FakeChannel("email")
    out = notify.send_weekly(demo_engine, NOW, [email], renderer=StubRenderer())  # Wednesday: Monday's is >24 h old
    assert out.skipped == "weekly report 2026-W41 not due (sent on Mondays)" and email.messages == []
    forced = notify.send_weekly(demo_engine, NOW, [email], renderer=StubRenderer(), force_due=True)  # CLI --weekly
    assert forced.sent and len(email.messages) == 1


def test_weekly_build_failure_marks_channels_failed(demo_engine, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def broken(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("Chrome not found")

    monkeypatch.setattr(weekly, "gather", broken)
    email = FakeChannel("email")
    out = notify.send_weekly(demo_engine, NOW, [email], week_label="2026-W41", renderer=StubRenderer())
    assert out.errors == ["report: RuntimeError: Chrome not found"] and email.messages == []
    assert ledger(demo_engine) == [("weekly", "2026-W41", "email", "failed")]
