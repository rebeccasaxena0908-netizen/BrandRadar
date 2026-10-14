"""Notifications: new spike alerts after each collector cycle, and the weekly PDF every Monday morning.

    python -m reports.notify --test      # one test message to every configured channel
    python -m reports.notify --alerts    # send alerts not sent yet (the collector does this every cycle)
    python -m reports.notify --weekly    # send last week's report now, if it hasn't gone out yet

Optional and off unless configured in .env: email over SMTP (works with a Gmail app password) and/or a
Slack incoming webhook. Each alert and each weekly report is sent at most once per channel: a row in
`notifications` is claimed *before* the message goes out (so a crash can lose a message but never repeat
one), and a failed send is retried up to MAX_ATTEMPTS times in later cycles. Only alerts from the last 7
days are sent, so switching notifications on doesn't deliver old history. Slack webhooks can't carry
files, so Slack gets a short summary and the PDFs go by email. Demo mode never sends.
"""

from __future__ import annotations

import argparse
import html
import logging
import smtplib
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import requests
from sqlalchemy import Engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from config import Settings, settings
from db.models import Alert, Brand, Notification, Topic
from reports import quiet_streamlit
from sources.base import utc_now

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
ALERT_WINDOW = timedelta(days=7)  # older unsent alerts are history, not news
WEEKLY_CATCH_UP = timedelta(hours=24)  # a collector that was off on Monday morning still sends that day
_LOCK = threading.Lock()  # the weekly job and a cycle's notify stage can overlap in the scheduler


# ---------------------------------------------------------------- messages and channels


@dataclass
class Message:
    subject: str
    text: str  # plain-text email body
    html: str  # HTML email body
    slack: str  # Slack mrkdwn
    attachments: list[Path] = field(default_factory=list)


class Channel(Protocol):
    name: str

    def send(self, message: Message) -> None: ...


class EmailChannel:
    name = "email"

    def __init__(self, cfg: Settings = settings, smtp: Callable[..., Any] = smtplib.SMTP,
                 smtp_ssl: Callable[..., Any] = smtplib.SMTP_SSL) -> None:
        self.cfg, self.smtp, self.smtp_ssl = cfg, smtp, smtp_ssl

    def send(self, message: Message) -> None:
        email = EmailMessage()
        email["Subject"], email["From"], email["To"] = message.subject, self.cfg.smtp_from, ", ".join(self.cfg.notify_email_to)
        email.set_content(message.text)
        email.add_alternative(message.html, subtype="html")
        for path in message.attachments:
            email.add_attachment(path.read_bytes(), maintype="application", subtype="pdf", filename=path.name)
        # 465 = TLS from the start; anything else (587) upgrades with STARTTLS
        client = (self.smtp_ssl if self.cfg.smtp_port == 465 else self.smtp)(self.cfg.smtp_host, self.cfg.smtp_port, timeout=30)
        try:
            if self.cfg.smtp_port != 465:
                client.starttls()
            if self.cfg.smtp_user:
                client.login(self.cfg.smtp_user, self.cfg.smtp_password)
            client.send_message(email)
        finally:
            try:
                client.quit()
            except Exception:  # noqa: BLE001 — the message is already sent or the error is already raised
                pass


class SlackChannel:
    name = "slack"

    def __init__(self, cfg: Settings = settings, post: Callable[..., Any] = requests.post) -> None:
        self.cfg, self.post = cfg, post

    def send(self, message: Message) -> None:
        # The webhook URL is a secret: it never appears in errors, logs or the notifications table.
        response = self.post(self.cfg.slack_webhook_url, json={"text": message.slack}, timeout=15)
        if response.status_code != 200:
            raise RuntimeError(f"Slack answered {response.status_code}: {response.text[:120]}")


def configured_channels(cfg: Settings = settings) -> list[Channel]:
    channels: list[Channel] = []
    if cfg.email_enabled:
        channels.append(EmailChannel(cfg))
    if cfg.slack_enabled:
        channels.append(SlackChannel(cfg))
    return channels


def safe_error(exc: Exception, cfg: Settings = settings) -> str:
    text = f"{type(exc).__name__}: {exc}"
    for secret in (cfg.slack_webhook_url, cfg.smtp_password):
        if secret:
            text = text.replace(secret, "***")
    return text[:1000]


# ---------------------------------------------------------------- ledger


def claim(engine: Engine, kind: str, ref: str, channel: str) -> int | None:
    """Reserve (kind, ref, channel) before sending. Returns the row id, or None if it was already sent,
    is being sent, or failed MAX_ATTEMPTS times."""
    with Session(engine) as session:
        row = session.scalar(select(Notification).where(
            Notification.kind == kind, Notification.ref == ref, Notification.channel == channel))
        if row is None:
            row = Notification(kind=kind, ref=ref, channel=channel, status="sending", attempts=1, created_at=utc_now())
            session.add(row)
            try:
                session.commit()
            except IntegrityError:  # someone else claimed it a moment ago
                return None
            return row.id
        if row.status == "failed" and row.attempts < MAX_ATTEMPTS:
            row.status, row.attempts = "sending", row.attempts + 1
            session.commit()
            return row.id
        return None


def finish(engine: Engine, ids: Sequence[int], error: str | None) -> None:
    with Session(engine) as session:
        for row in session.scalars(select(Notification).where(Notification.id.in_(ids))):
            row.status = "failed" if error else "sent"
            row.error = error
            row.sent_at = None if error else utc_now()
        session.commit()


@dataclass
class Outcome:
    sent: list[str] = field(default_factory=list)  # "email: 2 alert(s)"
    errors: list[str] = field(default_factory=list)
    skipped: str | None = None

    @property
    def summary(self) -> str:
        if self.skipped:
            return self.skipped
        parts = self.sent + [f"FAILED {e}" for e in self.errors]
        return "; ".join(parts) if parts else "nothing new to send"


def _deliver(engine: Engine, channel: Channel, kind: str, refs: Sequence[str], build: Callable[[list[str]], Message],
             outcome: Outcome, what: str) -> None:
    """Claim the refs on one channel, send one message about the claimed ones, record the result."""
    claimed = [(ref, row_id) for ref in refs if (row_id := claim(engine, kind, ref, channel.name)) is not None]
    if not claimed:
        return
    ids = [row_id for _, row_id in claimed]
    try:
        channel.send(build([ref for ref, _ in claimed]))
    except Exception as exc:  # noqa: BLE001 — recorded, retried next cycle
        error = safe_error(exc)
        finish(engine, ids, error)
        outcome.errors.append(f"{channel.name}: {error}")
        log.warning("notify %s failed: %s", channel.name, error)
        return
    finish(engine, ids, None)
    outcome.sent.append(f"{channel.name}: {len(claimed)} {what}")


def _refuse(channels: Sequence[Channel]) -> str | None:
    if settings.demo:
        return "skipped (demo mode never sends)"
    if not channels:
        return "notifications off (no SMTP or Slack settings)"
    return None


# ---------------------------------------------------------------- alerts


@dataclass
class AlertInfo:
    id: int
    brand: str
    group: str
    label: str
    category: str | None
    this_week: int
    last_week: int
    created_at: datetime
    quote: str | None


def recent_alerts(engine: Engine, now: datetime) -> list[AlertInfo]:
    from ui import data, insights

    with Session(engine) as session:
        rows = session.execute(
            select(Alert, Brand.name, Brand.brand_group, Topic.label, Topic.category)
            .join(Brand, Brand.id == Alert.brand_id).outerjoin(Topic, Topic.id == Alert.topic_id)
            .where(Alert.created_at >= now - ALERT_WINDOW).order_by(Alert.created_at)
        ).all()
    topic_ids = tuple(a.topic_id for a, *_ in rows if a.topic_id)
    texts = data.query_topic_texts(engine, topic_ids)
    out = []
    for alert, brand, group, label, category in rows:
        quotes = insights.topic_quotes(texts, alert.topic_id, n=1, max_len=160) if alert.topic_id else []
        out.append(AlertInfo(alert.id, brand, group, label or alert.message, category, int(alert.this_week or 0),
                             int(alert.last_week or 0), alert.created_at, quotes[0] if quotes else None))
    return out


def alert_message(alerts: Sequence[AlertInfo], cfg: Settings = settings) -> Message:
    from ui.insights import trend_text

    brands = ", ".join(dict.fromkeys(a.brand for a in alerts))
    n = len(alerts)
    subject = f"Brand Radar: {n} new complaint spike{'s' if n != 1 else ''} ({brands})"
    lines, items, slack = [], [], []
    for a in alerts:
        ratio = trend_text(a.this_week, a.last_week)
        category = f" ({a.category})" if a.category else ""
        lines.append(f"- {a.brand} [{a.group}]: {a.label}{category}. {a.this_week} complaints this week vs "
                     f"{a.last_week} last week ({ratio})." + (f'\n  "{a.quote}"' if a.quote else ""))
        items.append(
            f'<li style="margin:0 0 12px"><b>{html.escape(a.brand)}</b> <span style="color:#9C9899">{html.escape(a.group)}</span><br>'
            f'<span style="color:#D9536A;font-weight:600">{html.escape(a.label)}</span>{html.escape(category)} · '
            f'{a.this_week} this week vs {a.last_week} last week (<b>{ratio}</b>)'
            + (f'<br><i style="color:#6B5F62">“{html.escape(a.quote)}”</i>' if a.quote else "") + "</li>")
        slack.append(f"• *{a.brand}* · {a.label}{category}: {a.this_week} this week vs {a.last_week} last week ({ratio})"
                     + (f"\n> {a.quote}" if a.quote else ""))
    link = cfg.dashboard_url
    return Message(
        subject=subject,
        text=f"Complaints about these topics at least doubled week over week:\n\n" + "\n".join(lines)
             + f"\n\nDashboard: {link}\n",
        html=_html_mail("New complaint spikes", "Complaints about these topics at least doubled week over week.",
                        f'<ul style="padding-left:18px">{"".join(items)}</ul>', link),
        slack=f"*Brand Radar* · {n} new complaint spike{'s' if n != 1 else ''}\n" + "\n".join(slack)
              + f"\n<{link}|Open the dashboard>",
    )


def send_alerts(engine: Engine, now: datetime | None = None, channels: Sequence[Channel] | None = None) -> Outcome:
    """Send every alert from the last 7 days that a channel hasn't had yet, as one message per channel."""
    channels = configured_channels() if channels is None else channels
    if (reason := _refuse(channels)):
        return Outcome(skipped=reason)
    now = now or utc_now()
    outcome = Outcome()
    with _LOCK:
        alerts = {str(a.id): a for a in recent_alerts(engine, now)}
        for channel in channels:
            _deliver(engine, channel, "alert", list(alerts), lambda refs: alert_message([alerts[r] for r in refs]),
                     outcome, "alert(s)")
    return outcome


# ---------------------------------------------------------------- weekly report


def weekly_due(now_utc: datetime, cfg: Settings = settings) -> tuple[str, datetime]:
    """(ISO week label, due time in UTC) of the most recent weekly report whose send time has passed:
    Monday NOTIFY_WEEKLY_HOUR in NOTIFY_TIMEZONE, covering the Monday-Sunday week before it."""
    from reports.weeks import Week

    tz = ZoneInfo(cfg.notify_timezone)
    local = now_utc.replace(tzinfo=ZoneInfo("UTC")).astimezone(tz)
    monday = local.date() - timedelta(days=local.weekday())
    due = datetime.combine(monday, time(cfg.notify_weekly_hour), tzinfo=tz)
    if local < due:
        monday -= timedelta(days=7)
        due -= timedelta(days=7)
    week = Week.containing(monday - timedelta(days=1))
    return week.label, due.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)


def weekly_message(reports: Sequence[tuple[Any, Path]], week: Any, email_on: bool, cfg: Settings = settings) -> Message:
    def strip(markup: str) -> str:
        return html.unescape(markup.replace("<b>", "").replace("</b>", ""))

    def net(rep: Any) -> str:
        k = rep.kpis
        if k.net_sentiment is None:
            return "no mentions"
        change = "" if k.net_sentiment_change is None else f" ({k.net_sentiment_change:+.0f} pts)"
        return f"net sentiment {k.net_sentiment:+.0f}{change} · {k.mentions} mentions"

    subject = f"Brand Radar weekly report · {week.label} ({week.span})"
    text_lines, html_items, slack_lines = [], [], []
    for rep, path in reports:
        # the AI-written briefing for the week when there is one (Step 10), else the rule-based risk line
        lead = rep.briefing["text"] if rep.briefing else strip(rep.summary.risk)
        text_lines.append(f"{rep.name} [{rep.brand['brand_group']}]: {net(rep)}\n  {lead}\n  PDF: {path.name}")
        html_items.append(f'<li style="margin:0 0 12px"><b>{html.escape(rep.name)}</b> '
                          f'<span style="color:#9C9899">{html.escape(str(rep.brand["brand_group"]))}</span><br>'
                          f'{html.escape(net(rep))}<br><span style="color:#6B5F62">'
                          + (html.escape(rep.briefing["text"]) + " <i>(AI-written)</i>" if rep.briefing else rep.summary.risk)
                          + "</span></li>")
        slack_lines.append(f"• *{rep.name}* ({rep.brand['brand_group']}): {net(rep)}\n   "
                           + (f"{rep.briefing['text']} _(AI-written)_" if rep.briefing else strip(rep.summary.risk)))
    where = "The PDF reports are attached." if email_on else ""
    slack_where = ("PDF reports sent by email." if email_on
                   else "PDF reports are saved in reports/output on the collector machine.")
    return Message(
        subject=subject,
        text=f"Weekly report for {week.span}.\n\n" + "\n\n".join(text_lines) + f"\n\n{where}\nDashboard: {cfg.dashboard_url}\n",
        html=_html_mail(f"Weekly report · {week.span}", where, f'<ul style="padding-left:18px">{"".join(html_items)}</ul>',
                        cfg.dashboard_url),
        slack=f"*Brand Radar weekly report* · {week.span}\n" + "\n".join(slack_lines) + f"\n{slack_where} <{cfg.dashboard_url}|Open the dashboard>",
        attachments=[path for _, path in reports],
    )


def send_weekly(engine: Engine, now: datetime | None = None, channels: Sequence[Channel] | None = None,
                week_label: str | None = None, renderer: Any = None, force_due: bool = False) -> Outcome:
    """Send the weekly report once per week and channel. Without week_label it sends the report that is
    due (Monday morning) if that was less than WEEKLY_CATCH_UP ago, so enabling notifications mid-week
    doesn't send last week's report out of the blue; force_due (CLI --weekly) skips that window."""
    from reports import weekly

    channels = configured_channels() if channels is None else channels
    if (reason := _refuse(channels)):
        return Outcome(skipped=reason)
    now = now or utc_now()
    if week_label is None:
        week_label, due = weekly_due(now)
        if not force_due and now - due > WEEKLY_CATCH_UP:
            return Outcome(skipped=f"weekly report {week_label} not due (sent on Mondays)")
    week = weekly.Week.parse(week_label)
    outcome = Outcome()
    with _LOCK:
        claimed = {c.name: row for c in channels if (row := claim(engine, "weekly", week.label, c.name)) is not None}
        if not claimed:
            return outcome
        try:
            brands = weekly.data.query_brands(engine)
            refs = weekly.default_brands(brands, settings.notify_weekly_brands)
            reports: list[tuple[Any, Path]] = []

            def build(r: Any) -> None:
                for ref in refs:
                    rep = weekly.gather(engine, ref, week, now)
                    path = weekly.report_path(rep.name, week)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(weekly.build_pdf(rep, r))
                    reports.append((rep, path))

            if renderer is not None:
                build(renderer)
            else:
                with weekly.renderer() as r:
                    build(r)
        except Exception as exc:  # noqa: BLE001 — the report couldn't be built: every claimed channel failed
            error = f"report: {safe_error(exc)}"
            finish(engine, list(claimed.values()), error)
            outcome.errors.append(error)
            return outcome
        email_on = any(c.name == "email" for c in channels)  # PDFs travel by email; Slack only says so
        for channel in channels:
            if channel.name not in claimed:
                continue
            message = weekly_message(reports, week, email_on)
            if channel.name != "email":
                message.attachments = []
            try:
                channel.send(message)
            except Exception as exc:  # noqa: BLE001
                error = safe_error(exc)
                finish(engine, [claimed[channel.name]], error)
                outcome.errors.append(f"{channel.name}: {error}")
                continue
            finish(engine, [claimed[channel.name]], None)
            outcome.sent.append(f"{channel.name}: weekly {week.label} ({len(reports)} report(s))")
    return outcome


def after_cycle(engine: Engine, now: datetime | None = None) -> Outcome:
    """The collector's notify stage: new alerts, plus the weekly report if Monday's send was missed."""
    channels = configured_channels()
    if (reason := _refuse(channels)):
        return Outcome(skipped=reason)
    alerts = send_alerts(engine, now, channels)
    weekly_out = send_weekly(engine, now, channels)
    return Outcome(sent=alerts.sent + weekly_out.sent, errors=alerts.errors + weekly_out.errors)


# ---------------------------------------------------------------- email layout


def _html_mail(title: str, intro: str, body: str, link: str) -> str:
    return (
        '<div style="font-family:Figtree,Helvetica,Arial,sans-serif;color:#242424;background:#FBEFF1;padding:24px">'
        '<div style="max-width:560px;margin:auto;background:#ffffff;border-radius:16px;padding:22px 24px;'
        'border:1px solid #F1D9DD">'
        '<div style="font-family:Montserrat,Helvetica,Arial,sans-serif;font-weight:700;letter-spacing:3px;'
        'font-size:11px;color:#9E4A5A">BRAND RADAR</div>'
        f'<h2 style="font-family:Montserrat,Helvetica,Arial,sans-serif;margin:8px 0 6px">{html.escape(title)}</h2>'
        f'<p style="color:#6B5F62;margin:0 0 14px">{html.escape(intro)}</p>{body}'
        f'<p><a href="{html.escape(link)}" style="color:#9E4A5A;font-weight:600">Open the dashboard</a></p>'
        "</div></div>"
    )


# ---------------------------------------------------------------- CLI


def test_message() -> Message:
    now = utc_now()
    return Message(
        subject="Brand Radar test message",
        text=f"This is a test from Brand Radar ({now:%d %b %Y %H:%M} UTC). Notifications are set up correctly.\n",
        html=_html_mail("Test message", f"Sent {now:%d %b %Y %H:%M} UTC. Notifications are set up correctly.", "",
                        settings.dashboard_url),
        slack=f"*Brand Radar* test message ({now:%d %b %Y %H:%M} UTC). Notifications are set up correctly.",
    )


def send_test(channels: Sequence[Channel] | None = None, cfg: Settings = settings, say: Callable[[str], None] = print) -> int:
    """Not recorded in the ledger; works in demo mode too. Returns the number of failed channels."""
    channels = configured_channels(cfg) if channels is None else channels
    failed = 0
    names = {c.name for c in channels}
    for name, hint in (("email", "set SMTP_HOST, SMTP_USER, SMTP_PASSWORD and NOTIFY_EMAIL_TO"),
                       ("slack", "set SLACK_WEBHOOK_URL")):
        if name not in names:
            say(f"  - {name:<6} skipped (not configured: {hint})")
    for channel in channels:
        try:
            channel.send(test_message())
            target = ", ".join(cfg.notify_email_to) if channel.name == "email" else "the Slack channel"
            say(f"  ✔ {channel.name:<6} sent to {target}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            say(f"  ✘ {channel.name:<6} {safe_error(exc, cfg)}")
    return failed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--test", action="store_true", help="send a test message to every configured channel")
    action.add_argument("--alerts", action="store_true", help="send alerts that haven't been sent yet")
    action.add_argument("--weekly", action="store_true", help="send last week's report now (once per week)")
    parser.add_argument("--week", help="with --weekly: an ISO week such as 2026-W41")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    quiet_streamlit()

    if args.test:
        print("Brand Radar notification test")
        return 1 if send_test() else 0
    from db.session import get_engine

    engine = get_engine()
    outcome = send_alerts(engine) if args.alerts else send_weekly(engine, week_label=args.week, force_due=True)
    print(outcome.summary)
    return 1 if outcome.errors else 0


if __name__ == "__main__":
    sys.exit(main())
