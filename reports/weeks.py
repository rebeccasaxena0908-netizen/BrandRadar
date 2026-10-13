"""ISO weeks (Monday-Sunday, UTC dates) for the weekly report. Kept tiny so the dashboard can use it
without loading reportlab or kaleido."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class Week:
    year: int
    number: int

    @classmethod
    def parse(cls, text: str) -> Week:
        match = re.fullmatch(r"(\d{4})-?W(\d{1,2})", text.strip(), re.IGNORECASE)
        if not match:
            raise ValueError(f"week must look like 2026-W41, not {text!r}")
        week = cls(int(match.group(1)), int(match.group(2)))
        date.fromisocalendar(week.year, week.number, 1)  # raises for W53 in a 52-week year
        return week

    @classmethod
    def containing(cls, day: date) -> Week:
        iso = day.isocalendar()
        return cls(iso.year, iso.week)

    @classmethod
    def last_complete(cls, today: date) -> Week:
        """The most recent Monday-Sunday week that has ended before `today`."""
        return cls.containing(today - timedelta(days=today.isoweekday()))

    @property
    def start(self) -> date:
        return date.fromisocalendar(self.year, self.number, 1)

    @property
    def end(self) -> date:
        return self.start + timedelta(days=6)

    @property
    def label(self) -> str:
        return f"{self.year}-W{self.number:02d}"

    @property
    def span(self) -> str:
        s, e = self.start, self.end
        left = f"{s.day} {s:%b}" if s.month != e.month else f"{s.day}"
        return f"{left}–{e.day} {e:%b %Y}"

    def previous(self) -> Week:
        return Week.containing(self.start - timedelta(days=1))
