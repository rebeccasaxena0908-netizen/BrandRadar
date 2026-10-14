"""Timing, counts, tokens and cost of one LLM stage, written to `llm_runs` and logged."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from sqlalchemy.orm import Session, sessionmaker

from db.models import LlmRun
from nlp.llm.provider import LLMResult, Provider
from sources.base import utc_now

log = logging.getLogger("nlp.llm")


@dataclass
class StageRun:
    stage: str
    provider: str
    model: str
    started_at: object = field(default_factory=utc_now)
    t0: float = field(default_factory=time.perf_counter)
    items: int = 0
    generated: int = 0
    cached: int = 0
    fallbacks: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    errors: list[str] = field(default_factory=list)
    seconds: float = 0.0

    @classmethod
    def start(cls, stage: str, provider: Provider) -> StageRun:
        return cls(stage, provider.name, provider.model)

    def add(self, result: LLMResult) -> None:
        self.input_tokens += result.input_tokens
        self.output_tokens += result.output_tokens
        self.cost_usd += result.cost_usd

    def fallback(self, reason: str) -> None:
        self.fallbacks += 1
        if reason not in self.errors:
            self.errors.append(reason)

    @property
    def summary(self) -> str:
        parts = [f"{self.generated} generated", f"{self.cached} cached"]
        if self.fallbacks:
            parts.append(f"{self.fallbacks} rule-based fallback(s)")
        tokens = f", {self.input_tokens:,}+{self.output_tokens:,} tokens" if self.input_tokens else ""
        cost = f", ${self.cost_usd:.4f}" if self.cost_usd else ""
        return f"{self.stage}: {', '.join(parts)} in {self.seconds:.1f}s ({self.provider} {self.model}{tokens}{cost})"

    def finish(self, factory: sessionmaker[Session] | None) -> StageRun:
        self.seconds = round(time.perf_counter() - self.t0, 2)
        if factory is not None:
            with factory() as session:
                session.add(LlmRun(
                    stage=self.stage, provider=self.provider, model=self.model, started_at=self.started_at,
                    seconds=self.seconds, items=self.items, generated=self.generated, cached=self.cached,
                    fallbacks=self.fallbacks, input_tokens=self.input_tokens, output_tokens=self.output_tokens,
                    cost_usd=round(self.cost_usd, 6), error="; ".join(self.errors)[:1024] or None,
                ))
                session.commit()
        log.info("%s", self.summary)
        return self
