"""LLM briefing: 4-5 written sentences per brand (biggest risk, biggest win vs competitors in the same group,
top request, notable change), grounded only in a facts block built from the dashboard's own numbers
(ui/data + ui/insights: the same picks as the rule-based executive summary) and three quotes.

Grounding check: every number in the text must appear in the facts (format variants such as 9 and 9.0 are
fine) and numbers may not be written as words; anything else is rejected and the rule-based summary is
shown instead. Two periods are stored in `briefings`:
- 'last7': the 7 days up to today, for the dashboard; regenerated only when its facts change (facts_hash)
- 'week':  an ISO week, for the PDF and the Monday email; written once after the week ends
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import date, timedelta
from typing import Any

from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session, sessionmaker

from config import settings
from db.models import Briefing, Brand
from nlp.llm.provider import LLMError, LLMUnavailable, Provider
from nlp.llm.runlog import StageRun
from nlp.profiles import get_profile
from reports.weeks import Week
from sources.base import utc_now

RULES = (
    "Use ONLY the facts you are given. Every number you write must appear in the facts exactly as given, with "
    "the same sign (+39 stays +39, -14 stays -14); write numbers as digits, never as words (2 requests, not two "
    "requests), and do not compute new numbers (no percentages, differences or ratios of your own). Net "
    "sentiment is a score from -100 to +100. Do not call anything a risk, a win or a request unless the facts "
    "name it as one, and leave out everything listed under 'Nothing to report'. Do not guess causes beyond "
    "what the quotes say. Do not describe a change in a different direction or size than the facts state (if "
    "the facts say 'the same as', do not call it an increase). No headings, bullet points or markdown. Answer "
    "with the briefing text only."
)


def system_prompt(facts: dict[str, Any]) -> str:
    """The parts to cover are the ones the facts have: asked for a risk that isn't there, a small model
    invents one ("the biggest risk is the net sentiment of +53")."""
    parts = [name for name, present in (
        ("the biggest risk", facts["biggest_risk"] is not None),
        ("the largest complaint area", bool(facts.get("largest_complaint_area"))),
        ("the biggest win against the competitors", bool(facts["biggest_win"] and facts["biggest_win"]["beats_competitors"])),
        ("the top request", bool(facts["requests"]["count"])),
        ("the notable change", facts["rising_topic"] is not None),
    ) if present]
    if len(parts) < 3:
        parts.append("the overall volume and net sentiment")
    if len(parts) < 3:
        parts.append("what the quotes say")
    return ("You write a short briefing for a brand manager about public chatter on their brand. Write 3 to 5 "
            f"plain sentences covering, in this order: {', '.join(parts)}. " + RULES)


DIRECTION_SIGN = re.compile(r"\b(up|rose|risen|increased|grew|gained|down|fell|fallen|decreased|dropped|declined)"
                            r"( by)? ([-+−])(?=\d)", re.IGNORECASE)
UPWARD = {"up", "rose", "risen", "increased", "grew", "gained"}
NUMBER = re.compile(r"(?<![\w.])[-+−]?\d+(?:[.,]\d+)*")
NUMBER_WORDS = re.compile(
    r"\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
    r"seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|"
    r"million|dozen|half|twice|double[sd]?|triple[sd]?|tripled|quadrupled|threefold|twofold)\b", re.IGNORECASE)


# ---------------------------------------------------------------- facts


def _int(value: Any) -> int | None:
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else int(round(float(value)))


def build_facts(engine: Engine, brand_id: int, start: date, end: date, period_text: str) -> dict[str, Any] | None:
    """The numbers and quotes the briefing may use, or None when the brand has no mentions in the period."""
    from ui import data, insights

    brands = data.query_brands(engine)
    brand = brands[brands["id"] == brand_id].iloc[0]
    group = brands[brands["brand_group"] == brand["brand_group"]]
    comps = group[group["id"] != brand_id]
    ids = tuple(int(i) for i in group["id"])
    sources = tuple(data.query_sources_with_data(engine, ids))
    prev_start, prev_end = data.previous_period(start, end)
    current = data.query_mentions(engine, ids, start, end, sources)
    previous = data.query_mentions(engine, ids, prev_start, prev_end, sources)
    own, own_prev = current[current["brand_id"] == brand_id], previous[previous["brand_id"] == brand_id]
    if own.empty:
        return None
    k = data.compute_kpis(own, own_prev)
    profile = get_profile(brand["profile"])
    topics = data.query_topics(engine, brand_id)
    matrix = insights.topic_matrix(topics, settings.priority_volume_threshold, settings.priority_pain_threshold,
                                   settings.priority_praise_pain_max)
    table = insights.feature_table(data.query_feature_mentions(engine, ids), list(ids))

    risk = None
    if (pick := insights.top_risk(matrix)) is not None:
        kind, t = pick
        this, last = int(t["volume_7d"]), int(t["volume_prev_7d"])
        risk = {"topic": str(t["label"]), "category": t["category"], "priority": t["quadrant"],
                "complaints_last_7_days": this, "complaints_7_days_before": last,
                "times_last_week": round(this / last, 1) if last else None, "pain": round(float(t["pain"] or 0), 2),
                "urgent": kind != "watch"}
    themes = insights.topic_category_map(topics)
    area = insights.largest_area(
        insights.complaint_counts(own, profile, settings.topic_excluded_sources, themes),
        insights.complaint_counts(own_prev, profile, settings.topic_excluded_sources, themes), matrix)
    largest_area = None
    if area is not None and area.larger:  # only when it beats the top single theme: the roll-up says what themes can't
        largest_area = {"category": area.name, "complaints": area.this, "complaints_before": area.last,
                        "themes": area.themes, "biggest_single_theme_complaints": area.top_theme}
    win = None
    if (won := insights.biggest_win(table, brand_id, [int(c) for c in comps["id"]])) is not None:
        kind, feature, own_net, rival_net = won
        win = {"feature": insights.feature_name(feature), "net_sentiment": _int(own_net),
               "competitors_net_sentiment": _int(rival_net), "beats_competitors": kind == "beats"}
    total, feature, count = insights.top_request(current, brand_id, profile.request)
    rising = None
    if not matrix.empty:
        grown = matrix.assign(growth=matrix["volume_7d"] - matrix["volume_prev_7d"])
        grown = grown[(grown["growth"] >= 3) & (grown["volume_7d"] >= 2 * grown["volume_prev_7d"])]
        if not grown.empty and (risk is None or grown.sort_values("growth").iloc[-1]["label"] != risk["topic"]):
            g = grown.sort_values("growth").iloc[-1]
            rising = {"topic": str(g["label"]), "complaints_last_7_days": int(g["volume_7d"]),
                      "complaints_7_days_before": int(g["volume_prev_7d"])}
    quotes = []
    for sentiment, q in insights.pick_quotes(own).items():
        original = q.get("original") if isinstance(q.get("original"), str) else None
        quotes.append({"sentiment": sentiment, "text": " ".join(str(q["text"]).split())[:300],
                       "translated_from": (q.get("language") if original else None)})
    return {
        "brand": str(brand["name"]), "group": str(brand["brand_group"]),
        "industry": profile.name.replace("_", " "),
        "competitors": [str(c) for c in comps["name"]],
        "period": period_text,
        "mentions": k.mentions, "mentions_previous_period": int(len(own_prev)),
        "mentions_change_percent": _int(k.mentions_change_pct),
        "net_sentiment": _int(k.net_sentiment), "net_sentiment_change_points": _int(k.net_sentiment_change),
        "top_emotion": k.top_emotion,
        "biggest_risk": risk, "largest_complaint_area": largest_area, "biggest_win": win,
        "requests": {"label": profile.request, "count": total, "top_feature": insights.feature_name(feature) if feature else None,
                     "top_feature_count": count or None},
        "rising_topic": rising, "quotes": quotes,
    }


def _signed(value: int | None) -> str:
    return "n/a" if value is None else f"{value:+d}"


def _versus(now: int, before: int) -> str:
    if now == before:
        return f"{now}, the same as the 7 days before"
    return f"{now}, {'up' if now > before else 'down'} from {before} in the 7 days before"


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def fact_lines(facts: dict[str, Any]) -> str:
    """The facts as short ready-made lines with the numbers already formatted (small models copy these
    far more reliably than they read JSON)."""
    f = facts
    missing: list[str] = []
    lines = [f"Brand: {f['brand']} ({f['industry']}). Period: {f['period']}. "
             f"Competitors in the same group: {', '.join(f['competitors']) or 'none'}."]
    pct, before = f["mentions_change_percent"], f.get("mentions_previous_period")
    change = ("" if pct is None else ", the same as the previous period" if pct == 0 else
              f", {'up' if pct > 0 else 'down'} {abs(pct)}% from {before} in the previous period")
    lines.append(f"Mentions: {f['mentions']}{change}.")
    points = f["net_sentiment_change_points"]
    net_change = ("" if points is None else ", unchanged from the previous period" if points == 0 else
                  f", {'up' if points > 0 else 'down'} {abs(points)} points from the previous period")
    lines.append(f"Net sentiment: {_signed(f['net_sentiment'])}{net_change}.")
    r = f["biggest_risk"]
    if r is None:
        missing.append("an urgent risk")
    else:
        lead = "Biggest risk" if r["urgent"] else "No urgent risk; theme to watch"
        lines.append(f"{lead}: the theme \"{r['topic']}\" ({r['category'] or 'mixed'} complaints, priority {r['priority']}). "
                     f"Complaints in the last 7 days: {_versus(r['complaints_last_7_days'], r['complaints_7_days_before'])}.")
    a = f.get("largest_complaint_area")
    if a:
        lines.append(f"Largest complaint area: {a['category']}, spread across {a['themes']} themes. Complaints in "
                     f"the last 7 days: {_versus(a['complaints'], a['complaints_before'])}. The biggest single "
                     f"theme alone has {a['biggest_single_theme_complaints']}.")
    w = f["biggest_win"]
    if w is None:
        missing.append("a win against competitors")
    elif w["beats_competitors"]:
        lines.append(f"Biggest win against competitors: the feature {w['feature']}, where {f['brand']} has net "
                     f"sentiment {_signed(w['net_sentiment'])} and the competitors together {_signed(w['competitors_net_sentiment'])} "
                     "(one combined score, not any single competitor's).")
    else:  # a liked feature that competitors match is no win; small models call it one anyway
        missing.append("a win against competitors")
    q = f["requests"]
    if not q["count"]:
        missing.append(f"{q['label']}s")
    elif q["top_feature"]:
        lines.append(f"Top request: more or better {q['top_feature']} ({q['top_feature_count']} of the {q['count']} {q['label']}s).")
    else:
        lines.append(f"Requests: {_plural(q['count'], q['label'])}, no single topic stands out.")
    if not f["rising_topic"]:
        missing.append("a rising complaint theme")
    else:
        t = f["rising_topic"]
        lines.append(f"Notable change: the theme \"{t['topic']}\" is rising; complaints in the last 7 days: "
                     f"{_versus(t['complaints_last_7_days'], t['complaints_7_days_before'])}.")
    if missing:  # small models turn "risk: none" lines into nonsense ("the rising trend of no complaints")
        lines.append(f"Nothing to report (leave out entirely): {', '.join(missing)}.")
    for quote in f["quotes"]:
        source = f" (translated from {quote['translated_from']})" if quote["translated_from"] else ""
        lines.append(f"Quote, {quote['sentiment']}{source}: \"{quote['text']}\"")
    return "\n".join(lines)


WORD_DIGITS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split()) if i >= 2}


def digits_for_words(text: str) -> str:
    """"four complaints" -> "4 complaints": an exact rewrite, so the number check still sees every number."""
    return re.sub(r"\b(" + "|".join(WORD_DIGITS) + r")\b", lambda m: WORD_DIGITS[m.group(1).lower()], text,
                  flags=re.IGNORECASE)


def drop_redundant_signs(text: str) -> str:
    """"decreased by -5" -> "decreased by 5": an exact rewrite when the word and the sign agree. A clash
    ("increased by -5") is left alone, so check_briefing rejects it."""
    def fix(m: re.Match[str]) -> str:
        up = m.group(1).lower() in UPWARD
        agrees = (m.group(3) == "+") if up else (m.group(3) in "-−")
        return f"{m.group(1)}{m.group(2) or ''} " if agrees else m.group(0)

    return DIRECTION_SIGN.sub(fix, text)


def tidy(text: str) -> str:
    t = " ".join(str(text).split()).strip()
    t = re.sub(r"^(?:\*\*)?briefing:?(?:\*\*)?\s*", "", t, flags=re.IGNORECASE)  # "Briefing: ..." preamble
    if len(t) > 1 and t[0] == t[-1] == '"':
        t = t[1:-1].strip()
    return drop_redundant_signs(digits_for_words(t))


def facts_hash(facts: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(facts, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


# ---------------------------------------------------------------- grounding


def _numbers(text: str) -> list[tuple[float, bool]]:
    """(value, has an explicit sign) for every number in the text; a minus sign makes the value negative."""
    out = []
    for raw in NUMBER.findall(text):
        signed = raw[0] in "+-−"
        cleaned = raw.replace("−", "-").replace(",", "")
        try:
            out.append((float(cleaned), signed))
        except ValueError:
            continue
    return out


def allowed_numbers(facts: Any) -> set[float]:
    """Every number in the facts (signed), from values and from inside strings (labels, quotes, the period)."""
    found: set[float] = set()
    if isinstance(facts, dict):
        for value in facts.values():
            found |= allowed_numbers(value)
    elif isinstance(facts, list):
        for value in facts:
            found |= allowed_numbers(value)
    elif isinstance(facts, bool) or facts is None:
        pass
    elif isinstance(facts, (int, float)):
        found.add(float(facts))
    elif isinstance(facts, str):
        found |= {value for value, _ in _numbers(facts)}
    return found


def unknown_numbers(text: str, facts: dict[str, Any]) -> list[str]:
    """Numbers in the text that the facts don't contain. A signed number (+39, -14) must match with its sign;
    an unsigned one may match either sign (\"14 points\" for a change of -14)."""
    allowed = allowed_numbers(facts)
    try:  # the model sees the fact lines too ("in the last 7 days"): their numbers are fair to repeat
        allowed |= {value for value, _ in _numbers(fact_lines(facts))}
    except (KeyError, TypeError):  # a partial facts dict (tests, older rows)
        pass
    magnitudes = {abs(v) for v in allowed}
    bad = []
    for value, signed in _numbers(text):
        ok = (value in allowed) if signed else (abs(value) in magnitudes)
        if not ok:
            bad.append(f"{value:+g}" if signed else f"{value:g}")
    return bad


def check_briefing(text: str, facts: dict[str, Any]) -> str | None:
    """Why the briefing can't be used, or None."""
    t = " ".join(text.split())
    if not 120 <= len(t) <= 1100:
        return f"length {len(t)}"
    sentences = len(re.findall(r"[.!?](?:\s|$)", t))
    if not 3 <= sentences <= 6:
        return f"{sentences} sentences"
    if re.search(r"^\s*[-*#•]|\*\*", text, re.MULTILINE):
        return "markdown or bullets"
    if (both := re.search(r"\b(?:up|down|rose|fell|increased|decreased|dropped)(?: by)? [-+−]\d", t, re.IGNORECASE)):
        return f"direction word and sign together ({both.group(0)!r})"
    if (word := NUMBER_WORDS.search(t)):
        return f"number written as a word ({word.group(0)!r})"
    unknown = unknown_numbers(t, facts)
    if unknown:
        return f"numbers not in the facts: {', '.join(unknown)}"
    return None


# ---------------------------------------------------------------- generate and store


def write_briefing(provider: Provider, facts: dict[str, Any], run: StageRun | None = None) -> tuple[str, str | None]:
    """(text, problem): one attempt, plus one retry that names what was wrong."""
    prompt = "Facts:\n" + fact_lines(facts)
    briefing, problem = "", None
    for attempt in range(2):
        result = provider.text(system_prompt(facts), prompt, max_tokens=500)
        if run is not None:
            run.add(result)
        briefing = tidy(result.data.get("text", ""))
        problem = check_briefing(briefing, facts)
        if problem is None:
            return briefing, None
        prompt = ("Facts:\n" + fact_lines(facts) + f"\n\nYour previous briefing was rejected ({problem}). "
                  "Write it again using only the facts above, copying every number exactly.")
    return briefing, problem


def periods(today: date) -> list[tuple[str, str, date, date, str]]:
    """(period, label, start, end, text) for the dashboard's last 7 days and the last complete ISO week."""
    week = Week.last_complete(today)
    start7 = today - timedelta(days=6)
    return [("last7", "last7", start7, today, f"the last 7 days ({start7.day} {start7:%b}–{today.day} {today:%b %Y})"),
            ("week", week.label, week.start, week.end, f"the week of {week.span}")]


def refresh_briefings(factory: sessionmaker[Session], provider: Provider, brands: list[int] | None = None,
                      today: date | None = None, force: bool = False, record: bool = True) -> StageRun:
    engine: Engine = factory.kw["bind"]
    today = today or utc_now().date()
    run = StageRun.start("briefings", provider)
    available, down = True, ""
    with factory() as session:
        brand_ids = [b for b in session.scalars(select(Brand.id).order_by(Brand.id)) if brands is None or b in brands]
    for brand_id in brand_ids:
        for period, label, start, end, text in periods(today):
            run.items += 1
            with factory() as session:
                row = session.scalar(select(Briefing).where(Briefing.brand_id == brand_id, Briefing.period == period,
                                                            Briefing.period_label == label))
                if period == "week" and row is not None and not force:  # a finished week is written once
                    run.cached += 1
                    continue
                facts = build_facts(engine, brand_id, start, end, text)
                if facts is None:  # nothing to write about: the dashboard shows the rule-based summary
                    if row is not None:
                        session.delete(row)
                        session.commit()
                    continue
                digest = facts_hash(facts)
                if row is not None and row.facts_hash == digest and not force:
                    run.cached += 1
                    continue
                problem = None
                if not available:
                    problem = down
                else:
                    try:
                        briefing, problem = write_briefing(provider, facts, run)
                    except LLMUnavailable as exc:
                        available, problem = False, str(exc)
                        down = problem
                    except LLMError as exc:
                        problem = str(exc)[:200]
                if problem:
                    run.fallback(f"briefing rejected: {problem}" if available else problem)
                    if row is not None:  # its numbers are out of date now: show the rule-based summary instead
                        session.delete(row)
                        session.commit()
                    continue
                values = {"start_date": start, "end_date": end, "text": briefing, "facts": facts, "facts_hash": digest,
                          "provider": provider.name, "model": provider.model, "created_at": utc_now()}
                if row is None:
                    session.add(Briefing(brand_id=brand_id, period=period, period_label=label, **values))
                else:
                    for key, value in values.items():
                        setattr(row, key, value)
                session.commit()
                run.generated += 1
    provider.unload()
    return run.finish(factory if record else None)


def stored(session: Session, brand_id: int, period: str, label: str) -> Briefing | None:
    return session.scalar(select(Briefing).where(Briefing.brand_id == brand_id, Briefing.period == period,
                                                 Briefing.period_label == label))


def clear(session: Session, brand_ids: list[int] | None = None) -> int:
    stmt = delete(Briefing)
    if brand_ids is not None:
        stmt = stmt.where(Briefing.brand_id.in_(brand_ids))
    return int(session.execute(stmt).rowcount or 0)

