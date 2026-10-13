"""Build the demo database: realistic, fictional data for the 4 brands, kept apart from real data.

    python seed_demo.py                      # (re)builds brand_radar_demo
    BRAND_RADAR_DEMO=1 streamlit run app.py   # dashboard on the demo data ("Demo data" pill in the header)

Safe to re-run: the demo database is dropped and rebuilt from db/schema.sql each time. It refuses
to touch the real database (DB_NAME). Mentions, analysis rows and topic assignments are generated;
brands, topic_daily, priority snapshots and alerts are produced by the real code
(manage_brands.sync_brands, nlp.topics.rebuild_topic_daily, nlp.priority.run_priority).
No real people: authors are generated handles, texts are written from templates.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy import Engine, create_engine, func, select, text
from sqlalchemy.orm import Session, sessionmaker

from config import PROJECT_ROOT
from db.models import Alert, Analysis, Brand, CollectorRun, FetchRun, Mention, Topic, TopicDaily
from manage_brands import load_yaml, sync_brands
from nlp.models import EMOTIONS
from nlp.priority import run_priority
from nlp.topics import rebuild_topic_daily
from sources.base import utc_now

DAYS = 60
VOLUME = 1.4  # scales every theme's daily rate (not the spikes)
MODEL_VERSION = "demo"
SCHEMA_FILE = PROJECT_ROOT / "db" / "schema.sql"

# ---------------------------------------------------------------- content


@dataclass
class Theme:
    """A recurring thing people say about a brand. Complaint themes become topics."""

    label: str | None  # topic label; None = not clustered (praise, discussion)
    category: str
    feature: str | None
    sentiment: str  # negative / neutral / positive
    per_day: float  # average mentions per day over the 60 days
    texts: list[str]
    anger: float = 0.35  # how much anger+disgust the complaints carry (drives "pain")
    spike: int = 0  # extra mentions in the last 7 days (with a quiet week before) -> spike alert
    sources: tuple[str, ...] = ("appstore", "playstore", "hackernews")


THEMES: dict[str, list[Theme]] = {
    "Notion": [
        Theme("AI Credit Limits", "pricing", "AI", "negative", 0.55, spike=9, anger=0.42, texts=[
            "I hit the AI limit halfway through my lecture notes and now I have to pay for another plan.",
            "The AI features ran out after a few summaries. Charging extra on top of Plus feels greedy.",
            "Why is AI a separate add-on now? I already pay for the workspace every month.",
            "Ran out of AI responses on day three of the month. Please make the limits clearer before I upgrade.",
        ]),
        Theme("Offline Mode Gaps", "bug", "offline", "negative", 0.45, anger=0.30, texts=[
            "On the train with no signal and my pages would not open at all. Offline mode needs work.",
            "Edits I made offline disappeared once I reconnected. That is a scary bug for a notes app.",
            "Still can't rely on it without internet. I keep a separate notes app just for flights.",
        ]),
        Theme("Slow Large Databases", "performance", "databases", "negative", 0.5, anger=0.28, texts=[
            "Our project database has a few thousand rows and filtering it takes several seconds now.",
            "Large databases are painfully slow to load on my older laptop, scrolling stutters badly.",
            "Linked views of the big task database freeze for a moment every time I switch tabs.",
        ]),
        Theme("Mobile Editor Lag", "UX/design", "editor", "negative", 0.3, anger=0.18, texts=[
            "Typing on the Android app lags behind my keyboard and the cursor jumps around.",
            "The mobile editor is fiddly, moving blocks around on a phone is close to impossible.",
        ]),
        Theme(None, "praise", "templates", "positive", 0.9, texts=[
            "The templates saved me hours setting up my study planner. Love how flexible it is.",
            "I run my whole freelance business from one workspace now, the templates are excellent.",
            "Beautiful app, my reading list and journal finally live in the same place.",
        ]),
        Theme(None, "praise", "collaboration", "positive", 0.55, texts=[
            "Sharing pages with my team and commenting together works really smoothly.",
            "Real-time collaboration on our product docs is the reason our team switched.",
        ]),
        Theme(None, "praise", "databases", "positive", 0.4, texts=[
            "Databases with relations and rollups replaced three spreadsheets for me.",
            "The way databases link together is brilliant for planning content calendars.",
        ]),
        Theme(None, "feature request", "offline", "neutral", 0.5, texts=[
            "Please add a proper offline mode so I can pick which pages are always available.",
            "Would love an option to download a whole workspace for offline use before travelling.",
        ]),
        Theme(None, "general discussion", None, "neutral", 0.35, sources=("hackernews",), texts=[
            "We moved our internal wiki to it last year; search is fine but permissions took some planning.",
            "Interesting how many tools now copy the block-based editor approach it popularised.",
        ]),
    ],
    "Obsidian": [
        Theme("Sync Conflicts", "bug", "sync", "negative", 0.4, spike=8, anger=0.40, texts=[
            "Sync created duplicate copies of my daily note again, now I have conflict files everywhere.",
            "Edits made on my phone overwrote changes from my laptop. Sync conflicts are getting worse.",
            "After the update sync keeps showing errors and my vault on the iPad is out of date.",
        ]),
        Theme("Plugins Breaking After Updates", "bug", "plugins", "negative", 0.35, anger=0.30, texts=[
            "Every major update breaks one of my community plugins and I lose half a day fixing it.",
            "A plugin update corrupted the formatting in a few of my notes. Needs better testing.",
        ]),
        Theme("Steep Learning Curve", "UX/design", "editor", "negative", 0.3, anger=0.12, texts=[
            "Took me weeks to understand links, vaults and plugins. The onboarding could be much friendlier.",
            "Powerful but confusing for beginners, the settings screen alone is overwhelming.",
        ]),
        Theme(None, "praise", "plugins", "positive", 0.7, texts=[
            "The plugin ecosystem is incredible, I built exactly the workflow I wanted.",
            "Community plugins turned it into my task manager, calendar and reading list.",
        ]),
        Theme(None, "praise", "offline", "positive", 0.5, texts=[
            "Love that my notes are plain files on my own device and work offline everywhere.",
            "Local files and offline access mean I never worry about losing my notes.",
        ]),
        Theme(None, "praise", "editor", "positive", 0.45, texts=[
            "The markdown editor is fast and clean, writing long notes feels effortless.",
            "Live preview editing is lovely and it opens instantly even with a big vault.",
        ]),
        Theme(None, "feature request", "collaboration", "neutral", 0.35, texts=[
            "Real-time collaboration on shared vaults would make it perfect for our small team.",
            "Please add a simple way to share a single note with someone who doesn't use the app.",
        ]),
    ],
    "Evernote": [
        Theme("Price Rises Driving Switch", "pricing", "pricing", "negative", 1.1, anger=0.48, texts=[
            "The subscription price doubled again. After ten years I'm moving my notes elsewhere.",
            "Too expensive now for what it does. The price increase made me export everything.",
            "Paying this much for a notes app makes no sense anymore, cancelling my plan.",
            "Long-time user, the new pricing is the final straw. Switching to something simpler.",
        ]),
        Theme("Video Ads on Launch", "pricing", "pricing", "negative", 0.6, anger=0.52, texts=[
            "A full screen video ad plays every time I open the app just to write a quick note.",
            "Loud ads pushing the upgrade before I can even see my notebooks. Uninstalling.",
            "The unskippable promotions on launch make the free version almost unusable.",
        ]),
        Theme("Free Plan Device Limit", "pricing", "sync", "negative", 0.45, anger=0.38, texts=[
            "The free plan now only syncs to one device, so my phone and laptop are out of step.",
            "Limiting sync on the free version to a single device pushed me to look for alternatives.",
        ]),
        Theme("Search Got Slower", "performance", "search", "negative", 0.35, anger=0.20, texts=[
            "Search used to be the best feature, now it takes ages and misses notes I know exist.",
            "Finding anything in my older notebooks is slow since the redesign.",
        ]),
        Theme(None, "praise", "web clipper", "positive", 0.3, texts=[
            "The web clipper is still the best way I've found to save articles for later.",
            "Clipping recipes and receipts straight into notebooks still works great.",
        ]),
        Theme(None, "praise", "search", "positive", 0.2, texts=[
            "Searching text inside scanned documents and photos is still really impressive.",
        ]),
        Theme(None, "feature request", "pricing", "neutral", 0.3, texts=[
            "Please bring back a cheaper plan for people who only need basic notes and sync.",
        ]),
    ],
    "ClickUp": [
        Theme("Slow Task Loading", "performance", "task management", "negative", 0.35, spike=7, anger=0.34, texts=[
            "Opening a list with a few hundred tasks takes forever and the whole page freezes.",
            "The board view is so slow lately, our standups now start with everyone waiting for it to load.",
            "Task pages spin for several seconds before they show anything. Performance needs attention.",
        ]),
        Theme("Notification Overload", "UX/design", "notifications", "negative", 0.4, anger=0.25, texts=[
            "I get dozens of notifications for every small change and can't find the important ones.",
            "The notification settings are so complicated that the team just ignores all of them.",
        ]),
        Theme("Complex Setup", "UX/design", "task management", "negative", 0.3, anger=0.14, texts=[
            "There are so many settings that setting up a simple project took our team a whole week.",
            "Powerful, but the number of views and options makes it overwhelming for new teammates.",
        ]),
        Theme(None, "praise", "task management", "positive", 0.6, texts=[
            "Having tasks, docs and goals in one place has really simplified how our team works.",
            "Custom statuses and automations save our support team a lot of manual work.",
        ]),
        Theme(None, "praise", "integrations", "positive", 0.35, texts=[
            "The integrations with our chat and calendar tools work well and save time every day.",
            "Connecting it to our code repository keeps tasks and pull requests nicely in sync.",
        ]),
        Theme(None, "feature request", "integrations", "neutral", 0.3, texts=[
            "Would love a native integration with our invoicing tool so billable time flows across.",
        ]),
    ],
}

ADJECTIVES = ["quiet", "sunny", "brisk", "mellow", "clever", "rapid", "gentle", "lucky", "amber", "misty", "plucky", "tidy"]
NOUNS = ["fox", "otter", "heron", "maple", "comet", "pebble", "lark", "fern", "harbor", "willow", "finch", "ember"]


# ---------------------------------------------------------------- generation


@dataclass
class SeedSummary:
    brands: int = 0
    mentions: int = 0
    topics: int = 0
    alerts: int = 0
    per_brand: dict[str, int] = field(default_factory=dict)


def _handle(rng: random.Random) -> str:
    return f"{rng.choice(ADJECTIVES)}{rng.choice(NOUNS)}{rng.randint(2, 99)}"


def _emotions(rng: random.Random, sentiment: str, anger: float) -> dict[str, float]:
    if sentiment == "negative":
        weights = {"anger": anger * 0.6, "disgust": anger * 0.4, "sadness": 0.22, "fear": 0.06,
                   "surprise": 0.06, "joy": 0.02, "neutral": max(0.05, 0.6 - anger)}
    elif sentiment == "positive":
        weights = {"joy": 0.62, "surprise": 0.08, "neutral": 0.22, "sadness": 0.03, "anger": 0.02,
                   "disgust": 0.01, "fear": 0.02}
    else:
        weights = {"neutral": 0.7, "surprise": 0.08, "joy": 0.08, "sadness": 0.06, "anger": 0.03,
                   "disgust": 0.02, "fear": 0.03}
    noisy = {e: max(0.001, weights[e] * rng.uniform(0.75, 1.25)) for e in EMOTIONS}
    total = sum(noisy.values())
    return {e: round(v / total, 4) for e, v in noisy.items()}


def _score(rng: random.Random, sentiment: str) -> float:
    low, high = {"negative": (-0.95, -0.35), "neutral": (-0.2, 0.2), "positive": (0.4, 0.95)}[sentiment]
    return round(rng.uniform(low, high), 4)


def _rating(rng: random.Random, sentiment: str) -> int:
    return rng.choice({"negative": [1, 1, 2], "neutral": [3, 3, 4], "positive": [4, 5, 5]}[sentiment])


def _poisson(rng: random.Random, mean: float) -> int:
    # Knuth; means here are < 3, so this is fast
    limit, k, p = 2.718281828 ** -mean, 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


def seed(engine: Engine, now: datetime | None = None, seed_value: int = 7) -> SeedSummary:
    """Fill an empty database (same schema as the real one) with the demo dataset."""
    rng = random.Random(seed_value)
    now = (now or utc_now()).replace(microsecond=0)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    summary = SeedSummary()

    with factory() as session:
        sync_brands(session, load_yaml(PROJECT_ROOT / "brands.yaml"))
        session.commit()
        brands = {b.name: b for b in session.scalars(select(Brand))}
    summary.brands = len(brands)

    with factory() as session:
        n = 0
        for brand_name, themes in THEMES.items():
            brand = brands[brand_name]
            count = 0
            for theme in themes:
                topic_id = None
                if theme.label:
                    topic = Topic(brand_id=brand.id, label=theme.label, category=theme.category,
                                  keywords=[w.lower() for w in theme.label.split()])
                    session.add(topic)
                    session.flush()
                    topic_id = topic.id
                for day in range(DAYS):
                    mean = theme.per_day * VOLUME
                    if theme.spike and day < 7:
                        mean += theme.spike / 7  # spike: this week ...
                    elif theme.spike and day < 14:
                        mean *= 0.35  # ... after a quiet week, so the ratio clears 2x
                    for _ in range(_poisson(rng, mean)):
                        source = rng.choice(theme.sources)
                        created = now - timedelta(days=day, hours=rng.uniform(0.5, 23.5))
                        text_ = rng.choice(theme.texts)
                        n += 1
                        count += 1
                        mention = Mention(
                            brand_id=brand.id, source=source, external_id=f"demo-{n}", author=_handle(rng),
                            text=text_, url=None, created_at=created, fetched_at=created + timedelta(minutes=30),
                            rating=_rating(rng, theme.sentiment) if source != "hackernews" else None,
                        )
                        session.add(mention)
                        session.flush()
                        session.add(Analysis(
                            mention_id=mention.id, sentiment=theme.sentiment, sentiment_score=_score(rng, theme.sentiment),
                            emotion=None, emotion_scores=_emotions(rng, theme.sentiment, theme.anger),
                            category=theme.category, feature=theme.feature if rng.random() < 0.85 else None,
                            topic_id=topic_id, model_version=MODEL_VERSION,
                        ))
            summary.per_brand[brand_name] = count
        # top emotion per mention, from its scores
        for analysis in session.scalars(select(Analysis)):
            analysis.emotion = max(analysis.emotion_scores, key=analysis.emotion_scores.get)
        # topic first/last seen
        for topic in session.scalars(select(Topic)):
            dates = session.execute(
                select(func.min(Mention.created_at), func.max(Mention.created_at))
                .join(Analysis, Analysis.mention_id == Mention.id).where(Analysis.topic_id == topic.id)
            ).one()
            topic.first_seen, topic.last_seen = dates
        for brand in brands.values():
            rebuild_topic_daily(session, brand.id)
        # a few fetch runs and a recent successful collector cycle, so the header status has something to show
        for brand in brands.values():
            for source in ("hackernews", "appstore", "playstore"):
                session.add(FetchRun(source=source, brand_id=brand.id, started_at=now - timedelta(minutes=9),
                                     finished_at=now - timedelta(minutes=8), new_rows=rng.randint(0, 6)))
        session.add(CollectorRun(started_at=now - timedelta(minutes=9), finished_at=now - timedelta(minutes=4),
                                 status="ok", new_mentions=14, analysed=14, topics=sum(1 for t in THEMES for th in THEMES[t] if th.label),
                                 alerts_created=0, stage_seconds={"fetch": 41.0, "nlp": 162.0, "topics": 31.0, "priority": 0.4}))
        session.commit()
        summary.mentions = n

    _, alerts = run_priority(factory, as_of=now)
    with factory() as session:
        summary.topics = session.scalar(select(func.count()).select_from(Topic)) or 0
        summary.alerts = session.scalar(select(func.count()).select_from(Alert)) or 0
    return summary


# ---------------------------------------------------------------- MySQL database (re)build


def schema_statements(db_name: str) -> list[str]:
    """db/schema.sql with the database name swapped, split into statements."""
    sql = SCHEMA_FILE.read_text(encoding="utf-8")
    sql = re.sub(r"(?m)^CREATE DATABASE IF NOT EXISTS brand_radar\b", f"CREATE DATABASE IF NOT EXISTS {db_name}", sql)
    sql = re.sub(r"(?m)^USE brand_radar;", f"USE {db_name};", sql)
    sql = "\n".join(line for line in sql.splitlines() if not line.strip().startswith("--"))
    return [s.strip() for s in sql.split(";") if s.strip()]


def rebuild_database(server_url: str, db_name: str) -> None:
    engine = create_engine(server_url, future=True)
    with engine.begin() as conn:
        conn.execute(text(f"DROP DATABASE IF EXISTS `{db_name}`"))
        for statement in schema_statements(db_name):
            conn.execute(text(statement))
    engine.dispose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args(argv)
    from urllib.parse import quote_plus

    from config import settings

    demo_db = os.getenv("DEMO_DB_NAME", "brand_radar_demo")
    real_db = os.getenv("DB_NAME", "brand_radar")
    if not re.fullmatch(r"[A-Za-z0-9_]+", demo_db):
        print(f"✘ invalid DEMO_DB_NAME {demo_db!r}")
        return 1
    if demo_db == real_db:
        print(f"✘ DEMO_DB_NAME is the real database ({real_db}); refusing to overwrite it.")
        return 1

    server = (f"mysql+pymysql://{quote_plus(settings.db_user)}:{quote_plus(settings.db_password)}"
              f"@{settings.db_host}:{settings.db_port}/?charset=utf8mb4")
    try:
        rebuild_database(server, demo_db)
    except Exception as exc:  # noqa: BLE001
        print(f"✘ could not (re)create {demo_db}: {exc}\n"
              f"  Grant the radar user access once, with XAMPP's client:\n"
              f"  /Applications/XAMPP/xamppfiles/bin/mysql -u root -e "
              f"\"GRANT ALL PRIVILEGES ON {demo_db}.* TO 'radar'@'localhost', 'radar'@'127.0.0.1'\"")
        return 1
    engine = create_engine(server.replace("/?charset", f"/{demo_db}?charset"), future=True)
    summary = seed(engine)
    engine.dispose()
    print(f"✔ rebuilt {demo_db}: {summary.brands} brands, {summary.mentions} mentions "
          f"({', '.join(f'{k} {v}' for k, v in summary.per_brand.items())}), {summary.topics} topics, {summary.alerts} alerts")
    print("  Run the dashboard on it:  BRAND_RADAR_DEMO=1 streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
