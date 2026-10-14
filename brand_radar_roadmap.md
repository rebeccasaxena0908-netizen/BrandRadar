# Brand Radar

Automated brand sentiment and competitor feature radar. Pulls public chatter about your product and its
competitors, runs local transformer models to find sentiment, emotions and feature requests, and ranks
complaints in a priority matrix on a glassmorphism Streamlit dashboard. Everything is stored in a MySQL
(MariaDB) database running in XAMPP.

**Current version:** Step 8 – social sources (Reddit ready but optional; Mastodon and dev.to live). Build it step by step with Claude Code (see the bottom of this file).

---

## How it works

```
 Sources                 Collector (scheduled)        Database (XAMPP MySQL)        Dashboard (Streamlit)
 ───────                 ─────────────────────        ──────────────────────        ─────────────────────
 Hacker News  ──┐                                     brands / competitors          KPI glass cards
 App Store    ──┼──►  fetch → clean → dedupe  ──►     mentions (raw text)    ──►    emotion wheel
 Google Play  ──┤         │                           analysis (scores)             word clouds
 Reddit*      ──┤         ▼                           topics / clusters             priority matrix
 CSV import   ──┘     NLP pipeline (local models)     alerts / fetch_runs           alert badges + summary feed
                      sentiment · emotion · category
                      topic clustering · priority score
```

The collector and the dashboard are two separate processes. The collector writes to MySQL on a schedule;
the dashboard only reads from MySQL and refreshes itself. That keeps the UI fast and stops a slow API
call or model load from freezing the page.

---

## Data sources (read this before you start)

| Source | How | Key needed | Notes |
|---|---|---|---|
| Hacker News | Firebase API for items + Algolia HN Search API (`hn.algolia.com/api/v1/search`) for keyword search | No | Free and reliable. Algolia is what makes brand keyword search possible; Firebase alone has no search. |
| App Store | iTunes customer reviews RSS JSON feed (`itunes.apple.com/{country}/rss/customerreviews/id={app_id}/sortBy=mostRecent/json`) | No | Returns recent reviews only (a few hundred per country). Loop over countries (`us`, `in`, `gb` …) for more. |
| Google Play | `google-play-scraper` Python package | No | Unofficial scraper; can break if Google changes the page. Wrap it in try/except. |
| Reddit | PRAW with an approved OAuth app | Yes, and approval | **Risky.** Reddit closed self-service API access in Nov 2025, stops accepting new access requests on **31 Oct 2026**, and plans to end remaining public API access in **March 2027**. Build Reddit as an optional plug-in, never as a required source. |
| Twitter/X | – | Paid | Left out: the X API has no useful free tier. Can be added later through the CSV importer. |
| CSV import | Upload a CSV (`source, author, text, date, rating`) in the dashboard | No | Lets you load exports from any tool, and gives you demo data for presentations. |

Every source implements the same small interface, so adding or removing one never touches the rest of the code:

```python
class Source(Protocol):
    name: str
    def fetch(self, brand: Brand, since: datetime) -> list[RawMention]: ...
```

---

## Local AI models (Hugging Face, run on your machine)

| Job | Model | Output |
|---|---|---|
| Sentiment | `cardiffnlp/twitter-roberta-base-sentiment-latest` | negative / neutral / positive + score |
| Emotion | `j-hartmann/emotion-english-distilroberta-base` | anger, disgust, fear, joy, neutral, sadness, surprise (feeds the emotion wheel) |
| Category | `facebook/bart-large-mnli` zero-shot | bug, feature request, pricing, performance, UX/design, support, praise, general discussion (catch-all for neutral talk) |
| Feature / aspect | keyword gate on each brand's feature synonyms in `brands.yaml`, then zero-shot confirms and picks among the matches (zero-shot alone matched almost everything) | which feature each mention talks about, or none |
| Topic clustering | `BERTopic` + `sentence-transformers/all-MiniLM-L6-v2`, KeyBERTInspired + MMR keyphrases | groups of similar complaints with Title Case labels ("Unskippable Ads"); manual overrides in `topic_labels.yaml` |

Models download once (about 2 GB total) into the Hugging Face cache. Mentions are analysed in batches
and each row is analysed only once (`analysis` has a foreign key to `mentions`).

**Priority score** for each complaint topic (shown in the 2×2 matrix):

- **Volume** (x-axis): number of mentions in the last 7 days, log-scaled
- **Pain** (y-axis): average negativity × average P(anger)+P(disgust) (a probability share, so small topics don't score 0)
- **Trend**: this week's volume ÷ last week's volume; above 2× (with at least 5 mentions) raises an alert badge
- Quadrants: *Fix now* (high volume, high pain), *Quick wins* (low volume, high pain), *Monitor* (high volume, low pain), *Ignore*;
  thresholds `PRIORITY_VOLUME_THRESHOLD` / `PRIORITY_PAIN_THRESHOLD`; snapshot stored on `topics`

---

## Database (XAMPP)

1. Install **XAMPP** from apachefriends.org (Windows or macOS) and start **MySQL** (and **Apache** for phpMyAdmin) in the XAMPP control panel.
2. Open `http://localhost/phpmyadmin`, go to **SQL**, and run `db/schema.sql` (Claude Code writes this in Step 1).
3. Create a dedicated user instead of using `root` with no password:

   ```sql
   CREATE USER 'radar'@'localhost' IDENTIFIED BY 'choose-a-password';
   GRANT ALL PRIVILEGES ON brand_radar.* TO 'radar'@'localhost';
   ```

4. Put the details in `.env` (copied from `.env.example`). Never commit `.env`.

Python talks to MySQL through **SQLAlchemy + PyMySQL** (`mysql+pymysql://radar:...@localhost:3306/brand_radar`).

### Tables

| Table | Key columns |
|---|---|
| `brands` | `id`, `name`, `is_own` (your product vs competitor), `keywords` (JSON), `app_store_id`, `play_store_id`, `features` (JSON) |
| `mentions` | `id`, `brand_id`, `source`, `external_id`, `author`, `text`, `url`, `rating`, `created_at`, `fetched_at` — **unique (`brand_id`, `source`, `external_id`)** so re-fetching never duplicates, while one post that mentions two brands counts for both |
| `analysis` | `mention_id`, `sentiment`, `sentiment_score`, `emotion`, `emotion_scores` (JSON), `category`, `feature`, `topic_id`, `model_version` |
| `topics` | `id`, `brand_id`, `label`, `keywords`, `first_seen`, `last_seen` |
| `topic_daily` | `topic_id`, `day`, `mentions`, `avg_sentiment` (precomputed for fast charts) |
| `alerts` | `id`, `brand_id`, `topic_id`, `kind`, `message`, `created_at`, `seen` |
| `fetch_runs` | `id`, `source`, `brand_id`, `started_at`, `finished_at`, `new_rows`, `error` |

Use `utf8mb4` for all text so emoji in reviews don't break inserts.

---

## The dashboard

Streamlit with custom CSS injected through `st.markdown(..., unsafe_allow_html=True)`:

- **Glassmorphism look:** dark gradient background, cards with `backdrop-filter: blur(16px)`, translucent white borders, soft shadows
- **Header:** brand selector, competitor multiselect, date range, source filter
- **KPI cards:** net sentiment, mentions this week, % change, top emotion
- **Emotion wheel:** Plotly `barpolar` with one segment per emotion, own brand vs competitor overlaid
- **Sentiment trend:** daily line per brand
- **Word clouds:** separate clouds for complaints and for praise (`wordcloud` package, rendered to an image)
- **Priority matrix:** Plotly scatter of topics in the four quadrants, bubble size = volume, hover shows example quotes
- **Feature radar:** radar chart of sentiment per feature, your brand vs each competitor ("they win on sync, we win on price")
- **Alert badges:** pulsing red pills for spiking complaint topics
- **Executive summary feed:** short bullet summary of the week built from the numbers (biggest risk, biggest win, top requested feature) plus three representative quotes
- **Live updates:** wrap the live panels in `@st.fragment(run_every="60s")` so they refresh without reloading the page
- Cache database queries with `st.cache_data(ttl=60)` and model loading with `st.cache_resource`

---

## Project files (target layout)

| File | What it does |
|---|---|
| `config.py` | Settings loaded from `.env` (DB URL, fetch interval, thresholds) |
| `db/schema.sql` | Creates the database and all tables |
| `db/models.py` | SQLAlchemy models matching the schema |
| `db/session.py` | Engine and session helper |
| `sources/base.py` | `Source` interface and `RawMention` dataclass |
| `sources/hackernews.py` | Algolia search + Firebase item details |
| `sources/appstore.py` | iTunes review feed |
| `sources/playstore.py` | `google-play-scraper` wrapper |
| `sources/reddit.py` | Optional PRAW source, skipped if no credentials |
| `sources/csv_import.py` | CSV importer |
| `nlp/pipeline.py` | Cleans text, batches it through the models, writes `analysis` rows |
| `nlp/topics.py` | BERTopic clustering and `topic_daily` aggregation |
| `nlp/priority.py` | Priority scores, quadrants and alerts |
| `collector.py` | Scheduler (APScheduler): fetch → analyse → aggregate every N minutes |
| `app.py` | Streamlit dashboard entry point |
| `ui/styles.css` | Glassmorphism styles |
| `ui/components.py` | Glass card, badge, KPI and chart helpers |
| `seed_demo.py` | Fills the DB with realistic demo data so the UI works with no API keys |
| `check_setup.py` | Checks Python, packages, MySQL connection and model downloads |
| `tests/` | pytest tests for sources (mocked HTTP), pipeline and priority maths |

---

## Setup (once the code exists)

```bash
python3.11 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # then fill in DB password and brand settings
python check_setup.py
python seed_demo.py               # optional demo data
python collector.py --once        # one fetch + analysis run
streamlit run app.py
```

Main packages: `streamlit`, `plotly`, `wordcloud`, `sqlalchemy`, `pymysql`, `python-dotenv`, `requests`,
`google-play-scraper`, `praw`, `transformers`, `torch`, `sentence-transformers`, `bertopic`, `apscheduler`, `pandas`, `pytest`.

---

## Roadmap

| Step | Feature | Status |
|---|---|---|
| 1 | Project skeleton, `.env`, XAMPP MySQL schema, SQLAlchemy models, `check_setup.py` | Done |
| 2 | Sources: Hacker News, App Store, Google Play, CSV import, dedupe, `fetch_runs` logging | Done |
| 3 | NLP pipeline: sentiment, emotion, zero-shot category and feature, batch processing | Done |
| 4 | Topic clustering, daily aggregates, priority score, trend alerts | Done |
| 5 | Streamlit dashboard: glass CSS, KPI cards, emotion wheel, trend line, word clouds | Done |
| 6 | Priority matrix, feature radar vs competitors, alert badges, executive summary feed | Done |
| 7 | Collector scheduler + live-refreshing fragments; demo seed data | Done |
| 8 | Optional Reddit source (only if you have approved access); free social sources: Mastodon, dev.to | Done |
| 9 | Export: PDF/CSV weekly report, email or Slack alert | Done |
| 10 | Optional: LLM-written executive summary, multi-language reviews (AI topic labels, grounded weekly briefings, translation, profile drafts; Ollama or Claude) | Done |

---

## Building it with Claude Code

Keep this file in the project root (you can also copy the key rules into `CLAUDE.md`). Do one roadmap
step per session, run it, then commit before moving on.

**Rules to give Claude Code:**

- Python 3.11, type hints, small modules matching the file table above
- All settings in `config.py` / `.env`; no secrets in code
- Every source must fail gracefully: log the error to `fetch_runs` and carry on with the others
- The dashboard only reads from MySQL; it never calls external APIs or loads the big models directly
- Write pytest tests with mocked HTTP for each source and for the priority maths

**First prompt:**

> Read `brand_radar_roadmap.md`. Do Step 1 only: create the folder layout, `requirements.txt`,
> `.env.example`, `config.py`, `db/schema.sql` (MySQL/MariaDB, utf8mb4, all tables listed, unique key on
> mentions brand_id+source+external_id), `db/models.py`, `db/session.py` and `check_setup.py` that tests the
> MySQL connection to XAMPP. Explain how to run the schema in phpMyAdmin. Don't build later steps yet.
