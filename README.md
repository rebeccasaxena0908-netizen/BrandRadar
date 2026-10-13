# Brand Radar

**Brand sentiment and competitor radar.** Brand Radar collects public chatter about a product and its
competitors from Hacker News, the App Store, Google Play, Mastodon and dev.to (Reddit optional). It analyses that chatter with local
transformer models (sentiment, emotion, category, feature) and clusters the complaints into topics.
Each topic gets a priority score, and the results appear on a live Streamlit dashboard in a
"Blush glass" design. Everything is stored in MySQL (MariaDB in XAMPP).

It tracks Notion against Obsidian, Evernote and ClickUp out of the box; brands are configured in `brands.yaml`.

![Brand Radar dashboard, desktop](docs/screenshots/step6-desktop.png)

## Features

- **Collector** (`collector.py`): every 6 hours it fetches new mentions, analyses them, re-clusters
  topics and recomputes priorities. NLP models are only loaded when there is something new.
- **Local NLP**: sentiment, emotion, zero-shot category and keyword-gated feature detection. It runs on
  Apple Silicon (MPS) or CPU, with no API keys or paid services.
- **Complaint topics** with BERTopic, readable labels and manual overrides (`topic_labels.yaml`).
- **Priority matrix**: complaint volume vs pain, in four quadrants (Fix now, Quick wins, Monitor, Ignore).
- **Spike alerts** when a topic's complaints at least double week over week.
- **Dashboard**:
  - KPI cards and an emotion wheel
  - sentiment trend
  - complaint and praise word clouds
  - priority matrix and feature radar vs competitors
  - alert pills
  - a 20-second executive summary built from the numbers, with representative quotes

  The KPIs, alerts and "Updated … ago" status refresh every 60 s without reloading the page.
- **Demo mode** with a separate, fictional dataset, so the dashboard can be shown without real data.

## Tech stack

| Area | Tools |
|---|---|
| Language | Python 3.11 |
| Storage | MySQL / MariaDB (XAMPP), SQLAlchemy 2, PyMySQL |
| Sources | Algolia HN Search + Firebase, iTunes RSS reviews feed, `google-play-scraper`, Mastodon and dev.to public APIs, optional PRAW (Reddit) |
| NLP | PyTorch, Hugging Face Transformers, sentence-transformers, BERTopic (UMAP + HDBSCAN) |
| Scheduling | APScheduler |
| Dashboard | Streamlit, Plotly, `wordcloud` |
| Tests | pytest with mocked HTTP and models (no downloads), Streamlit AppTest, Playwright screenshots |

## How it works

```
 Sources                    Collector  (every 6 h, one cycle at a time)            MySQL                 Dashboard
 ───────                    ───────────────────────────────────────────            ─────                 ─────────
 Hacker News  ──┐   1 fetch      new mentions since the last success, deduped ──►  mentions, fetch_runs
 App Store    ──┤
 Google Play  ──┼─► 2 NLP*       sentiment · emotion · category · feature    ──►  analysis              Streamlit
 Mastodon     ──┤   3 topics*    BERTopic per brand on complaints**          ──►  topics, topic_daily ──► reads only,
 dev.to       ──┤   4 priority   volume × pain, quadrants, spike alerts      ──►  topics, alerts        cached 60 s
 Reddit (opt) ──┤                cycle record                                ──►  collector_runs
 CSV import   ──┘
                    * runs in a short-lived child process, so the models' memory is freed after each cycle
                    ** dev.to is left out of complaint topics, the matrix and alerts (TOPIC_EXCLUDED_SOURCES)
```

The collector and the dashboard are separate processes. The dashboard only reads MySQL; it never calls an
external API or loads a model, so it stays fast and light.

## Setup

Works on macOS and Windows (Linux too). You need **Python 3.11** and **XAMPP**.

> **Note:** the Windows steps below haven't been tested on a Windows machine yet; Brand Radar was built and tested on macOS.

**1. Install XAMPP** from [apachefriends.org](https://www.apachefriends.org) and start **MySQL** in the
XAMPP Control Panel. XAMPP's MySQL client is here (its `root` user has no password by default):

| macOS | Windows |
|---|---|
| `/Applications/XAMPP/xamppfiles/bin/mysql` | `C:\xampp\mysql\bin\mysql.exe` |

**2. Get the code and create a Python environment**

```bash
git clone <this repository> BrandRadar
cd BrandRadar
```

| macOS / Linux | Windows (PowerShell) |
|---|---|
| `python3.11 -m venv venv` | `py -3.11 -m venv venv` |
| `source venv/bin/activate` | `venv\Scripts\Activate.ps1` |
| `pip install -r requirements.txt` | `pip install -r requirements.txt` |

**3. Create the database and a dedicated user.** Run these from the project folder with XAMPP's client
(replace `mysql` with the full path above if it is not on your PATH, and choose your own password).
`source` works the same in Terminal, PowerShell and cmd:

```bash
mysql -u root -e "source db/schema.sql"
mysql -u root -e "CREATE USER 'radar'@'localhost' IDENTIFIED BY 'choose-a-password'; CREATE USER 'radar'@'127.0.0.1' IDENTIFIED BY 'choose-a-password'; GRANT ALL PRIVILEGES ON brand_radar.* TO 'radar'@'localhost', 'radar'@'127.0.0.1'; GRANT ALL PRIVILEGES ON brand_radar_demo.* TO 'radar'@'localhost', 'radar'@'127.0.0.1';"
```

**4. Configuration:** copy `.env.example` to `.env` (`cp` on macOS, `copy` on Windows) and set
`DB_PASSWORD` to the password you chose. Every setting (intervals, thresholds, sources) lives in `.env`;
secrets never go in code, and `.env` is git-ignored.

**5. Brands, models, first data**

```bash
python manage_brands.py            # brands.yaml -> brands table
python -m nlp.pipeline --download  # ~2.5 GB of models into the Hugging Face cache (once)
python collector.py --once         # first fetch + analysis
python check_setup.py              # checks Python, packages, models, database, collector, demo data
```

The first analysis takes about 15 minutes on an 8 GB Apple Silicon Mac (GPU via MPS). On Windows the models
run on the CPU (or an NVIDIA GPU if PyTorch sees one), so expect it to take longer.

## Running it

| What | macOS / Linux | Windows (PowerShell) |
|---|---|---|
| Collector + dashboard together (Ctrl+C stops both) | `scripts/start.sh` | run the next two lines in two terminals |
| Collector, every `COLLECTOR_INTERVAL_HOURS` (default 6) | `python collector.py` | `python collector.py` |
| Dashboard | `streamlit run app.py` | `streamlit run app.py` |
| One collector cycle | `python collector.py --once` | `python collector.py --once` |
| Build the demo database | `python seed_demo.py` | `python seed_demo.py` |
| Dashboard on demo data | `BRAND_RADAR_DEMO=1 streamlit run app.py` | `$env:BRAND_RADAR_DEMO=1; streamlit run app.py` |
| Import a CSV (`source, author, text, date, rating`) | `python fetch.py --csv file.csv --brand Notion` | same |
| Tests | `python -m pytest -q` | same |

The dashboard opens at <http://localhost:8501>. In PowerShell, `$env:BRAND_RADAR_DEMO` stays set for that
terminal; close it (or run `Remove-Item Env:BRAND_RADAR_DEMO`) before going back to real data.

Only one collector runs at a time (a lock file plus APScheduler `max_instances=1`); a second one exits
with a message. If a stage fails, the error is logged to `collector_runs` and the later stages still run.

**Demo mode** reads a separate `brand_radar_demo` database with about 1,100 fictional mentions over 60
days, built from templates (no real people). Your real data is never touched: the seed script refuses to
write to `DB_NAME`, and the collector refuses to run in demo mode. Re-running `seed_demo.py` rebuilds it.

## Data sources and their limits

| Source | Notes |
|---|---|
| Hacker News | Algolia keyword search (stories and comments) plus Firebase. Free and reliable. Common words like "notion" need case-sensitive matching and exclude lists (`brands.yaml`). |
| App Store | The public reviews feed only returns **recent** reviews (up to ~500 per country); we read the US, India, the UK, Canada and Australia. |
| Google Play | `google-play-scraper` is unofficial and can break when Google changes its pages; failures are logged, not fatal. English, US store. |
| Reddit | **Optional, off by default.** Posts and top-level comments that name the brand, plus the brand's own subreddits. Needs approved API credentials; see [Enabling Reddit](#enabling-reddit). Skipped cleanly without them. |
| Mastodon | Public hashtag timelines (#notion, #obsidian…) on mastodon.social, fosstodon.org and hachyderm.io (`MASTODON_INSTANCES`), no login. English posts only; people who opted out of search or discovery, bots, boosts and non-public posts are skipped; the post must still name the brand. Full-text search needs a login, so hashtags are the open route. About 40 mentions a month, almost all Notion and Obsidian. |
| dev.to | Articles tagged with the brand plus their top-level comments, via the public Forem API (no key). Mostly makers writing about their own tools, so it feeds sentiment, the feature radar and word clouds but is **left out of complaint topics, the matrix and alerts** (`TOPIC_EXCLUDED_SOURCES`). At most 3 items per author per brand and run. |
| CSV import | Any other source (e.g. exported social posts), via `fetch.py --csv`. |

## Enabling Reddit

The Reddit source is built and tested, but stays off until you add API credentials. Getting them is
no longer self-service:

1. **Read Reddit's rules first:** the [Responsible Builder Policy](https://support.reddithelp.com/hc/en-us/articles/42728983564564-Responsible-Builder-Policy),
   the [Reddit Data API Wiki](https://support.reddithelp.com/hc/en-us/articles/16160319875092-Reddit-Data-API-Wiki)
   and [Developer Platform & Accessing Reddit Data](https://support.reddithelp.com/hc/en-us/articles/14945211791892-Developer-Platform-Accessing-Reddit-Data).
   Since the policy took effect (November 2025) every app needs explicit approval before it can use the API.
   Using Reddit data for a business needs separate written approval.
2. **Request access** with Reddit's [developer support form](https://support.reddithelp.com/hc/en-us/requests/new?ticket_form_id=21879292693140).
   Describe Brand Radar honestly: read-only, low volume (a few hundred requests every 6 hours), public posts
   and comments about specific products, analysed locally, never redistributed. Reddit asks developers to use
   its Developer Platform (Devvit) where possible; this form is for tools Devvit can't support, such as this
   external analysis. Approval is not guaranteed, and replies can take a while.
3. **Mind the deadlines** (as reported by the press, e.g.
   [TechRepublic](https://www.techrepublic.com/article/news-reddit-rss-public-api-shutdown/) and
   [Decrypt](https://decrypt.co/379892/reddit-killing-rss-feeds-public-api-access-ai-scrapers)):
   Reddit stops accepting new public API requests on **31 October 2026**, and remaining public Data API
   access is due to end in **March 2027**. After that, this source will stop working even with credentials.
4. **Once approved**, fill in these values in `.env` and restart the collector:

   ```ini
   REDDIT_CLIENT_ID=your-client-id
   REDDIT_CLIENT_SECRET=your-client-secret
   REDDIT_USER_AGENT=python:brand-radar:0.1 (by u/your-reddit-username)
   ```

   `python check_setup.py` then shows Reddit as configured, and the next collector cycle fetches it. Which
   subreddits count as a brand's own is set by `subreddits` in `brands.yaml`
   (`Notion`, `ObsidianMD`, `Evernote`, `clickup`).

## Model quality (measured on the real data)

| Check | Result |
|---|---|
| Sentiment vs star rating, App Store | **79.2%** agree (87.9% leaving out 3★ reviews) |
| Sentiment vs star rating, Google Play | **76.5%** agree (81.3% leaving out 3★ reviews) |
| Mentions with a detected feature | 27–48% per brand (keyword gate + zero-shot confirmation) |
| Topic ids across re-runs | stable (same data -> same ids; new data joins existing topics) |
| Collector cycle, nothing new | ~1 min, no models loaded; idle collector ~55 MB |
| First full analysis, 816 mentions | ~13 min on an 8 GB M2 (MPS) |

Models: `cardiffnlp/twitter-roberta-base-sentiment-latest`, `j-hartmann/emotion-english-distilroberta-base`,
`facebook/bart-large-mnli` (zero-shot) and `sentence-transformers/all-MiniLM-L6-v2` (topics).

## Known limitations

- **English only.** Non-English reviews get analysed as English; very long non-English texts are kept out of topics.
- **Topic themes can drift.** Topic ids follow their mentions across re-clustering, but with small data
  the clusters reshape when new mentions arrive, so a manual label in `topic_labels.yaml` can end up on a
  changed theme. Re-read and re-map them after a big data change (labels that follow their example
  mentions are planned).
- **Few free social sources.** Evernote and ClickUp get almost nothing from Mastodon and dev.to; Reddit
  (optional, closing in 2027) was the main social source for them.
- **One feature per mention.** A review about sync *and* pricing gets one feature; about 15% of mentions name several.
- **Keyword-based topic labels.** Some are clunky; fix them in `topic_labels.yaml` (labels written by an LLM are a planned extra).
- **Some praise lands in the complaint set** (positive reviews the model put in a complaint category). These
  topics score zero pain, so the matrix hides them by default.
- **The emotion model is weak on single reviews.** The dashboard only shows emotion in aggregate.
- **The App Store and Play history is shallow**, so trends need the collector running over time.
- **Small data.** With about 800 mentions, topics for small brands (ClickUp) can be skipped (30 complaints minimum).

## Project layout

```
app.py              Streamlit dashboard          collector.py     scheduled fetch -> NLP -> topics -> priority
fetch.py            fetch runner + CSV import    seed_demo.py     demo database
manage_brands.py    brands.yaml -> brands        check_setup.py   environment / database / model checks
config.py           settings from .env           scripts/start.sh collector + dashboard together
db/                 schema.sql, models, session
sources/            hackernews, appstore, playstore, reddit, csv_import, shared HTTP + matching
nlp/                pipeline (models, text, features), topics, priority
ui/                 styles.css (approved theme), streamlit.css, components, data, insights
tests/              pytest suite (mocked HTTP and models)
docs/               ui-decisions.md, screenshots/
```

Design decisions for the dashboard, with screenshots of each step, are in
**[docs/ui-decisions.md](docs/ui-decisions.md)**. The step-by-step build plan is in
[brand_radar_roadmap.md](brand_radar_roadmap.md).

## License

[MIT](LICENSE) © 2026 Rebecca Saxena
