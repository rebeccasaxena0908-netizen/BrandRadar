# Brand Radar

Automated brand sentiment and competitor feature radar. A **collector** pulls public chatter about our
product and its competitors (Hacker News, App Store, Google Play, CSV import, optional Reddit), runs local
Hugging Face models (sentiment, emotion, zero-shot category/feature, BERTopic clustering), computes
priority scores and alerts, and stores everything in MySQL. A separate **Streamlit dashboard**
(glassmorphism style) reads from MySQL and shows KPIs, emotion wheel, word clouds, priority matrix,
feature radar, alert badges and an executive summary.

The full spec and the step-by-step roadmap live in [brand_radar_roadmap.md](brand_radar_roadmap.md).

## Permanent rules

- Python 3.11 in a venv (`venv/`), type hints everywhere, small modules matching the file table in the roadmap
- All settings in `config.py` loaded from `.env`; no secrets in code; `.env` is in `.gitignore`
- MySQL/MariaDB in XAMPP on `localhost:3306` via SQLAlchemy + PyMySQL, `utf8mb4` everywhere
- Every data source fails gracefully and logs errors to `fetch_runs`
- The Streamlit dashboard only reads from MySQL; it never calls external APIs or loads the big models
- Reddit is optional and skipped cleanly when no credentials exist
- pytest tests with mocked HTTP for sources and for the priority maths
- At the end of each step: run the tests, run `check_setup.py`, then git commit with a clear message
- Only build the roadmap step that is asked for; stop and summarise when done

## Environment

- macOS (Apple Silicon), XAMPP MariaDB 10.4
- XAMPP MySQL client: `/Applications/XAMPP/xamppfiles/bin/mysql` (root, no password)
- App DB user: `radar` (`@localhost` and `@127.0.0.1`), privileges on `brand_radar.*` only; credentials in `.env`

## Commands

```bash
source venv/bin/activate
pip install -r requirements.txt
/Applications/XAMPP/xamppfiles/bin/mysql -u root < db/schema.sql   # create/refresh schema (idempotent)
python -m pytest -q                                                 # tests
python check_setup.py                                               # environment + DB check, row counts
python manage_brands.py                                             # sync brands.yaml -> brands table
python fetch.py [--brand X] [--source hackernews] [--since-days 30] # fetch mentions
python fetch.py --csv file.csv --brand Notion                       # CSV import
python -m nlp.pipeline [--brand X] [--limit N]                      # analyse mentions without analysis
python -m nlp.pipeline --reanalyse [--brand X] [--model-version V|outdated]
python -m nlp.pipeline --download                                   # fetch the models into the HF cache
```

## Layout notes

- `db/schema.sql` is the source of truth for the schema; `db/models.py` must match it
  (`tests/test_models.py` checks this; `check_setup.py` checks the live unique keys).
  `schema.sql` only creates missing tables, so schema changes to an existing DB need an `ALTER TABLE`.
- `mentions` is unique on `(brand_id, source, external_id)`: `external_id` is the source's real ID, and
  one post that mentions two brands is stored once per brand.
- Brands live in `brands.yaml`; `python manage_brands.py` syncs them into the `brands` table.
  `brands.keywords` holds search keywords plus exclude phrases prefixed with `-`.
- Sources live in `sources/` and implement `fetch(brand, since) -> list[RawMention]`; `fetch.py` runs them,
  dedupes and writes one `fetch_runs` row per (source, brand).
- `db/session.py` exposes `get_engine()`, `get_session_factory()` and `get_session()`.
- NLP: `nlp/models.py` wraps the Hugging Face pipelines behind the `Analyzer` protocol (tests use a fake;
  never download models in tests). `nlp/pipeline.py` writes one `analysis` row per mention;
  `sentiment_score` is P(positive) − P(negative) in [-1, 1]. Features are keyword-gated (`nlp/features.py`,
  synonyms in `brands.yaml`) and confirmed by zero-shot with `FEATURE_THRESHOLD`; never zero-shot over
  the full feature list alone (bart is overconfident on short labels). Bump `PIPELINE_VERSION` when labels,
  hypotheses, cleaning, thresholds or truncation change, then run `--reanalyse --model-version outdated`.
- This machine is an 8 GB M2: MPS memory is shared with the system. Keep zero-shot input short
  (`NLP_ZERO_SHOT_MAX_TOKENS`), batches small, and free MPS memory between batches.
  bart-large-mnli zero-shot is ~95% of the runtime (~1 s per mention).
