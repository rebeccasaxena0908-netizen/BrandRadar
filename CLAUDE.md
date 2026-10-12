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
```

## Layout notes

- `db/schema.sql` is the source of truth for the schema; `db/models.py` must match it
  (`tests/test_models.py` checks this).
- `db/session.py` exposes `get_engine()` and `get_session()`.
