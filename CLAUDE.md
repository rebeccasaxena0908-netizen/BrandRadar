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
python -m nlp.topics [--brand X]                                    # cluster complaints, rebuild topic_daily
python -m nlp.priority [--as-of DATE] [--report]                    # priority snapshot + spike alerts
python manage_brands.py --recheck-mentions [--apply]                # after changing excludes
streamlit run app.py                                                # dashboard
python tools/screenshot.py --url http://localhost:8501 --name stepN --motion   # 1440px + 390px captures
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
- Topics: BERTopic per brand on complaints (negative, or a complaint category), HDBSCAN `leaf` with a
  min size that scales with the data, non-English texts skipped. Re-runs keep topic ids by member
  overlap (`match_clusters`); never delete topics on re-run (alerts and history point at them).
  Labels: KeyBERTInspired + MMR keyphrases, Title Case, unique per brand; `topics.category` holds the
  dominant category. Manual labels go in `topic_labels.yaml` (topic id -> label) and always win.
  Priority snapshot lives on `topics` (volume_7d, pain, trend, quadrant); max one spike alert per topic per ISO week.
- Dashboard: `app.py` (layout) + `ui/data.py` (read-only cached queries, pure aggregations) +
  `ui/components.py` (Plotly "blush" template, glass/KPI/empty-state helpers, background HTML).
  Glass cards are `st.container(key="glass-…")`. Restart Streamlit after editing ui/*.py (modules are cached).
  Set chart backgrounds explicitly in `show_chart` (Streamlit paints Plotly paper white otherwise).
  AppTest can't rerun pages with dynamic multiselect options: set `session_state` before the first run.
  Step 6 logic (matrix, feature radar, summary, quotes) is pure code in `ui/insights.py`.
  Glass + padding go on the wrapper (`stVerticalBlockBorderWrapper:has(> div > .st-key-glass-…)`), never on
  the keyed block: Streamlit sizes Plotly from the block, so padding there makes charts overflow.
  UI decisions and screenshots: `docs/ui-decisions.md`.
- Brand matching: `exclude` phrases are removed before matching; `exclude_if` words drop the mention.
- This machine is an 8 GB M2: MPS memory is shared with the system. Keep zero-shot input short
  (`NLP_ZERO_SHOT_MAX_TOKENS`), batches small, and free MPS memory between batches.
  bart-large-mnli zero-shot is ~95% of the runtime (~1 s per mention).

## Design system (use for all UI work)

- Theme: "Blush glass", a light pink-and-white glassmorphism
- Full CSS lives in ui/styles.css (approved theme; don't change its colours, fonts, blur values or animation timings; you may only add new classes)
- Background: animated drifting blurred blobs in #F4A3B1, #E58A9A, #F3BAC1, #FFFFFF (as in the approved ui/styles.css) over a white-to-blush gradient, plus 16 rising translucent glass bubbles; respect prefers-reduced-motion
- New CSS goes in ui/streamlit.css (loaded after styles.css); ui/styles.css itself stays byte-for-byte unchanged
- Glass cards: rgba(255,255,255,0.52), blur(18px) saturate(150%), 1px white border, radius 20px, soft berry shadow
- Text: #242424 primary, #6B5F62 secondary, #9C9899 labels
- Brand colours: own brand #9E4A5A (berry), competitors #E79AA6, #B7AEB0, #F3BAC1
- Sentiment: positive #6FA98A, neutral #B7AEB0, negative #D9536A
- Emotions: joy #E9B949, anger #D9536A, sadness #7FA7D9, fear #A78BC9, disgust #8FB26B, surprise #E79AA6, neutral #B7AEB0
- Fonts: Montserrat (headings, numbers), Figtree (text), from Google Fonts
- Plotly: transparent backgrounds, #E7CFCF gridlines, Figtree font, these colours

### Moving background (must match the approved theme exactly)

- Inject it once in app.py with st.markdown(unsafe_allow_html=True): the .br-sky wrapper, the 4 .br-blob clouds and 16 .br-bubble bubbles
- st.markdown does not run JavaScript, so generate the 16 bubbles in Python with inline styles: width/height 18–88px, left 0–96vw, --sway ±20–80px, animation-duration 12–26s, negative animation-delay. Use a fixed random seed so they don't jump on every rerun
- Keep it behind all content (z-index) with pointer-events: none so it never blocks clicks
