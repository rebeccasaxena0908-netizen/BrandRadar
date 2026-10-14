# Brand Radar – UI decisions

Design decided 12 Oct 2026. Step 5 done 13 Oct 2026 (commit `346fc9f`); Step 6 done 13 Oct 2026 (`e20e791`);
Step 7 done 13 Oct 2026 (`bb239fa`).

## Chosen design: "Blush glass"

- Light pink-and-white glassmorphism, palette sampled from Rebecca's resume
- Moving background: 4 drifting blurred colour clouds (9–13 s cycles) and 16 rising translucent glass bubbles;
  paused under `prefers-reduced-motion`
- The approved CSS is [`ui/styles.css`](../ui/styles.css) and is never edited. Additions live in
  [`ui/streamlit.css`](../ui/streamlit.css) (new classes and Streamlit hooks only)

| Role | Colours |
|---|---|
| Surfaces | white `#FFFFFF`, blush `#F0DCDB`, petal `#F2E0E0`, rose rule `#E7CFCF` |
| Roses | photo rose `#F3BAC1`, deep rose `#E79AA6`, berry `#9E4A5A` |
| Text | ink `#242424`, ink soft `#6B5F62`, muted `#9C9899` |
| Background blobs (approved file) | `#F4A3B1`, `#E58A9A`, `#F3BAC1`, `#FFFFFF` |
| Sentiment | positive `#6FA98A`, neutral `#B7AEB0`, negative `#D9536A` |
| Brands | selected brand `#9E4A5A`; competitors `#E79AA6`, `#B7AEB0`, `#F3BAC1` |
| Emotions | joy `#E9B949`, anger `#D9536A`, sadness `#7FA7D9`, fear `#A78BC9`, disgust `#8FB26B`, surprise `#E79AA6`, neutral `#B7AEB0` |

Fonts: Montserrat (headings, numbers) and Figtree (text), from Google Fonts. Word clouds use a bundled
Montserrat SemiBold file (`ui/fonts/`), because the `wordcloud` package needs a font file.

## Step 5 – header, KPIs, emotion wheel, trend, word clouds

| | |
|---|---|
| Desktop (1440 px) | [step5-desktop.png](screenshots/step5-desktop.png) |
| Phone (390 px) | [step5-mobile.png](screenshots/step5-mobile.png) |

- **Default date range: the last 14 days.** We hold about 30 days of data, so the previous 14 days exist and
  every KPI card shows a real change. "— no earlier data" is only a fallback for longer ranges.
- **Emotion wheel: mean emotion probability, neutral left out.** Neutral is about 45% of all emotion and
  would dwarf the rest; the Top emotion KPI still shows it. The emotion model is unreliable per mention, so
  only aggregates are shown. The wheel compares against the competitor with the most mentions.
- **Daily trend: 3-day rolling mean, weighted by mentions.** Single days with one or two mentions swing too much.
- **The brand being viewed is drawn in berry**, even when it is a competitor such as ClickUp.
- **Thin data shows a designed empty state** (radar icon, "Not enough data yet", a one-line reason) instead
  of an error or a blank chart.
- **Glass cards are `st.container(key="glass-…")`.** The moving background sits in a `.br-sky-layer`
  wrapper at `z-index: -1`, because Streamlit renders it inside `.block-container`, which the theme makes a
  stacking context; otherwise bubbles could paint over cards.

## Step 6 – priority matrix, alerts, feature radar, executive summary

| | |
|---|---|
| Desktop (1440 px) | [step6-desktop.png](screenshots/step6-desktop.png) |
| Phone (390 px) | [step6-mobile.png](screenshots/step6-mobile.png) |

Layout: on desktop the priority matrix takes the wide column, with the alerts and an "Act on these first"
list beside it; the feature radar and executive summary sit side by side below. On a phone everything stacks.

**Priority matrix**
- x = complaints in the last 7 days on a log scale (ticks show real counts), y = pain, from the priority
  snapshot stored on `topics`. The caption shows the snapshot time; the date and source filters do not apply.
- Quadrants split at `PRIORITY_VOLUME_THRESHOLD` (5) and `PRIORITY_PAIN_THRESHOLD` (0.15), with very soft
  rose tints (Fix now slightly stronger). Quadrants are recomputed from config, so the colours always match
  the dividing lines.
- **Praise and discussion topics are hidden by default** when pain is below `PRIORITY_PRAISE_PAIN_MAX`
  (0.02) and mean sentiment is positive. "Pain = 0" alone only caught 2 of the 5 praise topics clustered in
  Step 4. A "Show all topics" toggle brings them back.
- Only Fix now, Quick wins and Monitor bubbles carry labels; Ignore topics cluster at low pain and their
  labels collided, so they are hover-only. Quadrant names sit at the top of each quadrant for the same reason.
- "Act on these first" (Fix now, then Quick wins, by volume × pain) is its own card under the alerts,
  which balances the two columns.

**Trending complaint alerts**
- Pills show the topic's *current* label (stored alert messages can hold an older one), the multiplier and
  "N this week". Seen alerts are dimmed; the dashboard never marks alerts as seen.
- The numbers come from new `alerts.this_week` / `alerts.last_week` columns, written by `nlp/priority.py`
  (the topic snapshot only holds today's numbers). The one older alert was backfilled from its message.
- No alerts shows a calm "No spikes this week" state, not an empty card.

**Feature radar**
- Uses **all available data (about 30 days), not the date filter**, and says so in the caption. In 14 days
  Notion had data for only 3 of 6 qualifying features; over 30 days 11 features qualify.
- A feature is shown when at least one shown brand has 5+ mentions; a brand's point needs 3+ mentions.
  Missing points are bridged and the hover says "too few mentions". Fewer than 3 features shows the empty state.
- Scale −100 to +100 net sentiment with a dotted 0 ring; the selected brand is filled berry, competitors
  are outlines in their colours.

**Executive summary** (numbers only, no LLM)
- Biggest risk: the top Fix now topic, else the top Quick win *with complaints this week*, else
  "Watch: …" for the busiest Monitor topic (so it agrees with the alerts), else "no urgent themes".
- Biggest win: the feature where the brand beats the competitors' average by most, only with 5+ mentions.
- Top request: the feature most named in "feature request" mentions, only when named at least twice.
- Changes are written in words ("up from 2 last week", "same as last week"); pills and hovers keep "3.5×".
- Three quotes (negative, neutral, positive): readable English, 60–220 characters, no links; the strongest
  negative/positive and the most neutral, with source, star rating and date.

## Step 7 – live refresh, collector status, demo mode

| | |
|---|---|
| Demo data, desktop (1440 px) | [step7-demo-desktop.png](screenshots/step7-demo-desktop.png) |

- **Live parts** (`@st.fragment(run_every="60s")`): the KPI row, the alert pills and the header status.
  Charts keep `st.cache_data(ttl=60)` and update on the next interaction.
- **Header status** under the subtitle: a green dot and "Updated 12 min ago", from the last ok/partial
  collector cycle. An amber badge appears when the last cycle had errors or failed, or when the data is
  older than 2 × `COLLECTOR_INTERVAL_HOURS` ("overdue"). "· collecting…" shows while a cycle runs.
- **Demo data pill**: a small berry pill on its own line under the subtitle. Placing it after the
  wordmark pushed it into the Brand selector, and inline with the subtitle made the subtitle wrap.

## Step 10 – AI-written briefing, topic descriptions, translated quotes

| | |
|---|---|
| Real data, Note apps, desktop (1440 px) | [step10-desktop.png](screenshots/step10-desktop.png) |
| Real data, Note apps, phone (390 px) | [step10-mobile.png](screenshots/step10-mobile.png) |
| Real data, Skincare (thin data), desktop | [step10-skincare-desktop.png](screenshots/step10-skincare-desktop.png) |

- **Briefing in the Executive summary card**, replacing the rule-based text only when a stored briefing
  exists for the brand and period. Under it a small berry **AI-written** pill plus the period, model and
  "every number checked against the data", so readers can tell it apart from computed text. Without a
  briefing the card is exactly the Step 6 summary.
- **Topic descriptions** (one sentence) sit under each topic in "Act on these first" and in the matrix
  hover, in the secondary text colour, so the label stays the scannable part.
- **Translated quotes** show the English text as the quote and the original below it in a smaller,
  muted line ("Translated from Bengali · …"), so nobody mistakes the translation for the author's words.

## Step 11 – Track a new brand

| | |
|---|---|
| Nivea group (Skin care products), desktop | [step11-nivea-desktop.png](screenshots/step11-nivea-desktop.png) |
| McDonald's group (Fast food restaurants), desktop | [step11-mcdonalds-desktop.png](screenshots/step11-mcdonalds-desktop.png) |
| The Track card with a review open | [step11-review.png](screenshots/step11-review.png) |

- **One card at the bottom of the page**, after Export: tracking a brand is occasional work, not something
  to read every visit. A text box and a berry Track button; errors and confirmations appear as a soft
  coloured line under it and stay for 8 s (an action is often followed by a second rerun that would
  swallow a show-once message).
- **Request list** as glass rows: name, the new group, a status pill (berry while working, solid berry for
  "Waiting for your review", green Ready, amber Not enough data yet, red Failed, grey Cancelled/Removed),
  the collector's current step ("Fetching YouTube · Dove (11/18)") and the age. It refreshes every 10 s in
  a fragment; finished requests show a small table of mentions per brand and source, the top topics and
  the time taken.
- **The review panel lives outside the refreshing fragment**, so an edit form is never redrawn under the
  cursor; the list asks for one full rerun when a request reaches review. It shows the industry, profile
  and group, a table of each brand's rules as chips (keywords, cut out, drop if, must mention, YouTube and
  the app found), the research notes, the dry-run counts, and 10 matched / 10 rejected samples side by side
  with the source and the rejection reason in small muted type.
- **Edit is a plain form** with the same field names as brands.yaml in brackets, so what you change here
  is what you would change in the file. A request that needs input (no LLM) opens straight on the form,
  with its own Cancel button.
- **Removal is in a collapsed expander** and needs the group's name typed exactly; the delete button stays
  disabled until it matches. Only groups added from the dashboard are listed.

## Lessons for future UI work

- **Put padding on the glass wrapper, never on the keyed block.** Streamlit sizes Plotly charts from the
  keyed block's full width, so padding there made every chart 2 × 20 px too wide for its card (also in the
  Step 5 screenshots). The glass now sits on `stVerticalBlockBorderWrapper:has(> div > .st-key-glass-…)`.
- **Set Plotly backgrounds explicitly** (`show_chart` does): Streamlit paints the paper white otherwise.
- **Plotly clips text to its own SVG**, so overflow checks on the card miss cut-off axis labels; check
  label boxes against the SVG too (the radar needed 64 px side margins for "Performance" on a phone).
- **Screenshots** (`tools/screenshot.py`) use the installed Google Chrome. Pages taller than one screen
  are stitched from slices; because the background is fixed to the viewport, faint horizontal seams at the
  joins are capture artifacts, not part of the page.
