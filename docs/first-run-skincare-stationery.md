# First run: Skincare and Stationery groups (13 Oct 2026)

The first real collection for the two non-software groups added in `4366e7c` / `d779b7d`: 30 days back,
the sources each profile enables by default (YouTube, news via GDELT, Mastodon for skincare; Reddit is
not configured). It shows that the pipeline works for any industry and where the data is thin.

## Brands

| Group | Own brand | Competitors | Profile |
|---|---|---|---|
| Skincare | Laneige | SKIN1004, Cosrx, Innisfree | `beauty_skincare` |
| Stationery | DOMS | Apsara, Nataraj, Camlin | `stationery` |

Matching was tuned with `manage_brands.py dry-run` before anything was stored. The last change: the
brand is officially **Nataraj** (Hindustan Pencils), so it is now the display name and main keyword,
with "Natraj" kept as the common spelling. Lowercase keywords already match "Nataraj 621" and "Nataraj
pencil", so those go in `youtube_queries` only, and "621" (the flagship HB pencil) counts as a context
word. The Nataraj dry-run matched 9 video titles, all about the pencils, and rejected the Bigg Boss contestant and
temple/cinema uses.

## What was collected

| Brand | Fetched | Kept after cleanup | Analysed (English) | Not English | Net sentiment* |
|---|---:|---:|---:|---:|---:|
| **Laneige** | 26 | 21 | 20 | 1 | +15 |
| SKIN1004 | 23 | 18 | 15 | 3 | +41 |
| Cosrx | 10 | 10 | 6 | 4 | +3 |
| Innisfree | 21 | 20 | 7 | 13 | +22 |
| **DOMS** | 16 | 4 | 2 | 0 | +53 |
| Apsara | 121 | 41 | 29 | 12 | +29 |
| Nataraj | 19 | 11 | 9 | 2 | +38 |
| Camlin | 26 | 14 | 11 | 3 | +60 |

\* mean of P(positive) − P(negative) × 100 over the analysed mentions; with these counts it is indicative only.

- **Almost everything is YouTube comments.** Mastodon found one Laneige post. GDELT news returned no
  article that passed the matching rules, and it dropped the connection or rate-limited (429) for
  Laneige and Cosrx. Those runs are logged in `fetch_runs`, and the collector retries them next cycle.
- **Cleanup: 123 of 262 YouTube comments removed.** One Apsara video asked viewers to vote in the
  comments, which gave 101 of Apsara's 121 mentions, mostly "Sketch book". Zero-shot labelled 41 of
  them *smudging*, so they would have formed a fake complaint topic. A DOMS giveaway video drew children's
  names. New rule: a YouTube comment under 3 words that doesn't name the brand is skipped
  (`too_short_unnamed` in `sources/base.py`). `still_matches` applies the same rule, so
  `manage_brands.py --recheck-mentions --apply` removed the ones already stored. Nothing of substance
  went: the dropped texts are emoji, "Nice", "Very good", names, and poll answers.
- **Language detection is mostly right.** It filters out Bengali, Vietnamese Shopee affiliate spam
  (most of Innisfree's 13 "not English") and romanised Hindi. It misses on very short comments: 3–4
  short English comments ("doms offer better in same price") were tagged nl/de and skipped. Some
  romanised Hindi and Bengali was tagged es/de/id rather than `hinglish`, and some romanised Bengali
  ("apu …") passed as English.

## Topics, priority and alerts

No brand in either group reaches the 30-complaint minimum (most to fewest: Laneige 9, Apsara 6,
SKIN1004 5, the rest 0–2), so no complaint topics, priority quadrants or spike alerts exist yet. The
dashboard shows its empty states for those panels. The Note apps priorities did not change: 27 active
topics, with Evernote's Pop Ups, Share Notes and Free in *Fix now*.

## Early signals (small samples, not conclusions)

- **Laneige** is the only brand with real criticism: the 2026 advent calendar is called overpriced for its
  contents, one person says the Bouncy & Firm mask dries their skin, and people are sad the
  Philippines online store is closing. The praise is for the same advent calendar unboxings.
- **SKIN1004 and Cosrx** comments are mostly questions ("daily use?", "pregnancy safe?") and affiliate links.
- **Stationery** is overwhelmingly positive or neutral: unboxings, price talk ("₹100 geometry box"), and
  brand-vs-brand tests (DOMS vs Apsara, Natraj vs DOMS). There are no complaints yet.
- **DOMS is the thinnest brand** even though it has the most videos (91 remembered). They are mostly
  Shorts with few comments, and the 25 newest are read per cycle (`YOUTUBE_MAX_VIDEOS`).

## What would make these groups useful

1. Let the collector run: YouTube comments on remembered videos accumulate every 6 h, and GDELT will retry.
2. Import reviews by CSV (Amazon, Nykaa, Flipkart exports) with `fetch.py --csv`. That is where product
   complaints live for physical goods, and it is the quickest way past the 30-complaint minimum.
3. Consider broader DOMS `youtube_queries` (e.g. product lines: "DOMS geometry box", "DOMS brush pen").
4. Improve language detection for short romanised Hindi/Bengali comments (planned with multi-language support, roadmap step 10).

Screenshots: [Skincare](screenshots/industry-skincare-desktop.png) ·
[Stationery](screenshots/industry-stationery-desktop.png)
