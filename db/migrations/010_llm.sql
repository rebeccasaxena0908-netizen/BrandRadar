-- Step 10 (LLM insights, translation) for databases created before it. Safe to re-run (MariaDB 10.0+).
--   /Applications/XAMPP/xamppfiles/bin/mysql -u root brand_radar < db/migrations/010_llm.sql
-- then run db/schema.sql, which creates the new tables (briefings, llm_runs).

ALTER TABLE topics
  ADD COLUMN IF NOT EXISTS label_source  VARCHAR(8)   NOT NULL DEFAULT 'keyword',
  ADD COLUMN IF NOT EXISTS description   VARCHAR(512) NULL,
  ADD COLUMN IF NOT EXISTS keyword_label VARCHAR(255) NULL,
  ADD COLUMN IF NOT EXISTS llm_label     VARCHAR(255) NULL,
  ADD COLUMN IF NOT EXISTS llm_members   JSON         NULL,
  ADD COLUMN IF NOT EXISTS llm_at        DATETIME     NULL;

-- the current labels become the keyword fallback; `python -m nlp.topics` marks manual ones on its next run
UPDATE topics SET keyword_label = label WHERE keyword_label IS NULL;

ALTER TABLE mentions
  ADD COLUMN IF NOT EXISTS text_en            MEDIUMTEXT  NULL,
  ADD COLUMN IF NOT EXISTS translation_status VARCHAR(16) NULL,
  ADD COLUMN IF NOT EXISTS translated_by      VARCHAR(64) NULL,
  ADD COLUMN IF NOT EXISTS translated_at      DATETIME    NULL;
