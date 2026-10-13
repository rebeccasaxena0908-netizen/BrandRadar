-- Brand Radar schema for MySQL / MariaDB (XAMPP).
-- Safe to run more than once: nothing is dropped.
-- Run with:  /Applications/XAMPP/xamppfiles/bin/mysql -u root < db/schema.sql
-- or paste into phpMyAdmin > SQL.
-- All timestamps are stored in UTC.

CREATE DATABASE IF NOT EXISTS brand_radar
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE brand_radar;

-- Our own product and its competitors
CREATE TABLE IF NOT EXISTS brands (
  id             INT UNSIGNED NOT NULL AUTO_INCREMENT,
  name           VARCHAR(100) NOT NULL,
  is_own         TINYINT(1)   NOT NULL DEFAULT 0,
  keywords       JSON         NULL,
  app_store_id   VARCHAR(32)  NULL,
  play_store_id  VARCHAR(255) NULL,
  features       JSON         NULL,
  brand_group    VARCHAR(64)  NOT NULL DEFAULT 'Default',           -- one own brand + its competitors
  profile        VARCHAR(32)  NOT NULL DEFAULT 'consumer_product',  -- industry profile (profiles.yaml)
  config         JSON         NULL,      -- require_context, youtube_queries, sources
  PRIMARY KEY (id),
  UNIQUE KEY uq_brands_name (name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Raw text pulled from every source.
-- external_id is the source's own ID; unique per brand so one comment can count for several brands.
CREATE TABLE IF NOT EXISTS mentions (
  id           BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  brand_id     INT UNSIGNED    NOT NULL,
  source       VARCHAR(32)     NOT NULL,
  external_id  VARCHAR(255)    NOT NULL,
  author       VARCHAR(255)    NULL,
  text         MEDIUMTEXT      NOT NULL,
  url          VARCHAR(1024)   NULL,
  rating       TINYINT UNSIGNED NULL,
  created_at   DATETIME        NOT NULL,
  fetched_at   DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
  language     VARCHAR(16)     NULL,      -- 'en', 'hi', 'hinglish', 'ko', 'und'… (nlp/language.py)
  PRIMARY KEY (id),
  UNIQUE KEY uq_mentions_brand_source_external (brand_id, source, external_id),
  KEY ix_mentions_brand_created (brand_id, created_at),
  CONSTRAINT fk_mentions_brand FOREIGN KEY (brand_id)
    REFERENCES brands (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Complaint / praise clusters found by BERTopic
CREATE TABLE IF NOT EXISTS topics (
  id          INT UNSIGNED NOT NULL AUTO_INCREMENT,
  brand_id    INT UNSIGNED NOT NULL,
  label       VARCHAR(255) NOT NULL,
  category    VARCHAR(32)  NULL,      -- dominant analysis.category of the topic's mentions
  keywords    JSON         NULL,
  first_seen  DATETIME     NULL,
  last_seen   DATETIME     NULL,
  -- priority snapshot, written by nlp/priority.py
  volume_7d       INT UNSIGNED NOT NULL DEFAULT 0,
  volume_prev_7d  INT UNSIGNED NOT NULL DEFAULT 0,
  pain            FLOAT        NULL,
  trend           FLOAT        NULL,
  quadrant        VARCHAR(16)  NULL,
  scored_at       DATETIME     NULL,
  centroid        JSON         NULL,      -- mean member embedding; keeps topic ids on the same theme
  PRIMARY KEY (id),
  KEY ix_topics_brand (brand_id),
  CONSTRAINT fk_topics_brand FOREIGN KEY (brand_id)
    REFERENCES brands (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- One analysis row per mention (model outputs)
CREATE TABLE IF NOT EXISTS analysis (
  mention_id       BIGINT UNSIGNED NOT NULL,
  sentiment        VARCHAR(16)  NULL,
  sentiment_score  FLOAT        NULL,
  emotion          VARCHAR(16)  NULL,
  emotion_scores   JSON         NULL,
  category         VARCHAR(32)  NULL,
  feature          VARCHAR(64)  NULL,
  topic_id         INT UNSIGNED NULL,
  model_version    VARCHAR(64)  NULL,
  PRIMARY KEY (mention_id),
  KEY ix_analysis_topic (topic_id),
  KEY ix_analysis_category (category),
  CONSTRAINT fk_analysis_mention FOREIGN KEY (mention_id)
    REFERENCES mentions (id) ON DELETE CASCADE,
  CONSTRAINT fk_analysis_topic FOREIGN KEY (topic_id)
    REFERENCES topics (id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Precomputed daily counts per topic for fast charts
CREATE TABLE IF NOT EXISTS topic_daily (
  topic_id       INT UNSIGNED NOT NULL,
  day            DATE         NOT NULL,
  mentions       INT UNSIGNED NOT NULL DEFAULT 0,
  avg_sentiment  FLOAT        NULL,
  PRIMARY KEY (topic_id, day),
  CONSTRAINT fk_topic_daily_topic FOREIGN KEY (topic_id)
    REFERENCES topics (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Spike alerts shown as badges in the dashboard
CREATE TABLE IF NOT EXISTS alerts (
  id          INT UNSIGNED NOT NULL AUTO_INCREMENT,
  brand_id    INT UNSIGNED NOT NULL,
  topic_id    INT UNSIGNED NULL,
  kind        VARCHAR(32)  NOT NULL,
  message     VARCHAR(1024) NOT NULL,
  this_week   INT UNSIGNED NULL,      -- complaint mentions in the 7 days before the alert
  last_week   INT UNSIGNED NULL,      -- ...and in the 7 days before that
  created_at  DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  seen        TINYINT(1)   NOT NULL DEFAULT 0,
  PRIMARY KEY (id),
  KEY ix_alerts_brand_seen (brand_id, seen),
  CONSTRAINT fk_alerts_brand FOREIGN KEY (brand_id)
    REFERENCES brands (id) ON DELETE CASCADE,
  CONSTRAINT fk_alerts_topic FOREIGN KEY (topic_id)
    REFERENCES topics (id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- One row per source fetch, including failures
CREATE TABLE IF NOT EXISTS fetch_runs (
  id           BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  source       VARCHAR(32)  NOT NULL,
  brand_id     INT UNSIGNED NULL,
  started_at   DATETIME     NOT NULL,
  finished_at  DATETIME     NULL,
  new_rows     INT UNSIGNED NOT NULL DEFAULT 0,
  error        TEXT         NULL,
  PRIMARY KEY (id),
  KEY ix_fetch_runs_source_started (source, started_at),
  CONSTRAINT fk_fetch_runs_brand FOREIGN KEY (brand_id)
    REFERENCES brands (id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- One row per collector cycle (fetch -> NLP -> topics -> priority); the dashboard's "Updated … ago"
CREATE TABLE IF NOT EXISTS collector_runs (
  id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  started_at      DATETIME     NOT NULL,
  finished_at     DATETIME     NULL,
  status          VARCHAR(16)  NOT NULL DEFAULT 'running',  -- running / ok / partial / failed
  new_mentions    INT UNSIGNED NOT NULL DEFAULT 0,
  analysed        INT UNSIGNED NOT NULL DEFAULT 0,
  analysis_failed INT UNSIGNED NOT NULL DEFAULT 0,
  topics          INT UNSIGNED NOT NULL DEFAULT 0,
  alerts_created  INT UNSIGNED NOT NULL DEFAULT 0,
  stage_seconds   JSON         NULL,
  error           TEXT         NULL,
  PRIMARY KEY (id),
  KEY ix_collector_runs_started (started_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- YouTube videos whose title matched a brand: their comments are re-read on later cycles without
-- spending search calls (search.list has its own small daily cap)
CREATE TABLE IF NOT EXISTS youtube_videos (
  id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  brand_id        INT UNSIGNED  NOT NULL,
  video_id        VARCHAR(32)   NOT NULL,
  title           VARCHAR(512)  NOT NULL,
  channel         VARCHAR(255)  NULL,
  published_at    DATETIME      NOT NULL,
  first_seen_at   DATETIME      NOT NULL,
  comments_checked_at DATETIME  NULL,
  comments_disabled BOOLEAN     NOT NULL DEFAULT FALSE,
  PRIMARY KEY (id),
  UNIQUE KEY uq_youtube_videos_brand_video (brand_id, video_id),
  CONSTRAINT fk_youtube_videos_brand FOREIGN KEY (brand_id)
    REFERENCES brands (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Quota spent per API and day (YouTube's quota day is Pacific time); sources stop at the budget
CREATE TABLE IF NOT EXISTS api_usage (
  id     INT UNSIGNED NOT NULL AUTO_INCREMENT,
  api    VARCHAR(32)  NOT NULL,            -- 'youtube'
  bucket VARCHAR(32)  NOT NULL,            -- 'search' (calls) / 'units'
  day    DATE         NOT NULL,
  used   INT UNSIGNED NOT NULL DEFAULT 0,
  PRIMARY KEY (id),
  UNIQUE KEY uq_api_usage_api_bucket_day (api, bucket, day)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Notifications sent (email / Slack). One row per (kind, ref, channel) so nothing is ever sent twice:
-- a row is claimed ('sending') before the message goes out; 'failed' rows are retried a few times.
CREATE TABLE IF NOT EXISTS notifications (
  id          INT UNSIGNED NOT NULL AUTO_INCREMENT,
  kind        VARCHAR(16)  NOT NULL,           -- 'alert' / 'weekly'
  ref         VARCHAR(64)  NOT NULL,           -- alert id, or the ISO week ('2026-W41')
  channel     VARCHAR(16)  NOT NULL,           -- 'email' / 'slack'
  status      VARCHAR(16)  NOT NULL,           -- 'sending' / 'sent' / 'failed'
  attempts    INT UNSIGNED NOT NULL DEFAULT 0,
  error       VARCHAR(1024) NULL,
  created_at  DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  sent_at     DATETIME     NULL,
  PRIMARY KEY (id),
  UNIQUE KEY uq_notifications_kind_ref_channel (kind, ref, channel)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
