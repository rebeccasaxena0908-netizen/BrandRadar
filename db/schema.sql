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
  keywords    JSON         NULL,
  first_seen  DATETIME     NULL,
  last_seen   DATETIME     NULL,
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
