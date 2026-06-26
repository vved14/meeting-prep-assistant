-- SETUP FILE — ready to use, no changes needed.
--
-- The seed file (tldv-backfill.seed.sql) is DATA ONLY: it has no CREATE TABLE,
-- and every INSERT ends with
--     ON CONFLICT (tenant_id, source, source_external_ref, occurred_at) DO NOTHING
-- so the load needs a table that already exists with a UNIQUE constraint on those
-- four columns. Without it you get:
--     "there is no unique or exclusion constraint matching the ON CONFLICT spec".
--
-- Column types were read from the seed's INSERT values.
--
-- Run order:
--     createdb lighthouse
--     psql -d lighthouse -f schema.sql
--     psql -d lighthouse -f tldv-backfill.seed.sql

CREATE SCHEMA IF NOT EXISTS core;

CREATE TABLE IF NOT EXISTS core.events (
    tenant_id           uuid        NOT NULL,
    source              text        NOT NULL,
    source_external_ref text        NOT NULL,   -- 'meeting:tldv:<id>'
    kind                text        NOT NULL,   -- e.g. meeting.transcript.ready
    actor_id            uuid,
    occurred_at         timestamptz NOT NULL,
    payload             jsonb       NOT NULL,   -- the meeting data lives in here
    classification      text,
    audience_kind       text,
    personal_owner_id   uuid,
    consent_required    boolean,
    CONSTRAINT events_natural_key
        UNIQUE (tenant_id, source, source_external_ref, occurred_at)
);

-- Speeds up grouping events by meeting and the time filter you will write in db.py.
CREATE INDEX IF NOT EXISTS events_source_ref_idx
    ON core.events (source, source_external_ref);
