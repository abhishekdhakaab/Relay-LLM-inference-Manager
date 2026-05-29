-- Migration 005: Per-user identity within a tenant
-- All statements are idempotent.

CREATE TABLE IF NOT EXISTS users (
    id                    UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id               TEXT        NOT NULL,
    tenant_id             TEXT        NOT NULL,
    display_name          TEXT,
    role                  TEXT        NOT NULL DEFAULT 'member',   -- 'member' | 'admin'
    per_hour_token_limit  INT         NOT NULL DEFAULT 100000,     -- soft cap per user per hour
    is_active             BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at          TIMESTAMPTZ,
    UNIQUE (user_id, tenant_id)
);

CREATE INDEX IF NOT EXISTS idx_users_tenant ON users (tenant_id);

-- Seed users that map to the dev API keys
INSERT INTO users (user_id, tenant_id, display_name, role, per_hour_token_limit)
VALUES
    ('user-alice', 'default', 'Alice (default)', 'admin', 200000),
    ('user-bob',   'default', 'Bob (default)',   'member', 100000),
    ('user-carol', 'premium', 'Carol (premium)', 'admin', 500000)
ON CONFLICT (user_id, tenant_id) DO NOTHING;

-- Add user_id column to request_traces
ALTER TABLE request_traces
    ADD COLUMN IF NOT EXISTS user_id TEXT;

CREATE INDEX IF NOT EXISTS idx_request_traces_user_id ON request_traces (user_id);
