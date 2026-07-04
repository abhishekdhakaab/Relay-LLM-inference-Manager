-- Migration 004: API key authentication
-- All statements are idempotent.

CREATE TABLE IF NOT EXISTS api_keys (
    id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    key_prefix    VARCHAR(8)  NOT NULL,               -- first 8 chars, shown in UI
    key_hash      VARCHAR(64) NOT NULL UNIQUE,        -- SHA-256 hex of the raw key
    tenant_id     TEXT        NOT NULL,
    description   TEXT,
    scopes        TEXT[]      NOT NULL DEFAULT '{"chat"}',  -- "chat" | "admin"
    is_active     BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_used_at  TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_api_keys_hash     ON api_keys (key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_tenant   ON api_keys (tenant_id);

-- These public fixtures are for local development only. Replace them before
-- exposing the relay to any network you do not control.
--   default tenant:  relay-dev-default-key-1234
--   premium tenant:  relay-dev-premium-key-5678
--   admin access:    relay-dev-admin-key-9999

INSERT INTO api_keys (key_prefix, key_hash, tenant_id, description, scopes)
SELECT 'relay-de',
       encode(sha256('relay-dev-default-key-1234'::bytea), 'hex'),
       'default',
       'Dev key for default tenant',
       ARRAY['chat']
WHERE NOT EXISTS (SELECT 1 FROM api_keys WHERE key_prefix = 'relay-de' AND tenant_id = 'default');

INSERT INTO api_keys (key_prefix, key_hash, tenant_id, description, scopes)
SELECT 'relay-de',
       encode(sha256('relay-dev-premium-key-5678'::bytea), 'hex'),
       'premium',
       'Dev key for premium tenant',
       ARRAY['chat']
WHERE NOT EXISTS (SELECT 1 FROM api_keys WHERE key_prefix = 'relay-de' AND tenant_id = 'premium');

INSERT INTO api_keys (key_prefix, key_hash, tenant_id, description, scopes)
SELECT 'relay-de',
       encode(sha256('relay-dev-admin-key-9999'::bytea), 'hex'),
       'default',
       'Dev admin key',
       ARRAY['chat', 'admin']
WHERE NOT EXISTS (SELECT 1 FROM api_keys WHERE key_prefix = 'relay-de' AND scopes @> ARRAY['admin']);
