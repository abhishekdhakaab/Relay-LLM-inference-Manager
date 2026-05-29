-- Migration 002: Cost Router + quality escalation columns on request_traces,
--                cost_summary aggregation table, and SLO / error tracking columns.
-- All statements are idempotent (IF NOT EXISTS / ADD COLUMN IF NOT EXISTS).

-- ── Cost Router columns ──────────────────────────────────────────────────────
ALTER TABLE request_traces
  ADD COLUMN IF NOT EXISTS complexity_score       FLOAT,
  ADD COLUMN IF NOT EXISTS intent_type            VARCHAR(64),
  ADD COLUMN IF NOT EXISTS tier_selected          VARCHAR(16),
  ADD COLUMN IF NOT EXISTS forced_by_tool_flag    VARCHAR(64),
  ADD COLUMN IF NOT EXISTS needs_computation      BOOLEAN,
  ADD COLUMN IF NOT EXISTS needs_code             BOOLEAN,
  ADD COLUMN IF NOT EXISTS needs_retrieval        BOOLEAN,
  ADD COLUMN IF NOT EXISTS needs_personal_context BOOLEAN,
  ADD COLUMN IF NOT EXISTS is_multi_hop           BOOLEAN,
  ADD COLUMN IF NOT EXISTS actual_cost_usd        FLOAT,
  ADD COLUMN IF NOT EXISTS would_cost_usd         FLOAT,
  ADD COLUMN IF NOT EXISTS escalated              BOOLEAN DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS escalated_from_tier    VARCHAR(16),
  ADD COLUMN IF NOT EXISTS classifier_latency_ms  FLOAT;

-- ── OTel trace correlation ────────────────────────────────────────────────────
ALTER TABLE request_traces
  ADD COLUMN IF NOT EXISTS otel_trace_id VARCHAR(32);

-- ── Error tracking (used by SLO checker) ─────────────────────────────────────
ALTER TABLE request_traces
  ADD COLUMN IF NOT EXISTS error       BOOLEAN DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS error_type  VARCHAR(64);

-- ── Cost summary (pre-aggregated per tenant per time window) ──────────────────
CREATE TABLE IF NOT EXISTS cost_summary (
    id                     SERIAL PRIMARY KEY,
    window_start           TIMESTAMPTZ NOT NULL,
    window_end             TIMESTAMPTZ NOT NULL,
    tenant_id              VARCHAR(128) NOT NULL,
    total_requests         INT         NOT NULL,
    total_actual_cost_usd  FLOAT       NOT NULL,
    total_would_cost_usd   FLOAT       NOT NULL,
    total_savings_usd      FLOAT       NOT NULL,
    savings_pct            FLOAT       NOT NULL,
    escalation_count       INT         NOT NULL,
    simple_count           INT         NOT NULL,
    medium_count           INT         NOT NULL,
    complex_count          INT         NOT NULL,
    created_at             TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cost_summary_tenant_window
    ON cost_summary (tenant_id, window_start DESC);
