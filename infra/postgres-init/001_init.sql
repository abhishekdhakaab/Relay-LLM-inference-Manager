CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- for Semantic vector database
-- pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS semantic_cache_entries (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id TEXT NOT NULL,
  plan_sig TEXT NOT NULL,

  request_hash TEXT,
  prompt_text TEXT,

  -- embedding dimension must match your embedding model output (bge-small-en-v1.5 = 384)
  embedding vector(384) NOT NULL,

  response_json JSONB NOT NULL,

  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  expires_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sem_cache_tenant_plan_exp
  ON semantic_cache_entries (tenant_id, plan_sig, expires_at);


CREATE INDEX IF NOT EXISTS idx_sem_cache_embedding_ivfflat
  ON semantic_cache_entries
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);


--for Request Traces

CREATE TABLE IF NOT EXISTS request_traces (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  request_id TEXT NOT NULL UNIQUE,
  tenant_id TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

  endpoint TEXT NOT NULL,
  model TEXT,
  status_code INT,

  -- normalized request hash for caching/replay alignment
  request_hash TEXT,

  -- timings in ms
  latency_ms INT,
  backend_latency_ms INT,
  queue_wait_ms INT,
  backend_ttft_ms INT,

  -- token counts (best effort)
  prompt_tokens INT,
  completion_tokens INT,
  total_tokens INT,

  -- JSON blobs for rapid iteration (we'll normalize later if needed)
  request_json JSONB,
  response_json JSONB,
  error_json JSONB,

  policy_version TEXT,
  plan_json JSONB,
  decision_trace_json JSONB,
  cache_json JSONB
);

CREATE INDEX IF NOT EXISTS idx_request_traces_created_at ON request_traces (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_request_traces_tenant_id ON request_traces (tenant_id);
CREATE INDEX IF NOT EXISTS idx_request_traces_request_hash ON request_traces (request_hash);
