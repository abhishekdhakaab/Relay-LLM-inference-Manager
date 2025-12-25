# LLM Relay Platform — Architecture

## What this is
LLM Relay is a policy-driven inference gateway that exposes OpenAI-compatible APIs while optimizing for:
- latency (p95/p99 + TTFT-ready),
- cost proxy (tokens + compute),
- quality stability (regression gates),
- multi-tenant fairness under load.

It treats inference as a platform problem (routing, caching, scheduling, observability), not a single API call.

## Core components

### 1) API Layer (FastAPI)
- OpenAI-compatible endpoint: `/v1/chat/completions`
- Tenant isolation via `X-Tenant-Id`
- Request normalization to canonical form (used for caching + reproducibility)

### 2) Policy Engine (YAML + validation)
- Converts request features into an explicit ExecutionPlan:
  - tier, decoding profile, max_tokens, temperature
  - cache flags (exact/semantic)
  - policy version recorded per request
- Stores a decision trace explaining “why this plan was chosen”.

### 3) Caching
#### Exact Cache (Redis)
- Keyed by tenant + normalized request hash + plan signature
- Safe reuse for identical requests
- Provenance stored in trace

#### Semantic Cache (Postgres + pgvector)
- Stores embeddings + cached responses per tenant + plan signature
- Lookup: nearest vector match + similarity threshold
- Provenance includes similarity score + source entry id

### 4) Scheduler (Tail latency)
- Two-lane queues: short vs long
- Per-tenant fair scheduling (round robin)
- Admission control:
  - degrade max_tokens when predicted SLO miss
  - reject early (429) with retry-after under overload
- Queue wait time recorded in traces

### 5) Observability
- Structured logs with request_id
- Postgres trace store for request/response + plan + cache provenance + timings
- Admin trace viewer: `/admin/traces`

## Data model
- `request_traces`: durable record of every request, including:
  - plan_json, decision_trace_json
  - cache_json (exact + semantic + provenance)
  - timings (latency_ms, backend_latency_ms, queue_wait_ms)
- `semantic_cache_entries`: embedding + response store with expiration and vector index

## Why this design
- Explicit execution plans make optimization controllable and explainable.
- Tail latency is addressed with queue lanes + fairness + admission control.
- Caching is treated as a product feature with provenance and policy knobs.
- Regression harness prevents “silent regressions” in latency/cost/quality.

## Known limitations / next improvements
- Streaming responses + TTFT measurement could be added.
- Verifier mode for semantic cache can add safer reuse for high-risk tenants.
- Admission control estimates can be learned from recent traces instead of constants.
