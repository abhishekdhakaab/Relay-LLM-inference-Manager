# LLM Relay

LLM Relay is a local-first inference gateway that exposes an OpenAI-compatible chat endpoint while routing requests across Ollama models. It combines semantic caching, cost-aware model selection, fair scheduling, token budgets, and production-style observability in one FastAPI service.

The interesting part is the request path: each cache miss is classified using text heuristics and pgvector intent prototypes, assigned the least expensive suitable model tier, admitted against an estimated latency SLO, and recorded as a queryable trace.

## What it demonstrates

- Multi-signal routing across simple, medium, and complex model tiers
- Tenant-isolated exact caching in Redis and semantic caching in PostgreSQL/pgvector
- Two-lane asynchronous scheduling with round-robin tenant fairness
- Redis-backed tenant and user token budgets with atomic reservations
- Circuit breaking and response-quality escalation around Ollama
- OpenTelemetry traces, rolling SLO checks, and lightweight admin dashboards
- Repeatable smoke, regression, stress, and GPU benchmark tooling

## Architecture

```text
Client
  |
  v
FastAPI + bearer auth
  |
  +--> exact cache (Redis) -----------------------------+
  |                                                     |
  +--> embedding --> semantic cache (Postgres/pgvector) +--> response
  |                                                     |
  +--> complexity classifier --> cost router            |
                              |                          |
                              v                          |
                 tenant/user budgets (Redis)            |
                              |                          |
                  fair two-lane scheduler               |
                              |                          |
                    circuit breaker --> Ollama ----------+

Every terminal path writes a PostgreSQL trace and OpenTelemetry span data.
```

The policy files in `policies/` control cache thresholds, model tiers, token limits, SLOs, and admission behavior without changing application code.

## Stack

Python 3.11, FastAPI, Pydantic, Redis, PostgreSQL 16 with pgvector, SQLAlchemy async, Ollama, FastEmbed, OpenTelemetry, Docker Compose, Poetry, pytest, and Locust.

## Run locally

Prerequisites:

- Python 3.11+
- Poetry
- Docker with Compose
- Ollama for real inference; CI and tests can use the mock backend

Create local configuration and start the service:

```bash
cp .env.example .env
docker compose -f infra/docker-compose.yml up -d
cd relay
poetry install
poetry run uvicorn app.main:app --reload
```

The example configuration uses PostgreSQL on host port `5434`, Redis on `6379`, and Ollama on `11434`. Pull the configured models before using the real backend:

```bash
ollama pull llama3.2:1b
ollama pull llama3.1:8b
```

For a dependency-free inference smoke run, set `BACKEND_MODE=mock` in `.env`. The SQL migrations seed public local-development API keys; they are listed in `.env.example` and must be replaced before the service is exposed outside a trusted development machine.

Check liveness:

```bash
curl http://localhost:8000/health
```

Send a chat request:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Authorization: Bearer relay-dev-default-key-1234' \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-ollama","messages":[{"role":"user","content":"Explain semantic caching briefly."}]}'
```

Admin pages are available under `/admin/traces`, `/admin/cost`, and `/admin/slo` with the development admin key.

## Tests and benchmarks

From the repository root:

```bash
make test
make benchmark
```

Useful focused commands include:

```bash
cd relay
poetry run python ../scripts/smoke_test.py
poetry run locust -f ../scripts/locustfile.py --host http://localhost:8000
```

`scripts/run_gpu.sh` can provision a Linux GPU host, start its dependencies, run tests, and execute the full benchmark. It is intentionally more opinionated than the Docker-based local setup.

## Final benchmark

The retained report is [`benchmark_gpu_final_run.json`](benchmark_gpu_final_run.json), recorded on May 30, 2026 against 120 routing prompts and the full regression suite.

| Metric | Result |
|---|---:|
| Cost-router accuracy | 88.2% (97/110 routed requests; 10 cache hits excluded) |
| Estimated cost savings vs. always using the complex tier | 81.2% |
| Exact-cache hit rate | 100% |
| Exact-cache latency | 15.1 ms p50 / 24.0 ms p95 |
| Semantic easy-rephrasing hit rate | 100% |
| Semantic hard-near-miss rejection rate | 80% |
| Burst success rate | 100% |
| Token-budget, SLO, and regression gates | Passed |

The circuit-breaker benchmark validates state and metric instrumentation in mock mode; a full trip-and-recovery test requires deliberately taking a real backend offline.

## Repository layout

```text
relay/app/          API, routing, scheduling, persistence, and telemetry
infra/              Docker Compose and PostgreSQL migrations
policies/           Development, demo, and benchmark policy profiles
eval/               Gold evaluation datasets
scripts/            Smoke, load, regression, and benchmark tools
```
