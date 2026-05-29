#!/usr/bin/env bash
# LLM Relay — GPU Cluster Single Entrypoint
#
# Does everything needed to go from a fresh clone to a finished benchmark:
#   1. Validate environment
#   2. Install Python dependencies (poetry)
#   3. Start infrastructure (Docker Compose if available, else checks native services)
#   4. Apply all DB migrations (idempotent)
#   5. Start relay server (background)
#   6. Run smoke test — aborts on failure so you catch issues early
#   7. Run full benchmark suite
#
# Usage:
#   bash scripts/run_gpu.sh                   # full run
#   bash scripts/run_gpu.sh --smoke-only      # validate only, no benchmark
#   bash scripts/run_gpu.sh --bench-only      # skip smoke test
#   bash scripts/run_gpu.sh --skip-slo-wait   # skip the 90s SLO loop wait
#
# Configuration — set these in .env (copy .env.example → .env and edit):
#   DATABASE_URL   REDIS_URL   OLLAMA_BASE_URL   BACKEND_MODE   POLICY_PATH
#
# Prerequisites on GPU cluster (not managed by this script):
#   - PostgreSQL running and accessible
#   - Redis running and accessible
#   - Ollama running with models: llama3.2:1b and llama3.1:8b
#   - psql CLI installed (for migrations)
#   - poetry installed  (pip install poetry)

set -euo pipefail

# ── Locate repo root (script lives in scripts/, repo root is one level up) ───
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Parse flags ───────────────────────────────────────────────────────────────
SMOKE_ONLY=0
BENCH_ONLY=0
SKIP_SLO=0

for arg in "$@"; do
  case "$arg" in
    --smoke-only)   SMOKE_ONLY=1 ;;
    --bench-only)   BENCH_ONLY=1 ;;
    --skip-slo-wait) SKIP_SLO=1 ;;
    *)
      echo "Unknown argument: $arg"
      echo "Usage: bash scripts/run_gpu.sh [--smoke-only|--bench-only|--skip-slo-wait]"
      exit 1
      ;;
  esac
done

# ── Read config from .env (source it so vars are available in this shell too) ─
ENV_FILE="${ROOT}/.env"
if [ ! -f "${ENV_FILE}" ]; then
  echo "WARNING: .env not found at ${ENV_FILE}"
  echo "         Copying .env.example → .env  (fill in your cluster values)"
  cp "${ROOT}/.env.example" "${ENV_FILE}"
  echo "         Edit ${ENV_FILE} before re-running."
  exit 1
fi
set -o allexport
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +o allexport

# Derived defaults (can still be overridden by explicit env vars)
HOST="http://localhost:${RELAY_PORT:-8000}"
POLICY="${POLICY_PATH:-policies/policy.bench.yaml}"
API_KEY="${DEV_KEY_DEFAULT:-relay-dev-default-key-1234}"
ADMIN_KEY="${DEV_KEY_ADMIN:-relay-dev-admin-key-9999}"
GOLD="${ROOT}/eval/gold_150.jsonl"
COST_GOLD="${ROOT}/eval/cost_router_gold.jsonl"
BENCH_OUT="${ROOT}/eval/benchmark_gpu_$(date +%Y%m%d_%H%M%S).json"

# Convert async DB URL to plain psql URL (strip +asyncpg driver tag)
DB_URL_SYNC="${DATABASE_URL//+asyncpg/}"

echo "╔══════════════════════════════════════════════════╗"
echo "║         LLM Relay — GPU Cluster Run              ║"
echo "╚══════════════════════════════════════════════════╝"
echo "  Root      : ${ROOT}"
echo "  Policy    : ${POLICY}"
echo "  Backend   : ${BACKEND_MODE:-ollama}"
echo "  Host      : ${HOST}"
echo "  Output    : ${BENCH_OUT}"
echo ""

# ── 1. Validate environment ───────────────────────────────────────────────────
echo "── [1/6] Validating environment ──"

if ! command -v python3 &>/dev/null; then
  echo "ERROR: python3 not found. Install Python 3.11+."
  exit 1
fi

if ! command -v poetry &>/dev/null; then
  echo "ERROR: poetry not found. Install with: pip install poetry"
  exit 1
fi

BACKEND="${BACKEND_MODE:-ollama}"
if [ "${BACKEND}" = "ollama" ]; then
  OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
  echo "  Checking Ollama at ${OLLAMA_URL}..."
  if ! curl -sf "${OLLAMA_URL}/api/tags" &>/dev/null; then
    echo "ERROR: Ollama is not running at ${OLLAMA_URL}."
    echo "       Start Ollama with GPU support before running this script."
    echo "       Required models: llama3.2:1b  llama3.1:8b"
    exit 1
  fi
  echo "  Ollama: OK"
fi

echo "  Environment: OK"

# ── 2. Install Python dependencies ───────────────────────────────────────────
echo ""
echo "── [2/6] Installing Python dependencies ──"
cd "${ROOT}/relay"
poetry install --no-interaction --quiet
echo "  Dependencies: OK"

# ── 3. Start infrastructure ───────────────────────────────────────────────────
echo ""
echo "── [3/6] Infrastructure ──"

if command -v docker &>/dev/null && docker compose version &>/dev/null 2>&1; then
  echo "  Docker found — starting services via docker compose..."
  docker compose -f "${ROOT}/infra/docker-compose.yml" up -d

  echo "  Waiting for Postgres to be ready..."
  for i in $(seq 1 30); do
    if docker exec infra-postgres-1 pg_isready -U relay -d relay &>/dev/null 2>&1; then
      echo "  Postgres: ready (${i}s)"
      break
    fi
    sleep 1
  done

  echo "  Waiting for Redis to be ready..."
  for i in $(seq 1 15); do
    if docker exec infra-redis-1 redis-cli ping &>/dev/null 2>&1; then
      echo "  Redis: ready (${i}s)"
      break
    fi
    sleep 1
  done
else
  echo "  Docker not found — assuming Postgres and Redis are already running."
  echo "  DB URL : ${DB_URL_SYNC}"
  echo "  Redis  : ${REDIS_URL:-redis://localhost:6379/0}"
fi

# ── 4. Apply DB migrations (idempotent — safe to re-run) ─────────────────────
echo ""
echo "── [4/6] Applying DB migrations ──"

run_migration() {
  local file="$1"
  local name
  name="$(basename "$file")"
  if command -v psql &>/dev/null; then
    if PGPASSWORD="${DB_URL_SYNC##*:}" psql "${DB_URL_SYNC}" -f "$file" -q \
        2>&1 | grep -v "already exists" | grep -v "^$"; then
      :
    fi
    echo "  Applied: ${name}"
  else
    echo "  SKIP: psql not found — run migrations manually:"
    echo "        psql \"\${DATABASE_URL_SYNC}\" -f ${file}"
  fi
}

for sql_file in \
  "${ROOT}/infra/postgres-init/001_init.sql" \
  "${ROOT}/infra/postgres-init/002_cost_router.sql" \
  "${ROOT}/infra/postgres-init/003_prototypes.sql" \
  "${ROOT}/infra/postgres-init/004_auth.sql" \
  "${ROOT}/infra/postgres-init/005_users.sql"; do
  run_migration "${sql_file}"
done

# ── 5. Start relay ────────────────────────────────────────────────────────────
echo ""
echo "── [5/6] Starting relay ──"

# Kill any existing relay on the same port
pkill -f "uvicorn app.main:app" 2>/dev/null || true
sleep 1

cd "${ROOT}/relay"
RELAY_LOG="/tmp/relay_gpu.log"
poetry run uvicorn app.main:app \
  --host "${RELAY_HOST:-0.0.0.0}" \
  --port "${RELAY_PORT:-8000}" \
  --workers 1 \
  > "${RELAY_LOG}" 2>&1 &
RELAY_PID=$!
echo "  Relay PID: ${RELAY_PID}  (log: ${RELAY_LOG})"

echo "  Waiting for relay to be healthy..."
HEALTHY=0
for i in $(seq 1 30); do
  if curl -sf "${HOST}/health" &>/dev/null; then
    HEALTHY=1
    echo "  Relay: healthy (${i}s)"
    break
  fi
  if ! kill -0 "${RELAY_PID}" 2>/dev/null; then
    echo "ERROR: Relay process died. Last 20 lines of log:"
    tail -20 "${RELAY_LOG}"
    exit 1
  fi
  sleep 1
done

if [ "${HEALTHY}" -eq 0 ]; then
  echo "ERROR: Relay did not become healthy within 30s. Log:"
  tail -20 "${RELAY_LOG}"
  exit 1
fi

# Wait for prototype-embedding background task.
# On first run fastembed downloads BAAI/bge-small-en-v1.5 (~30s on GPU,
# up to 3 min on slow network). We poll the relay log for the completion marker.
echo "  Waiting for prototype embeddings (first run may download ~45 MB model)..."
for i in $(seq 1 120); do
  if grep -q "prototype_embeddings" "${RELAY_LOG}" 2>/dev/null; then
    echo "  Embeddings: ready (${i}s)"
    break
  fi
  sleep 2
done

# ── 6. Smoke test ────────────────────────────────────────────────────────────
if [ "${BENCH_ONLY}" -eq 0 ]; then
  echo ""
  echo "── [6a/6] Smoke test ──"
  cd "${ROOT}/relay"
  if ! poetry run python "${SCRIPT_DIR}/smoke_test.py" \
      --host "${HOST}" \
      --api-key "${API_KEY}" \
      --admin-key "${ADMIN_KEY}"; then
    echo ""
    echo "SMOKE TEST FAILED — aborting before benchmark."
    echo "Fix the issues above, then re-run with --bench-only to skip the smoke test."
    echo "Relay log: ${RELAY_LOG}"
    exit 1
  fi
fi

if [ "${SMOKE_ONLY}" -eq 1 ]; then
  echo ""
  echo "Smoke-only mode complete."
  exit 0
fi

# ── 7. Full benchmark ─────────────────────────────────────────────────────────
echo ""
echo "── [6b/6] Full benchmark ──"
echo "  Gold       : ${GOLD}"
echo "  Cost-gold  : ${COST_GOLD}"
echo "  Output     : ${BENCH_OUT}"

SLO_FLAG=""
[ "${SKIP_SLO}" -eq 1 ] && SLO_FLAG="--skip-slo-wait"

cd "${ROOT}/relay"
poetry run python "${SCRIPT_DIR}/benchmark_v3.py" \
  --host "${HOST}" \
  --gold "${GOLD}" \
  --cost-gold "${COST_GOLD}" \
  --out "${BENCH_OUT}" \
  --api-key "${API_KEY}" \
  ${SLO_FLAG}

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  Done. Report written to:                        ║"
echo "║  ${BENCH_OUT}"
echo "╚══════════════════════════════════════════════════╝"
