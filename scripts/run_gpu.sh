#!/usr/bin/env bash
# LLM Relay — GPU Cluster Single Entrypoint
#
# Full pipeline from fresh clone to finished benchmark report:
#   1. Validate environment (python3, poetry, ollama)
#   2. Install Python dependencies
#   3. Start infrastructure (Docker Compose or native services)
#   4. Apply all DB migrations (idempotent — safe to re-run)
#   5. Run pytest test suite (79 tests, no running relay needed)
#   6. Start relay server
#   7. Run smoke test (12 checks — aborts pipeline on failure)
#   8. Run full benchmark (phases 6-11) and write JSON report
#
# Usage:
#   bash scripts/run_gpu.sh                    # full pipeline
#   bash scripts/run_gpu.sh --smoke-only       # steps 1-7 only, no benchmark
#   bash scripts/run_gpu.sh --bench-only       # skip smoke test (step 7), do step 8
#   bash scripts/run_gpu.sh --skip-tests       # skip pytest (step 5), useful for re-runs
#   bash scripts/run_gpu.sh --skip-slo-wait    # skip the 90s SLO loop wait in benchmark
#
# Before running:
#   cp .env.example .env   # then edit with your cluster values
#
# Prerequisites (not managed by this script — must already be running):
#   • PostgreSQL accessible at DATABASE_URL
#   • Redis accessible at REDIS_URL
#   • Ollama running with GPU and models loaded: llama3.2:1b  llama3.1:8b
#   • psql CLI installed  (for migrations)
#   • poetry installed    (pip install poetry)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Flags ─────────────────────────────────────────────────────────────────────
SMOKE_ONLY=0
BENCH_ONLY=0
SKIP_TESTS=0
SKIP_SLO=0

for arg in "$@"; do
  case "$arg" in
    --smoke-only)    SMOKE_ONLY=1 ;;
    --bench-only)    BENCH_ONLY=1 ;;
    --skip-tests)    SKIP_TESTS=1 ;;
    --skip-slo-wait) SKIP_SLO=1 ;;
    *)
      echo "Unknown argument: $arg"
      echo "Usage: bash scripts/run_gpu.sh [--smoke-only|--bench-only|--skip-tests|--skip-slo-wait]"
      exit 1
      ;;
  esac
done

# ── Load .env ─────────────────────────────────────────────────────────────────
ENV_FILE="${ROOT}/.env"
if [ ! -f "${ENV_FILE}" ]; then
  echo "ERROR: .env not found at ${ENV_FILE}"
  echo "       Run:  cp .env.example .env  then fill in your cluster values."
  exit 1
fi
set -o allexport
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +o allexport

# Derived config
HOST="http://localhost:${RELAY_PORT:-8000}"
API_KEY="${DEV_KEY_DEFAULT:-relay-dev-default-key-1234}"
ADMIN_KEY="${DEV_KEY_ADMIN:-relay-dev-admin-key-9999}"
GOLD="${ROOT}/eval/gold_150.jsonl"
COST_GOLD="${ROOT}/eval/cost_router_gold.jsonl"
BENCH_OUT="${ROOT}/eval/benchmark_gpu_$(date +%Y%m%d_%H%M%S).json"
RELAY_LOG="${ROOT}/relay_gpu.log"

# psql-compatible URL: strip the +asyncpg SQLAlchemy driver tag
DB_URL_PSQL="${DATABASE_URL//+asyncpg/}"

echo "╔══════════════════════════════════════════════════════╗"
echo "║          LLM Relay — GPU Cluster Run                 ║"
echo "╚══════════════════════════════════════════════════════╝"
echo "  Root      : ${ROOT}"
echo "  Policy    : ${POLICY_PATH:-policies/policy.bench.yaml}"
echo "  Backend   : ${BACKEND_MODE:-ollama}"
echo "  Host      : ${HOST}"
echo "  Report    : ${BENCH_OUT}"
echo ""

# ── [1/8] Validate environment ────────────────────────────────────────────────
echo "── [1/8] Validating environment ──"

command -v python3 &>/dev/null || { echo "ERROR: python3 not found"; exit 1; }
command -v poetry  &>/dev/null || { echo "ERROR: poetry not found. Install: pip install poetry"; exit 1; }

if [ "${BACKEND_MODE:-ollama}" = "ollama" ]; then
  OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
  echo "  Checking Ollama at ${OLLAMA_URL}..."
  if ! curl -sf "${OLLAMA_URL}/api/tags" &>/dev/null; then
    echo "ERROR: Ollama not reachable at ${OLLAMA_URL}"
    echo "       Start Ollama with GPU support and load models:"
    echo "         ollama pull llama3.2:1b && ollama pull llama3.1:8b"
    exit 1
  fi
  echo "  Ollama: OK"
fi
echo "  Environment: OK"

# ── [2/8] Install Python dependencies ────────────────────────────────────────
echo ""
echo "── [2/8] Installing Python dependencies ──"
cd "${ROOT}/relay"
poetry install --no-interaction --quiet
echo "  Dependencies: OK"

# ── [3/8] Infrastructure ──────────────────────────────────────────────────────
echo ""
echo "── [3/8] Infrastructure ──"

if command -v docker &>/dev/null && docker compose version &>/dev/null 2>&1; then
  echo "  Docker found — starting services..."
  docker compose -f "${ROOT}/infra/docker-compose.yml" up -d

  echo "  Waiting for Postgres..."
  for i in $(seq 1 30); do
    docker exec infra-postgres-1 pg_isready -U relay -d relay &>/dev/null 2>&1 \
      && { echo "  Postgres: ready (${i}s)"; break; }
    sleep 1
  done

  echo "  Waiting for Redis..."
  for i in $(seq 1 15); do
    docker exec infra-redis-1 redis-cli ping &>/dev/null 2>&1 \
      && { echo "  Redis: ready (${i}s)"; break; }
    sleep 1
  done
else
  echo "  Docker not found — expecting native Postgres and Redis already running."
  echo "  DB  : ${DB_URL_PSQL}"
  echo "  Redis: ${REDIS_URL:-redis://localhost:6379/0}"
fi

# ── [4/8] DB migrations (idempotent — IF NOT EXISTS throughout) ──────────────
echo ""
echo "── [4/8] Applying DB migrations ──"

apply_migration() {
  local file="$1"
  local name
  name="$(basename "$file")"
  if command -v psql &>/dev/null; then
    # psql accepts the full postgresql://user:pass@host:port/db URI directly.
    # Suppress "already exists" noise; any real error will still print.
    psql "${DB_URL_PSQL}" -f "$file" -q 2>&1 \
      | grep -v "already exists" \
      | grep -v "^$" \
      || true
    echo "  Applied: ${name}"
  else
    echo "  WARN: psql not found — skipping ${name}"
    echo "        Run manually: psql \"${DB_URL_PSQL}\" -f ${file}"
  fi
}

for sql in \
  "${ROOT}/infra/postgres-init/001_init.sql" \
  "${ROOT}/infra/postgres-init/002_cost_router.sql" \
  "${ROOT}/infra/postgres-init/003_prototypes.sql" \
  "${ROOT}/infra/postgres-init/004_auth.sql" \
  "${ROOT}/infra/postgres-init/005_users.sql"
do
  apply_migration "${sql}"
done

# ── [5/8] Pytest test suite ───────────────────────────────────────────────────
echo ""
echo "── [5/8] Running pytest (79 tests) ──"

if [ "${SKIP_TESTS}" -eq 1 ]; then
  echo "  Skipped (--skip-tests)"
else
  cd "${ROOT}/relay"
  if ! poetry run pytest app/tests/ -q --tb=short 2>&1; then
    echo ""
    echo "TESTS FAILED — fix failures before running the benchmark."
    echo "Re-run with --skip-tests to bypass after verifying failures are known."
    exit 1
  fi
fi

# ── [6/8] Start relay ─────────────────────────────────────────────────────────
echo ""
echo "── [6/8] Starting relay ──"
pkill -f "uvicorn app.main:app" 2>/dev/null || true
sleep 1

cd "${ROOT}/relay"
poetry run uvicorn app.main:app \
  --host "${RELAY_HOST:-0.0.0.0}" \
  --port "${RELAY_PORT:-8000}" \
  --workers 1 \
  > "${RELAY_LOG}" 2>&1 &
RELAY_PID=$!
echo "  PID: ${RELAY_PID}  log: ${RELAY_LOG}"

HEALTHY=0
for i in $(seq 1 30); do
  curl -sf "${HOST}/health" &>/dev/null && { HEALTHY=1; echo "  Relay: healthy (${i}s)"; break; }
  kill -0 "${RELAY_PID}" 2>/dev/null || { echo "ERROR: relay died — check ${RELAY_LOG}"; tail -20 "${RELAY_LOG}"; exit 1; }
  sleep 1
done
[ "${HEALTHY}" -eq 0 ] && { echo "ERROR: relay not healthy after 30s"; tail -20 "${RELAY_LOG}"; exit 1; }

# Wait for prototype embeddings background task.
# First run: fastembed downloads BAAI/bge-small-en-v1.5 (~45 MB, up to 3 min).
# Subsequent runs: already in DB, completes in <1s.
echo "  Waiting for prototype embeddings (first run may download ~45 MB model)..."
for i in $(seq 1 150); do
  grep -q "prototype_embeddings" "${RELAY_LOG}" 2>/dev/null && { echo "  Embeddings: ready (${i}s × 2)"; break; }
  sleep 2
done

# ── [7/8] Smoke test ──────────────────────────────────────────────────────────
if [ "${BENCH_ONLY}" -eq 0 ]; then
  echo ""
  echo "── [7/8] Smoke test (12 checks) ──"
  cd "${ROOT}/relay"
  if ! poetry run python "${SCRIPT_DIR}/smoke_test.py" \
      --host "${HOST}" \
      --api-key "${API_KEY}" \
      --admin-key "${ADMIN_KEY}"; then
    echo ""
    echo "SMOKE TEST FAILED — aborting before benchmark."
    echo "Fix the issues above. Use --bench-only to skip smoke test on retry."
    echo "Relay log: ${RELAY_LOG}"
    exit 1
  fi
fi

[ "${SMOKE_ONLY}" -eq 1 ] && { echo ""; echo "Smoke-only run complete."; exit 0; }

# ── [8/8] Full benchmark ──────────────────────────────────────────────────────
echo ""
echo "── [8/8] Full benchmark (phases 6–11) ──"
echo "  Gold      : ${GOLD}"
echo "  Cost-gold : ${COST_GOLD}"
echo "  Output    : ${BENCH_OUT}"

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
echo "╔══════════════════════════════════════════════════════╗"
echo "║  All steps complete.                                 ║"
echo "║  Report: ${BENCH_OUT}"
echo "╚══════════════════════════════════════════════════════╝"
