#!/usr/bin/env bash
# LLM Relay — GPU Cluster Single Entrypoint
#
# Auto-installs ALL missing prerequisites then runs the full pipeline:
#   1. Install missing tools  (poetry, ollama, redis, postgresql+pgvector)
#   2. GPU detection + Ollama (pins to free GPUs, starts daemon, pulls models)
#   3. Python dependencies    (poetry install)
#   4. Start services         (Redis + PostgreSQL)
#   5. DB migrations          (idempotent — safe to re-run)
#   6. Pytest suite           (79 tests)
#   7. Start relay server
#   8. Smoke test             (12 checks)
#   9. Full benchmark         (phases 6-11, writes JSON report)
#
# Usage:
#   bash scripts/run_gpu.sh                    # full pipeline
#   bash scripts/run_gpu.sh --smoke-only       # steps 1-8 only, no benchmark
#   bash scripts/run_gpu.sh --bench-only       # skip smoke test, run benchmark
#   bash scripts/run_gpu.sh --skip-tests       # skip pytest (step 6)
#   bash scripts/run_gpu.sh --skip-slo-wait    # skip 90s SLO wait in benchmark
#
# Before running:
#   cp .env.example .env   # edit with your cluster values
#
# Auto-install support matrix:
#   sudo + apt-get   → Ubuntu / Debian (most GPU clusters)
#   sudo + yum/dnf   → RHEL / CentOS / Rocky
#   conda / mamba    → Anaconda/Miniconda (common on research clusters)
#   No sudo + no conda → Redis built from source; Postgres must be pre-installed

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

# Derived config (set defaults before .env overrides DATABASE_URL below)
HOST="http://localhost:${RELAY_PORT:-8000}"
API_KEY="${DEV_KEY_DEFAULT:-relay-dev-default-key-1234}"
ADMIN_KEY="${DEV_KEY_ADMIN:-relay-dev-admin-key-9999}"
GOLD="${ROOT}/eval/gold_150.jsonl"
COST_GOLD="${ROOT}/eval/cost_router_gold.jsonl"
BENCH_OUT="${ROOT}/eval/benchmark_gpu_$(date +%Y%m%d_%H%M%S).json"
RELAY_LOG="${ROOT}/relay_gpu.log"
DB_URL_PSQL="${DATABASE_URL//+asyncpg/}"

echo "╔══════════════════════════════════════════════════════╗"
echo "║          LLM Relay — GPU Cluster Run                 ║"
echo "╚══════════════════════════════════════════════════════╝"
echo "  Root   : ${ROOT}"
echo "  Policy : ${POLICY_PATH:-policies/policy.bench.yaml}"
echo "  Backend: ${BACKEND_MODE:-ollama}"
echo "  Host   : ${HOST}"
echo "  Report : ${BENCH_OUT}"
echo ""

# ── Detect available package managers ─────────────────────────────────────────
# Extend PATH so user-local binaries (poetry, ollama) are found
export PATH="${HOME}/.local/bin:${PATH}"

HAS_SUDO=0
sudo -n true 2>/dev/null && HAS_SUDO=1

PKG_MGR=""
if   [ "${HAS_SUDO}" -eq 1 ] && command -v apt-get &>/dev/null; then PKG_MGR="apt"
elif [ "${HAS_SUDO}" -eq 1 ] && command -v dnf     &>/dev/null; then PKG_MGR="dnf"
elif [ "${HAS_SUDO}" -eq 1 ] && command -v yum     &>/dev/null; then PKG_MGR="yum"
fi

CONDA_CMD=""
command -v mamba &>/dev/null && CONDA_CMD="mamba"
{ command -v conda &>/dev/null && [ -z "${CONDA_CMD}" ]; } && CONDA_CMD="conda"

# Whether Postgres was freshly installed this run (needs initdb on conda path)
PG_FRESH_INSTALL=0
# How we'll connect to postgres for migrations
PG_MANAGED=0   # 1 = we started postgres ourselves (conda/manual), 0 = system service

# ── Helper functions ───────────────────────────────────────────────────────────

install_pgvector_apt() {
  # Try versioned packages; fall back to building from source
  local pg_ver
  pg_ver=$(psql --version 2>/dev/null | grep -oP '\d+' | head -1)
  sudo apt-get install -y -q "postgresql-${pg_ver}-pgvector" &>/dev/null && return 0

  echo "  pgvector apt package not found — building from source..."
  sudo apt-get install -y -q build-essential postgresql-server-dev-all git 2>/dev/null
  local tmpdir
  tmpdir=$(mktemp -d)
  git clone --quiet --depth 1 https://github.com/pgvector/pgvector.git "${tmpdir}/pgvector"
  cd "${tmpdir}/pgvector" && make -j"$(nproc)" 2>/dev/null && sudo make install
  cd "${ROOT}"
  rm -rf "${tmpdir}"
}

install_redis_source() {
  # Build Redis from source — requires only make and a C compiler.
  echo "  Building Redis from source (no-sudo fallback)..."
  local build_dir="${HOME}/.local/build/redis"
  mkdir -p "${HOME}/.local/bin" "${build_dir}"
  curl -fsSL https://download.redis.io/redis-stable.tar.gz \
    | tar -xz -C "${build_dir}" --strip-components=1
  make -C "${build_dir}" -j"$(nproc)" 2>&1 | tail -2
  cp "${build_dir}/src/redis-server" "${build_dir}/src/redis-cli" "${HOME}/.local/bin/"
  echo "  Redis built → ${HOME}/.local/bin/redis-server"
}

start_redis_native() {
  if redis-cli ping &>/dev/null 2>&1; then
    echo "  Redis: already running"
    return
  fi
  echo "  Starting Redis..."
  redis-server --daemonize yes \
    --loglevel warning \
    --logfile "${ROOT}/redis_gpu.log" \
    --dir /tmp
  sleep 1
  redis-cli ping &>/dev/null \
    && echo "  Redis: started OK" \
    || { echo "ERROR: Redis failed to start — check ${ROOT}/redis_gpu.log"; exit 1; }
}

start_postgres_system() {
  # Used when postgres was installed via apt/yum — it has a system service
  if pg_isready -q 2>/dev/null; then
    echo "  PostgreSQL: already running"
  else
    echo "  Starting PostgreSQL (system service)..."
    sudo systemctl start postgresql 2>/dev/null \
      || sudo service postgresql start 2>/dev/null \
      || { echo "ERROR: could not start postgresql service"; exit 1; }
    for i in $(seq 1 20); do
      pg_isready -q 2>/dev/null && { echo "  PostgreSQL: ready (${i}s)"; break; }
      sleep 1
    done
    pg_isready -q || { echo "ERROR: PostgreSQL did not start in 20s"; exit 1; }
  fi
  # Create relay user + database under the postgres superuser (idempotent).
  # Password is required: Ubuntu/Debian default pg_hba.conf uses scram-sha-256
  # for TCP connections (host 127.0.0.1), so the user must have a password.
  sudo -u postgres psql -c "CREATE USER relay WITH SUPERUSER PASSWORD 'relay';" 2>/dev/null || true
  sudo -u postgres psql -c "CREATE DATABASE relay OWNER relay;" 2>/dev/null || true
  DB_URL_PSQL="postgresql://relay:relay@localhost/relay"
  export DATABASE_URL="postgresql+asyncpg://relay:relay@localhost/relay"
}

start_postgres_managed() {
  # Used when postgres was installed via conda — we run pg_ctl ourselves
  local pgdata="${HOME}/.local/share/relay-pgdata"
  local pg_log="${ROOT}/postgres_gpu.log"

  if pg_isready -q 2>/dev/null; then
    echo "  PostgreSQL: already running"
  else
    if [ ! -d "${pgdata}/global" ]; then
      echo "  Initialising PostgreSQL data directory..."
      initdb -D "${pgdata}" -U relay --auth=trust --no-instructions 2>/dev/null \
        || initdb -D "${pgdata}" -U relay --auth=trust
    fi
    echo "  Starting PostgreSQL..."
    pg_ctl -D "${pgdata}" -l "${pg_log}" start
    for i in $(seq 1 20); do
      pg_isready -q 2>/dev/null && { echo "  PostgreSQL: ready (${i}s)"; break; }
      sleep 1
    done
    pg_isready -q || { echo "ERROR: PostgreSQL did not start — check ${pg_log}"; exit 1; }
  fi
  # Create database (idempotent)
  createdb -U relay relay 2>/dev/null || true
  DB_URL_PSQL="postgresql://relay@localhost:5432/relay"
  export DATABASE_URL="postgresql+asyncpg://relay@localhost:5432/relay"
}

# ── [1/9] Install prerequisites ───────────────────────────────────────────────
echo "── [1/9] Installing prerequisites ──"

# ── poetry ─────────────────────────────────────────────────────────────────────
if ! command -v poetry &>/dev/null; then
  echo "  Installing poetry..."
  pip install --user -q poetry
  hash -r 2>/dev/null || true
fi
echo "  poetry  : $(poetry --version 2>/dev/null)"

# ── pciutils — Ollama installer needs lspci to detect NVIDIA GPUs ──────────────
# Also triggers a re-install of Ollama if it was previously installed without it.
REINSTALL_OLLAMA=0
if ! command -v lspci &>/dev/null; then
  if [ "${PKG_MGR}" = "apt" ] && sudo apt-get install -y -q pciutils &>/dev/null; then
    echo "  pciutils: installed (GPU detection now available)"
    REINSTALL_OLLAMA=1
  elif [ "${PKG_MGR}" = "dnf" ] && sudo dnf install -y -q pciutils &>/dev/null; then
    REINSTALL_OLLAMA=1
  elif [ "${PKG_MGR}" = "yum" ] && sudo yum install -y -q pciutils &>/dev/null; then
    REINSTALL_OLLAMA=1
  fi
fi

# ── Ollama ─────────────────────────────────────────────────────────────────────
# Re-run installer if: (a) ollama missing, or (b) lspci was just installed so
# the installer can now detect and enable CUDA support for H100/A100/etc.
if ! command -v ollama &>/dev/null || [ "${REINSTALL_OLLAMA}" -eq 1 ]; then
  echo "  Installing Ollama (GPU-aware)..."
  curl -fsSL https://ollama.com/install.sh | sh
  hash -r 2>/dev/null || true
fi
echo "  ollama  : OK ($(ollama --version 2>/dev/null | head -1))"

# ── Redis ──────────────────────────────────────────────────────────────────────
if ! command -v redis-server &>/dev/null; then
  echo "  Installing Redis..."
  # Each branch is in the if-condition so a package-manager failure falls through
  # to the next option instead of aborting the script (set -e safe).
  if   [ "${PKG_MGR}" = "apt" ] && sudo apt-get install -y -q redis-server &>/dev/null; then :
  elif [ "${PKG_MGR}" = "dnf" ] && sudo dnf install -y -q redis &>/dev/null;             then :
  elif [ "${PKG_MGR}" = "yum" ] && sudo yum install -y -q redis &>/dev/null;             then :
  elif [ -n "${CONDA_CMD}" ]   && "${CONDA_CMD}" install -y -q -c conda-forge redis;     then :
  else install_redis_source
  fi
  hash -r 2>/dev/null || true
fi
echo "  redis   : OK"

# ── PostgreSQL + pgvector ──────────────────────────────────────────────────────
if ! command -v psql &>/dev/null; then
  echo "  Installing PostgreSQL + pgvector..."
  if [ "${PKG_MGR}" = "apt" ] \
      && sudo apt-get update -qq &>/dev/null \
      && sudo apt-get install -y -q postgresql postgresql-contrib &>/dev/null; then
    install_pgvector_apt
    PG_MANAGED=0
    PG_FRESH_INSTALL=1
  elif [ "${PKG_MGR}" = "dnf" ] \
      && sudo dnf install -y -q postgresql-server postgresql-contrib &>/dev/null; then
    sudo postgresql-setup --initdb 2>/dev/null || true
    PG_MANAGED=0
    PG_FRESH_INSTALL=1
  elif [ "${PKG_MGR}" = "yum" ] \
      && sudo yum install -y -q postgresql-server postgresql-contrib &>/dev/null; then
    sudo postgresql-setup initdb 2>/dev/null || true
    PG_MANAGED=0
    PG_FRESH_INSTALL=1
  elif [ -n "${CONDA_CMD}" ]; then
    echo "  Using conda to install postgresql + pgvector..."
    "${CONDA_CMD}" install -y -q -c conda-forge postgresql pgvector
    hash -r 2>/dev/null || true
    PG_MANAGED=1
    PG_FRESH_INSTALL=1
  else
    echo "ERROR: Cannot install PostgreSQL — no working apt/dnf/yum or conda found."
    echo "       Install PostgreSQL 14+ with pgvector manually, then re-run."
    exit 1
  fi
fi
echo "  psql    : $(psql --version 2>/dev/null)"
echo "  Prerequisites: OK"

# ── [2/9] GPU detection + Ollama ──────────────────────────────────────────────
echo ""
echo "── [2/9] GPU setup ──"

FREE_GPUS=""
if command -v nvidia-smi &>/dev/null; then
  echo "  GPU inventory:"
  nvidia-smi --query-gpu=index,name,memory.used,memory.free \
    --format=csv,noheader | sed 's/^/    /'
  # Collect indices of GPUs with 0 MiB used
  FREE_GPUS=$(nvidia-smi --query-gpu=index,memory.used \
    --format=csv,noheader,nounits \
    | awk -F', ' '$2 == 0 {printf sep $1; sep=","} END{print ""}' \
    | tr -d '[:space:]')
fi

if [ -z "${FREE_GPUS}" ]; then
  echo "  WARNING: No completely free GPUs found."
  echo "           Ollama will use default GPU assignment."
  echo "           To avoid interfering with other jobs, wait for a free GPU"
  echo "           then re-run."
else
  export CUDA_VISIBLE_DEVICES="${FREE_GPUS}"
  echo "  Free GPUs → CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"

if ! curl -sf "${OLLAMA_URL}/api/tags" &>/dev/null; then
  echo "  Starting Ollama..."
  nohup ollama serve > "${ROOT}/ollama_gpu.log" 2>&1 &
  OLLAMA_PID=$!
  echo "  Ollama PID: ${OLLAMA_PID}  log: ${ROOT}/ollama_gpu.log"
  for i in $(seq 1 30); do
    curl -sf "${OLLAMA_URL}/api/tags" &>/dev/null && { echo "  Ollama: ready (${i}s)"; break; }
    sleep 1
  done
  curl -sf "${OLLAMA_URL}/api/tags" &>/dev/null \
    || { echo "ERROR: Ollama did not start — check ${ROOT}/ollama_gpu.log"; exit 1; }
else
  echo "  Ollama: already running"
fi

# Pull models only if not already present
pull_model() {
  local model="$1"
  if ollama list 2>/dev/null | tail -n +2 | awk '{print $1}' | grep -qF "${model}"; then
    echo "  ${model}: already present"
  else
    echo "  Pulling ${model} (first run — may take several minutes)..."
    ollama pull "${model}"
    echo "  ${model}: pulled OK"
  fi
}

pull_model "llama3.2:1b"
pull_model "llama3.1:8b"

# ── [3/9] Python dependencies ─────────────────────────────────────────────────
echo ""
echo "── [3/9] Installing Python dependencies ──"
cd "${ROOT}/relay"
poetry install --no-interaction --quiet
echo "  Dependencies: OK"

# ── [4/9] Start Redis + PostgreSQL ────────────────────────────────────────────
echo ""
echo "── [4/9] Starting services ──"

start_redis_native

if [ "${PG_MANAGED}" -eq 1 ]; then
  start_postgres_managed
elif [ "${PG_FRESH_INSTALL}" -eq 1 ]; then
  start_postgres_system
else
  # Pre-existing postgres — figure out if it's system-managed or user-managed
  if pg_isready -q 2>/dev/null; then
    echo "  PostgreSQL: already running"
  elif command -v pg_ctl &>/dev/null && [ -d "${HOME}/.local/share/relay-pgdata/global" ]; then
    start_postgres_managed
  else
    start_postgres_system
  fi
fi

# Confirm final connection string
echo "  DATABASE_URL: ${DATABASE_URL}"

# ── [5/9] DB migrations ───────────────────────────────────────────────────────
echo ""
echo "── [5/9] Applying DB migrations ──"

apply_migration() {
  local file="$1"
  local name
  name="$(basename "$file")"
  psql "${DB_URL_PSQL}" -f "$file" -q 2>&1 \
    | grep -v "already exists" \
    | grep -v "^$" \
    || true
  echo "  Applied: ${name}"
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

# ── [6/9] Pytest suite ────────────────────────────────────────────────────────
echo ""
echo "── [6/9] Running pytest (79 tests) ──"

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

# ── [7/9] Start relay server ──────────────────────────────────────────────────
echo ""
echo "── [7/9] Starting relay ──"
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
  kill -0 "${RELAY_PID}" 2>/dev/null \
    || { echo "ERROR: relay died — check ${RELAY_LOG}"; tail -20 "${RELAY_LOG}"; exit 1; }
  sleep 1
done
[ "${HEALTHY}" -eq 0 ] && { echo "ERROR: relay not healthy after 30s"; tail -20 "${RELAY_LOG}"; exit 1; }

# Wait for prototype embeddings background task.
# First run: fastembed downloads BAAI/bge-small-en-v1.5 (~45 MB, up to 3 min).
echo "  Waiting for prototype embeddings (first run may download ~45 MB model)..."
for i in $(seq 1 150); do
  grep -q "prototype_embeddings" "${RELAY_LOG}" 2>/dev/null && { echo "  Embeddings: ready (${i}s × 2)"; break; }
  sleep 2
done

# ── [8/9] Smoke test ──────────────────────────────────────────────────────────
if [ "${BENCH_ONLY}" -eq 0 ]; then
  echo ""
  echo "── [8/9] Smoke test (12 checks) ──"
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

# ── [9/9] Full benchmark ──────────────────────────────────────────────────────
echo ""
echo "── [9/9] Full benchmark (phases 6–11) ──"
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
