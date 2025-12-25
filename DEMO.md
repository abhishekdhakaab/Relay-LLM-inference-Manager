**LLM Relay Platform — End-to-End Demo**

This demo shows how the relay:

* routes requests via policy,
* uses exact + semantic cache,
* controls tail latency with a scheduler,
* records full provenance for every decision.

Everything runs **locally**.

---

## 0) Prerequisites

Make sure you have:

* Docker running
* Ollama running locally
* `.env` configured with:

```env
POLICY_PATH=policies/policy.demo.yaml
```

---

## 1) Start the system

```bash
make dev
```

Wait until you see:

* Postgres healthy
* Redis healthy
* FastAPI listening on `localhost:8000`

Verify health:

```bash
curl -s http://localhost:8000/health
```

Expected:

```json
{"status":"ok"}
```

---

## 2) Open the Trace Viewer

Open in browser:

```
http://localhost:8000/admin/traces
```

Leave this page open — you’ll use it throughout the demo.

---

## 3) Basic OpenAI-compatible request

Send a normal chat completion:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Tenant-Id: default' \
  -d '{
    "model": "local-ollama",
    "messages": [
      {"role": "user", "content": "Explain what an API gateway does in 2 lines."}
    ]
  }' | python -m json.tool
```

### What to show in the trace viewer

Click the newest trace and explain:

* **plan_json**

  * which bucket (short/medium/long)
  * max_tokens / temperature
* **decision_trace_json**

  * why this plan was selected
* **scheduler info**

  * lane (short)
  * queue_wait_ms (likely small)
* **cache_json**

  * no hit on first request

This establishes: *policy-driven routing with explainability*.

---

## 4) Exact cache demo (safe reuse)

Run the same request twice:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Tenant-Id: default' \
  -d '{
    "model": "local-ollama",
    "messages": [
      {"role": "user", "content": "Say hello in exactly 5 words."}
    ]
  }' >/dev/null

curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Tenant-Id: default' \
  -d '{
    "model": "local-ollama",
    "messages": [
      {"role": "user", "content": "Say hello in exactly 5 words."}
    ]
  }' >/dev/null
```

### What to show

Open the second trace:

* `cache_json.exact.hit = true`
* `backend_latency_ms = null`
* latency dramatically lower

Explain:

> Identical prompts are reused safely using an exact cache keyed by tenant + normalized prompt + plan signature.

---

## 5) Semantic cache demo (pgvector)

Send two **similar but not identical** prompts:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Tenant-Id: default' \
  -d '{
    "model": "local-ollama",
    "messages": [
      {"role": "user", "content": "Explain what an API gateway does in 2 lines."}
    ]
  }' >/dev/null

curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Tenant-Id: default' \
  -d '{
    "model": "local-ollama",
    "messages": [
      {"role": "user", "content": "In two lines, what does an API gateway do?"}
    ]
  }' >/dev/null
```

### What to show

Open the second trace:

* `cache_json.semantic.hit = true`
* similarity score (e.g. `0.91`)
* `entry_id` (provenance)
* same plan signature

Explain:

> Semantic reuse is backed by pgvector in Postgres, scoped per tenant and plan, with a similarity threshold and full provenance logging.

---

## 6) Scheduler & tail-latency demo

Generate a burst of concurrent requests:

```bash
for i in {1..25}; do
  curl -s http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -H 'X-Tenant-Id: default' \
    -d '{
      "model": "local-ollama",
      "messages": [
        {"role": "user", "content": "Write one sentence explaining caching."}
      ]
    }' >/dev/null &
done
wait
```

### What to show

In recent traces, point out:

* `scheduler.lane = short`
* `queue_wait_ms` > 0 on some requests
* no system meltdown
* some requests may show:

  * degraded `max_tokens`
  * or early rejection (if queue is full)

Explain:

> Instead of FIFO overload, the relay uses two lanes with fairness and admission control to protect p95/p99 latency.

---

## 7) Long vs short routing

Send a long prompt:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Tenant-Id: default' \
  -d '{
    "model": "local-ollama",
    "messages": [
      {"role": "user", "content": "'"$(python - <<'PY'
print("Explain distributed systems tradeoffs. " * 120)
PY
)"'"}
    ]
  }' >/dev/null
```

### What to show

In the trace:

* routed to `long` lane
* higher `max_tokens`
* higher predicted compute
* separate scheduling from short jobs

Explain:

> Short jobs are protected from long jobs, which is critical for tail latency.

---

## 8) Load test (optional but powerful)

Start Locust:

```bash
make loadtest
```

Open:

```
http://localhost:8089
```

Suggested run:

* Users: 30
* Spawn rate: 10/s
* Duration: ~2 minutes

Afterwards, query metrics from Postgres or just reference the trace viewer.

---

## 9) Regression gates (CI discipline)

Run baseline evaluation:

```bash
make eval_baseline
```

Make a small policy change (e.g., lower semantic threshold), then:

```bash
make eval_candidate
make eval_gate
```

Explain:

> Every change is gated on latency, cost proxy, and quality similarity to prevent silent regressions.

---

## 10) Key takeaway (what to say out loud)

> This project treats LLM inference as a platform problem — combining policy-driven routing, caching with provenance, tail-latency control, and regression discipline — all observable and explainable per request.

