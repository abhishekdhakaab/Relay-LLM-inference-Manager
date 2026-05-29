#!/usr/bin/env python3
"""
LLM Relay — Smoke Test
Verifies the relay is working end-to-end before a full benchmark run.
Tests: health, auth gating, inference, response structure, exact cache,
       response headers (tier + budget), admin auth, stream rejection.

Usage (from repo root):
    cd relay && poetry run python ../scripts/smoke_test.py
    cd relay && poetry run python ../scripts/smoke_test.py --host http://gpu-node:8000

Exit 0 = all checks passed.  Exit 1 = at least one check failed.
"""
from __future__ import annotations

import argparse
import sys
import time

import httpx


def _check(results: list[bool], name: str, ok: bool, detail: str = "") -> bool:
    icon = "✓" if ok else "✗"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{icon}] {name}{suffix}")
    results.append(ok)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--api-key", default="relay-dev-default-key-1234",
                    help="Chat-scoped API key")
    ap.add_argument("--admin-key", default="relay-dev-admin-key-9999",
                    help="Admin-scoped API key (required for /admin/* tests)")
    args = ap.parse_args()

    host = args.host.rstrip("/")
    key = args.api_key
    admin_key = args.admin_key
    auth = {"Authorization": f"Bearer {key}"}
    admin_auth = {"Authorization": f"Bearer {admin_key}"}
    ct = {"Content-Type": "application/json"}
    hdrs = {**auth, **ct}

    print("LLM Relay — Smoke Test")
    print(f"Target: {host}")
    print("=" * 50)

    results: list[bool] = []

    # ── 1. Health endpoint ────────────────────────────────────────────────────
    try:
        r = httpx.get(f"{host}/health", timeout=5)
        _check(results, "Health → 200 {status: ok}",
               r.status_code == 200 and r.json().get("status") == "ok",
               f"HTTP {r.status_code}")
    except Exception as e:
        _check(results, "Health → 200 {status: ok}", False, str(e))
        print("\nFATAL: relay is unreachable. Aborting smoke test.")
        return 1

    # ── 2. Auth: no token → 401 ───────────────────────────────────────────────
    r = httpx.post(f"{host}/v1/chat/completions",
                   headers=ct,
                   json={"model": "local-ollama",
                         "messages": [{"role": "user", "content": "hi"}]},
                   timeout=5)
    _check(results, "No auth token → 401", r.status_code == 401,
           f"HTTP {r.status_code}")

    # ── 3. Wrong scheme → 401 ────────────────────────────────────────────────
    r = httpx.post(f"{host}/v1/chat/completions",
                   headers={**ct, "Authorization": "Basic dXNlcjpwYXNz"},
                   json={"model": "local-ollama",
                         "messages": [{"role": "user", "content": "hi"}]},
                   timeout=5)
    _check(results, "Basic-scheme token → 401", r.status_code == 401,
           f"HTTP {r.status_code}")

    # ── 4. Valid token: live inference ────────────────────────────────────────
    # 300s: on first run, fastembed downloads BAAI/bge-small-en-v1.5 (~30s)
    # and Ollama may cold-load the model (~60-120s on CPU, faster on GPU).
    print("  [~] Live inference (300s timeout; first run may download embedding model)...")
    try:
        r = httpx.post(f"{host}/v1/chat/completions",
                       headers=hdrs,
                       json={"model": "local-ollama",
                             "messages": [{"role": "user", "content": "What is 2+2?"}]},
                       timeout=300)
        ok_status = r.status_code == 200
        body = r.json() if ok_status else {}
        has_choices = bool(body.get("choices"))
        has_content = has_choices and bool(
            body["choices"][0].get("message", {}).get("content", "").strip()
        )
        _check(results, "Live inference → 200", ok_status, f"HTTP {r.status_code}")
        _check(results, "Response has choices[0].message.content", has_content)
    except Exception as e:
        _check(results, "Live inference → 200", False, str(e))
        _check(results, "Response has choices[0].message.content", False)
        r = None

    # ── 5. Response headers: tier + budget ────────────────────────────────────
    if r is not None and r.status_code == 200:
        has_tier = "x-tier-selected" in r.headers
        _check(results, "X-Tier-Selected header present", has_tier,
               r.headers.get("x-tier-selected", "missing"))
        has_budget = "x-budget-remaining" in r.headers
        _check(results, "X-Budget-Remaining header present", has_budget,
               r.headers.get("x-budget-remaining", "missing"))
    else:
        _check(results, "X-Tier-Selected header present", False, "skipped (no 200)")
        _check(results, "X-Budget-Remaining header present", False, "skipped (no 200)")

    # ── 6. Exact cache hit (warm + re-send) ───────────────────────────────────
    warmup = {"model": "local-ollama",
              "messages": [{"role": "user",
                            "content": "What is the capital of France?"}]}
    print("  [~] Warming exact cache...")
    try:
        httpx.post(f"{host}/v1/chat/completions", headers=hdrs,
                   json=warmup, timeout=120)
        t0 = time.perf_counter()
        r2 = httpx.post(f"{host}/v1/chat/completions", headers=hdrs,
                        json=warmup, timeout=10)
        cache_ms = (time.perf_counter() - t0) * 1000
        cache_hit = r2.status_code == 200 and cache_ms < 80
        _check(results, "Exact cache hit < 80ms on repeat request", cache_hit,
               f"{cache_ms:.1f}ms  HTTP {r2.status_code}")
    except Exception as e:
        _check(results, "Exact cache hit < 80ms on repeat request", False, str(e))

    # ── 7. stream=True → 400 ─────────────────────────────────────────────────
    r = httpx.post(f"{host}/v1/chat/completions", headers=hdrs,
                   json={"model": "local-ollama",
                         "messages": [{"role": "user", "content": "hi"}],
                         "stream": True},
                   timeout=5)
    _check(results, "stream=True → 400", r.status_code == 400,
           f"HTTP {r.status_code}")

    # ── 8. Admin: no token → 401 ──────────────────────────────────────────────
    r = httpx.get(f"{host}/admin/traces", timeout=5)
    _check(results, "Admin no token → 401", r.status_code == 401,
           f"HTTP {r.status_code}")

    # ── 9. Admin: admin-scoped key → not 401/403 ──────────────────────────────
    r = httpx.get(f"{host}/admin/traces", headers=admin_auth, timeout=10)
    _check(results, "Admin with admin-scoped key → not 401/403",
           r.status_code not in (401, 403),
           f"HTTP {r.status_code}")

    # ── 10. Admin stats JSON endpoint ─────────────────────────────────────────
    r = httpx.get(f"{host}/admin/stats.json", headers=admin_auth, timeout=10)
    _check(results, "Admin /stats.json → not 401/403",
           r.status_code not in (401, 403),
           f"HTTP {r.status_code}")

    # ── Summary ───────────────────────────────────────────────────────────────
    n_pass = sum(results)
    n_total = len(results)
    label = "PASS" if all(results) else "FAIL"
    print(f"\n{'=' * 50}")
    print(f"{label}: {n_pass}/{n_total} checks passed")

    if not all(results):
        print("\nFailed checks detected. Review relay logs before running the benchmark.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
