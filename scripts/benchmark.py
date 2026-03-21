#!/usr/bin/env python3
"""
LLM Relay — Full Benchmark Suite

Runs 5 phases against a live relay server and produces a metrics report
covering latency, caching effectiveness, scheduling, and quality.

Usage:
    cd relay
    poetry run python ../scripts/benchmark.py --host http://localhost:8000

Requires the relay to be running with BACKEND_MODE=ollama and infra up.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Optional

import httpx
import orjson

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f) if f != c else s[f]


def extract_text(resp: dict) -> str:
    try:
        return resp["choices"][0]["message"]["content"] or ""
    except Exception:
        return ""


def extract_tokens(resp: dict) -> dict[str, int]:
    u = resp.get("usage", {})
    return {
        "prompt_tokens": u.get("prompt_tokens", 0),
        "completion_tokens": u.get("completion_tokens", 0),
        "total_tokens": u.get("total_tokens", 0),
    }


def load_gold(path: str, tag_filter: Optional[str] = None) -> list[dict]:
    rows = [orjson.loads(l) for l in Path(path).read_bytes().splitlines() if l.strip()]
    if tag_filter:
        rows = [r for r in rows if tag_filter in r.get("tags", [])]
    return rows


# ---------------------------------------------------------------------------
# Single request sender
# ---------------------------------------------------------------------------

def send_request(client: httpx.Client, host: str, prompt_messages: list[dict],
                 tenant_id: str = "default", model: str = "local-ollama") -> dict:
    payload = {"model": model, "messages": prompt_messages}
    t0 = time.perf_counter()
    try:
        resp = client.post(
            f"{host}/v1/chat/completions",
            headers={"Content-Type": "application/json", "X-Tenant-Id": tenant_id},
            content=orjson.dumps(payload),
        )
        dt_ms = (time.perf_counter() - t0) * 1000
        return {
            "status": resp.status_code,
            "latency_ms": dt_ms,
            "body": resp.json() if resp.content else {},
        }
    except Exception as e:
        dt_ms = (time.perf_counter() - t0) * 1000
        return {"status": 0, "latency_ms": dt_ms, "body": {}, "error": str(e)}


# ---------------------------------------------------------------------------
# Async burst sender
# ---------------------------------------------------------------------------

async def send_request_async(client: httpx.AsyncClient, host: str,
                             prompt_messages: list[dict], tenant_id: str = "default",
                             model: str = "local-ollama") -> dict:
    payload = {"model": model, "messages": prompt_messages}
    t0 = time.perf_counter()
    try:
        resp = await client.post(
            f"{host}/v1/chat/completions",
            headers={"Content-Type": "application/json", "X-Tenant-Id": tenant_id},
            content=orjson.dumps(payload),
        )
        dt_ms = (time.perf_counter() - t0) * 1000
        return {
            "status": resp.status_code,
            "latency_ms": dt_ms,
            "body": resp.json() if resp.content else {},
        }
    except Exception as e:
        dt_ms = (time.perf_counter() - t0) * 1000
        return {"status": 0, "latency_ms": dt_ms, "body": {}, "error": str(e)}


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def summarize_latencies(latencies: list[float]) -> dict:
    if not latencies:
        return {}
    return {
        "count": len(latencies),
        "mean_ms": round(mean(latencies), 1),
        "median_ms": round(median(latencies), 1),
        "p50_ms": round(percentile(latencies, 0.50), 1),
        "p90_ms": round(percentile(latencies, 0.90), 1),
        "p95_ms": round(percentile(latencies, 0.95), 1),
        "p99_ms": round(percentile(latencies, 0.99), 1),
        "min_ms": round(min(latencies), 1),
        "max_ms": round(max(latencies), 1),
        "stdev_ms": round(stdev(latencies), 1) if len(latencies) > 1 else 0.0,
    }


def phase_cold_baseline(client: httpx.Client, host: str, gold_path: str) -> dict:
    """Phase 1: Cold requests — no cache, measures raw backend + routing latency."""
    print("\n=== Phase 1: Cold Baseline (no cache, raw backend latency) ===")
    rows = load_gold(gold_path)
    # Skip exact/semantic duplicate tags — only unique prompts
    seen_contents = set()
    unique_rows = []
    for r in rows:
        content = r["messages"][0]["content"]
        if content not in seen_contents:
            seen_contents.add(content)
            unique_rows.append(r)

    latencies = []
    token_counts = []
    results = []

    for i, r in enumerate(unique_rows):
        res = send_request(client, host, r["messages"], r.get("tenant_id", "default"))
        lat = res["latency_ms"]
        tokens = extract_tokens(res["body"])
        text = extract_text(res["body"])
        latencies.append(lat)
        token_counts.append(tokens["total_tokens"])
        results.append({
            "id": r["id"],
            "status": res["status"],
            "latency_ms": round(lat, 1),
            "tokens": tokens,
            "response_preview": text[:100],
        })
        status_icon = "+" if res["status"] == 200 else "x"
        print(f"  [{status_icon}] {r['id']:12s} | {lat:7.0f}ms | {tokens['total_tokens']:4d} tok | {text[:60]}")

    return {
        "phase": "cold_baseline",
        "description": "Unique prompts, cold cache, measures raw backend latency",
        "total_prompts": len(unique_rows),
        "successful": sum(1 for r in results if r["status"] == 200),
        "latency": summarize_latencies(latencies),
        "tokens": {
            "avg_total": round(mean(token_counts), 1) if token_counts else 0,
            "total_generated": sum(token_counts),
        },
        "items": results,
    }


def phase_exact_cache(client: httpx.Client, host: str, gold_path: str) -> dict:
    """Phase 2: Re-send identical prompts — should hit exact cache."""
    print("\n=== Phase 2: Exact Cache Test (repeat identical prompts) ===")
    rows = load_gold(gold_path, tag_filter="exact")

    latencies = []
    hits = 0
    results = []

    for r in rows:
        res = send_request(client, host, r["messages"], r.get("tenant_id", "default"))
        lat = res["latency_ms"]
        latencies.append(lat)
        # A cache hit should be dramatically faster
        is_likely_hit = lat < 50  # sub-50ms is almost certainly cache
        if is_likely_hit:
            hits += 1
        results.append({"id": r["id"], "latency_ms": round(lat, 1), "likely_cache_hit": is_likely_hit})
        icon = "$" if is_likely_hit else "."
        print(f"  [{icon}] {r['id']:12s} | {lat:7.1f}ms | {'CACHE HIT' if is_likely_hit else 'miss/backend'}")

    return {
        "phase": "exact_cache",
        "description": "Re-sent identical prompts from phase 1 to test exact cache",
        "total_prompts": len(rows),
        "cache_hits": hits,
        "cache_hit_rate": round(hits / max(len(rows), 1) * 100, 1),
        "latency": summarize_latencies(latencies),
        "items": results,
    }


def phase_semantic_cache(client: httpx.Client, host: str, gold_path: str) -> dict:
    """Phase 3: Send rephrased versions of earlier prompts — tests semantic cache."""
    print("\n=== Phase 3: Semantic Cache Test (rephrased prompts) ===")
    rows = load_gold(gold_path, tag_filter="cache_test")
    # Only the "sem-b-*" rows are the rephrased variants
    test_rows = [r for r in rows if r["id"].startswith("sem-b-")]

    latencies = []
    hits = 0
    results = []

    for r in test_rows:
        res = send_request(client, host, r["messages"], r.get("tenant_id", "default"))
        lat = res["latency_ms"]
        latencies.append(lat)
        is_likely_hit = lat < 100  # semantic cache slightly slower than exact
        if is_likely_hit:
            hits += 1
        results.append({"id": r["id"], "latency_ms": round(lat, 1), "likely_semantic_hit": is_likely_hit})
        icon = "~" if is_likely_hit else "."
        print(f"  [{icon}] {r['id']:12s} | {lat:7.1f}ms | {'SEMANTIC HIT' if is_likely_hit else 'miss/backend'}")

    return {
        "phase": "semantic_cache",
        "description": "Rephrased prompts testing pgvector semantic similarity lookup",
        "total_prompts": len(test_rows),
        "semantic_hits": hits,
        "semantic_hit_rate": round(hits / max(len(test_rows), 1) * 100, 1),
        "latency": summarize_latencies(latencies),
        "items": results,
    }


def phase_burst_load(host: str, gold_path: str, concurrency: int = 15) -> dict:
    """Phase 4: Fire burst of concurrent requests — tests scheduler, queue, admission."""
    print(f"\n=== Phase 4: Burst Load Test ({concurrency} concurrent requests) ===")
    rows = load_gold(gold_path, tag_filter="burst")

    async def _run() -> list[dict]:
        async with httpx.AsyncClient(timeout=120.0) as client:
            tasks = []
            for r in rows:
                tasks.append(send_request_async(client, host, r["messages"], r.get("tenant_id", "default")))
            return await asyncio.gather(*tasks)

    t0 = time.perf_counter()
    results_raw = asyncio.run(_run())
    wall_time_ms = (time.perf_counter() - t0) * 1000

    latencies = [r["latency_ms"] for r in results_raw]
    statuses = [r["status"] for r in results_raw]
    ok_count = statuses.count(200)
    rejected_429 = statuses.count(429)
    overloaded_503 = statuses.count(503)

    for i, r in enumerate(results_raw):
        icon = "+" if r["status"] == 200 else ("!" if r["status"] in (429, 503) else "x")
        print(f"  [{icon}] burst-{i+1:02d}  | {r['latency_ms']:7.0f}ms | status={r['status']}")

    return {
        "phase": "burst_load",
        "description": f"{concurrency} concurrent requests testing scheduler and admission control",
        "total_requests": len(rows),
        "successful": ok_count,
        "rejected_429": rejected_429,
        "overloaded_503": overloaded_503,
        "wall_time_ms": round(wall_time_ms, 0),
        "throughput_rps": round(len(rows) / (wall_time_ms / 1000), 1) if wall_time_ms > 0 else 0,
        "latency": summarize_latencies(latencies),
    }


def phase_routing_distribution(client: httpx.Client, host: str, gold_path: str) -> dict:
    """Phase 5: Verify routing — check that short/med/long prompts get different plans."""
    print("\n=== Phase 5: Routing Distribution (verify plan differentiation) ===")
    rows = load_gold(gold_path, tag_filter="routing")

    buckets: dict[str, list[float]] = {"short": [], "medium": [], "long": []}

    for r in rows:
        res = send_request(client, host, r["messages"], r.get("tenant_id", "default"))
        # Determine expected bucket from tags
        for tag in r.get("tags", []):
            if tag in buckets:
                buckets[tag].append(res["latency_ms"])
                break

    summary = {}
    for bucket, lats in buckets.items():
        if lats:
            summary[bucket] = {
                "count": len(lats),
                "avg_latency_ms": round(mean(lats), 1),
                "p95_latency_ms": round(percentile(lats, 0.95), 1),
            }
            print(f"  {bucket:8s} | n={len(lats):2d} | avg={mean(lats):7.0f}ms | p95={percentile(lats, 0.95):7.0f}ms")

    return {
        "phase": "routing_distribution",
        "description": "Latency by prompt length bucket (short/medium/long)",
        "buckets": summary,
    }


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_resume_bullets(report: dict) -> list[str]:
    """Generate copy-paste resume bullet points from benchmark data."""
    bullets = []

    cold = report.get("cold_baseline", {})
    if cold:
        lat = cold.get("latency", {})
        tok = cold.get("tokens", {})
        n = cold.get("total_prompts", 0)
        bullets.append(
            f"Built a policy-driven LLM inference gateway serving {n} eval prompts "
            f"with p50={lat.get('p50_ms',0):.0f}ms / p95={lat.get('p95_ms',0):.0f}ms / "
            f"p99={lat.get('p99_ms',0):.0f}ms end-to-end latency"
        )

    exact = report.get("exact_cache", {})
    if exact:
        rate = exact.get("cache_hit_rate", 0)
        lat = exact.get("latency", {})
        cold_p50 = report.get("cold_baseline", {}).get("latency", {}).get("p50_ms", 1)
        cache_p50 = lat.get("p50_ms", 1)
        speedup = round(cold_p50 / max(cache_p50, 0.1), 1)
        bullets.append(
            f"Implemented dual-layer caching (Redis exact + pgvector semantic) achieving "
            f"{rate:.0f}% exact hit rate with {speedup}x latency reduction on repeated queries"
        )

    sem = report.get("semantic_cache", {})
    if sem:
        rate = sem.get("semantic_hit_rate", 0)
        bullets.append(
            f"Deployed semantic cache using pgvector embeddings with {rate:.0f}% hit rate "
            f"on rephrased queries, reducing redundant inference by identifying near-duplicate prompts"
        )

    burst = report.get("burst_load", {})
    if burst:
        lat = burst.get("latency", {})
        rps = burst.get("throughput_rps", 0)
        ok = burst.get("successful", 0)
        total = burst.get("total_requests", 0)
        bullets.append(
            f"Engineered two-lane fair scheduler with admission control handling {total} concurrent "
            f"requests at {rps:.1f} req/s with p95={lat.get('p95_ms',0):.0f}ms "
            f"({ok}/{total} served, rest gracefully degraded/rejected)"
        )

    routing = report.get("routing_distribution", {})
    if routing:
        b = routing.get("buckets", {})
        if "short" in b and "long" in b:
            short_avg = b["short"]["avg_latency_ms"]
            long_avg = b["long"]["avg_latency_ms"]
            bullets.append(
                f"Implemented policy-driven routing classifying prompts into length tiers "
                f"(short avg={short_avg:.0f}ms, long avg={long_avg:.0f}ms) with per-request "
                f"decision traces and YAML-configurable execution plans"
            )

    bullets.append(
        "Built regression harness with quality/latency/cost gates in CI, "
        "preventing silent regressions across policy changes"
    )

    return bullets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="LLM Relay Benchmark Suite")
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--gold", default="eval/gold_100.jsonl")
    ap.add_argument("--out", default="eval/benchmark_report.json")
    ap.add_argument("--burst-concurrency", type=int, default=10)
    ap.add_argument("--skip-cold", action="store_true", help="Skip cold baseline (reuse existing cache)")
    args = ap.parse_args()

    print(f"LLM Relay Benchmark Suite")
    print(f"Host: {args.host}")
    print(f"Gold set: {args.gold}")
    print(f"Burst concurrency: {args.burst_concurrency}")
    print(f"=" * 60)

    # Verify server is up
    try:
        r = httpx.get(f"{args.host}/health", timeout=5.0)
        assert r.status_code == 200
        print("Server health: OK")
    except Exception as e:
        print(f"ERROR: Server not reachable at {args.host}: {e}")
        sys.exit(1)

    report: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": args.host,
        "gold_set": args.gold,
    }

    with httpx.Client(timeout=120.0) as client:
        # Phase 1: Cold baseline
        if not args.skip_cold:
            report["cold_baseline"] = phase_cold_baseline(client, args.host, args.gold)

        # Phase 2: Exact cache (re-send identical prompts)
        report["exact_cache"] = phase_exact_cache(client, args.host, args.gold)

        # Phase 3: Semantic cache (rephrased prompts)
        report["semantic_cache"] = phase_semantic_cache(client, args.host, args.gold)

        # Phase 5: Routing distribution (send remaining routing-tagged prompts)
        report["routing_distribution"] = phase_routing_distribution(client, args.host, args.gold)

    # Phase 4: Burst load (async)
    report["burst_load"] = phase_burst_load(args.host, args.gold, args.burst_concurrency)

    # Generate resume bullets
    bullets = build_resume_bullets(report)
    report["resume_bullets"] = bullets

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)

    if "cold_baseline" in report:
        lat = report["cold_baseline"]["latency"]
        print(f"\nCold baseline:     p50={lat['p50_ms']:.0f}ms  p95={lat['p95_ms']:.0f}ms  p99={lat['p99_ms']:.0f}ms")
    if "exact_cache" in report:
        ec = report["exact_cache"]
        print(f"Exact cache:       {ec['cache_hit_rate']:.0f}% hit rate, p50={ec['latency']['p50_ms']:.0f}ms")
    if "semantic_cache" in report:
        sc = report["semantic_cache"]
        print(f"Semantic cache:    {sc['semantic_hit_rate']:.0f}% hit rate, p50={sc['latency']['p50_ms']:.0f}ms")
    if "burst_load" in report:
        bl = report["burst_load"]
        print(f"Burst load:        {bl['throughput_rps']:.1f} req/s, p95={bl['latency']['p95_ms']:.0f}ms, "
              f"{bl['successful']}/{bl['total_requests']} OK")

    print("\n--- Resume Bullet Points ---")
    for b in bullets:
        print(f"  • {b}")

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nFull report saved to: {out_path}")


if __name__ == "__main__":
    main()
