#!/usr/bin/env python3
"""
LLM Relay — Unified Benchmark Suite v2

Single script, single dataset, all tests:
  Phase 1: Cold baseline (unique prompts, no cache)
  Phase 2: Exact cache (identical re-sends)
  Phase 3: Semantic cache (easy hits + hard near-misses)
  Phase 4: Burst load (concurrent unique prompts, tests scheduler)
  Phase 5: Scheduler stress (escalating waves of unique prompts)

Usage:
    cd relay
    poetry run python ../scripts/benchmark_v2.py \
        --host http://localhost:8000 \
        --gold ../eval/gold_150.jsonl \
        --out ../eval/full_benchmark.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
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


def load_gold(path: str, tag_filter: Optional[str] = None) -> list[dict]:
    rows = [orjson.loads(l) for l in Path(path).read_bytes().splitlines() if l.strip()]
    if tag_filter:
        rows = [r for r in rows if tag_filter in r.get("tags", [])]
    return rows


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


def summarize_latencies(latencies: list[float]) -> dict:
    if not latencies:
        return {}
    return {
        "count": len(latencies),
        "mean_ms": round(mean(latencies), 1),
        "p50_ms": round(percentile(latencies, 0.50), 1),
        "p90_ms": round(percentile(latencies, 0.90), 1),
        "p95_ms": round(percentile(latencies, 0.95), 1),
        "p99_ms": round(percentile(latencies, 0.99), 1),
        "min_ms": round(min(latencies), 1),
        "max_ms": round(max(latencies), 1),
        "stdev_ms": round(stdev(latencies), 1) if len(latencies) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# Request senders
# ---------------------------------------------------------------------------

def send_sync(client: httpx.Client, host: str, messages: list[dict],
              tenant: str = "default") -> dict:
    payload = {"model": "local-ollama", "messages": messages}
    t0 = time.perf_counter()
    try:
        resp = client.post(
            f"{host}/v1/chat/completions",
            headers={"Content-Type": "application/json", "X-Tenant-Id": tenant},
            content=orjson.dumps(payload),
        )
        dt = (time.perf_counter() - t0) * 1000
        body = resp.json() if resp.content else {}
        return {"status": resp.status_code, "latency_ms": dt, "body": body}
    except Exception as e:
        dt = (time.perf_counter() - t0) * 1000
        return {"status": 0, "latency_ms": dt, "body": {}, "error": str(e)}


async def send_async(client: httpx.AsyncClient, host: str, messages: list[dict],
                     tenant: str = "default") -> dict:
    payload = {"model": "local-ollama", "messages": messages}
    t0 = time.perf_counter()
    try:
        resp = await client.post(
            f"{host}/v1/chat/completions",
            headers={"Content-Type": "application/json", "X-Tenant-Id": tenant},
            content=orjson.dumps(payload),
        )
        dt = (time.perf_counter() - t0) * 1000
        body = resp.json() if resp.content else {}
        return {"status": resp.status_code, "latency_ms": dt, "body": body}
    except Exception as e:
        dt = (time.perf_counter() - t0) * 1000
        return {"status": 0, "latency_ms": dt, "body": {}, "error": str(e)}


# ---------------------------------------------------------------------------
# Phase 1: Cold Baseline
# ---------------------------------------------------------------------------

def phase_cold(client: httpx.Client, host: str, gold_path: str) -> dict:
    """Send all unique prompts with cold cache."""
    print("\n" + "=" * 60)
    print("PHASE 1: Cold Baseline")
    print("=" * 60)

    rows = load_gold(gold_path, "cold")
    # Also send semantic warmup prompts (they'll be in cache for phase 3)
    warmup_hit = load_gold(gold_path, "sem_hit_warmup")
    warmup_miss = load_gold(gold_path, "sem_miss_warmup")
    all_rows = rows + warmup_hit + warmup_miss

    # Deduplicate by content
    seen = set()
    unique = []
    for r in all_rows:
        c = r["messages"][0]["content"]
        if c not in seen:
            seen.add(c)
            unique.append(r)

    latencies, token_counts, items = [], [], []

    for r in unique:
        res = send_sync(client, host, r["messages"], r.get("tenant_id", "default"))
        lat = res["latency_ms"]
        tokens = extract_tokens(res["body"])
        text = extract_text(res["body"])
        latencies.append(lat)
        token_counts.append(tokens["total_tokens"])
        items.append({"id": r["id"], "status": res["status"], "latency_ms": round(lat, 1), "tokens": tokens})
        icon = "+" if res["status"] == 200 else "x"
        print(f"  [{icon}] {r['id']:12s} | {lat:7.0f}ms | {tokens['total_tokens']:4d} tok | {text[:50]}")

    return {
        "phase": "cold_baseline",
        "total_prompts": len(unique),
        "successful": sum(1 for i in items if i["status"] == 200),
        "latency": summarize_latencies(latencies),
        "tokens": {
            "avg_total": round(mean(token_counts), 1) if token_counts else 0,
            "total_generated": sum(token_counts),
        },
        "items": items,
    }


# ---------------------------------------------------------------------------
# Phase 2: Exact Cache
# ---------------------------------------------------------------------------

def phase_exact(client: httpx.Client, host: str, gold_path: str) -> dict:
    """Re-send identical prompts — should hit exact cache."""
    print("\n" + "=" * 60)
    print("PHASE 2: Exact Cache")
    print("=" * 60)

    rows = load_gold(gold_path, "cache_exact")
    latencies, items = [], []
    hits = 0

    for r in rows:
        res = send_sync(client, host, r["messages"], r.get("tenant_id", "default"))
        lat = res["latency_ms"]
        latencies.append(lat)
        is_hit = lat < 50
        if is_hit:
            hits += 1
        items.append({"id": r["id"], "latency_ms": round(lat, 1), "hit": is_hit})
        icon = "$" if is_hit else "."
        print(f"  [{icon}] {r['id']:12s} | {lat:7.1f}ms | {'HIT' if is_hit else 'MISS'}")

    return {
        "phase": "exact_cache",
        "total": len(rows),
        "hits": hits,
        "misses": len(rows) - hits,
        "hit_rate_pct": round(hits / max(len(rows), 1) * 100, 1),
        "latency": summarize_latencies(latencies),
        "items": items,
    }


# ---------------------------------------------------------------------------
# Phase 3: Semantic Cache (hits + near-misses)
# ---------------------------------------------------------------------------

def phase_semantic(client: httpx.Client, host: str, gold_path: str) -> dict:
    """Test semantic cache with easy rephrasings AND hard near-misses."""
    print("\n" + "=" * 60)
    print("PHASE 3: Semantic Cache (hits + near-misses)")
    print("=" * 60)

    rows = load_gold(gold_path, "cache_semantic")

    all_items = []
    hit_items = []
    miss_items = []

    for r in rows:
        expected = "should_hit" if "sem_hit" in r.get("tags", []) else "should_miss"
        res = send_sync(client, host, r["messages"], r.get("tenant_id", "default"))
        lat = res["latency_ms"]
        is_cache_hit = lat < 100  # sub-100ms means it didn't go to backend

        item = {
            "id": r["id"],
            "latency_ms": round(lat, 1),
            "cache_hit": is_cache_hit,
            "expected": expected,
            "correct": (is_cache_hit and expected == "should_hit") or (not is_cache_hit and expected == "should_miss"),
        }
        all_items.append(item)

        if "sem_hit" in r.get("tags", []):
            hit_items.append(item)
        elif "sem_miss" in r.get("tags", []):
            miss_items.append(item)

        icon = "~" if is_cache_hit else "."
        correct = "OK" if item["correct"] else "WRONG"
        print(f"  [{icon}] {r['id']:12s} | {lat:7.1f}ms | {expected:11s} | actual={'HIT' if is_cache_hit else 'MISS':4s} | {correct}")

    # Calculate rates
    easy_hits = sum(1 for i in hit_items if i["cache_hit"])
    hard_misses = sum(1 for i in miss_items if not i["cache_hit"])
    total_hits = sum(1 for i in all_items if i["cache_hit"])
    total_correct = sum(1 for i in all_items if i["correct"])

    easy_latencies = [i["latency_ms"] for i in hit_items]
    hard_latencies = [i["latency_ms"] for i in miss_items]
    all_latencies = [i["latency_ms"] for i in all_items]

    return {
        "phase": "semantic_cache",
        "total_queries": len(all_items),
        "overall_hit_rate_pct": round(total_hits / max(len(all_items), 1) * 100, 1),
        "accuracy_pct": round(total_correct / max(len(all_items), 1) * 100, 1),
        "easy_rephrasings": {
            "total": len(hit_items),
            "hits": easy_hits,
            "hit_rate_pct": round(easy_hits / max(len(hit_items), 1) * 100, 1),
            "latency": summarize_latencies(easy_latencies),
        },
        "hard_near_misses": {
            "total": len(miss_items),
            "correctly_missed": hard_misses,
            "false_hits": len(miss_items) - hard_misses,
            "miss_rate_pct": round(hard_misses / max(len(miss_items), 1) * 100, 1),
            "latency": summarize_latencies(hard_latencies),
        },
        "latency_all": summarize_latencies(all_latencies),
        "items": all_items,
    }


# ---------------------------------------------------------------------------
# Phase 4: Burst Load
# ---------------------------------------------------------------------------

def phase_burst(host: str, gold_path: str) -> dict:
    """Fire burst prompts concurrently — these are unique so they hit backend."""
    print("\n" + "=" * 60)
    print("PHASE 4: Burst Load (concurrent unique prompts)")
    print("=" * 60)

    rows = load_gold(gold_path, "burst")

    async def _run():
        async with httpx.AsyncClient(timeout=120.0) as client:
            tasks = [send_async(client, host, r["messages"], r.get("tenant_id", "default")) for r in rows]
            return await asyncio.gather(*tasks)

    t0 = time.perf_counter()
    results = asyncio.run(_run())
    wall_ms = (time.perf_counter() - t0) * 1000

    latencies = [r["latency_ms"] for r in results]
    statuses = [r["status"] for r in results]

    for i, r in enumerate(results):
        icon = "+" if r["status"] == 200 else "!"
        print(f"  [{icon}] burst-{i+1:02d}  | {r['latency_ms']:7.0f}ms | status={r['status']}")

    return {
        "phase": "burst_load",
        "total": len(rows),
        "successful": statuses.count(200),
        "rejected_429": statuses.count(429),
        "errors": sum(1 for s in statuses if s not in (200, 429, 503)),
        "wall_time_ms": round(wall_ms, 0),
        "throughput_rps": round(len(rows) / (wall_ms / 1000), 2) if wall_ms > 0 else 0,
        "latency": summarize_latencies(latencies),
    }


# ---------------------------------------------------------------------------
# Phase 5: Scheduler Stress
# ---------------------------------------------------------------------------

STRESS_TOPICS = [
    "consistent hashing", "Raft consensus", "TCP congestion control",
    "database MVCC", "Bloom filters", "skip lists", "LSM trees",
    "gossip protocols", "vector clocks", "CRDTs", "B-tree indexing",
    "write-ahead logging", "connection pooling", "circuit breakers",
    "rate limiting algorithms", "load shedding", "backpressure mechanisms",
    "thread pool tuning", "lock-free queues", "memory-mapped I/O",
]

def phase_scheduler_stress(host: str, timeout: float = 180.0) -> dict:
    """Escalating waves of unique concurrent requests."""
    print("\n" + "=" * 60)
    print("PHASE 5: Scheduler Stress (escalating waves)")
    print("=" * 60)

    waves = [
        {"label": "5 concurrent",  "n": 5},
        {"label": "10 concurrent", "n": 10},
        {"label": "20 concurrent", "n": 20},
        {"label": "30 concurrent", "n": 30},
    ]

    uid = 5000
    wave_results = {}

    for wave in waves:
        prompts = []
        for i in range(wave["n"]):
            topic = random.choice(STRESS_TOPICS)
            prompt = f"Explain {topic} in detail. (uid #{uid + i})"
            prompts.append([{"role": "user", "content": prompt}])

        async def _run(ps=prompts):
            async with httpx.AsyncClient(timeout=timeout) as client:
                tasks = [send_async(client, host, p) for p in ps]
                return await asyncio.gather(*tasks)

        print(f"\n  Wave: {wave['label']}")
        t0 = time.perf_counter()
        results = asyncio.run(_run())
        wall_ms = (time.perf_counter() - t0) * 1000
        uid += wave["n"]

        ok = sum(1 for r in results if r["status"] == 200)
        errs = sum(1 for r in results if r["status"] not in (200, 429, 503))
        latencies = [r["latency_ms"] for r in results if r["status"] == 200]

        print(f"    OK: {ok}/{wave['n']}  Errors: {errs}  Wall: {wall_ms:.0f}ms")
        if latencies:
            print(f"    p50={percentile(latencies, 0.5):.0f}ms  p95={percentile(latencies, 0.95):.0f}ms")

        wave_results[wave["label"]] = {
            "total": wave["n"],
            "successful": ok,
            "errors": errs,
            "wall_time_ms": round(wall_ms, 0),
            "latency_ok": summarize_latencies(latencies),
        }

    # Pull trace analysis
    trace_analysis = {}
    try:
        resp = httpx.get(f"{host}/admin/traces.json?limit=500", timeout=10.0)
        traces = resp.json()
        degraded = 0
        queue_waits = []
        for t in traces:
            cache = t.get("cache_json") or {}
            if isinstance(cache, str):
                cache = json.loads(cache)
            sched = cache.get("scheduler", {})
            if sched.get("degraded"):
                degraded += 1
            qw = t.get("queue_wait_ms")
            if qw is not None and isinstance(qw, (int, float)):
                queue_waits.append(float(qw))

        trace_analysis = {
            "total_traces": len(traces),
            "degraded_requests": degraded,
            "degradation_rate_pct": round(degraded / max(len(traces), 1) * 100, 1),
            "queue_wait_ms": summarize_latencies(queue_waits) if queue_waits else {},
        }
        print(f"\n  Trace analysis: {len(traces)} traces, {degraded} degraded ({trace_analysis['degradation_rate_pct']}%)")
    except Exception as e:
        print(f"  Could not fetch traces: {e}")

    return {
        "phase": "scheduler_stress",
        "waves": wave_results,
        "trace_analysis": trace_analysis,
    }


# ---------------------------------------------------------------------------
# Resume bullet generator
# ---------------------------------------------------------------------------

def generate_bullets(report: dict) -> list[str]:
    bullets = []

    cold = report.get("cold_baseline", {})
    n = cold.get("total_prompts", 0)
    lat = cold.get("latency", {})
    bullets.append(
        f"Built a local-first LLM gateway that routes requests through YAML-driven policies and logs every "
        f"decision per request \u2014 tested on a {n}-prompt eval suite"
    )

    exact = report.get("exact_cache", {})
    sem = report.get("semantic_cache", {})
    cold_p50 = lat.get("p50_ms", 1)
    ex_p50 = exact.get("latency", {}).get("p50_ms", 1)
    sem_easy = sem.get("easy_rephrasings", {})
    sem_hard = sem.get("hard_near_misses", {})
    sem_easy_rate = sem_easy.get("hit_rate_pct", 0)
    sem_miss_rate = sem_hard.get("miss_rate_pct", 0)
    overall_sem = sem.get("overall_hit_rate_pct", 0)
    exact_speedup = round(cold_p50 / max(ex_p50, 0.1), 0)

    bullets.append(
        f"Added two-layer caching using Redis for exact matches ({exact.get('hit_rate_pct', 0):.0f}% hit rate, "
        f"{exact_speedup:.0f}x speedup) and pgvector for semantic similarity "
        f"({overall_sem:.0f}% overall hit rate \u2014 {sem_easy_rate:.0f}% on rephrasings, "
        f"{sem_miss_rate:.0f}% correctly rejected on different questions)"
    )

    sched = report.get("scheduler_stress", {})
    trace = sched.get("trace_analysis", {})
    deg_rate = trace.get("degradation_rate_pct", 0)
    bullets.append(
        f"Designed a fair scheduler with two priority lanes and admission control that kept the system alive "
        f"under escalating load (5\u201330 concurrent) \u2014 auto-degraded {deg_rate:.0f}% of requests instead of dropping them"
    )

    return bullets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Unified Benchmark v2")
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--gold", default="eval/gold_150.jsonl")
    ap.add_argument("--out", default="eval/full_benchmark.json")
    ap.add_argument("--skip-stress", action="store_true", help="Skip scheduler stress (saves ~10 min)")
    args = ap.parse_args()

    print("LLM Relay \u2014 Unified Benchmark v2")
    print(f"Host: {args.host}  |  Gold: {args.gold}")
    print("=" * 60)

    try:
        r = httpx.get(f"{args.host}/health", timeout=5.0)
        assert r.status_code == 200
        print("Server: OK")
    except Exception as e:
        print(f"ERROR: Server not reachable: {e}")
        sys.exit(1)

    report: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": args.host,
        "gold_set": args.gold,
    }

    with httpx.Client(timeout=120.0) as client:
        report["cold_baseline"] = phase_cold(client, args.host, args.gold)
        report["exact_cache"] = phase_exact(client, args.host, args.gold)
        report["semantic_cache"] = phase_semantic(client, args.host, args.gold)

    report["burst_load"] = phase_burst(args.host, args.gold)

    if not args.skip_stress:
        report["scheduler_stress"] = phase_scheduler_stress(args.host)

    bullets = generate_bullets(report)
    report["resume_bullets"] = bullets

    # Final summary
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)

    cold = report["cold_baseline"]
    print(f"\nCold baseline: {cold['total_prompts']} prompts, p50={cold['latency']['p50_ms']:.0f}ms")

    ex = report["exact_cache"]
    print(f"Exact cache:   {ex['hit_rate_pct']}% hit rate ({ex['hits']}/{ex['total']}), p50={ex['latency']['p50_ms']:.1f}ms")

    sem = report["semantic_cache"]
    easy = sem["easy_rephrasings"]
    hard = sem["hard_near_misses"]
    print(f"Semantic cache: {sem['overall_hit_rate_pct']}% overall | easy rephrasings: {easy['hit_rate_pct']}% hit | hard near-misses: {hard['miss_rate_pct']}% correctly rejected")

    bl = report["burst_load"]
    print(f"Burst load:    {bl['successful']}/{bl['total']} OK, p95={bl['latency'].get('p95_ms', 0):.0f}ms")

    if "scheduler_stress" in report:
        ta = report["scheduler_stress"].get("trace_analysis", {})
        print(f"Scheduler:     {ta.get('degradation_rate_pct', 0)}% degraded, {ta.get('total_traces', 0)} traces")

    print("\n--- Resume Bullets ---")
    for b in bullets:
        print(f"  \u2022 {b}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nReport: {out_path}")


if __name__ == "__main__":
    main()
