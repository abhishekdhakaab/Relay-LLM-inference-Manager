#!/usr/bin/env python3
"""
LLM Relay — Scheduler Stress Test

Sends bursts of UNIQUE prompts concurrently to bypass cache and force
every request through the scheduler → backend pipeline.

Measures: queue wait, admission control (degrade/reject), throughput,
and p95/p99 under real load.

Usage:
    cd relay
    poetry run python ../scripts/scheduler_stress.py --host http://localhost:8000
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

import httpx
import orjson


def percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f) if f != c else s[f]


# Prompt templates — each call appends a unique suffix to prevent cache hits
SHORT_TEMPLATES = [
    "Explain {topic} in 2 sentences. (request #{uid})",
    "Define {topic} briefly. (request #{uid})",
    "What is {topic}? One paragraph. (request #{uid})",
    "Give a simple explanation of {topic}. (request #{uid})",
]

LONG_TEMPLATES = [
    "Write a detailed technical explanation of {topic}. Cover the core concepts, "
    "how it works internally, common use cases, tradeoffs, and at least 3 specific "
    "examples. Be thorough and technical. (request #{uid})",
]

TOPICS = [
    "consistent hashing", "Raft consensus", "TCP congestion control",
    "database MVCC", "Bloom filters", "skip lists", "LSM trees",
    "gossip protocols", "vector clocks", "CRDTs",
    "B-tree indexing", "write-ahead logging", "connection pooling",
    "circuit breakers", "rate limiting algorithms", "load shedding",
    "backpressure mechanisms", "thread pool tuning", "lock-free queues",
    "memory-mapped I/O", "io_uring", "epoll vs kqueue",
    "TLS handshake", "HTTP/2 multiplexing", "gRPC streaming",
    "service discovery", "leader election", "split-brain resolution",
    "cache invalidation strategies", "read-repair in Cassandra",
]


def make_unique_prompt(uid: int, lane: str = "short") -> str:
    topic = random.choice(TOPICS)
    if lane == "short":
        template = random.choice(SHORT_TEMPLATES)
    else:
        template = random.choice(LONG_TEMPLATES)
    return template.format(topic=topic, uid=uid)


async def send_one(
    client: httpx.AsyncClient, host: str, prompt: str, tenant: str, uid: int
) -> dict[str, Any]:
    payload = {
        "model": "local-ollama",
        "messages": [{"role": "user", "content": prompt}],
    }
    t0 = time.perf_counter()
    try:
        resp = await client.post(
            f"{host}/v1/chat/completions",
            headers={"Content-Type": "application/json", "X-Tenant-Id": tenant},
            content=orjson.dumps(payload),
        )
        dt_ms = (time.perf_counter() - t0) * 1000
        body = resp.json() if resp.content else {}
        tokens = body.get("usage", {}).get("total_tokens", 0)
        return {
            "uid": uid,
            "status": resp.status_code,
            "latency_ms": dt_ms,
            "tokens": tokens,
            "prompt_chars": len(prompt),
        }
    except Exception as e:
        dt_ms = (time.perf_counter() - t0) * 1000
        return {"uid": uid, "status": 0, "latency_ms": dt_ms, "tokens": 0, "error": str(e)}


async def run_wave(
    host: str, n_requests: int, short_pct: float, start_uid: int, timeout: float
) -> list[dict]:
    """Send n_requests concurrently, mix of short/long."""
    prompts = []
    for i in range(n_requests):
        uid = start_uid + i
        lane = "short" if random.random() < short_pct else "long"
        prompt = make_unique_prompt(uid, lane)
        prompts.append((uid, prompt, lane))

    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [
            send_one(client, host, prompt, "default", uid)
            for uid, prompt, _lane in prompts
        ]
        results = await asyncio.gather(*tasks)

    # Attach lane info
    for i, (uid, prompt, lane) in enumerate(prompts):
        results[i]["lane"] = lane
    return list(results)


def summarize(results: list[dict], label: str) -> dict:
    ok = [r for r in results if r["status"] == 200]
    rejected_429 = [r for r in results if r["status"] == 429]
    rejected_503 = [r for r in results if r["status"] == 503]
    errors = [r for r in results if r["status"] not in (200, 429, 503)]

    latencies = [r["latency_ms"] for r in ok]
    all_latencies = [r["latency_ms"] for r in results]

    summary = {
        "label": label,
        "total": len(results),
        "successful": len(ok),
        "rejected_429": len(rejected_429),
        "rejected_503": len(rejected_503),
        "errors": len(errors),
    }

    if latencies:
        summary["latency_ok"] = {
            "mean_ms": round(mean(latencies), 1),
            "p50_ms": round(percentile(latencies, 0.50), 1),
            "p95_ms": round(percentile(latencies, 0.95), 1),
            "p99_ms": round(percentile(latencies, 0.99), 1),
            "max_ms": round(max(latencies), 1),
        }

    if all_latencies:
        summary["latency_all"] = {
            "mean_ms": round(mean(all_latencies), 1),
            "p50_ms": round(percentile(all_latencies, 0.50), 1),
            "p95_ms": round(percentile(all_latencies, 0.95), 1),
            "p99_ms": round(percentile(all_latencies, 0.99), 1),
        }

    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Scheduler Stress Test")
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--out", default="eval/scheduler_stress.json")
    ap.add_argument("--timeout", type=float, default=180.0, help="Per-request timeout seconds")
    args = ap.parse_args()

    print("=" * 60)
    print("LLM Relay — Scheduler Stress Test")
    print(f"Host: {args.host}")
    print("=" * 60)

    # Verify server
    try:
        r = httpx.get(f"{args.host}/health", timeout=5.0)
        assert r.status_code == 200
        print("Server: OK\n")
    except Exception as e:
        print(f"ERROR: Server not reachable: {e}")
        sys.exit(1)

    waves = [
        {"label": "Wave 1 — Light (5 concurrent, 80% short)",   "n": 5,  "short_pct": 0.8},
        {"label": "Wave 2 — Medium (10 concurrent, 70% short)", "n": 10, "short_pct": 0.7},
        {"label": "Wave 3 — Heavy (20 concurrent, 60% short)",  "n": 20, "short_pct": 0.6},
        {"label": "Wave 4 — Burst (30 concurrent, 50% short)",  "n": 30, "short_pct": 0.5},
        {"label": "Wave 5 — Overload (40 concurrent, 50% short)", "n": 40, "short_pct": 0.5},
    ]

    all_results = {}
    uid_counter = 1000

    for wave in waves:
        print(f"\n--- {wave['label']} ---")
        t0 = time.perf_counter()
        results = asyncio.run(
            run_wave(args.host, wave["n"], wave["short_pct"], uid_counter, args.timeout)
        )
        wall_ms = (time.perf_counter() - t0) * 1000
        uid_counter += wave["n"]

        summary = summarize(results, wave["label"])
        summary["wall_time_ms"] = round(wall_ms, 0)
        summary["throughput_rps"] = round(wave["n"] / (wall_ms / 1000), 2) if wall_ms > 0 else 0

        # Print results
        ok = summary["successful"]
        rej429 = summary["rejected_429"]
        rej503 = summary["rejected_503"]
        print(f"  Sent: {wave['n']}  |  OK: {ok}  |  429: {rej429}  |  503: {rej503}")
        print(f"  Wall time: {wall_ms:.0f}ms  |  Throughput: {summary['throughput_rps']} req/s")
        if "latency_ok" in summary:
            lat = summary["latency_ok"]
            print(f"  Latency (OK): p50={lat['p50_ms']:.0f}ms  p95={lat['p95_ms']:.0f}ms  p99={lat['p99_ms']:.0f}ms  max={lat['max_ms']:.0f}ms")

        all_results[wave["label"]] = summary

    # Fetch trace stats to see scheduler behavior
    print("\n\n--- Checking traces for scheduler behavior ---")
    try:
        traces_resp = httpx.get(f"{args.host}/admin/traces.json?limit=200", timeout=10.0)
        traces = traces_resp.json()

        degraded_count = 0
        queue_waits = []
        lanes = {"short": 0, "long": 0}

        for t in traces:
            cache = t.get("cache_json") or {}
            if isinstance(cache, str):
                cache = json.loads(cache)
            sched = cache.get("scheduler", {})

            if sched.get("degraded"):
                degraded_count += 1

            qw = t.get("queue_wait_ms")
            if qw is not None and isinstance(qw, (int, float)):
                queue_waits.append(float(qw))

            plan = t.get("plan_json") or {}
            if isinstance(plan, str):
                plan = json.loads(plan)
            pname = plan.get("plan_name", "")
            if pname in lanes:
                lanes[pname] += 1

        print(f"  Total traces analyzed: {len(traces)}")
        print(f"  Degraded requests (admission control): {degraded_count}")
        print(f"  Routing: short={lanes['short']}, long={lanes['long']}")
        if queue_waits:
            print(f"  Queue wait: mean={mean(queue_waits):.0f}ms  p50={percentile(queue_waits, 0.5):.0f}ms  p95={percentile(queue_waits, 0.95):.0f}ms  max={max(queue_waits):.0f}ms")

        all_results["trace_analysis"] = {
            "total_traces": len(traces),
            "degraded_requests": degraded_count,
            "routing": lanes,
            "queue_wait_ms": {
                "mean": round(mean(queue_waits), 1) if queue_waits else None,
                "p50": round(percentile(queue_waits, 0.5), 1) if queue_waits else None,
                "p95": round(percentile(queue_waits, 0.95), 1) if queue_waits else None,
                "max": round(max(queue_waits), 1) if queue_waits else None,
            },
        }
    except Exception as e:
        print(f"  Could not fetch traces: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SCHEDULER STRESS TEST COMPLETE")
    print("=" * 60)
    print("\nScaling behavior:")
    for wave in waves:
        s = all_results.get(wave["label"], {})
        ok = s.get("successful", 0)
        total = s.get("total", 0)
        rej = s.get("rejected_429", 0) + s.get("rejected_503", 0)
        lat = s.get("latency_ok", {})
        rps = s.get("throughput_rps", 0)
        p95 = lat.get("p95_ms", 0)
        print(f"  {wave['n']:2d} concurrent → {ok}/{total} OK, {rej} rejected, p95={p95:.0f}ms, {rps} req/s")

    # Save report
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nReport saved: {out_path}")


if __name__ == "__main__":
    main()
