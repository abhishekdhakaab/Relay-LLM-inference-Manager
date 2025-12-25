from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import median
from typing import Any

import httpx
import orjson

from app.core.embeddings import embed_text



def percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def cosine(a: list[float], b: list[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    denom = (na**0.5) * (nb**0.5)
    return 0.0 if denom == 0 else dot / denom


def extract_text(resp: dict[str, Any]) -> str:
    try:
        return resp["choices"][0]["message"]["content"] or ""
    except Exception:
        return ""


def extract_tokens(resp: dict[str, Any]) -> int:
    usage = resp.get("usage") or {}
    for k in ("total_tokens", "completion_tokens", "prompt_tokens"):
        v = usage.get(k)
        if isinstance(v, int):
            return v
    # fallback proxy: chars/4 roughly
    txt = extract_text(resp)
    return max(1, len(txt) // 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--gold", default="eval/gold.jsonl")
    ap.add_argument("--out", default="eval/out.json")
    ap.add_argument("--policy-label", default="candidate")
    ap.add_argument("--baseline-out", default="", help="If provided, compute quality similarity vs baseline outputs.")
    args = ap.parse_args()

    gold_path = Path(args.gold)
    out_path = Path(args.out)
    rows = [orjson.loads(line) for line in gold_path.read_bytes().splitlines() if line.strip()]

    baseline_map: dict[str, str] = {}
    if args.baseline_out:
        b = json.loads(Path(args.baseline_out).read_text())
        for item in b.get("items", []):
            baseline_map[item["id"]] = item.get("text", "")

    lat_ms: list[float] = []
    tokens: list[int] = []
    qual_sims: list[float] = []

    items: list[dict[str, Any]] = []

    with httpx.Client(timeout=120.0) as client:
        for r in rows:
            rid = r["id"]
            tenant_id = r.get("tenant_id", "default")
            payload = {"model": "local-ollama", "messages": r["messages"]}

            t0 = time.perf_counter()
            resp = client.post(
                f"{args.host}/v1/chat/completions",
                headers={"Content-Type": "application/json", "X-Tenant-Id": tenant_id},
                content=orjson.dumps(payload),
            )
            dt = (time.perf_counter() - t0) * 1000.0

            ok = resp.status_code == 200
            resp_json = resp.json() if resp.content else {}
            text = extract_text(resp_json)
            tok = extract_tokens(resp_json)

            lat_ms.append(dt)
            tokens.append(tok)

            sim = None
            if rid in baseline_map:
                a = embed_text(text)
                b = embed_text(baseline_map[rid])
                sim = cosine(a, b)
                qual_sims.append(sim)

            items.append(
                {
                    "id": rid,
                    "status": resp.status_code,
                    "latency_ms": dt,
                    "tokens_proxy": tok,
                    "text": text,
                    "quality_similarity_vs_baseline": sim,
                    "error": None if ok else resp_json,
                }
            )

    report = {
        "label": args.policy_label,
        "n": len(items),
        "latency_ms": {
            "p50": percentile(lat_ms, 0.50),
            "p95": percentile(lat_ms, 0.95),
            "p99": percentile(lat_ms, 0.99),
            "median": median(lat_ms) if lat_ms else 0.0,
        },
        "tokens_proxy": {
            "avg": (sum(tokens) / len(tokens)) if tokens else 0.0,
            "p95": percentile([float(t) for t in tokens], 0.95),
        },
        "quality_similarity_vs_baseline": {
            "avg": (sum(qual_sims) / len(qual_sims)) if qual_sims else None,
            "p10": percentile(qual_sims, 0.10) if qual_sims else None,
        },
        "items": items,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote report: {out_path}")


if __name__ == "__main__":
    main()
