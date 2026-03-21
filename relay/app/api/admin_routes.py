from __future__ import annotations

import json
from html import escape
from typing import Any

import orjson
from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse, Response

from app.db.traces_read import get_trace, list_traces

admin = APIRouter(prefix="/admin", tags=["admin"])


def _pretty_json(val: Any) -> str:
    try:
        if val is None:
            return "null"
        if isinstance(val, (dict, list)):
            return json.dumps(val, indent=2)
        if isinstance(val, (bytes, bytearray)):
            val = val.decode("utf-8")
        if isinstance(val, str):
            # try parse JSON
            try:
                obj = json.loads(val)
                return json.dumps(obj, indent=2)
            except Exception:
                return val
        return str(val)
    except Exception:
        return str(val)


@admin.get("/traces", response_class=HTMLResponse)
async def traces_page(limit: int = Query(default=50, ge=1, le=500)) -> HTMLResponse:
    rows = await list_traces(limit=limit)

    # Minimal HTML (fast, readable)
    parts = [
        "<html><head><title>Relay Traces</title>"
        "<style>body{font-family:ui-sans-serif,system-ui; padding:16px;} "
        "table{border-collapse:collapse; width:100%;} "
        "th,td{border:1px solid #ddd; padding:8px; font-size:14px;} "
        "th{background:#f6f6f6; text-align:left;} "
        "code{font-family:ui-monospace,Menlo,monospace; font-size:12px;} "
        "</style></head><body>"
    ]
    parts.append(f"<h2>Recent Traces (limit={limit})</h2>")
    parts.append("<p>Tip: open a trace to see routing, cache provenance, scheduler lane, and timings.</p>")
    parts.append("<table>")
    parts.append(
        "<tr><th>created_at</th><th>request_id</th><th>tenant</th><th>status</th>"
        "<th>latency_ms</th><th>queue_wait_ms</th><th>backend_ms</th><th>cache</th><th>plan</th></tr>"
    )

    for r in rows:
        rid = escape(str(r["request_id"]))
        created_at = escape(str(r["created_at"]))
        tenant = escape(str(r["tenant_id"]))
        status = escape(str(r["status_code"]))
        lat = escape(str(r["latency_ms"]))
        qwait = escape(str(r.get("queue_wait_ms")))
        bms = escape(str(r.get("backend_latency_ms")))

        cache = _pretty_json(r.get("cache_json"))
        plan = _pretty_json(r.get("plan_json"))
        cache_short = escape(cache[:120] + ("..." if len(cache) > 120 else ""))
        plan_short = escape(plan[:120] + ("..." if len(plan) > 120 else ""))

        parts.append(
            "<tr>"
            f"<td>{created_at}</td>"
            f"<td><a href='/admin/traces/{rid}'>{rid}</a></td>"
            f"<td>{tenant}</td>"
            f"<td>{status}</td>"
            f"<td>{lat}</td>"
            f"<td>{qwait}</td>"
            f"<td>{bms}</td>"
            f"<td><code>{cache_short}</code></td>"
            f"<td><code>{plan_short}</code></td>"
            "</tr>"
        )

    parts.append("</table>")
    parts.append("<p>JSON endpoints: <code>/admin/traces.json</code>, <code>/admin/traces/{request_id}.json</code></p>")
    parts.append("</body></html>")
    return HTMLResponse("".join(parts))


@admin.get("/traces.json")
async def traces_json(limit: int = Query(default=50, ge=1, le=500)) -> Response:
    rows = await list_traces(limit=limit)
    return Response(content=orjson.dumps(rows), media_type="application/json")


@admin.get("/traces/{request_id}", response_class=HTMLResponse)
async def trace_detail_page(request_id: str) -> HTMLResponse:
    row = await get_trace(request_id)
    if not row:
        return HTMLResponse(f"<h3>Not found</h3><p>{escape(request_id)}</p>", status_code=404)

    def block(title: str, content: Any) -> str:
        return (
            f"<h3>{escape(title)}</h3>"
            f"<pre style='background:#f6f6f6; padding:12px; overflow:auto; border-radius:8px;'>"
            f"{escape(_pretty_json(content))}"
            "</pre>"
        )

    parts = [
        "<html><head><title>Trace Detail</title>"
        "<style>body{font-family:ui-sans-serif,system-ui; padding:16px;} a{color:#0366d6;} "
        "pre{font-family:ui-monospace,Menlo,monospace; font-size:12px;}</style>"
        "</head><body>"
    ]
    parts.append("<p><a href='/admin/traces'>&larr; Back to list</a></p>")
    parts.append(f"<h2>Trace: {escape(request_id)}</h2>")

    # Header summary
    parts.append("<ul>")
    for k in ["created_at", "tenant_id", "endpoint", "model", "status_code", "latency_ms", "queue_wait_ms", "backend_latency_ms"]:
        parts.append(f"<li><b>{escape(k)}</b>: {escape(str(row.get(k)))}</li>")
    parts.append("</ul>")

    # The money sections
    parts.append(block("Plan (plan_json)", row.get("plan_json")))
    parts.append(block("Decision Trace (decision_trace_json)", row.get("decision_trace_json")))
    parts.append(block("Cache Provenance (cache_json)", row.get("cache_json")))

    parts.append(block("Request (request_json)", row.get("request_json")))
    parts.append(block("Response (response_json)", row.get("response_json")))
    parts.append(block("Error (error_json)", row.get("error_json")))

    parts.append(f"<p>JSON endpoint: <code>/admin/traces/{escape(request_id)}.json</code></p>")
    parts.append("</body></html>")
    return HTMLResponse("".join(parts))


@admin.get("/traces/{request_id}.json")
async def trace_detail_json(request_id: str) -> Response:
    row = await get_trace(request_id)
    if not row:
        return Response(content=orjson.dumps({"error": "not_found"}), media_type="application/json", status_code=404)
    return Response(content=orjson.dumps(row), media_type="application/json")


@admin.get("/stats.json")
async def stats_json() -> Response:
    from app.db.traces_read import get_stats
    stats = await get_stats()
    return Response(content=orjson.dumps(stats), media_type="application/json")


@admin.get("/metrics")
async def prometheus_metrics() -> Response:
    from app.db.traces_read import get_stats
    stats = await get_stats()

    lines: list[str] = []

    def gauge(name: str, help_text: str, value: Any) -> None:
        if value is not None:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")

    gauge("relay_requests_total", "Total requests processed", stats.get("total_requests"))
    gauge("relay_requests_ok", "Requests with 200 status", stats.get("ok_requests"))
    gauge("relay_requests_rejected", "Requests rejected (429/503)", stats.get("rejected_requests"))

    for pct in ("p50", "p95", "p99"):
        v = (stats.get("latency") or {}).get(pct)
        gauge(f"relay_latency_{pct}_ms", f"End-to-end latency {pct}", v)

    for pct in ("p50", "p95", "p99"):
        v = (stats.get("backend_latency") or {}).get(pct)
        gauge(f"relay_backend_latency_{pct}_ms", f"Backend latency {pct}", v)

    gauge("relay_cache_exact_hits", "Exact cache hits (estimated)", stats.get("cache_exact_hits"))
    gauge("relay_cache_semantic_hits", "Semantic cache hits (estimated)", stats.get("cache_semantic_hits"))

    gauge("relay_tokens_total", "Total tokens generated", stats.get("total_tokens"))

    for bucket in ("short", "medium", "long"):
        v = (stats.get("routing_distribution") or {}).get(bucket)
        gauge(f"relay_routing_{bucket}_count", f"Requests routed to {bucket} plan", v)

    body = "\n".join(lines) + "\n"
    return Response(content=body, media_type="text/plain; version=0.0.4; charset=utf-8")

