from __future__ import annotations

import json
from html import escape
from typing import Any

import orjson
from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse, Response

from app.auth import require_admin_auth
from app.db.traces_read import get_trace, list_traces

admin = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(require_admin_auth)],
)


# ── Shared layout ─────────────────────────────────────────────────────────────

def _layout(title: str, active: str, content: str) -> str:
    nav_items = [
        ("traces",  "/admin/traces",  "Traces"),
        ("cost",    "/admin/cost",    "Cost"),
        ("slo",     "/admin/slo",     "SLO"),
        ("metrics", "/admin/metrics", "Metrics"),
    ]
    nav_html = "".join(
        f'<a href="{href}" class="nav-item {"active" if key == active else ""}">{label}</a>'
        for key, href, label in nav_items
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{escape(title)} — LLM Relay</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  :root {{
    --bg:          #f9fafb;
    --surface:     #ffffff;
    --border:      #e5e7eb;
    --border-dark: #d1d5db;
    --text-1:      #111827;
    --text-2:      #6b7280;
    --text-3:      #9ca3af;
    --accent:      #2563eb;
    --accent-bg:   #eff6ff;
    --sidebar:     #111827;
    --sidebar-text:#9ca3af;
    --sidebar-act: #ffffff;
    --mono:        'JetBrains Mono', monospace;
    --sans:        'Inter', system-ui, sans-serif;
  }}

  body {{
    font-family: var(--sans);
    background: var(--bg);
    color: var(--text-1);
    display: flex;
    min-height: 100vh;
    font-size: 14px;
    line-height: 1.5;
  }}

  /* Sidebar */
  .sidebar {{
    width: 220px;
    min-height: 100vh;
    background: var(--sidebar);
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
    position: sticky;
    top: 0;
    height: 100vh;
  }}
  .sidebar-logo {{
    padding: 24px 20px 8px;
    font-size: 13px;
    font-weight: 600;
    color: #ffffff;
    letter-spacing: .04em;
    border-bottom: 1px solid #1f2937;
    padding-bottom: 16px;
    margin-bottom: 8px;
  }}
  .sidebar-logo span {{ color: #60a5fa; }}
  .sidebar-section {{
    padding: 6px 12px 2px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: #4b5563;
  }}
  .nav-item {{
    display: flex;
    align-items: center;
    padding: 8px 20px;
    font-size: 13.5px;
    color: var(--sidebar-text);
    text-decoration: none;
    border-radius: 6px;
    margin: 1px 8px;
    transition: background .12s, color .12s;
  }}
  .nav-item:hover {{ background: #1f2937; color: #e5e7eb; }}
  .nav-item.active {{ background: #1d4ed8; color: var(--sidebar-act); font-weight: 500; }}

  /* Main content */
  .main {{ flex: 1; min-width: 0; padding: 32px 40px; max-width: 1200px; }}

  /* Page header */
  .page-header {{ margin-bottom: 28px; }}
  .page-header h1 {{
    font-size: 20px;
    font-weight: 600;
    color: var(--text-1);
  }}
  .page-header p {{ color: var(--text-2); margin-top: 4px; font-size: 13px; }}

  /* Stats row */
  .stats-row {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 16px;
    margin-bottom: 28px;
  }}
  .stat-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px 20px;
  }}
  .stat-label {{
    font-size: 12px;
    font-weight: 500;
    color: var(--text-2);
    text-transform: uppercase;
    letter-spacing: .04em;
    margin-bottom: 6px;
  }}
  .stat-value {{
    font-size: 26px;
    font-weight: 600;
    color: var(--text-1);
    font-family: var(--mono);
  }}
  .stat-sub {{ font-size: 12px; color: var(--text-3); margin-top: 3px; }}

  /* Table */
  .table-wrap {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
  }}
  .table-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 20px;
    border-bottom: 1px solid var(--border);
  }}
  .table-header h2 {{
    font-size: 14px;
    font-weight: 600;
    color: var(--text-1);
  }}
  .table-meta {{ font-size: 12px; color: var(--text-2); }}

  table {{
    width: 100%;
    border-collapse: collapse;
  }}
  thead th {{
    padding: 10px 16px;
    font-size: 11.5px;
    font-weight: 600;
    text-align: left;
    color: var(--text-2);
    text-transform: uppercase;
    letter-spacing: .05em;
    background: #f9fafb;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }}
  tbody tr {{
    border-bottom: 1px solid var(--border);
    transition: background .1s;
  }}
  tbody tr:last-child {{ border-bottom: none; }}
  tbody tr:hover {{ background: #f9fafb; }}
  td {{
    padding: 11px 16px;
    font-size: 13px;
    color: var(--text-1);
    vertical-align: middle;
  }}

  /* Mono values */
  .mono {{ font-family: var(--mono); font-size: 12px; }}

  /* Badges */
  .badge {{
    display: inline-flex;
    align-items: center;
    padding: 2px 8px;
    border-radius: 99px;
    font-size: 11.5px;
    font-weight: 500;
    white-space: nowrap;
  }}
  .badge-green  {{ background:#dcfce7; color:#15803d; }}
  .badge-red    {{ background:#fee2e2; color:#b91c1c; }}
  .badge-amber  {{ background:#fef9c3; color:#92400e; }}
  .badge-blue   {{ background:#dbeafe; color:#1d4ed8; }}
  .badge-gray   {{ background:#f3f4f6; color:#6b7280; }}

  /* Links */
  a {{ color: var(--accent); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}

  /* Detail blocks */
  .detail-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-bottom: 24px;
  }}
  .detail-block {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
  }}
  .detail-block-title {{
    padding: 12px 16px;
    font-size: 12px;
    font-weight: 600;
    color: var(--text-2);
    text-transform: uppercase;
    letter-spacing: .05em;
    background: #f9fafb;
    border-bottom: 1px solid var(--border);
  }}
  pre.detail-pre {{
    padding: 14px 16px;
    font-family: var(--mono);
    font-size: 12px;
    line-height: 1.6;
    overflow: auto;
    max-height: 260px;
    color: var(--text-1);
    white-space: pre-wrap;
    word-break: break-word;
  }}
</style>
</head>
<body>
  <aside class="sidebar">
    <div class="sidebar-logo">LLM <span>Relay</span></div>
    <div class="sidebar-section">Observability</div>
    {nav_html}
  </aside>
  <main class="main">
    {content}
  </main>
</body>
</html>"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pretty_json(val: Any) -> str:
    try:
        if val is None:
            return "null"
        if isinstance(val, (dict, list)):
            return json.dumps(val, indent=2)
        if isinstance(val, (bytes, bytearray)):
            val = val.decode("utf-8")
        if isinstance(val, str):
            try:
                return json.dumps(json.loads(val), indent=2)
            except Exception:
                return val
        return str(val)
    except Exception:
        return str(val)


def _status_badge(code: Any) -> str:
    code = int(code) if code else 0
    if code == 200:
        return '<span class="badge badge-green">200 OK</span>'
    if code in (429, 503):
        return f'<span class="badge badge-amber">{code}</span>'
    if code >= 500:
        return f'<span class="badge badge-red">{code}</span>'
    return f'<span class="badge badge-gray">{code}</span>'


def _latency_badge(ms: Any) -> str:
    if ms is None:
        return '<span class="mono" style="color:#9ca3af">—</span>'
    ms = int(ms)
    cls = "badge-green" if ms < 1000 else ("badge-amber" if ms < 5000 else "badge-red")
    return f'<span class="badge {cls}">{ms:,} ms</span>'


def _cache_badge(cache_json: Any) -> str:
    try:
        obj = json.loads(cache_json) if isinstance(cache_json, str) else (cache_json or {})
        if obj.get("exact", {}).get("hit"):
            return '<span class="badge badge-blue">exact</span>'
        if obj.get("semantic", {}).get("hit"):
            return '<span class="badge badge-blue">semantic</span>'
    except Exception:
        pass
    return '<span class="badge badge-gray">miss</span>'


# ── Routes ────────────────────────────────────────────────────────────────────

@admin.get("/traces", response_class=HTMLResponse)
async def traces_page(limit: int = Query(default=50, ge=1, le=500)) -> HTMLResponse:
    rows = await list_traces(limit=limit)

    rows_html = ""
    for r in rows:
        rid     = escape(str(r["request_id"]))
        ts      = escape(str(r.get("created_at", ""))[:19].replace("T", " "))
        tenant  = escape(str(r.get("tenant_id", "—")))
        lat     = _latency_badge(r.get("latency_ms"))
        backend = _latency_badge(r.get("backend_latency_ms"))
        qwait   = _latency_badge(r.get("queue_wait_ms"))
        status  = _status_badge(r.get("status_code"))
        cache   = _cache_badge(r.get("cache_json"))
        plan_raw = _pretty_json(r.get("plan_json"))
        plan_short = escape(plan_raw[:60] + ("…" if len(plan_raw) > 60 else ""))
        rows_html += f"""<tr>
          <td class="mono" style="color:#6b7280">{ts}</td>
          <td><a href="/admin/traces/{rid}" class="mono" style="font-size:11px">{rid[:16]}…</a></td>
          <td><span class="badge badge-gray">{tenant}</span></td>
          <td>{status}</td>
          <td>{lat}</td>
          <td>{qwait}</td>
          <td>{backend}</td>
          <td>{cache}</td>
          <td class="mono" style="font-size:11px;color:#6b7280">{plan_short}</td>
        </tr>"""

    content = f"""
    <div class="page-header">
      <h1>Request Traces</h1>
      <p>Last {limit} requests — click a row to inspect routing, cache provenance, and timings.</p>
    </div>
    <div class="table-wrap">
      <div class="table-header">
        <h2>Traces</h2>
        <span class="table-meta">{len(rows)} rows &nbsp;·&nbsp;
          <a href="/admin/traces.json">JSON export</a>
        </span>
      </div>
      <table>
        <thead><tr>
          <th>Time</th><th>Request ID</th><th>Tenant</th><th>Status</th>
          <th>Latency</th><th>Queue wait</th><th>Backend</th><th>Cache</th><th>Plan</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>"""
    return HTMLResponse(_layout(f"Traces (last {limit})", "traces", content))


@admin.get("/traces.json")
async def traces_json(limit: int = Query(default=50, ge=1, le=500)) -> Response:
    rows = await list_traces(limit=limit)
    return Response(content=orjson.dumps(rows), media_type="application/json")


@admin.get("/traces/{request_id}", response_class=HTMLResponse)
async def trace_detail_page(request_id: str) -> HTMLResponse:
    row = await get_trace(request_id)
    if not row:
        content = f"<div class='page-header'><h1>Trace not found</h1><p>{escape(request_id)}</p></div>"
        return HTMLResponse(_layout("Not found", "traces", content), status_code=404)

    meta_items = [
        ("Time",           str(row.get("created_at", ""))[:19].replace("T", " ")),
        ("Tenant",         str(row.get("tenant_id", "—"))),
        ("User",           str(row.get("user_id", "—"))),
        ("Status",         str(row.get("status_code", "—"))),
        ("End-to-end",     f"{row.get('latency_ms', '—')} ms"),
        ("Queue wait",     f"{row.get('queue_wait_ms', '—')} ms"),
        ("Backend",        f"{row.get('backend_latency_ms', '—')} ms"),
        ("Model",          str(row.get("model", "—"))),
        ("Policy version", str(row.get("policy_version", "—"))),
    ]
    meta_html = "".join(
        f"<tr><td style='color:#6b7280;padding:7px 0;width:140px'>{escape(k)}</td>"
        f"<td style='padding:7px 0;font-family:var(--mono);font-size:12px'>{escape(v)}</td></tr>"
        for k, v in meta_items
    )

    def block(title: str, val: Any) -> str:
        return (
            f'<div class="detail-block">'
            f'<div class="detail-block-title">{escape(title)}</div>'
            f'<pre class="detail-pre">{escape(_pretty_json(val))}</pre>'
            f'</div>'
        )

    content = f"""
    <div class="page-header">
      <h1>Trace detail</h1>
      <p><a href="/admin/traces">&larr; Back to traces</a>
         &nbsp;·&nbsp; <a href="/admin/traces/{escape(request_id)}.json">JSON</a></p>
    </div>
    <div style="background:var(--surface);border:1px solid var(--border);border-radius:10px;
                padding:20px 24px;margin-bottom:24px;">
      <table style="width:100%;border-collapse:collapse">{meta_html}</table>
    </div>
    <div class="detail-grid">
      {block("Plan", row.get("plan_json"))}
      {block("Decision trace", row.get("decision_trace_json"))}
      {block("Cache provenance", row.get("cache_json"))}
      {block("Error", row.get("error_json"))}
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px">
      {block("Request", row.get("request_json"))}
      {block("Response", row.get("response_json"))}
    </div>"""
    return HTMLResponse(_layout(f"Trace {request_id[:16]}…", "traces", content))


@admin.get("/traces/{request_id}.json")
async def trace_detail_json(request_id: str) -> Response:
    row = await get_trace(request_id)
    if not row:
        return Response(content=orjson.dumps({"error": "not_found"}),
                        media_type="application/json", status_code=404)
    return Response(content=orjson.dumps(row), media_type="application/json")


@admin.get("/stats.json")
async def stats_json() -> Response:
    from app.db.traces_read import get_stats
    stats = await get_stats()
    return Response(content=orjson.dumps(stats), media_type="application/json")


@admin.get("/cost", response_class=HTMLResponse)
async def cost_dashboard() -> HTMLResponse:
    from sqlalchemy import text as sa_text
    from app.db.postgres import get_sessionmaker

    async with get_sessionmaker()() as session:
        res = await session.execute(sa_text("""
            SELECT
                tenant_id, tier_selected,
                COUNT(*)                                                       AS requests,
                COALESCE(SUM(actual_cost_usd), 0)                             AS total_cost_usd,
                COALESCE(SUM(would_cost_usd), 0)                              AS would_cost_usd,
                SUM(CASE WHEN escalated THEN 1 ELSE 0 END)                    AS escalations,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY classifier_latency_ms)
                    FILTER (WHERE classifier_latency_ms IS NOT NULL)           AS clf_p50_ms,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY classifier_latency_ms)
                    FILTER (WHERE classifier_latency_ms IS NOT NULL)           AS clf_p95_ms
            FROM request_traces
            WHERE created_at > NOW() - INTERVAL '24 hours'
              AND tier_selected IS NOT NULL
            GROUP BY tenant_id, tier_selected
            ORDER BY tenant_id, tier_selected
        """))
        rows = res.mappings().all()

    total_cost    = sum(r["total_cost_usd"] for r in rows)
    total_would   = sum(r["would_cost_usd"] for r in rows)
    total_savings = total_would - total_cost
    savings_pct   = (total_savings / total_would * 100) if total_would > 0 else 0

    stats_html = f"""
    <div class="stats-row">
      <div class="stat-card">
        <div class="stat-label">Actual cost (24h)</div>
        <div class="stat-value">${total_cost:.4f}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Would-have cost</div>
        <div class="stat-value">${total_would:.4f}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Savings</div>
        <div class="stat-value">${total_savings:.4f}</div>
        <div class="stat-sub">{savings_pct:.1f}% reduction</div>
      </div>
    </div>"""

    rows_html = ""
    for r in rows:
        cost    = r["total_cost_usd"]
        would   = r["would_cost_usd"]
        savings = would - cost
        esc_rate = (r["escalations"] / max(r["requests"], 1)) * 100
        esc_badge = (
            '<span class="badge badge-green">' if esc_rate < 5 else
            ('<span class="badge badge-amber">' if esc_rate < 15 else '<span class="badge badge-red">')
        ) + f'{esc_rate:.1f}%</span>'
        p50 = f"{r['clf_p50_ms']:.1f} ms" if r["clf_p50_ms"] else "—"
        p95 = f"{r['clf_p95_ms']:.1f} ms" if r["clf_p95_ms"] else "—"
        rows_html += (
            f"<tr>"
            f"<td><span class='badge badge-gray'>{escape(str(r['tenant_id']))}</span></td>"
            f"<td><span class='badge badge-blue'>{escape(str(r['tier_selected']))}</span></td>"
            f"<td class='mono'>{r['requests']:,}</td>"
            f"<td class='mono'>${cost:.6f}</td>"
            f"<td class='mono'>${would:.6f}</td>"
            f"<td class='mono'>${savings:.6f}</td>"
            f"<td>{esc_badge}</td>"
            f"<td class='mono'>{p50}</td>"
            f"<td class='mono'>{p95}</td>"
            f"</tr>"
        )

    content = f"""
    <div class="page-header">
      <h1>Cost Router</h1>
      <p>Per-tier cost breakdown and classifier performance — last 24 hours.</p>
    </div>
    {stats_html}
    <div class="table-wrap">
      <div class="table-header"><h2>Tier breakdown</h2></div>
      <table>
        <thead><tr>
          <th>Tenant</th><th>Tier</th><th>Requests</th>
          <th>Actual ($)</th><th>Would-have ($)</th><th>Savings ($)</th>
          <th>Escalation rate</th><th>Clf p50</th><th>Clf p95</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>"""
    return HTMLResponse(_layout("Cost Router", "cost", content))


@admin.get("/slo", response_class=HTMLResponse)
async def slo_dashboard() -> HTMLResponse:
    import time as _time
    from app.core.slo_checker import get_slo_gauges, get_last_checked_ts
    from app.core.settings import settings

    gauges      = get_slo_gauges()
    last_ts     = get_last_checked_ts()
    policy      = settings.load_policy()
    last_str    = f"{int(_time.time() - last_ts)}s ago" if last_ts > 0 else "not yet run"

    rows_html = ""
    for tenant_id, tenant_cfg in policy.tenants.items():
        slo = tenant_cfg.slo

        def gv_status(metric: str) -> tuple[str, str]:
            for status in ("breached", "ok"):
                key = f'{metric}{{tenant_id="{tenant_id}",status="{status}"}}'
                if key in gauges:
                    return f"{gauges[key]:.0f}", status
            return "—", "unknown"

        health_val = gauges.get(f'slo_health{{tenant_id="{tenant_id}"}}')
        if health_val is None:
            health_badge = '<span class="badge badge-gray">no data</span>'
        elif health_val >= 1.0:
            health_badge = '<span class="badge badge-green">Healthy</span>'
        else:
            health_badge = '<span class="badge badge-red">Breached</span>'

        def metric_cell(metric: str, threshold: int) -> str:
            val, status = gv_status(metric)
            cls = "badge-red" if status == "breached" else ("badge-green" if status == "ok" else "badge-gray")
            return (
                f'<td><span class="badge {cls} mono">{val}</span>'
                f'<span style="color:#9ca3af;font-size:11px;margin-left:4px">/ {threshold}</span></td>'
            )

        rows_html += (
            f"<tr>"
            f"<td><span class='badge badge-gray'>{escape(tenant_id)}</span></td>"
            f"<td>{health_badge}</td>"
            + metric_cell("slo_latency_p50_ms", slo.p50_ms)
            + metric_cell("slo_latency_p95_ms", slo.p95_ms)
            + metric_cell("slo_latency_p99_ms", slo.p99_ms)
            + metric_cell("slo_error_rate_pct", int(slo.error_rate_max_pct))
            + "</tr>"
        )

    content = f"""
    <div class="page-header">
      <h1>SLO Health</h1>
      <p>Per-tenant latency and error rate — refreshes every 60 s. Last checked: <strong>{escape(last_str)}</strong></p>
    </div>
    <div class="table-wrap">
      <div class="table-header"><h2>SLO status</h2></div>
      <table>
        <thead><tr>
          <th>Tenant</th><th>Health</th>
          <th>p50 / target (ms)</th><th>p95 / target (ms)</th>
          <th>p99 / target (ms)</th><th>Error rate / max (%)</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>"""
    return HTMLResponse(_layout("SLO Health", "slo", content))


@admin.get("/metrics")
async def prometheus_metrics() -> Response:
    from app.db.traces_read import get_stats
    from app.core.slo_checker import get_slo_gauges
    from app.core.ollama_adapter import get_circuit_breaker

    stats      = await get_stats()
    slo_gauges = get_slo_gauges()
    cb         = get_circuit_breaker()
    lines: list[str] = []

    def gauge(name: str, help_text: str, value: Any, labels: str = "") -> None:
        if value is not None:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} gauge")
            metric = f"{name}{{{labels}}}" if labels else name
            lines.append(f"{metric} {value}")

    gauge("relay_requests_total",    "Total requests processed",       stats.get("total_requests"))
    gauge("relay_requests_ok",       "Requests with 200 status",       stats.get("ok_requests"))
    gauge("relay_requests_rejected", "Requests rejected (429/503)",    stats.get("rejected_requests"))
    for pct in ("p50", "p95", "p99"):
        gauge(f"relay_latency_{pct}_ms",         f"E2E latency {pct}",     (stats.get("latency") or {}).get(pct))
        gauge(f"relay_backend_latency_{pct}_ms", f"Backend latency {pct}", (stats.get("backend_latency") or {}).get(pct))
    gauge("relay_cache_exact_hits",    "Exact cache hits",    stats.get("cache_exact_hits"))
    gauge("relay_cache_semantic_hits", "Semantic cache hits", stats.get("cache_semantic_hits"))
    gauge("relay_tokens_total",        "Total tokens",        stats.get("total_tokens"))
    if cb is not None:
        gauge("relay_circuit_breaker_state",          "CB state (0=CLOSED,1=HALF_OPEN,2=OPEN)", cb.state_gauge())
        gauge("relay_circuit_breaker_opens_total",    "Total CB opens",    cb.total_opens)
        gauge("relay_circuit_breaker_rejected_total", "Total CB rejected", cb.total_rejected)
    if slo_gauges:
        lines.append("# HELP relay_slo SLO gauges per tenant")
        lines.append("# TYPE relay_slo gauge")
        for key, value in slo_gauges.items():
            lines.append(f"relay_{key} {value}")

    return Response(content="\n".join(lines) + "\n",
                    media_type="text/plain; version=0.0.4; charset=utf-8")


@admin.get("/users/{tenant_id}", response_class=HTMLResponse)
async def users_page(tenant_id: str) -> HTMLResponse:
    from sqlalchemy import text as sa_text
    from app.db.postgres import get_sessionmaker

    async with get_sessionmaker()() as session:
        res = await session.execute(sa_text("""
            SELECT
                u.user_id, u.display_name, u.role,
                u.per_hour_token_limit, u.last_seen_at,
                COUNT(t.id)                       AS requests_24h,
                COALESCE(SUM(t.total_tokens), 0)  AS tokens_24h
            FROM users u
            LEFT JOIN request_traces t
                ON  t.user_id   = u.user_id
                AND t.tenant_id = u.tenant_id
                AND t.created_at > NOW() - INTERVAL '24 hours'
            WHERE u.tenant_id = :tid
            GROUP BY u.user_id, u.display_name, u.role,
                     u.per_hour_token_limit, u.last_seen_at
            ORDER BY tokens_24h DESC
        """), {"tid": tenant_id})
        rows = res.mappings().all()

    rows_html = ""
    for r in rows:
        limit    = r["per_hour_token_limit"] or 0
        used     = r["tokens_24h"] or 0
        pct      = (used / limit * 100) if limit else 0
        bar_cls  = "badge-green" if pct < 70 else ("badge-amber" if pct < 100 else "badge-red")
        rows_html += (
            f"<tr>"
            f"<td class='mono' style='font-size:12px'>{escape(r['user_id'])}</td>"
            f"<td>{escape(r['display_name'] or '—')}</td>"
            f"<td><span class='badge badge-gray'>{escape(r['role'])}</span></td>"
            f"<td class='mono'>{r['requests_24h']:,}</td>"
            f"<td><span class='badge {bar_cls} mono'>{used:,}</span>"
            f"  <span style='color:#9ca3af;font-size:11px'>/ {limit:,} ({pct:.0f}%)</span></td>"
            f"<td class='mono' style='font-size:11px;color:#6b7280'>"
            f"{str(r['last_seen_at'] or '—')[:19].replace('T',' ')}</td>"
            f"</tr>"
        )

    content = f"""
    <div class="page-header">
      <h1>Users — {escape(tenant_id)}</h1>
      <p>Token usage per user in the last 24 hours.</p>
    </div>
    <div class="table-wrap">
      <div class="table-header">
        <h2>Users</h2>
        <span class="table-meta">{len(rows)} users</span>
      </div>
      <table>
        <thead><tr>
          <th>User ID</th><th>Name</th><th>Role</th>
          <th>Requests (24h)</th><th>Tokens used / limit</th><th>Last seen</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>"""
    return HTMLResponse(_layout(f"Users — {tenant_id}", "traces", content))
