"""Read models for trace detail and aggregate dashboard queries."""

from __future__ import annotations
from typing import Any, Optional
from sqlalchemy import text
from app.db.postgres import get_sessionmaker

async def list_traces(limit:int=50) ->list[dict[str,Any]]:
    """Return the newest request traces for the admin list."""
    q = text(
        """
        SELECT
          request_id,
          tenant_id,
          created_at,
          status_code,
          model,
          latency_ms,
          backend_latency_ms,
          queue_wait_ms,
          request_hash,
          policy_version,
          cache_json,
          plan_json
        FROM request_traces
        ORDER BY created_at DESC
        LIMIT :limit
        """
    )
    async with get_sessionmaker()() as session:
        res = await session.execute(q,{'limit':limit})
        rows = res.mappings().all()
        return [dict(r) for r in rows]
    
async def get_trace(request_id:str)->Optional[dict[str,Any]]:
    """Return one complete request trace by ID."""
    q = text(
        """
        SELECT
          request_id,
          tenant_id,
          created_at,
          endpoint,
          model,
          status_code,
          request_hash,
          latency_ms,
          backend_latency_ms,
          queue_wait_ms,
          backend_ttft_ms,
          prompt_tokens,
          completion_tokens,
          total_tokens,
          policy_version,
          plan_json,
          decision_trace_json,
          cache_json,
          request_json,
          response_json,
          error_json
        FROM request_traces
        WHERE request_id = :request_id
        LIMIT 1
        """
    )
    async with get_sessionmaker()() as session:
        res = await session.execute(q, {"request_id": request_id})
        row = res.mappings().first()
        return dict(row) if row else None


async def get_stats() -> dict[str, Any]:
    """Aggregate latency, cache, token, and routing metrics across traces."""
    q = text(
        """
        SELECT
          count(*)                                              AS total_requests,
          count(*) FILTER (WHERE status_code = 200)             AS ok_requests,
          count(*) FILTER (WHERE status_code IN (429, 503))     AS rejected_requests,

          percentile_cont(0.50) WITHIN GROUP (ORDER BY latency_ms)
            FILTER (WHERE latency_ms IS NOT NULL)               AS latency_p50,
          percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms)
            FILTER (WHERE latency_ms IS NOT NULL)               AS latency_p95,
          percentile_cont(0.99) WITHIN GROUP (ORDER BY latency_ms)
            FILTER (WHERE latency_ms IS NOT NULL)               AS latency_p99,

          percentile_cont(0.50) WITHIN GROUP (ORDER BY backend_latency_ms)
            FILTER (WHERE backend_latency_ms IS NOT NULL)       AS backend_p50,
          percentile_cont(0.95) WITHIN GROUP (ORDER BY backend_latency_ms)
            FILTER (WHERE backend_latency_ms IS NOT NULL)       AS backend_p95,
          percentile_cont(0.99) WITHIN GROUP (ORDER BY backend_latency_ms)
            FILTER (WHERE backend_latency_ms IS NOT NULL)       AS backend_p99,

          coalesce(sum(total_tokens), 0)                        AS total_tokens,

          count(*) FILTER (WHERE backend_latency_ms IS NULL AND status_code = 200)
                                                                AS cache_exact_hits,
          count(*) FILTER (WHERE cache_json::text LIKE '%%"semantic"%%-"hit": true%%')
                                                                AS cache_semantic_hits
        FROM request_traces
        """
    )

    plan_q = text(
        """
        SELECT
          plan_json->>'plan_name' AS bucket,
          count(*)                AS cnt
        FROM request_traces
        WHERE plan_json IS NOT NULL
        GROUP BY plan_json->>'plan_name'
        """
    )

    async with get_sessionmaker()() as session:
        res = await session.execute(q)
        row = dict(res.mappings().first() or {})

        plan_res = await session.execute(plan_q)
        routing = {str(r["bucket"]): int(r["cnt"]) for r in plan_res.mappings().all()}

    def safe_round(v: Any) -> Any:
        # PostgreSQL percentile values may be Decimal rather than JSON-native floats.
        if v is None:
            return None
        return round(float(v), 1)

    return {
        "total_requests": row.get("total_requests", 0),
        "ok_requests": row.get("ok_requests", 0),
        "rejected_requests": row.get("rejected_requests", 0),
        "latency": {
            "p50": safe_round(row.get("latency_p50")),
            "p95": safe_round(row.get("latency_p95")),
            "p99": safe_round(row.get("latency_p99")),
        },
        "backend_latency": {
            "p50": safe_round(row.get("backend_p50")),
            "p95": safe_round(row.get("backend_p95")),
            "p99": safe_round(row.get("backend_p99")),
        },
        "total_tokens": row.get("total_tokens", 0),
        "cache_exact_hits": row.get("cache_exact_hits", 0),
        "cache_semantic_hits": row.get("cache_semantic_hits", 0),
        "routing_distribution": routing,
    }
