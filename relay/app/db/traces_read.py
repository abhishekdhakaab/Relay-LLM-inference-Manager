from __future__ import annotations
from typing import Any, Optional
from sqlalchemy import text
from app.db.postgres import get_sessionmaker

async def list_traces(limit:int=50) ->list[dict[str,Any]]:
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