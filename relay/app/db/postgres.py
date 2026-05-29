from __future__ import annotations

from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.settings import settings


_engine: Optional[AsyncEngine] = None
_sessionmaker: Optional[async_sessionmaker[AsyncSession]] = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _sessionmaker


async def insert_trace(payload: dict[str, Any]) -> None:
    """
    Insert a request trace row.

    The payload dict may contain any subset of columns — missing keys default
    to NULL so existing callers (exact/semantic cache hits) don't need to be
    updated to pass new columns; they simply won't be populated.
    """
    # Provide None defaults for every column so SQLAlchemy doesn't error on
    # missing keys.  The ON CONFLICT DO NOTHING guard prevents duplicate inserts
    # if the same request_id is submitted twice (defensive).
    defaults: dict[str, Any] = {
        # Original columns
        "request_id": None, "tenant_id": None, "user_id": None, "endpoint": None,
        "model": None, "status_code": None, "request_hash": None,
        "latency_ms": None, "backend_latency_ms": None, "queue_wait_ms": None,
        "backend_ttft_ms": None, "prompt_tokens": None, "completion_tokens": None,
        "total_tokens": None, "request_json": "null", "response_json": "null",
        "error_json": "null", "policy_version": None, "plan_json": "null",
        "decision_trace_json": "null", "cache_json": "null",
        # Feature 1 — Cost Router
        "complexity_score": None, "intent_type": None, "tier_selected": None,
        "forced_by_tool_flag": None, "needs_computation": None, "needs_code": None,
        "needs_retrieval": None, "needs_personal_context": None, "is_multi_hop": None,
        "actual_cost_usd": None, "would_cost_usd": None,
        "escalated": False, "escalated_from_tier": None,
        "classifier_latency_ms": None,
        # Feature 3 — OTel
        "otel_trace_id": None,
        # Feature 5 — SLO error tracking
        "error": False, "error_type": None,
    }
    row = {**defaults, **payload}

    stmt = text(
        """
        INSERT INTO request_traces (
          request_id, tenant_id, user_id, endpoint, model, status_code,
          request_hash, latency_ms, backend_latency_ms, queue_wait_ms, backend_ttft_ms,
          prompt_tokens, completion_tokens, total_tokens,
          request_json, response_json, error_json,
          policy_version, plan_json, decision_trace_json, cache_json,
          complexity_score, intent_type, tier_selected, forced_by_tool_flag,
          needs_computation, needs_code, needs_retrieval, needs_personal_context,
          is_multi_hop, actual_cost_usd, would_cost_usd,
          escalated, escalated_from_tier, classifier_latency_ms,
          otel_trace_id, error, error_type
        )
        VALUES (
          :request_id, :tenant_id, :user_id, :endpoint, :model, :status_code,
          :request_hash, :latency_ms, :backend_latency_ms, :queue_wait_ms, :backend_ttft_ms,
          :prompt_tokens, :completion_tokens, :total_tokens,
          CAST(:request_json AS JSONB), CAST(:response_json AS JSONB), CAST(:error_json AS JSONB),
          :policy_version, CAST(:plan_json AS JSONB), CAST(:decision_trace_json AS JSONB),
          CAST(:cache_json AS JSONB),
          :complexity_score, :intent_type, :tier_selected, :forced_by_tool_flag,
          :needs_computation, :needs_code, :needs_retrieval, :needs_personal_context,
          :is_multi_hop, :actual_cost_usd, :would_cost_usd,
          :escalated, :escalated_from_tier, :classifier_latency_ms,
          :otel_trace_id, :error, :error_type
        )
        ON CONFLICT (request_id) DO NOTHING
        """
    )

    async with get_sessionmaker()() as session:
        await session.execute(stmt, row)
        await session.commit()
