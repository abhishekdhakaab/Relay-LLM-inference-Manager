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
    stmt = text(
        """
        INSERT INTO request_traces (
          request_id, tenant_id, endpoint, model, status_code,
          request_hash, latency_ms, backend_latency_ms, queue_wait_ms, backend_ttft_ms,
          prompt_tokens, completion_tokens, total_tokens,
          request_json, response_json, error_json,
          policy_version, plan_json, decision_trace_json, cache_json
        )
        VALUES (
          :request_id, :tenant_id, :endpoint, :model, :status_code,
          :request_hash, :latency_ms, :backend_latency_ms, :queue_wait_ms, :backend_ttft_ms,
          :prompt_tokens, :completion_tokens, :total_tokens,
          CAST(:request_json AS JSONB), CAST(:response_json AS JSONB), CAST(:error_json AS JSONB),
          :policy_version, CAST(:plan_json AS JSONB), CAST(:decision_trace_json AS JSONB), CAST(:cache_json AS JSONB)
        )
        """
    )

    async with get_sessionmaker()() as session:
        await session.execute(stmt, payload)
        await session.commit()
