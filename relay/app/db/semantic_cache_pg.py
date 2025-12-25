from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import orjson
from sqlalchemy import text

from app.db.postgres import get_sessionmaker


def _vec_literal(vec: list[float]) -> str:
    # pgvector accepts: '[1,2,3]'::vector
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


async def semantic_lookup(
    *,
    tenant_id: str,
    plan_sig: str,
    query_vec: list[float],
) -> Optional[dict[str, Any]]:
    """
    Returns best match: {id, response_json, similarity}
    Using cosine distance (<=>) with vector_cosine_ops.
    similarity ≈ 1 - cosine_distance
    """
    q = text(
        """
        SELECT
          id::text AS id,
          response_json,
          (1 - (embedding <=> (:qvec)::vector)) AS similarity
        FROM semantic_cache_entries
        WHERE tenant_id = :tenant_id
          AND plan_sig = :plan_sig
          AND expires_at > now()
        ORDER BY embedding <=> (:qvec)::vector
        LIMIT 1
        """
    )

    params = {
        "tenant_id": tenant_id,
        "plan_sig": plan_sig,
        "qvec": _vec_literal(query_vec),
    }

    async with get_sessionmaker()() as session:
        res = await session.execute(q, params)
        row = res.mappings().first()
        return dict(row) if row else None


async def semantic_store(
    *,
    tenant_id: str,
    plan_sig: str,
    request_hash: str,
    prompt_text: str,
    embedding: list[float],
    response_obj: dict[str, Any],
    ttl_seconds: int,
) -> str:
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)

    q = text(
        """
        INSERT INTO semantic_cache_entries
          (tenant_id, plan_sig, request_hash, prompt_text, embedding, response_json, expires_at)
        VALUES
          (:tenant_id, :plan_sig, :request_hash, :prompt_text, (:embedding)::vector, CAST(:response_json AS JSONB), :expires_at)
        RETURNING id::text AS id
        """
    )

    params = {
        "tenant_id": tenant_id,
        "plan_sig": plan_sig,
        "request_hash": request_hash,
        "prompt_text": prompt_text,
        "embedding": _vec_literal(embedding),
        "response_json": orjson.dumps(response_obj).decode("utf-8"),
        "expires_at": expires_at,
    }

    async with get_sessionmaker()() as session:
        res = await session.execute(q, params)
        await session.commit()
        return str(res.scalar_one())
    


async def semantic_score(*,tenant_id:str, plan_sig:str, request_hash:str,prompt_text:str,embedding:list[float], response_obj:dict[str,Any], ttl_seconds:int)->str:
    expires_at = datetime.now(timezone.utc) + timedelta(seconds = ttl_seconds)

    q = text(
        """
        INSERT INTO semantic_cache_entries
          (tenant_id, plan_sig, request_hash, prompt_text, embedding, response_json, expires_at)
        VALUES
          (:tenant_id, :plan_sig, :request_hash, :prompt_text, (:embedding)::vector, CAST(:response_json AS JSONB), :expires_at)
        RETURNING id::text AS id
        """
    )

    params = {
        "tenant_id": tenant_id,
        "plan_sig": plan_sig,
        "request_hash": request_hash,
        "prompt_text": prompt_text,
        "embedding": _vec_literal(embedding),
        "response_json": orjson.dumps(response_obj).decode("utf-8"),
        "expires_at": expires_at,
    }
    async with get_sessionmaker()() as session:
        res = await session.execute(q,params)
        await session.commit()
        return str(res.scalar_one())