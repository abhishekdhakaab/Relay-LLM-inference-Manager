"""Bearer-key authentication and optional per-user identity resolution."""

from __future__ import annotations

import hashlib
from typing import Optional

from fastapi import Depends, Header, HTTPException
from sqlalchemy import text as sa_text

from app.core.logging import get_logger
from app.db.postgres import get_sessionmaker

log = get_logger(component="auth")


def _hash_key(raw_key: str) -> str:
    """SHA-256 hex digest of the raw bearer token."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


async def _lookup_key(raw_key: str) -> Optional[dict]:
    """Return an active API-key row and update its last-used timestamp."""
    key_hash = _hash_key(raw_key)
    q = sa_text(
        """
        SELECT id, tenant_id, scopes, is_active
        FROM api_keys
        WHERE key_hash = :key_hash
        LIMIT 1
        """
    )
    async with get_sessionmaker()() as session:
        res = await session.execute(q, {"key_hash": key_hash})
        row = res.mappings().first()
        if row and row["is_active"]:
            await session.execute(
                sa_text("UPDATE api_keys SET last_used_at = now() WHERE key_hash = :h"),
                {"h": key_hash},
            )
            await session.commit()
            return dict(row)
    return None


async def require_chat_auth(
    authorization: str = Header(default=""),
    x_tenant_id: str = Header(default=""),
) -> dict:
    """Require chat scope and derive tenant identity from the bearer key."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={"error": "missing_api_key", "message": "Authorization: Bearer <key> required"},
        )
    raw_key = authorization.removeprefix("Bearer ").strip()
    row = await _lookup_key(raw_key)
    if row is None:
        log.warning("auth_failed", reason="invalid_or_inactive_key")
        raise HTTPException(
            status_code=401,
            detail={"error": "invalid_api_key", "message": "API key is invalid or inactive"},
        )
    if "chat" not in row["scopes"]:
        raise HTTPException(
            status_code=403,
            detail={"error": "insufficient_scope", "message": "This key does not have chat scope"},
        )
    return row


async def require_admin_auth(
    authorization: str = Header(default=""),
) -> dict:
    """FastAPI dependency for /admin/* routes."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail={"error": "missing_api_key"})
    raw_key = authorization.removeprefix("Bearer ").strip()
    row = await _lookup_key(raw_key)
    if row is None:
        raise HTTPException(status_code=401, detail={"error": "invalid_api_key"})
    if "admin" not in row["scopes"]:
        raise HTTPException(
            status_code=403,
            detail={"error": "insufficient_scope", "message": "Admin scope required"},
        )
    return row


async def resolve_user(
    x_user_id: str = Header(default="anonymous"),
    auth: dict = Depends(require_chat_auth),
) -> dict:
    """Combine key identity with an optional user record in the same tenant."""
    tenant_id = auth["tenant_id"]
    user_id = x_user_id.strip() or "anonymous"

    q = sa_text(
        """
        SELECT user_id, tenant_id, role, per_hour_token_limit, is_active
        FROM users
        WHERE user_id = :uid AND tenant_id = :tid
        LIMIT 1
        """
    )
    user_row = None
    async with get_sessionmaker()() as session:
        res = await session.execute(q, {"uid": user_id, "tid": tenant_id})
        row = res.mappings().first()
        if row:
            user_row = dict(row)
            await session.execute(
                sa_text(
                    "UPDATE users SET last_seen_at = now() "
                    "WHERE user_id = :uid AND tenant_id = :tid"
                ),
                {"uid": user_id, "tid": tenant_id},
            )
            await session.commit()

    return {
        "tenant_id": tenant_id,
        "user_id": user_id,
        "key_row": auth,
        "user_row": user_row,
    }
