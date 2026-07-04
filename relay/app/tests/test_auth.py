"""
Tests for API-key authorization and per-tenant user identity.
All DB calls are mocked — no Postgres or Redis required.
"""
from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.auth import _hash_key, require_chat_auth, require_admin_auth, resolve_user


# ── _hash_key ──────────────────────────────────────────────────────────────────

def test_hash_key_matches_sha256():
    raw = "relay-dev-default-key-1234"
    expected = hashlib.sha256(raw.encode()).hexdigest()
    assert _hash_key(raw) == expected


def test_hash_key_deterministic():
    assert _hash_key("some-key") == _hash_key("some-key")


def test_hash_key_different_inputs_produce_different_digests():
    assert _hash_key("key-a") != _hash_key("key-b")


def test_hash_key_is_64_hex_chars():
    result = _hash_key("anything")
    assert len(result) == 64
    assert all(c in "0123456789abcdef" for c in result)


# ── require_chat_auth — missing / wrong header ────────────────────────────────

@pytest.mark.asyncio
async def test_chat_auth_no_header_raises_401():
    with pytest.raises(HTTPException) as exc:
        await require_chat_auth(authorization="", x_tenant_id="")
    assert exc.value.status_code == 401
    assert exc.value.detail["error"] == "missing_api_key"


@pytest.mark.asyncio
async def test_chat_auth_basic_scheme_raises_401():
    with pytest.raises(HTTPException) as exc:
        await require_chat_auth(authorization="Basic dXNlcjpwYXNz", x_tenant_id="")
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_chat_auth_bearer_prefix_only_raises_401():
    """'Bearer ' with no key after it — DB lookup will return None."""
    with patch("app.auth._lookup_key", new=AsyncMock(return_value=None)):
        with pytest.raises(HTTPException) as exc:
            await require_chat_auth(authorization="Bearer ", x_tenant_id="")
    assert exc.value.status_code == 401


# ── require_chat_auth — key lookup failures ────────────────────────────────────

@pytest.mark.asyncio
async def test_chat_auth_unknown_key_raises_401():
    with patch("app.auth._lookup_key", new=AsyncMock(return_value=None)):
        with pytest.raises(HTTPException) as exc:
            await require_chat_auth(authorization="Bearer bad-key", x_tenant_id="")
    assert exc.value.status_code == 401
    assert exc.value.detail["error"] == "invalid_api_key"


@pytest.mark.asyncio
async def test_chat_auth_admin_only_key_raises_403():
    """A key that exists but only has 'admin' scope cannot use chat endpoints."""
    row = {"id": "x", "tenant_id": "default", "scopes": ["admin"], "is_active": True}
    with patch("app.auth._lookup_key", new=AsyncMock(return_value=row)):
        with pytest.raises(HTTPException) as exc:
            await require_chat_auth(authorization="Bearer relay-dev-admin-only", x_tenant_id="")
    assert exc.value.status_code == 403
    assert exc.value.detail["error"] == "insufficient_scope"


# ── require_chat_auth — valid keys ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chat_auth_valid_chat_key_returns_row():
    row = {"id": "x", "tenant_id": "default", "scopes": ["chat"], "is_active": True}
    with patch("app.auth._lookup_key", new=AsyncMock(return_value=row)):
        result = await require_chat_auth(
            authorization="Bearer relay-dev-default-key-1234", x_tenant_id=""
        )
    assert result["tenant_id"] == "default"
    assert "chat" in result["scopes"]


@pytest.mark.asyncio
async def test_chat_auth_admin_key_with_chat_scope_passes():
    """Admin key that also holds chat scope must be accepted on chat routes."""
    row = {"id": "x", "tenant_id": "default", "scopes": ["chat", "admin"], "is_active": True}
    with patch("app.auth._lookup_key", new=AsyncMock(return_value=row)):
        result = await require_chat_auth(
            authorization="Bearer relay-dev-admin-key-9999", x_tenant_id=""
        )
    assert result is not None


@pytest.mark.asyncio
async def test_chat_auth_strips_whitespace_from_key():
    """Trailing spaces in the bearer value must be stripped before lookup."""
    row = {"id": "x", "tenant_id": "default", "scopes": ["chat"], "is_active": True}
    captured: list[str] = []

    async def fake_lookup(raw_key: str):
        captured.append(raw_key)
        return row

    with patch("app.auth._lookup_key", new=fake_lookup):
        await require_chat_auth(authorization="Bearer  my-key  ", x_tenant_id="")

    assert captured[0] == "my-key"


# ── require_admin_auth ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_admin_auth_no_header_raises_401():
    with pytest.raises(HTTPException) as exc:
        await require_admin_auth(authorization="")
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_admin_auth_chat_only_key_raises_403():
    row = {"id": "x", "tenant_id": "default", "scopes": ["chat"], "is_active": True}
    with patch("app.auth._lookup_key", new=AsyncMock(return_value=row)):
        with pytest.raises(HTTPException) as exc:
            await require_admin_auth(authorization="Bearer relay-dev-default-key-1234")
    assert exc.value.status_code == 403
    assert exc.value.detail["error"] == "insufficient_scope"


@pytest.mark.asyncio
async def test_admin_auth_valid_admin_key_returns_row():
    row = {"id": "x", "tenant_id": "default", "scopes": ["chat", "admin"], "is_active": True}
    with patch("app.auth._lookup_key", new=AsyncMock(return_value=row)):
        result = await require_admin_auth(authorization="Bearer relay-dev-admin-key-9999")
    assert result["tenant_id"] == "default"
    assert "admin" in result["scopes"]


@pytest.mark.asyncio
async def test_admin_auth_invalid_key_raises_401():
    with patch("app.auth._lookup_key", new=AsyncMock(return_value=None)):
        with pytest.raises(HTTPException) as exc:
            await require_admin_auth(authorization="Bearer nonexistent-key")
    assert exc.value.status_code == 401


# ── resolve_user ───────────────────────────────────────────────────────────────

def _mock_session(row):
    """Return a context-manager-compatible async mock that yields `row` from first()."""
    mock_result = MagicMock()
    mock_result.mappings.return_value.first.return_value = row

    session = AsyncMock()
    session.execute = AsyncMock(return_value=mock_result)
    session.commit = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=session)


@pytest.mark.asyncio
async def test_resolve_user_known_user_populates_user_row():
    key_row = {"id": "k", "tenant_id": "default", "scopes": ["chat"], "is_active": True}
    db_user = {
        "user_id": "user-alice", "tenant_id": "default",
        "role": "admin", "per_hour_token_limit": 200000, "is_active": True,
    }
    with patch("app.auth.get_sessionmaker", return_value=_mock_session(db_user)):
        ctx = await resolve_user(x_user_id="user-alice", auth=key_row)

    assert ctx["tenant_id"] == "default"
    assert ctx["user_id"] == "user-alice"
    assert ctx["user_row"] is not None
    assert ctx["user_row"]["role"] == "admin"


@pytest.mark.asyncio
async def test_resolve_user_unknown_user_id_sets_user_row_to_none():
    key_row = {"id": "k", "tenant_id": "default", "scopes": ["chat"], "is_active": True}
    with patch("app.auth.get_sessionmaker", return_value=_mock_session(None)):
        ctx = await resolve_user(x_user_id="ghost-user", auth=key_row)

    assert ctx["user_id"] == "ghost-user"
    assert ctx["user_row"] is None


@pytest.mark.asyncio
async def test_resolve_user_blank_header_defaults_to_anonymous():
    key_row = {"id": "k", "tenant_id": "default", "scopes": ["chat"], "is_active": True}
    with patch("app.auth.get_sessionmaker", return_value=_mock_session(None)):
        ctx = await resolve_user(x_user_id="   ", auth=key_row)

    assert ctx["user_id"] == "anonymous"


@pytest.mark.asyncio
async def test_resolve_user_tenant_always_from_key_not_header():
    """tenant_id must come from the validated API key, not any user-supplied header."""
    key_row = {"id": "k", "tenant_id": "premium", "scopes": ["chat"], "is_active": True}
    with patch("app.auth.get_sessionmaker", return_value=_mock_session(None)):
        ctx = await resolve_user(x_user_id="user-carol", auth=key_row)

    assert ctx["tenant_id"] == "premium"


@pytest.mark.asyncio
async def test_resolve_user_key_row_preserved_in_context():
    key_row = {"id": "k", "tenant_id": "default", "scopes": ["chat"], "is_active": True}
    with patch("app.auth.get_sessionmaker", return_value=_mock_session(None)):
        ctx = await resolve_user(x_user_id="u", auth=key_row)

    assert ctx["key_row"] is key_row
