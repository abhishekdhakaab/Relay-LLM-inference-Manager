"""
Tests for Feature 2 — per-user Redis token budget.
Redis is mocked; no running Redis required.
"""
from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import pytest

from app.core.user_budget import (
    UserBudgetResult,
    _reset_in,
    _window_bucket,
    check_user_budget,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def test_window_bucket_returns_int():
    assert isinstance(_window_bucket(), int)


def test_window_bucket_advances_with_time():
    """Bucket for now should be same as floor(now/3600)."""
    import math
    expected = math.floor(time.time() / 3600)
    assert _window_bucket(3600) == expected


def test_reset_in_positive_and_lte_window():
    r = _reset_in(3600)
    assert 0 < r <= 3600


def test_reset_in_small_window():
    r = _reset_in(60)
    assert 0 < r <= 60


# ── Budget logic ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_allowed_well_under_limit():
    mock_redis = AsyncMock()
    mock_redis.incrby = AsyncMock(return_value=1000)
    mock_redis.expire = AsyncMock()

    with patch("app.core.user_budget.get_redis", return_value=mock_redis):
        result = await check_user_budget("default", "user-alice", 1000, limit=100_000)

    assert result.allowed is True
    assert result.remaining == 99_000


@pytest.mark.asyncio
async def test_allowed_exactly_at_limit():
    """Hitting the soft cap exactly still lets the request through."""
    mock_redis = AsyncMock()
    mock_redis.incrby = AsyncMock(return_value=100_000)
    mock_redis.expire = AsyncMock()

    with patch("app.core.user_budget.get_redis", return_value=mock_redis):
        result = await check_user_budget("default", "user-alice", 100_000, limit=100_000)

    assert result.allowed is True
    assert result.remaining == 0


@pytest.mark.asyncio
async def test_allowed_in_grace_band_between_1x_and_2x():
    """150 % of limit is still within the soft-cap grace band → allowed."""
    mock_redis = AsyncMock()
    mock_redis.incrby = AsyncMock(return_value=150_000)  # 1.5× limit
    mock_redis.expire = AsyncMock()

    with patch("app.core.user_budget.get_redis", return_value=mock_redis):
        result = await check_user_budget("default", "user-alice", 50_000, limit=100_000)

    assert result.allowed is True


@pytest.mark.asyncio
async def test_blocked_above_2x_limit():
    """Above 2× limit → hard block, counter rolled back."""
    mock_redis = AsyncMock()
    mock_redis.incrby = AsyncMock(return_value=210_000)  # > 2 × 100_000
    mock_redis.expire = AsyncMock()
    mock_redis.decrby = AsyncMock()

    with patch("app.core.user_budget.get_redis", return_value=mock_redis):
        result = await check_user_budget("default", "user-alice", 10_000, limit=100_000)

    assert result.allowed is False
    assert result.remaining == 0
    assert result.reset_in_seconds > 0
    mock_redis.decrby.assert_called_once_with(
        f"user_budget:default:user-alice:{_window_bucket()}", 10_000
    )


@pytest.mark.asyncio
async def test_rollback_not_called_when_allowed():
    """decrby must NOT be called when the request is allowed."""
    mock_redis = AsyncMock()
    mock_redis.incrby = AsyncMock(return_value=5_000)
    mock_redis.expire = AsyncMock()
    mock_redis.decrby = AsyncMock()

    with patch("app.core.user_budget.get_redis", return_value=mock_redis):
        await check_user_budget("default", "user-alice", 5_000, limit=100_000)

    mock_redis.decrby.assert_not_called()


# ── Key namespacing ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_redis_key_is_tenant_and_user_scoped():
    """Different tenants for the same user_id must use different Redis keys."""
    captured: list[str] = []

    mock_redis = AsyncMock()

    async def record_incrby(key: str, amount: int) -> int:
        captured.append(key)
        return amount  # well under any limit

    mock_redis.incrby = record_incrby
    mock_redis.expire = AsyncMock()

    with patch("app.core.user_budget.get_redis", return_value=mock_redis):
        await check_user_budget("tenant-a", "shared-user", 100, limit=10_000)
        await check_user_budget("tenant-b", "shared-user", 100, limit=10_000)

    assert len(captured) == 2
    assert captured[0] != captured[1]
    assert "tenant-a" in captured[0]
    assert "tenant-b" in captured[1]


@pytest.mark.asyncio
async def test_redis_key_contains_bucket_for_ttl_isolation():
    """The Redis key must embed the time bucket so windows auto-expire."""
    captured: list[str] = []

    mock_redis = AsyncMock()

    async def record_incrby(key: str, amount: int) -> int:
        captured.append(key)
        return amount

    mock_redis.incrby = record_incrby
    mock_redis.expire = AsyncMock()

    with patch("app.core.user_budget.get_redis", return_value=mock_redis):
        await check_user_budget("t", "u", 1, limit=1_000)

    key = captured[0]
    bucket = str(_window_bucket())
    assert bucket in key


@pytest.mark.asyncio
async def test_expire_called_with_double_window():
    """TTL set on the Redis key must be 2× the window so it overlaps resets."""
    mock_redis = AsyncMock()
    mock_redis.incrby = AsyncMock(return_value=1)
    mock_redis.expire = AsyncMock()

    window = 3600
    with patch("app.core.user_budget.get_redis", return_value=mock_redis):
        await check_user_budget("t", "u", 1, limit=10_000, window_seconds=window)

    _, ttl_arg = mock_redis.expire.call_args[0]
    assert ttl_arg == window * 2
