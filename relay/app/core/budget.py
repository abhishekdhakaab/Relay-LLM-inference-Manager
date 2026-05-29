from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

from app.core.settings import TokenBudgetConfig
from app.db.redis_client import get_redis


@dataclass
class BudgetResult:
    allowed: bool
    degraded: bool = False
    allowed_tokens: Optional[int] = None  # set when degraded=True
    remaining: int = 0
    reset_in_seconds: int = 0


def _window_bucket(window_seconds: int) -> int:
    """Current rolling-window bucket index."""
    return math.floor(time.time() / window_seconds)


def _redis_key(tenant_id: str, window_bucket: int) -> str:
    return f"budget:{tenant_id}:{window_bucket}"


def _reset_in_seconds(window_seconds: int) -> int:
    """Seconds until the current window bucket rolls over."""
    bucket = _window_bucket(window_seconds)
    next_reset = (bucket + 1) * window_seconds
    return max(0, int(next_reset - time.time()))


async def check_and_reserve(
    tenant_id: str,
    estimated_tokens: int,
    config: TokenBudgetConfig,
) -> BudgetResult:
    """
    Atomically reserve estimated_tokens from the tenant's rolling window budget.

    Key schema: budget:{tenant_id}:{window_bucket}
    TTL is set to 2× the window so stale keys self-clean.

    Spec §4.3 algorithm:
    1. INCRBY key estimated_tokens
    2. If result > limit:
       a. hard_reject=True  → DECRBY back, return 429
       b. hard_reject=False → return degraded allocation with remaining tokens
    3. Otherwise return allowed with remaining tokens.
    """
    redis = get_redis()
    bucket = _window_bucket(config.window_seconds)
    key = _redis_key(tenant_id, bucket)
    ttl = config.window_seconds * 2

    # Atomic increment
    new_total = await redis.incrby(key, estimated_tokens)
    # Ensure TTL is set on the key (only costs an extra round-trip if not set)
    await redis.expire(key, ttl)

    reset_in = _reset_in_seconds(config.window_seconds)

    if new_total > config.limit:
        if config.hard_reject:
            # Roll back the reservation
            await redis.decrby(key, estimated_tokens)
            return BudgetResult(
                allowed=False,
                remaining=0,
                reset_in_seconds=reset_in,
            )
        else:
            # Soft degrade: give the remaining available tokens
            current_before = new_total - estimated_tokens
            available = max(0, config.limit - current_before)
            # Roll back and re-reserve only the available amount
            await redis.decrby(key, estimated_tokens)
            if available > 0:
                await redis.incrby(key, available)
            return BudgetResult(
                allowed=True,
                degraded=True,
                allowed_tokens=available,
                remaining=0,
                reset_in_seconds=reset_in,
            )

    remaining = max(0, config.limit - new_total)
    return BudgetResult(
        allowed=True,
        remaining=remaining,
        reset_in_seconds=reset_in,
    )


async def record_actual_usage(
    tenant_id: str,
    actual_tokens: int,
    estimated_tokens: int,
    config: TokenBudgetConfig,
) -> None:
    """
    Correct the budget reservation after generation completes.
    Adjusts the counter by (actual - estimated).  This keeps the running
    total precise without requiring a read-modify-write cycle per request.
    """
    delta = actual_tokens - estimated_tokens
    if delta == 0:
        return

    redis = get_redis()
    bucket = _window_bucket(config.window_seconds)
    key = _redis_key(tenant_id, bucket)

    if delta > 0:
        await redis.incrby(key, delta)
    else:
        # delta is negative — we over-estimated, release the difference
        await redis.decrby(key, -delta)
