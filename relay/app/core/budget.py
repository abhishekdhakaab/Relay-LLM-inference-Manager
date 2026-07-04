"""Tenant token budgets backed by atomic Redis counters."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

from app.core.settings import TokenBudgetConfig
from app.db.redis_client import get_redis


@dataclass
class BudgetResult:
    """Outcome of a token reservation attempt."""

    allowed: bool
    degraded: bool = False
    allowed_tokens: Optional[int] = None  # set when degraded=True
    remaining: int = 0
    reset_in_seconds: int = 0

def _window_bucket(window_seconds: int) -> int:
    """Return the fixed-window bucket containing the current time."""
    return math.floor(time.time() / window_seconds)

def _redis_key(tenant_id: str, window_bucket: int) -> str:
    return f"budget:{tenant_id}:{window_bucket}"

def _reset_in_seconds(window_seconds: int) -> int:
    """Seconds until the current window bucket rolls over."""
    bucket = _window_bucket(window_seconds)
    next_reset = (bucket + 1) * window_seconds
    return max(0, int(next_reset - time.time()))

async def check_and_reserve(tenant_id: str,estimated_tokens: int,config: TokenBudgetConfig,) -> BudgetResult:
    """Atomically reserve tokens, rejecting or shrinking requests over budget."""

    redis = get_redis()
    bucket = _window_bucket(config.window_seconds)
    key = _redis_key(tenant_id, bucket)
    # Keeping the previous bucket for one extra window makes late corrections safe.
    ttl = config.window_seconds * 2


    # Reserve before checking the limit so concurrent requests cannot oversubscribe.
    new_total = await redis.incrby(key, estimated_tokens)
    await redis.expire(key, ttl)

    reset_in = _reset_in_seconds(config.window_seconds)

    if new_total > config.limit:
        if config.hard_reject:
            # A rejected request must not consume budget.
            await redis.decrby(key, estimated_tokens)
            return BudgetResult(
                allowed=False,
                remaining=0,
                reset_in_seconds=reset_in,
            )
        else:
            current_before = new_total - estimated_tokens
            available = max(0, config.limit - current_before)
            # Replace the optimistic reservation with only what was available.
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

async def record_actual_usage(tenant_id: str,actual_tokens: int,estimated_tokens: int,config: TokenBudgetConfig,) -> None:
    """Reconcile an estimate with the token count returned by the backend."""
    delta = actual_tokens - estimated_tokens
    if delta == 0:
        return

    redis = get_redis()
    bucket = _window_bucket(config.window_seconds)
    key = _redis_key(tenant_id, bucket)

    if delta > 0:
        await redis.incrby(key, delta)
    else:
        # Negative deltas release an overestimate back to the tenant.
        await redis.decrby(key, -delta)
