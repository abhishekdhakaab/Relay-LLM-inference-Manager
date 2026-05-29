from __future__ import annotations

import math
import time
from dataclasses import dataclass

from app.db.redis_client import get_redis


@dataclass
class UserBudgetResult:
    allowed: bool
    remaining: int
    reset_in_seconds: int


def _window_bucket(window_seconds: int = 3600) -> int:
    return math.floor(time.time() / window_seconds)


def _reset_in(window_seconds: int = 3600) -> int:
    bucket = _window_bucket(window_seconds)
    return max(0, int((bucket + 1) * window_seconds - time.time()))


async def check_user_budget(
    tenant_id: str,
    user_id: str,
    estimated_tokens: int,
    limit: int,
    window_seconds: int = 3600,
) -> UserBudgetResult:
    """
    Soft per-user token cap. Unlike the tenant hard-reject budget, this
    returns allowed=False only when the user is MORE than 2x over their limit.
    Under 2x they are allowed through with a warning in the trace.
    """
    redis = get_redis()
    bucket = _window_bucket(window_seconds)
    key = f"user_budget:{tenant_id}:{user_id}:{bucket}"
    ttl = window_seconds * 2

    new_total = await redis.incrby(key, estimated_tokens)
    await redis.expire(key, ttl)
    remaining = max(0, limit - new_total)

    # Hard block only above 2x limit to avoid degrading legitimate users
    if new_total > limit * 2:
        await redis.decrby(key, estimated_tokens)
        return UserBudgetResult(
            allowed=False,
            remaining=0,
            reset_in_seconds=_reset_in(window_seconds),
        )

    return UserBudgetResult(
        allowed=True,
        remaining=remaining,
        reset_in_seconds=_reset_in(window_seconds),
    )
