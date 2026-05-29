from __future__ import annotations
from typing import Optional
import redis.asyncio as redis
from app.core.settings import settings
_client: Optional[redis.Redis] = None

def get_redis() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.from_url(
            settings.redis_url,
            decode_responses=False,
            max_connections=32,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
        )
    return _client


