"""Shared asynchronous Redis client."""

from __future__ import annotations
from typing import Optional
import redis.asyncio as redis
from app.core.settings import settings
_client: Optional[redis.Redis] = None
def get_redis() -> redis.Redis:
    """Create the bounded connection pool on first use."""
    global _client
    if _client is None:
        _client = redis.from_url(
            settings.redis_url,
            # Cache payloads stay as bytes until their serializer decodes them.
            decode_responses=False,

            max_connections=32,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
        )
    return _client
