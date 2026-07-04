"""Stable signatures and tenant-scoped keys for exact caching."""

from __future__ import annotations
import hashlib
from typing import Any
import orjson

def plan_signature(plan:dict[str,Any]) ->str:
    """Hash generation-affecting plan values independent of key order."""

    b = orjson.dumps(plan, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(b).hexdigest()[:16]

def exact_cache_key(*,tenant_id:str, request_hash : str, plan_sig : str) -> str:
    """Scope cache hits to the same tenant, plan, and normalized request."""
    return f"exact:{tenant_id}:{plan_sig}:{request_hash}"
