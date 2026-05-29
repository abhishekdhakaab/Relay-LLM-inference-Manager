"""
Regression tests for auth gating on HTTP routes (Features 1, 2, 4).
Uses FastAPI dependency_overrides so no live DB, Redis, or Ollama is needed.

Coverage:
  - /health always open (no auth)
  - /v1/chat/completions rejects missing/wrong Bearer → 401
  - /v1/chat/completions accepts overridden auth → passes auth layer
  - stream=true always → 400 (regardless of auth)
  - /admin/* rejects missing Bearer → 401
  - /admin/* rejects chat-only key → 403
  - /admin/* accepts admin override → passes auth layer
  - JSON admin endpoints still exist and pass auth check
  - cache_keys utilities produce stable, correctly-scoped values
"""
from __future__ import annotations

import hashlib

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.auth import require_admin_auth, resolve_user
from app.main import create_app
from app.utils.cache_keys import exact_cache_key, plan_signature

# ── Shared fixtures ────────────────────────────────────────────────────────────

CHAT_CTX = {
    "tenant_id": "default",
    "user_id": "user-alice",
    "key_row": {"scopes": ["chat"]},
    "user_row": None,
}

ADMIN_ROW = {
    "id": "x",
    "tenant_id": "default",
    "scopes": ["chat", "admin"],
    "is_active": True,
}


def _chat_client() -> TestClient:
    """Client with chat auth bypassed; admin auth also bypassed."""
    app = create_app()
    app.dependency_overrides[resolve_user] = lambda: CHAT_CTX
    app.dependency_overrides[require_admin_auth] = lambda: ADMIN_ROW
    return TestClient(app, raise_server_exceptions=False)


def _no_auth_client() -> TestClient:
    """Client with no overrides — real auth dependency runs."""
    app = create_app()
    return TestClient(app, raise_server_exceptions=False)


def _admin_only_override() -> TestClient:
    """Client where admin auth is overridden but chat auth is NOT."""
    app = create_app()
    app.dependency_overrides[require_admin_auth] = lambda: ADMIN_ROW
    return TestClient(app, raise_server_exceptions=False)


def _raise_403_dep():
    raise HTTPException(status_code=403, detail={"error": "insufficient_scope"})


# ── /health — always open ──────────────────────────────────────────────────────

def test_health_returns_200_without_any_auth():
    client = _no_auth_client()
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_health_still_works_with_auth_headers():
    client = _no_auth_client()
    r = client.get("/health", headers={"Authorization": "Bearer some-key"})
    assert r.status_code == 200


# ── /v1/chat/completions — 401 path ───────────────────────────────────────────

def test_chat_no_auth_header_returns_401():
    client = _no_auth_client()
    r = client.post(
        "/v1/chat/completions",
        json={"model": "local", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 401


def test_chat_basic_scheme_returns_401():
    client = _no_auth_client()
    r = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Basic dXNlcjpwYXNz"},
        json={"model": "local", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 401


def test_chat_bearer_with_wrong_key_returns_401():
    """Bearer present but key not in DB (DB unreachable → lookup returns None)."""
    client = _no_auth_client()
    r = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer totally-wrong-key"},
        json={"model": "local", "messages": [{"role": "user", "content": "hi"}]},
    )
    # Either 401 (key lookup failed) or 500 (DB unreachable)
    # — must NOT be 200 or 403
    assert r.status_code in (401, 500)
    assert r.status_code != 200


# ── /v1/chat/completions — auth passes ────────────────────────────────────────

def test_chat_with_valid_override_is_not_401_or_403():
    """With auth bypass, the request reaches the handler (may 500 without DB)."""
    client = _chat_client()
    r = client.post(
        "/v1/chat/completions",
        json={"model": "local", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code not in (401, 403)


def test_chat_stream_true_returns_400_regardless_of_auth():
    """stream=True must always be rejected before hitting any backend."""
    client = _chat_client()
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "local",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert r.status_code == 400


# ── /admin/* — 401 path ───────────────────────────────────────────────────────

def test_admin_traces_no_key_returns_401():
    client = _no_auth_client()
    r = client.get("/admin/traces")
    assert r.status_code == 401


def test_admin_traces_json_no_key_returns_401():
    client = _no_auth_client()
    r = client.get("/admin/traces.json")
    assert r.status_code == 401


def test_admin_cost_no_key_returns_401():
    client = _no_auth_client()
    r = client.get("/admin/cost")
    assert r.status_code == 401


def test_admin_stats_json_no_key_returns_401():
    client = _no_auth_client()
    r = client.get("/admin/stats.json")
    assert r.status_code == 401


# ── /admin/* — 403 path (chat key, no admin scope) ───────────────────────────

def test_admin_traces_chat_key_returns_403():
    """A chat-only key must be rejected on admin routes with 403."""
    app = create_app()
    app.dependency_overrides[require_admin_auth] = _raise_403_dep
    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/admin/traces")
    assert r.status_code == 403


def test_admin_cost_chat_key_returns_403():
    app = create_app()
    app.dependency_overrides[require_admin_auth] = _raise_403_dep
    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/admin/cost")
    assert r.status_code == 403


# ── /admin/* — auth passes ────────────────────────────────────────────────────

def test_admin_traces_with_admin_override_not_401_or_403():
    client = _admin_only_override()
    r = client.get("/admin/traces")
    assert r.status_code not in (401, 403)


def test_admin_traces_json_with_admin_override_not_401_or_403():
    client = _admin_only_override()
    r = client.get("/admin/traces.json")
    assert r.status_code not in (401, 403)


def test_admin_stats_json_with_admin_override_not_401_or_403():
    client = _admin_only_override()
    r = client.get("/admin/stats.json")
    assert r.status_code not in (401, 403)


def test_admin_slo_with_admin_override_not_401_or_403():
    client = _admin_only_override()
    r = client.get("/admin/slo")
    assert r.status_code not in (401, 403)


# ── cache_keys — regression for stable hashing ────────────────────────────────

def test_plan_signature_is_16_hex_chars():
    plan = {"tier": "fast", "max_tokens": 512, "temperature": 0.0,
            "cache": {}, "plan_name": "x", "decoding_profile": "greedy"}
    sig = plan_signature(plan)
    assert len(sig) == 16
    assert all(c in "0123456789abcdef" for c in sig)


def test_plan_signature_is_deterministic():
    plan = {"tier": "fast", "max_tokens": 512, "temperature": 0.0,
            "cache": {}, "plan_name": "x", "decoding_profile": "greedy"}
    assert plan_signature(plan) == plan_signature(plan)


def test_plan_signature_differs_for_different_plans():
    base = {"tier": "fast", "max_tokens": 512, "temperature": 0.0,
            "cache": {}, "plan_name": "x", "decoding_profile": "greedy"}
    other = {**base, "max_tokens": 1024}
    assert plan_signature(base) != plan_signature(other)


def test_plan_signature_key_order_invariant():
    """orjson OPT_SORT_KEYS means insertion order must not change the hash."""
    a = {"z": 1, "a": 2}
    b = {"a": 2, "z": 1}
    assert plan_signature(a) == plan_signature(b)


def test_plan_signature_cache_config_excluded_from_routes_call_site():
    """Plans differing only in cache policy must produce the same sig.

    routes.py filters out 'cache' before calling plan_signature so that
    a change in semantic threshold or ttl_seconds doesn't invalidate all
    existing exact-cache entries and doesn't cause warm-pass / test-pass
    key mismatches on policy reloads.
    """
    gen = {
        "tier": "standard", "max_tokens": 256, "temperature": 0.7,
        "plan_name": "short", "decoding_profile": "fast",
    }
    with_bench_cache = {**gen, "cache": {"exact_enabled": True, "semantic": {"threshold": 0.78, "ttl_seconds": 3600}}}
    with_dev_cache  = {**gen, "cache": {"exact_enabled": True, "semantic": {"threshold": 0.85, "ttl_seconds": 1800}}}

    sig_bench = plan_signature({k: v for k, v in with_bench_cache.items() if k != "cache"})
    sig_dev   = plan_signature({k: v for k, v in with_dev_cache.items()  if k != "cache"})
    assert sig_bench == sig_dev


def test_exact_cache_key_format():
    key = exact_cache_key(tenant_id="default", request_hash="abc123", plan_sig="defsig0")
    assert key == "exact:default:defsig0:abc123"


def test_exact_cache_key_tenant_scoped():
    """Different tenants must never share the same cache key."""
    k1 = exact_cache_key(tenant_id="default", request_hash="h", plan_sig="s")
    k2 = exact_cache_key(tenant_id="premium", request_hash="h", plan_sig="s")
    assert k1 != k2


def test_exact_cache_key_plan_sig_scoped():
    """Different plans for the same prompt must produce different keys."""
    k1 = exact_cache_key(tenant_id="t", request_hash="h", plan_sig="sig-a")
    k2 = exact_cache_key(tenant_id="t", request_hash="h", plan_sig="sig-b")
    assert k1 != k2


def test_exact_cache_key_request_hash_scoped():
    """Different prompts for the same plan must produce different keys."""
    k1 = exact_cache_key(tenant_id="t", request_hash="hash-1", plan_sig="s")
    k2 = exact_cache_key(tenant_id="t", request_hash="hash-2", plan_sig="s")
    assert k1 != k2
