"""
Tests for Feature 4 — Redis-backed scheduler queue depth and async admission_check.
No live Redis or Postgres needed.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from app.core.policy_engine import ExecutionPlan
from app.core.scheduler import QueueFullError, Scheduler, ScheduledJob
from app.core.settings import PolicyConfig, SchedulerConfig, TenantPolicy


# ── Helpers ────────────────────────────────────────────────────────────────────

def _policy(
    *,
    workers: int = 2,
    max_depth: int = 200,
    admission_enabled: bool = True,
    degrade_enabled: bool = True,
    reject_enabled: bool = True,
    short_max_chars: int = 1200,
) -> PolicyConfig:
    from app.core.settings import (
        SchedulerAdmission,
        SchedulerAdmissionComputeMs,
        SchedulerDegrade,
        SchedulerReject,
        CostRouterConfig,
        CircuitBreakerConfig,
    )

    return PolicyConfig(
        policy_version="test-1",
        tenants={"default": TenantPolicy()},
        routing={},
        plans={},
        scheduler=SchedulerConfig(
            short_max_prompt_chars=short_max_chars,
            workers=workers,
            max_queue_depth_per_lane=max_depth,
            admission=SchedulerAdmission(
                enabled=admission_enabled,
                default_compute_ms=SchedulerAdmissionComputeMs(short=1200, long=3500),
                degrade=SchedulerDegrade(
                    enabled=degrade_enabled, max_tokens_floor=128, max_tokens_scale=0.5
                ),
                reject=SchedulerReject(enabled=reject_enabled, retry_after_seconds=2),
            ),
        ),
    )


def _make_job(lane: str = "short", tenant: str = "t1") -> ScheduledJob:
    loop = asyncio.get_event_loop()
    fut: asyncio.Future[object] = loop.create_future()

    async def noop() -> str:
        return "ok"

    plan = ExecutionPlan(
        plan_name="test", tier="fast", decoding_profile="greedy",
        max_tokens=512, temperature=0.0, cache={},
    )
    return ScheduledJob(
        request_id="r1", tenant_id=tenant, lane=lane,
        created_at=0.0, slo_ms=5000, plan=plan,
        run=noop, fut=fut, queue_entered_at=0.0,
    )


# ── Lane assignment ────────────────────────────────────────────────────────────

def test_short_lane_for_chars_at_boundary():
    s = Scheduler(_policy(short_max_chars=1200))
    assert s.lane_for_prompt_chars(1200) == "short"


def test_long_lane_one_char_over_boundary():
    s = Scheduler(_policy(short_max_chars=1200))
    assert s.lane_for_prompt_chars(1201) == "long"


def test_short_lane_for_tiny_prompt():
    s = Scheduler(_policy())
    assert s.lane_for_prompt_chars(10) == "short"


def test_long_lane_for_large_prompt():
    s = Scheduler(_policy())
    assert s.lane_for_prompt_chars(99_999) == "long"


# ── admission_check is a coroutine (Feature 4 contract) ───────────────────────

def test_admission_check_returns_coroutine():
    """After Feature 4, admission_check must be async."""
    s = Scheduler(_policy())
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=b"0")
    with patch("app.core.scheduler.get_redis", return_value=mock_redis):
        coro = s.admission_check(lane="short", tenant_slo_ms=5000, prompt_chars=100)
    assert asyncio.iscoroutine(coro), "admission_check must return a coroutine (async def)"
    coro.close()  # clean up without running


# ── Admission decisions ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_admission_within_slo_empty_queue():
    """Depth = 0 → predicted wait = 0 ms → within_slo."""
    s = Scheduler(_policy(workers=2))
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=b"0")

    with patch("app.core.scheduler.get_redis", return_value=mock_redis):
        admission, wait_ms = await s.admission_check(
            lane="short", tenant_slo_ms=5000, prompt_chars=100
        )

    assert admission.accepted is True
    assert admission.degraded is False
    assert admission.rejected is False
    assert admission.reason == "within_slo"
    assert wait_ms == 0


@pytest.mark.asyncio
async def test_admission_degrades_under_load():
    """
    depth=10, avg_compute=1200, workers=2 →
    predicted_wait = (10 × 1200) / 2 = 6000 ms
    predicted_total = 6000 + 1200 = 7200 ms > SLO 5000 ms → degrade
    """
    s = Scheduler(_policy(workers=2))
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=b"10")

    with patch("app.core.scheduler.get_redis", return_value=mock_redis):
        admission, _ = await s.admission_check(
            lane="short", tenant_slo_ms=5000, prompt_chars=100
        )

    assert admission.degraded is True
    assert admission.rejected is False
    assert admission.reason == "degrade_to_meet_slo"


@pytest.mark.asyncio
async def test_admission_rejects_when_both_degrade_disabled():
    """degrade off, reject on → rejected."""
    s = Scheduler(_policy(degrade_enabled=False, reject_enabled=True))
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=b"10")

    with patch("app.core.scheduler.get_redis", return_value=mock_redis):
        admission, _ = await s.admission_check(
            lane="short", tenant_slo_ms=5000, prompt_chars=100
        )

    assert admission.rejected is True
    assert admission.retry_after_seconds == 2
    assert admission.reason == "reject_predicted_slo_miss"


@pytest.mark.asyncio
async def test_admission_disabled_always_accepts():
    """admission.enabled=False skips all checks immediately."""
    s = Scheduler(_policy(admission_enabled=False))
    mock_redis = AsyncMock()

    with patch("app.core.scheduler.get_redis", return_value=mock_redis):
        admission, wait_ms = await s.admission_check(
            lane="short", tenant_slo_ms=1, prompt_chars=100  # absurdly tight SLO
        )

    assert admission.accepted is True
    assert admission.reason == "admission_disabled"
    assert wait_ms == 0


@pytest.mark.asyncio
async def test_admission_accepts_when_both_flags_off():
    """degrade off, reject off → default accept."""
    s = Scheduler(_policy(degrade_enabled=False, reject_enabled=False))
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=b"10")

    with patch("app.core.scheduler.get_redis", return_value=mock_redis):
        admission, _ = await s.admission_check(
            lane="short", tenant_slo_ms=5000, prompt_chars=100
        )

    assert admission.accepted is True
    assert admission.rejected is False
    assert admission.degraded is False


# ── Redis depth helpers ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_incr_depth_uses_lane_key():
    s = Scheduler(_policy())
    mock_redis = AsyncMock()
    mock_redis.incr = AsyncMock()

    with patch("app.core.scheduler.get_redis", return_value=mock_redis):
        await s._incr_depth("short")

    mock_redis.incr.assert_called_once_with("scheduler:depth:short")


@pytest.mark.asyncio
async def test_decr_depth_normal_decrement():
    s = Scheduler(_policy())
    mock_redis = AsyncMock()
    mock_redis.decr = AsyncMock(return_value=3)
    mock_redis.set = AsyncMock()

    with patch("app.core.scheduler.get_redis", return_value=mock_redis):
        await s._decr_depth("short")

    mock_redis.decr.assert_called_once_with("scheduler:depth:short")
    mock_redis.set.assert_not_called()  # value was positive, no clamp needed


@pytest.mark.asyncio
async def test_decr_depth_clamps_to_zero_on_negative():
    """If Redis goes negative (counter drift), must reset to 0."""
    s = Scheduler(_policy())
    mock_redis = AsyncMock()
    mock_redis.decr = AsyncMock(return_value=-1)
    mock_redis.set = AsyncMock()

    with patch("app.core.scheduler.get_redis", return_value=mock_redis):
        await s._decr_depth("short")

    mock_redis.set.assert_called_once_with("scheduler:depth:short", 0)


@pytest.mark.asyncio
async def test_get_redis_depth_returns_int_from_redis():
    s = Scheduler(_policy())
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=b"7")

    with patch("app.core.scheduler.get_redis", return_value=mock_redis):
        depth = await s._get_redis_depth("short")

    assert depth == 7


@pytest.mark.asyncio
async def test_get_redis_depth_falls_back_on_redis_error():
    """Redis failure → fall back to local in-process queue depth."""
    s = Scheduler(_policy())
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(side_effect=ConnectionError("redis down"))

    with patch("app.core.scheduler.get_redis", return_value=mock_redis):
        depth = await s._get_redis_depth("short")

    assert depth == 0  # empty local queues


@pytest.mark.asyncio
async def test_get_redis_depth_none_falls_back_to_local():
    """Redis returns None (key missing) → fall back to local depth."""
    s = Scheduler(_policy())
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)

    with patch("app.core.scheduler.get_redis", return_value=mock_redis):
        depth = await s._get_redis_depth("short")

    assert depth == 0


# ── submit + Redis incr ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_submit_increments_redis_depth():
    """Every successful submit must call _incr_depth → Redis INCR."""
    s = Scheduler(_policy())
    mock_redis = AsyncMock()
    mock_redis.incr = AsyncMock()

    job = _make_job("short")
    with patch("app.core.scheduler.get_redis", return_value=mock_redis):
        await s.submit(job)

    mock_redis.incr.assert_called_once_with("scheduler:depth:short")


@pytest.mark.asyncio
async def test_submit_raises_queue_full_at_cap():
    """submit beyond max_queue_depth_per_lane raises QueueFullError."""
    s = Scheduler(_policy(max_depth=1))
    mock_redis = AsyncMock()
    mock_redis.incr = AsyncMock()

    job1 = _make_job("short", "t1")
    job2 = _make_job("short", "t1")

    with patch("app.core.scheduler.get_redis", return_value=mock_redis):
        await s.submit(job1)
        with pytest.raises(QueueFullError):
            await s.submit(job2)


@pytest.mark.asyncio
async def test_submit_different_tenants_share_lane_depth():
    """Two different tenants in the same lane share the lane depth counter."""
    s = Scheduler(_policy(max_depth=2))
    mock_redis = AsyncMock()
    mock_redis.incr = AsyncMock()

    job_a = _make_job("short", "tenant-a")
    job_b = _make_job("short", "tenant-b")

    with patch("app.core.scheduler.get_redis", return_value=mock_redis):
        await s.submit(job_a)
        await s.submit(job_b)

    assert mock_redis.incr.call_count == 2
