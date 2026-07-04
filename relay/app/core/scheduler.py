"""Two-lane scheduler with tenant fairness and SLO-aware admission control."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, Optional

from app.core.policy_engine import ExecutionPlan
from app.core.settings import PolicyConfig
from app.db.redis_client import get_redis


@dataclass
class ScheduledJob:
    """A backend call waiting for a scheduler worker."""

    request_id: str
    tenant_id: str
    lane: str  # "short" | "long"
    created_at: float
    slo_ms: int
    plan: ExecutionPlan
    run: Callable[[], Awaitable[object]]  # returns backend result (opaque)
    fut: asyncio.Future[object]
    queue_entered_at: float


@dataclass(frozen=True)
class AdmissionResult:
    """Admission decision returned before a job enters a lane."""

    accepted: bool
    degraded: bool
    rejected: bool
    reason: str
    retry_after_seconds: int | None = None


class Scheduler:
    """Serve short work first while rotating fairly across tenants."""

    def __init__(self, policy: PolicyConfig):
        self.policy = policy

        self._lock = asyncio.Lock()

        # Separate tenant queues prevent one tenant from monopolizing a lane.
        self._queues: Dict[str, Dict[str, asyncio.Queue[ScheduledJob]]] = {
            "short": {},
            "long": {},
        }

        # The cursor advances only after a tenant supplies a job.
        self._rr_order: Dict[str, list[str]] = {"short": [], "long": []}
        self._rr_index: Dict[str, int] = {"short": 0, "long": 0}

        self._workers: list[asyncio.Task[None]] = []
        self._stop = asyncio.Event()
    def start(self) -> None:
        """Start the configured number of in-process workers."""
        workers = int(self.policy.scheduler.workers)
        for i in range(workers):
            self._workers.append(asyncio.create_task(self._worker_loop(i)))

    async def stop(self) -> None:
        """Cancel workers and wait for their cleanup."""
        self._stop.set()
        for t in self._workers:
            t.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)

    def lane_for_prompt_chars(self, prompt_chars: int) -> str:
        return "short" if prompt_chars <= int(self.policy.scheduler.short_max_prompt_chars) else "long"

    async def _incr_depth(self, lane: str) -> None:
        """Increment the shared Redis depth counter for this lane."""
        try:
            redis = get_redis()
            await redis.incr(f"scheduler:depth:{lane}")
        except Exception:
            pass  # Admission can still use local depth during a Redis outage.

    async def _decr_depth(self, lane: str) -> None:
        """Decrement the shared Redis depth counter for this lane."""
        try:
            redis = get_redis()
            key = f"scheduler:depth:{lane}"
            new = await redis.decr(key)
            if new < 0:
                await redis.set(key, 0)  # Retries can otherwise drift below zero.
        except Exception:
            pass

    async def _get_redis_depth(self, lane: str) -> int:
        """Read total depth from Redis; fall back to in-process depth on error."""
        try:
            redis = get_redis()
            val = await redis.get(f"scheduler:depth:{lane}")
            return int(val) if val is not None else self._local_depth(lane)
        except Exception:
            return self._local_depth(lane)

    def _local_depth(self, lane: str) -> int:
        tmap = self._queues.get(lane, {})
        return sum(q.qsize() for q in tmap.values())

    async def submit(self, job: ScheduledJob) -> None:
        """Enqueue an admitted job without exceeding the lane-wide cap."""
        async with self._lock:
            lane = job.lane
            tenant = job.tenant_id

            tmap = self._queues[lane]
            if tenant not in tmap:
                tmap[tenant] = asyncio.Queue()
                self._rr_order[lane].append(tenant)

            # The cap spans every tenant queue in this lane.
            total_depth = sum(q.qsize() for q in tmap.values())
            if total_depth >= int(self.policy.scheduler.max_queue_depth_per_lane):
                raise QueueFullError(f"{lane} queue full")

            await tmap[tenant].put(job)
            await self._incr_depth(lane)
    async def _worker_loop(self, worker_id: int) -> None:
        """Resolve each job's future with its backend result or exception."""

        while not self._stop.is_set():
            job = await self._dequeue_fair()
            if job is None:
                await asyncio.sleep(0.005)  # Small polling delay avoids a busy loop.
                continue

            if job.fut.cancelled():
                continue

            await self._decr_depth(job.lane)

            try:
                res = await job.run()
                if not job.fut.done():
                    job.fut.set_result(res)
            except Exception as e:
                if not job.fut.done():
                    job.fut.set_exception(e)
    async def _dequeue_fair(self) -> Optional[ScheduledJob]:
        async with self._lock:
            # Short-first service protects interactive tail latency.
            job = self._dequeue_lane("short")
            if job is not None:
                return job
            return self._dequeue_lane("long")

    def _dequeue_lane(self, lane: str) -> Optional[ScheduledJob]:
        tenants = self._rr_order[lane]
        if not tenants:
            return None
        tmap = self._queues[lane]

        n = len(tenants)
        start = self._rr_index[lane] % n

        for offset in range(n):
            idx = (start + offset) % n
            tenant = tenants[idx]
            q = tmap.get(tenant)
            if q is None or q.qsize() == 0:
                continue
            self._rr_index[lane] = idx + 1
            return q.get_nowait()

        return None

    async def admission_check(
        self,
        *,
        lane: str,
        tenant_slo_ms: int,
        prompt_chars: int,
    ) -> tuple[AdmissionResult, int]:
        """Predict queue delay and accept, degrade, or reject the request."""

        adm = self.policy.scheduler.admission
        if not adm.enabled:
            return AdmissionResult(True, False, False, "admission_disabled"), 0

        workers = max(1, int(self.policy.scheduler.workers))
        avg_compute = adm.default_compute_ms.short if lane == "short" else adm.default_compute_ms.long

        depth = await self._get_redis_depth(lane)
        predicted_wait_ms = int((depth * avg_compute) / workers)

        predicted_total_ms = predicted_wait_ms + avg_compute

        if predicted_total_ms <= tenant_slo_ms:
            return AdmissionResult(True, False, False, "within_slo"), predicted_wait_ms

        if adm.degrade.enabled:
            return AdmissionResult(True, True, False, "degrade_to_meet_slo"), predicted_wait_ms

        if adm.reject.enabled:
            return AdmissionResult(False, False, True, "reject_predicted_slo_miss", adm.reject.retry_after_seconds), predicted_wait_ms

        # If neither policy is enabled, observability wins over silent rejection.
        return AdmissionResult(True, False, False, "accept_even_if_slo_miss"), predicted_wait_ms


class QueueFullError(RuntimeError):
    pass
