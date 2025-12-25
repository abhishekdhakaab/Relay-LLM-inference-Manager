from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, Optional

from app.core.policy_engine import ExecutionPlan
from app.core.settings import PolicyConfig


@dataclass
class ScheduledJob:
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
    accepted: bool
    degraded: bool
    rejected: bool
    reason: str
    retry_after_seconds: int | None = None


class Scheduler:
    """
    Two-lane (short/long) fair scheduler with basic admission control.
    - Per-lane: per-tenant FIFO deques (implemented as asyncio Queues per tenant)
    - Fairness: round-robin across tenants that have queued work
    """

    def __init__(self, policy: PolicyConfig):
        self.policy = policy

        self._lock = asyncio.Lock()

        # lane -> tenant -> queue
        self._queues: Dict[str, Dict[str, asyncio.Queue[ScheduledJob]]] = {
            "short": {},
            "long": {},
        }

        # lane -> round robin tenant order (maintained)
        self._rr_order: Dict[str, list[str]] = {"short": [], "long": []}
        self._rr_index: Dict[str, int] = {"short": 0, "long": 0}

        self._workers: list[asyncio.Task[None]] = []
        self._stop = asyncio.Event()

    def start(self) -> None:
        workers = int(self.policy.scheduler.workers)
        for i in range(workers):
            self._workers.append(asyncio.create_task(self._worker_loop(i)))

    async def stop(self) -> None:
        self._stop.set()
        for t in self._workers:
            t.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)

    def lane_for_prompt_chars(self, prompt_chars: int) -> str:
        return "short" if prompt_chars <= int(self.policy.scheduler.short_max_prompt_chars) else "long"

    async def submit(self, job: ScheduledJob) -> None:
        async with self._lock:
            lane = job.lane
            tenant = job.tenant_id

            # create queue if needed
            tmap = self._queues[lane]
            if tenant not in tmap:
                tmap[tenant] = asyncio.Queue()
                self._rr_order[lane].append(tenant)

            # enforce max queue depth per lane (global cap, simple)
            total_depth = sum(q.qsize() for q in tmap.values())
            if total_depth >= int(self.policy.scheduler.max_queue_depth_per_lane):
                raise QueueFullError(f"{lane} queue full")

            await tmap[tenant].put(job)

    async def _worker_loop(self, worker_id: int) -> None:
        # naive strategy: prefer short lane, then long
        while not self._stop.is_set():
            job = await self._dequeue_fair()
            if job is None:
                await asyncio.sleep(0.005)
                continue

            if job.fut.cancelled():
                continue

            try:
                res = await job.run()
                if not job.fut.done():
                    job.fut.set_result(res)
            except Exception as e:
                if not job.fut.done():
                    job.fut.set_exception(e)

    async def _dequeue_fair(self) -> Optional[ScheduledJob]:
        async with self._lock:
            # Prefer short to reduce tail latency
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

    def admission_check(
        self,
        *,
        lane: str,
        tenant_slo_ms: int,
        prompt_chars: int,
    ) -> tuple[AdmissionResult, int]:

        adm = self.policy.scheduler.admission
        if not adm.enabled:
            return AdmissionResult(True, False, False, "admission_disabled"), 0

        workers = max(1, int(self.policy.scheduler.workers))
        avg_compute = adm.default_compute_ms.short if lane == "short" else adm.default_compute_ms.long

        # Approximate queue depth in this lane
        tmap = self._queues[lane]
        depth = sum(q.qsize() for q in tmap.values())
        predicted_wait_ms = int((depth * avg_compute) / workers)

        ## now to decide if we should admit this question: two factor
        ## 1.average compute = hardcoded based on the size of the prompt ( short or long)
        ## 2. predicted_wait_ms = (total number of all the requests in that particular lane * avg computer)/number of workers
        predicted_total_ms = predicted_wait_ms + avg_compute




        if predicted_total_ms <= tenant_slo_ms:
            return AdmissionResult(True, False, False, "within_slo"), predicted_wait_ms

        # Try degrade
        if adm.degrade.enabled:
            return AdmissionResult(True, True, False, "degrade_to_meet_slo"), predicted_wait_ms

        # Else reject
        if adm.reject.enabled:
            return AdmissionResult(False, False, True, "reject_predicted_slo_miss", adm.reject.retry_after_seconds), predicted_wait_ms

        # Default accept
        return AdmissionResult(True, False, False, "accept_even_if_slo_miss"), predicted_wait_ms


class QueueFullError(RuntimeError):
    pass
