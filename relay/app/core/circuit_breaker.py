"""Async circuit breaker that contains repeated backend failures."""
from __future__ import annotations

import asyncio
import time
from enum import Enum
from typing import Optional

from app.core.settings import CircuitBreakerConfig

class State(Enum):
    CLOSED = "CLOSED"       # normal — all requests pass through
    OPEN = "OPEN"           # failing — all requests rejected immediately
    HALF_OPEN = "HALF_OPEN" # recovering — one probe allowed through


class CircuitOpenError(Exception):
    """Raised by before_call() when the circuit is OPEN."""

    def __init__(self, retry_after_seconds: int) -> None:
        super().__init__(f"Circuit breaker OPEN; retry in {retry_after_seconds}s")
        self.retry_after_seconds = retry_after_seconds


class CircuitBreaker:
    """
    Async-safe three-state circuit breaker.

    State transitions:
        CLOSED    → OPEN      : failure_threshold consecutive failures
        OPEN      → HALF_OPEN : after recovery_timeout_seconds
        HALF_OPEN → CLOSED    : success_threshold consecutive successes
        HALF_OPEN → OPEN      : any single failure
    """

    def __init__(self, config: CircuitBreakerConfig) -> None:
        self._cfg = config
        self._state = State.CLOSED
        self._failure_streak: int = 0
        self._success_streak: int = 0
        self._opened_at: Optional[float] = None
        self._last_failure_ts: Optional[float] = None
        self._total_opens: int = 0
        self._total_rejected: int = 0
        self._lock = asyncio.Lock()

    # Metrics are approximate by design; locking every scrape would add contention.

    @property
    def state(self) -> State:
        return self._state

    @property
    def total_opens(self) -> int:
        return self._total_opens

    @property
    def total_rejected(self) -> int:
        return self._total_rejected

    @property
    def last_failure_timestamp(self) -> Optional[float]:
        return self._last_failure_ts

    async def before_call(self) -> None:
        """Reject open-circuit calls or admit a probe after the cooldown."""
        async with self._lock:
            if self._state == State.CLOSED:
                return

            if self._state == State.OPEN:
                elapsed = time.monotonic() - (self._opened_at or 0.0)
                remaining = int(self._cfg.recovery_timeout_seconds - elapsed)
                if elapsed >= self._cfg.recovery_timeout_seconds:
                    # The first request after cooldown becomes the recovery probe.
                    self._state = State.HALF_OPEN
                    self._success_streak = 0
                else:
                    self._total_rejected += 1
                    raise CircuitOpenError(retry_after_seconds=max(remaining, 1))

            # HALF_OPEN traffic is evaluated by on_success/on_failure.

    async def on_success(self) -> None:
        """Call after a successful backend response."""
        async with self._lock:
            self._failure_streak = 0
            self._last_failure_ts = None

            if self._state == State.HALF_OPEN:
                self._success_streak += 1
                if self._success_streak >= self._cfg.success_threshold:
                    self._state = State.CLOSED
                    self._success_streak = 0
                    self._opened_at = None

    async def on_failure(self) -> None:
        """Call after a backend failure (timeout or 5xx)."""
        async with self._lock:
            self._last_failure_ts = time.time()
            self._failure_streak += 1

            if self._state == State.HALF_OPEN:
                # A failed recovery probe reopens the circuit immediately.
                self._state = State.OPEN
                self._opened_at = time.monotonic()
                self._total_opens += 1
                self._success_streak = 0
                return

            if (
                self._state == State.CLOSED
                and self._failure_streak >= self._cfg.failure_threshold
            ):
                self._state = State.OPEN
                self._opened_at = time.monotonic()
                self._total_opens += 1

    def state_gauge(self) -> int:
        """Map states to stable numeric values for metrics export."""
        return {State.CLOSED: 0, State.HALF_OPEN: 1, State.OPEN: 2}[self._state]
