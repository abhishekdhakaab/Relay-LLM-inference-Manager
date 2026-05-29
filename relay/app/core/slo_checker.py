from __future__ import annotations

import time
from typing import Any

import numpy as np
from sqlalchemy import text as sa_text

from app.core.logging import get_logger
from app.core.settings import PolicyConfig

log = get_logger(component="slo_checker")

# In-memory store for Prometheus-compatible SLO gauges.
# Keyed by (tenant_id, metric_name) → (value, labels_dict).
# admin_routes.py reads this dict when serving /admin/metrics.
_slo_gauges: dict[str, float] = {}
_slo_last_checked: float = 0.0


def get_slo_gauges() -> dict[str, float]:
    """Read-only snapshot of current SLO gauges for the metrics endpoint."""
    return dict(_slo_gauges)


def get_last_checked_ts() -> float:
    return _slo_last_checked


async def run_slo_check(db_session: Any, policy: PolicyConfig) -> None:
    """
    Query the last 5 minutes of traces per tenant, compute p50/p95/p99 and
    error rate, compare against configured SLO thresholds, and update
    Prometheus-style in-memory gauges.

    Called by the background loop in main.py every 60 seconds.
    """
    global _slo_last_checked
    _slo_last_checked = time.time()

    for tenant_id, tenant_cfg in policy.tenants.items():
        slo = tenant_cfg.slo

        try:
            rows = await _fetch_recent_traces(db_session, tenant_id)
        except Exception as exc:
            log.warning("slo_check_db_error", tenant_id=tenant_id, error=str(exc))
            continue

        if not rows:
            # No traffic in last 5 minutes — health is "OK" (no breach evidence)
            _write_gauge(tenant_id, "slo_health", 1.0)
            continue

        latencies = np.array([r["latency_ms"] for r in rows if r["latency_ms"] is not None], dtype=float)
        error_count = sum(1 for r in rows if r.get("error"))
        total = len(rows)
        error_rate_pct = (error_count / total) * 100 if total > 0 else 0.0

        breached = False

        if len(latencies) > 0:
            p50 = float(np.percentile(latencies, 50))
            p95 = float(np.percentile(latencies, 95))
            p99 = float(np.percentile(latencies, 99))
        else:
            p50 = p95 = p99 = 0.0

        # Write per-dimension gauges and check breaches
        for metric, actual, threshold in [
            ("slo_latency_p50_ms", p50, slo.p50_ms),
            ("slo_latency_p95_ms", p95, slo.p95_ms),
            ("slo_latency_p99_ms", p99, slo.p99_ms),
        ]:
            status = "breached" if actual > threshold else "ok"
            key = f'{metric}{{tenant_id="{tenant_id}",status="{status}"}}'
            _slo_gauges[key] = actual
            if status == "breached":
                breached = True
                log.warning(
                    "slo_breach",
                    tenant_id=tenant_id,
                    dimension=metric,
                    threshold_ms=threshold,
                    actual_ms=round(actual, 1),
                )

        err_status = "breached" if error_rate_pct > slo.error_rate_max_pct else "ok"
        err_key = f'slo_error_rate_pct{{tenant_id="{tenant_id}",status="{err_status}"}}'
        _slo_gauges[err_key] = error_rate_pct
        if err_status == "breached":
            breached = True
            log.warning(
                "slo_error_rate_breach",
                tenant_id=tenant_id,
                threshold_pct=slo.error_rate_max_pct,
                actual_pct=round(error_rate_pct, 2),
            )

        health = 0.0 if breached else 1.0
        _write_gauge(tenant_id, "slo_health", health)

        log.info(
            "slo_check_complete",
            tenant_id=tenant_id,
            samples=total,
            p50_ms=round(p50, 1),
            p95_ms=round(p95, 1),
            p99_ms=round(p99, 1),
            error_rate_pct=round(error_rate_pct, 2),
            health=health,
        )


async def _fetch_recent_traces(db_session: Any, tenant_id: str) -> list[dict]:
    q = sa_text(
        """
        SELECT latency_ms, error
        FROM request_traces
        WHERE tenant_id = :tenant_id
          AND created_at > NOW() - INTERVAL '5 minutes'
        """
    )
    res = await db_session.execute(q, {"tenant_id": tenant_id})
    return [dict(row) for row in res.mappings().all()]


def _write_gauge(tenant_id: str, metric: str, value: float) -> None:
    key = f'{metric}{{tenant_id="{tenant_id}"}}'
    _slo_gauges[key] = value
