"""FastAPI application lifecycle and background maintenance tasks."""

from __future__ import annotations

import asyncio

from fastapi import FastAPI

from app.api.routes import router
from app.api.admin_routes import admin
from app.core.logging import configure_logging, get_logger
from app.core.settings import settings
from app.core.runtime import init_scheduler
from app.core.telemetry import configure_tracing
from app.core.slo_checker import run_slo_check
from app.db.postgres import get_sessionmaker

log = get_logger(component="main")

_slo_task: asyncio.Task | None = None


def create_app() -> FastAPI:
    """Configure and assemble the relay application."""

    configure_logging(settings.relay_log_level)
    configure_tracing()  # Safe to call more than once during test app creation.
    app = FastAPI(title="LLM Relay", version="0.3.0")

    app.include_router(router)
    app.include_router(admin)

    @app.on_event("startup")
    async def _startup() -> None:
        global _slo_task
        policy = settings.load_policy()
        init_scheduler(policy)
        # The first model download is slow, so prototype setup must not block startup.
        asyncio.create_task(_populate_prototype_embeddings(), name="embed_prototypes")
        _slo_task = asyncio.create_task(_slo_loop(policy), name="slo_checker")
        log.info("startup_complete", version="0.3.0", policy_version=policy.policy_version)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        global _slo_task
        if _slo_task is not None and not _slo_task.done():
            _slo_task.cancel()
            try:
                await _slo_task
            except asyncio.CancelledError:
                pass

    return app


async def _populate_prototype_embeddings() -> None:
    """Fill missing intent-prototype embeddings without recomputing stored rows."""
    from sqlalchemy import text as sa_text
    from app.core.embeddings import embed_text

    def _vec_literal(vec: list[float]) -> str:
        return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

    async with get_sessionmaker()() as session:
        res = await session.execute(
            sa_text("SELECT id, prompt_text FROM complexity_prototypes WHERE embedding IS NULL")
        )
        rows = res.mappings().all()

    if not rows:
        log.info("prototype_embeddings_ok", count=0)
        return

    log.info("populating_prototype_embeddings", count=len(rows))
    async with get_sessionmaker()() as session:
        for row in rows:
            vec = embed_text(row["prompt_text"])
            await session.execute(
                sa_text(
                    "UPDATE complexity_prototypes SET embedding = (:vec)::vector WHERE id = :id"
                ),
                {"vec": _vec_literal(vec), "id": row["id"]},
            )
        await session.commit()
    log.info("prototype_embeddings_populated", count=len(rows))


async def _slo_loop(policy) -> None:
    """Evaluate SLO compliance once per minute until shutdown."""
    while True:
        try:
            async with get_sessionmaker()() as session:
                await run_slo_check(session, policy)
        except Exception as exc:
            log.warning("slo_loop_error", error=str(exc))
        await asyncio.sleep(60)


app = create_app()
