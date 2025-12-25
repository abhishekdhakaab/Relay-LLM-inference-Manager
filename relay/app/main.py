from fastapi import FastAPI

from app.api.routes import router
from app.api.admin_routes import admin   # <-- must exist
from app.core.logging import configure_logging
from app.core.settings import settings
from app.core.runtime import init_scheduler


def create_app() -> FastAPI:
    configure_logging(settings.relay_log_level)
    app = FastAPI(title="LLM Relay", version="0.1.0")

    app.include_router(router)
    app.include_router(admin) 

    @app.on_event("startup")
    async def _startup() -> None:
        policy = settings.load_policy()
        init_scheduler(policy)

    return app


app = create_app()
