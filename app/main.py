import asyncio
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, status
from fastapi.responses import JSONResponse
from slowapi.middleware import SlowAPIMiddleware
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.rate_limiter import (
    check_redis_availability,
    close_rate_limiter_storage,
    limiter,
)
from app.core.security import verify_access_token

from app.api.auth import router as auth_router
from app.api.compliance import router as compliance_router
from app.api.customer import router as customer_router
from app.api.response import router as response_router
from app.api.response_history import router as response_history_router
from app.api.prompt import router as prompt_router

from app.database.database import SessionLocal, get_db
from app.exceptions.handlers import register_exception_handlers
from app.logging.logger import logger
from app.logging.request_context import RequestIdMiddleware
from app.services import huggingface_service, ollama_service
from app.services.ai_provider import check_availability
from app.services.pii_metrics_service import close_pii_metrics_client
from app.services.response_cache_service import close_response_cache
from app.services.response_history_service import purge_expired_response_history


async def _response_history_purge_loop() -> None:
    """
    Runs purge_expired_response_history on a recurring interval
    (settings.response_history_purge_interval_seconds) for the life of
    the app process - the "scheduled purge" half of response_history's
    retention policy, since this codebase has no external job
    scheduler to hook into. Purges once immediately on startup too, so
    rows that expired while the app was down don't linger for a full
    interval. A single failed purge is logged and retried next
    interval, never crashes the loop - see purge_expired_response_
    history's docstring for why running this redundantly across
    multiple worker processes is safe.
    """

    while True:
        try:
            async with SessionLocal() as db:
                deleted = await purge_expired_response_history(db)

            if deleted:
                logger.info(
                    "Purged expired response history rows",
                    extra={"deleted_count": deleted},
                )
        except Exception:
            logger.warning(
                "Response history purge failed - will retry next interval",
                exc_info=True,
            )

        await asyncio.sleep(settings.response_history_purge_interval_seconds)


@asynccontextmanager
async def lifespan(app: FastAPI):
    purge_task = asyncio.create_task(_response_history_purge_loop())

    yield

    purge_task.cancel()

    try:
        await purge_task
    except asyncio.CancelledError:
        pass

    # Both AI provider clients are created unconditionally at import time
    # (ai_provider.py imports both modules regardless of which one is
    # active), so both need releasing on shutdown - not just whichever
    # settings.ai_provider currently selects.
    try:
        await huggingface_service.client.close()
    except Exception:
        logger.warning("Error closing Hugging Face client", exc_info=True)

    try:
        await ollama_service.client.aclose()
    except Exception:
        logger.warning("Error closing Ollama client", exc_info=True)

    close_rate_limiter_storage()
    close_response_cache()
    close_pii_metrics_client()


app = FastAPI(
    title=settings.app_name,
    description="AI-powered response rewriting and correction service.",
    version=settings.app_version,
    lifespan=lifespan,
)


app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(RequestIdMiddleware)

register_exception_handlers(app)


app.include_router(auth_router)

protected = [Depends(verify_access_token)]

app.include_router(customer_router, dependencies=protected)
app.include_router(response_router, dependencies=protected)
app.include_router(response_history_router, dependencies=protected)
app.include_router(prompt_router, dependencies=protected)
app.include_router(compliance_router, dependencies=protected)


@app.get("/")
def home():
    return {
        "message": "AI Response Creator API is running!"
    }


@app.get("/health")
async def health(db: AsyncSession = Depends(get_db)):
    try:
        await db.execute(text("SELECT 1"))
        database_ok = True
    except Exception:
        database_ok = False

    ai_provider_ok = await check_availability()

    # Informational only - not included in `healthy`. Rate limiting fails
    # open when Redis is down, so the API keeps working either way; this
    # just makes that otherwise-silent degradation visible.
    redis_ok = check_redis_availability()

    healthy = database_ok and ai_provider_ok

    return JSONResponse(
        status_code=(
            status.HTTP_200_OK if healthy
            else status.HTTP_503_SERVICE_UNAVAILABLE
        ),
        content={
            "status": "ok" if healthy else "degraded",
            "database": "ok" if database_ok else "unreachable",
            "ai_provider": "ok" if ai_provider_ok else "unreachable",
            "rate_limiter": "ok" if redis_ok else "degraded",
        },
    )
