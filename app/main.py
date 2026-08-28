from fastapi import Depends, FastAPI, status
from fastapi.responses import JSONResponse
from slowapi.middleware import SlowAPIMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.rate_limiter import limiter
from app.core.security import verify_access_token

from app.api.auth import router as auth_router
from app.api.customer import router as customer_router
from app.api.response import router as response_router
from app.api.prompt import router as prompt_router

from app.database.database import get_db
from app.exceptions.handlers import register_exception_handlers
from app.logging.request_context import RequestIdMiddleware
from app.services.ai_provider import check_availability


app = FastAPI(
    title=settings.app_name,
    description="AI-powered response rewriting and correction service.",
    version=settings.app_version
)


app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(RequestIdMiddleware)

register_exception_handlers(app)


app.include_router(auth_router)

protected = [Depends(verify_access_token)]

app.include_router(customer_router, dependencies=protected)
app.include_router(response_router, dependencies=protected)
app.include_router(prompt_router, dependencies=protected)


@app.get("/")
def home():
    return {
        "message": "AI Response Creator API is running!"
    }


@app.get("/health")
def health(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        database_ok = True
    except Exception:
        database_ok = False

    ai_provider_ok = check_availability()

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
        },
    )
