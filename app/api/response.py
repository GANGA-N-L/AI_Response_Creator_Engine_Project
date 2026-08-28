from fastapi import APIRouter, Depends, Header, Request
from sqlalchemy.orm import Session

from app.core.rate_limiter import get_auth_subject_or_ip, limiter
from app.database.database import get_db
from app.schemas.response import ErrorResponse, ResponseRequest, SuccessResponse
from app.services import ai_response_service


router = APIRouter(
    prefix="/ai",
    tags=["AI Response"]
)


@router.post(
    "/generate-response",
    response_model=SuccessResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    }
)
@limiter.limit("60/minute", key_func=get_auth_subject_or_ip)
def generate_ai_response(
    request: Request,
    payload: ResponseRequest,
    db: Session = Depends(get_db),
    idempotency_key: str | None = Header(
        default=None,
        alias="Idempotency-Key",
        max_length=255,
        description=(
            "Optional client-generated key. Retrying the same request "
            "with the same key returns the original response instead of "
            "consuming credits or calling the AI provider again."
        ),
    ),
):
    return ai_response_service.generate_ai_response(
        db=db,
        customer_id=payload.customer_id,
        prompt_id=payload.prompt_id,
        text=payload.text,
        idempotency_key=idempotency_key,
    )
