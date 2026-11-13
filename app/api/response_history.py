from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.database import get_db
from app.schemas.response_history import ResponseHistoryEntry
from app.services import response_history_service

router = APIRouter(
    prefix="/customers/{customer_id}/response-history",
    tags=["Response History"],
)


@router.get("/", response_model=list[ResponseHistoryEntry])
async def get_response_history(
    customer_id: int,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """
    Support/dispute-resolution lookup: what this customer actually
    submitted and received, newest first. Always scoped to the
    customer_id in the path - there is no endpoint that lists across
    customers, so this is no broader a read than any other per-
    customer resource already exposed behind the same bearer-token
    check (e.g. GET /customers/{customer_id}).

    Retention-limited - see settings.response_history_retention_days
    and response_history_service.purge_expired_response_history. A row
    past its retention window is simply gone, not flagged as deleted.
    """

    return await response_history_service.get_response_history(
        db, customer_id, limit, offset
    )
