from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.logging.logger import logger
from app.models.response_history import ResponseHistory


async def record_response_history(
    db: AsyncSession,
    customer_id: int,
    prompt_id: int,
    input_text: str,
    output_text: str,
) -> None:
    """
    Always written for every request that produces an answer (a fresh
    generation or a response-cache hit) - a cache hit is still a real
    submission a customer may later ask support about. Not written on
    an idempotency-key replay (ai_response_service.py's early return),
    since that's re-serving a fact already recorded the first time the
    same input was submitted, not new information.

    Purely additive bookkeeping, same fail-open contract as
    ai_usage_log_service.record_ai_usage and pii_metrics_service.
    record_pii_visibility: a logging failure must never cost the
    customer the answer they were just charged a credit for.
    """

    expires_at = datetime.now(timezone.utc) + timedelta(
        days=settings.response_history_retention_days
    )

    try:
        db.add(
            ResponseHistory(
                customer_id=customer_id,
                prompt_id=prompt_id,
                input_text=input_text,
                output_text=output_text,
                expires_at=expires_at,
            )
        )
        await db.commit()
    except Exception:
        logger.warning(
            "Failed to record response history",
            exc_info=True,
            extra={"customer_id": customer_id, "prompt_id": prompt_id},
        )
        await db.rollback()


async def get_response_history(
    db: AsyncSession, customer_id: int, limit: int, offset: int
) -> list[ResponseHistory]:
    """
    Scoped strictly to one explicit customer_id, newest first - there
    is no query path that returns another customer's rows or lists
    across customers, matching every other customer-scoped read in
    this app (see customer_service.get_customer).

    Orders by id as a tiebreaker after created_at: Postgres's now()
    returns the transaction start time, not the statement time, so two
    rows committed inside the same transaction (or just close enough
    together) can tie on created_at - id is monotonically increasing
    and never ties, so it keeps "newest first" deterministic.
    """

    result = await db.execute(
        select(ResponseHistory)
        .where(ResponseHistory.customer_id == customer_id)
        .order_by(ResponseHistory.created_at.desc(), ResponseHistory.id.desc())
        .offset(offset)
        .limit(limit)
    )

    return list(result.scalars().all())


async def purge_expired_response_history(db: AsyncSession) -> int:
    """
    Deletes every response-history row past its retention window and
    returns the number of rows removed, for the caller to log. Called
    on a recurring interval by the background loop in app.main's
    lifespan. Idempotent - safe to run redundantly from multiple
    worker processes, each on its own schedule, since a row already
    deleted by another worker simply isn't matched again.
    """

    result = await db.execute(
        delete(ResponseHistory).where(
            ResponseHistory.expires_at < datetime.now(timezone.utc)
        )
    )
    await db.commit()

    return result.rowcount
