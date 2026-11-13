import hashlib
import json

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.idempotency_key import IdempotencyKey


def compute_request_hash(customer_id: int, prompt_id: int, text: str) -> str:
    """
    Computes a stable fingerprint of the parts of a request that must
    match for an idempotency key to be reused safely.
    """

    payload = json.dumps(
        {"customer_id": customer_id, "prompt_id": prompt_id, "text": text},
        sort_keys=True,
    )

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def get_cached_response(
    db: AsyncSession,
    customer_id: int,
    idempotency_key: str,
) -> IdempotencyKey | None:
    result = await db.execute(
        select(IdempotencyKey).where(
            IdempotencyKey.customer_id == customer_id,
            IdempotencyKey.idempotency_key == idempotency_key,
        )
    )

    return result.scalar_one_or_none()


async def store_response(
    db: AsyncSession,
    customer_id: int,
    idempotency_key: str,
    request_hash: str,
    status_code: int,
    response_body: dict,
) -> IdempotencyKey:
    """
    Stores the response produced for an idempotency key, so a retried
    request with the same key can be answered without repeating the
    AI provider call.

    If a concurrent request already stored a response for the same key
    first, the unique constraint on (customer_id, idempotency_key)
    rejects this insert; in that case, the already-stored record is
    returned instead of raising.
    """

    record = IdempotencyKey(
        customer_id=customer_id,
        idempotency_key=idempotency_key,
        request_hash=request_hash,
        response_status_code=status_code,
        response_body=response_body,
    )

    db.add(record)

    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        existing = await get_cached_response(db, customer_id, idempotency_key)

        if existing is not None:
            return existing

        raise

    await db.refresh(record)

    return record
