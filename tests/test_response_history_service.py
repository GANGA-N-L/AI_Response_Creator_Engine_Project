from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

from sqlalchemy import select

from app.models.customer import Customer
from app.models.response_history import ResponseHistory
from app.services.response_history_service import (
    get_response_history,
    purge_expired_response_history,
    record_response_history,
)


async def test_record_response_history_writes_row(customer, prompt, db):
    await record_response_history(
        db, customer.id, prompt.id, "Thnks for your help", "Thanks for your help."
    )

    result = await db.execute(
        select(ResponseHistory).where(ResponseHistory.customer_id == customer.id)
    )
    row = result.scalar_one()

    assert row.prompt_id == prompt.id
    assert row.input_text == "Thnks for your help"
    assert row.output_text == "Thanks for your help."
    assert row.expires_at > datetime.now(timezone.utc)


# Purely additive bookkeeping - a logging failure must never cost the
# customer the answer they already received, same fail-open contract
# as ai_usage_log_service.record_ai_usage.
async def test_record_response_history_fails_open_on_db_error(customer, prompt, db):
    with patch.object(db, "commit", AsyncMock(side_effect=Exception("db down"))):
        await record_response_history(
            db, customer.id, prompt.id, "text", "answer"
        )


async def test_get_response_history_scoped_to_customer(customer, prompt, db):
    other_customer = Customer(customer_name="Other Customer")
    db.add(other_customer)
    await db.flush()

    await record_response_history(db, customer.id, prompt.id, "mine", "mine reply")
    await record_response_history(
        db, other_customer.id, prompt.id, "theirs", "theirs reply"
    )

    rows = await get_response_history(db, customer.id, limit=50, offset=0)

    assert len(rows) == 1
    assert rows[0].input_text == "mine"


async def test_get_response_history_orders_newest_first(customer, prompt, db):
    await record_response_history(db, customer.id, prompt.id, "first", "first reply")
    await record_response_history(db, customer.id, prompt.id, "second", "second reply")

    rows = await get_response_history(db, customer.id, limit=50, offset=0)

    assert [row.input_text for row in rows] == ["second", "first"]


async def test_purge_expired_response_history_deletes_only_expired_rows(
    customer, prompt, db
):
    await record_response_history(db, customer.id, prompt.id, "fresh", "fresh reply")

    expired = ResponseHistory(
        customer_id=customer.id,
        prompt_id=prompt.id,
        input_text="stale",
        output_text="stale reply",
        expires_at=datetime.now(timezone.utc) - timedelta(days=1),
    )
    db.add(expired)
    await db.commit()

    deleted_count = await purge_expired_response_history(db)

    assert deleted_count == 1

    rows = await get_response_history(db, customer.id, limit=50, offset=0)
    assert [row.input_text for row in rows] == ["fresh"]
