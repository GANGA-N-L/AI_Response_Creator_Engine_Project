from unittest.mock import AsyncMock, patch

from sqlalchemy import select

from app.models.ai_usage_log import AiUsageLog
from app.services.ai_usage_log_service import record_ai_usage
from app.services.token_usage import TokenUsage


async def test_record_ai_usage_writes_row(customer, prompt, db):
    usage = TokenUsage(prompt_tokens=12, completion_tokens=7, total_tokens=19)

    await record_ai_usage(
        db,
        customer.id,
        prompt.id,
        "Thnks for your help",
        "Thanks for your help.",
        usage,
    )

    result = await db.execute(
        select(AiUsageLog).where(AiUsageLog.customer_id == customer.id)
    )
    row = result.scalar_one()

    assert row.prompt_id == prompt.id
    assert row.input_text_length == len("Thnks for your help")
    assert row.output_text_length == len("Thanks for your help.")
    assert row.prompt_tokens == 12
    assert row.completion_tokens == 7
    assert row.total_tokens == 19


# Purely additive bookkeeping - a logging failure must never cost the
# customer the answer they already received, same fail-open contract
# as pii_metrics_service.record_pii_visibility.
async def test_record_ai_usage_fails_open_on_db_error(customer, prompt, db):
    usage = TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)

    with patch.object(db, "commit", AsyncMock(side_effect=Exception("db down"))):
        await record_ai_usage(
            db, customer.id, prompt.id, "text", "answer", usage
        )
