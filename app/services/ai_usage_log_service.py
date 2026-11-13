from sqlalchemy.ext.asyncio import AsyncSession

from app.logging.logger import logger
from app.models.ai_usage_log import AiUsageLog
from app.services.token_usage import TokenUsage


async def record_ai_usage(
    db: AsyncSession,
    customer_id: int,
    prompt_id: int,
    input_text: str,
    output_text: str,
    usage: TokenUsage,
) -> None:
    """
    Logs one row of real AI-provider token spend. Originally this only
    fired for a full rewrite generation - a cache hit or idempotency
    replay spent no new tokens, so callers skipped this entirely
    rather than logging zeros. That's no longer true now that
    ai_response_service's check_self_harm_risk runs unconditionally on
    every request, including cache hits and idempotency replays: those
    callers pass output_text="" for a safety-check-only row, since no
    rewrite was generated on that call - output_text_length of 0
    unambiguously flags this apart from a real rewrite row, because a
    genuine AI-generated rewrite is never empty (see
    ai_output_validator.py's empty-content rejection). Callers should
    still skip calling this entirely when usage.total_tokens is 0 (the
    "don't log zero-token rows" rule the cache-hit/replay case
    originally existed for still applies - see
    _record_self_harm_only_usage). Purely additive bookkeeping, same
    fail-open contract as pii_metrics_service.record_pii_visibility: a
    logging failure must never cost the customer the answer they were
    just charged no credits for but did spend real AI-provider tokens
    on.
    """

    try:
        db.add(
            AiUsageLog(
                customer_id=customer_id,
                prompt_id=prompt_id,
                input_text_length=len(input_text),
                output_text_length=len(output_text),
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            )
        )
        await db.commit()
    except Exception:
        logger.warning(
            "Failed to record AI usage log", exc_info=True,
            extra={"customer_id": customer_id, "prompt_id": prompt_id},
        )
        await db.rollback()
