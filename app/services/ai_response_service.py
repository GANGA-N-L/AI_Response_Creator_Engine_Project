from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.rate_limiter import check_customer_rate_limit
from app.exceptions.ai_exceptions import AIProviderError
from app.exceptions.response_exceptions import (
    CustomerNotFoundError,
    DatabaseError,
    IdempotencyKeyConflictError,
    InputValidationError,
    InsufficientCreditsError,
    PromptNotFoundError,
    RateLimitExceededError,
)
from app.logging.logger import logger
from app.models.customer import Customer
from app.models.prompt import Prompt
from app.schemas.response import SuccessResponse
from app.services.ai_provider import generate_response
from app.services.credit_service import (
    calculate_credit_cost,
    consume_credit,
    refund_credit,
)
from app.services.idempotency_service import (
    compute_request_hash,
    get_cached_response,
    store_response,
)
from app.validators.input_validator import InputValidator


def generate_ai_response(
    db: Session,
    customer_id: int,
    prompt_id: int,
    text: str,
    idempotency_key: str | None = None,
) -> SuccessResponse:
    """
    Runs the full AI-response-generation workflow: per-customer rate
    limiting, input validation, customer/prompt lookup, idempotency
    replay, credit consumption, the AI provider call, and
    refund-on-failure.

    Raises a typed AppException subclass for every expected failure
    mode; FastAPI's registered exception handlers turn those into the
    corresponding HTTP response, so HTTP callers never need to catch
    anything themselves. Plain Python callers (e.g. tests) can catch
    these exceptions directly, with no FastAPI machinery involved.
    """

    if not check_customer_rate_limit(customer_id):
        raise RateLimitExceededError(
            "This customer has made too many requests recently. "
            "Please slow down and try again shortly."
        )

    validation_result = InputValidator.validate(text)

    if not validation_result.is_valid:
        raise InputValidationError(
            error_code=validation_result.error_code,
            message=validation_result.reason,
        )

    customer = db.query(Customer).filter(Customer.id == customer_id).first()

    if customer is None:
        raise CustomerNotFoundError("Customer not found.")

    prompt = db.query(Prompt).filter(Prompt.id == prompt_id).first()

    if prompt is None:
        raise PromptNotFoundError("Prompt not found.")

    request_hash = None

    if idempotency_key:
        request_hash = compute_request_hash(customer_id, prompt_id, text)

        cached = get_cached_response(db, customer_id, idempotency_key)

        if cached is not None:
            if cached.request_hash != request_hash:
                raise IdempotencyKeyConflictError(
                    "This idempotency key was already used for a "
                    "different request."
                )

            return SuccessResponse.model_validate(cached.response_body)

    credit_cost = calculate_credit_cost(text)

    try:
        credit_consumed = consume_credit(db, customer_id, credit_cost)
    except SQLAlchemyError:
        raise DatabaseError("Unable to update customer credits.")

    if not credit_consumed:
        raise InsufficientCreditsError(
            f"Insufficient credits. This request requires "
            f"{credit_cost} credit(s)."
        )

    try:
        ai_response = generate_response(prompt.prompt_text, text, customer_id)

    except AIProviderError as exc:
        try:
            refund_credit(db, customer_id, credit_cost)
        except SQLAlchemyError:
            logger.critical(
                "Credit refund failed after AI provider failure - "
                "manual reconciliation required",
                extra={"customer_id": customer_id},
            )

        # Always surface the generic default message to callers, regardless
        # of the specific internal reason (connection failure, malformed
        # output, etc.) - matches the original behavior, where response.py
        # hardcoded this same generic text rather than exposing exc.message.
        raise AIProviderError() from exc

    except Exception:
        try:
            refund_credit(db, customer_id, credit_cost)
        except SQLAlchemyError:
            logger.critical(
                "Credit refund failed after unexpected error - "
                "manual reconciliation required",
                extra={"customer_id": customer_id},
            )

        raise

    success_response = SuccessResponse(
        status_code=200,
        answer=ai_response,
        credits=customer.credits,
        credits_used=credit_cost,
    )

    if idempotency_key:
        store_response(
            db,
            customer_id,
            idempotency_key,
            request_hash,
            200,
            success_response.model_dump(),
        )

    return success_response
