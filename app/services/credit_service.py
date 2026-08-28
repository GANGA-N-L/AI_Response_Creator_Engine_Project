import math

from sqlalchemy.orm import Session
from sqlalchemy import update

from app.core.config import settings
from app.logging.logger import logger
from app.models.credit_transaction import CreditTransaction
from app.models.customer import Customer


def calculate_credit_cost(text: str) -> int:
    """
    Computes how many credits a request costs, based on input length.

    Cost scales with input size (settings.characters_per_credit
    characters per credit), with a minimum charge of 1 credit.
    """

    return max(1, math.ceil(len(text) / settings.characters_per_credit))


def consume_credit(
    db: Session,
    customer_id: int,
    amount: int = 1,
    reason: str = "AI response generation",
) -> bool:
    """
    Atomically consumes credits from a customer and records the
    transaction in the credit ledger.

    Returns:
        True  - credits successfully consumed
        False - customer has insufficient available credits
    """

    try:
        result = db.execute(
            update(Customer)
            .where(
                Customer.id == customer_id,
                Customer.credits >= amount
            )
            .values(
                credits=Customer.credits - amount
            )
            .returning(Customer.credits)
        )

        row = result.first()

        if row is None:
            return False

        db.add(
            CreditTransaction(
                customer_id=customer_id,
                amount=-amount,
                balance_after=row[0],
                transaction_type="CONSUME",
                reason=reason,
            )
        )

        db.commit()

        return True

    except Exception:
        logger.error(
            "Credit consumption failed",
            extra={"customer_id": customer_id, "amount": amount},
            exc_info=True,
        )
        db.rollback()
        raise


def refund_credit(
    db: Session,
    customer_id: int,
    amount: int = 1,
    reason: str = "Refund for failed AI provider call",
) -> bool:
    """
    Atomically refunds credits to a customer and records the
    transaction in the credit ledger, compensating for credits that
    were consumed but not honored (e.g. the AI provider call that
    followed consumption failed).

    Returns:
        True  - credits successfully refunded
        False - customer no longer exists
    """

    try:
        result = db.execute(
            update(Customer)
            .where(Customer.id == customer_id)
            .values(
                credits=Customer.credits + amount
            )
            .returning(Customer.credits)
        )

        row = result.first()

        if row is None:
            return False

        db.add(
            CreditTransaction(
                customer_id=customer_id,
                amount=amount,
                balance_after=row[0],
                transaction_type="REFUND",
                reason=reason,
            )
        )

        db.commit()

        return True

    except Exception:
        logger.error(
            "Credit refund failed",
            extra={"customer_id": customer_id, "amount": amount},
            exc_info=True,
        )
        db.rollback()
        raise
