from sqlalchemy.orm import Session

from app.core.config import settings
from app.exceptions.response_exceptions import CustomerNotFoundError
from app.models.credit_transaction import CreditTransaction
from app.models.customer import Customer


def create_customer(db: Session, customer_name: str) -> Customer:
    new_customer = Customer(
        customer_name=customer_name,
        credits=settings.default_credits,
    )

    db.add(new_customer)
    db.commit()
    db.refresh(new_customer)

    return new_customer


def list_customers(db: Session, limit: int, offset: int) -> list[Customer]:
    return (
        db.query(Customer)
        .offset(offset)
        .limit(limit)
        .all()
    )


def get_customer(db: Session, customer_id: int) -> Customer:
    customer = (
        db.query(Customer)
        .filter(Customer.id == customer_id)
        .first()
    )

    if customer is None:
        raise CustomerNotFoundError("Customer not found")

    return customer


def update_customer_credits(
    db: Session,
    customer_id: int,
    credits: int,
) -> Customer:
    customer = (
        db.query(Customer)
        .filter(Customer.id == customer_id)
        .with_for_update()
        .first()
    )

    if customer is None:
        raise CustomerNotFoundError("Customer not found")

    previous_credits = customer.credits
    customer.credits = credits

    delta = customer.credits - previous_credits

    if delta != 0:
        db.add(
            CreditTransaction(
                customer_id=customer.id,
                amount=delta,
                balance_after=customer.credits,
                transaction_type="ADMIN_ADJUSTMENT",
                reason="Manual credit adjustment via API",
            )
        )

    db.commit()
    db.refresh(customer)

    return customer


def list_customer_transactions(
    db: Session,
    customer_id: int,
    limit: int,
    offset: int,
) -> list[CreditTransaction]:
    customer = (
        db.query(Customer)
        .filter(Customer.id == customer_id)
        .first()
    )

    if customer is None:
        raise CustomerNotFoundError("Customer not found")

    return (
        db.query(CreditTransaction)
        .filter(CreditTransaction.customer_id == customer_id)
        .order_by(CreditTransaction.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
