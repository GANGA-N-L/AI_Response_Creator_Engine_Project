from unittest.mock import patch

import pytest
from sqlalchemy.exc import SQLAlchemyError

from app.models.customer import Customer
from app.services.credit_service import consume_credit, refund_credit


def test_consume_credit_success(customer, db):
    result = consume_credit(
        db,
        customer.id,
    )

    db.refresh(customer)

    assert result is True
    assert customer.credits == 19


def test_consume_credit_with_zero_credits(db):
    customer = Customer(
        customer_name="Credit Zero Test Customer",
        credits=0,
    )

    db.add(customer)
    db.commit()
    db.refresh(customer)

    result = consume_credit(
        db,
        customer.id,
    )

    db.refresh(customer)

    assert result is False
    assert customer.credits == 0


def test_consume_credit_customer_not_found(db):
    result = consume_credit(
        db,
        999999,
    )

    assert result is False


def test_consume_credit_database_failure(customer, db):
    with patch.object(
        db,
        "execute",
        side_effect=SQLAlchemyError("Simulated database failure"),
    ):
        with pytest.raises(SQLAlchemyError):
            consume_credit(
                db,
                customer.id,
            )

    db.refresh(customer)

    assert customer.credits == 20


def test_refund_credit_success(customer, db):
    consume_credit(db, customer.id)

    result = refund_credit(
        db,
        customer.id,
    )

    db.refresh(customer)

    assert result is True
    assert customer.credits == 20


def test_refund_credit_customer_not_found(db):
    result = refund_credit(
        db,
        999999,
    )

    assert result is False


def test_refund_credit_database_failure(customer, db):
    with patch.object(
        db,
        "execute",
        side_effect=SQLAlchemyError("Simulated database failure"),
    ):
        with pytest.raises(SQLAlchemyError):
            refund_credit(
                db,
                customer.id,
            )

    db.refresh(customer)

    assert customer.credits == 20