from unittest.mock import patch

from sqlalchemy.exc import SQLAlchemyError


def fake_consume_credit(db, customer_id, amount=1):
    raise SQLAlchemyError("Simulated database failure")


def test_response_database_failure(client, customer, prompt, db):
    credits_before = customer.credits

    with patch(
        "app.services.ai_response_service.consume_credit",
        side_effect=fake_consume_credit,
    ):
        response = client.post(
            "/ai/generate-response",
            json={
                "customer_id": customer.id,
                "prompt_id": prompt.id,
                "text": "Thnks for your help",
            },
        )

    db.refresh(customer)

    assert response.status_code == 500

    assert response.json() == {
        "status_code": 500,
        "error_code": "DATABASE_ERROR",
        "message": "Unable to update customer credits.",
    }

    assert customer.credits == credits_before
