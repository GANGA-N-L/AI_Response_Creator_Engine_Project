from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from app.exceptions.ai_exceptions import AIProviderError
from app.main import app
from app.models.customer import Customer
from app.models.prompt import Prompt


# Testing the successful API request
def test_generate_response_success(client, customer, prompt):
    with patch(
        "app.services.ai_response_service.generate_response",
        return_value="Thanks for your help.",
    ):
        response = client.post(
            "/ai/generate-response",
            json={
                "customer_id": customer.id,
                "prompt_id": prompt.id,
                "text": "Thnks for your help",
            },
        )

    assert response.status_code == 200

    assert response.json() == {
        "status_code": 200,
        "answer": "Thanks for your help.",
        "credits": 19,
        "credits_used": 1,
    }


# Idempotency key: retrying the same request returns the cached response
# without consuming another credit or calling the AI provider again
def test_generate_response_idempotency_key_replays_cached_response(
    client, customer, prompt, db
):
    payload = {
        "customer_id": customer.id,
        "prompt_id": prompt.id,
        "text": "Thnks for your help",
    }

    with patch(
        "app.services.ai_response_service.generate_response",
        return_value="Thanks for your help.",
    ) as mock_generate_response:

        first = client.post(
            "/ai/generate-response",
            json=payload,
            headers={"Idempotency-Key": "test-key-123"},
        )

        second = client.post(
            "/ai/generate-response",
            json=payload,
            headers={"Idempotency-Key": "test-key-123"},
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()

    mock_generate_response.assert_called_once()

    db.refresh(customer)

    assert customer.credits == 19


# Idempotency key: reusing a key with a different request body is a conflict
def test_generate_response_idempotency_key_conflict(client, customer, prompt, db):
    with patch(
        "app.services.ai_response_service.generate_response",
        return_value="Thanks for your help.",
    ) as mock_generate_response:

        first = client.post(
            "/ai/generate-response",
            json={
                "customer_id": customer.id,
                "prompt_id": prompt.id,
                "text": "Thnks for your help",
            },
            headers={"Idempotency-Key": "test-key-456"},
        )

        second = client.post(
            "/ai/generate-response",
            json={
                "customer_id": customer.id,
                "prompt_id": prompt.id,
                "text": "A completely different message",
            },
            headers={"Idempotency-Key": "test-key-456"},
        )

    assert first.status_code == 200
    assert second.status_code == 409

    assert second.json() == {
        "status_code": 409,
        "error_code": "IDEMPOTENCY_KEY_CONFLICT",
        "message": "This idempotency key was already used for a different request.",
    }

    mock_generate_response.assert_called_once()

    db.refresh(customer)

    assert customer.credits == 19


# Empty-input API test
def test_generate_response_empty_input(client, customer, prompt):
    response = client.post(
        "/ai/generate-response",
        json={
            "customer_id": customer.id,
            "prompt_id": prompt.id,
            "text": "",
        },
    )

    assert response.status_code == 422

    assert response.json() == {
        "status_code": 422,
        "error_code": "EMPTY_INPUT",
        "message": "Input cannot be empty or whitespace only.",
    }


# Invalid Customer ID API test
def test_generate_response_customer_not_found(client, prompt):
    with patch(
        "app.services.ai_response_service.generate_response"
    ) as mock_generate_response:

        response = client.post(
            "/ai/generate-response",
            json={
                "customer_id": 999999,
                "prompt_id": prompt.id,
                "text": "Thnks for your help",
            },
        )

    assert response.status_code == 404

    assert response.json() == {
        "status_code": 404,
        "error_code": "CUSTOMER_NOT_FOUND",
        "message": "Customer not found.",
    }

    mock_generate_response.assert_not_called()


# Invalid Prompt ID API test
def test_generate_response_prompt_not_found(client, customer):
    with patch(
        "app.services.ai_response_service.generate_response"
    ) as mock_generate_response:

        response = client.post(
            "/ai/generate-response",
            json={
                "customer_id": customer.id,
                "prompt_id": 999999,
                "text": "Thnks for your help",
            },
        )

    assert response.status_code == 404

    assert response.json() == {
        "status_code": 404,
        "error_code": "PROMPT_NOT_FOUND",
        "message": "Prompt not found.",
    }

    mock_generate_response.assert_not_called()


# Insufficient Credits API test
def test_generate_response_insufficient_credits(client, db, prompt):
    customer = Customer(
        customer_name="API Insufficient Credits Test",
        credits=0,
    )

    db.add(customer)
    db.commit()
    db.refresh(customer)

    with patch(
        "app.services.ai_response_service.generate_response"
    ) as mock_generate_response:

        response = client.post(
            "/ai/generate-response",
            json={
                "customer_id": customer.id,
                "prompt_id": prompt.id,
                "text": "Thnks for your help",
            },
        )

    assert response.status_code == 400

    assert response.json() == {
        "status_code": 400,
        "error_code": "INSUFFICIENT_CREDITS",
        "message": "Insufficient credits. This request requires 1 credit(s).",
    }

    mock_generate_response.assert_not_called()


# AI Provider Failure API test
def test_generate_response_provider_failure(
    client,
    customer,
    prompt,
    db,
):
    credits_before = customer.credits

    with patch(
        "app.services.ai_response_service.generate_response",
        side_effect=AIProviderError(),
    ):
        response = client.post(
            "/ai/generate-response",
            json={
                "customer_id": customer.id,
                "prompt_id": prompt.id,
                "text": "Thnks for your help",
            },
        )

    assert response.status_code == 503

    assert response.json() == {
        "status_code": 503,
        "error_code": "AI_PROVIDER_UNAVAILABLE",
        "message": (
            "AI service is temporarily unavailable. "
            "Please try again later."
        ),
    }

    db.refresh(customer)

    assert customer.credits == credits_before


# Unexpected (non-AIProviderError) failure during the AI call
def test_generate_response_unexpected_provider_error(
    client,
    customer,
    prompt,
    db,
):
    credits_before = customer.credits

    lenient_client = TestClient(app, raise_server_exceptions=False)
    lenient_client.headers = client.headers

    with patch(
        "app.services.ai_response_service.generate_response",
        side_effect=RuntimeError("something unrelated broke"),
    ):
        response = lenient_client.post(
            "/ai/generate-response",
            json={
                "customer_id": customer.id,
                "prompt_id": prompt.id,
                "text": "Thnks for your help",
            },
        )

    assert response.status_code == 500

    assert response.json() == {
        "status_code": 500,
        "error_code": "INTERNAL_SERVER_ERROR",
        "message": (
            "An unexpected error occurred. "
            "Please try again later."
        ),
    }

    db.refresh(customer)

    assert customer.credits == credits_before


# Database Failure API test
def test_generate_response_database_failure(
    client,
    customer,
    prompt,
    db,
):
    credits_before = customer.credits

    with patch(
        "app.services.ai_response_service.generate_response",
        return_value="Thanks for your help.",
    ), patch(
        "app.services.ai_response_service.consume_credit",
        side_effect=SQLAlchemyError("Simulated database failure"),
    ):

        response = client.post(
            "/ai/generate-response",
            json={
                "customer_id": customer.id,
                "prompt_id": prompt.id,
                "text": "Thnks for your help",
            },
        )

    assert response.status_code == 500

    assert response.json() == {
        "status_code": 500,
        "error_code": "DATABASE_ERROR",
        "message": "Unable to update customer credits.",
    }

    db.refresh(customer)

    assert customer.credits == credits_before


# Repeated requests beyond the per-caller limit are rejected with 429,
# regardless of the underlying request outcome. Cycles across several
# customer_ids (each well under the per-customer limit) so it's the
# global per-caller limit that trips, not the per-customer one.
def test_generate_response_rate_limited(client):
    for i in range(60):
        response = client.post(
            "/ai/generate-response",
            json={
                "customer_id": 999_900 + (i % 6),
                "prompt_id": 999_999,
                "text": "Hello there",
            },
        )
        assert response.status_code == 404

    response = client.post(
        "/ai/generate-response",
        json={
            "customer_id": 999_900,
            "prompt_id": 999_999,
            "text": "Hello there",
        },
    )

    assert response.status_code == 429
    assert response.json()["error_code"] == "RATE_LIMIT_EXCEEDED"


# Repeated requests for the same customer_id are rejected with 429 once
# that customer's own limit is hit, independent of the global limit
def test_generate_response_customer_rate_limited(client):
    payload = {
        "customer_id": 999_888,
        "prompt_id": 999_999,
        "text": "Hello there",
    }

    for _ in range(20):
        response = client.post("/ai/generate-response", json=payload)
        assert response.status_code == 404

    limited = client.post("/ai/generate-response", json=payload)

    assert limited.status_code == 429
    assert limited.json()["error_code"] == "RATE_LIMIT_EXCEEDED"
    assert "customer" in limited.json()["message"].lower()

    # A different customer_id is unaffected - the limit is per-customer
    other_customer = client.post(
        "/ai/generate-response",
        json={**payload, "customer_id": 999_777},
    )

    assert other_customer.status_code == 404