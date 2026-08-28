from unittest.mock import patch

from app.exceptions.ai_exceptions import AIProviderError


def fake_generate_response(prompt, text, customer_id):
    raise AIProviderError()


def test_response_provider_failure(client, customer, prompt, db):
    credits_before = customer.credits

    with patch(
        "app.services.ai_response_service.generate_response",
        side_effect=fake_generate_response,
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

    assert response.status_code == 503

    assert response.json() == {
        "status_code": 503,
        "error_code": "AI_PROVIDER_UNAVAILABLE",
        "message": (
            "AI service is temporarily unavailable. "
            "Please try again later."
        ),
    }

    assert customer.credits == credits_before
