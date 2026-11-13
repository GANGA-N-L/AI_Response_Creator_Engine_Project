from unittest.mock import patch

from app.exceptions.ai_exceptions import AIProviderError

from tests.conftest import assert_error_body


def fake_generate_response(prompt, text, customer_id):
    raise AIProviderError()


async def test_response_provider_failure(client, customer, prompt):
    with patch(
        "app.services.ai_response_service.generate_response",
        side_effect=fake_generate_response,
    ):
        response = await client.post(
            "/ai/generate-response",
            json={
                "customer_id": customer.id,
                "prompt_id": prompt.id,
                "text": "Thnks for your help",
            },
        )

    assert response.status_code == 503

    assert_error_body(response, {
        "status_code": 503,
        "error_code": "AI_PROVIDER_UNAVAILABLE",
        "message": (
            "AI service is temporarily unavailable. "
            "Please try again later."
        ),
    })
