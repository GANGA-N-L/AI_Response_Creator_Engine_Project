from unittest.mock import patch

from app.models.customer import Customer
from app.services.token_usage import TokenUsage

_USAGE = TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)


async def test_response_history_requires_auth(unauthenticated_client, customer):
    response = await unauthenticated_client.get(
        f"/customers/{customer.id}/response-history/"
    )

    assert response.status_code == 401


async def test_response_history_empty_state(client, customer):
    response = await client.get(f"/customers/{customer.id}/response-history/")

    assert response.status_code == 200
    assert response.json() == []


# Real requests through the generate-response endpoint are what feeds the
# history - not a direct call into the service.
async def test_response_history_reflects_real_traffic(client, customer, prompt):
    with patch(
        "app.services.ai_response_service.generate_response",
        return_value=("Thanks for your help.", _USAGE),
    ):
        await client.post(
            "/ai/generate-response",
            json={
                "customer_id": customer.id,
                "prompt_id": prompt.id,
                "text": "Thnks for your help",
            },
        )

    response = await client.get(f"/customers/{customer.id}/response-history/")

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 1
    assert data[0]["prompt_id"] == prompt.id
    assert data[0]["input_text"] == "Thnks for your help"
    assert data[0]["output_text"] == "Thanks for your help."


# A response-cache hit still records history - it's a real submission a
# customer may later ask support about, same as a fresh generation.
async def test_response_history_records_cache_hits(client, db, prompt):
    first_customer = Customer(customer_name="History Cache One")
    second_customer = Customer(customer_name="History Cache Two")

    db.add_all([first_customer, second_customer])
    await db.commit()
    await db.refresh(first_customer)
    await db.refresh(second_customer)

    with patch(
        "app.services.ai_response_service.generate_response",
        return_value=("Thanks for your help.", _USAGE),
    ):
        await client.post(
            "/ai/generate-response",
            json={
                "customer_id": first_customer.id,
                "prompt_id": prompt.id,
                "text": "Thnks for your help",
            },
        )
        await client.post(
            "/ai/generate-response",
            json={
                "customer_id": second_customer.id,
                "prompt_id": prompt.id,
                "text": "Thnks for your help",
            },
        )

    response = await client.get(
        f"/customers/{second_customer.id}/response-history/"
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 1
    assert data[0]["output_text"] == "Thanks for your help."


# Scoped strictly to the customer_id in the path - a different customer's
# history is never returned, even though both are behind the same
# service-level bearer token.
async def test_response_history_does_not_leak_other_customers(client, db, prompt):
    first_customer = Customer(customer_name="Scope Test One")
    second_customer = Customer(customer_name="Scope Test Two")

    db.add_all([first_customer, second_customer])
    await db.commit()
    await db.refresh(first_customer)
    await db.refresh(second_customer)

    with patch(
        "app.services.ai_response_service.generate_response",
        return_value=("Reply for customer one.", _USAGE),
    ):
        await client.post(
            "/ai/generate-response",
            json={
                "customer_id": first_customer.id,
                "prompt_id": prompt.id,
                "text": "Text from customer one",
            },
        )

    response = await client.get(
        f"/customers/{second_customer.id}/response-history/"
    )

    assert response.status_code == 200
    assert response.json() == []
