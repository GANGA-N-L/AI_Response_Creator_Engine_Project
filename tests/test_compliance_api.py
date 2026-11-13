from unittest.mock import patch

from app.models.customer import Customer
from app.services.token_usage import TokenUsage

_USAGE = TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)


async def test_pii_metrics_requires_auth(unauthenticated_client):
    response = await unauthenticated_client.get("/compliance/pii-metrics")

    assert response.status_code == 401


async def test_pii_metrics_empty_state(client):
    response = await client.get("/compliance/pii-metrics")

    assert response.status_code == 200
    assert response.json() == {
        "total_requests": 0,
        "requests_with_pii": 0,
        "pii_percentage": 0.0,
        "by_category": {},
    }


# Real requests through the generate-response endpoint are what feeds
# the aggregate - not a direct call into the metrics service
async def test_pii_metrics_reflects_real_traffic(client, customer, prompt):
    with patch(
        "app.services.ai_response_service.generate_response",
        return_value=("Thanks, reach me at §PII0§ soon.", _USAGE),
    ):
        await client.post(
            "/ai/generate-response",
            json={
                "customer_id": customer.id,
                "prompt_id": prompt.id,
                "text": "Thanks, reach me at jane@example.com soon.",
            },
        )

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

    response = await client.get("/compliance/pii-metrics")

    assert response.status_code == 200

    data = response.json()

    assert data["total_requests"] == 2
    assert data["requests_with_pii"] == 1
    assert data["pii_percentage"] == 50.0
    assert data["by_category"] == {"email": 1}


async def test_cache_metrics_requires_auth(unauthenticated_client):
    response = await unauthenticated_client.get("/compliance/cache-metrics")

    assert response.status_code == 401


async def test_cache_metrics_empty_state(client):
    response = await client.get("/compliance/cache-metrics")

    assert response.status_code == 200
    assert response.json() == {
        "hits": 0,
        "misses": 0,
        "hit_rate_percentage": 0.0,
    }


# Real requests through the generate-response endpoint are what feeds
# the aggregate - not a direct call into the cache metrics service
async def test_cache_metrics_reflects_real_traffic(client, db, prompt):
    first_customer = Customer(customer_name="Cache Metrics One")
    second_customer = Customer(customer_name="Cache Metrics Two")

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

    response = await client.get("/compliance/cache-metrics")

    assert response.status_code == 200

    data = response.json()

    assert data["hits"] == 1
    assert data["misses"] == 1
    assert data["hit_rate_percentage"] == 50.0
