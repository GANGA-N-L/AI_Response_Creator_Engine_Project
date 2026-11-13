from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import jwt

from app.core.config import settings

from tests.conftest import assert_error_body


# Token issuance: correct credentials
async def test_issue_token_success(unauthenticated_client):
    response = await unauthenticated_client.post(
        "/auth/token",
        json={
            "client_id": settings.auth_client_id,
            "client_secret": settings.auth_client_secret,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["token_type"] == "bearer"
    assert data["expires_in"] == 3600
    assert isinstance(data["access_token"], str)
    assert len(data["access_token"]) > 0


# Token issuance: wrong credentials
async def test_issue_token_invalid_credentials(unauthenticated_client):
    response = await unauthenticated_client.post(
        "/auth/token",
        json={
            "client_id": settings.auth_client_id,
            "client_secret": "definitely-wrong",
        },
    )

    assert response.status_code == 401

    assert_error_body(response, {
        "status_code": 401,
        "error_code": "UNAUTHORIZED",
        "message": "Invalid client credentials",
    })


# Protected endpoint with no Authorization header at all
async def test_protected_endpoint_without_token(unauthenticated_client):
    response = await unauthenticated_client.get("/customers/")

    assert response.status_code == 401

    assert_error_body(response, {
        "status_code": 401,
        "error_code": "UNAUTHORIZED",
        "message": "Not authenticated",
    })


# Protected endpoint with a garbage token
async def test_protected_endpoint_with_invalid_token(unauthenticated_client):
    response = await unauthenticated_client.get(
        "/customers/",
        headers={"Authorization": "Bearer not-a-real-token"},
    )

    assert response.status_code == 401


# Protected endpoint with a well-formed but expired token
async def test_protected_endpoint_with_expired_token(unauthenticated_client):
    now = datetime.now(timezone.utc)

    expired_token = jwt.encode(
        {
            "sub": settings.auth_client_id,
            "iat": now - timedelta(minutes=120),
            "exp": now - timedelta(minutes=60),
        },
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )

    response = await unauthenticated_client.get(
        "/customers/",
        headers={"Authorization": f"Bearer {expired_token}"},
    )

    assert response.status_code == 401


# Protected endpoint with a valid token succeeds
async def test_protected_endpoint_with_valid_token(client):
    response = await client.get("/customers/")

    assert response.status_code == 200


# Root and health endpoints are intentionally public
async def test_root_and_health_do_not_require_auth(unauthenticated_client):
    root_response = await unauthenticated_client.get("/")
    assert root_response.status_code == 200

    with patch("app.main.check_availability", return_value=True):
        health_response = await unauthenticated_client.get("/health")
        assert health_response.status_code == 200


# Repeated token requests beyond the limit are rejected with 429
async def test_issue_token_rate_limited(unauthenticated_client):
    payload = {
        "client_id": settings.auth_client_id,
        "client_secret": settings.auth_client_secret,
    }

    for _ in range(5):
        response = await unauthenticated_client.post("/auth/token", json=payload)
        assert response.status_code == 200

    response = await unauthenticated_client.post("/auth/token", json=payload)

    assert response.status_code == 429
    assert response.json()["error_code"] == "RATE_LIMIT_EXCEEDED"
