from unittest.mock import patch

from sqlalchemy.exc import SQLAlchemyError


# Fully healthy: database, AI provider, and Redis all reachable
async def test_health_all_ok(unauthenticated_client):
    with patch("app.main.check_availability", return_value=True):
        response = await unauthenticated_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "database": "ok",
        "ai_provider": "ok",
        "rate_limiter": "ok",
    }


# AI provider unreachable: database is fine, but the core feature is down
async def test_health_ai_provider_unreachable(unauthenticated_client):
    with patch("app.main.check_availability", return_value=False):
        response = await unauthenticated_client.get("/health")

    assert response.status_code == 503
    assert response.json() == {
        "status": "degraded",
        "database": "ok",
        "ai_provider": "unreachable",
        "rate_limiter": "ok",
    }


# Database unreachable: reported explicitly instead of bubbling up as a
# generic 500
async def test_health_database_unreachable(unauthenticated_client, db):
    with patch.object(
        db, "execute", side_effect=SQLAlchemyError("Simulated database failure")
    ), patch("app.main.check_availability", return_value=True):
        response = await unauthenticated_client.get("/health")

    assert response.status_code == 503
    assert response.json() == {
        "status": "degraded",
        "database": "unreachable",
        "ai_provider": "ok",
        "rate_limiter": "ok",
    }


# Both dependencies down at once
async def test_health_everything_unreachable(unauthenticated_client, db):
    with patch.object(
        db, "execute", side_effect=SQLAlchemyError("Simulated database failure")
    ), patch("app.main.check_availability", return_value=False):
        response = await unauthenticated_client.get("/health")

    assert response.status_code == 503
    assert response.json() == {
        "status": "degraded",
        "database": "unreachable",
        "ai_provider": "unreachable",
        "rate_limiter": "ok",
    }


# Redis unreachable: reported for visibility, but doesn't drag down the
# overall status - rate limiting fails open, so the API is still healthy
async def test_health_redis_unreachable(unauthenticated_client):
    with patch("app.main.check_availability", return_value=True), patch(
        "app.main.check_redis_availability", return_value=False
    ):
        response = await unauthenticated_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "database": "ok",
        "ai_provider": "ok",
        "rate_limiter": "degraded",
    }
