from unittest.mock import patch

from sqlalchemy.exc import SQLAlchemyError


# Fully healthy: database and AI provider both reachable
def test_health_all_ok(unauthenticated_client):
    with patch("app.main.check_availability", return_value=True):
        response = unauthenticated_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "database": "ok",
        "ai_provider": "ok",
    }


# AI provider unreachable: database is fine, but the core feature is down
def test_health_ai_provider_unreachable(unauthenticated_client):
    with patch("app.main.check_availability", return_value=False):
        response = unauthenticated_client.get("/health")

    assert response.status_code == 503
    assert response.json() == {
        "status": "degraded",
        "database": "ok",
        "ai_provider": "unreachable",
    }


# Database unreachable: reported explicitly instead of bubbling up as a
# generic 500
def test_health_database_unreachable(unauthenticated_client, db):
    with patch.object(
        db, "execute", side_effect=SQLAlchemyError("Simulated database failure")
    ), patch("app.main.check_availability", return_value=True):
        response = unauthenticated_client.get("/health")

    assert response.status_code == 503
    assert response.json() == {
        "status": "degraded",
        "database": "unreachable",
        "ai_provider": "ok",
    }


# Both dependencies down at once
def test_health_everything_unreachable(unauthenticated_client, db):
    with patch.object(
        db, "execute", side_effect=SQLAlchemyError("Simulated database failure")
    ), patch("app.main.check_availability", return_value=False):
        response = unauthenticated_client.get("/health")

    assert response.status_code == 503
    assert response.json() == {
        "status": "degraded",
        "database": "unreachable",
        "ai_provider": "unreachable",
    }
