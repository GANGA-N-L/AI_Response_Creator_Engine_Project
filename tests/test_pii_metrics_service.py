from unittest.mock import patch

from app.services.pii_metrics_service import (
    _metrics_client,
    get_pii_metrics,
    record_pii_visibility,
)


def test_record_and_get_metrics_with_no_pii():
    record_pii_visibility("Please make this sound more professional.")

    metrics = get_pii_metrics()

    assert metrics["total_requests"] == 1
    assert metrics["requests_with_pii"] == 0
    assert metrics["pii_percentage"] == 0.0
    assert metrics["by_category"] == {}


def test_record_and_get_metrics_with_pii():
    record_pii_visibility("Reach me at jane@example.com please.")
    record_pii_visibility("No PII in this one.")

    metrics = get_pii_metrics()

    assert metrics["total_requests"] == 2
    assert metrics["requests_with_pii"] == 1
    assert metrics["pii_percentage"] == 50.0
    assert metrics["by_category"] == {"email": 1}


def test_by_category_covers_formats_not_masked():
    record_pii_visibility("SSN 123-45-6789, card 4111 1111 1111 1111.")

    metrics = get_pii_metrics()

    assert metrics["by_category"]["ssn"] == 1
    assert metrics["by_category"]["credit_card"] == 1


def test_get_pii_metrics_empty_state():
    metrics = get_pii_metrics()

    assert metrics == {
        "total_requests": 0,
        "requests_with_pii": 0,
        "pii_percentage": 0.0,
        "by_category": {},
    }


# Fails open: an unreachable metrics store must not break the request
def test_record_pii_visibility_fails_open_on_redis_error():
    with patch.object(
        _metrics_client, "pipeline", side_effect=ConnectionError("Redis unreachable")
    ):
        record_pii_visibility("jane@example.com")


def test_get_pii_metrics_fails_open_on_redis_error():
    with patch.object(
        _metrics_client, "get", side_effect=ConnectionError("Redis unreachable")
    ):
        metrics = get_pii_metrics()

    assert metrics == {
        "total_requests": 0,
        "requests_with_pii": 0,
        "pii_percentage": 0.0,
        "by_category": {},
    }
