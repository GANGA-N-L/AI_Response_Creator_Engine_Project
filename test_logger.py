import io
import json
import logging
import uuid

from app.core.config import settings
from app.logging.logger import JsonFormatter, _handler


def _make_record(**kwargs):
    defaults = dict(
        name="ai_response_creator",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="Test message",
        args=(),
        exc_info=None,
    )
    defaults.update(kwargs)

    return logging.LogRecord(**defaults)


def test_json_formatter_basic_fields():
    record = _make_record()

    payload = json.loads(JsonFormatter().format(record))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "ai_response_creator"
    assert payload["message"] == "Test message"
    assert "timestamp" in payload


def test_json_formatter_includes_extra_fields():
    record = _make_record()
    record.customer_id = 42
    record.amount = 3

    payload = json.loads(JsonFormatter().format(record))

    assert payload["customer_id"] == 42
    assert payload["amount"] == 3


def test_json_formatter_omits_none_extra_fields():
    record = _make_record()
    record.request_id = None

    payload = json.loads(JsonFormatter().format(record))

    assert "request_id" not in payload


def test_json_formatter_includes_exception_info():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys
        record = _make_record(exc_info=sys.exc_info())

    payload = json.loads(JsonFormatter().format(record))

    assert "ValueError" in payload["exception"]
    assert "boom" in payload["exception"]


# Caller-supplied request IDs are echoed back unchanged, so IDs stay
# consistent across a chain of services
def test_request_id_echoed_back_when_provided(unauthenticated_client):
    response = unauthenticated_client.get(
        "/health", headers={"X-Request-ID": "test-request-id-123"}
    )

    assert response.headers["X-Request-ID"] == "test-request-id-123"


# No inbound header: the server generates a fresh one
def test_request_id_generated_when_missing(unauthenticated_client):
    response = unauthenticated_client.get("/health")

    generated_id = response.headers["X-Request-ID"]

    assert uuid.UUID(generated_id)


# The request ID established by the middleware reaches the app's real
# log output (JSON, via the actual configured handler) for records
# produced during that request - exercised end to end, not just at the
# filter/formatter unit level. Swaps the handler's stream directly
# rather than relying on capsys, since the handler bound its stream
# reference at import time, before any per-test capture fixture exists.
def test_request_id_appears_in_log_output(unauthenticated_client):
    buffer = io.StringIO()
    original_stream = _handler.stream
    _handler.stream = buffer

    try:
        unauthenticated_client.post(
            "/auth/token",
            json={
                "client_id": settings.auth_client_id,
                "client_secret": "definitely-wrong",
            },
            headers={"X-Request-ID": "trace-me-456"},
        )
    finally:
        _handler.stream = original_stream

    entries = [
        json.loads(line)
        for line in buffer.getvalue().splitlines()
        if line.strip()
    ]

    assert any(entry.get("request_id") == "trace-me-456" for entry in entries)
