from app.models.idempotency_key import IdempotencyKey
from app.services.idempotency_service import (
    compute_request_hash,
    get_cached_response,
    store_response,
)


def test_compute_request_hash_is_stable():
    h1 = compute_request_hash(1, 2, "hello")
    h2 = compute_request_hash(1, 2, "hello")

    assert h1 == h2


def test_compute_request_hash_differs_for_different_input():
    h1 = compute_request_hash(1, 2, "hello")
    h2 = compute_request_hash(1, 2, "goodbye")

    assert h1 != h2


def test_store_response_success(customer, db):
    record = store_response(
        db,
        customer.id,
        "unit-test-key",
        "somehash",
        200,
        {"answer": "hi"},
    )

    assert record.customer_id == customer.id
    assert record.idempotency_key == "unit-test-key"
    assert record.response_status_code == 200
    assert record.response_body == {"answer": "hi"}

    fetched = get_cached_response(db, customer.id, "unit-test-key")
    assert fetched is not None
    assert fetched.id == record.id


def test_store_response_concurrent_duplicate_returns_existing(customer, db):
    # simulate a concurrent request that already committed a response
    # for this key, by inserting it directly before calling store_response
    existing = IdempotencyKey(
        customer_id=customer.id,
        idempotency_key="race-key",
        request_hash="hash-a",
        response_status_code=200,
        response_body={"answer": "first writer wins"},
    )
    db.add(existing)
    db.commit()
    db.refresh(existing)

    result = store_response(
        db,
        customer.id,
        "race-key",
        "hash-a",
        200,
        {"answer": "second writer"},
    )

    assert result.id == existing.id
    assert result.response_body == {"answer": "first writer wins"}
