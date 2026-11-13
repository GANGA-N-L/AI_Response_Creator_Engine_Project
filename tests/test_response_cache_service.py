from unittest.mock import patch

from app.services.response_cache_service import (
    _cache_client,
    build_cache_key,
    get_cache_metrics,
    get_cached_answer,
    store_cached_answer,
)


def test_build_cache_key_is_stable():
    key_a = build_cache_key("Fix grammar.", "Thnks for your help")
    key_b = build_cache_key("Fix grammar.", "Thnks for your help")

    assert key_a == key_b


def test_build_cache_key_distinguishes_prompt_and_text():
    base = build_cache_key("Fix grammar.", "Thnks for your help")
    different_text = build_cache_key("Fix grammar.", "Something else")
    different_prompt = build_cache_key("Rewrite formally.", "Thnks for your help")

    assert base != different_text
    assert base != different_prompt


def test_get_cached_answer_returns_none_when_absent():
    result = get_cached_answer("Fix grammar.", "no such entry")

    assert result is None


def test_store_then_get_round_trip():
    store_cached_answer("Fix grammar.", "Thnks for your help", "Thanks for your help.")

    result = get_cached_answer("Fix grammar.", "Thnks for your help")

    assert result == "Thanks for your help."


# Fails open: an unreachable cache must not break the request, just behave
# as though nothing was cached.
def test_get_cached_answer_fails_open_on_redis_error():
    with patch.object(
        _cache_client, "get", side_effect=ConnectionError("Redis unreachable")
    ):
        result = get_cached_answer("Fix grammar.", "Thnks for your help")

    assert result is None


def test_store_cached_answer_fails_open_on_redis_error():
    with patch.object(
        _cache_client, "set", side_effect=ConnectionError("Redis unreachable")
    ):
        store_cached_answer("Fix grammar.", "Thnks for your help", "Thanks.")


def test_get_cache_metrics_empty_state():
    assert get_cache_metrics() == {
        "hits": 0,
        "misses": 0,
        "hit_rate_percentage": 0.0,
    }


def test_get_cached_answer_records_hit_and_miss_metrics():
    get_cached_answer("Fix grammar.", "no such entry")  # miss

    store_cached_answer("Fix grammar.", "Thnks for your help", "Thanks for your help.")
    get_cached_answer("Fix grammar.", "Thnks for your help")  # hit

    metrics = get_cache_metrics()

    assert metrics["hits"] == 1
    assert metrics["misses"] == 1
    assert metrics["hit_rate_percentage"] == 50.0


# A read failure is neither a hit nor a miss - excluded from both so the
# ratio stays meaningful, matching get_cached_answer's own docstring.
def test_get_cached_answer_does_not_record_metric_on_redis_error():
    with patch.object(
        _cache_client, "get", side_effect=ConnectionError("Redis unreachable")
    ):
        get_cached_answer("Fix grammar.", "Thnks for your help")

    assert get_cache_metrics() == {
        "hits": 0,
        "misses": 0,
        "hit_rate_percentage": 0.0,
    }


def test_get_cache_metrics_fails_open_on_redis_error():
    with patch.object(
        _cache_client, "get", side_effect=ConnectionError("Redis unreachable")
    ):
        metrics = get_cache_metrics()

    assert metrics == {"hits": 0, "misses": 0, "hit_rate_percentage": 0.0}
