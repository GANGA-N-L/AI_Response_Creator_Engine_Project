import hashlib

import redis

from app.core.config import settings
from app.logging.logger import logger

CACHE_KEY_PREFIX = "response_cache:"

# Cumulative hit/miss counters, kept separate from CACHE_KEY_PREFIX so
# clear_response_cache's scan (test-only, clears cache entries) doesn't
# also wipe them - see clear_cache_metrics for the metrics-only reset.
_METRICS_KEY_PREFIX = "response_cache_metrics:"
_HITS_KEY = f"{_METRICS_KEY_PREFIX}hits"
_MISSES_KEY = f"{_METRICS_KEY_PREFIX}misses"

# Synchronous client, matching rate_limiter.py's precedent: Redis calls are
# fast/local, so calling this directly (no await) from async request
# handlers is the accepted tradeoff here rather than an async client, whose
# pooled connections would otherwise be tied to whichever event loop first
# used them.
_cache_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)


def build_cache_key(prompt_text: str, text: str) -> str:
    """
    Fingerprint of the (prompt, input text) pair a cached response was
    generated for. Keyed on the prompt's actual text rather than its id,
    so editing a prompt naturally invalidates old cache entries instead
    of serving output generated under the previous wording.
    """

    payload = f"{prompt_text}\x00{text}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    return f"{CACHE_KEY_PREFIX}{digest}"


def get_cached_answer(
    prompt_text: str, text: str, customer_id: int | None = None
) -> str | None:
    """
    Fails open: a Redis error is treated as a cache miss rather than
    surfaced to the caller, so an unreachable cache just means "call the
    AI provider every time" instead of breaking the request.

    Logs the outcome at DEBUG and records it in the cumulative hit/miss
    counters (get_cache_metrics) - a read failure is neither, so it's
    excluded from both to keep the ratio meaningful. customer_id is
    accepted only for the log line's context, same convention as
    unmask_and_verify/validate_ai_content.
    """

    key = build_cache_key(prompt_text, text)

    try:
        result = _cache_client.get(key)
    except Exception:
        logger.warning(
            "Response cache read failed - treating as a miss",
            exc_info=True,
            extra={"customer_id": customer_id},
        )
        return None

    hit = result is not None

    logger.debug(
        "Response cache hit" if hit else "Response cache miss",
        extra={"customer_id": customer_id},
    )

    try:
        _cache_client.incr(_HITS_KEY if hit else _MISSES_KEY)
    except Exception:
        logger.warning("Failed to record response cache metric", exc_info=True)

    return result


def store_cached_answer(prompt_text: str, text: str, answer: str) -> None:
    key = build_cache_key(prompt_text, text)

    try:
        _cache_client.set(key, answer, ex=settings.response_cache_ttl_seconds)
    except Exception:
        logger.warning("Response cache write failed", exc_info=True)


def get_cache_metrics() -> dict:
    """
    Cumulative hit/miss counts for the exact-match response cache, plus
    the derived hit-rate percentage - not a live operational signal,
    just enough visibility to confirm the cache is actually reducing
    AI-provider calls on real traffic. Fails open to all-zero on a
    Redis error, same contract as pii_metrics_service.get_pii_metrics.
    """

    try:
        hits = int(_cache_client.get(_HITS_KEY) or 0)
        misses = int(_cache_client.get(_MISSES_KEY) or 0)
    except Exception:
        logger.warning("Failed to read response cache metrics", exc_info=True)
        return {"hits": 0, "misses": 0, "hit_rate_percentage": 0.0}

    total = hits + misses
    hit_rate = round((hits / total * 100), 2) if total else 0.0

    return {"hits": hits, "misses": misses, "hit_rate_percentage": hit_rate}


def clear_response_cache() -> None:
    """Test-only helper: removes every cached response."""

    for key in _cache_client.scan_iter(match=f"{CACHE_KEY_PREFIX}*"):
        _cache_client.delete(key)


def clear_cache_metrics() -> None:
    """Test-only helper: resets the hit/miss counters."""

    for key in _cache_client.scan_iter(match=f"{_METRICS_KEY_PREFIX}*"):
        _cache_client.delete(key)


def close_response_cache() -> None:
    try:
        _cache_client.close()
    except Exception:
        logger.warning(
            "Error closing response cache Redis client", exc_info=True
        )
