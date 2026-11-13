import redis

from app.core.config import settings
from app.logging.logger import logger
from app.services.ner_pii_detector import detect_ner_categories
from app.services.pii_detector import detect_pii_categories

METRICS_KEY_PREFIX = "pii_metrics:"
_TOTAL_KEY = f"{METRICS_KEY_PREFIX}total_requests"
_WITH_PII_KEY = f"{METRICS_KEY_PREFIX}requests_with_pii"
_CATEGORY_KEY_PREFIX = f"{METRICS_KEY_PREFIX}category:"

# Same synchronous-client precedent as rate_limiter.py/
# response_cache_service.py: Redis calls are fast/local, so calling
# this directly (no await) from async request handlers is the accepted
# tradeoff over an async client whose pooled connections would
# otherwise be tied to whichever event loop first used them.
_metrics_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)


def record_pii_visibility(text: str) -> None:
    """
    Records whether `text` contains PII, by category, in a cumulative
    counter - not a live operational signal, but the raw numbers for a
    data-handling/DPA conversation with the AI provider ("what fraction
    of real traffic actually carries personal data"). Purely additive
    bookkeeping: never raises, never affects whether a request
    succeeds. Fails open on a Redis error, matching every other
    secondary/non-critical Redis use in this app.

    Combines the fast regex/library categories (pii_detector.py) with
    the NER-based ones (names/addresses, ner_pii_detector.py) - the
    latter meaningfully slower (a model inference per call, not
    near-instant pattern matching), which is why this whole function
    is a deliberate, isolated add-on to the request path rather than
    something folded into input validation.
    """

    categories = detect_pii_categories(text) | detect_ner_categories(text)

    try:
        pipe = _metrics_client.pipeline()
        pipe.incr(_TOTAL_KEY)

        if categories:
            pipe.incr(_WITH_PII_KEY)

            for category in categories:
                pipe.incr(f"{_CATEGORY_KEY_PREFIX}{category}")

        pipe.execute()
    except Exception:
        logger.warning(
            "Failed to record PII visibility metrics", exc_info=True
        )
        return

    if categories:
        logger.info(
            "PII detected in request text",
            extra={"categories": sorted(categories)},
        )


def get_pii_metrics() -> dict:
    """
    Returns the current cumulative PII-visibility metrics: total
    tracked requests, how many contained PII, the percentage, and a
    per-category breakdown. Fails open to all-zero on a Redis error -
    this is a reporting endpoint, not something worth breaking over.
    """

    try:
        total = int(_metrics_client.get(_TOTAL_KEY) or 0)
        with_pii = int(_metrics_client.get(_WITH_PII_KEY) or 0)

        by_category = {
            key[len(_CATEGORY_KEY_PREFIX):]: int(_metrics_client.get(key) or 0)
            for key in _metrics_client.scan_iter(match=f"{_CATEGORY_KEY_PREFIX}*")
        }
    except Exception:
        logger.warning(
            "Failed to read PII visibility metrics", exc_info=True
        )
        return {
            "total_requests": 0,
            "requests_with_pii": 0,
            "pii_percentage": 0.0,
            "by_category": {},
        }

    percentage = round((with_pii / total * 100), 2) if total else 0.0

    return {
        "total_requests": total,
        "requests_with_pii": with_pii,
        "pii_percentage": percentage,
        "by_category": by_category,
    }


def clear_pii_metrics() -> None:
    """Test-only helper: resets every PII-visibility counter."""

    for key in _metrics_client.scan_iter(match=f"{METRICS_KEY_PREFIX}*"):
        _metrics_client.delete(key)


def close_pii_metrics_client() -> None:
    try:
        _metrics_client.close()
    except Exception:
        logger.warning(
            "Error closing PII metrics Redis client", exc_info=True
        )
