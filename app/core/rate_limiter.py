import jwt
import redis
from fastapi import Request
from limits import RateLimitItemPerMinute
from limits.storage import storage_from_string
from limits.strategies import FixedWindowRateLimiter
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import settings
from app.logging.logger import logger


def get_auth_subject_or_ip(request: Request) -> str:
    """
    Rate-limit key: the authenticated JWT subject if the request carries
    a valid bearer token, falling back to the client IP otherwise.
    """

    auth_header = request.headers.get("Authorization", "")

    if auth_header.startswith("Bearer "):
        token = auth_header[len("Bearer "):]

        try:
            payload = jwt.decode(
                token,
                settings.jwt_secret_key,
                algorithms=[settings.jwt_algorithm],
            )
        except jwt.PyJWTError:
            return get_remote_address(request)

        subject = payload.get("sub")

        if subject:
            return subject

    return get_remote_address(request)


limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=settings.redis_url,
    swallow_errors=True,
)


# Per-customer_id limit for /ai/generate-response. Keyed off a field in the
# JSON body, which slowapi's Request-only, synchronous key_func can't reach
# (see slowapi/extension.py:502), so it's enforced independently of the
# global per-caller limit above rather than through the @limiter.limit
# decorator.
#
# Shared Redis storage (rather than in-process memory) so this limit is
# enforced correctly across multiple worker processes/instances, not just
# within one. If Redis is unreachable, this fails open (see
# check_customer_rate_limit below) rather than taking the API down over a
# secondary dependency.
_customer_limit_storage = storage_from_string(settings.redis_url)
_customer_limiter = FixedWindowRateLimiter(_customer_limit_storage)
CUSTOMER_RATE_LIMIT = RateLimitItemPerMinute(20)


def check_customer_rate_limit(customer_id: int) -> bool:
    """
    Records a hit for this customer and returns whether they're still
    within their rate limit (fixed-window: every call counts toward the
    window, including ones that end up rejected).

    Fails open if the storage backend (Redis) is unreachable: the
    customer is treated as within their limit rather than the request
    being rejected or erroring out.
    """

    try:
        return _customer_limiter.hit(CUSTOMER_RATE_LIMIT, str(customer_id))
    except Exception:
        logger.warning(
            "Per-customer rate limit check failed - failing open",
            extra={"customer_id": customer_id},
            exc_info=True,
        )
        return True


def reset_customer_rate_limits() -> None:
    _customer_limit_storage.reset()


# Dedicated client for the /health check, with a short fail-fast timeout
# independent of the rate limiters' own connection handling - mirrors the
# huggingface_service/ollama_service check_availability() pattern.
_health_check_client = redis.Redis.from_url(
    settings.redis_url, socket_connect_timeout=5, socket_timeout=5
)


def check_redis_availability() -> bool:
    """
    Lightweight reachability check for /health. Informational only: rate
    limiting fails open when Redis is unreachable, so the API keeps
    working either way - this just gives an operator visibility into
    limits silently not being enforced, rather than that going unnoticed.
    """

    try:
        return bool(_health_check_client.ping())
    except Exception:
        logger.warning("Redis availability check failed", exc_info=True)
        return False


def close_rate_limiter_storage() -> None:
    """
    Releases both limiters' Redis connection pools on shutdown.

    The per-customer limiter's storage is ours to close directly. The
    global slowapi Limiter only accepts a storage_uri string at
    construction and builds its storage internally with no public way to
    get a reference back - `_storage` is the only path in. Each close is
    independently defensive: if a future slowapi version renames or
    removes that attribute, this becomes a no-op (logged) rather than
    breaking shutdown.
    """

    try:
        _customer_limit_storage.storage.close()
    except Exception:
        logger.warning(
            "Error closing per-customer rate limiter's Redis client",
            exc_info=True,
        )

    try:
        limiter._storage.storage.close()
    except Exception:
        logger.warning(
            "Error closing global rate limiter's Redis client",
            exc_info=True,
        )
