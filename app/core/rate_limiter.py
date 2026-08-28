import jwt
from fastapi import Request
from limits import RateLimitItemPerMinute
from limits.storage import MemoryStorage
from limits.strategies import FixedWindowRateLimiter
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import settings


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


limiter = Limiter(key_func=get_remote_address)


# Per-customer_id limit for /ai/generate-response. Keyed off a field in the
# JSON body, which slowapi's Request-only, synchronous key_func can't reach
# (see slowapi/extension.py:502), so it's enforced independently of the
# global per-caller limit above rather than through the @limiter.limit
# decorator.
_customer_limit_storage = MemoryStorage()
_customer_limiter = FixedWindowRateLimiter(_customer_limit_storage)
CUSTOMER_RATE_LIMIT = RateLimitItemPerMinute(20)


def check_customer_rate_limit(customer_id: int) -> bool:
    """
    Records a hit for this customer and returns whether they're still
    within their rate limit (fixed-window: every call counts toward the
    window, including ones that end up rejected).
    """

    return _customer_limiter.hit(CUSTOMER_RATE_LIMIT, str(customer_id))


def reset_customer_rate_limits() -> None:
    _customer_limit_storage.reset()
