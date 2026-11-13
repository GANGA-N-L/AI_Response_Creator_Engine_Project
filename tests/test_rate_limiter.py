from unittest.mock import patch

from app.core.rate_limiter import _customer_limiter, check_customer_rate_limit


# If the Redis-backed storage is unreachable, the customer must still be
# allowed through rather than the request erroring out - rate limiting is
# a secondary protection, not something that should be able to take the
# API down on its own.
def test_check_customer_rate_limit_fails_open_on_storage_error():
    with patch.object(
        _customer_limiter,
        "hit",
        side_effect=ConnectionError("Redis unreachable"),
    ):
        result = check_customer_rate_limit(1)

    assert result is True
