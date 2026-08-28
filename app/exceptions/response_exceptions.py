from app.exceptions.base import AppException


class InputValidationError(AppException):
    """Raised when submitted text fails InputValidator's checks."""

    status_code = 422

    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code


class CustomerNotFoundError(AppException):
    """Raised when the referenced customer does not exist."""

    status_code = 404
    error_code = "CUSTOMER_NOT_FOUND"


class PromptNotFoundError(AppException):
    """Raised when the referenced prompt does not exist."""

    status_code = 404
    error_code = "PROMPT_NOT_FOUND"


class InsufficientCreditsError(AppException):
    """Raised when a customer does not have enough credits for a request."""

    status_code = 400
    error_code = "INSUFFICIENT_CREDITS"


class RateLimitExceededError(AppException):
    """Raised when a customer exceeds their own request rate limit."""

    status_code = 429
    error_code = "RATE_LIMIT_EXCEEDED"


class IdempotencyKeyConflictError(AppException):
    """Raised when an idempotency key is reused for a different request."""

    status_code = 409
    error_code = "IDEMPOTENCY_KEY_CONFLICT"


class DatabaseError(AppException):
    """Raised when a database operation required to serve the request fails."""

    status_code = 500
    error_code = "DATABASE_ERROR"
