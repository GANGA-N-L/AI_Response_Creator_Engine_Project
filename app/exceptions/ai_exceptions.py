from app.exceptions.base import AppException


class AIProviderError(AppException):
    """Raised when the AI provider fails to generate a response."""

    status_code: int = 503
    error_code: str = "AI_PROVIDER_UNAVAILABLE"

    def __init__(
        self,
        message: str = (
            "AI service is temporarily unavailable. "
            "Please try again later."
        )
    ) -> None:
        super().__init__(message)
