class AppException(Exception):
    """Base exception for application-specific errors."""

    status_code: int = 500
    error_code: str = "APPLICATION_ERROR"

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)
