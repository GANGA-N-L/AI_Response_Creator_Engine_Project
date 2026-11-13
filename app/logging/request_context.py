import logging
import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.types import ASGIApp

REQUEST_ID_HEADER = "X-Request-ID"

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


class RequestIdFilter(logging.Filter):
    """
    Attaches the current request's ID (if any) to every log record, so
    it flows through to any handler attached to the logger - including
    test tooling like pytest's caplog, not just the app's own handler.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class RequestIdMiddleware(BaseHTTPMiddleware):
    """
    Assigns each request an ID - reusing an inbound X-Request-ID header
    if present (so IDs stay consistent across a chain of services),
    otherwise generating a new one - and echoes it back on the response
    so a caller can quote it when reporting an issue.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())

        # Deliberately not reset in a finally: Starlette handles a fully
        # unhandled exception (one with no registered AppException/
        # HTTPException/RequestValidationError handler) via
        # ServerErrorMiddleware, which wraps *outside* this middleware -
        # resetting here would clear the ID before that handler ever runs,
        # so it couldn't include one in the error body or response header.
        # Safe to leave set: every request re-sets it at the top of this
        # method regardless, so nothing downstream ever reads a stale value.
        request_id_var.set(request_id)

        response = await call_next(request)

        response.headers[REQUEST_ID_HEADER] = request_id
        return response
