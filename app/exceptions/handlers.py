from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.exceptions.base import AppException
from app.logging.logger import logger
from app.logging.request_context import REQUEST_ID_HEADER, request_id_var
from app.schemas.response import ErrorField, ErrorResponse


def register_exception_handlers(app: FastAPI) -> None:

    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_exceeded_handler(
        request: Request,
        exc: RateLimitExceeded
    ):
        logger.warning(
            "Rate limit exceeded",
            extra={"path": request.url.path, "limit": exc.detail},
        )

        response = JSONResponse(
            status_code=429,
            content=ErrorResponse(
                status_code=429,
                error_code="RATE_LIMIT_EXCEEDED",
                message=(
                    "Too many requests. Please slow down and try again "
                    "shortly."
                ),
                request_id=request_id_var.get(),
            ).model_dump(exclude_none=True)
        )

        return request.app.state.limiter._inject_headers(
            response, request.state.view_rate_limit
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError
    ):
        logger.warning(
            "Request validation failed",
            extra={"path": request.url.path, "errors": exc.errors()},
        )

        fields = [
            ErrorField(
                field=".".join(str(part) for part in error["loc"][1:])
                or str(error["loc"][-1]),
                message=error["msg"],
            )
            for error in exc.errors()
        ]

        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                status_code=422,
                error_code="VALIDATION_ERROR",
                message="Invalid request data.",
                fields=fields or None,
                request_id=request_id_var.get(),
            ).model_dump(exclude_none=True)
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request,
        exc: StarletteHTTPException
    ):
        logger.warning(
            "HTTP exception",
            extra={
                "path": request.url.path,
                "status_code": exc.status_code,
                "detail": exc.detail,
            },
        )

        try:
            error_code = HTTPStatus(exc.status_code).name
        except ValueError:
            error_code = "HTTP_ERROR"

        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                status_code=exc.status_code,
                error_code=error_code,
                message=str(exc.detail),
                request_id=request_id_var.get(),
            ).model_dump(exclude_none=True)
        )

    @app.exception_handler(AppException)
    async def app_exception_handler(
        request: Request,
        exc: AppException
    ):
        logger.error(
            "Application exception",
            extra={
                "path": request.url.path,
                "error_type": type(exc).__name__,
            },
        )

        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                status_code=exc.status_code,
                error_code=exc.error_code,
                message=exc.message,
                request_id=request_id_var.get(),
            ).model_dump(exclude_none=True)
        )

    @app.exception_handler(Exception)
    async def unexpected_exception_handler(
        request: Request,
        exc: Exception
    ):
        logger.exception(
            "Unexpected application error",
            extra={"path": request.url.path},
        )

        request_id = request_id_var.get()

        response = JSONResponse(
            status_code=500,
            content=ErrorResponse(
                status_code=500,
                error_code="INTERNAL_SERVER_ERROR",
                message=(
                    "An unexpected error occurred. "
                    "Please try again later."
                ),
                request_id=request_id,
            ).model_dump(exclude_none=True)
        )

        # This handler is Starlette's true fallback for anything with no
        # registered AppException/HTTPException/RequestValidationError
        # handler - it runs inside ServerErrorMiddleware, which wraps
        # *outside* RequestIdMiddleware, so that middleware's own
        # response.headers[...] = request_id line never executes on this
        # path. Set it directly here instead.
        if request_id:
            response.headers[REQUEST_ID_HEADER] = request_id

        return response
