import json
import logging
from datetime import datetime, timezone

from app.core.config import settings
from app.logging.request_context import RequestIdFilter

_BASE_RECORD_ATTRS = set(logging.makeLogRecord({}).__dict__.keys())


class JsonFormatter(logging.Formatter):
    """
    Renders each log record as one JSON object. Any field passed via
    extra={...} (or attached by a filter, e.g. request_id) is lifted
    into a top-level key rather than left embedded in the message text.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key not in _BASE_RECORD_ATTRS and value is not None:
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


_handler = logging.StreamHandler()
_handler.setFormatter(JsonFormatter())
_handler.addFilter(RequestIdFilter())

logging.basicConfig(level=settings.log_level, handlers=[_handler])

# Silences the HTTP stack's own verbose internals (full request/response
# bodies, TLS handshake steps, connection pooling) so LOG_LEVEL=DEBUG
# surfaces only this app's own debug output, not every library's -
# found live that "httpx2" (below) isn't a real logger namespace either
# httpx or httpcore ever uses, so it silenced nothing: httpcore's own
# loggers ("httpcore.connection"/"httpcore.http11") still flooded
# through at DEBUG once app/services/ai_response_service.py's
# violation-diagnostic logging prompted enabling LOG_LEVEL=DEBUG.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


logger = logging.getLogger("ai_response_creator")
