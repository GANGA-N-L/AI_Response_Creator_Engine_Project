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

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpx2").setLevel(logging.WARNING)


logger = logging.getLogger("ai_response_creator")
