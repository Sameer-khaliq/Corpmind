"""Structured JSON logging. Every AuditLogEntry write should also emit a
line through this logger so ops/debugging can grep JSON instead of prose.
"""
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from corpmind.config import settings

_ALWAYS_EXCLUDE = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName", "processName",
    "process", "taskName",
}


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Any extra={} field a caller attaches shows up automatically —
        # no fixed allowlist to maintain as new node/agent fields get added.
        for key, value in record.__dict__.items():
            if key not in _ALWAYS_EXCLUDE:
                log_data[key] = value

        return json.dumps(log_data, default=str)


def setup_logging() -> None:
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(console_handler)

    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    root_logger.setLevel(log_level)


setup_logging()
