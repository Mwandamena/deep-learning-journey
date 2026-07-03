import logging
import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    """Each app passes its own name, e.g. get_logger('digit-classifier')."""
    return logging.getLogger(name)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every request + response, with duration.
    Express equivalent:
        app.use(morgan('dev'))
    """

    def __init__(self, app, logger_name: str = "api"):
        super().__init__(app)
        self.logger = get_logger(logger_name)

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000

        self.logger.info(
            "%s %s -> %s (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response