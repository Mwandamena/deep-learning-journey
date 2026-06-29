import logging
import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("mnist-api")

class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every request + response, with duration.
    Express equivalent:
        app.use(morgan('dev'))
        -- or a custom:
        app.use((req, res, next) => {
            const start = Date.now()
            res.on('finish', () => {
                console.log(`${req.method} ${req.url} ${res.statusCode} ${Date.now()-start}ms`)
            })
            next()
        })
    """

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "%s %s -> %s (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response