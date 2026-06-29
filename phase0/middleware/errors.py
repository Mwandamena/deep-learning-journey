from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from middleware.logging import logger


def _error_response(status: int, message: str, detail=None) -> JSONResponse:
    """Single shape for every error response — tweak once, applies everywhere."""
    body = {"status": "error", "message": message}
    if detail:
        body["detail"] = detail
    return JSONResponse(status_code=status, content=body)


def register_error_handlers(app: FastAPI) -> None:
    """
    Call this in App.__init__() to wire up all handlers.

    Express equivalent:
        // 404
        app.use('*', (req, res) => res.status(404).json({...}))

        // validation errors (express-validator style)
        app.use((err, req, res, next) => {
            if (err.type === 'validation') res.status(422).json({...})
        })

        // catch-all
        app.use((err, req, res, next) => res.status(500).json({...}))
    """

    @app.exception_handler(404)
    async def not_found_handler(request: Request, _exc):
        logger.warning("404 Not Found: %s %s", request.method, request.url.path)
        return _error_response(404, f"Route {request.url.path} does not exist")

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        logger.warning("422 Validation error on %s: %s", request.url.path, exc.errors())
        return _error_response(422, "Invalid request body", detail=exc.errors())

    @app.exception_handler(Exception)
    async def global_handler(request: Request, exc: Exception):
        logger.error(
            "500 Unhandled exception on %s %s: %s",
            request.method,
            request.url.path,
            str(exc),
            exc_info=True, 
        )
        return _error_response(500, "Internal server error")