"""Routes FastAPI optionnelles (extra ``bouture[fastapi]``). Le cœur n'importe jamais ce module."""

from .router import bouture_exception_handler, error_response, make_router, sse_line

__all__ = ["bouture_exception_handler", "error_response", "make_router", "sse_line"]
