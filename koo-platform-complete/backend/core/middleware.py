"""
Simplified Middleware for Single-User KOO Platform
Essential security without complex user management
"""

from fastapi import Request, Response, HTTPException, status
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response as StarletteResponse
import time
from typing import Callable
import logging

from .config import settings
from .security import SecurityHeaders, validate_request_size, check_allowed_origins

logger = logging.getLogger(__name__)

class SecurityMiddleware(BaseHTTPMiddleware):
    """Essential security middleware for single-user application"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        # Add security headers
        response = await call_next(request)

        # Add security headers to response
        security_headers = SecurityHeaders.get_security_headers()
        for header, value in security_headers.items():
            response.headers[header] = value

        # Add processing time header
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)

        return response

class SimpleRateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware"""

    def __init__(self, app, calls_per_minute: int = 60):
        super().__init__(app)
        self.calls_per_minute = calls_per_minute
        self.client_requests = {}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not settings.RATE_LIMIT_ENABLED:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()

        # Clean old entries
        self.client_requests = {
            ip: [req_time for req_time in requests if current_time - req_time < 60]
            for ip, requests in self.client_requests.items()
        }

        # Check rate limit
        if client_ip not in self.client_requests:
            self.client_requests[client_ip] = []

        if len(self.client_requests[client_ip]) >= self.calls_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )

        # Add current request
        self.client_requests[client_ip].append(current_time)

        return await call_next(request)

class RequestValidationMiddleware(BaseHTTPMiddleware):
    """Basic request validation middleware"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Validate request size
        try:
            await validate_request_size(request, settings.UPLOAD_MAX_SIZE)
        except HTTPException as e:
            return StarletteResponse(
                content=f"Request too large: {e.detail}",
                status_code=e.status_code
            )

        # Validate content type for POST/PUT requests
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")
            if content_type and not any(
                allowed in content_type.lower()
                for allowed in ["application/json", "multipart/form-data", "text/"]
            ):
                return StarletteResponse(
                    content="Unsupported content type",
                    status_code=415
                )

        return await call_next(request)

class LoggingMiddleware(BaseHTTPMiddleware):
    """Simple request logging middleware"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        # Log request
        logger.info(f"Request: {request.method} {request.url.path}")

        response = await call_next(request)

        # Log response
        process_time = time.time() - start_time
        logger.info(
            f"Response: {response.status_code} - {process_time:.4f}s - {request.method} {request.url.path}"
        )

        return response

def setup_middleware(app):
    """Setup all middleware for the application"""

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom middleware (order matters)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(SecurityMiddleware)
    app.add_middleware(RequestValidationMiddleware)

    if settings.RATE_LIMIT_ENABLED:
        app.add_middleware(
            SimpleRateLimitMiddleware,
            calls_per_minute=settings.RATE_LIMIT_PER_MINUTE
        )

    return app