"""
Comprehensive Logging System for KOO Platform
Structured logging with request tracking and performance monitoring
"""

import logging
import json
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional
from contextlib import contextmanager

from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse

from .config import settings

# Custom JSON Formatter
class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if present
        if hasattr(record, 'request_id'):
            log_entry["request_id"] = record.request_id

        if hasattr(record, 'user_id'):
            log_entry["user_id"] = record.user_id

        if hasattr(record, 'path'):
            log_entry["path"] = record.path

        if hasattr(record, 'method'):
            log_entry["method"] = record.method

        if hasattr(record, 'status_code'):
            log_entry["status_code"] = record.status_code

        if hasattr(record, 'duration_ms'):
            log_entry["duration_ms"] = record.duration_ms

        if hasattr(record, 'error_type'):
            log_entry["error_type"] = record.error_type

        # Add any additional extra data
        if hasattr(record, 'extra_data'):
            log_entry.update(record.extra_data)

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)

# Request Context Logger
class RequestContextLogger:
    """Logger that maintains request context"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.request_id: Optional[str] = None
        self.user_id: Optional[int] = None
        self.path: Optional[str] = None
        self.method: Optional[str] = None

    def set_context(self, request_id: str, user_id: Optional[int] = None,
                   path: Optional[str] = None, method: Optional[str] = None):
        """Set request context"""
        self.request_id = request_id
        self.user_id = user_id
        self.path = path
        self.method = method

    def _add_context(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        """Add request context to log entry"""
        if self.request_id:
            extra['request_id'] = self.request_id
        if self.user_id:
            extra['user_id'] = self.user_id
        if self.path:
            extra['path'] = self.path
        if self.method:
            extra['method'] = self.method
        return extra

    def debug(self, message: str, **kwargs):
        extra = self._add_context(kwargs)
        self.logger.debug(message, extra=extra)

    def info(self, message: str, **kwargs):
        extra = self._add_context(kwargs)
        self.logger.info(message, extra=extra)

    def warning(self, message: str, **kwargs):
        extra = self._add_context(kwargs)
        self.logger.warning(message, extra=extra)

    def error(self, message: str, **kwargs):
        extra = self._add_context(kwargs)
        self.logger.error(message, extra=extra)

    def critical(self, message: str, **kwargs):
        extra = self._add_context(kwargs)
        self.logger.critical(message, extra=extra)

# Request Logging Middleware
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging all requests and responses"""

    def __init__(self, app, logger: logging.Logger):
        super().__init__(app)
        self.logger = logger

    async def dispatch(self, request: Request, call_next) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Extract client info
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")

        # Start timing
        start_time = time.time()

        # Log request
        self.logger.info(
            f"Incoming request: {request.method} {request.url.path}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "query_params": str(request.query_params),
                "client_ip": client_ip,
                "user_agent": user_agent,
                "content_type": request.headers.get("content-type"),
                "content_length": request.headers.get("content-length"),
            }
        )

        try:
            # Process request
            response = await call_next(request)

            # Calculate duration
            duration_ms = round((time.time() - start_time) * 1000, 2)

            # Log response
            self.logger.info(
                f"Request completed: {request.method} {request.url.path} - {response.status_code}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                    "response_size": response.headers.get("content-length"),
                }
            )

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as e:
            # Calculate duration
            duration_ms = round((time.time() - start_time) * 1000, 2)

            # Log error
            self.logger.error(
                f"Request failed: {request.method} {request.url.path} - {type(e).__name__}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "duration_ms": duration_ms,
                },
                exc_info=True
            )

            # Re-raise the exception
            raise

# Performance Monitor
class PerformanceMonitor:
    """Monitor and log performance metrics"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    @contextmanager
    def monitor_operation(self, operation_name: str, **context):
        """Context manager to monitor operation performance"""
        start_time = time.time()

        try:
            yield

            duration_ms = round((time.time() - start_time) * 1000, 2)

            # Log successful operation
            self.logger.info(
                f"Operation completed: {operation_name}",
                extra={
                    "operation": operation_name,
                    "duration_ms": duration_ms,
                    "status": "success",
                    **context
                }
            )

        except Exception as e:
            duration_ms = round((time.time() - start_time) * 1000, 2)

            # Log failed operation
            self.logger.error(
                f"Operation failed: {operation_name}",
                extra={
                    "operation": operation_name,
                    "duration_ms": duration_ms,
                    "status": "failed",
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    **context
                },
                exc_info=True
            )

            raise

# Application-specific loggers
def get_logger(name: str) -> RequestContextLogger:
    """Get a context-aware logger"""
    base_logger = logging.getLogger(name)
    return RequestContextLogger(base_logger)

# Database operation logger
def log_database_operation(operation: str, table: str, duration_ms: float,
                         error: Optional[Exception] = None, **context):
    """Log database operations"""
    logger = logging.getLogger("koo.database")

    if error:
        logger.error(
            f"Database operation failed: {operation} on {table}",
            extra={
                "operation": operation,
                "table": table,
                "duration_ms": duration_ms,
                "error_type": type(error).__name__,
                "error_message": str(error),
                **context
            },
            exc_info=True
        )
    else:
        logger.info(
            f"Database operation: {operation} on {table}",
            extra={
                "operation": operation,
                "table": table,
                "duration_ms": duration_ms,
                **context
            }
        )

# API operation logger
def log_api_operation(endpoint: str, method: str, status_code: int,
                     duration_ms: float, user_id: Optional[int] = None,
                     error: Optional[Exception] = None, **context):
    """Log API operations"""
    logger = logging.getLogger("koo.api")

    if error:
        logger.error(
            f"API operation failed: {method} {endpoint}",
            extra={
                "endpoint": endpoint,
                "method": method,
                "status_code": status_code,
                "duration_ms": duration_ms,
                "user_id": user_id,
                "error_type": type(error).__name__,
                "error_message": str(error),
                **context
            },
            exc_info=True
        )
    else:
        logger.info(
            f"API operation: {method} {endpoint}",
            extra={
                "endpoint": endpoint,
                "method": method,
                "status_code": status_code,
                "duration_ms": duration_ms,
                "user_id": user_id,
                **context
            }
        )

# Setup logging configuration
def setup_logging():
    """Setup application logging configuration"""

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))

    # Remove default handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler()

    if settings.ENVIRONMENT == "production":
        # Use JSON formatter in production
        console_handler.setFormatter(JSONFormatter())
    else:
        # Use simple formatter in development
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)

    # Configure specific loggers
    loggers = [
        "koo",
        "koo.api",
        "koo.database",
        "koo.auth",
        "koo.security",
        "uvicorn.access",
        "uvicorn.error"
    ]

    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))

    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

# Initialize performance monitor
performance_monitor = PerformanceMonitor(logging.getLogger("koo.performance"))

# Usage examples:
# logger = get_logger(__name__)
# logger.info("User performed action", action="create_chapter", chapter_id=123)
#
# with performance_monitor.monitor_operation("chapter_creation", user_id=1):
#     # perform operation
#     pass