"""
Comprehensive Error Handling for KOO Platform
Custom exceptions and error handlers
"""

from typing import Any, Dict, Optional, Union
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, DatabaseError
import logging
import traceback
from datetime import datetime

logger = logging.getLogger(__name__)

# Custom Exception Classes
class KOOPlatformException(Exception):
    """Base exception for KOO Platform"""
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

class ValidationError(KOOPlatformException):
    """Input validation errors"""
    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(message, status.HTTP_400_BAD_REQUEST)
        if field:
            self.details = {"field": field}

class DatabaseError(KOOPlatformException):
    """Database operation errors"""
    def __init__(self, message: str = "Database operation failed"):
        super().__init__(message, status.HTTP_500_INTERNAL_SERVER_ERROR)

class ResourceNotFoundError(KOOPlatformException):
    """Resource not found errors"""
    def __init__(self, resource: str, identifier: Union[str, int]):
        message = f"{resource} with id '{identifier}' not found"
        super().__init__(message, status.HTTP_404_NOT_FOUND)
        self.details = {"resource": resource, "identifier": str(identifier)}

class ResourceAlreadyExistsError(KOOPlatformException):
    """Resource already exists errors"""
    def __init__(self, resource: str, identifier: Union[str, int]):
        message = f"{resource} with id '{identifier}' already exists"
        super().__init__(message, status.HTTP_409_CONFLICT)
        self.details = {"resource": resource, "identifier": str(identifier)}

class ExternalServiceError(KOOPlatformException):
    """External service errors (PubMed, AI APIs, etc.)"""
    def __init__(self, service: str, message: str = "External service unavailable"):
        super().__init__(f"{service}: {message}", status.HTTP_503_SERVICE_UNAVAILABLE)
        self.details = {"service": service}

class APIKeyError(KOOPlatformException):
    """API key related errors"""
    def __init__(self, provider: str, message: str = "API key error"):
        super().__init__(f"{provider}: {message}", status.HTTP_401_UNAUTHORIZED)
        self.details = {"provider": provider}

class APIKeyValidationError(APIKeyError):
    """API key validation specific errors"""
    def __init__(self, provider: str, message: str = "API key validation failed"):
        super().__init__(provider, message)

class APIKeyConfigurationError(APIKeyError):
    """API key configuration specific errors"""
    def __init__(self, provider: str, message: str = "API key configuration error"):
        super().__init__(provider, message)

class RateLimitError(KOOPlatformException):
    """Rate limiting errors"""
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, status.HTTP_429_TOO_MANY_REQUESTS)

class AuthenticationError(KOOPlatformException):
    """Authentication errors"""
    def __init__(self, message: str = "Authentication required"):
        super().__init__(message, status.HTTP_401_UNAUTHORIZED)

# Error Response Schema
class ErrorResponse:
    """Standardized error response format"""

    @staticmethod
    def create_response(
        error: Exception,
        request: Request,
        include_traceback: bool = False
    ) -> Dict[str, Any]:
        """Create standardized error response"""

        timestamp = datetime.utcnow().isoformat()
        path = str(request.url.path)
        method = request.method

        if isinstance(error, KOOPlatformException):
            response = {
                "error": {
                    "type": error.__class__.__name__,
                    "message": error.message,
                    "status_code": error.status_code,
                    "details": error.details,
                    "timestamp": timestamp,
                    "path": path,
                    "method": method
                }
            }
        elif isinstance(error, HTTPException):
            response = {
                "error": {
                    "type": "HTTPException",
                    "message": error.detail,
                    "status_code": error.status_code,
                    "timestamp": timestamp,
                    "path": path,
                    "method": method
                }
            }
        else:
            # Generic error
            response = {
                "error": {
                    "type": error.__class__.__name__,
                    "message": "Internal server error",
                    "status_code": 500,
                    "timestamp": timestamp,
                    "path": path,
                    "method": method
                }
            }

        # Add traceback in development
        if include_traceback:
            response["error"]["traceback"] = traceback.format_exc()

        return response

# Exception Handlers
async def koo_platform_exception_handler(request: Request, exc: KOOPlatformException):
    """Handle custom KOO Platform exceptions"""
    logger.error(f"KOO Platform Exception: {exc.message}", extra={
        "path": request.url.path,
        "method": request.method,
        "status_code": exc.status_code,
        "details": exc.details
    })

    response = ErrorResponse.create_response(exc, request)
    return JSONResponse(
        status_code=exc.status_code,
        content=response
    )

async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle FastAPI HTTP exceptions"""
    logger.warning(f"HTTP Exception: {exc.detail}", extra={
        "path": request.url.path,
        "method": request.method,
        "status_code": exc.status_code
    })

    response = ErrorResponse.create_response(exc, request)
    return JSONResponse(
        status_code=exc.status_code,
        content=response
    )

async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError):
    """Handle SQLAlchemy database exceptions"""
    logger.error(f"Database Error: {str(exc)}", extra={
        "path": request.url.path,
        "method": request.method,
        "exception_type": exc.__class__.__name__
    })

    # Convert to appropriate KOO exception
    if isinstance(exc, IntegrityError):
        koo_exc = ResourceAlreadyExistsError("Resource", "unknown")
        koo_exc.message = "Resource already exists or constraint violation"
    else:
        koo_exc = DatabaseError("Database operation failed")

    response = ErrorResponse.create_response(koo_exc, request)
    return JSONResponse(
        status_code=koo_exc.status_code,
        content=response
    )

async def general_exception_handler(request: Request, exc: Exception):
    """Handle any unhandled exceptions"""
    logger.error(f"Unhandled Exception: {str(exc)}", extra={
        "path": request.url.path,
        "method": request.method,
        "exception_type": exc.__class__.__name__,
        "traceback": traceback.format_exc()
    })

    koo_exc = KOOPlatformException("Internal server error occurred")
    response = ErrorResponse.create_response(
        koo_exc,
        request,
        include_traceback=True  # Include in development
    )

    return JSONResponse(
        status_code=500,
        content=response
    )

# Validation Helpers
def validate_positive_integer(value: Any, field_name: str) -> int:
    """Validate positive integer input"""
    try:
        int_value = int(value)
        if int_value <= 0:
            raise ValidationError(f"{field_name} must be a positive integer", field_name)
        return int_value
    except (ValueError, TypeError):
        raise ValidationError(f"{field_name} must be a valid integer", field_name)

def validate_string_length(value: str, field_name: str, min_length: int = 1, max_length: int = 1000) -> str:
    """Validate string length"""
    if not isinstance(value, str):
        raise ValidationError(f"{field_name} must be a string", field_name)

    if len(value) < min_length:
        raise ValidationError(f"{field_name} must be at least {min_length} characters", field_name)

    if len(value) > max_length:
        raise ValidationError(f"{field_name} must be no more than {max_length} characters", field_name)

    return value.strip()

def validate_email(email: str) -> str:
    """Basic email validation"""
    import re
    email = email.strip().lower()
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        raise ValidationError("Invalid email format", "email")
    return email

# Error Logging Context Manager
class ErrorContext:
    """Context manager for error handling with additional context"""

    def __init__(self, operation: str, **context):
        self.operation = operation
        self.context = context

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.error(f"Error in {self.operation}: {exc_val}", extra={
                "operation": self.operation,
                "context": self.context,
                "exception_type": exc_type.__name__
            })
        return False  # Don't suppress exceptions