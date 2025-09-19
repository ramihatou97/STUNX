"""
Simplified Dependencies for Single-User KOO Platform
No complex authentication - you are always the admin user
Includes proper database session management and error handling
"""

from fastapi import Depends, HTTPException, status, Header
from sqlalchemy.ext.asyncio import AsyncSession
import os
from typing import Optional, AsyncGenerator

from .config import settings
from .database import get_db_session
from .exceptions import AuthenticationError, ValidationError

# Simplified Authentication
class CurrentUser:
    """Single admin user - always you"""
    def __init__(self):
        self.id = 1
        self.username = "admin"
        self.email = settings.ADMIN_EMAIL
        self.full_name = settings.ADMIN_NAME
        self.role = "super_admin"
        self.is_active = True
        self.is_verified = True

async def get_current_user(api_key: Optional[str] = Header(None, alias="X-API-Key")) -> CurrentUser:
    """
    Get current user (always admin in single-user mode)
    Optional API key check for external access
    """
    try:
        # If API key is provided, validate it
        if api_key:
            if api_key != settings.ADMIN_API_KEY:
                raise AuthenticationError("Invalid API key")

        # Always return admin user
        return CurrentUser()
    except Exception as e:
        if isinstance(e, AuthenticationError):
            raise
        raise AuthenticationError("Authentication failed")

async def require_admin() -> CurrentUser:
    """Ensure admin access (always true in single-user mode)"""
    return CurrentUser()

async def optional_auth(api_key: Optional[str] = Header(None, alias="X-API-Key")) -> Optional[CurrentUser]:
    """Optional authentication for public endpoints"""
    try:
        return await get_current_user(api_key)
    except AuthenticationError:
        return None

# Database Dependencies
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Get database session with proper error handling
    This is the main dependency for database access
    """
    async for session in get_db_session():
        yield session

# Rate limiting helper (simple in-memory)
from collections import defaultdict
from datetime import datetime, timedelta
import asyncio

class SimpleRateLimiter:
    def __init__(self):
        self.requests = defaultdict(list)
        self.max_requests = settings.RATE_LIMIT_PER_MINUTE
        self.time_window = timedelta(minutes=1)

    async def check_rate_limit(self, client_ip: str) -> bool:
        """Check if client is within rate limits"""
        now = datetime.now()

        # Clean old requests
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if now - req_time < self.time_window
        ]

        # Check current count
        if len(self.requests[client_ip]) >= self.max_requests:
            return False

        # Add current request
        self.requests[client_ip].append(now)
        return True

rate_limiter = SimpleRateLimiter()

async def check_rate_limit(client_ip: str = "127.0.0.1"):
    """Simple rate limiting with proper error handling"""
    if not settings.RATE_LIMIT_ENABLED:
        return True

    try:
        if not await rate_limiter.check_rate_limit(client_ip):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later."
            )
        return True
    except HTTPException:
        raise
    except Exception as e:
        # Log error but don't block request
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Rate limiting error: {e}")
        return True

# Input validation dependencies
async def validate_pagination(
    skip: int = 0,
    limit: int = 50
) -> dict:
    """Validate pagination parameters"""
    try:
        if skip < 0:
            raise ValidationError("Skip parameter must be non-negative", "skip")

        if limit <= 0 or limit > 100:
            raise ValidationError("Limit must be between 1 and 100", "limit")

        return {"skip": skip, "limit": limit}
    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(f"Pagination validation failed: {str(e)}")

# Request context for logging
class RequestContext:
    """Request context for logging and error tracking"""

    def __init__(self, request_id: str, user_id: int, path: str, method: str):
        self.request_id = request_id
        self.user_id = user_id
        self.path = path
        self.method = method
        self.start_time = datetime.utcnow()

async def get_request_context(
    request,
    current_user: CurrentUser = Depends(get_current_user)
) -> RequestContext:
    """Get request context for logging"""
    import uuid

    request_id = str(uuid.uuid4())
    return RequestContext(
        request_id=request_id,
        user_id=current_user.id,
        path=request.url.path,
        method=request.method
    )

# Health check dependencies
async def check_database_health() -> bool:
    """Check database health for health endpoints"""
    try:
        from .database import db_manager
        return await db_manager.health_check()
    except Exception:
        return False

async def check_external_services() -> dict:
    """Check external service health"""
    services = {
        "database": await check_database_health(),
        "redis": True,  # Simplified - would check Redis if configured
    }

    # Check AI services if keys are configured
    if settings.GEMINI_API_KEY:
        services["gemini"] = True  # Would do actual health check
    if settings.CLAUDE_API_KEY:
        services["claude"] = True  # Would do actual health check
    if settings.PUBMED_API_KEY:
        services["pubmed"] = True  # Would do actual health check

    return services