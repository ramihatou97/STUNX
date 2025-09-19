"""
Simplified Security for Single-User KOO Platform
Focus on essential security without complex user management
"""

from fastapi import Request, HTTPException, status
from fastapi.security.utils import get_authorization_scheme_param
import os
import secrets
import hashlib
from typing import Optional

class SimpleSecurity:
    """Simplified security utilities for single-user application"""

    @staticmethod
    def generate_api_key() -> str:
        """Generate a secure API key for external access"""
        return secrets.token_urlsafe(32)

    @staticmethod
    def hash_content(content: str) -> str:
        """Simple content hashing for integrity checks"""
        return hashlib.sha256(content.encode()).hexdigest()

    @staticmethod
    def validate_input_length(content: str, max_length: int = 10000) -> bool:
        """Basic input validation"""
        return len(content) <= max_length

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Basic filename sanitization"""
        import re
        # Remove path separators and special characters
        filename = re.sub(r'[^\w\-_\.]', '', filename)
        return filename[:255]  # Limit length

    @staticmethod
    def is_safe_url(url: str) -> bool:
        """Basic URL validation"""
        import re
        # Allow only HTTP/HTTPS URLs
        pattern = r'^https?://[^\s<>"]+$'
        return bool(re.match(pattern, url))

async def validate_request_size(request: Request, max_size: int = 50 * 1024 * 1024):  # 50MB
    """Validate request size to prevent large uploads"""
    content_length = request.headers.get('content-length')
    if content_length and int(content_length) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Request too large"
        )

def check_allowed_origins(origin: str) -> bool:
    """Check if origin is allowed (simple CORS)"""
    allowed_origins = [
        "http://localhost:3000",
        "http://localhost:3001",
        "https://app.koo-platform.com",
        "https://localhost"
    ]
    return origin in allowed_origins

class SecurityHeaders:
    """Security headers for responses"""

    @staticmethod
    def get_security_headers() -> dict:
        """Get basic security headers"""
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }

# Input validation helpers
def validate_chapter_content(content: str) -> bool:
    """Validate chapter content"""
    if not content or len(content.strip()) == 0:
        return False
    if len(content) > 100000:  # 100KB limit
        return False
    return True

def validate_search_query(query: str) -> bool:
    """Validate search query"""
    if not query or len(query.strip()) == 0:
        return False
    if len(query) > 500:  # Reasonable search query limit
        return False
    # Basic injection prevention
    dangerous_chars = ['<', '>', 'script', 'javascript:', 'data:']
    query_lower = query.lower()
    return not any(char in query_lower for char in dangerous_chars)

def sanitize_output(text: str) -> str:
    """Basic output sanitization"""
    import html
    return html.escape(text)

# Environment-based security settings
def get_security_config() -> dict:
    """Get security configuration based on environment"""
    is_production = os.getenv("ENVIRONMENT", "development") == "production"

    return {
        "force_https": is_production,
        "secure_cookies": is_production,
        "debug_mode": not is_production,
        "cors_origins": [
            "https://app.koo-platform.com" if is_production else "http://localhost:3000",
            "http://localhost:3001"  # Development
        ]
    }