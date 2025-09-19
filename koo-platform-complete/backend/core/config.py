"""
KOO Platform Configuration Management
Simplified single-user configuration with essential security
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Simplified settings for single-user KOO Platform"""

    # Application
    PROJECT_NAME: str = "KOO Platform - Personal Edition"
    VERSION: str = "2.0.0"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")

    # Single User Configuration
    ADMIN_NAME: str = os.getenv("ADMIN_NAME", "Admin User")
    ADMIN_EMAIL: str = os.getenv("ADMIN_EMAIL", "admin@koo-platform.com")
    ADMIN_API_KEY: str = os.getenv("ADMIN_API_KEY", "koo-admin-key-change-this")

    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://koo_user:password@localhost:5432/koo_development"
    )
    DATABASE_POOL_SIZE: int = int(os.getenv("DATABASE_POOL_SIZE", "10"))
    DATABASE_MAX_OVERFLOW: int = int(os.getenv("DATABASE_MAX_OVERFLOW", "20"))

    # Cache
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # AI API Keys
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    CLAUDE_API_KEY: Optional[str] = os.getenv("CLAUDE_API_KEY")
    PUBMED_API_KEY: Optional[str] = os.getenv("PUBMED_API_KEY")
    PERPLEXITY_API_KEY: Optional[str] = os.getenv("PERPLEXITY_API_KEY")

    # Hybrid AI Configuration
    GEMINI_ACCESS_METHOD: str = os.getenv("GEMINI_ACCESS_METHOD", "hybrid")
    CLAUDE_ACCESS_METHOD: str = os.getenv("CLAUDE_ACCESS_METHOD", "hybrid")
    PERPLEXITY_ACCESS_METHOD: str = os.getenv("PERPLEXITY_ACCESS_METHOD", "api")

    # Web Interface Settings
    GEMINI_WEB_ENABLED: bool = os.getenv("GEMINI_WEB_ENABLED", "true").lower() == "true"
    CLAUDE_WEB_ENABLED: bool = os.getenv("CLAUDE_WEB_ENABLED", "true").lower() == "true"
    PERPLEXITY_WEB_ENABLED: bool = os.getenv("PERPLEXITY_WEB_ENABLED", "true").lower() == "true"

    # Cost Management
    GEMINI_DAILY_BUDGET: float = float(os.getenv("GEMINI_DAILY_BUDGET", "15.0"))
    CLAUDE_DAILY_BUDGET: float = float(os.getenv("CLAUDE_DAILY_BUDGET", "20.0"))
    PERPLEXITY_DAILY_BUDGET: float = float(os.getenv("PERPLEXITY_DAILY_BUDGET", "10.0"))

    # Browser Automation
    BROWSER_HEADLESS: bool = os.getenv("BROWSER_HEADLESS", "true").lower() == "true"
    BROWSER_TIMEOUT: int = int(os.getenv("BROWSER_TIMEOUT", "30000"))
    BROWSER_USER_DATA_DIR: str = os.getenv("BROWSER_USER_DATA_DIR", "./data/browser_sessions")

    # API Configuration
    API_V1_STR: str = "/api/v1"
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1"]

    # Simplified Security Settings
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "https://app.koo-platform.com"
    ]

    # Rate Limiting (Simple)
    RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))

    # File Upload
    UPLOAD_MAX_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_FILE_TYPES: List[str] = ["pdf", "doc", "docx", "txt", "md", "png", "jpg", "jpeg"]

    # Monitoring
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    SENTRY_DSN: Optional[str] = os.getenv("SENTRY_DSN")

    # Security Headers
    FORCE_HTTPS: bool = ENVIRONMENT == "production"
    SECURE_COOKIES: bool = ENVIRONMENT == "production"

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()