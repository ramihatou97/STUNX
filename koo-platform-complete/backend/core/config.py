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
    DATABASE_POOL_TIMEOUT: int = int(os.getenv("DATABASE_POOL_TIMEOUT", "30"))
    DATABASE_POOL_RECYCLE: int = int(os.getenv("DATABASE_POOL_RECYCLE", "3600"))

    # Database Circuit Breaker
    DB_CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = int(os.getenv("DB_CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5"))
    DB_CIRCUIT_BREAKER_RECOVERY_TIMEOUT: int = int(os.getenv("DB_CIRCUIT_BREAKER_RECOVERY_TIMEOUT", "60"))
    DB_CIRCUIT_BREAKER_SUCCESS_THRESHOLD: int = int(os.getenv("DB_CIRCUIT_BREAKER_SUCCESS_THRESHOLD", "2"))

    # Database Retry Configuration
    DB_MAX_RETRIES: int = int(os.getenv("DB_MAX_RETRIES", "3"))
    DB_RETRY_BASE_DELAY: float = float(os.getenv("DB_RETRY_BASE_DELAY", "1.0"))
    DB_RETRY_MAX_DELAY: float = float(os.getenv("DB_RETRY_MAX_DELAY", "30.0"))
    DB_RETRY_BACKOFF_MULTIPLIER: float = float(os.getenv("DB_RETRY_BACKOFF_MULTIPLIER", "2.0"))

    # Redis Cache Configuration
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    REDIS_MAX_CONNECTIONS: int = int(os.getenv("REDIS_MAX_CONNECTIONS", "20"))
    REDIS_RETRY_ON_TIMEOUT: bool = os.getenv("REDIS_RETRY_ON_TIMEOUT", "true").lower() == "true"
    REDIS_SOCKET_KEEPALIVE: bool = os.getenv("REDIS_SOCKET_KEEPALIVE", "true").lower() == "true"
    REDIS_HEALTH_CHECK_INTERVAL: int = int(os.getenv("REDIS_HEALTH_CHECK_INTERVAL", "30"))

    # Cache Configuration
    CACHE_DEFAULT_TTL: int = int(os.getenv("CACHE_DEFAULT_TTL", "3600"))
    CACHE_COMPRESSION_THRESHOLD: int = int(os.getenv("CACHE_COMPRESSION_THRESHOLD", "1024"))
    CACHE_MAX_VALUE_SIZE: int = int(os.getenv("CACHE_MAX_VALUE_SIZE", "10485760"))  # 10MB

    # Celery Configuration
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")
    CELERY_TASK_TIME_LIMIT: int = int(os.getenv("CELERY_TASK_TIME_LIMIT", "1800"))  # 30 minutes
    CELERY_TASK_SOFT_TIME_LIMIT: int = int(os.getenv("CELERY_TASK_SOFT_TIME_LIMIT", "1500"))  # 25 minutes
    CELERY_WORKER_PREFETCH_MULTIPLIER: int = int(os.getenv("CELERY_WORKER_PREFETCH_MULTIPLIER", "1"))
    CELERY_TASK_ACKS_LATE: bool = os.getenv("CELERY_TASK_ACKS_LATE", "true").lower() == "true"
    CELERY_RESULT_EXPIRES: int = int(os.getenv("CELERY_RESULT_EXPIRES", "3600"))  # 1 hour

    # AI API Keys
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    CLAUDE_API_KEY: Optional[str] = os.getenv("CLAUDE_API_KEY")
    PUBMED_API_KEY: Optional[str] = os.getenv("PUBMED_API_KEY")
    PERPLEXITY_API_KEY: Optional[str] = os.getenv("PERPLEXITY_API_KEY")

    # Hybrid AI Configuration
    GEMINI_ACCESS_METHOD: str = os.getenv("GEMINI_ACCESS_METHOD", "hybrid")
    CLAUDE_ACCESS_METHOD: str = os.getenv("CLAUDE_ACCESS_METHOD", "hybrid")
    PERPLEXITY_ACCESS_METHOD: str = os.getenv("PERPLEXITY_ACCESS_METHOD", "api")

    # AI Service Error Handling
    AI_CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = int(os.getenv("AI_CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5"))
    AI_CIRCUIT_BREAKER_RECOVERY_TIMEOUT: int = int(os.getenv("AI_CIRCUIT_BREAKER_RECOVERY_TIMEOUT", "60"))
    AI_CIRCUIT_BREAKER_SUCCESS_THRESHOLD: int = int(os.getenv("AI_CIRCUIT_BREAKER_SUCCESS_THRESHOLD", "2"))

    # AI Service Retry Configuration
    AI_MAX_RETRIES: int = int(os.getenv("AI_MAX_RETRIES", "3"))
    AI_RETRY_BASE_DELAY: float = float(os.getenv("AI_RETRY_BASE_DELAY", "1.0"))
    AI_RETRY_MAX_DELAY: float = float(os.getenv("AI_RETRY_MAX_DELAY", "30.0"))
    AI_RETRY_BACKOFF_MULTIPLIER: float = float(os.getenv("AI_RETRY_BACKOFF_MULTIPLIER", "2.0"))

    # AI Service Rate Limiting
    AI_DEFAULT_REQUESTS_PER_MINUTE: int = int(os.getenv("AI_DEFAULT_REQUESTS_PER_MINUTE", "60"))
    AI_DEFAULT_REQUESTS_PER_HOUR: int = int(os.getenv("AI_DEFAULT_REQUESTS_PER_HOUR", "1000"))
    AI_DEFAULT_REQUESTS_PER_DAY: int = int(os.getenv("AI_DEFAULT_REQUESTS_PER_DAY", "10000"))
    AI_DEFAULT_DAILY_BUDGET: float = float(os.getenv("AI_DEFAULT_DAILY_BUDGET", "10.0"))

    # Health Check Configuration
    AI_HEALTH_CHECK_INTERVAL: int = int(os.getenv("AI_HEALTH_CHECK_INTERVAL", "60"))
    DB_HEALTH_CHECK_INTERVAL: int = int(os.getenv("DB_HEALTH_CHECK_INTERVAL", "30"))
    HEALTH_CHECK_TIMEOUT: int = int(os.getenv("HEALTH_CHECK_TIMEOUT", "5"))

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

    # PDF Processing Configuration
    PDF_MAX_FILE_SIZE: int = int(os.getenv("PDF_MAX_FILE_SIZE", str(100 * 1024 * 1024)))  # 100MB
    PDF_MAX_PAGES: int = int(os.getenv("PDF_MAX_PAGES", "1000"))
    PDF_MEMORY_LIMIT: int = int(os.getenv("PDF_MEMORY_LIMIT", str(512 * 1024 * 1024)))  # 512MB
    PDF_CHUNK_SIZE: int = int(os.getenv("PDF_CHUNK_SIZE", str(8 * 1024 * 1024)))  # 8MB chunks
    PDF_POOL_SIZE: int = int(os.getenv("PDF_POOL_SIZE", "5"))  # Parser pool size
    PDF_PROCESSING_TIMEOUT: int = int(os.getenv("PDF_PROCESSING_TIMEOUT", "1800"))  # 30 minutes
    PDF_PAGE_BATCH_SIZE: int = int(os.getenv("PDF_PAGE_BATCH_SIZE", "10"))  # Pages per batch
    PDF_CACHE_TTL: int = int(os.getenv("PDF_CACHE_TTL", "86400"))  # 24 hours
    PDF_CHECKPOINT_INTERVAL: int = int(os.getenv("PDF_CHECKPOINT_INTERVAL", "50"))  # Pages between checkpoints
    PDF_MEMORY_CHECK_INTERVAL: int = int(os.getenv("PDF_MEMORY_CHECK_INTERVAL", "10"))  # Pages between memory checks

    # Textbook Reference Library
    TEXTBOOKS_PATH: str = os.getenv("TEXTBOOKS_PATH", "./data/textbooks")
    TEXTBOOK_PROCESSING_BATCH_SIZE: int = int(os.getenv("TEXTBOOK_PROCESSING_BATCH_SIZE", "10"))
    TEXTBOOK_SEARCH_LIMIT: int = int(os.getenv("TEXTBOOK_SEARCH_LIMIT", "50"))
    TEXTBOOK_SUMMARY_MAX_LENGTH: int = int(os.getenv("TEXTBOOK_SUMMARY_MAX_LENGTH", "1000"))

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