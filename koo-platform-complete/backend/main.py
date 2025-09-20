"""
KOO Platform - Simplified Single-User AI-Driven Medical Knowledge Management
Main FastAPI application with comprehensive error handling and database management
Version: 2.0.0
"""

from contextlib import asynccontextmanager
from typing import Dict, Any
import logging

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

# Core imports
from core.config import settings
from core.dependencies import get_current_user, get_db, CurrentUser, check_external_services
from core.middleware import setup_middleware
from core.database import init_database, close_database, db_manager
from core.exceptions import (
    KOOPlatformException,
    koo_platform_exception_handler,
    http_exception_handler,
    sqlalchemy_exception_handler,
    general_exception_handler
)

# API route imports
from api import chapters, research, admin, ai_services, knowledge_pipeline, enhanced_research, ai_knowledge

# Enhanced logging setup
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management with proper error handling"""
    logger.info("Starting KOO Platform (Single-User Edition)")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Admin: {settings.ADMIN_NAME} ({settings.ADMIN_EMAIL})")

    # Initialize services
    try:
        # Initialize database
        await init_database()
        logger.info("Database initialized successfully")

        # Initialize AI services (placeholder)
        logger.info("AI services ready")

        # Application is ready
        logger.info("KOO Platform startup completed successfully")

        yield

    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    finally:
        # Cleanup
        logger.info("Shutting down KOO Platform")
        try:
            await close_database()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Personal AI-driven medical knowledge management platform",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

# Setup middleware
app = setup_middleware(app)

# Register exception handlers
app.add_exception_handler(KOOPlatformException, koo_platform_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(SQLAlchemyError, sqlalchemy_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# Health check endpoints with database verification
@app.get("/health")
async def health_check():
    """Basic health check with database status"""
    try:
        db_healthy = await db_manager.health_check() if db_manager.is_initialized else False

        return {
            "status": "healthy" if db_healthy else "degraded",
            "version": settings.VERSION,
            "database": "connected" if db_healthy else "disconnected",
            "environment": settings.ENVIRONMENT
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": "Health check failed",
                "version": settings.VERSION
            }
        )

@app.get("/ready")
async def readiness_check():
    """Readiness check for deployment with external service status"""
    try:
        services = await check_external_services()
        all_healthy = all(services.values())

        return {
            "status": "ready" if all_healthy else "not_ready",
            "services": services,
            "environment": settings.ENVIRONMENT,
            "version": settings.VERSION
        }
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "error": "Readiness check failed"
            }
        )

# Admin info endpoint with enhanced information
@app.get("/api/v1/admin/info")
async def admin_info(current_user: CurrentUser = Depends(get_current_user)):
    """Get admin user information and system status"""
    try:
        services = await check_external_services()

        return {
            "user": {
                "id": current_user.id,
                "username": current_user.username,
                "email": current_user.email,
                "full_name": current_user.full_name,
                "role": current_user.role
            },
            "platform": {
                "name": settings.PROJECT_NAME,
                "version": settings.VERSION,
                "environment": settings.ENVIRONMENT,
                "debug": settings.DEBUG
            },
            "services": services,
            "configuration": {
                "rate_limiting": settings.RATE_LIMIT_ENABLED,
                "database_initialized": db_manager.is_initialized
            }
        }
    except Exception as e:
        logger.error(f"Admin info error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve admin information")

# Database status endpoint
@app.get("/api/v1/admin/database/status")
async def database_status(current_user: CurrentUser = Depends(get_current_user)):
    """Get detailed database status"""
    try:
        if not db_manager.is_initialized:
            return {"status": "not_initialized", "message": "Database not initialized"}

        is_healthy = await db_manager.health_check()

        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "initialized": db_manager.is_initialized,
            "connection_test": is_healthy,
            "database_url": settings.DATABASE_URL.split('@')[1] if '@' in settings.DATABASE_URL else "hidden"
        }
    except Exception as e:
        logger.error(f"Database status check failed: {e}")
        return {
            "status": "error",
            "message": "Failed to check database status",
            "error": str(e)
        }

# Include API routers
app.include_router(
    chapters.router,
    prefix=f"{settings.API_V1_STR}/chapters",
    tags=["chapters"]
)

app.include_router(
    research.router,
    prefix=f"{settings.API_V1_STR}/research",
    tags=["research"]
)

app.include_router(
    admin.router,
    prefix=f"{settings.API_V1_STR}",
    tags=["admin"]
)

app.include_router(
    ai_services.router,
    prefix=f"{settings.API_V1_STR}",
    tags=["ai-services"]
)

app.include_router(
    knowledge_pipeline.router,
    prefix=f"{settings.API_V1_STR}",
    tags=["knowledge-pipeline"]
)

app.include_router(
    enhanced_research.router,
    prefix=f"{settings.API_V1_STR}/enhanced-research",
    tags=["enhanced-research"]
)

app.include_router(
    ai_knowledge.router,
    prefix=f"{settings.API_V1_STR}/ai-knowledge",
    tags=["ai-knowledge"]
)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system information"""
    try:
        return {
            "message": f"Welcome to {settings.PROJECT_NAME}",
            "version": settings.VERSION,
            "docs": "/docs" if settings.DEBUG else "Disabled in production",
            "admin": settings.ADMIN_NAME,
            "environment": settings.ENVIRONMENT,
            "status": "running"
        }
    except Exception as e:
        logger.error(f"Root endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Service temporarily unavailable")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )