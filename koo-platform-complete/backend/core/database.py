"""
Database Session Management for KOO Platform
Async SQLAlchemy setup with proper session handling
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
import logging

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import StaticPool
from sqlalchemy import event, text
from sqlalchemy.exc import SQLAlchemyError, DisconnectionError

from .config import settings
from .exceptions import DatabaseError, ErrorContext

logger = logging.getLogger(__name__)

# Database Base Class
class Base(DeclarativeBase):
    """Base class for all database models"""
    pass

# Database Manager
class DatabaseManager:
    """Manages database connections and sessions"""

    def __init__(self):
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None
        self._is_initialized = False

    async def initialize(self) -> None:
        """Initialize database connection"""
        if self._is_initialized:
            return

        try:
            with ErrorContext("database_initialization"):
                # Create async engine
                self._engine = create_async_engine(
                    settings.DATABASE_URL,
                    echo=settings.DEBUG,
                    pool_size=settings.DATABASE_POOL_SIZE,
                    max_overflow=settings.DATABASE_MAX_OVERFLOW,
                    pool_pre_ping=True,  # Verify connections before use
                    pool_recycle=3600,   # Recycle connections after 1 hour
                    connect_args={
                        "server_settings": {
                            "application_name": "koo_platform",
                        }
                    } if "postgresql" in settings.DATABASE_URL else {}
                )

                # Add connection event listeners
                self._setup_event_listeners()

                # Create session factory
                self._session_factory = async_sessionmaker(
                    bind=self._engine,
                    class_=AsyncSession,
                    expire_on_commit=False,
                    autoflush=True,
                    autocommit=False
                )

                # Test connection
                await self.health_check()

                self._is_initialized = True
                logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise DatabaseError(f"Database initialization failed: {str(e)}")

    def _setup_event_listeners(self) -> None:
        """Setup SQLAlchemy event listeners"""
        if not self._engine:
            return

        @event.listens_for(self._engine.sync_engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """Set SQLite pragmas if using SQLite"""
            if "sqlite" in settings.DATABASE_URL:
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        @event.listens_for(self._engine.sync_engine, "checkout")
        def receive_checkout(dbapi_connection, connection_record, connection_proxy):
            """Log connection checkout"""
            logger.debug("Database connection checked out")

        @event.listens_for(self._engine.sync_engine, "checkin")
        def receive_checkin(dbapi_connection, connection_record):
            """Log connection checkin"""
            logger.debug("Database connection checked in")

    async def get_session(self) -> AsyncSession:
        """Get a new database session"""
        if not self._is_initialized:
            raise DatabaseError("Database not initialized")

        if not self._session_factory:
            raise DatabaseError("Session factory not available")

        return self._session_factory()

    @asynccontextmanager
    async def session_scope(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Provide a transactional scope around a series of operations.
        Automatically commits on success, rolls back on error.
        """
        if not self._session_factory:
            raise DatabaseError("Database not initialized")

        session = self._session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise DatabaseError(f"Database operation failed: {str(e)}")
        finally:
            await session.close()

    async def health_check(self) -> bool:
        """Check database health"""
        if not self._engine:
            return False

        try:
            async with self._engine.begin() as conn:
                result = await conn.execute(text("SELECT 1"))
                return result.scalar() == 1
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    async def close(self) -> None:
        """Close database connections"""
        if self._engine:
            await self._engine.dispose()
            logger.info("Database connections closed")

    @property
    def is_initialized(self) -> bool:
        """Check if database is initialized"""
        return self._is_initialized

# Global database manager instance
db_manager = DatabaseManager()

# Dependency for FastAPI
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency to get database session
    Ensures proper session cleanup
    """
    if not db_manager.is_initialized:
        await db_manager.initialize()

    session = await db_manager.get_session()
    try:
        yield session
    except SQLAlchemyError as e:
        await session.rollback()
        logger.error(f"Database session error in dependency: {e}")
        raise DatabaseError(f"Database operation failed: {str(e)}")
    except Exception as e:
        await session.rollback()
        logger.error(f"Unexpected error in database session: {e}")
        raise
    finally:
        await session.close()

# Transactional decorator
def transactional(func):
    """
    Decorator for transactional database operations
    Automatically handles session management
    """
    async def wrapper(*args, **kwargs):
        async with db_manager.session_scope() as session:
            # Inject session if not provided
            if 'session' not in kwargs:
                kwargs['session'] = session
            return await func(*args, **kwargs)
    return wrapper

# Database utilities
class DatabaseUtils:
    """Utility functions for database operations"""

    @staticmethod
    async def execute_raw_sql(query: str, params: Optional[dict] = None) -> any:
        """Execute raw SQL query"""
        async with db_manager.session_scope() as session:
            result = await session.execute(text(query), params or {})
            return result

    @staticmethod
    async def check_table_exists(table_name: str) -> bool:
        """Check if table exists"""
        try:
            query = """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = :table_name
                );
            """
            result = await DatabaseUtils.execute_raw_sql(query, {"table_name": table_name})
            return result.scalar()
        except Exception:
            return False

    @staticmethod
    async def get_table_info(table_name: str) -> dict:
        """Get table information"""
        try:
            query = """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = :table_name
                ORDER BY ordinal_position;
            """
            result = await DatabaseUtils.execute_raw_sql(query, {"table_name": table_name})
            columns = []
            for row in result:
                columns.append({
                    "name": row[0],
                    "type": row[1],
                    "nullable": row[2] == "YES"
                })
            return {"table": table_name, "columns": columns}
        except Exception as e:
            logger.error(f"Error getting table info: {e}")
            return {"table": table_name, "columns": [], "error": str(e)}

# Connection retry logic
class ConnectionRetry:
    """Handle database connection retries"""

    def __init__(self, max_retries: int = 3, delay: float = 1.0):
        self.max_retries = max_retries
        self.delay = delay

    async def __call__(self, func):
        """Retry decorator for database operations"""
        for attempt in range(self.max_retries):
            try:
                return await func()
            except (DisconnectionError, ConnectionError) as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Database connection failed after {self.max_retries} attempts")
                    raise DatabaseError("Database connection failed")

                logger.warning(f"Database connection attempt {attempt + 1} failed, retrying...")
                await asyncio.sleep(self.delay * (2 ** attempt))  # Exponential backoff

        raise DatabaseError("Database connection failed after all retries")

# Initialize database on startup
async def init_database():
    """Initialize database on application startup"""
    try:
        await db_manager.initialize()
        logger.info("Database initialization completed")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

# Cleanup on shutdown
async def close_database():
    """Close database connections on application shutdown"""
    try:
        await db_manager.close()
        logger.info("Database cleanup completed")
    except Exception as e:
        logger.error(f"Database cleanup failed: {e}")