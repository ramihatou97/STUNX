"""
Database Session Management for KOO Platform
Enhanced async SQLAlchemy setup with advanced connection pooling,
monitoring, circuit breaker pattern, and comprehensive error handling
"""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import threading
from collections import defaultdict

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import StaticPool, QueuePool
from sqlalchemy import event, text
from sqlalchemy.exc import SQLAlchemyError, DisconnectionError, OperationalError, TimeoutError

from .config import settings
from .exceptions import DatabaseError, ErrorContext

logger = logging.getLogger(__name__)

class ConnectionState(Enum):
    """Connection pool states"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CIRCUIT_OPEN = "circuit_open"

@dataclass
class PoolMetrics:
    """Connection pool metrics"""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    checked_out_connections: int = 0
    overflow_connections: int = 0
    failed_connections: int = 0
    total_checkouts: int = 0
    total_checkins: int = 0
    total_connects: int = 0
    total_disconnects: int = 0
    average_checkout_time: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class CircuitBreakerState:
    """Circuit breaker state for database connections"""
    state: ConnectionState = ConnectionState.HEALTHY
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    next_attempt_time: Optional[datetime] = None
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    half_open_max_calls: int = 3

# Database Base Class
class Base(DeclarativeBase):
    """Base class for all database models"""
    pass

# Enhanced Database Manager with Connection Pooling and Circuit Breaker
class DatabaseManager:
    """
    Enhanced database manager with advanced connection pooling,
    monitoring, circuit breaker pattern, and comprehensive error handling
    """

    def __init__(self):
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None
        self._is_initialized = False

        # Enhanced monitoring and circuit breaker
        self._pool_metrics = PoolMetrics()
        self._circuit_breaker = CircuitBreakerState()
        self._checkout_times: Dict[int, float] = {}
        self._metrics_lock = threading.Lock()

        # Retry configuration from settings
        self._max_retries = settings.DB_MAX_RETRIES
        self._base_delay = settings.DB_RETRY_BASE_DELAY
        self._max_delay = settings.DB_RETRY_MAX_DELAY
        self._backoff_multiplier = settings.DB_RETRY_BACKOFF_MULTIPLIER

        # Health check configuration from settings
        self._health_check_interval = settings.DB_HEALTH_CHECK_INTERVAL
        self._last_health_check = None
        self._health_check_timeout = settings.HEALTH_CHECK_TIMEOUT

        # Circuit breaker configuration from settings
        self._circuit_breaker.config.failure_threshold = settings.DB_CIRCUIT_BREAKER_FAILURE_THRESHOLD
        self._circuit_breaker.config.recovery_timeout = settings.DB_CIRCUIT_BREAKER_RECOVERY_TIMEOUT
        self._circuit_breaker.config.success_threshold = settings.DB_CIRCUIT_BREAKER_SUCCESS_THRESHOLD

    async def initialize(self) -> None:
        """Initialize database connection with enhanced pooling and monitoring"""
        if self._is_initialized:
            return

        try:
            with ErrorContext("database_initialization"):
                # Enhanced connection pool configuration
                pool_kwargs = {
                    "poolclass": QueuePool,
                    "pool_size": settings.DATABASE_POOL_SIZE,
                    "max_overflow": settings.DATABASE_MAX_OVERFLOW,
                    "pool_pre_ping": True,  # Verify connections before use
                    "pool_recycle": settings.DATABASE_POOL_RECYCLE,   # Recycle connections
                    "pool_timeout": settings.DATABASE_POOL_TIMEOUT,     # Timeout for getting connection from pool
                    "pool_reset_on_return": "commit",  # Reset connections on return
                }

                # PostgreSQL specific optimizations
                connect_args = {}
                if "postgresql" in settings.DATABASE_URL:
                    connect_args = {
                        "server_settings": {
                            "application_name": "koo_platform",
                            "tcp_keepalives_idle": "600",
                            "tcp_keepalives_interval": "30",
                            "tcp_keepalives_count": "3",
                        },
                        "command_timeout": 60,
                    }

                # Create async engine with enhanced configuration
                self._engine = create_async_engine(
                    settings.DATABASE_URL,
                    echo=settings.DEBUG,
                    connect_args=connect_args,
                    **pool_kwargs
                )

                # Add enhanced connection event listeners
                self._setup_enhanced_event_listeners()

                # Create session factory
                self._session_factory = async_sessionmaker(
                    bind=self._engine,
                    class_=AsyncSession,
                    expire_on_commit=False,
                    autoflush=True,
                    autocommit=False
                )

                # Test connection and initialize circuit breaker
                await self._initialize_circuit_breaker()

                self._is_initialized = True
                logger.info("Database initialized successfully with enhanced pooling")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            self._circuit_breaker.state = ConnectionState.CIRCUIT_OPEN
            self._circuit_breaker.failure_count += 1
            self._circuit_breaker.last_failure_time = datetime.now()
            raise DatabaseError(f"Database initialization failed: {str(e)}")

    def _setup_enhanced_event_listeners(self) -> None:
        """Setup enhanced SQLAlchemy event listeners with metrics tracking"""
        if not self._engine:
            return

        @event.listens_for(self._engine.sync_engine, "connect")
        def on_connect(dbapi_connection, connection_record):
            """Handle new database connections"""
            if "sqlite" in settings.DATABASE_URL:
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

            with self._metrics_lock:
                self._pool_metrics.total_connects += 1
                self._pool_metrics.last_updated = datetime.now()

            logger.debug("New database connection established")

        @event.listens_for(self._engine.sync_engine, "checkout")
        def on_checkout(dbapi_connection, connection_record, connection_proxy):
            """Track connection checkout with timing"""
            connection_id = id(dbapi_connection)
            checkout_time = time.time()
            self._checkout_times[connection_id] = checkout_time

            with self._metrics_lock:
                self._pool_metrics.total_checkouts += 1
                self._pool_metrics.checked_out_connections += 1
                self._pool_metrics.last_updated = datetime.now()

            logger.debug(f"Database connection {connection_id} checked out")

        @event.listens_for(self._engine.sync_engine, "checkin")
        def on_checkin(dbapi_connection, connection_record):
            """Track connection checkin with timing"""
            connection_id = id(dbapi_connection)
            checkin_time = time.time()

            # Calculate checkout duration
            if connection_id in self._checkout_times:
                checkout_duration = checkin_time - self._checkout_times[connection_id]
                del self._checkout_times[connection_id]

                with self._metrics_lock:
                    # Update average checkout time
                    current_avg = self._pool_metrics.average_checkout_time
                    total_checkins = self._pool_metrics.total_checkins
                    new_avg = ((current_avg * total_checkins) + checkout_duration) / (total_checkins + 1)
                    self._pool_metrics.average_checkout_time = new_avg

            with self._metrics_lock:
                self._pool_metrics.total_checkins += 1
                self._pool_metrics.checked_out_connections = max(0, self._pool_metrics.checked_out_connections - 1)
                self._pool_metrics.last_updated = datetime.now()

            logger.debug(f"Database connection {connection_id} checked in")

        @event.listens_for(self._engine.sync_engine, "close")
        def on_close(dbapi_connection, connection_record):
            """Track connection closures"""
            with self._metrics_lock:
                self._pool_metrics.total_disconnects += 1
                self._pool_metrics.last_updated = datetime.now()

            logger.debug("Database connection closed")

        @event.listens_for(self._engine.sync_engine, "close_detached")
        def on_close_detached(dbapi_connection):
            """Track detached connection closures"""
            with self._metrics_lock:
                self._pool_metrics.total_disconnects += 1
                self._pool_metrics.last_updated = datetime.now()

            logger.debug("Detached database connection closed")

    async def _initialize_circuit_breaker(self) -> None:
        """Initialize circuit breaker with health check"""
        try:
            await self.health_check()
            self._circuit_breaker.state = ConnectionState.HEALTHY
            self._circuit_breaker.last_success_time = datetime.now()
            logger.info("Circuit breaker initialized in HEALTHY state")
        except Exception as e:
            self._circuit_breaker.state = ConnectionState.UNHEALTHY
            self._circuit_breaker.failure_count = 1
            self._circuit_breaker.last_failure_time = datetime.now()
            logger.warning(f"Circuit breaker initialized in UNHEALTHY state: {e}")

    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker allows operations"""
        now = datetime.now()

        if self._circuit_breaker.state == ConnectionState.HEALTHY:
            return True

        elif self._circuit_breaker.state == ConnectionState.CIRCUIT_OPEN:
            # Check if we should attempt recovery
            if (self._circuit_breaker.next_attempt_time and
                now >= self._circuit_breaker.next_attempt_time):
                self._circuit_breaker.state = ConnectionState.DEGRADED
                logger.info("Circuit breaker transitioning to DEGRADED state for recovery attempt")
                return True
            return False

        elif self._circuit_breaker.state == ConnectionState.DEGRADED:
            # Allow limited operations for recovery testing
            return True

        elif self._circuit_breaker.state == ConnectionState.UNHEALTHY:
            # Check if enough time has passed for recovery attempt
            if (self._circuit_breaker.last_failure_time and
                now - self._circuit_breaker.last_failure_time > timedelta(seconds=self._circuit_breaker.recovery_timeout)):
                self._circuit_breaker.state = ConnectionState.DEGRADED
                logger.info("Circuit breaker transitioning to DEGRADED state after timeout")
                return True
            return False

        return False

    def _record_success(self) -> None:
        """Record successful operation for circuit breaker"""
        self._circuit_breaker.failure_count = 0
        self._circuit_breaker.last_success_time = datetime.now()

        if self._circuit_breaker.state != ConnectionState.HEALTHY:
            self._circuit_breaker.state = ConnectionState.HEALTHY
            logger.info("Circuit breaker recovered to HEALTHY state")

    def _record_failure(self) -> None:
        """Record failed operation for circuit breaker"""
        self._circuit_breaker.failure_count += 1
        self._circuit_breaker.last_failure_time = datetime.now()

        if self._circuit_breaker.failure_count >= self._circuit_breaker.failure_threshold:
            self._circuit_breaker.state = ConnectionState.CIRCUIT_OPEN
            self._circuit_breaker.next_attempt_time = (
                datetime.now() + timedelta(seconds=self._circuit_breaker.recovery_timeout)
            )
            logger.error(f"Circuit breaker OPENED after {self._circuit_breaker.failure_count} failures")
        else:
            self._circuit_breaker.state = ConnectionState.UNHEALTHY
            logger.warning(f"Circuit breaker in UNHEALTHY state, failure count: {self._circuit_breaker.failure_count}")

    async def get_session(self) -> AsyncSession:
        """Get a new database session with circuit breaker protection"""
        if not self._is_initialized:
            raise DatabaseError("Database not initialized")

        if not self._session_factory:
            raise DatabaseError("Session factory not available")

        # Check circuit breaker
        if not self._check_circuit_breaker():
            raise DatabaseError("Database circuit breaker is open - service temporarily unavailable")

        try:
            session = self._session_factory()
            self._record_success()
            return session
        except Exception as e:
            self._record_failure()
            with self._metrics_lock:
                self._pool_metrics.failed_connections += 1
            logger.error(f"Failed to create database session: {e}")
            raise DatabaseError(f"Failed to create database session: {str(e)}")

    @asynccontextmanager
    async def session_scope(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Enhanced transactional scope with circuit breaker protection and retry logic.
        Automatically commits on success, rolls back on error.
        """
        if not self._session_factory:
            raise DatabaseError("Database not initialized")

        # Check circuit breaker
        if not self._check_circuit_breaker():
            raise DatabaseError("Database circuit breaker is open - service temporarily unavailable")

        session = None
        try:
            session = self._session_factory()
            yield session
            await session.commit()
            self._record_success()
        except Exception as e:
            if session:
                await session.rollback()
            self._record_failure()
            with self._metrics_lock:
                self._pool_metrics.failed_connections += 1
            logger.error(f"Database session error: {e}")
            raise DatabaseError(f"Database operation failed: {str(e)}")
        finally:
            if session:
                await session.close()

    async def session_scope_with_retry(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Session scope with automatic retry logic for transient failures
        """
        last_exception = None

        for attempt in range(self._max_retries + 1):
            try:
                async with self.session_scope() as session:
                    yield session
                    return  # Success, exit retry loop
            except DatabaseError as e:
                last_exception = e

                # Don't retry if circuit breaker is open
                if "circuit breaker is open" in str(e):
                    raise e

                if attempt < self._max_retries:
                    delay = min(
                        self._base_delay * (self._backoff_multiplier ** attempt),
                        self._max_delay
                    )
                    logger.warning(f"Database operation failed (attempt {attempt + 1}/{self._max_retries + 1}), retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Database operation failed after {self._max_retries + 1} attempts")
                    raise e

        # This should never be reached, but just in case
        if last_exception:
            raise last_exception

    async def health_check(self, timeout: Optional[float] = None) -> bool:
        """Enhanced database health check with timeout and detailed diagnostics"""
        if not self._engine:
            logger.error("Health check failed: Engine not initialized")
            return False

        timeout = timeout or self._health_check_timeout

        try:
            # Use asyncio.wait_for to enforce timeout
            async with asyncio.timeout(timeout):
                async with self._engine.begin() as conn:
                    # Test basic connectivity
                    result = await conn.execute(text("SELECT 1"))
                    if result.scalar() != 1:
                        logger.error("Health check failed: Unexpected result from SELECT 1")
                        return False

                    # Test current timestamp (ensures database is responsive)
                    if "postgresql" in settings.DATABASE_URL:
                        await conn.execute(text("SELECT NOW()"))
                    elif "sqlite" in settings.DATABASE_URL:
                        await conn.execute(text("SELECT datetime('now')"))

                    self._last_health_check = datetime.now()
                    logger.debug("Database health check passed")
                    return True

        except asyncio.TimeoutError:
            logger.error(f"Database health check timed out after {timeout}s")
            return False
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    def get_pool_metrics(self) -> Dict[str, Any]:
        """Get current connection pool metrics"""
        with self._metrics_lock:
            # Update current pool status if engine is available
            if self._engine and hasattr(self._engine.pool, 'size'):
                pool = self._engine.pool
                self._pool_metrics.total_connections = pool.size()
                self._pool_metrics.active_connections = pool.checkedout()
                self._pool_metrics.idle_connections = pool.checkedin()
                self._pool_metrics.overflow_connections = pool.overflow()

            return {
                "pool_status": {
                    "total_connections": self._pool_metrics.total_connections,
                    "active_connections": self._pool_metrics.active_connections,
                    "idle_connections": self._pool_metrics.idle_connections,
                    "checked_out_connections": self._pool_metrics.checked_out_connections,
                    "overflow_connections": self._pool_metrics.overflow_connections,
                    "failed_connections": self._pool_metrics.failed_connections,
                },
                "usage_statistics": {
                    "total_checkouts": self._pool_metrics.total_checkouts,
                    "total_checkins": self._pool_metrics.total_checkins,
                    "total_connects": self._pool_metrics.total_connects,
                    "total_disconnects": self._pool_metrics.total_disconnects,
                    "average_checkout_time": round(self._pool_metrics.average_checkout_time, 4),
                },
                "circuit_breaker": {
                    "state": self._circuit_breaker.state.value,
                    "failure_count": self._circuit_breaker.failure_count,
                    "last_failure_time": self._circuit_breaker.last_failure_time.isoformat() if self._circuit_breaker.last_failure_time else None,
                    "last_success_time": self._circuit_breaker.last_success_time.isoformat() if self._circuit_breaker.last_success_time else None,
                    "next_attempt_time": self._circuit_breaker.next_attempt_time.isoformat() if self._circuit_breaker.next_attempt_time else None,
                },
                "configuration": {
                    "pool_size": settings.DATABASE_POOL_SIZE,
                    "max_overflow": settings.DATABASE_MAX_OVERFLOW,
                    "failure_threshold": self._circuit_breaker.failure_threshold,
                    "recovery_timeout": self._circuit_breaker.recovery_timeout,
                },
                "last_updated": self._pool_metrics.last_updated.isoformat(),
                "last_health_check": self._last_health_check.isoformat() if self._last_health_check else None,
            }

    async def get_detailed_status(self) -> Dict[str, Any]:
        """Get comprehensive database status including health check"""
        metrics = self.get_pool_metrics()

        # Perform health check if it's been a while
        health_ok = True
        if (not self._last_health_check or
            datetime.now() - self._last_health_check > timedelta(seconds=self._health_check_interval)):
            health_ok = await self.health_check()

        metrics["health_status"] = {
            "healthy": health_ok,
            "initialized": self._is_initialized,
            "last_check": self._last_health_check.isoformat() if self._last_health_check else None,
        }

        return metrics

    async def close(self) -> None:
        """Close database connections and cleanup resources"""
        try:
            if self._engine:
                # Close all connections gracefully
                await self._engine.dispose()
                logger.info("Database connections closed")

            # Reset state
            self._is_initialized = False
            self._engine = None
            self._session_factory = None

            # Clear metrics
            with self._metrics_lock:
                self._pool_metrics = PoolMetrics()
                self._checkout_times.clear()

            # Reset circuit breaker
            self._circuit_breaker = CircuitBreakerState()

        except Exception as e:
            logger.error(f"Error during database cleanup: {e}")

    @property
    def is_initialized(self) -> bool:
        """Check if database is initialized"""
        return self._is_initialized

    @property
    def circuit_breaker_state(self) -> ConnectionState:
        """Get current circuit breaker state"""
        return self._circuit_breaker.state

    def reset_circuit_breaker(self) -> None:
        """Manually reset circuit breaker to healthy state"""
        self._circuit_breaker = CircuitBreakerState()
        self._circuit_breaker.state = ConnectionState.HEALTHY
        self._circuit_breaker.last_success_time = datetime.now()
        logger.info("Circuit breaker manually reset to HEALTHY state")

# Global database manager instance
db_manager = DatabaseManager()

# Enhanced Dependencies for FastAPI
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Enhanced FastAPI dependency to get database session with circuit breaker protection
    Ensures proper session cleanup and error handling
    """
    if not db_manager.is_initialized:
        await db_manager.initialize()

    # Check circuit breaker before creating session
    if not db_manager._check_circuit_breaker():
        raise DatabaseError("Database service temporarily unavailable")

    session = None
    try:
        session = await db_manager.get_session()
        yield session
    except SQLAlchemyError as e:
        if session:
            await session.rollback()
        db_manager._record_failure()
        logger.error(f"Database session error in dependency: {e}")
        raise DatabaseError(f"Database operation failed: {str(e)}")
    except Exception as e:
        if session:
            await session.rollback()
        db_manager._record_failure()
        logger.error(f"Unexpected error in database session: {e}")
        raise
    finally:
        if session:
            await session.close()

async def get_db_session_with_retry() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency with automatic retry logic for transient failures
    """
    if not db_manager.is_initialized:
        await db_manager.initialize()

    last_exception = None

    for attempt in range(db_manager._max_retries + 1):
        try:
            async with db_manager.session_scope() as session:
                yield session
                return  # Success, exit retry loop
        except DatabaseError as e:
            last_exception = e

            # Don't retry if circuit breaker is open
            if "circuit breaker is open" in str(e) or "temporarily unavailable" in str(e):
                raise e

            if attempt < db_manager._max_retries:
                delay = min(
                    db_manager._base_delay * (db_manager._backoff_multiplier ** attempt),
                    db_manager._max_delay
                )
                logger.warning(f"Database dependency failed (attempt {attempt + 1}/{db_manager._max_retries + 1}), retrying in {delay}s: {e}")
                await asyncio.sleep(delay)
            else:
                logger.error(f"Database dependency failed after {db_manager._max_retries + 1} attempts")
                raise e

    # This should never be reached, but just in case
    if last_exception:
        raise last_exception

# Enhanced Transactional Decorators
def transactional(func):
    """
    Enhanced decorator for transactional database operations with circuit breaker protection
    Automatically handles session management and error recovery
    """
    async def wrapper(*args, **kwargs):
        async with db_manager.session_scope() as session:
            # Inject session if not provided
            if 'session' not in kwargs:
                kwargs['session'] = session
            return await func(*args, **kwargs)
    return wrapper

def transactional_with_retry(max_retries: Optional[int] = None):
    """
    Decorator for transactional operations with automatic retry logic
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            retries = max_retries or db_manager._max_retries
            last_exception = None

            for attempt in range(retries + 1):
                try:
                    async with db_manager.session_scope() as session:
                        # Inject session if not provided
                        if 'session' not in kwargs:
                            kwargs['session'] = session
                        return await func(*args, **kwargs)
                except DatabaseError as e:
                    last_exception = e

                    # Don't retry if circuit breaker is open
                    if "circuit breaker is open" in str(e):
                        raise e

                    if attempt < retries:
                        delay = min(
                            db_manager._base_delay * (db_manager._backoff_multiplier ** attempt),
                            db_manager._max_delay
                        )
                        logger.warning(f"Transactional operation failed (attempt {attempt + 1}/{retries + 1}), retrying in {delay}s: {e}")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Transactional operation failed after {retries + 1} attempts")
                        raise e

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
        return wrapper
    return decorator

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

# Enhanced Connection Retry Logic
class ConnectionRetry:
    """Enhanced database connection retry handler with circuit breaker integration"""

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 30.0, backoff_multiplier: float = 2.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier

    async def __call__(self, func: Callable, *args, **kwargs):
        """Enhanced retry decorator for database operations with exponential backoff"""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                # Check circuit breaker before attempting operation
                if hasattr(db_manager, '_check_circuit_breaker') and not db_manager._check_circuit_breaker():
                    raise DatabaseError("Database circuit breaker is open")

                result = await func(*args, **kwargs)

                # Record success if we have circuit breaker
                if hasattr(db_manager, '_record_success'):
                    db_manager._record_success()

                return result

            except (DisconnectionError, OperationalError, TimeoutError, ConnectionError) as e:
                last_exception = e

                # Record failure if we have circuit breaker
                if hasattr(db_manager, '_record_failure'):
                    db_manager._record_failure()

                if attempt == self.max_retries:
                    logger.error(f"Database operation failed after {self.max_retries + 1} attempts: {e}")
                    raise DatabaseError(f"Database operation failed after all retries: {str(e)}")

                delay = min(
                    self.base_delay * (self.backoff_multiplier ** attempt),
                    self.max_delay
                )
                logger.warning(f"Database operation attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                await asyncio.sleep(delay)

            except DatabaseError as e:
                # Don't retry if circuit breaker is open
                if "circuit breaker is open" in str(e):
                    raise e
                last_exception = e

                if attempt == self.max_retries:
                    logger.error(f"Database operation failed after {self.max_retries + 1} attempts: {e}")
                    raise e

                delay = min(
                    self.base_delay * (self.backoff_multiplier ** attempt),
                    self.max_delay
                )
                logger.warning(f"Database operation attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                await asyncio.sleep(delay)

        # This should never be reached, but just in case
        if last_exception:
            raise DatabaseError(f"Database operation failed: {str(last_exception)}")

# Convenience retry instance
db_retry = ConnectionRetry()

# Enhanced Database Lifecycle Management
async def init_database():
    """Enhanced database initialization with comprehensive error handling"""
    try:
        await db_manager.initialize()

        # Log initial status
        status = await db_manager.get_detailed_status()
        logger.info(f"Database initialization completed successfully")
        logger.info(f"Circuit breaker state: {status['circuit_breaker']['state']}")
        logger.info(f"Pool configuration: {status['configuration']['pool_size']} connections, {status['configuration']['max_overflow']} overflow")

    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        # Ensure circuit breaker is in proper state
        if hasattr(db_manager, '_circuit_breaker'):
            db_manager._circuit_breaker.state = ConnectionState.CIRCUIT_OPEN
        raise

async def close_database():
    """Enhanced database cleanup with metrics logging"""
    try:
        # Log final metrics before closing
        if db_manager.is_initialized:
            final_metrics = db_manager.get_pool_metrics()
            logger.info(f"Final database metrics - Total checkouts: {final_metrics['usage_statistics']['total_checkouts']}, "
                       f"Failed connections: {final_metrics['pool_status']['failed_connections']}, "
                       f"Average checkout time: {final_metrics['usage_statistics']['average_checkout_time']}s")

        await db_manager.close()
        logger.info("Database cleanup completed successfully")

    except Exception as e:
        logger.error(f"Database cleanup failed: {e}")

# Health Check Utilities
async def check_database_health() -> Dict[str, Any]:
    """Comprehensive database health check"""
    if not db_manager.is_initialized:
        return {
            "healthy": False,
            "error": "Database not initialized",
            "timestamp": datetime.now().isoformat()
        }

    try:
        health_ok = await db_manager.health_check()
        status = await db_manager.get_detailed_status()

        return {
            "healthy": health_ok,
            "circuit_breaker_state": status['circuit_breaker']['state'],
            "pool_status": status['pool_status'],
            "last_health_check": status['last_health_check'],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "healthy": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }