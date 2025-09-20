"""
Enhanced Redis Caching Layer for KOO Platform
Comprehensive caching system with connection pooling, error handling, and monitoring
"""

import asyncio
import json
import time
import zlib
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, List, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict
import pickle
import hashlib

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool
from redis.exceptions import RedisError, ConnectionError, TimeoutError

from .config import settings
from .exceptions import CacheError

logger = logging.getLogger(__name__)

class CacheLevel(Enum):
    """Cache level enumeration"""
    APPLICATION = "app"
    DATABASE = "db"
    API = "api"
    SESSION = "session"

class CompressionType(Enum):
    """Compression type enumeration"""
    NONE = "none"
    ZLIB = "zlib"
    PICKLE = "pickle"

@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0
    total_size: int = 0
    avg_response_time: float = 0.0
    last_reset: datetime = field(default_factory=datetime.now)

@dataclass
class CacheConfig:
    """Cache configuration"""
    default_ttl: int = 3600
    max_connections: int = 20
    retry_on_timeout: bool = True
    socket_keepalive: bool = True
    socket_keepalive_options: Dict = field(default_factory=dict)
    health_check_interval: int = 30
    compression_threshold: int = 1024  # Compress data larger than 1KB
    max_value_size: int = 10 * 1024 * 1024  # 10MB max value size

class EnhancedRedisCache:
    """
    Enhanced Redis cache with connection pooling, error handling, 
    compression, and comprehensive monitoring
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.pool: Optional[ConnectionPool] = None
        self.redis_client: Optional[redis.Redis] = None
        
        # Metrics tracking
        self.metrics: Dict[CacheLevel, CacheMetrics] = {
            level: CacheMetrics() for level in CacheLevel
        }
        self.global_metrics = CacheMetrics()
        
        # Circuit breaker for Redis connection
        self.circuit_breaker_failures = 0
        self.circuit_breaker_last_failure = None
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_timeout = 60
        
        # Health check
        self.last_health_check = None
        self.is_healthy = False

    async def initialize(self) -> None:
        """Initialize Redis connection pool"""
        try:
            # Create connection pool
            self.pool = ConnectionPool.from_url(
                settings.REDIS_URL,
                max_connections=self.config.max_connections,
                retry_on_timeout=self.config.retry_on_timeout,
                socket_keepalive=self.config.socket_keepalive,
                socket_keepalive_options=self.config.socket_keepalive_options,
                decode_responses=False  # We handle encoding/decoding manually
            )
            
            # Create Redis client
            self.redis_client = redis.Redis(connection_pool=self.pool)
            
            # Test connection
            await self.health_check()
            
            logger.info("Redis cache initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {e}")
            raise CacheError(f"Redis initialization failed: {str(e)}")

    async def close(self) -> None:
        """Close Redis connections"""
        try:
            if self.redis_client:
                await self.redis_client.close()
            if self.pool:
                await self.pool.disconnect()
            logger.info("Redis cache connections closed")
        except Exception as e:
            logger.error(f"Error closing Redis connections: {e}")

    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker allows operations"""
        if self.circuit_breaker_failures < self.circuit_breaker_threshold:
            return True
            
        if (self.circuit_breaker_last_failure and 
            time.time() - self.circuit_breaker_last_failure > self.circuit_breaker_timeout):
            # Reset circuit breaker
            self.circuit_breaker_failures = 0
            self.circuit_breaker_last_failure = None
            logger.info("Redis circuit breaker reset")
            return True
            
        return False

    def _record_failure(self) -> None:
        """Record circuit breaker failure"""
        self.circuit_breaker_failures += 1
        self.circuit_breaker_last_failure = time.time()
        
        if self.circuit_breaker_failures >= self.circuit_breaker_threshold:
            logger.warning("Redis circuit breaker opened due to failures")

    def _generate_key(self, level: CacheLevel, key: str) -> str:
        """Generate namespaced cache key"""
        return f"koo:{level.value}:{key}"

    def _compress_data(self, data: bytes) -> tuple[bytes, CompressionType]:
        """Compress data if it exceeds threshold"""
        if len(data) < self.config.compression_threshold:
            return data, CompressionType.NONE
            
        try:
            compressed = zlib.compress(data)
            if len(compressed) < len(data):
                return compressed, CompressionType.ZLIB
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            
        return data, CompressionType.NONE

    def _decompress_data(self, data: bytes, compression_type: CompressionType) -> bytes:
        """Decompress data based on compression type"""
        if compression_type == CompressionType.ZLIB:
            return zlib.decompress(data)
        return data

    def _serialize_value(self, value: Any) -> tuple[bytes, CompressionType]:
        """Serialize and optionally compress value"""
        try:
            # Serialize to JSON first, fall back to pickle for complex objects
            try:
                serialized = json.dumps(value).encode('utf-8')
            except (TypeError, ValueError):
                serialized = pickle.dumps(value)
                
            # Check size limit
            if len(serialized) > self.config.max_value_size:
                raise CacheError(f"Value too large: {len(serialized)} bytes")
                
            return self._compress_data(serialized)
            
        except Exception as e:
            raise CacheError(f"Serialization failed: {str(e)}")

    def _deserialize_value(self, data: bytes, compression_type: CompressionType) -> Any:
        """Decompress and deserialize value"""
        try:
            decompressed = self._decompress_data(data, compression_type)
            
            # Try JSON first, fall back to pickle
            try:
                return json.loads(decompressed.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return pickle.loads(decompressed)
                
        except Exception as e:
            raise CacheError(f"Deserialization failed: {str(e)}")

    async def get(self, key: str, level: CacheLevel = CacheLevel.APPLICATION) -> Optional[Any]:
        """Get value from cache"""
        if not self._check_circuit_breaker():
            return None
            
        start_time = time.time()
        cache_key = self._generate_key(level, key)
        
        try:
            # Get value and metadata
            pipe = self.redis_client.pipeline()
            pipe.hget(cache_key, "value")
            pipe.hget(cache_key, "compression")
            pipe.hget(cache_key, "expires_at")
            
            results = await pipe.execute()
            value_data, compression_data, expires_at = results
            
            if value_data is None:
                self._record_miss(level, time.time() - start_time)
                return None
                
            # Check expiration
            if expires_at and float(expires_at) < time.time():
                await self.delete(key, level)
                self._record_miss(level, time.time() - start_time)
                return None
                
            # Deserialize value
            compression_type = CompressionType(compression_data.decode()) if compression_data else CompressionType.NONE
            value = self._deserialize_value(value_data, compression_type)
            
            self._record_hit(level, time.time() - start_time)
            return value
            
        except Exception as e:
            self._record_error(level)
            self._record_failure()
            logger.error(f"Cache get error for key {key}: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None, 
                  level: CacheLevel = CacheLevel.APPLICATION) -> bool:
        """Set value in cache"""
        if not self._check_circuit_breaker():
            return False
            
        start_time = time.time()
        cache_key = self._generate_key(level, key)
        ttl = ttl or self.config.default_ttl
        
        try:
            # Serialize and compress value
            value_data, compression_type = self._serialize_value(value)
            expires_at = time.time() + ttl
            
            # Store value with metadata
            await self.redis_client.hset(cache_key, mapping={
                "value": value_data,
                "compression": compression_type.value,
                "expires_at": expires_at,
                "created_at": time.time()
            })
            
            # Set expiration on the hash key
            await self.redis_client.expire(cache_key, ttl + 60)  # Add buffer
            
            self._record_set(level, time.time() - start_time, len(value_data))
            return True

        except Exception as e:
            self._record_error(level)
            self._record_failure()
            logger.error(f"Cache set error for key {key}: {e}")
            return False

    async def delete(self, key: str, level: CacheLevel = CacheLevel.APPLICATION) -> bool:
        """Delete value from cache"""
        if not self._check_circuit_breaker():
            return False

        cache_key = self._generate_key(level, key)

        try:
            result = await self.redis_client.delete(cache_key)
            self._record_delete(level)
            return bool(result)

        except Exception as e:
            self._record_error(level)
            self._record_failure()
            logger.error(f"Cache delete error for key {key}: {e}")
            return False

    async def exists(self, key: str, level: CacheLevel = CacheLevel.APPLICATION) -> bool:
        """Check if key exists in cache"""
        if not self._check_circuit_breaker():
            return False

        cache_key = self._generate_key(level, key)

        try:
            return bool(await self.redis_client.exists(cache_key))
        except Exception as e:
            self._record_error(level)
            logger.error(f"Cache exists error for key {key}: {e}")
            return False

    async def clear_level(self, level: CacheLevel) -> int:
        """Clear all keys for a specific cache level"""
        if not self._check_circuit_breaker():
            return 0

        pattern = f"koo:{level.value}:*"

        try:
            keys = []
            async for key in self.redis_client.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                return await self.redis_client.delete(*keys)
            return 0

        except Exception as e:
            self._record_error(level)
            logger.error(f"Cache clear error for level {level}: {e}")
            return 0

    async def clear_all(self) -> bool:
        """Clear all cache data"""
        if not self._check_circuit_breaker():
            return False

        try:
            await self.redis_client.flushdb()
            self._reset_metrics()
            return True

        except Exception as e:
            logger.error(f"Cache clear all error: {e}")
            return False

    async def get_keys(self, pattern: str, level: CacheLevel = CacheLevel.APPLICATION) -> List[str]:
        """Get keys matching pattern"""
        if not self._check_circuit_breaker():
            return []

        cache_pattern = f"koo:{level.value}:{pattern}"

        try:
            keys = []
            async for key in self.redis_client.scan_iter(match=cache_pattern):
                # Remove namespace prefix
                clean_key = key.decode().replace(f"koo:{level.value}:", "")
                keys.append(clean_key)
            return keys

        except Exception as e:
            logger.error(f"Cache get_keys error: {e}")
            return []

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on Redis connection"""
        health_status = {
            "healthy": False,
            "latency_ms": None,
            "memory_usage": None,
            "connected_clients": None,
            "error": None
        }

        try:
            start_time = time.time()

            # Test basic connectivity
            await self.redis_client.ping()

            latency = (time.time() - start_time) * 1000
            health_status["latency_ms"] = round(latency, 2)

            # Get Redis info
            info = await self.redis_client.info()
            health_status["memory_usage"] = info.get("used_memory_human")
            health_status["connected_clients"] = info.get("connected_clients")

            health_status["healthy"] = True
            self.is_healthy = True
            self.last_health_check = datetime.now()

        except Exception as e:
            health_status["error"] = str(e)
            self.is_healthy = False
            self._record_failure()

        return health_status

    def _record_hit(self, level: CacheLevel, response_time: float) -> None:
        """Record cache hit metrics"""
        self.metrics[level].hits += 1
        self.global_metrics.hits += 1
        self._update_response_time(level, response_time)

    def _record_miss(self, level: CacheLevel, response_time: float) -> None:
        """Record cache miss metrics"""
        self.metrics[level].misses += 1
        self.global_metrics.misses += 1
        self._update_response_time(level, response_time)

    def _record_set(self, level: CacheLevel, response_time: float, size: int) -> None:
        """Record cache set metrics"""
        self.metrics[level].sets += 1
        self.metrics[level].total_size += size
        self.global_metrics.sets += 1
        self.global_metrics.total_size += size
        self._update_response_time(level, response_time)

    def _record_delete(self, level: CacheLevel) -> None:
        """Record cache delete metrics"""
        self.metrics[level].deletes += 1
        self.global_metrics.deletes += 1

    def _record_error(self, level: CacheLevel) -> None:
        """Record cache error metrics"""
        self.metrics[level].errors += 1
        self.global_metrics.errors += 1

    def _update_response_time(self, level: CacheLevel, response_time: float) -> None:
        """Update average response time"""
        current_avg = self.metrics[level].avg_response_time
        total_ops = self.metrics[level].hits + self.metrics[level].misses + self.metrics[level].sets

        if total_ops > 0:
            self.metrics[level].avg_response_time = (
                (current_avg * (total_ops - 1) + response_time) / total_ops
            )

    def _reset_metrics(self) -> None:
        """Reset all metrics"""
        for level in CacheLevel:
            self.metrics[level] = CacheMetrics()
        self.global_metrics = CacheMetrics()

    def get_metrics(self, level: Optional[CacheLevel] = None) -> Dict[str, Any]:
        """Get cache metrics"""
        if level:
            metrics = self.metrics[level]
            return {
                "level": level.value,
                "hits": metrics.hits,
                "misses": metrics.misses,
                "hit_rate": metrics.hits / (metrics.hits + metrics.misses) if (metrics.hits + metrics.misses) > 0 else 0,
                "sets": metrics.sets,
                "deletes": metrics.deletes,
                "errors": metrics.errors,
                "total_size_bytes": metrics.total_size,
                "avg_response_time_ms": round(metrics.avg_response_time * 1000, 2),
                "last_reset": metrics.last_reset.isoformat()
            }
        else:
            # Return global metrics
            total_ops = self.global_metrics.hits + self.global_metrics.misses
            return {
                "global": {
                    "hits": self.global_metrics.hits,
                    "misses": self.global_metrics.misses,
                    "hit_rate": self.global_metrics.hits / total_ops if total_ops > 0 else 0,
                    "sets": self.global_metrics.sets,
                    "deletes": self.global_metrics.deletes,
                    "errors": self.global_metrics.errors,
                    "total_size_bytes": self.global_metrics.total_size,
                    "avg_response_time_ms": round(self.global_metrics.avg_response_time * 1000, 2),
                    "circuit_breaker_failures": self.circuit_breaker_failures,
                    "is_healthy": self.is_healthy,
                    "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None
                },
                "by_level": {level.value: self.get_metrics(level) for level in CacheLevel}
            }

# Cache decorators and utilities
def cache_key_generator(*args, **kwargs) -> str:
    """Generate cache key from function arguments"""
    key_parts = []

    # Add positional arguments
    for arg in args:
        if hasattr(arg, '__dict__'):
            # For objects, use class name and id
            key_parts.append(f"{arg.__class__.__name__}_{id(arg)}")
        else:
            key_parts.append(str(arg))

    # Add keyword arguments
    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}={v}")

    # Create hash for long keys
    key = ":".join(key_parts)
    if len(key) > 200:
        key = hashlib.md5(key.encode()).hexdigest()

    return key

def cached(ttl: int = 3600, level: CacheLevel = CacheLevel.APPLICATION,
          key_prefix: str = "", invalidate_on_error: bool = True):
    """
    Decorator for caching function results

    Args:
        ttl: Time to live in seconds
        level: Cache level
        key_prefix: Prefix for cache key
        invalidate_on_error: Whether to invalidate cache on function error
    """
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            # Generate cache key
            func_key = f"{func.__module__}.{func.__name__}"
            args_key = cache_key_generator(*args, **kwargs)
            cache_key = f"{key_prefix}:{func_key}:{args_key}" if key_prefix else f"{func_key}:{args_key}"

            # Try to get from cache
            cached_result = await redis_cache.get(cache_key, level)
            if cached_result is not None:
                return cached_result

            # Execute function
            try:
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)

                # Cache the result
                await redis_cache.set(cache_key, result, ttl, level)
                return result

            except Exception as e:
                if invalidate_on_error:
                    await redis_cache.delete(cache_key, level)
                raise

        return wrapper
    return decorator

# Global cache instance
redis_cache = EnhancedRedisCache()

# Convenience functions for different cache levels
async def cache_api_response(key: str, value: Any, ttl: int = 300) -> bool:
    """Cache API response with 5-minute default TTL"""
    return await redis_cache.set(key, value, ttl, CacheLevel.API)

async def get_cached_api_response(key: str) -> Optional[Any]:
    """Get cached API response"""
    return await redis_cache.get(key, CacheLevel.API)

async def cache_database_query(key: str, value: Any, ttl: int = 1800) -> bool:
    """Cache database query result with 30-minute default TTL"""
    return await redis_cache.set(key, value, ttl, CacheLevel.DATABASE)

async def get_cached_database_query(key: str) -> Optional[Any]:
    """Get cached database query result"""
    return await redis_cache.get(key, CacheLevel.DATABASE)

async def cache_application_data(key: str, value: Any, ttl: int = 3600) -> bool:
    """Cache application data with 1-hour default TTL"""
    return await redis_cache.set(key, value, ttl, CacheLevel.APPLICATION)

async def get_cached_application_data(key: str) -> Optional[Any]:
    """Get cached application data"""
    return await redis_cache.get(key, CacheLevel.APPLICATION)
