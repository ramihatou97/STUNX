"""
Simple in-memory cache for KOO Platform
Lightweight caching without external dependencies
"""

import asyncio
import time
from typing import Any, Optional, Dict
from functools import wraps
import hashlib
import json

class SimpleCache:
    """Simple in-memory cache with TTL support"""

    def __init__(self, default_ttl: int = 3600):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl

    def _get_key(self, key: str) -> str:
        """Generate cache key with hashing for long keys"""
        if len(key) > 100:
            return hashlib.md5(key.encode()).hexdigest()
        return key

    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry is expired"""
        return time.time() > entry.get('expires_at', 0)

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        cache_key = self._get_key(key)

        if cache_key not in self._cache:
            return None

        entry = self._cache[cache_key]

        if self._is_expired(entry):
            del self._cache[cache_key]
            return None

        return entry['value']

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with TTL"""
        cache_key = self._get_key(key)
        ttl = ttl or self.default_ttl

        self._cache[cache_key] = {
            'value': value,
            'expires_at': time.time() + ttl,
            'created_at': time.time()
        }

    async def delete(self, key: str) -> None:
        """Delete value from cache"""
        cache_key = self._get_key(key)

        if cache_key in self._cache:
            del self._cache[cache_key]

    async def clear(self) -> None:
        """Clear all cache entries"""
        self._cache.clear()

    def size(self) -> int:
        """Get current cache size"""
        return len(self._cache)

    def cleanup_expired(self) -> int:
        """Remove expired entries and return count removed"""
        expired_keys = []

        for key, entry in self._cache.items():
            if self._is_expired(entry):
                expired_keys.append(key)

        for key in expired_keys:
            del self._cache[key]

        return len(expired_keys)

# Global cache instance
cache = SimpleCache(default_ttl=3600)  # 1 hour default TTL

def cached(ttl: int = 3600, key_prefix: str = ""):
    """
    Decorator for caching function results

    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache keys
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            key_parts = [key_prefix, func.__name__]

            # Add string representation of args and kwargs
            if args:
                key_parts.append(str(args))
            if kwargs:
                key_parts.append(str(sorted(kwargs.items())))

            cache_key = ":".join(filter(None, key_parts))

            # Try to get from cache first
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            await cache.set(cache_key, result, ttl)
            return result

        return wrapper
    return decorator

# Utility functions for common caching patterns
async def cache_research_results(query: str, results: Any, ttl: int = 1800) -> None:
    """Cache research results with 30-minute TTL"""
    await cache.set(f"research:{query}", results, ttl)

async def get_cached_research_results(query: str) -> Optional[Any]:
    """Get cached research results"""
    return await cache.get(f"research:{query}")

async def cache_ai_response(prompt_hash: str, response: Any, ttl: int = 3600) -> None:
    """Cache AI responses with 1-hour TTL"""
    await cache.set(f"ai_response:{prompt_hash}", response, ttl)

async def get_cached_ai_response(prompt_hash: str) -> Optional[Any]:
    """Get cached AI response"""
    return await cache.get(f"ai_response:{prompt_hash}")

# Background task to clean up expired entries
async def cleanup_task():
    """Background task to periodically clean up expired cache entries"""
    while True:
        try:
            removed = cache.cleanup_expired()
            if removed > 0:
                print(f"Cache cleanup: removed {removed} expired entries")
        except Exception as e:
            print(f"Cache cleanup error: {e}")

        # Run cleanup every 10 minutes
        await asyncio.sleep(600)