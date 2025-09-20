"""
Cache Management API Endpoints
API for managing cache operations, monitoring, and maintenance
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Query, status
from pydantic import BaseModel

from core.dependencies import get_current_user, CurrentUser
from core.redis_cache import redis_cache, CacheLevel
from core.task_manager import task_manager, submit_cache_task

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models
class CacheKeyRequest(BaseModel):
    key: str
    level: str = "application"

class CacheSetRequest(BaseModel):
    key: str
    value: Any
    ttl: Optional[int] = None
    level: str = "application"

class CacheWarmingRequest(BaseModel):
    queries: List[Dict[str, Any]]

class CacheResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None

@router.get("/health", summary="Get cache health status")
async def get_cache_health(
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get comprehensive cache health status"""
    try:
        health_status = await redis_cache.health_check()
        return {
            "status": "healthy" if health_status["healthy"] else "unhealthy",
            "health_details": health_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Cache health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cache health check failed: {str(e)}"
        )

@router.get("/metrics", summary="Get cache metrics")
async def get_cache_metrics(
    level: Optional[str] = Query(None, description="Cache level to get metrics for"),
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get cache performance metrics"""
    try:
        cache_level = None
        if level:
            try:
                cache_level = CacheLevel(level)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid cache level: {level}"
                )
        
        metrics = redis_cache.get_metrics(cache_level)
        return {
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get cache metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cache metrics: {str(e)}"
        )

@router.get("/keys", summary="Get cache keys")
async def get_cache_keys(
    pattern: str = Query("*", description="Key pattern to search for"),
    level: str = Query("application", description="Cache level"),
    limit: int = Query(100, description="Maximum number of keys to return"),
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get cache keys matching pattern"""
    try:
        cache_level = CacheLevel(level)
        keys = await redis_cache.get_keys(pattern, cache_level)
        
        # Limit results
        if len(keys) > limit:
            keys = keys[:limit]
        
        return {
            "keys": keys,
            "total_found": len(keys),
            "pattern": pattern,
            "level": level,
            "timestamp": datetime.now().isoformat()
        }
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid cache level: {level}"
        )
    except Exception as e:
        logger.error(f"Failed to get cache keys: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cache keys: {str(e)}"
        )

@router.get("/get", summary="Get cache value")
async def get_cache_value(
    request: CacheKeyRequest,
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get value from cache"""
    try:
        cache_level = CacheLevel(request.level)
        value = await redis_cache.get(request.key, cache_level)
        
        return {
            "key": request.key,
            "value": value,
            "found": value is not None,
            "level": request.level,
            "timestamp": datetime.now().isoformat()
        }
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid cache level: {request.level}"
        )
    except Exception as e:
        logger.error(f"Failed to get cache value: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cache value: {str(e)}"
        )

@router.post("/set", summary="Set cache value")
async def set_cache_value(
    request: CacheSetRequest,
    current_user: CurrentUser = Depends(get_current_user)
) -> CacheResponse:
    """Set value in cache"""
    try:
        cache_level = CacheLevel(request.level)
        success = await redis_cache.set(
            request.key, 
            request.value, 
            request.ttl, 
            cache_level
        )
        
        return CacheResponse(
            success=success,
            message=f"Cache value {'set' if success else 'failed to set'} for key: {request.key}",
            data={
                "key": request.key,
                "level": request.level,
                "ttl": request.ttl
            }
        )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid cache level: {request.level}"
        )
    except Exception as e:
        logger.error(f"Failed to set cache value: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to set cache value: {str(e)}"
        )

@router.delete("/delete", summary="Delete cache value")
async def delete_cache_value(
    request: CacheKeyRequest,
    current_user: CurrentUser = Depends(get_current_user)
) -> CacheResponse:
    """Delete value from cache"""
    try:
        cache_level = CacheLevel(request.level)
        success = await redis_cache.delete(request.key, cache_level)
        
        return CacheResponse(
            success=success,
            message=f"Cache value {'deleted' if success else 'not found'} for key: {request.key}",
            data={
                "key": request.key,
                "level": request.level
            }
        )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid cache level: {request.level}"
        )
    except Exception as e:
        logger.error(f"Failed to delete cache value: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete cache value: {str(e)}"
        )

@router.post("/clear", summary="Clear cache level")
async def clear_cache_level(
    level: str = Query(..., description="Cache level to clear"),
    current_user: CurrentUser = Depends(get_current_user)
) -> CacheResponse:
    """Clear all keys for a specific cache level"""
    try:
        cache_level = CacheLevel(level)
        cleared_count = await redis_cache.clear_level(cache_level)
        
        return CacheResponse(
            success=True,
            message=f"Cleared {cleared_count} keys from {level} cache",
            data={
                "level": level,
                "cleared_count": cleared_count
            }
        )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid cache level: {level}"
        )
    except Exception as e:
        logger.error(f"Failed to clear cache level: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache level: {str(e)}"
        )

@router.post("/clear-all", summary="Clear all cache")
async def clear_all_cache(
    current_user: CurrentUser = Depends(get_current_user)
) -> CacheResponse:
    """Clear all cache data (use with caution)"""
    try:
        success = await redis_cache.clear_all()
        
        return CacheResponse(
            success=success,
            message="All cache data cleared" if success else "Failed to clear cache",
            data={"timestamp": datetime.now().isoformat()}
        )
    except Exception as e:
        logger.error(f"Failed to clear all cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear all cache: {str(e)}"
        )

@router.post("/maintenance", summary="Start cache maintenance")
async def start_cache_maintenance(
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """Start cache maintenance task"""
    try:
        task_id = await submit_cache_task(
            "koo.tasks.cache.maintenance"
        )
        
        return {
            "task_id": task_id,
            "message": "Cache maintenance task started",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to start cache maintenance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start cache maintenance: {str(e)}"
        )

@router.post("/warm", summary="Warm cache with critical data")
async def warm_cache(
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """Start cache warming task"""
    try:
        task_id = await submit_cache_task(
            "koo.tasks.cache.warm_critical_data"
        )
        
        return {
            "task_id": task_id,
            "message": "Cache warming task started",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to start cache warming: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start cache warming: {str(e)}"
        )

@router.post("/cleanup", summary="Clean up expired cache entries")
async def cleanup_expired_cache(
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """Start cache cleanup task"""
    try:
        task_id = await submit_cache_task(
            "koo.tasks.cache.cleanup_expired"
        )
        
        return {
            "task_id": task_id,
            "message": "Cache cleanup task started",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to start cache cleanup: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start cache cleanup: {str(e)}"
        )
