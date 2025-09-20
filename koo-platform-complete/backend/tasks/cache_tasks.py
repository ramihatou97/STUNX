"""
Cache Management Background Tasks
Tasks for cache warming, cleanup, and maintenance
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from celery import current_task
from core.task_manager import celery_app, BaseKOOTask
from core.redis_cache import redis_cache, CacheLevel
from core.database import db_manager

logger = logging.getLogger(__name__)

@celery_app.task(base=BaseKOOTask, bind=True, name='koo.tasks.cache.cleanup_expired')
def cleanup_expired_cache(self) -> Dict[str, Any]:
    """
    Clean up expired cache entries across all cache levels
    
    Returns:
        Cleanup statistics
    """
    try:
        current_task.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Starting cache cleanup'})
        
        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            cleanup_stats = {
                'total_cleaned': 0,
                'by_level': {},
                'timestamp': datetime.now().isoformat()
            }
            
            # Clean up each cache level
            for i, level in enumerate(CacheLevel):
                progress = int(((i + 1) / len(CacheLevel)) * 80) + 10
                current_task.update_state(
                    state='PROGRESS',
                    meta={'progress': progress, 'status': f'Cleaning {level.value} cache'}
                )
                
                # Get all keys for this level
                pattern = f"koo:{level.value}:*"
                keys_to_check = []
                
                # Scan for keys
                async for key in redis_cache.redis_client.scan_iter(match=pattern):
                    keys_to_check.append(key)
                
                cleaned_count = 0
                for key in keys_to_check:
                    try:
                        # Check if key has expiration metadata
                        expires_at = await redis_cache.redis_client.hget(key, "expires_at")
                        if expires_at and float(expires_at) < asyncio.get_event_loop().time():
                            await redis_cache.redis_client.delete(key)
                            cleaned_count += 1
                    except Exception as e:
                        logger.warning(f"Error checking key {key}: {e}")
                        continue
                
                cleanup_stats['by_level'][level.value] = cleaned_count
                cleanup_stats['total_cleaned'] += cleaned_count
            
            current_task.update_state(state='PROGRESS', meta={'progress': 100, 'status': 'Cleanup complete'})
            
            logger.info(f"Cache cleanup completed: {cleanup_stats['total_cleaned']} entries removed")
            return cleanup_stats
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Cache cleanup task failed: {e}")
        raise

@celery_app.task(base=BaseKOOTask, bind=True, name='koo.tasks.cache.warm_critical_data')
def warm_critical_data(self) -> Dict[str, Any]:
    """
    Warm cache with critical frequently-accessed data
    
    Returns:
        Cache warming statistics
    """
    try:
        current_task.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Starting cache warming'})
        
        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            warming_stats = {
                'chapters_warmed': 0,
                'research_warmed': 0,
                'ai_responses_warmed': 0,
                'total_warmed': 0,
                'timestamp': datetime.now().isoformat()
            }
            
            # Warm frequently accessed chapters
            current_task.update_state(state='PROGRESS', meta={'progress': 20, 'status': 'Warming chapter data'})
            
            # Get most accessed chapters from database
            async with db_manager.session_scope() as session:
                # This would be implemented based on your chapter access tracking
                # For now, we'll simulate warming top chapters
                top_chapters = await _get_top_chapters(session)
                
                for chapter in top_chapters:
                    cache_key = f"chapter:{chapter['id']}"
                    await redis_cache.set(
                        cache_key,
                        chapter,
                        ttl=7200,  # 2 hours
                        level=CacheLevel.APPLICATION
                    )
                    warming_stats['chapters_warmed'] += 1
            
            # Warm frequently used research queries
            current_task.update_state(state='PROGRESS', meta={'progress': 50, 'status': 'Warming research data'})
            
            frequent_queries = [
                "brain tumor classification",
                "neurosurgical techniques",
                "spinal cord injury treatment",
                "craniotomy procedures",
                "neurological assessment"
            ]
            
            for query in frequent_queries:
                # Check if we have cached research results
                cache_key = f"research:{query}"
                cached_result = await redis_cache.get(cache_key, CacheLevel.APPLICATION)
                
                if cached_result:
                    # Refresh the cache with extended TTL
                    await redis_cache.set(
                        cache_key,
                        cached_result,
                        ttl=10800,  # 3 hours
                        level=CacheLevel.APPLICATION
                    )
                    warming_stats['research_warmed'] += 1
            
            # Warm common AI responses
            current_task.update_state(state='PROGRESS', meta={'progress': 80, 'status': 'Warming AI responses'})
            
            common_ai_queries = [
                {
                    'service': 'claude',
                    'prompt': 'What are the key considerations for neurosurgical planning?'
                },
                {
                    'service': 'gemini',
                    'prompt': 'Explain the anatomy of the brain stem'
                },
                {
                    'service': 'claude',
                    'prompt': 'What are the latest advances in minimally invasive neurosurgery?'
                }
            ]
            
            for query in common_ai_queries:
                cache_key = f"ai_query:{query['service']}:{hash(query['prompt'])}"
                cached_result = await redis_cache.get(cache_key, CacheLevel.API)
                
                if cached_result:
                    # Refresh with extended TTL
                    await redis_cache.set(
                        cache_key,
                        cached_result,
                        ttl=7200,  # 2 hours
                        level=CacheLevel.API
                    )
                    warming_stats['ai_responses_warmed'] += 1
            
            warming_stats['total_warmed'] = (
                warming_stats['chapters_warmed'] +
                warming_stats['research_warmed'] +
                warming_stats['ai_responses_warmed']
            )
            
            current_task.update_state(state='PROGRESS', meta={'progress': 100, 'status': 'Warming complete'})
            
            logger.info(f"Cache warming completed: {warming_stats['total_warmed']} entries warmed")
            return warming_stats
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Cache warming task failed: {e}")
        raise

@celery_app.task(base=BaseKOOTask, bind=True, name='koo.tasks.cache.maintenance')
def cache_maintenance(self) -> Dict[str, Any]:
    """
    Perform comprehensive cache maintenance
    
    Returns:
        Maintenance statistics
    """
    try:
        current_task.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Starting cache maintenance'})
        
        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            maintenance_stats = {
                'health_check': {},
                'memory_usage': {},
                'key_analysis': {},
                'optimization_applied': False,
                'timestamp': datetime.now().isoformat()
            }
            
            # Perform health check
            current_task.update_state(state='PROGRESS', meta={'progress': 20, 'status': 'Health check'})
            
            health_status = await redis_cache.health_check()
            maintenance_stats['health_check'] = health_status
            
            # Analyze memory usage
            current_task.update_state(state='PROGRESS', meta={'progress': 40, 'status': 'Analyzing memory usage'})
            
            info = await redis_cache.redis_client.info('memory')
            maintenance_stats['memory_usage'] = {
                'used_memory': info.get('used_memory'),
                'used_memory_human': info.get('used_memory_human'),
                'used_memory_peak': info.get('used_memory_peak'),
                'used_memory_peak_human': info.get('used_memory_peak_human'),
                'memory_fragmentation_ratio': info.get('mem_fragmentation_ratio')
            }
            
            # Analyze key distribution
            current_task.update_state(state='PROGRESS', meta={'progress': 60, 'status': 'Analyzing key distribution'})
            
            key_stats = {}
            for level in CacheLevel:
                pattern = f"koo:{level.value}:*"
                count = 0
                total_size = 0
                
                async for key in redis_cache.redis_client.scan_iter(match=pattern):
                    count += 1
                    try:
                        size = await redis_cache.redis_client.memory_usage(key)
                        if size:
                            total_size += size
                    except:
                        pass
                
                key_stats[level.value] = {
                    'count': count,
                    'total_size_bytes': total_size,
                    'avg_size_bytes': total_size / count if count > 0 else 0
                }
            
            maintenance_stats['key_analysis'] = key_stats
            
            # Apply optimizations if needed
            current_task.update_state(state='PROGRESS', meta={'progress': 80, 'status': 'Applying optimizations'})
            
            # Check if memory usage is high
            used_memory = info.get('used_memory', 0)
            max_memory = info.get('maxmemory', 0)
            
            if max_memory > 0 and used_memory / max_memory > 0.8:
                # High memory usage - clean up old entries
                cutoff_time = datetime.now() - timedelta(hours=6)
                cleaned = await _cleanup_old_cache_entries(cutoff_time)
                maintenance_stats['optimization_applied'] = True
                maintenance_stats['cleaned_entries'] = cleaned
                logger.info(f"Applied memory optimization: cleaned {cleaned} old entries")
            
            current_task.update_state(state='PROGRESS', meta={'progress': 100, 'status': 'Maintenance complete'})
            
            logger.info("Cache maintenance completed successfully")
            return maintenance_stats
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Cache maintenance task failed: {e}")
        raise

@celery_app.task(base=BaseKOOTask, bind=True, name='koo.tasks.cache.preload_frequent_queries')
def preload_frequent_queries(self, queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Preload frequently used queries into cache
    
    Args:
        queries: List of query configurations to preload
    
    Returns:
        Preloading statistics
    """
    try:
        current_task.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Starting query preloading'})
        
        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            preload_stats = {
                'total_queries': len(queries),
                'preloaded_count': 0,
                'skipped_count': 0,
                'failed_count': 0,
                'timestamp': datetime.now().isoformat()
            }
            
            for i, query_config in enumerate(queries):
                progress = int((i / len(queries)) * 90) + 10
                current_task.update_state(
                    state='PROGRESS',
                    meta={'progress': progress, 'status': f'Preloading query {i+1}/{len(queries)}'}
                )
                
                try:
                    query_type = query_config.get('type')
                    cache_key = query_config.get('cache_key')
                    ttl = query_config.get('ttl', 3600)
                    level = CacheLevel(query_config.get('level', 'application'))
                    
                    # Check if already cached
                    existing = await redis_cache.get(cache_key, level)
                    if existing:
                        preload_stats['skipped_count'] += 1
                        continue
                    
                    # Execute query based on type
                    if query_type == 'database':
                        result = await _execute_database_query(query_config)
                    elif query_type == 'api':
                        result = await _execute_api_query(query_config)
                    elif query_type == 'computation':
                        result = await _execute_computation(query_config)
                    else:
                        logger.warning(f"Unknown query type: {query_type}")
                        preload_stats['failed_count'] += 1
                        continue
                    
                    # Cache the result
                    await redis_cache.set(cache_key, result, ttl, level)
                    preload_stats['preloaded_count'] += 1
                    
                except Exception as e:
                    logger.error(f"Failed to preload query {i}: {e}")
                    preload_stats['failed_count'] += 1
                    continue
            
            current_task.update_state(state='PROGRESS', meta={'progress': 100, 'status': 'Preloading complete'})
            
            logger.info(f"Query preloading completed: {preload_stats['preloaded_count']} queries preloaded")
            return preload_stats
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Query preloading task failed: {e}")
        raise

# Helper functions

async def _get_top_chapters(session) -> List[Dict[str, Any]]:
    """Get most frequently accessed chapters"""
    # This would be implemented based on your access tracking
    # For now, return mock data
    return [
        {'id': 1, 'title': 'Brain Anatomy', 'content': '...'},
        {'id': 2, 'title': 'Surgical Techniques', 'content': '...'},
        {'id': 3, 'title': 'Neurological Assessment', 'content': '...'}
    ]

async def _cleanup_old_cache_entries(cutoff_time: datetime) -> int:
    """Clean up cache entries older than cutoff time"""
    cleaned_count = 0
    
    for level in CacheLevel:
        pattern = f"koo:{level.value}:*"
        
        async for key in redis_cache.redis_client.scan_iter(match=pattern):
            try:
                created_at = await redis_cache.redis_client.hget(key, "created_at")
                if created_at:
                    created_time = datetime.fromtimestamp(float(created_at))
                    if created_time < cutoff_time:
                        await redis_cache.redis_client.delete(key)
                        cleaned_count += 1
            except Exception:
                continue
    
    return cleaned_count

async def _execute_database_query(query_config: Dict[str, Any]) -> Any:
    """Execute database query for preloading"""
    # Implementation would depend on specific query
    return {"result": "database_query_result"}

async def _execute_api_query(query_config: Dict[str, Any]) -> Any:
    """Execute API query for preloading"""
    # Implementation would depend on specific API
    return {"result": "api_query_result"}

async def _execute_computation(query_config: Dict[str, Any]) -> Any:
    """Execute computation for preloading"""
    # Implementation would depend on specific computation
    return {"result": "computation_result"}
