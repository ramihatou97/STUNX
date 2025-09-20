"""
AI Service Background Tasks
Tasks for handling AI service operations asynchronously
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from celery import current_task
from core.task_manager import celery_app, BaseKOOTask
from core.redis_cache import redis_cache, CacheLevel
from services.hybrid_ai_manager import hybrid_ai_manager, query_ai
from core.ai_error_handling import ai_error_handler

logger = logging.getLogger(__name__)

@celery_app.task(base=BaseKOOTask, bind=True, name='koo.tasks.ai_service.process_query')
def process_ai_query(self, service: str, prompt: str, **kwargs) -> Dict[str, Any]:
    """
    Process AI query asynchronously
    
    Args:
        service: AI service name (claude, gemini, perplexity)
        prompt: Query prompt
        **kwargs: Additional parameters
    
    Returns:
        Dict containing response and metadata
    """
    try:
        # Update task progress
        current_task.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Starting AI query'})
        
        # Create event loop for async operations
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Check cache first
            cache_key = f"ai_query:{service}:{hash(prompt)}"
            cached_result = loop.run_until_complete(
                redis_cache.get(cache_key, CacheLevel.API)
            )
            
            if cached_result:
                logger.info(f"AI query cache hit for service {service}")
                return {
                    'response': cached_result,
                    'service': service,
                    'cached': True,
                    'timestamp': datetime.now().isoformat()
                }
            
            current_task.update_state(state='PROGRESS', meta={'progress': 30, 'status': 'Querying AI service'})
            
            # Execute AI query
            response = loop.run_until_complete(
                query_ai(service, prompt, **kwargs)
            )
            
            current_task.update_state(state='PROGRESS', meta={'progress': 80, 'status': 'Caching result'})
            
            # Cache the result
            loop.run_until_complete(
                redis_cache.set(cache_key, response, ttl=3600, level=CacheLevel.API)
            )
            
            current_task.update_state(state='PROGRESS', meta={'progress': 100, 'status': 'Complete'})
            
            return {
                'response': response,
                'service': service,
                'cached': False,
                'timestamp': datetime.now().isoformat()
            }
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"AI query task failed: {e}")
        raise

@celery_app.task(base=BaseKOOTask, bind=True, name='koo.tasks.ai_service.batch_queries')
def batch_ai_queries(self, queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process multiple AI queries in batch
    
    Args:
        queries: List of query dictionaries with 'service', 'prompt', and optional kwargs
    
    Returns:
        List of response dictionaries
    """
    try:
        results = []
        total_queries = len(queries)
        
        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            for i, query in enumerate(queries):
                progress = int((i / total_queries) * 100)
                current_task.update_state(
                    state='PROGRESS', 
                    meta={'progress': progress, 'status': f'Processing query {i+1}/{total_queries}'}
                )
                
                service = query.get('service')
                prompt = query.get('prompt')
                kwargs = query.get('kwargs', {})
                
                # Check cache
                cache_key = f"ai_query:{service}:{hash(prompt)}"
                cached_result = loop.run_until_complete(
                    redis_cache.get(cache_key, CacheLevel.API)
                )
                
                if cached_result:
                    result = {
                        'response': cached_result,
                        'service': service,
                        'cached': True,
                        'query_index': i
                    }
                else:
                    # Execute query
                    response = loop.run_until_complete(
                        query_ai(service, prompt, **kwargs)
                    )
                    
                    # Cache result
                    loop.run_until_complete(
                        redis_cache.set(cache_key, response, ttl=3600, level=CacheLevel.API)
                    )
                    
                    result = {
                        'response': response,
                        'service': service,
                        'cached': False,
                        'query_index': i
                    }
                
                results.append(result)
            
            return results
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Batch AI queries task failed: {e}")
        raise

@celery_app.task(base=BaseKOOTask, bind=True, name='koo.tasks.ai_service.health_check')
def ai_service_health_check(self, service: Optional[str] = None) -> Dict[str, Any]:
    """
    Perform health check on AI services
    
    Args:
        service: Specific service to check, or None for all services
    
    Returns:
        Health check results
    """
    try:
        current_task.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Starting health check'})
        
        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            if service:
                # Check specific service
                health_status = loop.run_until_complete(
                    hybrid_ai_manager.health_check(service)
                )
                
                # Cache health status
                loop.run_until_complete(
                    redis_cache.set(
                        f"health:ai_service:{service}",
                        health_status,
                        ttl=300,  # 5 minutes
                        level=CacheLevel.APPLICATION
                    )
                )
                
                return {
                    'service': service,
                    'health': health_status,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                # Check all services
                services = ['claude', 'gemini', 'perplexity']
                results = {}
                
                for i, svc in enumerate(services):
                    progress = int(((i + 1) / len(services)) * 100)
                    current_task.update_state(
                        state='PROGRESS',
                        meta={'progress': progress, 'status': f'Checking {svc}'}
                    )
                    
                    health_status = loop.run_until_complete(
                        hybrid_ai_manager.health_check(svc)
                    )
                    
                    # Cache health status
                    loop.run_until_complete(
                        redis_cache.set(
                            f"health:ai_service:{svc}",
                            health_status,
                            ttl=300,
                            level=CacheLevel.APPLICATION
                        )
                    )
                    
                    results[svc] = health_status
                
                return {
                    'services': results,
                    'timestamp': datetime.now().isoformat()
                }
                
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"AI service health check task failed: {e}")
        raise

@celery_app.task(base=BaseKOOTask, bind=True, name='koo.tasks.ai_service.warm_cache')
def warm_ai_cache(self, queries: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Warm AI service cache with frequently used queries
    
    Args:
        queries: List of queries to pre-cache
    
    Returns:
        Cache warming results
    """
    try:
        current_task.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Starting cache warming'})
        
        warmed_count = 0
        total_queries = len(queries)
        
        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            for i, query in enumerate(queries):
                progress = int((i / total_queries) * 100)
                current_task.update_state(
                    state='PROGRESS',
                    meta={'progress': progress, 'status': f'Warming query {i+1}/{total_queries}'}
                )
                
                service = query.get('service')
                prompt = query.get('prompt')
                
                # Check if already cached
                cache_key = f"ai_query:{service}:{hash(prompt)}"
                cached_result = loop.run_until_complete(
                    redis_cache.get(cache_key, CacheLevel.API)
                )
                
                if not cached_result:
                    try:
                        # Execute query and cache result
                        response = loop.run_until_complete(
                            query_ai(service, prompt)
                        )
                        
                        loop.run_until_complete(
                            redis_cache.set(cache_key, response, ttl=7200, level=CacheLevel.API)
                        )
                        
                        warmed_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to warm cache for query {i}: {e}")
                        continue
            
            return {
                'total_queries': total_queries,
                'warmed_count': warmed_count,
                'timestamp': datetime.now().isoformat()
            }
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"AI cache warming task failed: {e}")
        raise

@celery_app.task(base=BaseKOOTask, bind=True, name='koo.tasks.ai_service.cleanup_errors')
def cleanup_ai_errors(self) -> Dict[str, Any]:
    """
    Clean up AI service error states and reset circuit breakers if appropriate
    
    Returns:
        Cleanup results
    """
    try:
        current_task.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Starting error cleanup'})
        
        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Get all AI services
            services = ['claude', 'gemini', 'perplexity']
            reset_count = 0
            
            for i, service in enumerate(services):
                progress = int(((i + 1) / len(services)) * 100)
                current_task.update_state(
                    state='PROGRESS',
                    meta={'progress': progress, 'status': f'Checking {service}'}
                )
                
                # Check service health
                health_status = loop.run_until_complete(
                    hybrid_ai_manager.health_check(service)
                )
                
                # Reset circuit breaker if service is healthy but circuit is open
                if health_status.get('healthy') and not health_status.get('circuit_breaker_closed', True):
                    ai_error_handler.reset_circuit_breaker(service)
                    reset_count += 1
                    logger.info(f"Reset circuit breaker for {service}")
            
            return {
                'services_checked': len(services),
                'circuit_breakers_reset': reset_count,
                'timestamp': datetime.now().isoformat()
            }
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"AI error cleanup task failed: {e}")
        raise
