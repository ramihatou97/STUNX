"""
Health Check Background Tasks
Tasks for monitoring system health and service availability
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from celery import current_task
from core.task_manager import celery_app, BaseKOOTask
from core.redis_cache import redis_cache, CacheLevel
from core.database import db_manager
from services.hybrid_ai_manager import hybrid_ai_manager

logger = logging.getLogger(__name__)

@celery_app.task(base=BaseKOOTask, bind=True, name='koo.tasks.health_check.check_all_services')
def check_all_services(self) -> Dict[str, Any]:
    """
    Comprehensive health check for all system services
    
    Returns:
        Complete health status report
    """
    try:
        current_task.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Starting health checks'})
        
        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            health_report = {
                'overall_status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'services': {},
                'alerts': [],
                'summary': {}
            }
            
            # Check database health
            current_task.update_state(state='PROGRESS', meta={'progress': 20, 'status': 'Checking database'})
            
            db_health = await _check_database_health()
            health_report['services']['database'] = db_health
            
            if not db_health['healthy']:
                health_report['overall_status'] = 'degraded'
                health_report['alerts'].append({
                    'severity': 'critical',
                    'service': 'database',
                    'message': db_health.get('error', 'Database health check failed')
                })
            
            # Check Redis cache health
            current_task.update_state(state='PROGRESS', meta={'progress': 40, 'status': 'Checking cache'})
            
            cache_health = await redis_cache.health_check()
            health_report['services']['cache'] = cache_health
            
            if not cache_health['healthy']:
                health_report['overall_status'] = 'degraded'
                health_report['alerts'].append({
                    'severity': 'warning',
                    'service': 'cache',
                    'message': cache_health.get('error', 'Cache health check failed')
                })
            
            # Check AI services health
            current_task.update_state(state='PROGRESS', meta={'progress': 60, 'status': 'Checking AI services'})
            
            ai_services = ['claude', 'gemini', 'perplexity']
            ai_health = {}
            
            for service in ai_services:
                try:
                    service_health = await hybrid_ai_manager.health_check(service)
                    ai_health[service] = service_health
                    
                    if not service_health.get('healthy', False):
                        health_report['alerts'].append({
                            'severity': 'warning',
                            'service': f'ai_{service}',
                            'message': f'AI service {service} is not healthy'
                        })
                        
                except Exception as e:
                    ai_health[service] = {
                        'healthy': False,
                        'error': str(e)
                    }
                    health_report['alerts'].append({
                        'severity': 'warning',
                        'service': f'ai_{service}',
                        'message': f'AI service {service} health check failed: {str(e)}'
                    })
            
            health_report['services']['ai_services'] = ai_health
            
            # Check system resources
            current_task.update_state(state='PROGRESS', meta={'progress': 80, 'status': 'Checking system resources'})
            
            system_health = await _check_system_resources()
            health_report['services']['system'] = system_health
            
            if system_health.get('memory_usage_percent', 0) > 90:
                health_report['alerts'].append({
                    'severity': 'warning',
                    'service': 'system',
                    'message': f"High memory usage: {system_health['memory_usage_percent']}%"
                })
            
            # Generate summary
            total_services = len(health_report['services'])
            healthy_services = sum(1 for service in health_report['services'].values() 
                                 if service.get('healthy', False))
            
            health_report['summary'] = {
                'total_services': total_services,
                'healthy_services': healthy_services,
                'unhealthy_services': total_services - healthy_services,
                'total_alerts': len(health_report['alerts']),
                'critical_alerts': len([a for a in health_report['alerts'] if a['severity'] == 'critical']),
                'warning_alerts': len([a for a in health_report['alerts'] if a['severity'] == 'warning'])
            }
            
            # Determine overall status
            if health_report['summary']['critical_alerts'] > 0:
                health_report['overall_status'] = 'critical'
            elif health_report['summary']['warning_alerts'] > 0:
                health_report['overall_status'] = 'degraded'
            else:
                health_report['overall_status'] = 'healthy'
            
            # Cache health report
            await redis_cache.set(
                'system_health_report',
                health_report,
                ttl=300,  # 5 minutes
                level=CacheLevel.APPLICATION
            )
            
            current_task.update_state(state='PROGRESS', meta={'progress': 100, 'status': 'Health check complete'})
            
            logger.info(f"Health check completed: {health_report['overall_status']}")
            return health_report
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Health check task failed: {e}")
        raise

@celery_app.task(base=BaseKOOTask, bind=True, name='koo.tasks.health_check.check_database')
def check_database_health(self) -> Dict[str, Any]:
    """
    Detailed database health check
    
    Returns:
        Database health status
    """
    try:
        current_task.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Starting database check'})
        
        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            db_health = await _check_database_health()
            
            # Cache result
            await redis_cache.set(
                'health:database',
                db_health,
                ttl=300,
                level=CacheLevel.APPLICATION
            )
            
            current_task.update_state(state='PROGRESS', meta={'progress': 100, 'status': 'Database check complete'})
            
            return db_health
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Database health check task failed: {e}")
        raise

@celery_app.task(base=BaseKOOTask, bind=True, name='koo.tasks.health_check.check_ai_services')
def check_ai_services_health(self) -> Dict[str, Any]:
    """
    Detailed AI services health check
    
    Returns:
        AI services health status
    """
    try:
        current_task.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Starting AI services check'})
        
        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            ai_services = ['claude', 'gemini', 'perplexity']
            ai_health = {
                'overall_healthy': True,
                'services': {},
                'timestamp': datetime.now().isoformat()
            }
            
            for i, service in enumerate(ai_services):
                progress = int(((i + 1) / len(ai_services)) * 80) + 10
                current_task.update_state(
                    state='PROGRESS',
                    meta={'progress': progress, 'status': f'Checking {service}'}
                )
                
                try:
                    service_health = await hybrid_ai_manager.health_check(service)
                    ai_health['services'][service] = service_health
                    
                    if not service_health.get('healthy', False):
                        ai_health['overall_healthy'] = False
                        
                except Exception as e:
                    ai_health['services'][service] = {
                        'healthy': False,
                        'error': str(e)
                    }
                    ai_health['overall_healthy'] = False
            
            # Cache result
            await redis_cache.set(
                'health:ai_services',
                ai_health,
                ttl=300,
                level=CacheLevel.APPLICATION
            )
            
            current_task.update_state(state='PROGRESS', meta={'progress': 100, 'status': 'AI services check complete'})
            
            return ai_health
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"AI services health check task failed: {e}")
        raise

@celery_app.task(base=BaseKOOTask, bind=True, name='koo.tasks.health_check.check_cache')
def check_cache_health(self) -> Dict[str, Any]:
    """
    Detailed cache health check
    
    Returns:
        Cache health status
    """
    try:
        current_task.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Starting cache check'})
        
        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            cache_health = await redis_cache.health_check()
            
            # Add additional cache metrics
            cache_metrics = redis_cache.get_metrics()
            cache_health['metrics'] = cache_metrics
            
            # Cache result (but not in Redis to avoid recursion)
            current_task.update_state(state='PROGRESS', meta={'progress': 100, 'status': 'Cache check complete'})
            
            return cache_health
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Cache health check task failed: {e}")
        raise

@celery_app.task(base=BaseKOOTask, bind=True, name='koo.tasks.health_check.performance_test')
def performance_test(self) -> Dict[str, Any]:
    """
    Run performance tests on critical system components
    
    Returns:
        Performance test results
    """
    try:
        current_task.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Starting performance tests'})
        
        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            perf_results = {
                'timestamp': datetime.now().isoformat(),
                'tests': {}
            }
            
            # Test database performance
            current_task.update_state(state='PROGRESS', meta={'progress': 25, 'status': 'Testing database performance'})
            
            db_perf = await _test_database_performance()
            perf_results['tests']['database'] = db_perf
            
            # Test cache performance
            current_task.update_state(state='PROGRESS', meta={'progress': 50, 'status': 'Testing cache performance'})
            
            cache_perf = await _test_cache_performance()
            perf_results['tests']['cache'] = cache_perf
            
            # Test AI service performance
            current_task.update_state(state='PROGRESS', meta={'progress': 75, 'status': 'Testing AI service performance'})
            
            ai_perf = await _test_ai_performance()
            perf_results['tests']['ai_services'] = ai_perf
            
            # Cache results
            await redis_cache.set(
                'performance_test_results',
                perf_results,
                ttl=1800,  # 30 minutes
                level=CacheLevel.APPLICATION
            )
            
            current_task.update_state(state='PROGRESS', meta={'progress': 100, 'status': 'Performance tests complete'})
            
            return perf_results
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Performance test task failed: {e}")
        raise

# Helper functions

async def _check_database_health() -> Dict[str, Any]:
    """Check database health and connectivity"""
    try:
        start_time = asyncio.get_event_loop().time()
        
        # Test basic connectivity
        async with db_manager.session_scope() as session:
            await session.execute("SELECT 1")
        
        response_time = (asyncio.get_event_loop().time() - start_time) * 1000
        
        # Get pool metrics
        pool_metrics = db_manager.get_pool_metrics()
        
        return {
            'healthy': True,
            'response_time_ms': round(response_time, 2),
            'pool_metrics': pool_metrics,
            'circuit_breaker_open': not db_manager._check_circuit_breaker()
        }
        
    except Exception as e:
        return {
            'healthy': False,
            'error': str(e),
            'circuit_breaker_open': not db_manager._check_circuit_breaker()
        }

async def _check_system_resources() -> Dict[str, Any]:
    """Check system resource usage"""
    import psutil
    
    try:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'healthy': True,
            'memory_usage_percent': memory.percent,
            'memory_available_gb': round(memory.available / (1024**3), 2),
            'disk_usage_percent': disk.percent,
            'disk_free_gb': round(disk.free / (1024**3), 2),
            'cpu_usage_percent': psutil.cpu_percent(interval=1)
        }
        
    except Exception as e:
        return {
            'healthy': False,
            'error': str(e)
        }

async def _test_database_performance() -> Dict[str, Any]:
    """Test database performance"""
    try:
        start_time = asyncio.get_event_loop().time()
        
        # Simple query test
        async with db_manager.session_scope() as session:
            await session.execute("SELECT COUNT(*) FROM chapters")
        
        query_time = (asyncio.get_event_loop().time() - start_time) * 1000
        
        return {
            'query_response_time_ms': round(query_time, 2),
            'status': 'pass' if query_time < 1000 else 'slow'
        }
        
    except Exception as e:
        return {
            'status': 'fail',
            'error': str(e)
        }

async def _test_cache_performance() -> Dict[str, Any]:
    """Test cache performance"""
    try:
        # Test set operation
        start_time = asyncio.get_event_loop().time()
        await redis_cache.set('perf_test', {'test': 'data'}, ttl=60)
        set_time = (asyncio.get_event_loop().time() - start_time) * 1000
        
        # Test get operation
        start_time = asyncio.get_event_loop().time()
        result = await redis_cache.get('perf_test')
        get_time = (asyncio.get_event_loop().time() - start_time) * 1000
        
        # Cleanup
        await redis_cache.delete('perf_test')
        
        return {
            'set_time_ms': round(set_time, 2),
            'get_time_ms': round(get_time, 2),
            'status': 'pass' if (set_time < 100 and get_time < 50) else 'slow'
        }
        
    except Exception as e:
        return {
            'status': 'fail',
            'error': str(e)
        }

async def _test_ai_performance() -> Dict[str, Any]:
    """Test AI service performance"""
    try:
        # Simple health check for each service
        services = ['claude', 'gemini', 'perplexity']
        results = {}
        
        for service in services:
            start_time = asyncio.get_event_loop().time()
            health = await hybrid_ai_manager.health_check(service)
            response_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            results[service] = {
                'response_time_ms': round(response_time, 2),
                'healthy': health.get('healthy', False),
                'status': 'pass' if health.get('healthy', False) and response_time < 5000 else 'slow'
            }
        
        return results
        
    except Exception as e:
        return {
            'status': 'fail',
            'error': str(e)
        }
