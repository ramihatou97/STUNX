"""
Background Task Management System for KOO Platform
Comprehensive task queue system with Celery, monitoring, and error handling
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import traceback
import uuid

from celery import Celery, Task
from celery.result import AsyncResult
from celery.signals import task_prerun, task_postrun, task_failure, task_success
from kombu import Queue

from .config import settings
from .redis_cache import redis_cache, CacheLevel
from .exceptions import TaskError

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """Task status enumeration"""
    PENDING = "PENDING"
    STARTED = "STARTED"
    RETRY = "RETRY"
    FAILURE = "FAILURE"
    SUCCESS = "SUCCESS"
    REVOKED = "REVOKED"

class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class TaskCategory(Enum):
    """Task category enumeration"""
    AI_SERVICE = "ai_service"
    PDF_PROCESSING = "pdf_processing"
    DATABASE_MAINTENANCE = "database_maintenance"
    HEALTH_CHECK = "health_check"
    RESEARCH_SYNC = "research_sync"
    CACHE_WARMING = "cache_warming"
    NOTIFICATION = "notification"

@dataclass
class TaskMetrics:
    """Task execution metrics"""
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    retried_tasks: int = 0
    avg_execution_time: float = 0.0
    last_execution: Optional[datetime] = None
    execution_times: List[float] = field(default_factory=list)

@dataclass
class TaskInfo:
    """Comprehensive task information"""
    task_id: str
    name: str
    category: TaskCategory
    priority: TaskPriority
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    progress: int = 0
    metadata: dict = field(default_factory=dict)

# Celery configuration
def create_celery_app() -> Celery:
    """Create and configure Celery application"""
    
    # Celery configuration
    celery_config = {
        'broker_url': settings.CELERY_BROKER_URL,
        'result_backend': settings.CELERY_RESULT_BACKEND,
        'task_serializer': 'json',
        'accept_content': ['json'],
        'result_serializer': 'json',
        'timezone': 'UTC',
        'enable_utc': True,
        'task_track_started': True,
        'task_time_limit': 30 * 60,  # 30 minutes
        'task_soft_time_limit': 25 * 60,  # 25 minutes
        'worker_prefetch_multiplier': 1,
        'task_acks_late': True,
        'worker_disable_rate_limits': False,
        'task_compression': 'gzip',
        'result_compression': 'gzip',
        'result_expires': 3600,  # 1 hour
        'task_default_retry_delay': 60,  # 1 minute
        'task_max_retries': 3,
        'task_routes': {
            'koo.tasks.ai_service.*': {'queue': 'ai_service'},
            'koo.tasks.pdf_processing.*': {'queue': 'pdf_processing'},
            'koo.tasks.database.*': {'queue': 'database'},
            'koo.tasks.health_check.*': {'queue': 'health_check'},
            'koo.tasks.research.*': {'queue': 'research'},
            'koo.tasks.cache.*': {'queue': 'cache'},
        },
        'task_default_queue': 'default',
        'task_queues': (
            Queue('default', routing_key='default'),
            Queue('ai_service', routing_key='ai_service'),
            Queue('pdf_processing', routing_key='pdf_processing'),
            Queue('database', routing_key='database'),
            Queue('health_check', routing_key='health_check'),
            Queue('research', routing_key='research'),
            Queue('cache', routing_key='cache'),
        ),
        'beat_schedule': {
            'health-check-all-services': {
                'task': 'koo.tasks.health_check.check_all_services',
                'schedule': 300.0,  # Every 5 minutes
            },
            'cleanup-expired-cache': {
                'task': 'koo.tasks.cache.cleanup_expired',
                'schedule': 600.0,  # Every 10 minutes
            },
            'database-maintenance': {
                'task': 'koo.tasks.database.maintenance',
                'schedule': 3600.0,  # Every hour
            },
            'warm-critical-cache': {
                'task': 'koo.tasks.cache.warm_critical_data',
                'schedule': 1800.0,  # Every 30 minutes
            },
        },
    }
    
    app = Celery('koo')
    app.config_from_object(celery_config)
    
    return app

# Global Celery instance
celery_app = create_celery_app()

class BaseKOOTask(Task):
    """Base task class with enhanced error handling and monitoring"""
    
    def __init__(self):
        self.task_info: Optional[TaskInfo] = None
        
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called when task is retried"""
        logger.warning(f"Task {task_id} retrying due to: {exc}")
        if self.task_info:
            self.task_info.retry_count += 1
            self.task_info.status = TaskStatus.RETRY
            
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when task fails"""
        logger.error(f"Task {task_id} failed: {exc}")
        if self.task_info:
            self.task_info.status = TaskStatus.FAILURE
            self.task_info.error = str(exc)
            self.task_info.completed_at = datetime.now()
            
    def on_success(self, retval, task_id, args, kwargs):
        """Called when task succeeds"""
        logger.info(f"Task {task_id} completed successfully")
        if self.task_info:
            self.task_info.status = TaskStatus.SUCCESS
            self.task_info.result = retval
            self.task_info.completed_at = datetime.now()

class TaskManager:
    """
    Comprehensive task management system with monitoring and error handling
    """
    
    def __init__(self):
        self.celery = celery_app
        self.active_tasks: Dict[str, TaskInfo] = {}
        self.task_metrics: Dict[TaskCategory, TaskMetrics] = {
            category: TaskMetrics() for category in TaskCategory
        }
        self.global_metrics = TaskMetrics()
        
    async def submit_task(self, 
                         task_name: str,
                         category: TaskCategory,
                         args: tuple = (),
                         kwargs: dict = None,
                         priority: TaskPriority = TaskPriority.NORMAL,
                         max_retries: int = 3,
                         countdown: int = 0,
                         eta: Optional[datetime] = None,
                         metadata: dict = None) -> str:
        """Submit a task for background execution"""
        
        kwargs = kwargs or {}
        metadata = metadata or {}
        
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Create task info
        task_info = TaskInfo(
            task_id=task_id,
            name=task_name,
            category=category,
            priority=priority,
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
            args=args,
            kwargs=kwargs,
            max_retries=max_retries,
            metadata=metadata
        )
        
        # Store task info
        self.active_tasks[task_id] = task_info
        
        try:
            # Submit to Celery
            celery_task = self.celery.send_task(
                task_name,
                args=args,
                kwargs=kwargs,
                task_id=task_id,
                countdown=countdown,
                eta=eta,
                retry=True,
                retry_policy={
                    'max_retries': max_retries,
                    'interval_start': 1,
                    'interval_step': 2,
                    'interval_max': 60,
                }
            )
            
            # Cache task info
            await redis_cache.set(
                f"task:{task_id}",
                task_info.__dict__,
                ttl=86400,  # 24 hours
                level=CacheLevel.APPLICATION
            )
            
            logger.info(f"Task {task_id} ({task_name}) submitted successfully")
            return task_id
            
        except Exception as e:
            task_info.status = TaskStatus.FAILURE
            task_info.error = str(e)
            logger.error(f"Failed to submit task {task_id}: {e}")
            raise TaskError(f"Task submission failed: {str(e)}")
    
    async def get_task_status(self, task_id: str) -> Optional[TaskInfo]:
        """Get task status and information"""
        
        # Try to get from active tasks first
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]
        
        # Try to get from cache
        cached_info = await redis_cache.get(f"task:{task_id}", CacheLevel.APPLICATION)
        if cached_info:
            return TaskInfo(**cached_info)
        
        # Try to get from Celery result backend
        try:
            result = AsyncResult(task_id, app=self.celery)
            
            task_info = TaskInfo(
                task_id=task_id,
                name="unknown",
                category=TaskCategory.AI_SERVICE,  # Default
                priority=TaskPriority.NORMAL,
                status=TaskStatus(result.status),
                created_at=datetime.now(),  # Unknown
                result=result.result if result.successful() else None,
                error=str(result.result) if result.failed() else None
            )
            
            return task_info
            
        except Exception as e:
            logger.error(f"Failed to get task status for {task_id}: {e}")
            return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task"""
        try:
            self.celery.control.revoke(task_id, terminate=True)
            
            # Update task info
            if task_id in self.active_tasks:
                self.active_tasks[task_id].status = TaskStatus.REVOKED
                self.active_tasks[task_id].completed_at = datetime.now()
            
            logger.info(f"Task {task_id} cancelled")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {e}")
            return False
    
    async def retry_task(self, task_id: str) -> bool:
        """Retry a failed task"""
        try:
            task_info = await self.get_task_status(task_id)
            if not task_info:
                return False
            
            # Submit new task with same parameters
            new_task_id = await self.submit_task(
                task_info.name,
                task_info.category,
                task_info.args,
                task_info.kwargs,
                task_info.priority,
                task_info.max_retries,
                metadata=task_info.metadata
            )
            
            logger.info(f"Task {task_id} retried as {new_task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to retry task {task_id}: {e}")
            return False

    async def get_active_tasks(self, category: Optional[TaskCategory] = None) -> List[TaskInfo]:
        """Get list of active tasks, optionally filtered by category"""
        tasks = []

        for task_info in self.active_tasks.values():
            if category is None or task_info.category == category:
                if task_info.status in [TaskStatus.PENDING, TaskStatus.STARTED, TaskStatus.RETRY]:
                    tasks.append(task_info)

        return tasks

    async def get_task_history(self,
                              category: Optional[TaskCategory] = None,
                              status: Optional[TaskStatus] = None,
                              limit: int = 100) -> List[TaskInfo]:
        """Get task history with optional filtering"""

        # Get tasks from cache
        pattern = "task:*"
        task_keys = await redis_cache.get_keys(pattern, CacheLevel.APPLICATION)

        tasks = []
        for key in task_keys[:limit]:
            cached_info = await redis_cache.get(key, CacheLevel.APPLICATION)
            if cached_info:
                task_info = TaskInfo(**cached_info)

                # Apply filters
                if category and task_info.category != category:
                    continue
                if status and task_info.status != status:
                    continue

                tasks.append(task_info)

        # Sort by creation time (newest first)
        tasks.sort(key=lambda x: x.created_at, reverse=True)
        return tasks[:limit]

    def get_task_metrics(self, category: Optional[TaskCategory] = None) -> Dict[str, Any]:
        """Get task execution metrics"""

        if category:
            metrics = self.task_metrics[category]
            return {
                "category": category.value,
                "total_tasks": metrics.total_tasks,
                "successful_tasks": metrics.successful_tasks,
                "failed_tasks": metrics.failed_tasks,
                "retried_tasks": metrics.retried_tasks,
                "success_rate": metrics.successful_tasks / metrics.total_tasks if metrics.total_tasks > 0 else 0,
                "avg_execution_time": metrics.avg_execution_time,
                "last_execution": metrics.last_execution.isoformat() if metrics.last_execution else None
            }
        else:
            # Return global metrics
            return {
                "global": {
                    "total_tasks": self.global_metrics.total_tasks,
                    "successful_tasks": self.global_metrics.successful_tasks,
                    "failed_tasks": self.global_metrics.failed_tasks,
                    "retried_tasks": self.global_metrics.retried_tasks,
                    "success_rate": self.global_metrics.successful_tasks / self.global_metrics.total_tasks if self.global_metrics.total_tasks > 0 else 0,
                    "avg_execution_time": self.global_metrics.avg_execution_time,
                    "last_execution": self.global_metrics.last_execution.isoformat() if self.global_metrics.last_execution else None
                },
                "by_category": {
                    category.value: self.get_task_metrics(category)
                    for category in TaskCategory
                }
            }

    async def cleanup_completed_tasks(self, older_than_hours: int = 24) -> int:
        """Clean up completed tasks older than specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        cleaned_count = 0

        # Clean from active tasks
        to_remove = []
        for task_id, task_info in self.active_tasks.items():
            if (task_info.completed_at and
                task_info.completed_at < cutoff_time and
                task_info.status in [TaskStatus.SUCCESS, TaskStatus.FAILURE, TaskStatus.REVOKED]):
                to_remove.append(task_id)

        for task_id in to_remove:
            del self.active_tasks[task_id]
            cleaned_count += 1

        # Clean from cache
        pattern = "task:*"
        task_keys = await redis_cache.get_keys(pattern, CacheLevel.APPLICATION)

        for key in task_keys:
            cached_info = await redis_cache.get(key, CacheLevel.APPLICATION)
            if cached_info:
                task_info = TaskInfo(**cached_info)
                if (task_info.completed_at and
                    task_info.completed_at < cutoff_time and
                    task_info.status in [TaskStatus.SUCCESS, TaskStatus.FAILURE, TaskStatus.REVOKED]):
                    await redis_cache.delete(key, CacheLevel.APPLICATION)
                    cleaned_count += 1

        logger.info(f"Cleaned up {cleaned_count} completed tasks")
        return cleaned_count

    def _update_metrics(self, task_info: TaskInfo) -> None:
        """Update task metrics"""
        category_metrics = self.task_metrics[task_info.category]

        # Update category metrics
        category_metrics.total_tasks += 1
        self.global_metrics.total_tasks += 1

        if task_info.status == TaskStatus.SUCCESS:
            category_metrics.successful_tasks += 1
            self.global_metrics.successful_tasks += 1
        elif task_info.status == TaskStatus.FAILURE:
            category_metrics.failed_tasks += 1
            self.global_metrics.failed_tasks += 1
        elif task_info.status == TaskStatus.RETRY:
            category_metrics.retried_tasks += 1
            self.global_metrics.retried_tasks += 1

        # Update execution time
        if task_info.execution_time:
            category_metrics.execution_times.append(task_info.execution_time)
            self.global_metrics.execution_times.append(task_info.execution_time)

            # Calculate new average
            category_metrics.avg_execution_time = sum(category_metrics.execution_times) / len(category_metrics.execution_times)
            self.global_metrics.avg_execution_time = sum(self.global_metrics.execution_times) / len(self.global_metrics.execution_times)

        # Update last execution time
        category_metrics.last_execution = datetime.now()
        self.global_metrics.last_execution = datetime.now()

# Global task manager instance
task_manager = TaskManager()

# Celery signal handlers
@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **kwds):
    """Handle task prerun signal"""
    logger.info(f"Task {task_id} started")

    # Update task info
    if task_id in task_manager.active_tasks:
        task_info = task_manager.active_tasks[task_id]
        task_info.status = TaskStatus.STARTED
        task_info.started_at = datetime.now()

@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, retval=None, state=None, **kwds):
    """Handle task postrun signal"""
    logger.info(f"Task {task_id} finished with state: {state}")

    # Update task info
    if task_id in task_manager.active_tasks:
        task_info = task_manager.active_tasks[task_id]
        task_info.completed_at = datetime.now()

        if task_info.started_at:
            task_info.execution_time = (task_info.completed_at - task_info.started_at).total_seconds()

        task_info.status = TaskStatus(state)
        task_info.result = retval

        # Update metrics
        task_manager._update_metrics(task_info)

@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwds):
    """Handle task failure signal"""
    logger.error(f"Task {task_id} failed: {exception}")

    # Update task info
    if task_id in task_manager.active_tasks:
        task_info = task_manager.active_tasks[task_id]
        task_info.status = TaskStatus.FAILURE
        task_info.error = str(exception)
        task_info.completed_at = datetime.now()

@task_success.connect
def task_success_handler(sender=None, task_id=None, result=None, **kwds):
    """Handle task success signal"""
    logger.info(f"Task {task_id} succeeded")

    # Update task info
    if task_id in task_manager.active_tasks:
        task_info = task_manager.active_tasks[task_id]
        task_info.status = TaskStatus.SUCCESS
        task_info.result = result
        task_info.completed_at = datetime.now()

# Convenience functions for common task operations
async def submit_ai_task(task_name: str, *args, **kwargs) -> str:
    """Submit AI service task"""
    return await task_manager.submit_task(
        task_name,
        TaskCategory.AI_SERVICE,
        args,
        kwargs,
        priority=TaskPriority.HIGH
    )

async def submit_pdf_processing_task(task_name: str, *args, **kwargs) -> str:
    """Submit PDF processing task"""
    return await task_manager.submit_task(
        task_name,
        TaskCategory.PDF_PROCESSING,
        args,
        kwargs,
        priority=TaskPriority.NORMAL
    )

async def submit_database_task(task_name: str, *args, **kwargs) -> str:
    """Submit database maintenance task"""
    return await task_manager.submit_task(
        task_name,
        TaskCategory.DATABASE_MAINTENANCE,
        args,
        kwargs,
        priority=TaskPriority.LOW
    )

async def submit_health_check_task(task_name: str, *args, **kwargs) -> str:
    """Submit health check task"""
    return await task_manager.submit_task(
        task_name,
        TaskCategory.HEALTH_CHECK,
        args,
        kwargs,
        priority=TaskPriority.CRITICAL
    )

async def submit_research_task(task_name: str, *args, **kwargs) -> str:
    """Submit research synchronization task"""
    return await task_manager.submit_task(
        task_name,
        TaskCategory.RESEARCH_SYNC,
        args,
        kwargs,
        priority=TaskPriority.NORMAL
    )

async def submit_cache_task(task_name: str, *args, **kwargs) -> str:
    """Submit cache management task"""
    return await task_manager.submit_task(
        task_name,
        TaskCategory.CACHE_WARMING,
        args,
        kwargs,
        priority=TaskPriority.LOW
    )
