"""
Task Management API Endpoints
API for managing background tasks, monitoring, and control
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Query, status
from pydantic import BaseModel

from core.dependencies import get_current_user, CurrentUser
from core.task_manager import (
    task_manager, TaskCategory, TaskPriority, TaskStatus,
    submit_ai_task, submit_pdf_processing_task, submit_database_task,
    submit_health_check_task, submit_research_task, submit_cache_task
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models
class TaskSubmissionRequest(BaseModel):
    task_name: str
    category: str
    args: List[Any] = []
    kwargs: Dict[str, Any] = {}
    priority: str = "normal"
    max_retries: int = 3
    countdown: int = 0
    metadata: Dict[str, Any] = {}

class TaskResponse(BaseModel):
    success: bool
    message: str
    task_id: Optional[str] = None
    data: Optional[Any] = None

@router.get("/status/{task_id}", summary="Get task status")
async def get_task_status(
    task_id: str,
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get status and information for a specific task"""
    try:
        task_info = await task_manager.get_task_status(task_id)
        
        if not task_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )
        
        return {
            "task_id": task_info.task_id,
            "name": task_info.name,
            "category": task_info.category.value,
            "priority": task_info.priority.value,
            "status": task_info.status.value,
            "created_at": task_info.created_at.isoformat(),
            "started_at": task_info.started_at.isoformat() if task_info.started_at else None,
            "completed_at": task_info.completed_at.isoformat() if task_info.completed_at else None,
            "execution_time": task_info.execution_time,
            "result": task_info.result,
            "error": task_info.error,
            "retry_count": task_info.retry_count,
            "max_retries": task_info.max_retries,
            "progress": task_info.progress,
            "metadata": task_info.metadata
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get task status: {str(e)}"
        )

@router.get("/active", summary="Get active tasks")
async def get_active_tasks(
    category: Optional[str] = Query(None, description="Filter by task category"),
    limit: int = Query(100, description="Maximum number of tasks to return"),
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get list of active tasks"""
    try:
        task_category = None
        if category:
            try:
                task_category = TaskCategory(category)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid task category: {category}"
                )
        
        active_tasks = await task_manager.get_active_tasks(task_category)
        
        # Limit results
        if len(active_tasks) > limit:
            active_tasks = active_tasks[:limit]
        
        tasks_data = []
        for task_info in active_tasks:
            tasks_data.append({
                "task_id": task_info.task_id,
                "name": task_info.name,
                "category": task_info.category.value,
                "priority": task_info.priority.value,
                "status": task_info.status.value,
                "created_at": task_info.created_at.isoformat(),
                "started_at": task_info.started_at.isoformat() if task_info.started_at else None,
                "progress": task_info.progress,
                "retry_count": task_info.retry_count
            })
        
        return {
            "active_tasks": tasks_data,
            "total_found": len(tasks_data),
            "category_filter": category,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get active tasks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get active tasks: {str(e)}"
        )

@router.get("/history", summary="Get task history")
async def get_task_history(
    category: Optional[str] = Query(None, description="Filter by task category"),
    status_filter: Optional[str] = Query(None, description="Filter by task status"),
    limit: int = Query(100, description="Maximum number of tasks to return"),
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get task execution history"""
    try:
        task_category = None
        if category:
            try:
                task_category = TaskCategory(category)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid task category: {category}"
                )
        
        task_status = None
        if status_filter:
            try:
                task_status = TaskStatus(status_filter)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid task status: {status_filter}"
                )
        
        task_history = await task_manager.get_task_history(
            task_category, task_status, limit
        )
        
        history_data = []
        for task_info in task_history:
            history_data.append({
                "task_id": task_info.task_id,
                "name": task_info.name,
                "category": task_info.category.value,
                "priority": task_info.priority.value,
                "status": task_info.status.value,
                "created_at": task_info.created_at.isoformat(),
                "started_at": task_info.started_at.isoformat() if task_info.started_at else None,
                "completed_at": task_info.completed_at.isoformat() if task_info.completed_at else None,
                "execution_time": task_info.execution_time,
                "retry_count": task_info.retry_count,
                "error": task_info.error
            })
        
        return {
            "task_history": history_data,
            "total_found": len(history_data),
            "filters": {
                "category": category,
                "status": status_filter
            },
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get task history: {str(e)}"
        )

@router.get("/metrics", summary="Get task metrics")
async def get_task_metrics(
    category: Optional[str] = Query(None, description="Get metrics for specific category"),
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get task execution metrics"""
    try:
        task_category = None
        if category:
            try:
                task_category = TaskCategory(category)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid task category: {category}"
                )
        
        metrics = task_manager.get_task_metrics(task_category)
        
        return {
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get task metrics: {str(e)}"
        )

@router.post("/submit", summary="Submit new task")
async def submit_task(
    request: TaskSubmissionRequest,
    current_user: CurrentUser = Depends(get_current_user)
) -> TaskResponse:
    """Submit a new background task"""
    try:
        # Validate category
        try:
            category = TaskCategory(request.category)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid task category: {request.category}"
            )
        
        # Validate priority
        try:
            priority = TaskPriority(request.priority.upper())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid task priority: {request.priority}"
            )
        
        # Submit task
        task_id = await task_manager.submit_task(
            task_name=request.task_name,
            category=category,
            args=tuple(request.args),
            kwargs=request.kwargs,
            priority=priority,
            max_retries=request.max_retries,
            countdown=request.countdown,
            metadata=request.metadata
        )
        
        return TaskResponse(
            success=True,
            message=f"Task submitted successfully",
            task_id=task_id,
            data={
                "task_name": request.task_name,
                "category": request.category,
                "priority": request.priority
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit task: {str(e)}"
        )

@router.post("/cancel/{task_id}", summary="Cancel task")
async def cancel_task(
    task_id: str,
    current_user: CurrentUser = Depends(get_current_user)
) -> TaskResponse:
    """Cancel a running task"""
    try:
        success = await task_manager.cancel_task(task_id)
        
        return TaskResponse(
            success=success,
            message=f"Task {'cancelled' if success else 'cancellation failed'}",
            task_id=task_id
        )
    except Exception as e:
        logger.error(f"Failed to cancel task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel task: {str(e)}"
        )

@router.post("/retry/{task_id}", summary="Retry failed task")
async def retry_task(
    task_id: str,
    current_user: CurrentUser = Depends(get_current_user)
) -> TaskResponse:
    """Retry a failed task"""
    try:
        success = await task_manager.retry_task(task_id)
        
        return TaskResponse(
            success=success,
            message=f"Task {'retried' if success else 'retry failed'}",
            task_id=task_id
        )
    except Exception as e:
        logger.error(f"Failed to retry task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retry task: {str(e)}"
        )

@router.post("/cleanup", summary="Clean up completed tasks")
async def cleanup_completed_tasks(
    hours_old: int = Query(24, description="Remove tasks completed more than this many hours ago"),
    current_user: CurrentUser = Depends(get_current_user)
) -> TaskResponse:
    """Clean up old completed tasks"""
    try:
        cleaned_count = await task_manager.cleanup_completed_tasks(hours_old)
        
        return TaskResponse(
            success=True,
            message=f"Cleaned up {cleaned_count} completed tasks",
            data={
                "cleaned_count": cleaned_count,
                "hours_old": hours_old
            }
        )
    except Exception as e:
        logger.error(f"Failed to cleanup tasks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cleanup tasks: {str(e)}"
        )

# Convenience endpoints for common task types

@router.post("/ai/query", summary="Submit AI query task")
async def submit_ai_query_task(
    service: str,
    prompt: str,
    current_user: CurrentUser = Depends(get_current_user)
) -> TaskResponse:
    """Submit AI query task"""
    try:
        task_id = await submit_ai_task(
            "koo.tasks.ai_service.process_query",
            service, prompt
        )
        
        return TaskResponse(
            success=True,
            message="AI query task submitted",
            task_id=task_id,
            data={"service": service, "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt}
        )
    except Exception as e:
        logger.error(f"Failed to submit AI query task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit AI query task: {str(e)}"
        )

@router.post("/health-check", summary="Submit health check task")
async def submit_health_check_task(
    current_user: CurrentUser = Depends(get_current_user)
) -> TaskResponse:
    """Submit comprehensive health check task"""
    try:
        task_id = await submit_health_check_task(
            "koo.tasks.health_check.check_all_services"
        )
        
        return TaskResponse(
            success=True,
            message="Health check task submitted",
            task_id=task_id
        )
    except Exception as e:
        logger.error(f"Failed to submit health check task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit health check task: {str(e)}"
        )
