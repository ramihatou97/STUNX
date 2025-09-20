"""
KOO Platform Monitoring API
Comprehensive monitoring endpoints for database and AI services
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from datetime import datetime

from core.dependencies import get_current_user, CurrentUser
from core.database import db_manager, check_database_health
from core.redis_cache import redis_cache
from core.task_manager import task_manager
from services.hybrid_ai_manager import hybrid_ai_manager
from core.ai_error_handling import ai_error_handler

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

# Pydantic models for responses
class DatabaseHealthResponse(BaseModel):
    healthy: bool
    circuit_breaker_state: str
    pool_status: Dict[str, Any]
    last_health_check: Optional[str]
    timestamp: str

class AIServiceHealthResponse(BaseModel):
    service_name: str
    state: str
    health: Dict[str, Any]
    metrics: Dict[str, Any]
    rate_limits: Dict[str, Any]
    timestamp: str

class SystemOverviewResponse(BaseModel):
    database: DatabaseHealthResponse
    ai_services: Dict[str, AIServiceHealthResponse]
    overall_health: str
    timestamp: str

@router.get("/health", summary="Get overall system health")
async def get_system_health(
    current_user: CurrentUser = Depends(get_current_user)
):
    """Get comprehensive system health status"""
    try:
        # Get database health
        db_health = await check_database_health()
        
        # Get AI services health
        ai_health = hybrid_ai_manager.get_all_services_status()
        
        # Calculate overall health
        db_healthy = db_health.get("healthy", False)
        ai_services_healthy = ai_health.get("system_health", {}).get("overall_health") == "healthy"
        
        overall_health = "healthy" if db_healthy and ai_services_healthy else "degraded"
        
        return {
            "overall_health": overall_health,
            "database": db_health,
            "ai_services": ai_health,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system health: {str(e)}"
        )

@router.get("/database", summary="Get database monitoring data")
async def get_database_monitoring(
    current_user: CurrentUser = Depends(get_current_user)
):
    """Get comprehensive database monitoring information"""
    try:
        # Get detailed database status
        detailed_status = await db_manager.get_detailed_status()
        
        return {
            "database_monitoring": detailed_status,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get database monitoring data: {str(e)}"
        )

@router.get("/database/pool-metrics", summary="Get database pool metrics")
async def get_database_pool_metrics(
    current_user: CurrentUser = Depends(get_current_user)
):
    """Get detailed database connection pool metrics"""
    try:
        pool_metrics = db_manager.get_pool_metrics()
        
        return {
            "pool_metrics": pool_metrics,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get pool metrics: {str(e)}"
        )

@router.post("/database/health-check", summary="Perform database health check")
async def perform_database_health_check(
    current_user: CurrentUser = Depends(get_current_user)
):
    """Perform active database health check"""
    try:
        health_result = await db_manager.health_check()
        detailed_status = await db_manager.get_detailed_status()
        
        return {
            "health_check_result": health_result,
            "detailed_status": detailed_status,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database health check failed: {str(e)}"
        )

@router.post("/database/reset-circuit-breaker", summary="Reset database circuit breaker")
async def reset_database_circuit_breaker(
    current_user: CurrentUser = Depends(get_current_user)
):
    """Manually reset database circuit breaker"""
    try:
        db_manager.reset_circuit_breaker()
        
        return {
            "message": "Database circuit breaker reset successfully",
            "new_state": db_manager.circuit_breaker_state.value,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset database circuit breaker: {str(e)}"
        )

@router.get("/ai-services", summary="Get AI services monitoring data")
async def get_ai_services_monitoring(
    current_user: CurrentUser = Depends(get_current_user)
):
    """Get comprehensive AI services monitoring information"""
    try:
        # Get all services status from hybrid manager
        services_status = hybrid_ai_manager.get_all_services_status()
        
        # Get error handler metrics
        error_metrics = ai_error_handler.get_all_services_status()
        
        return {
            "ai_services_monitoring": {
                "services_status": services_status,
                "error_metrics": error_metrics
            },
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get AI services monitoring data: {str(e)}"
        )

@router.get("/ai-services/{service}/metrics", summary="Get specific AI service metrics")
async def get_ai_service_metrics(
    service: str,
    current_user: CurrentUser = Depends(get_current_user)
):
    """Get detailed metrics for a specific AI service"""
    try:
        # Get service status from hybrid manager
        service_status = hybrid_ai_manager.get_enhanced_service_status(service)
        
        if "error" in service_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=service_status["error"]
            )
        
        return {
            "service_metrics": service_status,
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get service metrics: {str(e)}"
        )

@router.get("/alerts", summary="Get system alerts")
async def get_system_alerts(
    current_user: CurrentUser = Depends(get_current_user)
):
    """Get current system alerts and warnings"""
    try:
        alerts = []
        
        # Check database alerts
        db_status = await db_manager.get_detailed_status()
        circuit_breaker_state = db_status.get("circuit_breaker", {}).get("state")
        
        if circuit_breaker_state == "circuit_open":
            alerts.append({
                "type": "critical",
                "component": "database",
                "message": "Database circuit breaker is open",
                "timestamp": datetime.now().isoformat()
            })
        elif circuit_breaker_state in ["unhealthy", "degraded"]:
            alerts.append({
                "type": "warning",
                "component": "database",
                "message": f"Database is in {circuit_breaker_state} state",
                "timestamp": datetime.now().isoformat()
            })
        
        # Check AI services alerts
        ai_status = ai_error_handler.get_all_services_status()
        for service_name, service_data in ai_status.items():
            state = service_data.get("state", "unknown")
            
            if state == "circuit_open":
                alerts.append({
                    "type": "critical",
                    "component": f"ai_service_{service_name}",
                    "message": f"AI service {service_name} circuit breaker is open",
                    "timestamp": datetime.now().isoformat()
                })
            elif state in ["unhealthy", "degraded", "rate_limited"]:
                alerts.append({
                    "type": "warning",
                    "component": f"ai_service_{service_name}",
                    "message": f"AI service {service_name} is in {state} state",
                    "timestamp": datetime.now().isoformat()
                })
        
        return {
            "alerts": alerts,
            "alert_count": len(alerts),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system alerts: {str(e)}"
        )

@router.get("/dashboard", summary="Get monitoring dashboard data")
async def get_monitoring_dashboard(
    current_user: CurrentUser = Depends(get_current_user)
):
    """Get comprehensive monitoring dashboard data"""
    try:
        # Get all monitoring data
        db_health = await check_database_health()
        db_metrics = db_manager.get_pool_metrics()
        ai_health = hybrid_ai_manager.get_all_services_status()
        ai_metrics = ai_error_handler.get_all_services_status()
        
        # Get alerts
        alerts_response = await get_system_alerts(current_user)
        alerts = alerts_response["alerts"]
        
        # Calculate summary statistics
        total_services = len(hybrid_ai_manager.service_configs)
        healthy_ai_services = sum(1 for service_data in ai_metrics.values() 
                                 if service_data.get("state") == "healthy")
        
        dashboard_data = {
            "summary": {
                "overall_health": "healthy" if db_health.get("healthy") and healthy_ai_services >= total_services * 0.8 else "degraded",
                "database_healthy": db_health.get("healthy", False),
                "ai_services_healthy": f"{healthy_ai_services}/{total_services}",
                "active_alerts": len(alerts),
                "circuit_breakers_open": sum(1 for service_data in ai_metrics.values() 
                                            if service_data.get("state") == "circuit_open")
            },
            "database": {
                "health": db_health,
                "metrics": db_metrics
            },
            "ai_services": {
                "health": ai_health,
                "metrics": ai_metrics
            },
            "alerts": alerts,
            "timestamp": datetime.now().isoformat()
        }
        
        return dashboard_data
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get dashboard data: {str(e)}"
        )

@router.get("/cache", summary="Get cache monitoring data")
async def get_cache_monitoring(
    current_user: CurrentUser = Depends(get_current_user)
):
    """Get comprehensive cache monitoring information"""
    try:
        # Get cache health
        cache_health = await redis_cache.health_check()

        # Get cache metrics
        cache_metrics = redis_cache.get_metrics()

        return {
            "health": cache_health,
            "metrics": cache_metrics,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cache monitoring data: {str(e)}"
        )

@router.get("/tasks", summary="Get task monitoring data")
async def get_task_monitoring(
    current_user: CurrentUser = Depends(get_current_user)
):
    """Get comprehensive task monitoring information"""
    try:
        # Get task metrics
        task_metrics = task_manager.get_task_metrics()

        # Get active tasks summary
        active_tasks = await task_manager.get_active_tasks()
        active_summary = {
            "total_active": len(active_tasks),
            "by_category": {},
            "by_status": {}
        }

        for task in active_tasks:
            category = task.category.value
            status = task.status.value

            active_summary["by_category"][category] = active_summary["by_category"].get(category, 0) + 1
            active_summary["by_status"][status] = active_summary["by_status"].get(status, 0) + 1

        return {
            "metrics": task_metrics,
            "active_tasks": active_summary,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get task monitoring data: {str(e)}"
        )

@router.get("/performance", summary="Get system performance metrics")
async def get_performance_metrics(
    current_user: CurrentUser = Depends(get_current_user)
):
    """Get comprehensive system performance metrics"""
    try:
        # Get database performance
        db_metrics = db_manager.get_pool_metrics()

        # Get cache performance
        cache_metrics = redis_cache.get_metrics()

        # Get task performance
        task_metrics = task_manager.get_task_metrics()

        # Calculate overall performance score
        cache_hit_rate = cache_metrics.get("global", {}).get("hit_rate", 0)
        task_success_rate = task_metrics.get("global", {}).get("success_rate", 0)
        db_health_score = 1.0 if db_metrics.get("healthy", False) else 0.0

        overall_score = (cache_hit_rate + task_success_rate + db_health_score) / 3

        return {
            "overall_performance_score": round(overall_score, 3),
            "database": {
                "pool_metrics": db_metrics,
                "health_score": db_health_score
            },
            "cache": {
                "hit_rate": cache_hit_rate,
                "avg_response_time_ms": cache_metrics.get("global", {}).get("avg_response_time_ms", 0),
                "total_operations": cache_metrics.get("global", {}).get("hits", 0) + cache_metrics.get("global", {}).get("misses", 0)
            },
            "tasks": {
                "success_rate": task_success_rate,
                "avg_execution_time": task_metrics.get("global", {}).get("avg_execution_time", 0),
                "total_tasks": task_metrics.get("global", {}).get("total_tasks", 0)
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance metrics: {str(e)}"
        )
