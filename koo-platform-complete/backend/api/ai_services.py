"""
KOO Platform AI Services API
Endpoints for hybrid AI service management and queries
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from datetime import datetime

from ..core.dependencies import get_current_user, CurrentUser
from ..services.hybrid_ai_manager import (
    hybrid_ai_manager,
    query_ai,
    query_multiple_ai,
    get_ai_service_health,
    get_all_ai_services_health,
    perform_ai_health_check,
    reset_ai_service_errors,
    reset_all_ai_services_errors
)
from ..core.exceptions import ExternalServiceError
from ..core.ai_error_handling import ai_error_handler

router = APIRouter(prefix="/ai", tags=["ai-services"])

# Pydantic models
class AIQueryRequest(BaseModel):
    service: str
    prompt: str
    max_tokens: Optional[int] = 2000
    temperature: Optional[float] = 0.7
    use_web_fallback: Optional[bool] = True

class AIQueryResponse(BaseModel):
    service: str
    prompt: str
    response: str
    method_used: str  # "api" or "web"
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    timestamp: datetime

class BatchQueryRequest(BaseModel):
    queries: List[Dict[str, Any]]

class BatchQueryResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_queries: int
    successful_queries: int
    failed_queries: int
    total_cost: float

class ServiceStatus(BaseModel):
    service: str
    api_available: bool
    web_available: bool
    current_method: str
    daily_budget: float
    budget_used: float
    budget_remaining: float
    api_calls_today: int
    web_calls_today: int
    last_used: Optional[datetime] = None

class UsageStatsResponse(BaseModel):
    services: Dict[str, ServiceStatus]
    total_api_calls: int
    total_web_calls: int
    total_cost: float
    last_updated: datetime

class ServiceHealthResponse(BaseModel):
    service_name: str
    state: str
    health: Dict[str, Any]
    metrics: Dict[str, Any]
    rate_limits: Dict[str, Any]
    configuration: Dict[str, Any]
    last_updated: str

class SystemHealthResponse(BaseModel):
    overall_health: str
    health_percentage: float
    healthy_services: int
    total_services: int
    services: Dict[str, ServiceHealthResponse]
    timestamp: str

@router.post("/query", response_model=AIQueryResponse, summary="Query AI service")
async def query_ai_service(
    request: AIQueryRequest,
    current_user: CurrentUser = Depends(get_current_user)
):
    """Query a specific AI service with hybrid access"""
    try:
        # Start timing
        start_time = datetime.now()

        # Query the AI service
        response = await query_ai(
            service=request.service,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )

        # Determine method used (simplified - could be enhanced)
        stats = hybrid_ai_manager.get_usage_stats()
        service_stats = stats.get(request.service, {})
        method_used = "api" if service_stats.get("api_calls", 0) > 0 else "web"

        return AIQueryResponse(
            service=request.service,
            prompt=request.prompt,
            response=response,
            method_used=method_used,
            tokens_used=len(response.split()) * 1.3,  # Rough estimation
            cost=service_stats.get("api_cost", 0.0),
            timestamp=start_time
        )

    except ExternalServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"AI service error: {e.message}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query AI service: {str(e)}"
        )

@router.post("/batch-query", response_model=BatchQueryResponse, summary="Batch query multiple AI services")
async def batch_query_ai_services(
    request: BatchQueryRequest,
    current_user: CurrentUser = Depends(get_current_user)
):
    """Query multiple AI services in batch"""
    try:
        results = await query_multiple_ai(request.queries)

        successful_queries = sum(1 for r in results if r.get("success", False))
        failed_queries = len(results) - successful_queries

        # Calculate total cost
        stats = hybrid_ai_manager.get_usage_stats()
        total_cost = sum(service_stats.get("api_cost", 0.0) for service_stats in stats.values())

        return BatchQueryResponse(
            results=results,
            total_queries=len(request.queries),
            successful_queries=successful_queries,
            failed_queries=failed_queries,
            total_cost=total_cost
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch query failed: {str(e)}"
        )

@router.get("/services/status", response_model=UsageStatsResponse, summary="Get AI services status")
async def get_ai_services_status(current_user: CurrentUser = Depends(get_current_user)):
    """Get status and usage statistics for all AI services"""
    try:
        stats = hybrid_ai_manager.get_usage_stats()

        services = {}
        total_api_calls = 0
        total_web_calls = 0
        total_cost = 0.0

        for service_name, service_stats in stats.items():
            config = hybrid_ai_manager.service_configs.get(service_name, {})

            api_calls = service_stats.get("api_calls", 0)
            web_calls = service_stats.get("web_calls", 0)
            cost = service_stats.get("api_cost", 0.0)
            daily_budget = service_stats.get("daily_budget", 0.0)
            budget_used = service_stats.get("daily_budget_used", 0.0)

            services[service_name] = ServiceStatus(
                service=service_name,
                api_available=bool(config.get("api_available", False)),
                web_available=bool(config.get("web_available", True)),
                current_method="hybrid",  # Simplified
                daily_budget=daily_budget,
                budget_used=budget_used,
                budget_remaining=max(0, daily_budget - budget_used),
                api_calls_today=api_calls,
                web_calls_today=web_calls
            )

            total_api_calls += api_calls
            total_web_calls += web_calls
            total_cost += cost

        return UsageStatsResponse(
            services=services,
            total_api_calls=total_api_calls,
            total_web_calls=total_web_calls,
            total_cost=total_cost,
            last_updated=datetime.now()
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get services status: {str(e)}"
        )

@router.post("/services/{service}/test", summary="Test AI service")
async def test_ai_service(
    service: str,
    current_user: CurrentUser = Depends(get_current_user)
):
    """Test connectivity to a specific AI service"""
    try:
        test_prompt = "Hello, this is a test message. Please respond with 'Test successful.'"

        response = await query_ai(
            service=service,
            prompt=test_prompt,
            max_tokens=50
        )

        return {
            "service": service,
            "test_successful": True,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }

    except ExternalServiceError as e:
        return {
            "service": service,
            "test_successful": False,
            "error": e.message,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Test failed: {str(e)}"
        )

@router.post("/services/reset-usage", summary="Reset daily usage statistics")
async def reset_usage_stats(current_user: CurrentUser = Depends(get_current_user)):
    """Reset daily usage statistics for all services"""
    try:
        hybrid_ai_manager.reset_daily_stats()

        return {
            "message": "Daily usage statistics reset successfully",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset usage stats: {str(e)}"
        )

@router.get("/services/{service}/config", summary="Get service configuration")
async def get_service_config(
    service: str,
    current_user: CurrentUser = Depends(get_current_user)
):
    """Get configuration for a specific AI service"""
    try:
        if service not in hybrid_ai_manager.service_configs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Service '{service}' not found"
            )

        config = hybrid_ai_manager.service_configs[service]
        stats = hybrid_ai_manager.usage_stats.get(service, {})

        return {
            "service": service,
            "access_method": config.access_method.value if hasattr(config.access_method, 'value') else str(config.access_method),
            "api_available": config.api_available,
            "web_available": config.web_available,
            "daily_budget": config.daily_budget,
            "cost_per_1k_tokens": config.cost_per_1k_tokens,
            "max_tokens_per_request": config.max_tokens_per_request,
            "usage_stats": {
                "api_calls": getattr(stats, 'api_calls', 0),
                "web_calls": getattr(stats, 'web_calls', 0),
                "api_cost": getattr(stats, 'api_cost', 0.0),
                "tokens_used": getattr(stats, 'tokens_used', 0),
                "daily_budget_used": getattr(stats, 'daily_budget_used', 0.0)
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get service config: {str(e)}"
        )

@router.post("/services/initialize", summary="Initialize browser session")
async def initialize_browser_session(
    background_tasks: BackgroundTasks,
    current_user: CurrentUser = Depends(get_current_user)
):
    """Initialize browser session for web automation"""
    try:
        # Initialize in background to avoid blocking
        background_tasks.add_task(hybrid_ai_manager.initialize_browser)

        return {
            "message": "Browser session initialization started",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize browser: {str(e)}"
        )

@router.post("/services/cleanup", summary="Cleanup browser session")
async def cleanup_browser_session(current_user: CurrentUser = Depends(get_current_user)):
    """Cleanup browser session and resources"""
    try:
        await hybrid_ai_manager.close_browser()

        return {
            "message": "Browser session cleaned up successfully",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cleanup browser: {str(e)}"
        )

@router.get("/services/health", summary="Get comprehensive AI services health")
async def get_services_health(
    current_user: CurrentUser = Depends(get_current_user)
):
    """Get comprehensive health status for all AI services"""
    try:
        health_status = get_all_ai_services_health()
        return health_status
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get services health: {str(e)}"
        )

@router.get("/services/{service}/health", summary="Get specific service health")
async def get_service_health(
    service: str,
    current_user: CurrentUser = Depends(get_current_user)
):
    """Get comprehensive health status for a specific AI service"""
    try:
        health_status = get_ai_service_health(service)
        if "error" in health_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=health_status["error"]
            )
        return health_status
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get service health: {str(e)}"
        )

@router.post("/services/health-check", summary="Perform active health check")
async def perform_health_check(
    service: Optional[str] = None,
    current_user: CurrentUser = Depends(get_current_user)
):
    """Perform active health check for AI services"""
    try:
        health_check_result = await perform_ai_health_check(service)
        return health_check_result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )

@router.post("/services/{service}/reset-errors", summary="Reset service errors")
async def reset_service_errors(
    service: str,
    current_user: CurrentUser = Depends(get_current_user)
):
    """Reset error handling state for a specific AI service"""
    try:
        success = reset_ai_service_errors(service)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Service {service} not found or reset failed"
            )

        return {
            "message": f"Error state reset for {service}",
            "service": service,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset service errors: {str(e)}"
        )

@router.post("/services/reset-all-errors", summary="Reset all services errors")
async def reset_all_services_errors_endpoint(
    current_user: CurrentUser = Depends(get_current_user)
):
    """Reset error handling state for all AI services"""
    try:
        results = reset_all_ai_services_errors()

        return {
            "message": "Error state reset for all services",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset all service errors: {str(e)}"
        )

@router.get("/services/metrics", summary="Get detailed service metrics")
async def get_service_metrics(
    current_user: CurrentUser = Depends(get_current_user)
):
    """Get detailed metrics for all AI services"""
    try:
        # Get metrics from error handler
        error_handler_metrics = ai_error_handler.get_all_services_status()

        # Get usage stats from hybrid manager
        usage_stats = hybrid_ai_manager.get_usage_stats()

        # Combine metrics
        combined_metrics = {
            "error_handling_metrics": error_handler_metrics,
            "usage_statistics": usage_stats,
            "system_health": hybrid_ai_manager._get_system_health(),
            "timestamp": datetime.now().isoformat()
        }

        return combined_metrics
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get service metrics: {str(e)}"
        )