"""
KOO Platform Admin API - API Key Management
Endpoints for managing and validating API keys
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from datetime import datetime

from core.dependencies import get_current_user, CurrentUser
from core.api_key_manager import (
    api_key_manager,
    APIProvider,
    check_all_api_health,
    validate_provider_key
)

router = APIRouter(prefix="/admin", tags=["admin"])

# Pydantic models
class APIKeyRequest(BaseModel):
    provider: str
    key: str
    endpoint_url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None

class APIKeyResponse(BaseModel):
    provider: str
    is_configured: bool
    is_valid: bool
    last_validated: Optional[datetime] = None
    usage_count: int = 0
    rate_limit_remaining: Optional[int] = None
    rate_limit_reset: Optional[datetime] = None
    endpoint_url: Optional[str] = None

class HealthCheckResponse(BaseModel):
    timestamp: datetime
    total_providers: int
    valid_providers: int
    invalid_providers: int
    providers: Dict[str, Dict[str, Any]]

@router.get("/info", summary="Get admin user information")
async def get_admin_info(current_user: CurrentUser = Depends(get_current_user)):
    """Get current admin user information"""
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "role": current_user.role,
        "is_active": current_user.is_active,
        "is_verified": current_user.is_verified
    }

@router.get("/api-keys", response_model=List[APIKeyResponse], summary="Get all API key configurations")
async def get_api_keys(current_user: CurrentUser = Depends(get_current_user)):
    """Get status of all configured API keys"""
    api_keys = []

    for provider in APIProvider:
        config = api_key_manager.api_configs.get(provider)

        if config:
            api_keys.append(APIKeyResponse(
                provider=provider.value,
                is_configured=True,
                is_valid=config.is_valid,
                last_validated=config.last_validated,
                usage_count=config.usage_count,
                rate_limit_remaining=config.rate_limit_remaining,
                rate_limit_reset=config.rate_limit_reset,
                endpoint_url=config.endpoint_url
            ))
        else:
            api_keys.append(APIKeyResponse(
                provider=provider.value,
                is_configured=False,
                is_valid=False
            ))

    return api_keys

@router.post("/api-keys", summary="Add or update an API key")
async def add_api_key(
    request: APIKeyRequest,
    current_user: CurrentUser = Depends(get_current_user)
):
    """Add or update an API key for a provider"""
    try:
        # Validate provider
        try:
            provider = APIProvider(request.provider.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid provider: {request.provider}. Valid providers: {[p.value for p in APIProvider]}"
            )

        # Add the API key
        success = api_key_manager.add_api_key(
            provider=provider,
            key=request.key,
            endpoint_url=request.endpoint_url,
            headers=request.headers,
            encrypt=True
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to add API key for {provider.value}"
            )

        # Validate the key
        is_valid, message = await validate_provider_key(provider)

        return {
            "message": f"API key for {provider.value} added successfully",
            "provider": provider.value,
            "is_valid": is_valid,
            "validation_message": message
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add API key: {str(e)}"
        )

@router.delete("/api-keys/{provider}", summary="Remove an API key")
async def remove_api_key(
    provider: str,
    current_user: CurrentUser = Depends(get_current_user)
):
    """Remove an API key for a provider"""
    try:
        # Validate provider
        try:
            api_provider = APIProvider(provider.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid provider: {provider}"
            )

        success = api_key_manager.remove_api_key(api_provider)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No API key found for {provider}"
            )

        return {
            "message": f"API key for {provider} removed successfully",
            "provider": provider
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to remove API key: {str(e)}"
        )

@router.post("/api-keys/{provider}/validate", summary="Validate a specific API key")
async def validate_api_key(
    provider: str,
    current_user: CurrentUser = Depends(get_current_user)
):
    """Validate a specific API key"""
    try:
        # Validate provider
        try:
            api_provider = APIProvider(provider.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid provider: {provider}"
            )

        is_valid, message = await validate_provider_key(api_provider)

        return {
            "provider": provider,
            "is_valid": is_valid,
            "message": message,
            "validated_at": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate API key: {str(e)}"
        )

@router.post("/api-keys/validate-all", summary="Validate all API keys")
async def validate_all_api_keys(current_user: CurrentUser = Depends(get_current_user)):
    """Validate all configured API keys"""
    try:
        results = await api_key_manager.validate_all_keys()

        validation_results = {}
        for provider, (is_valid, message) in results.items():
            validation_results[provider.value] = {
                "is_valid": is_valid,
                "message": message,
                "validated_at": datetime.now().isoformat()
            }

        return {
            "total_providers": len(results),
            "valid_count": sum(1 for result in results.values() if result[0]),
            "invalid_count": sum(1 for result in results.values() if not result[0]),
            "results": validation_results
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate API keys: {str(e)}"
        )

@router.get("/health", response_model=HealthCheckResponse, summary="Health check for all API services")
async def admin_health_check(current_user: CurrentUser = Depends(get_current_user)):
    """Comprehensive health check of all API services"""
    try:
        health_data = await check_all_api_health()
        return HealthCheckResponse(**health_data)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )

@router.get("/api-keys/export", summary="Export API key configuration")
async def export_api_configuration(
    include_keys: bool = False,
    current_user: CurrentUser = Depends(get_current_user)
):
    """Export API key configuration for backup (optionally including encrypted keys)"""
    try:
        export_data = api_key_manager.export_configuration(include_keys=include_keys)
        return export_data

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export configuration: {str(e)}"
        )

@router.get("/api-keys/usage", summary="Get API usage statistics")
async def get_api_usage_stats(current_user: CurrentUser = Depends(get_current_user)):
    """Get usage statistics for all API providers"""
    try:
        usage_stats = {}

        for provider, config in api_key_manager.api_configs.items():
            usage_stats[provider.value] = {
                "usage_count": config.usage_count,
                "is_valid": config.is_valid,
                "last_validated": config.last_validated.isoformat() if config.last_validated else None,
                "rate_limit_remaining": config.rate_limit_remaining,
                "rate_limit_reset": config.rate_limit_reset.isoformat() if config.rate_limit_reset else None
            }

        return {
            "timestamp": datetime.now().isoformat(),
            "total_usage": sum(config.usage_count for config in api_key_manager.api_configs.values()),
            "providers": usage_stats
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get usage statistics: {str(e)}"
        )