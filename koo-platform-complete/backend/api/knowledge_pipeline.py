"""
KOO Platform Knowledge Pipeline API
Endpoints for managing research workflows and knowledge synthesis
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime

from ..core.dependencies import get_current_user, CurrentUser
from ..services.knowledge_integration_pipeline import (
    knowledge_pipeline,
    start_research_pipeline,
    get_pipeline_status,
    ResearchQuery,
    PipelineStage
)
from ..core.exceptions import ExternalServiceError, ValidationError

router = APIRouter(prefix="/knowledge", tags=["knowledge-pipeline"])

# Pydantic models
class ResearchQueryRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=500, description="Research topic")
    neurosurgical_focus: List[str] = Field(..., min_items=1, description="Neurosurgical focus areas")
    mesh_terms: Optional[List[str]] = Field(default=[], description="Additional MeSH terms")
    date_range: Optional[List[str]] = Field(default=None, description="Date range [start, end] in YYYY-MM-DD format")
    max_results: int = Field(default=50, ge=10, le=200, description="Maximum results per source")
    quality_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum quality threshold")

class PipelineExecutionResponse(BaseModel):
    execution_id: str
    message: str
    estimated_completion_time: str

class PipelineStatusResponse(BaseModel):
    execution_id: str
    stage: str
    is_active: bool
    results_count: int
    started_at: str
    completed_at: Optional[str] = None
    success: Optional[bool] = None
    error: Optional[str] = None
    progress_percentage: int

class KnowledgeSynthesisResponse(BaseModel):
    topic: str
    executive_summary: str
    key_findings: List[str]
    clinical_implications: List[str]
    surgical_techniques: List[str]
    evidence_quality: str
    conflicting_findings: List[str]
    research_gaps: List[str]
    recommendations: List[str]
    sources_used: List[str]
    confidence_score: float
    last_updated: str

class ActiveExecutionsResponse(BaseModel):
    active_count: int
    executions: List[Dict[str, Any]]

@router.post("/pipeline/start", response_model=PipelineExecutionResponse, summary="Start knowledge research pipeline")
async def start_knowledge_pipeline(
    request: ResearchQueryRequest,
    background_tasks: BackgroundTasks,
    current_user: CurrentUser = Depends(get_current_user)
):
    """Start a new knowledge integration pipeline execution"""
    try:
        # Validate neurosurgical focus areas
        valid_focus_areas = [
            "brain tumors", "spinal surgery", "vascular neurosurgery", "epilepsy surgery",
            "functional neurosurgery", "pediatric neurosurgery", "trauma neurosurgery",
            "minimally invasive surgery", "stereotactic surgery", "neuromonitoring",
            "brain mapping", "cranial surgery", "tumor resection", "deep brain stimulation"
        ]

        for focus in request.neurosurgical_focus:
            if not any(valid_area.lower() in focus.lower() for valid_area in valid_focus_areas):
                logger.warning(f"Non-standard focus area: {focus}")

        # Start pipeline in background
        execution_id = await start_research_pipeline(
            topic=request.topic,
            neurosurgical_focus=request.neurosurgical_focus,
            mesh_terms=request.mesh_terms,
            max_results=request.max_results
        )

        return PipelineExecutionResponse(
            execution_id=execution_id,
            message="Knowledge integration pipeline started successfully",
            estimated_completion_time="5-15 minutes depending on complexity"
        )

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {e.message}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start pipeline: {str(e)}"
        )

@router.get("/pipeline/{execution_id}/status", response_model=PipelineStatusResponse, summary="Get pipeline status")
async def get_pipeline_execution_status(
    execution_id: str,
    current_user: CurrentUser = Depends(get_current_user)
):
    """Get the current status of a pipeline execution"""
    try:
        status_info = await get_pipeline_status(execution_id)

        if not status_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pipeline execution {execution_id} not found"
            )

        # Calculate progress percentage
        stage_progress = {
            PipelineStage.INITIATED.value: 5,
            PipelineStage.RESEARCH_GATHERING.value: 25,
            PipelineStage.ANALYSIS.value: 50,
            PipelineStage.SYNTHESIS.value: 75,
            PipelineStage.QUALITY_CHECK.value: 90,
            PipelineStage.CHAPTER_UPDATE.value: 95,
            PipelineStage.COMPLETED.value: 100,
            PipelineStage.FAILED.value: 0
        }

        return PipelineStatusResponse(
            execution_id=execution_id,
            stage=status_info["stage"],
            is_active=status_info["is_active"],
            results_count=status_info["results_count"],
            started_at=status_info["started_at"],
            completed_at=status_info.get("completed_at"),
            success=status_info.get("success"),
            error=status_info.get("error"),
            progress_percentage=stage_progress.get(status_info["stage"], 0)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get pipeline status: {str(e)}"
        )

@router.get("/pipeline/{execution_id}/synthesis", response_model=KnowledgeSynthesisResponse, summary="Get knowledge synthesis")
async def get_knowledge_synthesis(
    execution_id: str,
    current_user: CurrentUser = Depends(get_current_user)
):
    """Get the knowledge synthesis results from a completed pipeline"""
    try:
        status_info = await get_pipeline_status(execution_id)

        if not status_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pipeline execution {execution_id} not found"
            )

        if status_info["stage"] != PipelineStage.COMPLETED.value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Pipeline not completed. Current stage: {status_info['stage']}"
            )

        synthesis = status_info.get("synthesis")
        if not synthesis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No synthesis results available"
            )

        return KnowledgeSynthesisResponse(**synthesis)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get synthesis: {str(e)}"
        )

@router.get("/pipeline/active", response_model=ActiveExecutionsResponse, summary="Get active pipeline executions")
async def get_active_pipeline_executions(current_user: CurrentUser = Depends(get_current_user)):
    """Get all currently active pipeline executions"""
    try:
        active_executions = knowledge_pipeline.get_active_executions()

        return ActiveExecutionsResponse(
            active_count=len(active_executions),
            executions=active_executions
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get active executions: {str(e)}"
        )

@router.post("/pipeline/{execution_id}/cancel", summary="Cancel pipeline execution")
async def cancel_pipeline_execution(
    execution_id: str,
    current_user: CurrentUser = Depends(get_current_user)
):
    """Cancel an active pipeline execution"""
    try:
        if execution_id not in knowledge_pipeline.active_executions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Active pipeline execution {execution_id} not found"
            )

        # Move to completed with failed status
        execution = knowledge_pipeline.active_executions[execution_id]
        execution.stage = PipelineStage.FAILED
        execution.error_message = "Cancelled by user"
        execution.completed_at = datetime.now()

        knowledge_pipeline.completed_executions.append(
            knowledge_pipeline.active_executions.pop(execution_id)
        )

        return {
            "message": f"Pipeline execution {execution_id} cancelled successfully",
            "execution_id": execution_id
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel pipeline: {str(e)}"
        )

@router.get("/pipeline/history", summary="Get pipeline execution history")
async def get_pipeline_history(
    limit: int = 20,
    offset: int = 0,
    current_user: CurrentUser = Depends(get_current_user)
):
    """Get history of completed pipeline executions"""
    try:
        completed = knowledge_pipeline.completed_executions

        # Sort by completion time (most recent first)
        sorted_executions = sorted(
            completed,
            key=lambda x: x.completed_at or x.started_at,
            reverse=True
        )

        # Apply pagination
        paginated = sorted_executions[offset:offset + limit]

        history = []
        for execution in paginated:
            history.append({
                "execution_id": execution.execution_id,
                "topic": execution.query.topic,
                "stage": execution.stage.value,
                "started_at": execution.started_at.isoformat(),
                "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                "success": execution.stage == PipelineStage.COMPLETED,
                "results_count": len(execution.results),
                "has_synthesis": execution.synthesis is not None,
                "error": execution.error_message
            })

        return {
            "total_count": len(completed),
            "returned_count": len(history),
            "offset": offset,
            "limit": limit,
            "history": history
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get pipeline history: {str(e)}"
        )

@router.get("/pipeline/metrics", summary="Get pipeline performance metrics")
async def get_pipeline_metrics(current_user: CurrentUser = Depends(get_current_user)):
    """Get performance metrics for the knowledge pipeline"""
    try:
        completed = knowledge_pipeline.completed_executions
        active = knowledge_pipeline.active_executions

        successful_executions = [e for e in completed if e.stage == PipelineStage.COMPLETED]
        failed_executions = [e for e in completed if e.stage == PipelineStage.FAILED]

        # Calculate average execution time for successful runs
        avg_execution_time = 0
        if successful_executions:
            total_time = sum(
                (e.completed_at - e.started_at).total_seconds()
                for e in successful_executions
                if e.completed_at
            )
            avg_execution_time = total_time / len(successful_executions)

        # Calculate average results per execution
        avg_results = 0
        if completed:
            avg_results = sum(len(e.results) for e in completed) / len(completed)

        return {
            "total_executions": len(completed),
            "successful_executions": len(successful_executions),
            "failed_executions": len(failed_executions),
            "active_executions": len(active),
            "success_rate": len(successful_executions) / len(completed) if completed else 0,
            "average_execution_time_seconds": avg_execution_time,
            "average_results_per_execution": avg_results,
            "metrics_updated_at": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get pipeline metrics: {str(e)}"
        )

@router.post("/pipeline/templates", summary="Create research query template")
async def create_research_template(
    name: str,
    template: ResearchQueryRequest,
    current_user: CurrentUser = Depends(get_current_user)
):
    """Save a research query template for reuse"""
    try:
        # This would integrate with your database
        # For now, we'll just return success
        return {
            "message": f"Research template '{name}' created successfully",
            "template_name": name,
            "template": template.dict()
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create template: {str(e)}"
        )

@router.get("/pipeline/templates", summary="Get research query templates")
async def get_research_templates(current_user: CurrentUser = Depends(get_current_user)):
    """Get saved research query templates"""
    try:
        # Predefined templates for common neurosurgical topics
        templates = [
            {
                "name": "Brain Tumor Research",
                "template": {
                    "topic": "Brain tumor treatment and surgical outcomes",
                    "neurosurgical_focus": ["brain tumors", "tumor resection", "surgical outcomes"],
                    "mesh_terms": ["Brain Neoplasms", "Neurosurgical Procedures", "Treatment Outcome"],
                    "max_results": 75
                }
            },
            {
                "name": "Minimally Invasive Techniques",
                "template": {
                    "topic": "Minimally invasive neurosurgical techniques",
                    "neurosurgical_focus": ["minimally invasive surgery", "endoscopic surgery"],
                    "mesh_terms": ["Minimally Invasive Surgical Procedures", "Endoscopy", "Neurosurgery"],
                    "max_results": 50
                }
            },
            {
                "name": "Spinal Surgery Innovations",
                "template": {
                    "topic": "Spinal surgery techniques and outcomes",
                    "neurosurgical_focus": ["spinal surgery", "spine fusion", "spinal instrumentation"],
                    "mesh_terms": ["Spinal Fusion", "Spinal Diseases", "Orthopedic Procedures"],
                    "max_results": 60
                }
            }
        ]

        return {
            "templates": templates,
            "count": len(templates)
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get templates: {str(e)}"
        )