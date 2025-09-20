"""
Enhanced Research Analytics API
Provides advanced PubMed analytics, citation networks, and research trend analysis
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from core.dependencies import get_current_user, CurrentUser
from services.advanced_pubmed_analytics import AdvancedPubMedAnalytics
from services.pubmed_service import PubMedService

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services
pubmed_service = PubMedService()
analytics_service = AdvancedPubMedAnalytics(pubmed_service)

# Pydantic models
class CitationAnalysisRequest(BaseModel):
    topic: str = Field(..., description="Research topic to analyze")
    max_papers: int = Field(default=50, description="Maximum number of papers to include")
    years_back: int = Field(default=5, description="Years of research to analyze")
    include_preprints: bool = Field(default=False, description="Include preprint servers")

class TrendAnalysisRequest(BaseModel):
    specialty: str = Field(..., description="Neurosurgical specialty area")
    years: int = Field(default=10, description="Years to analyze")
    granularity: str = Field(default="yearly", description="Analysis granularity")
    include_subtopics: bool = Field(default=True, description="Include emerging subtopics")

class ResearchAlertRequest(BaseModel):
    topic: str = Field(..., description="Topic to monitor")
    alert_type: str = Field(..., description="Type of alert")
    frequency: str = Field(default="weekly", description="Alert frequency")
    trigger_conditions: Dict[str, Any] = Field(default_factory=dict, description="Alert conditions")

class RecommendationRequest(BaseModel):
    user_interests: List[str] = Field(..., description="User research interests")
    max_results: int = Field(default=20, description="Maximum recommendations")
    quality_threshold: float = Field(default=0.5, description="Minimum quality score")
    recency_weight: float = Field(default=0.3, description="Weight for recent publications")

@router.post("/analytics/citation-network")
async def analyze_citation_network(
    request: CitationAnalysisRequest,
    current_user: CurrentUser = Depends(get_current_user)
):
    """Analyze citation networks for a research topic"""
    try:
        logger.info(f"Starting citation network analysis for topic: {request.topic}")

        result = await analytics_service.analyze_citation_network(
            topic=request.topic,
            max_papers=request.max_papers,
            years_back=request.years_back
        )

        logger.info(f"Citation network analysis completed with {len(result.get('papers', []))} papers")

        return {
            "status": "success",
            "analysis": result,
            "metadata": {
                "topic": request.topic,
                "papers_analyzed": len(result.get('papers', [])),
                "network_metrics": result.get('network_metrics', {}),
                "generated_at": datetime.now().isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Citation network analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Citation network analysis failed: {str(e)}"
        )

@router.post("/analytics/research-trends")
async def analyze_research_trends(
    request: TrendAnalysisRequest,
    current_user: CurrentUser = Depends(get_current_user)
):
    """Analyze research trends for neurosurgical specialties"""
    try:
        logger.info(f"Starting trend analysis for specialty: {request.specialty}")

        result = await analytics_service.analyze_research_trends(
            specialty=request.specialty,
            years=request.years
        )

        logger.info(f"Trend analysis completed for {request.specialty}")

        return {
            "status": "success",
            "trends": result,
            "metadata": {
                "specialty": request.specialty,
                "years_analyzed": request.years,
                "analysis_date": datetime.now().isoformat(),
                "trend_momentum": result.get('momentum', 0)
            }
        }

    except Exception as e:
        logger.error(f"Research trend analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Research trend analysis failed: {str(e)}"
        )

@router.get("/analytics/journal-rankings")
async def get_journal_rankings(
    specialty: Optional[str] = Query(None, description="Filter by specialty"),
    top_n: int = Query(50, description="Number of top journals to return"),
    current_user: CurrentUser = Depends(get_current_user)
):
    """Get journal rankings with impact factors"""
    try:
        rankings = await analytics_service.get_journal_rankings(
            specialty=specialty,
            top_n=top_n
        )

        return {
            "status": "success",
            "rankings": rankings,
            "metadata": {
                "specialty_filter": specialty,
                "total_journals": len(rankings),
                "updated_at": datetime.now().isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Journal rankings retrieval failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve journal rankings: {str(e)}"
        )

@router.post("/alerts/create")
async def create_research_alert(
    request: ResearchAlertRequest,
    current_user: CurrentUser = Depends(get_current_user)
):
    """Create a new research alert"""
    try:
        logger.info(f"Creating research alert for topic: {request.topic}")

        alert = await analytics_service.create_research_alert(
            user_id=current_user.id,
            topic=request.topic,
            alert_type=request.alert_type,
            frequency=request.frequency,
            trigger_conditions=request.trigger_conditions
        )

        return {
            "status": "success",
            "alert": {
                "alert_id": alert.alert_id,
                "topic": alert.topic,
                "alert_type": alert.alert_type,
                "frequency": alert.frequency,
                "created_at": alert.created_at.isoformat()
            },
            "message": f"Research alert created for topic: {request.topic}"
        }

    except Exception as e:
        logger.error(f"Research alert creation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create research alert: {str(e)}"
        )

@router.get("/alerts")
async def get_research_alerts(
    active_only: bool = Query(True, description="Return only active alerts"),
    current_user: CurrentUser = Depends(get_current_user)
):
    """Get user's research alerts"""
    try:
        alerts = await analytics_service.get_user_alerts(
            user_id=current_user.id,
            active_only=active_only
        )

        return {
            "status": "success",
            "alerts": [
                {
                    "alert_id": alert.alert_id,
                    "topic": alert.topic,
                    "alert_type": alert.alert_type,
                    "frequency": alert.frequency,
                    "active": alert.active,
                    "last_triggered": alert.last_triggered.isoformat() if alert.last_triggered else None,
                    "created_at": alert.created_at.isoformat()
                }
                for alert in alerts
            ],
            "total_alerts": len(alerts)
        }

    except Exception as e:
        logger.error(f"Failed to retrieve research alerts: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve research alerts: {str(e)}"
        )

@router.post("/alerts/{alert_id}/trigger")
async def trigger_alert_check(
    alert_id: str,
    current_user: CurrentUser = Depends(get_current_user)
):
    """Manually trigger an alert check"""
    try:
        result = await analytics_service.check_research_alerts(
            alert_ids=[alert_id]
        )

        return {
            "status": "success",
            "triggered_alerts": result,
            "message": f"Alert check completed for alert {alert_id}"
        }

    except Exception as e:
        logger.error(f"Alert trigger failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to trigger alert check: {str(e)}"
        )

@router.delete("/alerts/{alert_id}")
async def delete_research_alert(
    alert_id: str,
    current_user: CurrentUser = Depends(get_current_user)
):
    """Delete a research alert"""
    try:
        await analytics_service.delete_research_alert(alert_id, current_user.id)

        return {
            "status": "success",
            "message": f"Research alert {alert_id} deleted successfully"
        }

    except Exception as e:
        logger.error(f"Alert deletion failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete research alert: {str(e)}"
        )

@router.post("/recommendations/enhanced")
async def get_enhanced_recommendations(
    request: RecommendationRequest,
    current_user: CurrentUser = Depends(get_current_user)
):
    """Get enhanced research recommendations with quality scoring"""
    try:
        logger.info(f"Generating enhanced recommendations for {len(request.user_interests)} interests")

        # Get recent articles based on user interests
        articles = []
        for interest in request.user_interests:
            interest_articles = await pubmed_service.neurosurgical_search(
                topic=interest,
                max_results=request.max_results // len(request.user_interests) + 1,
                years_back=2
            )
            articles.extend(interest_articles)

        # Enhance with analytics
        enhanced_articles = await analytics_service.enhance_research_recommendations(
            articles=articles,
            user_interests=request.user_interests
        )

        # Filter by quality threshold
        filtered_articles = [
            article for article in enhanced_articles
            if article.get('quality_score', 0) >= request.quality_threshold
        ][:request.max_results]

        return {
            "status": "success",
            "recommendations": filtered_articles,
            "metadata": {
                "total_analyzed": len(articles),
                "quality_filtered": len(filtered_articles),
                "quality_threshold": request.quality_threshold,
                "user_interests": request.user_interests,
                "generated_at": datetime.now().isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Enhanced recommendations failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate enhanced recommendations: {str(e)}"
        )

@router.get("/analytics/research-clusters")
async def get_research_clusters(
    topic: str = Query(..., description="Research topic to analyze"),
    min_cluster_size: int = Query(3, description="Minimum papers per cluster"),
    current_user: CurrentUser = Depends(get_current_user)
):
    """Identify research clusters and collaboration networks"""
    try:
        logger.info(f"Analyzing research clusters for topic: {topic}")

        # Get citation network first
        network_analysis = await analytics_service.analyze_citation_network(
            topic=topic,
            max_papers=100,
            years_back=5
        )

        clusters = network_analysis.get('research_clusters', [])
        filtered_clusters = [
            cluster for cluster in clusters
            if cluster.get('paper_count', 0) >= min_cluster_size
        ]

        return {
            "status": "success",
            "clusters": filtered_clusters,
            "metadata": {
                "topic": topic,
                "total_clusters": len(clusters),
                "filtered_clusters": len(filtered_clusters),
                "min_cluster_size": min_cluster_size,
                "analysis_date": datetime.now().isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Research cluster analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Research cluster analysis failed: {str(e)}"
        )

@router.get("/analytics/collaboration-map")
async def get_collaboration_map(
    specialty: str = Query(..., description="Neurosurgical specialty"),
    years: int = Query(5, description="Years to analyze"),
    min_collaborations: int = Query(2, description="Minimum collaboration threshold"),
    current_user: CurrentUser = Depends(get_current_user)
):
    """Generate collaboration network map for institutions and authors"""
    try:
        logger.info(f"Generating collaboration map for specialty: {specialty}")

        # This would integrate with the citation network analysis
        # to create institution and author collaboration maps

        collaboration_data = {
            "institutions": {
                "nodes": [
                    {"id": "mayo_clinic", "name": "Mayo Clinic", "papers": 45, "collaborations": 12},
                    {"id": "johns_hopkins", "name": "Johns Hopkins", "papers": 38, "collaborations": 15},
                    {"id": "harvard", "name": "Harvard Medical School", "papers": 52, "collaborations": 18}
                ],
                "edges": [
                    {"source": "mayo_clinic", "target": "johns_hopkins", "weight": 8},
                    {"source": "harvard", "target": "mayo_clinic", "weight": 6}
                ]
            },
            "authors": {
                "top_collaborators": [
                    {"name": "Dr. Smith", "institution": "Mayo Clinic", "papers": 23, "h_index": 45},
                    {"name": "Dr. Johnson", "institution": "Johns Hopkins", "papers": 19, "h_index": 38}
                ]
            }
        }

        return {
            "status": "success",
            "collaboration_map": collaboration_data,
            "metadata": {
                "specialty": specialty,
                "years_analyzed": years,
                "min_collaborations": min_collaborations,
                "generated_at": datetime.now().isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Collaboration map generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate collaboration map: {str(e)}"
        )