"""
AI Knowledge Services API
Comprehensive API for AI-powered neurosurgical knowledge management
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from pydantic import BaseModel, Field

from core.dependencies import get_current_user, CurrentUser
from services.ai_chapter_generation import (
    ai_chapter_generator, ChapterType, GenerationQuality, GenerationRequest
)
from services.literature_summarization import (
    literature_summarizer, SummaryType, SummaryRequest
)
from services.semantic_search import (
    semantic_search_engine, SearchQuery, SearchType, ContentType
)
from services.smart_tagging import (
    neurosurgical_tagger, TagType
)
from services.intelligent_cross_referencing import (
    intelligent_cross_referencer, CrossReferenceType
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Pydantic Models for API

class ChapterGenerationRequest(BaseModel):
    topic: str = Field(..., description="Neurosurgical topic for chapter")
    chapter_type: str = Field(..., description="Type of chapter")
    quality_level: str = Field(default="standard", description="Quality level")
    target_audience: str = Field(default="residents", description="Target audience")
    specialty_focus: List[str] = Field(..., description="Neurosurgical specialties")
    max_length: int = Field(default=3000, description="Maximum chapter length")
    include_cases: bool = Field(default=True, description="Include case examples")
    include_images: bool = Field(default=True, description="Include image references")

class LiteratureSummaryRequest(BaseModel):
    topic: str = Field(..., description="Research topic to summarize")
    summary_type: str = Field(default="comprehensive", description="Type of summary")
    max_papers: int = Field(default=50, description="Maximum papers to analyze")
    years_back: int = Field(default=5, description="Years of literature to include")
    specialty_focus: List[str] = Field(default=[], description="Specialty filters")
    quality_threshold: float = Field(default=0.6, description="Quality threshold")

class SemanticSearchRequest(BaseModel):
    query_text: str = Field(..., description="Search query")
    search_type: str = Field(default="general", description="Type of search")
    content_types: List[str] = Field(default=["all"], description="Content types to search")
    specialty_filter: List[str] = Field(default=[], description="Specialty filters")
    max_results: int = Field(default=20, description="Maximum results")
    quality_threshold: float = Field(default=0.0, description="Quality threshold")

class ContentTaggingRequest(BaseModel):
    content_id: str = Field(..., description="Content identifier")
    content_text: str = Field(..., description="Content to tag")
    content_type: str = Field(default="chapter", description="Type of content")

class CrossReferenceRequest(BaseModel):
    chapter_id: str = Field(..., description="Chapter identifier")
    content: str = Field(..., description="Chapter content")
    existing_chapters: List[Dict[str, Any]] = Field(default=[], description="Existing chapters for reference")

class ConceptGraphRequest(BaseModel):
    central_concept: str = Field(..., description="Central concept for graph")
    depth: int = Field(default=2, description="Graph depth")

# Chapter Generation Endpoints

@router.post("/ai/generate-chapter")
async def generate_neurosurgical_chapter(
    request: ChapterGenerationRequest,
    background_tasks: BackgroundTasks,
    current_user: CurrentUser = Depends(get_current_user)
):
    """Generate a comprehensive neurosurgical chapter using AI"""
    try:
        logger.info(f"Generating chapter for topic: {request.topic}")

        # Convert string enums to actual enums
        chapter_type = ChapterType(request.chapter_type)
        quality_level = GenerationQuality(request.quality_level)

        # Create generation request
        gen_request = GenerationRequest(
            topic=request.topic,
            chapter_type=chapter_type,
            quality_level=quality_level,
            target_audience=request.target_audience,
            specialty_focus=request.specialty_focus,
            include_cases=request.include_cases,
            include_images=request.include_images,
            max_length=request.max_length
        )

        # Generate chapter
        generated_chapter = await ai_chapter_generator.generate_chapter(gen_request)

        return {
            "status": "success",
            "chapter": {
                "chapter_id": generated_chapter.chapter_id,
                "title": generated_chapter.title,
                "content": generated_chapter.content,
                "quality_score": generated_chapter.quality_score,
                "word_count": generated_chapter.metadata.get("word_count", 0),
                "citations": len(generated_chapter.citations),
                "cross_references": generated_chapter.cross_references,
                "research_gaps": generated_chapter.research_gaps,
                "generation_time": generated_chapter.generation_time.isoformat()
            },
            "metadata": generated_chapter.metadata
        }

    except Exception as e:
        logger.error(f"Chapter generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Chapter generation failed: {str(e)}"
        )

@router.post("/ai/regenerate-section")
async def regenerate_chapter_section(
    chapter_id: str,
    section_name: str,
    new_requirements: Dict[str, Any],
    current_user: CurrentUser = Depends(get_current_user)
):
    """Regenerate a specific section of a chapter"""
    try:
        regenerated_content = await ai_chapter_generator.regenerate_section(
            chapter_id, section_name, new_requirements
        )

        return {
            "status": "success",
            "section_name": section_name,
            "content": regenerated_content,
            "regenerated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Section regeneration failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Section regeneration failed: {str(e)}"
        )

# Literature Summarization Endpoints

@router.post("/ai/summarize-literature")
async def summarize_neurosurgical_literature(
    request: LiteratureSummaryRequest,
    current_user: CurrentUser = Depends(get_current_user)
):
    """Generate AI-powered literature summary for neurosurgical topics"""
    try:
        logger.info(f"Summarizing literature for: {request.topic}")

        # Convert string enum
        summary_type = SummaryType(request.summary_type)

        # Create summary request
        summary_request = SummaryRequest(
            topic=request.topic,
            summary_type=summary_type,
            max_papers=request.max_papers,
            years_back=request.years_back,
            specialty_focus=request.specialty_focus,
            quality_threshold=request.quality_threshold
        )

        # Generate summary
        synthesis = await literature_summarizer.summarize_literature(summary_request)

        return {
            "status": "success",
            "summary": {
                "summary_id": synthesis.summary_id,
                "topic": synthesis.topic,
                "executive_summary": synthesis.executive_summary,
                "key_findings": synthesis.key_findings,
                "clinical_implications": synthesis.clinical_implications,
                "research_gaps": synthesis.research_gaps,
                "recommendations": synthesis.recommendations,
                "papers_analyzed": len(synthesis.papers_included),
                "evidence_quality": synthesis.evidence_quality,
                "confidence_score": synthesis.confidence_score,
                "generated_at": synthesis.generated_at.isoformat()
            },
            "detailed_analysis": synthesis.detailed_analysis,
            "conflicts_identified": synthesis.conflicts_identified
        }

    except Exception as e:
        logger.error(f"Literature summarization failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Literature summarization failed: {str(e)}"
        )

# Semantic Search Endpoints

@router.post("/ai/semantic-search")
async def perform_semantic_search(
    request: SemanticSearchRequest,
    current_user: CurrentUser = Depends(get_current_user)
):
    """Perform semantic search across neurosurgical knowledge"""
    try:
        logger.info(f"Performing semantic search for: {request.query_text}")

        # Convert string enums
        search_type = SearchType(request.search_type)
        content_types = [ContentType(ct) for ct in request.content_types]

        # Create search query
        search_query = SearchQuery(
            query_text=request.query_text,
            search_type=search_type,
            content_types=content_types,
            specialty_filter=request.specialty_filter,
            max_results=request.max_results,
            quality_threshold=request.quality_threshold
        )

        # Perform search
        search_response = await semantic_search_engine.semantic_search(search_query)

        return {
            "status": "success",
            "search_results": {
                "total_results": search_response.total_results,
                "search_time_ms": search_response.search_time_ms,
                "results": [
                    {
                        "content_id": result.content_id,
                        "title": result.title,
                        "content_type": result.content_type.value,
                        "excerpt": result.excerpt,
                        "relevance_score": result.relevance_score,
                        "semantic_similarity": result.semantic_similarity,
                        "keyword_matches": result.keyword_matches,
                        "medical_concepts": result.medical_concepts,
                        "source": result.source,
                        "highlighted_text": result.highlighted_text
                    }
                    for result in search_response.results
                ],
                "query_expansion": search_response.query_expansion,
                "related_concepts": search_response.related_concepts,
                "semantic_clusters": search_response.semantic_clusters,
                "suggested_filters": search_response.suggested_filters
            }
        }

    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Semantic search failed: {str(e)}"
        )

@router.get("/ai/search-suggestions")
async def get_search_suggestions(
    partial_query: str = Query(..., description="Partial search query"),
    limit: int = Query(10, description="Maximum suggestions"),
    current_user: CurrentUser = Depends(get_current_user)
):
    """Get search suggestions for autocomplete"""
    try:
        suggestions = await semantic_search_engine.get_search_suggestions(partial_query, limit)

        return {
            "status": "success",
            "suggestions": suggestions,
            "count": len(suggestions)
        }

    except Exception as e:
        logger.error(f"Search suggestions failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search suggestions failed: {str(e)}"
        )

# Smart Tagging Endpoints

@router.post("/ai/tag-content")
async def tag_neurosurgical_content(
    request: ContentTaggingRequest,
    current_user: CurrentUser = Depends(get_current_user)
):
    """Tag neurosurgical content with AI-powered medical concepts"""
    try:
        logger.info(f"Tagging content: {request.content_id}")

        # Tag content
        tagging_result = await neurosurgical_tagger.tag_content(
            request.content_id,
            request.content_text,
            request.content_type
        )

        return {
            "status": "success",
            "tagging_result": {
                "content_id": tagging_result.content_id,
                "primary_topic": tagging_result.primary_topic,
                "anatomical_regions": tagging_result.anatomical_regions,
                "surgical_procedures": tagging_result.surgical_procedures,
                "disease_entities": tagging_result.disease_entities,
                "imaging_findings": tagging_result.imaging_findings,
                "clinical_significance": tagging_result.clinical_significance,
                "tags": [
                    {
                        "tag_id": tag.tag_id,
                        "tag_name": tag.tag_name,
                        "tag_type": tag.tag_type.value,
                        "confidence": tag.confidence,
                        "confidence_level": tag.confidence_level.value,
                        "context": tag.context,
                        "synonyms": tag.synonyms,
                        "related_concepts": tag.related_concepts,
                        "anatomical_location": tag.anatomical_location,
                        "pathological_type": tag.pathological_type,
                        "surgical_complexity": tag.surgical_complexity
                    }
                    for tag in tagging_result.tags
                ],
                "tagged_at": tagging_result.tagged_at.isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Content tagging failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Content tagging failed: {str(e)}"
        )

@router.get("/ai/tag-suggestions")
async def get_tag_suggestions(
    partial_tag: str = Query(..., description="Partial tag name"),
    tag_type: Optional[str] = Query(None, description="Tag type filter"),
    current_user: CurrentUser = Depends(get_current_user)
):
    """Get tag suggestions for autocomplete"""
    try:
        tag_type_enum = TagType(tag_type) if tag_type else None
        suggestions = await neurosurgical_tagger.get_tag_suggestions(partial_tag, tag_type_enum)

        return {
            "status": "success",
            "suggestions": suggestions,
            "count": len(suggestions)
        }

    except Exception as e:
        logger.error(f"Tag suggestions failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Tag suggestions failed: {str(e)}"
        )

@router.get("/ai/popular-tags")
async def get_popular_tags(
    tag_type: Optional[str] = Query(None, description="Tag type filter"),
    limit: int = Query(20, description="Maximum tags to return"),
    current_user: CurrentUser = Depends(get_current_user)
):
    """Get popular neurosurgical tags"""
    try:
        tag_type_enum = TagType(tag_type) if tag_type else None
        popular_tags = await neurosurgical_tagger.get_popular_tags(tag_type_enum, limit)

        return {
            "status": "success",
            "popular_tags": popular_tags,
            "total_count": len(popular_tags)
        }

    except Exception as e:
        logger.error(f"Popular tags retrieval failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Popular tags retrieval failed: {str(e)}"
        )

# Cross-Referencing Endpoints

@router.post("/ai/generate-cross-references")
async def generate_cross_references(
    request: CrossReferenceRequest,
    current_user: CurrentUser = Depends(get_current_user)
):
    """Generate intelligent cross-references for neurosurgical content"""
    try:
        logger.info(f"Generating cross-references for chapter: {request.chapter_id}")

        # Generate cross-references
        cross_references = await intelligent_cross_referencer.generate_cross_references(
            request.chapter_id,
            request.content,
            request.existing_chapters
        )

        return {
            "status": "success",
            "cross_references": [
                {
                    "reference_id": ref.reference_id,
                    "source_concept": ref.source_concept,
                    "target_concept": ref.target_concept,
                    "target_chapter_id": ref.target_chapter_id,
                    "reference_type": ref.reference_type.value,
                    "strength": ref.strength.value,
                    "confidence": ref.confidence,
                    "description": ref.description,
                    "clinical_relevance": ref.clinical_relevance,
                    "bidirectional": ref.bidirectional,
                    "context_keywords": ref.context_keywords,
                    "anatomical_context": ref.anatomical_context,
                    "pathological_context": ref.pathological_context
                }
                for ref in cross_references
            ],
            "total_references": len(cross_references)
        }

    except Exception as e:
        logger.error(f"Cross-reference generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Cross-reference generation failed: {str(e)}"
        )

@router.post("/ai/concept-graph")
async def build_concept_graph(
    request: ConceptGraphRequest,
    current_user: CurrentUser = Depends(get_current_user)
):
    """Build a concept graph for neurosurgical knowledge visualization"""
    try:
        logger.info(f"Building concept graph for: {request.central_concept}")

        # Build concept graph
        concept_graph = await intelligent_cross_referencer.build_concept_graph(
            request.central_concept,
            request.depth
        )

        return {
            "status": "success",
            "concept_graph": {
                "graph_id": concept_graph.graph_id,
                "central_concept": concept_graph.central_concept,
                "related_concepts": concept_graph.related_concepts,
                "concept_hierarchy": concept_graph.concept_hierarchy,
                "clinical_pathways": concept_graph.clinical_pathways,
                "cross_references": [
                    {
                        "source_concept": ref.source_concept,
                        "target_concept": ref.target_concept,
                        "reference_type": ref.reference_type.value,
                        "strength": ref.strength.value,
                        "confidence": ref.confidence,
                        "description": ref.description
                    }
                    for ref in concept_graph.cross_references
                ],
                "generated_at": concept_graph.generated_at.isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Concept graph building failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Concept graph building failed: {str(e)}"
        )

# Utility Endpoints

@router.get("/ai/capabilities")
async def get_ai_capabilities(
    current_user: CurrentUser = Depends(get_current_user)
):
    """Get available AI capabilities and features"""
    return {
        "status": "success",
        "capabilities": {
            "chapter_generation": {
                "available": True,
                "chapter_types": [ct.value for ct in ChapterType],
                "quality_levels": [ql.value for ql in GenerationQuality],
                "supported_specialties": [
                    "neuro-oncology", "vascular neurosurgery", "spine surgery",
                    "functional neurosurgery", "pediatric neurosurgery", "skull base surgery"
                ]
            },
            "literature_summarization": {
                "available": True,
                "summary_types": [st.value for st in SummaryType],
                "max_papers": 100,
                "years_back_limit": 10
            },
            "semantic_search": {
                "available": True,
                "search_types": [st.value for st in SearchType],
                "content_types": [ct.value for ct in ContentType],
                "features": ["query_expansion", "concept_clustering", "relevance_ranking"]
            },
            "smart_tagging": {
                "available": True,
                "tag_types": [tt.value for tt in TagType],
                "features": ["confidence_scoring", "synonym_detection", "concept_hierarchy"]
            },
            "cross_referencing": {
                "available": True,
                "reference_types": [rt.value for rt in CrossReferenceType],
                "features": ["concept_graphs", "clinical_pathways", "bidirectional_links"]
            }
        },
        "version": "1.0.0",
        "updated_at": datetime.now().isoformat()
    }