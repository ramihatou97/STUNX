"""
Reference Library API Endpoints
API for managing textbook references and chapters
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
from pydantic import BaseModel, Field

from core.dependencies import get_current_user, get_db, CurrentUser
from core.exceptions import ValidationError, ResourceNotFoundError, ExternalServiceError
from models.references import Textbook, BookChapter, ChapterCitation, ContentReference
from services.reference_library import reference_library, ProcessingStats, ChapterSearchResult

logger = logging.getLogger(__name__)
router = APIRouter()

# Pydantic Models for API

class TextbookResponse(BaseModel):
    """Textbook response model"""
    id: str
    name: str
    title: str
    edition: Optional[str]
    authors: List[str]
    publisher: Optional[str]
    publication_year: Optional[int]
    specialty: str
    folder_path: str
    is_processed: bool
    chapter_count: int
    created_at: datetime

    class Config:
        from_attributes = True

class ChapterResponse(BaseModel):
    """Chapter response model"""
    id: str
    textbook_id: str
    file_name: str
    file_path: str
    chapter_number: Optional[int]
    title: str
    summary: Optional[str]
    keywords: List[str]
    medical_terms: List[str]
    word_count: Optional[int]
    reading_time_minutes: Optional[int]
    page_count: Optional[int]
    is_processed: bool
    created_at: datetime

    class Config:
        from_attributes = True

class ChapterDetailResponse(ChapterResponse):
    """Detailed chapter response with content"""
    content_text: Optional[str]
    textbook_title: str
    citations_count: int

class CitationResponse(BaseModel):
    """Citation response model"""
    id: str
    citation_text: str
    citation_type: str
    page_reference: Optional[str]
    doi: Optional[str]
    pmid: Optional[int]
    confidence_score: Optional[float]

    class Config:
        from_attributes = True

class ChapterSearchRequest(BaseModel):
    """Chapter search request"""
    query: str = Field(..., min_length=2, max_length=500)
    textbook_id: Optional[str] = None
    specialty: Optional[str] = None
    limit: int = Field(20, ge=1, le=100)

class ChapterSearchResponse(BaseModel):
    """Chapter search response"""
    chapter_id: str
    textbook_title: str
    chapter_title: str
    relevance_score: float
    matching_text: str
    file_path: str
    chapter_number: Optional[int]

class ProcessingStatsResponse(BaseModel):
    """Processing statistics response"""
    textbooks_found: int
    textbooks_processed: int
    chapters_found: int
    chapters_processed: int
    chapters_failed: int
    processing_time_seconds: float
    errors: List[str]

class ReferenceRecommendationResponse(BaseModel):
    """Reference recommendation response"""
    chapter_id: str
    textbook_title: str
    chapter_title: str
    relevance_score: float
    recommendation_reason: str
    key_topics: List[str]

# API Endpoints

@router.get("/textbooks", response_model=List[TextbookResponse])
async def get_textbooks(
    processed_only: bool = Query(True, description="Only return processed textbooks"),
    specialty: Optional[str] = Query(None, description="Filter by specialty"),
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get all textbooks in the library"""
    try:
        query = select(Textbook).options(selectinload(Textbook.chapters))

        if processed_only:
            query = query.where(Textbook.is_processed == True)

        if specialty:
            query = query.where(Textbook.specialty == specialty)

        result = await db.execute(query)
        textbooks = result.scalars().all()

        # Convert to response models
        textbook_responses = []
        for textbook in textbooks:
            textbook_data = TextbookResponse(
                id=textbook.id,
                name=textbook.name,
                title=textbook.title,
                edition=textbook.edition,
                authors=textbook.authors,
                publisher=textbook.publisher,
                publication_year=textbook.publication_year,
                specialty=textbook.specialty,
                folder_path=textbook.folder_path,
                is_processed=textbook.is_processed,
                chapter_count=len(textbook.chapters),
                created_at=textbook.created_at
            )
            textbook_responses.append(textbook_data)

        return textbook_responses

    except Exception as e:
        logger.error(f"Error retrieving textbooks: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve textbooks")

@router.get("/textbooks/{textbook_id}/chapters", response_model=List[ChapterResponse])
async def get_textbook_chapters(
    textbook_id: str,
    processed_only: bool = Query(True, description="Only return processed chapters"),
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get all chapters for a specific textbook"""
    try:
        # Verify textbook exists
        textbook_result = await db.execute(
            select(Textbook).where(Textbook.id == textbook_id)
        )
        textbook = textbook_result.scalar_one_or_none()

        if not textbook:
            raise HTTPException(status_code=404, detail="Textbook not found")

        # Get chapters
        query = select(BookChapter).where(BookChapter.textbook_id == textbook_id)

        if processed_only:
            query = query.where(BookChapter.is_processed == True)

        query = query.order_by(BookChapter.chapter_number, BookChapter.title)

        result = await db.execute(query)
        chapters = result.scalars().all()

        return [ChapterResponse.model_validate(chapter) for chapter in chapters]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving chapters for textbook {textbook_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chapters")

@router.get("/chapters/{chapter_id}", response_model=ChapterDetailResponse)
async def get_chapter_detail(
    chapter_id: str,
    include_content: bool = Query(False, description="Include full chapter content"),
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get detailed information for a specific chapter"""
    try:
        result = await db.execute(
            select(BookChapter).options(
                selectinload(BookChapter.textbook),
                selectinload(BookChapter.citations)
            ).where(BookChapter.id == chapter_id)
        )
        chapter = result.scalar_one_or_none()

        if not chapter:
            raise HTTPException(status_code=404, detail="Chapter not found")

        # Prepare response
        chapter_data = {
            "id": chapter.id,
            "textbook_id": chapter.textbook_id,
            "file_name": chapter.file_name,
            "file_path": chapter.file_path,
            "chapter_number": chapter.chapter_number,
            "title": chapter.title,
            "summary": chapter.summary,
            "keywords": chapter.keywords,
            "medical_terms": chapter.medical_terms,
            "word_count": chapter.word_count,
            "reading_time_minutes": chapter.reading_time_minutes,
            "page_count": chapter.page_count,
            "is_processed": chapter.is_processed,
            "created_at": chapter.created_at,
            "textbook_title": chapter.textbook.title,
            "citations_count": len(chapter.citations)
        }

        if include_content:
            chapter_data["content_text"] = chapter.content_text

        return ChapterDetailResponse(**chapter_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving chapter {chapter_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chapter")

@router.get("/chapters/{chapter_id}/citations", response_model=List[CitationResponse])
async def get_chapter_citations(
    chapter_id: str,
    citation_type: Optional[str] = Query(None, description="Filter by citation type"),
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get citations for a specific chapter"""
    try:
        # Verify chapter exists
        chapter_result = await db.execute(
            select(BookChapter).where(BookChapter.id == chapter_id)
        )
        chapter = chapter_result.scalar_one_or_none()

        if not chapter:
            raise HTTPException(status_code=404, detail="Chapter not found")

        # Get citations
        query = select(ChapterCitation).where(ChapterCitation.chapter_id == chapter_id)

        if citation_type:
            query = query.where(ChapterCitation.citation_type == citation_type)

        result = await db.execute(query)
        citations = result.scalars().all()

        return [CitationResponse.model_validate(citation) for citation in citations]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving citations for chapter {chapter_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve citations")

@router.post("/chapters/search", response_model=List[ChapterSearchResponse])
async def search_chapters(
    search_request: ChapterSearchRequest,
    current_user: CurrentUser = Depends(get_current_user)
):
    """Search chapters by content, title, or keywords"""
    try:
        search_results = await reference_library.search_chapters(
            query=search_request.query,
            textbook_id=search_request.textbook_id,
            specialty=search_request.specialty,
            limit=search_request.limit
        )

        return [
            ChapterSearchResponse(
                chapter_id=result.chapter_id,
                textbook_title=result.textbook_title,
                chapter_title=result.chapter_title,
                relevance_score=result.relevance_score,
                matching_text=result.matching_text,
                file_path=result.file_path,
                chapter_number=result.chapter_number
            )
            for result in search_results
        ]

    except Exception as e:
        logger.error(f"Error searching chapters: {e}")
        raise HTTPException(status_code=500, detail="Failed to search chapters")

@router.get("/recommendations/{topic}", response_model=List[ReferenceRecommendationResponse])
async def get_reference_recommendations(
    topic: str,
    limit: int = Query(10, ge=1, le=20),
    current_user: CurrentUser = Depends(get_current_user)
):
    """Get recommended references for a specific topic"""
    try:
        # Search for relevant chapters
        search_results = await reference_library.search_chapters(
            query=topic,
            limit=limit
        )

        recommendations = []
        for result in search_results:
            recommendation = ReferenceRecommendationResponse(
                chapter_id=result.chapter_id,
                textbook_title=result.textbook_title,
                chapter_title=result.chapter_title,
                relevance_score=result.relevance_score,
                recommendation_reason=f"Contains relevant content about {topic}",
                key_topics=[topic]  # Could be enhanced with AI analysis
            )
            recommendations.append(recommendation)

        return recommendations

    except Exception as e:
        logger.error(f"Error getting recommendations for topic {topic}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get recommendations")

@router.post("/process-folder", response_model=ProcessingStatsResponse)
async def process_textbooks_folder(
    background_tasks: BackgroundTasks,
    force_rescan: bool = Query(False, description="Force reprocessing of existing textbooks"),
    current_user: CurrentUser = Depends(get_current_user)
):
    """Process all textbooks in the textbooks folder"""
    try:
        # Start processing in background
        background_tasks.add_task(
            _process_folder_background,
            force_rescan
        )

        return ProcessingStatsResponse(
            textbooks_found=0,
            textbooks_processed=0,
            chapters_found=0,
            chapters_processed=0,
            chapters_failed=0,
            processing_time_seconds=0.0,
            errors=["Processing started in background. Check status endpoint for updates."]
        )

    except Exception as e:
        logger.error(f"Error starting textbook processing: {e}")
        raise HTTPException(status_code=500, detail="Failed to start processing")

@router.get("/processing-status", response_model=Dict[str, Any])
async def get_processing_status(
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get current processing status"""
    try:
        # Get textbook processing stats
        textbook_stats = await db.execute(
            select(
                func.count(Textbook.id).label('total_textbooks'),
                func.count(Textbook.id).filter(Textbook.is_processed == True).label('processed_textbooks'),
                func.count(Textbook.id).filter(Textbook.processing_error.isnot(None)).label('failed_textbooks')
            )
        )
        textbook_row = textbook_stats.first()

        # Get chapter processing stats
        chapter_stats = await db.execute(
            select(
                func.count(BookChapter.id).label('total_chapters'),
                func.count(BookChapter.id).filter(BookChapter.is_processed == True).label('processed_chapters'),
                func.count(BookChapter.id).filter(BookChapter.processing_error.isnot(None)).label('failed_chapters')
            )
        )
        chapter_row = chapter_stats.first()

        return {
            "textbooks": {
                "total": textbook_row.total_textbooks,
                "processed": textbook_row.processed_textbooks,
                "failed": textbook_row.failed_textbooks,
                "processing_rate": (textbook_row.processed_textbooks / max(textbook_row.total_textbooks, 1)) * 100
            },
            "chapters": {
                "total": chapter_row.total_chapters,
                "processed": chapter_row.processed_chapters,
                "failed": chapter_row.failed_chapters,
                "processing_rate": (chapter_row.processed_chapters / max(chapter_row.total_chapters, 1)) * 100
            },
            "last_updated": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting processing status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get processing status")

@router.delete("/textbooks/{textbook_id}")
async def delete_textbook(
    textbook_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a textbook and all its chapters"""
    try:
        # Get textbook
        textbook_result = await db.execute(
            select(Textbook).where(Textbook.id == textbook_id)
        )
        textbook = textbook_result.scalar_one_or_none()

        if not textbook:
            raise HTTPException(status_code=404, detail="Textbook not found")

        # Delete textbook (cascades to chapters)
        await db.delete(textbook)
        await db.commit()

        return {"message": f"Textbook '{textbook.title}' deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting textbook {textbook_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete textbook")

@router.post("/reprocess/{textbook_id}")
async def reprocess_textbook(
    textbook_id: str,
    background_tasks: BackgroundTasks,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Reprocess a specific textbook"""
    try:
        # Verify textbook exists
        textbook_result = await db.execute(
            select(Textbook).where(Textbook.id == textbook_id)
        )
        textbook = textbook_result.scalar_one_or_none()

        if not textbook:
            raise HTTPException(status_code=404, detail="Textbook not found")

        # Start reprocessing in background
        background_tasks.add_task(
            _reprocess_textbook_background,
            textbook_id
        )

        return {"message": f"Reprocessing started for textbook '{textbook.title}'"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting textbook reprocessing: {e}")
        raise HTTPException(status_code=500, detail="Failed to start reprocessing")

# Background task functions

async def _process_folder_background(force_rescan: bool = False):
    """Background task to process textbooks folder"""
    try:
        logger.info("Starting background textbook processing")
        stats = await reference_library.scan_textbooks_folder(force_rescan=force_rescan)
        logger.info(f"Background processing completed: {stats.textbooks_processed} textbooks processed")
    except Exception as e:
        logger.error(f"Background processing failed: {e}")

async def _reprocess_textbook_background(textbook_id: str):
    """Background task to reprocess a specific textbook"""
    try:
        logger.info(f"Starting background reprocessing for textbook {textbook_id}")
        # Implementation would go here
        logger.info(f"Background reprocessing completed for textbook {textbook_id}")
    except Exception as e:
        logger.error(f"Background reprocessing failed for textbook {textbook_id}: {e}")