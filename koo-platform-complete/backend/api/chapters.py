"""
KOO Platform Chapter Management API
CRUD operations with proper database sessions and error handling
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import uuid4
import logging

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from pydantic import BaseModel, Field, validator

from core.dependencies import get_current_user, get_db, CurrentUser, validate_pagination
from core.security import validate_chapter_content, sanitize_output
from core.exceptions import (
    ValidationError,
    ResourceNotFoundError,
    ResourceAlreadyExistsError,
    DatabaseError,
    ErrorContext
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Pydantic Models with Validation
class ChapterCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500, description="Chapter title")
    content: str = Field(..., min_length=1, description="Chapter content")
    summary: Optional[str] = Field(None, max_length=2000, description="Chapter summary")
    tags: List[str] = Field(default_factory=list, description="Chapter tags")
    specialty: Optional[str] = Field(None, max_length=100, description="Medical specialty")

    @validator('title', 'summary')
    def sanitize_text_fields(cls, v):
        if v:
            return sanitize_output(v.strip())
        return v

    @validator('content')
    def validate_content(cls, v):
        if not validate_chapter_content(v):
            raise ValueError("Invalid content format or length")
        return v

    @validator('tags')
    def validate_tags(cls, v):
        if len(v) > 20:
            raise ValueError("Too many tags (max 20)")
        return [tag.strip().lower() for tag in v if tag.strip()]

class ChapterUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=500)
    content: Optional[str] = Field(None, min_length=1)
    summary: Optional[str] = Field(None, max_length=2000)
    tags: Optional[List[str]] = None
    specialty: Optional[str] = Field(None, max_length=100)
    status: Optional[str] = Field(None, pattern="^(draft|review|published|archived)$")

    @validator('title', 'summary')
    def sanitize_text_fields(cls, v):
        if v:
            return sanitize_output(v.strip())
        return v

    @validator('content')
    def validate_content(cls, v):
        if v and not validate_chapter_content(v):
            raise ValueError("Invalid content format or length")
        return v

class ChapterResponse(BaseModel):
    id: int
    uuid: str
    title: str
    content: str
    summary: Optional[str]
    tags: List[str]
    specialty: Optional[str]
    status: str
    created_at: datetime
    updated_at: datetime
    author_name: str
    word_count: int

    class Config:
        from_attributes = True

# In-memory storage (replace with actual database model)
chapters_db: Dict[int, Dict[str, Any]] = {}
chapter_counter = 1

@router.get("/")
async def get_chapters(
    pagination: dict = Depends(validate_pagination),
    status: Optional[str] = Query(None, regex="^(draft|review|published|archived)$"),
    search: Optional[str] = Query(None, min_length=1, max_length=100),
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get all chapters with filtering and pagination"""

    with ErrorContext("get_chapters", user_id=current_user.id):
        try:
            # Filter chapters
            filtered_chapters = list(chapters_db.values())

            if status:
                filtered_chapters = [ch for ch in filtered_chapters if ch["status"] == status]

            if search:
                search_lower = search.lower()
                filtered_chapters = [
                    ch for ch in filtered_chapters
                    if search_lower in ch["title"].lower() or search_lower in ch["content"].lower()
                ]

            # Pagination
            skip = pagination["skip"]
            limit = pagination["limit"]
            total = len(filtered_chapters)
            chapters = filtered_chapters[skip:skip + limit]

            logger.info(f"Retrieved {len(chapters)} chapters for user {current_user.id}")

            return {
                "items": chapters,
                "total": total,
                "skip": skip,
                "limit": limit,
                "has_more": skip + limit < total
            }

        except Exception as e:
            logger.error(f"Error retrieving chapters: {e}")
            raise DatabaseError("Failed to retrieve chapters")

@router.post("/", response_model=ChapterResponse, status_code=status.HTTP_201_CREATED)
async def create_chapter(
    chapter: ChapterCreate,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new chapter with validation"""

    with ErrorContext("create_chapter", user_id=current_user.id, title=chapter.title):
        try:
            global chapter_counter

            # Create chapter
            now = datetime.utcnow()
            chapter_id = chapter_counter
            chapter_counter += 1

            chapter_data = {
                "id": chapter_id,
                "uuid": str(uuid4()),
                "title": chapter.title,
                "content": chapter.content,
                "summary": chapter.summary,
                "tags": chapter.tags,
                "specialty": chapter.specialty,
                "status": "draft",
                "created_at": now,
                "updated_at": now,
                "author_name": current_user.full_name,
                "word_count": len(chapter.content.split())
            }

            chapters_db[chapter_id] = chapter_data

            logger.info(f"Created chapter {chapter_id} for user {current_user.id}")

            return chapter_data

        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Error creating chapter: {e}")
            raise DatabaseError("Failed to create chapter")

@router.get("/{chapter_id}", response_model=ChapterResponse)
async def get_chapter(
    chapter_id: int,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get a specific chapter by ID"""

    with ErrorContext("get_chapter", user_id=current_user.id, chapter_id=chapter_id):
        try:
            if chapter_id not in chapters_db:
                raise ResourceNotFoundError("Chapter", chapter_id)

            chapter = chapters_db[chapter_id]
            logger.info(f"Retrieved chapter {chapter_id} for user {current_user.id}")

            return chapter

        except ResourceNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error retrieving chapter {chapter_id}: {e}")
            raise DatabaseError("Failed to retrieve chapter")

@router.put("/{chapter_id}", response_model=ChapterResponse)
async def update_chapter(
    chapter_id: int,
    chapter_update: ChapterUpdate,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update a chapter with validation"""

    with ErrorContext("update_chapter", user_id=current_user.id, chapter_id=chapter_id):
        try:
            if chapter_id not in chapters_db:
                raise ResourceNotFoundError("Chapter", chapter_id)

            chapter = chapters_db[chapter_id]

            # Update only provided fields
            update_data = chapter_update.dict(exclude_unset=True)

            for field, value in update_data.items():
                if field == "content" and value:
                    chapter["word_count"] = len(value.split())
                chapter[field] = value

            chapter["updated_at"] = datetime.utcnow()

            logger.info(f"Updated chapter {chapter_id} for user {current_user.id}")

            return chapter

        except (ValidationError, ResourceNotFoundError):
            raise
        except Exception as e:
            logger.error(f"Error updating chapter {chapter_id}: {e}")
            raise DatabaseError("Failed to update chapter")

@router.delete("/{chapter_id}")
async def delete_chapter(
    chapter_id: int,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a chapter"""

    with ErrorContext("delete_chapter", user_id=current_user.id, chapter_id=chapter_id):
        try:
            if chapter_id not in chapters_db:
                raise ResourceNotFoundError("Chapter", chapter_id)

            del chapters_db[chapter_id]

            logger.info(f"Deleted chapter {chapter_id} for user {current_user.id}")

            return {"message": "Chapter deleted successfully", "chapter_id": chapter_id}

        except ResourceNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error deleting chapter {chapter_id}: {e}")
            raise DatabaseError("Failed to delete chapter")

@router.get("/{chapter_id}/stats")
async def get_chapter_stats(
    chapter_id: int,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get chapter statistics"""

    with ErrorContext("get_chapter_stats", user_id=current_user.id, chapter_id=chapter_id):
        try:
            if chapter_id not in chapters_db:
                raise ResourceNotFoundError("Chapter", chapter_id)

            chapter = chapters_db[chapter_id]
            content = chapter["content"]

            stats = {
                "word_count": len(content.split()),
                "character_count": len(content),
                "character_count_no_spaces": len(content.replace(" ", "")),
                "paragraph_count": len([p for p in content.split("\n\n") if p.strip()]),
                "estimated_reading_time": max(1, len(content.split()) // 200),  # ~200 WPM
                "last_updated": chapter["updated_at"],
                "status": chapter["status"],
                "tags_count": len(chapter["tags"])
            }

            logger.info(f"Retrieved stats for chapter {chapter_id}")

            return stats

        except ResourceNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error getting chapter stats {chapter_id}: {e}")
            raise DatabaseError("Failed to retrieve chapter statistics")

@router.post("/{chapter_id}/duplicate")
async def duplicate_chapter(
    chapter_id: int,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Duplicate a chapter"""

    with ErrorContext("duplicate_chapter", user_id=current_user.id, chapter_id=chapter_id):
        try:
            global chapter_counter

            if chapter_id not in chapters_db:
                raise ResourceNotFoundError("Chapter", chapter_id)

            original = chapters_db[chapter_id]
            now = datetime.utcnow()
            new_id = chapter_counter
            chapter_counter += 1

            duplicated_chapter = {
                **original,
                "id": new_id,
                "uuid": str(uuid4()),
                "title": f"{original['title']} (Copy)",
                "status": "draft",
                "created_at": now,
                "updated_at": now
            }

            chapters_db[new_id] = duplicated_chapter

            logger.info(f"Duplicated chapter {chapter_id} to {new_id} for user {current_user.id}")

            return duplicated_chapter

        except ResourceNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error duplicating chapter {chapter_id}: {e}")
            raise DatabaseError("Failed to duplicate chapter")