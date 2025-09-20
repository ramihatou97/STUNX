"""
Reference Library Database Models
Database models for textbook and chapter management
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import uuid4
import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID, ARRAY, TEXT, JSONB

from .base import Base

class Textbook(Base):
    """Textbook model for reference library"""
    __tablename__ = "textbooks"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(sa.String(100), nullable=False, index=True)  # folder name
    title: Mapped[str] = mapped_column(sa.String(500), nullable=False)
    edition: Mapped[Optional[str]] = mapped_column(sa.String(50))
    authors: Mapped[List[str]] = mapped_column(ARRAY(TEXT), nullable=False)
    publisher: Mapped[Optional[str]] = mapped_column(sa.String(200))
    publication_year: Mapped[Optional[int]] = mapped_column(sa.Integer)
    isbn: Mapped[Optional[str]] = mapped_column(sa.String(20))
    specialty: Mapped[str] = mapped_column(sa.String(100), nullable=False, default="neurosurgery")
    folder_path: Mapped[str] = mapped_column(sa.String(500), nullable=False)
    textbook_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)

    # Processing status
    is_processed: Mapped[bool] = mapped_column(sa.Boolean, default=False)
    processing_started_at: Mapped[Optional[datetime]] = mapped_column(sa.DateTime)
    processing_completed_at: Mapped[Optional[datetime]] = mapped_column(sa.DateTime)
    processing_error: Mapped[Optional[str]] = mapped_column(TEXT)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(sa.DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(sa.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    chapters: Mapped[List["BookChapter"]] = relationship("BookChapter", back_populates="textbook", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Textbook(id={self.id}, title='{self.title}', edition='{self.edition}')>"

class BookChapter(Base):
    """Chapter model for individual PDF files"""
    __tablename__ = "book_chapters"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    textbook_id: Mapped[str] = mapped_column(UUID(as_uuid=False), sa.ForeignKey("textbooks.id"), nullable=False)

    # File information
    file_name: Mapped[str] = mapped_column(sa.String(200), nullable=False)
    file_path: Mapped[str] = mapped_column(sa.String(500), nullable=False, unique=True)
    file_size_mb: Mapped[Optional[float]] = mapped_column(sa.Float)

    # Chapter metadata
    chapter_number: Mapped[Optional[int]] = mapped_column(sa.Integer)
    title: Mapped[str] = mapped_column(sa.String(500), nullable=False)
    subtitle: Mapped[Optional[str]] = mapped_column(sa.String(500))

    # Content
    content_text: Mapped[Optional[str]] = mapped_column(TEXT)
    summary: Mapped[Optional[str]] = mapped_column(TEXT)
    page_count: Mapped[Optional[int]] = mapped_column(sa.Integer)

    # Extracted metadata
    keywords: Mapped[List[str]] = mapped_column(ARRAY(TEXT), default=list)
    medical_terms: Mapped[List[str]] = mapped_column(ARRAY(TEXT), default=list)
    topics: Mapped[List[str]] = mapped_column(ARRAY(TEXT), default=list)

    # Content analysis
    word_count: Mapped[Optional[int]] = mapped_column(sa.Integer)
    reading_time_minutes: Mapped[Optional[int]] = mapped_column(sa.Integer)
    complexity_score: Mapped[Optional[float]] = mapped_column(sa.Float)  # 0.0 to 1.0

    # Processing status
    is_processed: Mapped[bool] = mapped_column(sa.Boolean, default=False)
    processing_error: Mapped[Optional[str]] = mapped_column(TEXT)

    # Timestamps
    processed_at: Mapped[Optional[datetime]] = mapped_column(sa.DateTime)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(sa.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    textbook: Mapped["Textbook"] = relationship("Textbook", back_populates="chapters")
    citations: Mapped[List["ChapterCitation"]] = relationship("ChapterCitation", back_populates="chapter", cascade="all, delete-orphan")
    content_references: Mapped[List["ContentReference"]] = relationship("ContentReference", back_populates="chapter", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<BookChapter(id={self.id}, title='{self.title}', file='{self.file_name}')>"

class ChapterCitation(Base):
    """Citations found within chapters"""
    __tablename__ = "chapter_citations"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    chapter_id: Mapped[str] = mapped_column(UUID(as_uuid=False), sa.ForeignKey("book_chapters.id"), nullable=False)

    # Citation details
    citation_text: Mapped[str] = mapped_column(TEXT, nullable=False)
    citation_type: Mapped[str] = mapped_column(sa.String(50), nullable=False)  # "reference", "figure", "table", "internal"
    page_reference: Mapped[Optional[str]] = mapped_column(sa.String(20))
    section_reference: Mapped[Optional[str]] = mapped_column(sa.String(100))

    # Context
    context_before: Mapped[Optional[str]] = mapped_column(TEXT)
    context_after: Mapped[Optional[str]] = mapped_column(TEXT)

    # External references
    doi: Mapped[Optional[str]] = mapped_column(sa.String(100))
    pmid: Mapped[Optional[int]] = mapped_column(sa.Integer)
    isbn: Mapped[Optional[str]] = mapped_column(sa.String(20))
    url: Mapped[Optional[str]] = mapped_column(sa.String(500))

    # Quality metrics
    confidence_score: Mapped[Optional[float]] = mapped_column(sa.Float)  # 0.0 to 1.0
    is_verified: Mapped[bool] = mapped_column(sa.Boolean, default=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(sa.DateTime, default=datetime.utcnow)

    # Relationships
    chapter: Mapped["BookChapter"] = relationship("BookChapter", back_populates="citations")

    def __repr__(self):
        return f"<ChapterCitation(id={self.id}, type='{self.citation_type}', text='{self.citation_text[:50]}...')>"

class ContentReference(Base):
    """Links between AI-generated content and textbook references"""
    __tablename__ = "content_references"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))

    # AI content identification
    ai_content_id: Mapped[Optional[str]] = mapped_column(UUID(as_uuid=False))  # Links to generated content
    ai_content_type: Mapped[str] = mapped_column(sa.String(50), nullable=False)  # "chapter", "summary", "search_result"
    ai_content_hash: Mapped[Optional[str]] = mapped_column(sa.String(64))  # Content hash for deduplication

    # Reference information
    chapter_id: Mapped[str] = mapped_column(UUID(as_uuid=False), sa.ForeignKey("book_chapters.id"), nullable=False)
    reference_type: Mapped[str] = mapped_column(sa.String(50), nullable=False)  # "supporting", "contradicting", "related", "cited"

    # Relevance and quality
    relevance_score: Mapped[float] = mapped_column(sa.Float, nullable=False)  # 0.0 to 1.0
    confidence_score: Mapped[Optional[float]] = mapped_column(sa.Float)  # 0.0 to 1.0

    # Context
    citation_context: Mapped[Optional[str]] = mapped_column(TEXT)  # Where/how it was referenced
    matching_text: Mapped[Optional[str]] = mapped_column(TEXT)  # Specific text that matched

    # Usage tracking
    usage_count: Mapped[int] = mapped_column(sa.Integer, default=0)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(sa.DateTime)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(sa.DateTime, default=datetime.utcnow)

    # Relationships
    chapter: Mapped["BookChapter"] = relationship("BookChapter", back_populates="content_references")

    def __repr__(self):
        return f"<ContentReference(id={self.id}, type='{self.reference_type}', score={self.relevance_score})>"

class ReferenceSearchIndex(Base):
    """Search index for fast text search across chapters"""
    __tablename__ = "reference_search_index"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    chapter_id: Mapped[str] = mapped_column(UUID(as_uuid=False), sa.ForeignKey("book_chapters.id"), nullable=False)

    # Search vectors
    content_vector: Mapped[Optional[str]] = mapped_column(TEXT)  # For semantic search
    keyword_vector: Mapped[Optional[str]] = mapped_column(TEXT)  # For keyword search

    # Text chunks for context
    text_chunk: Mapped[str] = mapped_column(TEXT, nullable=False)
    chunk_position: Mapped[int] = mapped_column(sa.Integer, nullable=False)  # Position in chapter
    chunk_size: Mapped[int] = mapped_column(sa.Integer, nullable=False)

    # Metadata
    section_title: Mapped[Optional[str]] = mapped_column(sa.String(200))
    page_number: Mapped[Optional[int]] = mapped_column(sa.Integer)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(sa.DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(sa.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<ReferenceSearchIndex(id={self.id}, chapter_id={self.chapter_id}, position={self.chunk_position})>"

# Create indexes for performance
def create_indexes():
    """Create database indexes for optimal performance"""
    return [
        # Textbook indexes
        sa.Index('idx_textbooks_name', Textbook.name),
        sa.Index('idx_textbooks_specialty', Textbook.specialty),
        sa.Index('idx_textbooks_processed', Textbook.is_processed),

        # Chapter indexes
        sa.Index('idx_chapters_textbook_id', Chapter.textbook_id),
        sa.Index('idx_chapters_chapter_number', Chapter.chapter_number),
        sa.Index('idx_chapters_processed', Chapter.is_processed),
        sa.Index('idx_chapters_keywords', Chapter.keywords, postgresql_using='gin'),
        sa.Index('idx_chapters_medical_terms', Chapter.medical_terms, postgresql_using='gin'),
        sa.Index('idx_chapters_file_path', Chapter.file_path),

        # Citation indexes
        sa.Index('idx_citations_chapter_id', ChapterCitation.chapter_id),
        sa.Index('idx_citations_type', ChapterCitation.citation_type),
        sa.Index('idx_citations_doi', ChapterCitation.doi),
        sa.Index('idx_citations_pmid', ChapterCitation.pmid),

        # Content reference indexes
        sa.Index('idx_content_refs_chapter_id', ContentReference.chapter_id),
        sa.Index('idx_content_refs_ai_content_id', ContentReference.ai_content_id),
        sa.Index('idx_content_refs_type', ContentReference.reference_type),
        sa.Index('idx_content_refs_relevance', ContentReference.relevance_score),
        sa.Index('idx_content_refs_hash', ContentReference.ai_content_hash),

        # Search index indexes
        sa.Index('idx_search_chapter_id', ReferenceSearchIndex.chapter_id),
        sa.Index('idx_search_position', ReferenceSearchIndex.chunk_position),
    ]