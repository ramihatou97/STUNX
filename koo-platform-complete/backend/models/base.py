"""
Base Database Models for KOO Platform
Simple SQLAlchemy models for single-user application
"""

from datetime import datetime
from typing import Optional, List
from uuid import uuid4

from sqlalchemy import String, Text, Integer, DateTime, Boolean, JSON, ARRAY
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    """Base class for all database models"""
    pass

class TimestampMixin:
    """Mixin for created_at and updated_at timestamps"""

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )

class Chapter(Base, TimestampMixin):
    """Chapter model for storing medical knowledge"""

    __tablename__ = "chapters"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # UUID for external references
    uuid: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        default=lambda: str(uuid4()),
        unique=True,
        nullable=False
    )

    # Content fields
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[Optional[str]] = mapped_column(Text)

    # Metadata
    tags: Mapped[List[str]] = mapped_column(
        ARRAY(String),
        default=list,
        nullable=False
    )
    specialty: Mapped[Optional[str]] = mapped_column(String(100))

    # Status and versioning
    status: Mapped[str] = mapped_column(
        String(20),
        default="draft",
        nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    # Metrics
    word_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    view_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Author (always admin in single-user mode)
    author_name: Mapped[str] = mapped_column(String(100), nullable=False)

    # Additional metadata
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    def __repr__(self):
        return f"<Chapter(id={self.id}, title='{self.title[:50]}...')>"

    def to_dict(self) -> dict:
        """Convert model to dictionary"""
        return {
            "id": self.id,
            "uuid": self.uuid,
            "title": self.title,
            "content": self.content,
            "summary": self.summary,
            "tags": self.tags,
            "specialty": self.specialty,
            "status": self.status,
            "version": self.version,
            "word_count": self.word_count,
            "view_count": self.view_count,
            "author_name": self.author_name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata_json
        }

class SearchQuery(Base, TimestampMixin):
    """Model for storing search queries and results"""

    __tablename__ = "search_queries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    uuid: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        default=lambda: str(uuid4()),
        unique=True,
        nullable=False
    )

    # Query details
    query_text: Mapped[str] = mapped_column(String(500), nullable=False)
    query_type: Mapped[str] = mapped_column(String(50), default="general", nullable=False)

    # Results
    results_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    results_data: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    # Performance
    execution_time_ms: Mapped[float] = mapped_column(Integer, default=0, nullable=False)

    # User context (always admin in single-user)
    user_context: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    def __repr__(self):
        return f"<SearchQuery(id={self.id}, query='{self.query_text[:30]}...')>"

class ApiUsage(Base, TimestampMixin):
    """Model for tracking API usage and costs"""

    __tablename__ = "api_usage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # Service details
    service_name: Mapped[str] = mapped_column(String(50), nullable=False)
    endpoint: Mapped[str] = mapped_column(String(200), nullable=False)
    method: Mapped[str] = mapped_column(String(10), nullable=False)

    # Request details
    request_data: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    # Response details
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    response_time_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Cost tracking
    tokens_used: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cost_usd: Mapped[float] = mapped_column(Integer, default=0, nullable=False)

    # Error tracking
    error_message: Mapped[Optional[str]] = mapped_column(Text)

    def __repr__(self):
        return f"<ApiUsage(service={self.service_name}, endpoint={self.endpoint})>"

# Model registry for easy access
MODEL_REGISTRY = {
    "Chapter": Chapter,
    "SearchQuery": SearchQuery,
    "ApiUsage": ApiUsage,
}