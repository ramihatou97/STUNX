"""
KOO Platform Database Models
SQLAlchemy models for PostgreSQL with pgvector support
"""

import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy import (
    Column, Integer, String, Text, Boolean, DateTime, Float,
    ForeignKey, JSON, Index, UniqueConstraint, CheckConstraint,
    event, text
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class TimestampMixin:
    """Mixin for created_at and updated_at timestamps"""
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class UUIDMixin:
    """Mixin for UUID primary key"""
    id = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, nullable=False, index=True)


class User(Base, UUIDMixin, TimestampMixin):
    """User model with enhanced authentication and profile data"""
    __tablename__ = "users"

    # Basic information
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)

    # Profile information
    full_name = Column(String(100), nullable=False)
    title = Column(String(100))  # Dr., Prof., etc.
    institution = Column(String(200))
    department = Column(String(100))
    specialty = Column(String(100))

    # Authentication and authorization
    role = Column(String(20), nullable=False, default="viewer")
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    last_login = Column(DateTime(timezone=True))
    failed_login_attempts = Column(Integer, default=0, nullable=False)
    locked_until = Column(DateTime(timezone=True))

    # Preferences and settings
    preferences = Column(JSON, default=dict, nullable=False)
    notification_settings = Column(JSON, default=dict, nullable=False)

    # Statistics
    total_proposals = Column(Integer, default=0, nullable=False)
    approved_proposals = Column(Integer, default=0, nullable=False)
    total_research_queries = Column(Integer, default=0, nullable=False)

    # Relationships
    created_proposals = relationship("ProposedUpdate", foreign_keys="ProposedUpdate.created_by", back_populates="creator")
    approved_proposals_rel = relationship("ProposedUpdate", foreign_keys="ProposedUpdate.approved_by", back_populates="approver")
    thoughts = relationship("ThoughtStream", back_populates="user")

    # Constraints
    __table_args__ = (
        CheckConstraint("role IN ('admin', 'editor', 'viewer')", name="valid_role"),
        CheckConstraint("failed_login_attempts >= 0", name="non_negative_failed_attempts"),
        Index("idx_users_role_active", "role", "is_active"),
        Index("idx_users_last_login", "last_login"),
    )

    @validates('email')
    def validate_email(self, key, email):
        """Validate email format"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, email):
            raise ValueError("Invalid email format")
        return email.lower()

    @validates('role')
    def validate_role(self, key, role):
        """Validate user role"""
        valid_roles = ['admin', 'editor', 'viewer']
        if role not in valid_roles:
            raise ValueError(f"Role must be one of: {valid_roles}")
        return role

    def __repr__(self):
        return f"<User(username='{self.username}', role='{self.role}')>"


class Chapter(Base, UUIDMixin, TimestampMixin):
    """Enhanced chapter model with versioning and metadata"""
    __tablename__ = "chapters_v2"

    # Basic information
    title = Column(String(200), nullable=False, index=True)
    description = Column(Text)
    specialty = Column(String(100), index=True)

    # Versioning
    version = Column(Integer, default=1, nullable=False)
    parent_chapter_id = Column(Integer, ForeignKey("chapters_v2.id"))

    # Status and workflow
    status = Column(String(20), default="draft", nullable=False)
    visibility = Column(String(20), default="public", nullable=False)

    # Metadata and classification
    metadata = Column(JSON, default=dict, nullable=False)
    tags = Column(ARRAY(String), default=list, nullable=False)
    keywords = Column(ARRAY(String), default=list, nullable=False)

    # Statistics and metrics
    view_count = Column(Integer, default=0, nullable=False)
    edit_count = Column(Integer, default=0, nullable=False)
    proposal_count = Column(Integer, default=0, nullable=False)
    quality_score = Column(Float, default=0.0, nullable=False)
    last_quality_check = Column(DateTime(timezone=True))

    # Author information
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    last_modified_by = Column(Integer, ForeignKey("users.id"))

    # Relationships
    creator = relationship("User", foreign_keys=[created_by])
    modifier = relationship("User", foreign_keys=[last_modified_by])
    sections = relationship("ChapterSection", back_populates="chapter", cascade="all, delete-orphan")
    versions = relationship("ChapterVersion", back_populates="chapter")
    proposals = relationship("ProposedUpdate", back_populates="chapter")
    parent = relationship("Chapter", remote_side=[id])
    children = relationship("Chapter")

    # Constraints
    __table_args__ = (
        CheckConstraint("status IN ('draft', 'review', 'published', 'archived')", name="valid_status"),
        CheckConstraint("visibility IN ('public', 'private', 'restricted')", name="valid_visibility"),
        CheckConstraint("version > 0", name="positive_version"),
        CheckConstraint("quality_score >= 0.0 AND quality_score <= 1.0", name="valid_quality_score"),
        Index("idx_chapters_status_specialty", "status", "specialty"),
        Index("idx_chapters_quality_score", "quality_score"),
        Index("idx_chapters_tags", "tags", postgresql_using="gin"),
        Index("idx_chapters_keywords", "keywords", postgresql_using="gin"),
        Index("idx_chapters_fulltext", text("to_tsvector('english', title || ' ' || COALESCE(description, ''))"), postgresql_using="gin"),
    )

    def __repr__(self):
        return f"<Chapter(title='{self.title}', version={self.version})>"


class ChapterSection(Base, UUIDMixin, TimestampMixin):
    """Chapter sections with vector embeddings for semantic search"""
    __tablename__ = "chapter_sections_v2"

    # Basic information
    chapter_id = Column(Integer, ForeignKey("chapters_v2.id"), nullable=False, index=True)
    title = Column(String(200), index=True)
    content = Column(Text, nullable=False)
    position = Column(Integer, default=0, nullable=False)

    # Content analysis
    embedding = Column(Vector(1536))  # OpenAI/Sentence-Transformers compatible
    content_hash = Column(String(64), index=True)  # For change detection
    word_count = Column(Integer, default=0, nullable=False)
    section_type = Column(String(50), default="text", nullable=False)

    # Metadata
    metadata = Column(JSON, default=dict, nullable=False)
    tags = Column(ARRAY(String), default=list, nullable=False)

    # Quality metrics
    quality_score = Column(Float, default=0.0, nullable=False)
    readability_score = Column(Float, default=0.0, nullable=False)
    last_quality_check = Column(DateTime(timezone=True))

    # References and citations
    citations = Column(JSON, default=list, nullable=False)  # List of citation objects
    external_links = Column(JSON, default=list, nullable=False)

    # Relationships
    chapter = relationship("Chapter", back_populates="sections")
    proposals = relationship("ProposedUpdate", back_populates="section")
    nuance_merges = relationship("NuanceMerge", back_populates="section")

    # Constraints
    __table_args__ = (
        CheckConstraint("position >= 0", name="non_negative_position"),
        CheckConstraint("word_count >= 0", name="non_negative_word_count"),
        CheckConstraint("quality_score >= 0.0 AND quality_score <= 1.0", name="valid_quality_score"),
        CheckConstraint("readability_score >= 0.0 AND readability_score <= 1.0", name="valid_readability_score"),
        Index("idx_sections_chapter_position", "chapter_id", "position"),
        Index("idx_sections_embedding", "embedding", postgresql_using="ivfflat", postgresql_ops={"embedding": "vector_cosine_ops"}),
        Index("idx_sections_content_hash", "content_hash"),
        Index("idx_sections_fulltext", text("to_tsvector('english', COALESCE(title, '') || ' ' || content)"), postgresql_using="gin"),
        UniqueConstraint("chapter_id", "position", name="unique_chapter_position"),
    )

    def __repr__(self):
        return f"<ChapterSection(title='{self.title}', chapter_id={self.chapter_id})>"


class ThoughtStream(Base, UUIDMixin, TimestampMixin):
    """Thought stream for auto-evolution system"""
    __tablename__ = "thought_stream"

    # Basic information
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    chapter_id = Column(Integer, ForeignKey("chapters_v2.id"), index=True)
    thought_content = Column(Text, nullable=False)

    # Classification
    thought_type = Column(String(50), default="question", nullable=False)  # question, insight, concern, suggestion
    priority_score = Column(Float, default=0.5, nullable=False)

    # Semantic analysis
    embedding = Column(Vector(1536))
    keywords = Column(ARRAY(String), default=list, nullable=False)
    sentiment_score = Column(Float, default=0.0, nullable=False)

    # Processing
    processed = Column(Boolean, default=False, nullable=False)
    processed_at = Column(DateTime(timezone=True))
    processing_result = Column(JSON, default=dict, nullable=False)

    # Relationships
    user = relationship("User", back_populates="thoughts")
    chapter = relationship("Chapter")

    # Constraints
    __table_args__ = (
        CheckConstraint("thought_type IN ('question', 'insight', 'concern', 'suggestion', 'feedback')", name="valid_thought_type"),
        CheckConstraint("priority_score >= 0.0 AND priority_score <= 1.0", name="valid_priority_score"),
        CheckConstraint("sentiment_score >= -1.0 AND sentiment_score <= 1.0", name="valid_sentiment_score"),
        Index("idx_thoughts_user_chapter", "user_id", "chapter_id"),
        Index("idx_thoughts_processed", "processed", "created_at"),
        Index("idx_thoughts_embedding", "embedding", postgresql_using="ivfflat", postgresql_ops={"embedding": "vector_cosine_ops"}),
        Index("idx_thoughts_keywords", "keywords", postgresql_using="gin"),
    )

    def __repr__(self):
        return f"<ThoughtStream(type='{self.thought_type}', processed={self.processed})>"


class KnowledgeSource(Base, UUIDMixin, TimestampMixin):
    """Knowledge sources with enhanced metadata"""
    __tablename__ = "knowledge_sources"

    # Source identification
    source_type = Column(String(50), nullable=False, index=True)  # pubmed, arxiv, perplexity, textbook, manual
    external_id = Column(String(255), index=True)  # DOI, PMID, arXiv ID, etc.

    # Content information
    title = Column(Text, nullable=False)
    authors = Column(ARRAY(String), default=list, nullable=False)
    publication_info = Column(JSON, default=dict, nullable=False)
    url = Column(Text)
    doi = Column(String(255), index=True)

    # Content analysis
    content_summary = Column(Text)
    embedding = Column(Vector(1536))
    keywords = Column(ARRAY(String), default=list, nullable=False)

    # Quality assessment
    quality_score = Column(Float, default=0.0, nullable=False)
    credibility_score = Column(Float, default=0.0, nullable=False)
    relevance_scores = Column(JSON, default=dict, nullable=False)  # Per-chapter relevance
    evidence_level = Column(String(50))  # systematic_review, rct, etc.

    # Processing metadata
    metadata = Column(JSON, default=dict, nullable=False)
    extraction_method = Column(String(100))
    last_updated = Column(DateTime(timezone=True))
    indexed_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    proposals = relationship("ProposedUpdate", back_populates="source")

    # Constraints
    __table_args__ = (
        CheckConstraint("quality_score >= 0.0 AND quality_score <= 1.0", name="valid_quality_score"),
        CheckConstraint("credibility_score >= 0.0 AND credibility_score <= 1.0", name="valid_credibility_score"),
        Index("idx_sources_type_quality", "source_type", "quality_score"),
        Index("idx_sources_embedding", "embedding", postgresql_using="ivfflat", postgresql_ops={"embedding": "vector_cosine_ops"}),
        Index("idx_sources_keywords", "keywords", postgresql_using="gin"),
        Index("idx_sources_fulltext", text("to_tsvector('english', title || ' ' || COALESCE(content_summary, ''))"), postgresql_using="gin"),
        UniqueConstraint("source_type", "external_id", name="unique_source_external_id"),
    )

    def __repr__(self):
        return f"<KnowledgeSource(type='{self.source_type}', external_id='{self.external_id}')>"


class ProposedUpdate(Base, UUIDMixin, TimestampMixin):
    """Enhanced proposed updates with AI analysis"""
    __tablename__ = "proposed_updates_v2"

    # Target information
    chapter_id = Column(Integer, ForeignKey("chapters_v2.id"), nullable=False, index=True)
    section_id = Column(Integer, ForeignKey("chapter_sections_v2.id"), index=True)
    source_id = Column(Integer, ForeignKey("knowledge_sources.id"), index=True)

    # Proposal content
    proposed_content = Column(Text, nullable=False)
    content_type = Column(String(50), default="enhancement", nullable=False)
    change_summary = Column(Text)

    # AI analysis
    ai_confidence = Column(Float, default=0.0, nullable=False)
    quality_metrics = Column(JSON, default=dict, nullable=False)
    contradiction_analysis = Column(JSON, default=dict, nullable=False)
    integration_strategy = Column(Text)

    # Workflow
    status = Column(String(20), default="pending", nullable=False)
    priority = Column(String(10), default="medium", nullable=False)

    # Review information
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    reviewed_by = Column(Integer, ForeignKey("users.id"))
    review_notes = Column(Text)
    reviewed_at = Column(DateTime(timezone=True))

    # Processing metadata
    processing_time = Column(Float)  # Seconds
    model_used = Column(String(100))
    api_costs = Column(JSON, default=dict, nullable=False)

    # Relationships
    chapter = relationship("Chapter", back_populates="proposals")
    section = relationship("ChapterSection", back_populates="proposals")
    source = relationship("KnowledgeSource", back_populates="proposals")
    creator = relationship("User", foreign_keys=[created_by], back_populates="created_proposals")
    approver = relationship("User", foreign_keys=[reviewed_by], back_populates="approved_proposals_rel")
    nuance_merges = relationship("NuanceMerge", back_populates="update")

    # Constraints
    __table_args__ = (
        CheckConstraint("status IN ('pending', 'approved', 'rejected', 'skipped', 'review')", name="valid_status"),
        CheckConstraint("priority IN ('low', 'medium', 'high', 'critical')", name="valid_priority"),
        CheckConstraint("content_type IN ('enhancement', 'correction', 'addition', 'update', 'clarification')", name="valid_content_type"),
        CheckConstraint("ai_confidence >= 0.0 AND ai_confidence <= 1.0", name="valid_ai_confidence"),
        Index("idx_proposals_status_created", "status", "created_at"),
        Index("idx_proposals_chapter_status", "chapter_id", "status"),
        Index("idx_proposals_priority_created", "priority", "created_at"),
    )

    def __repr__(self):
        return f"<ProposedUpdate(status='{self.status}', chapter_id={self.chapter_id})>"


class EvolutionEvent(Base, UUIDMixin, TimestampMixin):
    """Evolution events tracking for auto-evolution system"""
    __tablename__ = "evolution_events"

    # Event identification
    chapter_id = Column(Integer, ForeignKey("chapters_v2.id"), nullable=False, index=True)
    event_type = Column(String(50), nullable=False, index=True)
    trigger_source = Column(String(50), index=True)  # thought_stream, scheduled, manual, api

    # Event data
    changes_made = Column(JSON, default=dict, nullable=False)
    before_state = Column(JSON, default=dict, nullable=False)
    after_state = Column(JSON, default=dict, nullable=False)

    # Impact assessment
    impact_score = Column(Float, default=0.0, nullable=False)
    affected_sections = Column(ARRAY(Integer), default=list, nullable=False)
    quality_change = Column(Float, default=0.0, nullable=False)

    # User feedback
    user_feedback = Column(JSON, default=dict, nullable=False)
    feedback_score = Column(Float)

    # Processing information
    processing_time = Column(Float)
    models_used = Column(ARRAY(String), default=list, nullable=False)

    # Relationships
    chapter = relationship("Chapter")

    # Constraints
    __table_args__ = (
        CheckConstraint("impact_score >= 0.0 AND impact_score <= 1.0", name="valid_impact_score"),
        CheckConstraint("quality_change >= -1.0 AND quality_change <= 1.0", name="valid_quality_change"),
        CheckConstraint("feedback_score IS NULL OR (feedback_score >= 0.0 AND feedback_score <= 1.0)", name="valid_feedback_score"),
        Index("idx_evolution_chapter_type", "chapter_id", "event_type"),
        Index("idx_evolution_trigger_created", "trigger_source", "created_at"),
    )

    def __repr__(self):
        return f"<EvolutionEvent(type='{self.event_type}', chapter_id={self.chapter_id})>"


class NuanceMerge(Base, UUIDMixin, TimestampMixin):
    """Enhanced nuance merges with detailed tracking"""
    __tablename__ = "nuance_merges_v2"

    # Reference information
    update_id = Column(Integer, ForeignKey("proposed_updates_v2.id"), nullable=False, index=True)
    chapter_id = Column(Integer, ForeignKey("chapters_v2.id"), nullable=False, index=True)
    section_id = Column(Integer, ForeignKey("chapter_sections_v2.id"), index=True)

    # Merge details
    original_content = Column(Text, nullable=False)
    merged_content = Column(Text, nullable=False)
    merge_strategy = Column(String(50), nullable=False)

    # Analysis
    similarity_score = Column(Float, nullable=False)
    nuance_differences = Column(JSON, default=dict, nullable=False)
    confidence_score = Column(Float, default=0.0, nullable=False)

    # Review process
    human_review_required = Column(Boolean, default=False, nullable=False)
    auto_merge_confidence = Column(Float, default=0.0, nullable=False)
    reviewed = Column(Boolean, default=False, nullable=False)
    review_feedback = Column(Text)

    # Processing metadata
    merged_by = Column(Integer, ForeignKey("users.id"))
    merge_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    processing_time = Column(Float)

    # Relationships
    update = relationship("ProposedUpdate", back_populates="nuance_merges")
    chapter = relationship("Chapter")
    section = relationship("ChapterSection", back_populates="nuance_merges")
    merger = relationship("User")

    # Constraints
    __table_args__ = (
        CheckConstraint("similarity_score >= 0.0 AND similarity_score <= 1.0", name="valid_similarity_score"),
        CheckConstraint("confidence_score >= 0.0 AND confidence_score <= 1.0", name="valid_confidence_score"),
        CheckConstraint("auto_merge_confidence >= 0.0 AND auto_merge_confidence <= 1.0", name="valid_auto_merge_confidence"),
        CheckConstraint("merge_strategy IN ('word_level', 'sentence_level', 'paragraph_level', 'manual')", name="valid_merge_strategy"),
        Index("idx_nuance_chapter_timestamp", "chapter_id", "merge_timestamp"),
        Index("idx_nuance_review_required", "human_review_required", "reviewed"),
    )

    def __repr__(self):
        return f"<NuanceMerge(strategy='{self.merge_strategy}', similarity={self.similarity_score:.2f})>"


class ChapterVersion(Base, UUIDMixin, TimestampMixin):
    """Chapter version history with enhanced metadata"""
    __tablename__ = "chapter_versions"

    # Version information
    chapter_id = Column(Integer, ForeignKey("chapters_v2.id"), nullable=False, index=True)
    version_number = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)

    # Change information
    change_summary = Column(Text)
    change_type = Column(String(50))  # major, minor, patch, hotfix
    changes_count = Column(Integer, default=0, nullable=False)

    # Metadata
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    tags = Column(ARRAY(String), default=list, nullable=False)
    metadata = Column(JSON, default=dict, nullable=False)

    # Quality metrics
    quality_score = Column(Float, default=0.0, nullable=False)
    content_hash = Column(String(64), index=True)
    word_count = Column(Integer, default=0, nullable=False)

    # Relationships
    chapter = relationship("Chapter", back_populates="versions")
    creator = relationship("User")

    # Constraints
    __table_args__ = (
        CheckConstraint("version_number > 0", name="positive_version_number"),
        CheckConstraint("changes_count >= 0", name="non_negative_changes_count"),
        CheckConstraint("quality_score >= 0.0 AND quality_score <= 1.0", name="valid_quality_score"),
        CheckConstraint("word_count >= 0", name="non_negative_word_count"),
        UniqueConstraint("chapter_id", "version_number", name="unique_chapter_version"),
        Index("idx_versions_chapter_version", "chapter_id", "version_number"),
    )

    def __repr__(self):
        return f"<ChapterVersion(chapter_id={self.chapter_id}, version={self.version_number})>"


class APIUsageLog(Base, UUIDMixin, TimestampMixin):
    """API usage tracking for cost management"""
    __tablename__ = "api_usage_log"

    # API information
    api_service = Column(String(50), nullable=False, index=True)
    endpoint = Column(String(255))
    method = Column(String(10))

    # Request information
    request_data = Column(JSON, default=dict, nullable=False)
    response_status = Column(Integer, index=True)
    response_time_ms = Column(Integer)

    # Usage metrics
    tokens_used = Column(Integer, default=0, nullable=False)
    cost_usd = Column(Float, default=0.0, nullable=False)

    # User information
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    ip_address = Column(String(45))  # IPv6 compatible
    user_agent = Column(Text)

    # Relationships
    user = relationship("User")

    # Constraints
    __table_args__ = (
        CheckConstraint("tokens_used >= 0", name="non_negative_tokens"),
        CheckConstraint("cost_usd >= 0.0", name="non_negative_cost"),
        CheckConstraint("response_time_ms >= 0", name="non_negative_response_time"),
        Index("idx_api_usage_service_created", "api_service", "created_at"),
        Index("idx_api_usage_user_created", "user_id", "created_at"),
        Index("idx_api_usage_cost_created", "cost_usd", "created_at"),
    )

    def __repr__(self):
        return f"<APIUsageLog(service='{self.api_service}', cost=${self.cost_usd:.4f})>"


# Event listeners for automatic updates

@event.listens_for(ChapterSection, 'before_insert')
@event.listens_for(ChapterSection, 'before_update')
def update_content_hash(mapper, connection, target):
    """Automatically update content hash when content changes"""
    import hashlib
    if target.content:
        target.content_hash = hashlib.sha256(target.content.encode()).hexdigest()
        # Simple word count
        target.word_count = len(target.content.split())


@event.listens_for(Chapter, 'before_update')
def increment_edit_count(mapper, connection, target):
    """Increment edit count when chapter is updated"""
    target.edit_count += 1


@event.listens_for(User, 'before_update')
def update_login_stats(mapper, connection, target):
    """Update login statistics"""
    if target.last_login and hasattr(target, '_is_login_update'):
        # This would be set by the authentication service
        target.failed_login_attempts = 0


# Database utility functions
def create_indexes():
    """Create additional indexes for performance"""
    from sqlalchemy import text

    indexes = [
        # Performance indexes
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chapters_created_at_desc ON chapters_v2 (created_at DESC)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sections_updated_at_desc ON chapter_sections_v2 (updated_at DESC)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_proposals_created_at_desc ON proposed_updates_v2 (created_at DESC)",

        # Composite indexes for common queries
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_proposals_status_priority_created ON proposed_updates_v2 (status, priority, created_at DESC)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_nuance_chapter_reviewed ON nuance_merges_v2 (chapter_id, reviewed, merge_timestamp DESC)",

        # Partial indexes for active data
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_active_users ON users (id, last_login) WHERE is_active = true",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_pending_proposals ON proposed_updates_v2 (created_at DESC) WHERE status = 'pending'",
    ]

    return indexes


def get_table_sizes():
    """Get table sizes for monitoring"""
    from sqlalchemy import text

    query = text("""
        SELECT
            schemaname,
            tablename,
            attname,
            n_distinct,
            correlation,
            most_common_vals,
            most_common_freqs
        FROM pg_stats
        WHERE schemaname = 'public'
        AND tablename IN (
            'users', 'chapters_v2', 'chapter_sections_v2',
            'proposed_updates_v2', 'knowledge_sources',
            'nuance_merges_v2', 'thought_stream'
        )
        ORDER BY tablename, attname;
    """)

    return query