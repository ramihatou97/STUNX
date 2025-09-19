"""
KOO Platform Pydantic Schemas
Request/response models for API endpoints
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from enum import Enum


class UserRole(str, Enum):
    """User roles"""
    admin = "admin"
    editor = "editor"
    viewer = "viewer"


class ChapterStatus(str, Enum):
    """Chapter status"""
    draft = "draft"
    review = "review"
    published = "published"
    archived = "archived"


class ProposalStatus(str, Enum):
    """Proposal status"""
    pending = "pending"
    approved = "approved"
    rejected = "rejected"
    skipped = "skipped"
    review = "review"


# Base response models
class APIResponse(BaseModel):
    """Standard API response"""
    message: str
    version: str
    documentation: Optional[str] = None
    health: str
    status: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    timestamp: Optional[str] = None
    services: Optional[Dict[str, Any]] = None
    environment: str
    error: Optional[str] = None


# User schemas
class UserBase(BaseModel):
    """Base user fields"""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
    full_name: str = Field(..., min_length=1, max_length=100)
    title: Optional[str] = Field(None, max_length=100)
    institution: Optional[str] = Field(None, max_length=200)
    department: Optional[str] = Field(None, max_length=100)
    specialty: Optional[str] = Field(None, max_length=100)


class UserCreate(UserBase):
    """User creation schema"""
    password: str = Field(..., min_length=8)
    role: UserRole = UserRole.viewer


class UserUpdate(BaseModel):
    """User update schema"""
    full_name: Optional[str] = Field(None, min_length=1, max_length=100)
    title: Optional[str] = Field(None, max_length=100)
    institution: Optional[str] = Field(None, max_length=200)
    department: Optional[str] = Field(None, max_length=100)
    specialty: Optional[str] = Field(None, max_length=100)
    preferences: Optional[Dict[str, Any]] = None
    notification_settings: Optional[Dict[str, Any]] = None


class UserResponse(UserBase):
    """User response schema"""
    id: int
    uuid: str
    role: UserRole
    is_active: bool
    is_verified: bool
    last_login: Optional[datetime] = None
    total_proposals: int = 0
    approved_proposals: int = 0
    total_research_queries: int = 0
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Authentication schemas
class TokenResponse(BaseModel):
    """Token response"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class LoginRequest(BaseModel):
    """Login request"""
    username: str
    password: str


# Chapter schemas
class ChapterBase(BaseModel):
    """Base chapter fields"""
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    specialty: Optional[str] = Field(None, max_length=100)
    tags: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChapterCreate(ChapterBase):
    """Chapter creation schema"""
    visibility: str = Field("public", regex="^(public|private|restricted)$")


class ChapterUpdate(BaseModel):
    """Chapter update schema"""
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    specialty: Optional[str] = Field(None, max_length=100)
    tags: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    status: Optional[ChapterStatus] = None
    visibility: Optional[str] = Field(None, regex="^(public|private|restricted)$")


class ChapterResponse(ChapterBase):
    """Chapter response schema"""
    id: int
    uuid: str
    version: int
    status: ChapterStatus
    visibility: str
    view_count: int
    edit_count: int
    proposal_count: int
    quality_score: float
    last_quality_check: Optional[datetime] = None
    created_by: int
    last_modified_by: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Chapter section schemas
class ChapterSectionBase(BaseModel):
    """Base chapter section fields"""
    title: Optional[str] = Field(None, max_length=200)
    content: str = Field(..., min_length=1)
    position: int = Field(0, ge=0)
    section_type: str = Field("text", max_length=50)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChapterSectionCreate(ChapterSectionBase):
    """Chapter section creation schema"""
    chapter_id: int


class ChapterSectionUpdate(BaseModel):
    """Chapter section update schema"""
    title: Optional[str] = Field(None, max_length=200)
    content: Optional[str] = Field(None, min_length=1)
    position: Optional[int] = Field(None, ge=0)
    section_type: Optional[str] = Field(None, max_length=50)
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class ChapterSectionResponse(ChapterSectionBase):
    """Chapter section response schema"""
    id: int
    uuid: str
    chapter_id: int
    content_hash: Optional[str] = None
    word_count: int
    quality_score: float
    readability_score: float
    last_quality_check: Optional[datetime] = None
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    external_links: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Research schemas
class ResearchQuery(BaseModel):
    """Research query schema"""
    query: str = Field(..., min_length=1, max_length=500)
    max_results: int = Field(50, ge=1, le=200)
    years_back: int = Field(5, ge=1, le=20)
    evidence_filter: Optional[str] = Field(None, regex="^(rct|systematic_review|clinical|case_series|cohort)$")
    include_clinical_trials: bool = True
    min_impact_factor: float = Field(0.0, ge=0.0)
    specialty_focus: Optional[str] = None


class NeurosurgicalPaperResponse(BaseModel):
    """Neurosurgical paper response schema"""
    pmid: str
    title: str
    authors: List[str]
    journal: str
    publication_date: str
    abstract: str
    mesh_terms: List[str]
    evidence_level: str
    surgical_relevance: float
    clinical_significance: float
    neurosurgical_score: float
    keywords: List[str]
    doi: Optional[str] = None
    pmc_id: Optional[str] = None
    url: Optional[str] = None
    citation_count: int = 0
    impact_factor: float = 0.0
    study_type: str = ""
    patient_count: Optional[int] = None
    follow_up_months: Optional[int] = None


class ResearchResponse(BaseModel):
    """Research response schema"""
    query: str
    total_results: int
    papers: List[NeurosurgicalPaperResponse]
    search_time_ms: int
    cache_hit: bool = False


# Proposal schemas
class ProposalBase(BaseModel):
    """Base proposal fields"""
    proposed_content: str = Field(..., min_length=1)
    content_type: str = Field("enhancement", regex="^(enhancement|correction|addition|update|clarification)$")
    change_summary: Optional[str] = None
    priority: str = Field("medium", regex="^(low|medium|high|critical)$")


class ProposalCreate(ProposalBase):
    """Proposal creation schema"""
    chapter_id: int
    section_id: Optional[int] = None
    source_id: Optional[int] = None


class ProposalUpdate(BaseModel):
    """Proposal update schema"""
    status: ProposalStatus
    review_notes: Optional[str] = None


class ProposalResponse(ProposalBase):
    """Proposal response schema"""
    id: int
    uuid: str
    chapter_id: int
    section_id: Optional[int] = None
    source_id: Optional[int] = None
    status: ProposalStatus
    ai_confidence: float
    quality_metrics: Dict[str, Any] = Field(default_factory=dict)
    contradiction_analysis: Dict[str, Any] = Field(default_factory=dict)
    integration_strategy: Optional[str] = None
    created_by: int
    reviewed_by: Optional[int] = None
    review_notes: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    processing_time: Optional[float] = None
    model_used: Optional[str] = None
    api_costs: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Knowledge source schemas
class KnowledgeSourceResponse(BaseModel):
    """Knowledge source response schema"""
    id: int
    uuid: str
    source_type: str
    external_id: Optional[str] = None
    title: str
    authors: List[str]
    publication_info: Dict[str, Any] = Field(default_factory=dict)
    url: Optional[str] = None
    doi: Optional[str] = None
    content_summary: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    quality_score: float
    credibility_score: float
    relevance_scores: Dict[str, Any] = Field(default_factory=dict)
    evidence_level: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    extraction_method: Optional[str] = None
    last_updated: Optional[datetime] = None
    indexed_at: datetime
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Pagination schemas
class PaginationParams(BaseModel):
    """Pagination parameters"""
    skip: int = Field(0, ge=0)
    limit: int = Field(50, ge=1, le=100)


class PaginatedResponse(BaseModel):
    """Paginated response"""
    items: List[Any]
    total: int
    skip: int
    limit: int
    has_next: bool
    has_previous: bool


# Error schemas
class ErrorDetail(BaseModel):
    """Error detail"""
    type: str
    message: str
    field: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response"""
    detail: str
    errors: Optional[List[ErrorDetail]] = None
    request_id: Optional[str] = None


# Analytics schemas
class AnalyticsQuery(BaseModel):
    """Analytics query parameters"""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    metric_type: Optional[str] = None
    granularity: str = Field("day", regex="^(hour|day|week|month)$")


class AnalyticsResponse(BaseModel):
    """Analytics response"""
    metric_name: str
    data_points: List[Dict[str, Any]]
    total_value: Optional[float] = None
    period_start: datetime
    period_end: datetime


# Admin schemas
class SystemStatus(BaseModel):
    """System status schema"""
    status: str
    version: str
    environment: str
    uptime_seconds: float
    database_healthy: bool
    cache_healthy: bool
    ai_services_healthy: bool
    active_users: int
    total_chapters: int
    pending_proposals: int
    api_usage_24h: int


class UserManagement(BaseModel):
    """User management schema"""
    user_id: int
    action: str = Field(..., regex="^(activate|deactivate|promote|demote|reset_password)$")
    role: Optional[UserRole] = None


# Validation helpers
@validator('tags', 'keywords', pre=True)
def validate_string_list(cls, v):
    """Validate string lists"""
    if isinstance(v, str):
        return [item.strip() for item in v.split(',') if item.strip()]
    return v or []


@validator('metadata', pre=True)
def validate_metadata(cls, v):
    """Validate metadata"""
    if v is None:
        return {}
    if isinstance(v, str):
        try:
            import json
            return json.loads(v)
        except json.JSONDecodeError:
            return {}
    return v