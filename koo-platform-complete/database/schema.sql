-- KOO Platform Database Schema v2.0
-- PostgreSQL 14+ with pgvector extension
-- Production-ready schema with full indexing and constraints

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Custom types
CREATE TYPE user_role AS ENUM ('viewer', 'editor', 'admin', 'super_admin');
CREATE TYPE chapter_status AS ENUM ('draft', 'review', 'published', 'archived');
CREATE TYPE conflict_type AS ENUM ('contradiction', 'outdated', 'uncertain', 'incomplete');
CREATE TYPE severity_level AS ENUM ('low', 'medium', 'high', 'critical');
CREATE TYPE api_source AS ENUM ('pubmed', 'semantic_scholar', 'perplexity', 'elsevier', 'biodigital');
CREATE TYPE insight_type AS ENUM ('suggestion', 'improvement', 'research_gap', 'contradiction');

-- Users table with enhanced authentication
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,

    -- Profile information
    full_name VARCHAR(100) NOT NULL,
    title VARCHAR(100),
    institution VARCHAR(200),
    department VARCHAR(100),
    specialty VARCHAR(100),
    bio TEXT,

    -- Authentication and authorization
    role user_role DEFAULT 'viewer' NOT NULL,
    is_active BOOLEAN DEFAULT true NOT NULL,
    is_verified BOOLEAN DEFAULT false NOT NULL,

    -- Security tracking
    last_login TIMESTAMPTZ,
    failed_login_attempts INTEGER DEFAULT 0 NOT NULL,
    locked_until TIMESTAMPTZ,
    password_reset_token VARCHAR(255),
    password_reset_expires TIMESTAMPTZ,
    email_verification_token VARCHAR(255),

    -- Preferences
    preferences JSONB DEFAULT '{}' NOT NULL,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- Chapters table with versioning and AI features
CREATE TABLE chapters (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    title VARCHAR(500) NOT NULL,
    slug VARCHAR(200) UNIQUE NOT NULL,

    -- Content
    summary TEXT,
    content TEXT NOT NULL,
    content_html TEXT,

    -- Metadata
    tags TEXT[] DEFAULT ARRAY[]::TEXT[],
    keywords TEXT[] DEFAULT ARRAY[]::TEXT[],
    specialty VARCHAR(100),
    difficulty_level INTEGER DEFAULT 1 CHECK (difficulty_level BETWEEN 1 AND 5),

    -- Status and versioning
    status chapter_status DEFAULT 'draft' NOT NULL,
    version INTEGER DEFAULT 1 NOT NULL,
    is_template BOOLEAN DEFAULT false,

    -- AI and quality metrics
    confidence_score FLOAT DEFAULT 0.0 CHECK (confidence_score BETWEEN 0 AND 1),
    last_ai_review TIMESTAMPTZ,
    last_content_update TIMESTAMPTZ,

    -- Vector embeddings for semantic search
    content_embedding vector(1536), -- OpenAI embedding size
    summary_embedding vector(1536),

    -- Relationships
    author_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    parent_chapter_id INTEGER REFERENCES chapters(id) ON DELETE CASCADE,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    published_at TIMESTAMPTZ
);

-- Chapter sections for granular content management
CREATE TABLE chapter_sections (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    chapter_id INTEGER REFERENCES chapters(id) ON DELETE CASCADE NOT NULL,

    -- Content
    title VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,
    content_html TEXT,
    order_index INTEGER NOT NULL,

    -- Quality metrics
    confidence_score FLOAT DEFAULT 0.0 CHECK (confidence_score BETWEEN 0 AND 1),
    last_verified TIMESTAMPTZ DEFAULT NOW(),
    verification_source VARCHAR(200),

    -- Sources and references
    sources TEXT[] DEFAULT ARRAY[]::TEXT[],
    references JSONB DEFAULT '[]' NOT NULL,

    -- Vector embedding
    content_embedding vector(1536),

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- Knowledge sources tracking
CREATE TABLE knowledge_sources (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,

    -- Source identification
    source_type api_source NOT NULL,
    external_id VARCHAR(100),
    title VARCHAR(1000) NOT NULL,
    authors TEXT[] DEFAULT ARRAY[]::TEXT[],

    -- Publication details
    journal VARCHAR(300),
    publication_date DATE,
    doi VARCHAR(100),
    pmid VARCHAR(20),
    url TEXT,

    -- Content
    abstract TEXT,
    full_text TEXT,
    keywords TEXT[] DEFAULT ARRAY[]::TEXT[],

    -- Quality and relevance
    relevance_score FLOAT DEFAULT 0.0 CHECK (relevance_score BETWEEN 0 AND 1),
    citation_count INTEGER DEFAULT 0,
    impact_factor FLOAT,

    -- Vector embeddings
    title_embedding vector(1536),
    abstract_embedding vector(1536),

    -- Metadata
    metadata JSONB DEFAULT '{}' NOT NULL,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    last_accessed TIMESTAMPTZ DEFAULT NOW()
);

-- Conflicts and contradictions tracking
CREATE TABLE knowledge_conflicts (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,

    -- Related entities
    chapter_id INTEGER REFERENCES chapters(id) ON DELETE CASCADE,
    section_id INTEGER REFERENCES chapter_sections(id) ON DELETE CASCADE,
    source_1_id INTEGER REFERENCES knowledge_sources(id),
    source_2_id INTEGER REFERENCES knowledge_sources(id),

    -- Conflict details
    conflict_type conflict_type NOT NULL,
    severity severity_level DEFAULT 'medium' NOT NULL,
    description TEXT NOT NULL,
    auto_detected BOOLEAN DEFAULT false,

    -- Resolution
    is_resolved BOOLEAN DEFAULT false,
    resolution_notes TEXT,
    resolved_by INTEGER REFERENCES users(id),
    resolved_at TIMESTAMPTZ,

    -- AI suggestions
    ai_suggestions JSONB DEFAULT '[]' NOT NULL,
    confidence_level FLOAT DEFAULT 0.0 CHECK (confidence_level BETWEEN 0 AND 1),

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- Thought stream for auto-evolution
CREATE TABLE thought_stream (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,

    -- Related entities
    chapter_id INTEGER REFERENCES chapters(id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,

    -- Stream content
    thought_type VARCHAR(50) NOT NULL, -- 'insight', 'question', 'update', 'research'
    content TEXT NOT NULL,
    context JSONB DEFAULT '{}' NOT NULL,

    -- AI processing
    is_processed BOOLEAN DEFAULT false,
    ai_analysis JSONB,
    processing_status VARCHAR(50) DEFAULT 'pending',

    -- Impact tracking
    influenced_updates JSONB DEFAULT '[]' NOT NULL,
    impact_score FLOAT DEFAULT 0.0,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    processed_at TIMESTAMPTZ
);

-- AI insights and suggestions
CREATE TABLE ai_insights (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,

    -- Related entities
    chapter_id INTEGER REFERENCES chapters(id) ON DELETE CASCADE,
    section_id INTEGER REFERENCES chapter_sections(id) ON DELETE CASCADE,

    -- Insight details
    insight_type insight_type NOT NULL,
    title VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,

    -- Relevance and quality
    relevance_score FLOAT DEFAULT 0.0 CHECK (relevance_score BETWEEN 0 AND 1),
    confidence_score FLOAT DEFAULT 0.0 CHECK (confidence_score BETWEEN 0 AND 1),

    -- User interaction
    is_reviewed BOOLEAN DEFAULT false,
    is_applied BOOLEAN DEFAULT false,
    user_feedback INTEGER CHECK (user_feedback BETWEEN 1 AND 5),

    -- Source information
    generated_by VARCHAR(100) NOT NULL, -- AI model identifier
    source_data JSONB DEFAULT '{}' NOT NULL,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    reviewed_at TIMESTAMPTZ,
    applied_at TIMESTAMPTZ
);

-- User bookmarks and favorites
CREATE TABLE user_bookmarks (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE NOT NULL,
    chapter_id INTEGER REFERENCES chapters(id) ON DELETE CASCADE,
    source_id INTEGER REFERENCES knowledge_sources(id) ON DELETE CASCADE,

    -- Bookmark details
    bookmark_type VARCHAR(20) NOT NULL, -- 'chapter', 'source'
    notes TEXT,
    tags TEXT[] DEFAULT ARRAY[]::TEXT[],

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,

    CONSTRAINT bookmark_entity_check CHECK (
        (chapter_id IS NOT NULL AND source_id IS NULL) OR
        (chapter_id IS NULL AND source_id IS NOT NULL)
    )
);

-- API usage tracking for cost management
CREATE TABLE api_usage_logs (
    id SERIAL PRIMARY KEY,

    -- API details
    service_name api_source NOT NULL,
    endpoint VARCHAR(200) NOT NULL,
    method VARCHAR(10) NOT NULL,

    -- Request details
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    request_id UUID DEFAULT uuid_generate_v4(),
    query_params JSONB DEFAULT '{}' NOT NULL,

    -- Response details
    status_code INTEGER NOT NULL,
    response_time_ms INTEGER,
    tokens_used INTEGER DEFAULT 0,
    cost_usd DECIMAL(10, 6) DEFAULT 0,

    -- Rate limiting
    rate_limit_remaining INTEGER,
    rate_limit_reset TIMESTAMPTZ,

    -- Error tracking
    error_message TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- System notifications
CREATE TABLE notifications (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,

    -- Recipient
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE NOT NULL,

    -- Notification details
    title VARCHAR(500) NOT NULL,
    message TEXT NOT NULL,
    notification_type VARCHAR(50) NOT NULL,
    priority INTEGER DEFAULT 1 CHECK (priority BETWEEN 1 AND 5),

    -- Status
    is_read BOOLEAN DEFAULT false,
    is_dismissed BOOLEAN DEFAULT false,

    -- Related entities
    related_entity_type VARCHAR(50),
    related_entity_id INTEGER,

    -- Metadata
    metadata JSONB DEFAULT '{}' NOT NULL,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    read_at TIMESTAMPTZ,
    dismissed_at TIMESTAMPTZ
);

-- Audit log for compliance and tracking
CREATE TABLE audit_logs (
    id SERIAL PRIMARY KEY,

    -- Actor
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    session_id VARCHAR(100),
    ip_address INET,
    user_agent TEXT,

    -- Action details
    action VARCHAR(100) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    entity_id INTEGER,

    -- Changes
    old_values JSONB,
    new_values JSONB,

    -- Context
    context JSONB DEFAULT '{}' NOT NULL,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- Performance and analytics
CREATE TABLE analytics_events (
    id SERIAL PRIMARY KEY,

    -- Event details
    event_name VARCHAR(100) NOT NULL,
    event_category VARCHAR(50) NOT NULL,

    -- User context
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    session_id VARCHAR(100),

    -- Event data
    properties JSONB DEFAULT '{}' NOT NULL,
    duration_ms INTEGER,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- INDEXES for performance optimization

-- Users indexes
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_role ON users(role);
CREATE INDEX idx_users_active ON users(is_active) WHERE is_active = true;
CREATE INDEX idx_users_last_login ON users(last_login DESC);

-- Chapters indexes
CREATE INDEX idx_chapters_status ON chapters(status);
CREATE INDEX idx_chapters_author ON chapters(author_id);
CREATE INDEX idx_chapters_tags ON chapters USING GIN(tags);
CREATE INDEX idx_chapters_keywords ON chapters USING GIN(keywords);
CREATE INDEX idx_chapters_specialty ON chapters(specialty);
CREATE INDEX idx_chapters_updated ON chapters(updated_at DESC);
CREATE INDEX idx_chapters_published ON chapters(published_at DESC) WHERE published_at IS NOT NULL;
CREATE INDEX idx_chapters_confidence ON chapters(confidence_score DESC);
CREATE INDEX idx_chapters_text_search ON chapters USING GIN(to_tsvector('english', title || ' ' || summary || ' ' || content));

-- Vector similarity indexes
CREATE INDEX idx_chapters_content_embedding ON chapters USING ivfflat (content_embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_chapters_summary_embedding ON chapters USING ivfflat (summary_embedding vector_cosine_ops) WITH (lists = 100);

-- Chapter sections indexes
CREATE INDEX idx_sections_chapter ON chapter_sections(chapter_id);
CREATE INDEX idx_sections_order ON chapter_sections(chapter_id, order_index);
CREATE INDEX idx_sections_confidence ON chapter_sections(confidence_score DESC);
CREATE INDEX idx_sections_updated ON chapter_sections(updated_at DESC);
CREATE INDEX idx_sections_embedding ON chapter_sections USING ivfflat (content_embedding vector_cosine_ops) WITH (lists = 100);

-- Knowledge sources indexes
CREATE INDEX idx_sources_type ON knowledge_sources(source_type);
CREATE INDEX idx_sources_external_id ON knowledge_sources(source_type, external_id);
CREATE INDEX idx_sources_doi ON knowledge_sources(doi) WHERE doi IS NOT NULL;
CREATE INDEX idx_sources_pmid ON knowledge_sources(pmid) WHERE pmid IS NOT NULL;
CREATE INDEX idx_sources_relevance ON knowledge_sources(relevance_score DESC);
CREATE INDEX idx_sources_date ON knowledge_sources(publication_date DESC);
CREATE INDEX idx_sources_updated ON knowledge_sources(updated_at DESC);
CREATE INDEX idx_sources_title_embedding ON knowledge_sources USING ivfflat (title_embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_sources_abstract_embedding ON knowledge_sources USING ivfflat (abstract_embedding vector_cosine_ops) WITH (lists = 100);

-- Conflicts indexes
CREATE INDEX idx_conflicts_chapter ON knowledge_conflicts(chapter_id);
CREATE INDEX idx_conflicts_section ON knowledge_conflicts(section_id);
CREATE INDEX idx_conflicts_type ON knowledge_conflicts(conflict_type);
CREATE INDEX idx_conflicts_severity ON knowledge_conflicts(severity);
CREATE INDEX idx_conflicts_unresolved ON knowledge_conflicts(is_resolved) WHERE is_resolved = false;
CREATE INDEX idx_conflicts_created ON knowledge_conflicts(created_at DESC);

-- Thought stream indexes
CREATE INDEX idx_thought_stream_chapter ON thought_stream(chapter_id);
CREATE INDEX idx_thought_stream_user ON thought_stream(user_id);
CREATE INDEX idx_thought_stream_type ON thought_stream(thought_type);
CREATE INDEX idx_thought_stream_unprocessed ON thought_stream(is_processed) WHERE is_processed = false;
CREATE INDEX idx_thought_stream_created ON thought_stream(created_at DESC);

-- AI insights indexes
CREATE INDEX idx_insights_chapter ON ai_insights(chapter_id);
CREATE INDEX idx_insights_section ON ai_insights(section_id);
CREATE INDEX idx_insights_type ON ai_insights(insight_type);
CREATE INDEX idx_insights_relevance ON ai_insights(relevance_score DESC);
CREATE INDEX idx_insights_unreviewed ON ai_insights(is_reviewed) WHERE is_reviewed = false;
CREATE INDEX idx_insights_created ON ai_insights(created_at DESC);

-- Bookmarks indexes
CREATE INDEX idx_bookmarks_user ON user_bookmarks(user_id);
CREATE INDEX idx_bookmarks_chapter ON user_bookmarks(chapter_id) WHERE chapter_id IS NOT NULL;
CREATE INDEX idx_bookmarks_source ON user_bookmarks(source_id) WHERE source_id IS NOT NULL;
CREATE INDEX idx_bookmarks_type ON user_bookmarks(bookmark_type);
CREATE INDEX idx_bookmarks_created ON user_bookmarks(created_at DESC);

-- API usage indexes
CREATE INDEX idx_api_usage_service ON api_usage_logs(service_name);
CREATE INDEX idx_api_usage_user ON api_usage_logs(user_id);
CREATE INDEX idx_api_usage_created ON api_usage_logs(created_at DESC);
CREATE INDEX idx_api_usage_cost ON api_usage_logs(created_at DESC, cost_usd DESC);

-- Notifications indexes
CREATE INDEX idx_notifications_user ON notifications(user_id);
CREATE INDEX idx_notifications_unread ON notifications(user_id, is_read) WHERE is_read = false;
CREATE INDEX idx_notifications_type ON notifications(notification_type);
CREATE INDEX idx_notifications_created ON notifications(created_at DESC);

-- Audit logs indexes
CREATE INDEX idx_audit_user ON audit_logs(user_id);
CREATE INDEX idx_audit_action ON audit_logs(action);
CREATE INDEX idx_audit_entity ON audit_logs(entity_type, entity_id);
CREATE INDEX idx_audit_created ON audit_logs(created_at DESC);

-- Analytics indexes
CREATE INDEX idx_analytics_event ON analytics_events(event_name);
CREATE INDEX idx_analytics_category ON analytics_events(event_category);
CREATE INDEX idx_analytics_user ON analytics_events(user_id);
CREATE INDEX idx_analytics_created ON analytics_events(created_at DESC);

-- TRIGGERS for automatic updates

-- Update timestamp trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply to all tables with updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_chapters_updated_at BEFORE UPDATE ON chapters FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_sections_updated_at BEFORE UPDATE ON chapter_sections FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_sources_updated_at BEFORE UPDATE ON knowledge_sources FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_conflicts_updated_at BEFORE UPDATE ON knowledge_conflicts FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Audit logging trigger
CREATE OR REPLACE FUNCTION audit_changes()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO audit_logs (action, entity_type, entity_id, new_values)
        VALUES ('INSERT', TG_TABLE_NAME, NEW.id, to_jsonb(NEW));
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit_logs (action, entity_type, entity_id, old_values, new_values)
        VALUES ('UPDATE', TG_TABLE_NAME, NEW.id, to_jsonb(OLD), to_jsonb(NEW));
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO audit_logs (action, entity_type, entity_id, old_values)
        VALUES ('DELETE', TG_TABLE_NAME, OLD.id, to_jsonb(OLD));
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ language 'plpgsql';

-- Apply audit triggers to critical tables
CREATE TRIGGER audit_users AFTER INSERT OR UPDATE OR DELETE ON users FOR EACH ROW EXECUTE FUNCTION audit_changes();
CREATE TRIGGER audit_chapters AFTER INSERT OR UPDATE OR DELETE ON chapters FOR EACH ROW EXECUTE FUNCTION audit_changes();

-- FUNCTIONS for enhanced functionality

-- Semantic search function using vector similarity
CREATE OR REPLACE FUNCTION semantic_search_chapters(
    query_embedding vector(1536),
    similarity_threshold float DEFAULT 0.7,
    result_limit int DEFAULT 20
)
RETURNS TABLE(
    chapter_id int,
    title varchar,
    summary text,
    similarity_score float
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.id,
        c.title,
        c.summary,
        1 - (c.content_embedding <=> query_embedding) as similarity
    FROM chapters c
    WHERE c.status = 'published'
        AND 1 - (c.content_embedding <=> query_embedding) >= similarity_threshold
    ORDER BY c.content_embedding <=> query_embedding
    LIMIT result_limit;
END;
$$ LANGUAGE plpgsql;

-- Get chapter conflicts summary
CREATE OR REPLACE FUNCTION get_chapter_conflicts_summary(chapter_id_param int)
RETURNS TABLE(
    total_conflicts bigint,
    high_severity bigint,
    unresolved bigint,
    avg_confidence float
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*) as total_conflicts,
        COUNT(*) FILTER (WHERE severity IN ('high', 'critical')) as high_severity,
        COUNT(*) FILTER (WHERE is_resolved = false) as unresolved,
        AVG(confidence_level) as avg_confidence
    FROM knowledge_conflicts
    WHERE chapter_id = chapter_id_param;
END;
$$ LANGUAGE plpgsql;

-- VIEWS for common queries

-- Active chapters with metrics
CREATE VIEW active_chapters_summary AS
SELECT
    c.id,
    c.uuid,
    c.title,
    c.status,
    c.confidence_score,
    c.updated_at,
    u.full_name as author_name,
    COUNT(cs.id) as section_count,
    COUNT(kc.id) FILTER (WHERE kc.is_resolved = false) as unresolved_conflicts,
    COUNT(ai.id) FILTER (WHERE ai.is_reviewed = false) as pending_insights
FROM chapters c
LEFT JOIN users u ON c.author_id = u.id
LEFT JOIN chapter_sections cs ON c.id = cs.chapter_id
LEFT JOIN knowledge_conflicts kc ON c.id = kc.chapter_id
LEFT JOIN ai_insights ai ON c.id = ai.chapter_id
WHERE c.status IN ('draft', 'review', 'published')
GROUP BY c.id, c.uuid, c.title, c.status, c.confidence_score, c.updated_at, u.full_name;

-- User activity summary
CREATE VIEW user_activity_summary AS
SELECT
    u.id,
    u.uuid,
    u.full_name,
    u.role,
    u.last_login,
    COUNT(DISTINCT c.id) as chapters_authored,
    COUNT(DISTINCT ub.id) as bookmarks_count,
    COUNT(DISTINCT al.id) as recent_actions
FROM users u
LEFT JOIN chapters c ON u.id = c.author_id
LEFT JOIN user_bookmarks ub ON u.id = ub.user_id
LEFT JOIN audit_logs al ON u.id = al.user_id AND al.created_at > NOW() - INTERVAL '30 days'
GROUP BY u.id, u.uuid, u.full_name, u.role, u.last_login;

-- API usage costs summary
CREATE VIEW api_costs_summary AS
SELECT
    service_name,
    DATE_TRUNC('day', created_at) as usage_date,
    COUNT(*) as request_count,
    SUM(tokens_used) as total_tokens,
    SUM(cost_usd) as total_cost,
    AVG(response_time_ms) as avg_response_time
FROM api_usage_logs
WHERE created_at > NOW() - INTERVAL '30 days'
GROUP BY service_name, DATE_TRUNC('day', created_at)
ORDER BY usage_date DESC, total_cost DESC;

-- CONSTRAINTS and security

-- Row Level Security policies
ALTER TABLE chapters ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_bookmarks ENABLE ROW LEVEL SECURITY;
ALTER TABLE notifications ENABLE ROW LEVEL SECURITY;

-- Users can only see published chapters or their own drafts
CREATE POLICY chapters_select_policy ON chapters
    FOR SELECT
    USING (
        status = 'published' OR
        author_id = current_setting('app.current_user_id')::int
    );

-- Users can only modify their own bookmarks
CREATE POLICY bookmarks_policy ON user_bookmarks
    FOR ALL
    USING (user_id = current_setting('app.current_user_id')::int);

-- Users can only see their own notifications
CREATE POLICY notifications_policy ON notifications
    FOR ALL
    USING (user_id = current_setting('app.current_user_id')::int);

-- Initial data
INSERT INTO users (username, email, password_hash, full_name, role, is_active, is_verified)
VALUES
    ('admin', 'admin@koo-platform.com', '$2b$12$dummy_hash', 'System Administrator', 'super_admin', true, true),
    ('demo', 'demo@koo-platform.com', '$2b$12$dummy_hash', 'Demo User', 'editor', true, true);

COMMIT;