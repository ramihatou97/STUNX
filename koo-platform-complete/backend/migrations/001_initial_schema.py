"""
Initial KOO Platform Database Schema
Creates all core tables with proper indexes and constraints
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

def upgrade():
    # Enable extensions
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    op.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto"')
    op.execute('CREATE EXTENSION IF NOT EXISTS "vector"')
    op.execute('CREATE EXTENSION IF NOT EXISTS "pg_trgm"')
    op.execute('CREATE EXTENSION IF NOT EXISTS "btree_gin"')

    # Create enums
    user_role = sa.Enum('viewer', 'editor', 'admin', 'super_admin', name='user_role')
    user_role.create(op.get_bind())

    chapter_status = sa.Enum('draft', 'review', 'published', 'archived', name='chapter_status')
    chapter_status.create(op.get_bind())

    # Create users table
    op.create_table('users',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('uuid', postgresql.UUID(), nullable=False, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('username', sa.String(50), nullable=False),
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('password_hash', sa.String(255), nullable=False),
        sa.Column('full_name', sa.String(100), nullable=False),
        sa.Column('title', sa.String(100)),
        sa.Column('institution', sa.String(200)),
        sa.Column('department', sa.String(100)),
        sa.Column('specialty', sa.String(100)),
        sa.Column('role', user_role, nullable=False, server_default='viewer'),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('is_verified', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('last_login', sa.DateTime(timezone=True)),
        sa.Column('failed_login_attempts', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('preferences', postgresql.JSONB(), nullable=False, server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint('username'),
        sa.UniqueConstraint('email'),
        sa.UniqueConstraint('uuid')
    )

    # Create chapters table with vector support
    op.create_table('chapters',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('uuid', postgresql.UUID(), nullable=False, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('title', sa.String(500), nullable=False),
        sa.Column('slug', sa.String(200), nullable=False),
        sa.Column('summary', sa.Text()),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('content_html', sa.Text()),
        sa.Column('tags', postgresql.ARRAY(sa.Text()), server_default='{}'),
        sa.Column('status', chapter_status, nullable=False, server_default='draft'),
        sa.Column('version', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('confidence_score', sa.Float(), server_default='0.0'),
        sa.Column('content_embedding', sa.dialects.postgresql.ARRAY(sa.Float())), # Vector placeholder
        sa.Column('author_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='SET NULL')),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('published_at', sa.DateTime(timezone=True)),
        sa.UniqueConstraint('slug'),
        sa.UniqueConstraint('uuid')
    )

    # Create indexes
    op.create_index('idx_users_email', 'users', ['email'])
    op.create_index('idx_users_username', 'users', ['username'])
    op.create_index('idx_chapters_status', 'chapters', ['status'])
    op.create_index('idx_chapters_author', 'chapters', ['author_id'])
    op.create_index('idx_chapters_updated', 'chapters', ['updated_at'])

def downgrade():
    op.drop_table('chapters')
    op.drop_table('users')

    # Drop enums
    sa.Enum(name='chapter_status').drop(op.get_bind())
    sa.Enum(name='user_role').drop(op.get_bind())