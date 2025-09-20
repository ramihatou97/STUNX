"""
Background Tasks for KOO Platform
Celery tasks organized by category
"""

from .ai_service_tasks import *
from .pdf_processing_tasks import *
from .database_tasks import *
from .health_check_tasks import *
from .research_tasks import *
from .cache_tasks import *

__all__ = [
    # AI Service Tasks
    'process_ai_query',
    'batch_ai_queries',
    'ai_service_health_check',
    
    # PDF Processing Tasks
    'process_pdf_document',
    'extract_pdf_text',
    'index_pdf_content',
    'batch_pdf_processing',
    
    # Database Tasks
    'database_maintenance',
    'cleanup_old_data',
    'optimize_database',
    'backup_database',
    
    # Health Check Tasks
    'check_all_services',
    'check_database_health',
    'check_ai_services_health',
    'check_cache_health',
    
    # Research Tasks
    'sync_research_data',
    'update_knowledge_base',
    'process_research_pipeline',
    
    # Cache Tasks
    'cleanup_expired_cache',
    'warm_critical_data',
    'cache_maintenance',
    'preload_frequent_queries'
]
