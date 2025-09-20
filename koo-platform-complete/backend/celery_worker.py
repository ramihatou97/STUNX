#!/usr/bin/env python3
"""
Celery Worker Entry Point for KOO Platform
Starts Celery worker with proper configuration and task imports
"""

import os
import sys
import logging
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import Celery app and tasks
from core.task_manager import celery_app

# Import all task modules to register them
import tasks.ai_service_tasks
import tasks.pdf_processing_tasks
import tasks.database_tasks
import tasks.health_check_tasks
import tasks.research_tasks
import tasks.cache_tasks

if __name__ == '__main__':
    logger.info("Starting KOO Platform Celery Worker")
    
    # Start the worker
    celery_app.start()
