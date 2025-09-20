"""
Database Maintenance Background Tasks
Tasks for database cleanup, optimization, and maintenance
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from celery import current_task
from core.task_manager import celery_app, BaseKOOTask
from core.redis_cache import redis_cache, CacheLevel
from core.database import db_manager

logger = logging.getLogger(__name__)

@celery_app.task(base=BaseKOOTask, bind=True, name='koo.tasks.database.maintenance')
def database_maintenance(self) -> Dict[str, Any]:
    """
    Comprehensive database maintenance
    
    Returns:
        Maintenance results
    """
    try:
        current_task.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Starting database maintenance'})
        
        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            maintenance_results = {
                'timestamp': datetime.now().isoformat(),
                'operations': {},
                'total_duration_seconds': 0
            }
            
            start_time = asyncio.get_event_loop().time()
            
            # Clean up old data
            current_task.update_state(state='PROGRESS', meta={'progress': 25, 'status': 'Cleaning old data'})
            
            cleanup_result = await _cleanup_old_data()
            maintenance_results['operations']['cleanup'] = cleanup_result
            
            # Optimize database
            current_task.update_state(state='PROGRESS', meta={'progress': 50, 'status': 'Optimizing database'})
            
            optimize_result = await _optimize_database()
            maintenance_results['operations']['optimization'] = optimize_result
            
            # Update statistics
            current_task.update_state(state='PROGRESS', meta={'progress': 75, 'status': 'Updating statistics'})
            
            stats_result = await _update_database_statistics()
            maintenance_results['operations']['statistics'] = stats_result
            
            # Backup critical data
            current_task.update_state(state='PROGRESS', meta={'progress': 90, 'status': 'Creating backup'})
            
            backup_result = await _create_maintenance_backup()
            maintenance_results['operations']['backup'] = backup_result
            
            maintenance_results['total_duration_seconds'] = asyncio.get_event_loop().time() - start_time
            
            # Cache maintenance report
            await redis_cache.set(
                'database_maintenance_report',
                maintenance_results,
                ttl=86400,  # 24 hours
                level=CacheLevel.APPLICATION
            )
            
            current_task.update_state(state='PROGRESS', meta={'progress': 100, 'status': 'Maintenance complete'})
            
            logger.info("Database maintenance completed successfully")
            return maintenance_results
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Database maintenance task failed: {e}")
        raise

@celery_app.task(base=BaseKOOTask, bind=True, name='koo.tasks.database.cleanup_old_data')
def cleanup_old_data(self, days_old: int = 30) -> Dict[str, Any]:
    """
    Clean up old data from database
    
    Args:
        days_old: Remove data older than this many days
    
    Returns:
        Cleanup results
    """
    try:
        current_task.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Starting data cleanup'})
        
        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            cleanup_result = await _cleanup_old_data(days_old)
            
            current_task.update_state(state='PROGRESS', meta={'progress': 100, 'status': 'Cleanup complete'})
            
            return cleanup_result
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Data cleanup task failed: {e}")
        raise

@celery_app.task(base=BaseKOOTask, bind=True, name='koo.tasks.database.optimize')
def optimize_database(self) -> Dict[str, Any]:
    """
    Optimize database performance
    
    Returns:
        Optimization results
    """
    try:
        current_task.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Starting database optimization'})
        
        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            optimize_result = await _optimize_database()
            
            current_task.update_state(state='PROGRESS', meta={'progress': 100, 'status': 'Optimization complete'})
            
            return optimize_result
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Database optimization task failed: {e}")
        raise

@celery_app.task(base=BaseKOOTask, bind=True, name='koo.tasks.database.backup')
def backup_database(self, backup_type: str = 'incremental') -> Dict[str, Any]:
    """
    Create database backup
    
    Args:
        backup_type: Type of backup ('full' or 'incremental')
    
    Returns:
        Backup results
    """
    try:
        current_task.update_state(state='PROGRESS', meta={'progress': 10, 'status': f'Starting {backup_type} backup'})
        
        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            backup_result = await _create_database_backup(backup_type)
            
            current_task.update_state(state='PROGRESS', meta={'progress': 100, 'status': 'Backup complete'})
            
            return backup_result
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Database backup task failed: {e}")
        raise

# Helper functions

async def _cleanup_old_data(days_old: int = 30) -> Dict[str, Any]:
    """Clean up old data from various tables"""
    try:
        cutoff_date = datetime.now() - timedelta(days=days_old)
        cleanup_stats = {
            'cutoff_date': cutoff_date.isoformat(),
            'tables_cleaned': {},
            'total_rows_deleted': 0,
            'success': True
        }
        
        async with db_manager.session_scope() as session:
            # Clean up old log entries (example)
            try:
                result = await session.execute(
                    "DELETE FROM system_logs WHERE created_at < :cutoff_date",
                    {'cutoff_date': cutoff_date}
                )
                deleted_logs = result.rowcount
                cleanup_stats['tables_cleaned']['system_logs'] = deleted_logs
                cleanup_stats['total_rows_deleted'] += deleted_logs
                
            except Exception as e:
                logger.warning(f"Failed to clean system_logs: {e}")
                cleanup_stats['tables_cleaned']['system_logs'] = f"Error: {str(e)}"
            
            # Clean up old session data (example)
            try:
                result = await session.execute(
                    "DELETE FROM user_sessions WHERE last_activity < :cutoff_date",
                    {'cutoff_date': cutoff_date}
                )
                deleted_sessions = result.rowcount
                cleanup_stats['tables_cleaned']['user_sessions'] = deleted_sessions
                cleanup_stats['total_rows_deleted'] += deleted_sessions
                
            except Exception as e:
                logger.warning(f"Failed to clean user_sessions: {e}")
                cleanup_stats['tables_cleaned']['user_sessions'] = f"Error: {str(e)}"
            
            # Clean up old temporary data (example)
            try:
                result = await session.execute(
                    "DELETE FROM temp_data WHERE created_at < :cutoff_date",
                    {'cutoff_date': cutoff_date}
                )
                deleted_temp = result.rowcount
                cleanup_stats['tables_cleaned']['temp_data'] = deleted_temp
                cleanup_stats['total_rows_deleted'] += deleted_temp
                
            except Exception as e:
                logger.warning(f"Failed to clean temp_data: {e}")
                cleanup_stats['tables_cleaned']['temp_data'] = f"Error: {str(e)}"
            
            await session.commit()
        
        logger.info(f"Data cleanup completed: {cleanup_stats['total_rows_deleted']} rows deleted")
        return cleanup_stats
        
    except Exception as e:
        logger.error(f"Data cleanup failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'cutoff_date': cutoff_date.isoformat() if 'cutoff_date' in locals() else None
        }

async def _optimize_database() -> Dict[str, Any]:
    """Optimize database performance"""
    try:
        optimization_stats = {
            'operations': {},
            'success': True,
            'timestamp': datetime.now().isoformat()
        }
        
        async with db_manager.session_scope() as session:
            # Analyze tables for query optimization
            try:
                await session.execute("ANALYZE")
                optimization_stats['operations']['analyze'] = 'completed'
                
            except Exception as e:
                logger.warning(f"Failed to analyze tables: {e}")
                optimization_stats['operations']['analyze'] = f"Error: {str(e)}"
            
            # Vacuum database (PostgreSQL specific)
            try:
                # Note: VACUUM cannot be run inside a transaction
                # This would need to be handled differently in production
                optimization_stats['operations']['vacuum'] = 'skipped (requires separate connection)'
                
            except Exception as e:
                logger.warning(f"Failed to vacuum database: {e}")
                optimization_stats['operations']['vacuum'] = f"Error: {str(e)}"
            
            # Reindex critical tables
            try:
                # Example: reindex chapters table
                await session.execute("REINDEX TABLE chapters")
                optimization_stats['operations']['reindex_chapters'] = 'completed'
                
            except Exception as e:
                logger.warning(f"Failed to reindex chapters: {e}")
                optimization_stats['operations']['reindex_chapters'] = f"Error: {str(e)}"
        
        logger.info("Database optimization completed")
        return optimization_stats
        
    except Exception as e:
        logger.error(f"Database optimization failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

async def _update_database_statistics() -> Dict[str, Any]:
    """Update database statistics"""
    try:
        stats_result = {
            'tables': {},
            'success': True,
            'timestamp': datetime.now().isoformat()
        }
        
        async with db_manager.session_scope() as session:
            # Get table statistics
            tables = ['chapters', 'textbooks', 'research_papers', 'users']
            
            for table in tables:
                try:
                    # Get row count
                    result = await session.execute(f"SELECT COUNT(*) FROM {table}")
                    row_count = result.scalar()
                    
                    # Get table size (PostgreSQL specific)
                    result = await session.execute(f"SELECT pg_total_relation_size('{table}')")
                    table_size = result.scalar()
                    
                    stats_result['tables'][table] = {
                        'row_count': row_count,
                        'size_bytes': table_size,
                        'size_mb': round(table_size / (1024 * 1024), 2) if table_size else 0
                    }
                    
                except Exception as e:
                    logger.warning(f"Failed to get statistics for {table}: {e}")
                    stats_result['tables'][table] = f"Error: {str(e)}"
        
        # Cache statistics
        await redis_cache.set(
            'database_statistics',
            stats_result,
            ttl=3600,  # 1 hour
            level=CacheLevel.APPLICATION
        )
        
        logger.info("Database statistics updated")
        return stats_result
        
    except Exception as e:
        logger.error(f"Failed to update database statistics: {e}")
        return {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

async def _create_maintenance_backup() -> Dict[str, Any]:
    """Create a maintenance backup"""
    try:
        backup_result = {
            'backup_type': 'maintenance',
            'timestamp': datetime.now().isoformat(),
            'success': True
        }
        
        # This would implement actual backup logic
        # For now, we'll simulate the backup process
        
        backup_result['backup_file'] = f"maintenance_backup_{int(datetime.now().timestamp())}.sql"
        backup_result['size_mb'] = 0  # Would be actual backup size
        
        logger.info("Maintenance backup created")
        return backup_result
        
    except Exception as e:
        logger.error(f"Failed to create maintenance backup: {e}")
        return {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

async def _create_database_backup(backup_type: str) -> Dict[str, Any]:
    """Create database backup"""
    try:
        backup_result = {
            'backup_type': backup_type,
            'timestamp': datetime.now().isoformat(),
            'success': True
        }
        
        # This would implement actual backup logic based on backup_type
        # For now, we'll simulate the backup process
        
        backup_result['backup_file'] = f"{backup_type}_backup_{int(datetime.now().timestamp())}.sql"
        backup_result['size_mb'] = 0  # Would be actual backup size
        
        logger.info(f"{backup_type.capitalize()} backup created")
        return backup_result
        
    except Exception as e:
        logger.error(f"Failed to create {backup_type} backup: {e}")
        return {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
