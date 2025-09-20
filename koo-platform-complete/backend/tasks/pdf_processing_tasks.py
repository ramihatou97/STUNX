"""
Optimized PDF Processing Background Tasks
Memory-efficient tasks for handling PDF document processing asynchronously
"""

import asyncio
import logging
import gc
import psutil
import tempfile
import weakref
from typing import Dict, List, Any, Optional, AsyncGenerator, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
import os
import json

from celery import current_task
from core.task_manager import celery_app, BaseKOOTask
from core.redis_cache import redis_cache, CacheLevel
from core.config import settings

logger = logging.getLogger(__name__)

# Memory monitoring and management
@dataclass
class MemoryStats:
    """Memory usage statistics"""
    process_memory_mb: float
    available_memory_mb: float
    memory_percent: float
    timestamp: datetime

@dataclass
class ProcessingCheckpoint:
    """Checkpoint data for resuming interrupted processing"""
    document_id: str
    file_path: str
    pages_processed: int
    total_pages: int
    extracted_text: List[Dict[str, Any]]
    timestamp: datetime
    memory_stats: MemoryStats

class PDFParserPool:
    """Object pool for PDF parser instances to reduce GC overhead"""

    def __init__(self, pool_size: int = 5):
        self.pool_size = pool_size
        self._pool = []
        self._in_use = weakref.WeakSet()
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Acquire a PDF parser from the pool"""
        async with self._lock:
            if self._pool:
                parser = self._pool.pop()
                self._in_use.add(parser)
                return parser

            # Create new parser if pool is empty
            try:
                import PyPDF2
                parser = PyPDF2.PdfReader
                self._in_use.add(parser)
                return parser
            except ImportError:
                logger.error("PyPDF2 not available for PDF processing")
                raise

    async def release(self, parser):
        """Release a PDF parser back to the pool"""
        async with self._lock:
            if parser in self._in_use:
                self._in_use.discard(parser)
                if len(self._pool) < self.pool_size:
                    self._pool.append(parser)

    async def cleanup(self):
        """Clean up the parser pool"""
        async with self._lock:
            self._pool.clear()
            self._in_use.clear()

# Global parser pool
pdf_parser_pool = PDFParserPool(settings.PDF_POOL_SIZE)

# Memory monitoring utilities
def get_memory_stats() -> MemoryStats:
    """Get current memory usage statistics"""
    process = psutil.Process()
    memory_info = process.memory_info()
    virtual_memory = psutil.virtual_memory()

    return MemoryStats(
        process_memory_mb=memory_info.rss / 1024 / 1024,
        available_memory_mb=virtual_memory.available / 1024 / 1024,
        memory_percent=virtual_memory.percent,
        timestamp=datetime.now()
    )

def check_memory_pressure() -> bool:
    """Check if system is under memory pressure"""
    stats = get_memory_stats()

    # Check if process memory exceeds limit
    if stats.process_memory_mb > (settings.PDF_MEMORY_LIMIT / 1024 / 1024):
        return True

    # Check if system memory usage is too high
    if stats.memory_percent > 90:
        return True

    return False

async def cleanup_memory():
    """Force garbage collection and memory cleanup"""
    gc.collect()
    await asyncio.sleep(0.1)  # Allow cleanup to complete

async def save_checkpoint(checkpoint: ProcessingCheckpoint) -> None:
    """Save processing checkpoint for recovery"""
    try:
        checkpoint_key = f"pdf_checkpoint:{checkpoint.document_id}"
        await redis_cache.set(
            checkpoint_key,
            asdict(checkpoint),
            ttl=settings.PDF_CACHE_TTL,
            level=CacheLevel.APPLICATION
        )
        logger.debug(f"Checkpoint saved for document {checkpoint.document_id}")
    except Exception as e:
        logger.warning(f"Failed to save checkpoint: {e}")

async def load_checkpoint(document_id: str) -> Optional[ProcessingCheckpoint]:
    """Load processing checkpoint for recovery"""
    try:
        checkpoint_key = f"pdf_checkpoint:{document_id}"
        checkpoint_data = await redis_cache.get(checkpoint_key, CacheLevel.APPLICATION)

        if checkpoint_data:
            # Convert timestamp strings back to datetime objects
            checkpoint_data['timestamp'] = datetime.fromisoformat(checkpoint_data['timestamp'])
            checkpoint_data['memory_stats']['timestamp'] = datetime.fromisoformat(
                checkpoint_data['memory_stats']['timestamp']
            )

            return ProcessingCheckpoint(**checkpoint_data)

        return None
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}")
        return None

@celery_app.task(base=BaseKOOTask, bind=True, name='koo.tasks.pdf_processing.process_document')
def process_pdf_document(self, file_path: str, document_id: str, resume: bool = False) -> Dict[str, Any]:
    """
    Process a PDF document asynchronously with memory optimization

    Args:
        file_path: Path to the PDF file
        document_id: Unique identifier for the document
        resume: Whether to resume from checkpoint if available

    Returns:
        Processing results
    """
    try:
        current_task.update_state(state='PROGRESS', meta={'progress': 5, 'status': 'Initializing PDF processing'})

        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(_process_pdf_document_async(file_path, document_id, resume))
            return result
        finally:
            loop.close()

    except Exception as e:
        logger.error(f"PDF processing task failed for {document_id}: {e}")
        raise

async def _process_pdf_document_async(file_path: str, document_id: str, resume: bool = False) -> Dict[str, Any]:
    """Async implementation of PDF document processing"""
    processing_result = {
        'document_id': document_id,
        'file_path': file_path,
        'status': 'processing',
        'timestamp': datetime.now().isoformat(),
        'pages_processed': 0,
        'text_extracted': False,
        'indexed': False,
        'errors': [],
        'memory_stats': [],
        'resumed_from_checkpoint': False
    }

    try:
        # Validate file before processing
        current_task.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Validating file'})

        validation_result = await _validate_pdf_file(file_path)
        if not validation_result['valid']:
            raise ValueError(f"PDF validation failed: {validation_result['error']}")

        # Check for existing checkpoint if resuming
        checkpoint = None
        if resume:
            checkpoint = await load_checkpoint(document_id)
            if checkpoint:
                processing_result['resumed_from_checkpoint'] = True
                processing_result['pages_processed'] = checkpoint.pages_processed
                logger.info(f"Resuming PDF processing from page {checkpoint.pages_processed}")

        # Extract text with memory optimization
        current_task.update_state(state='PROGRESS', meta={'progress': 20, 'status': 'Starting text extraction'})

        text_content = await _extract_pdf_text_optimized(
            file_path,
            document_id,
            start_page=checkpoint.pages_processed if checkpoint else 0,
            existing_text=checkpoint.extracted_text if checkpoint else []
        )

        processing_result['text_extracted'] = True
        processing_result['pages_processed'] = len(text_content.get('pages', []))
        processing_result['memory_stats'] = text_content.get('memory_stats', [])

        # Index content for search
        current_task.update_state(state='PROGRESS', meta={'progress': 70, 'status': 'Indexing content'})

        index_result = await _index_pdf_content_optimized(document_id, text_content)
        processing_result['indexed'] = index_result['success']

        # Cache extracted content
        current_task.update_state(state='PROGRESS', meta={'progress': 85, 'status': 'Caching content'})

        await _cache_pdf_content(document_id, text_content)

        # Clean up checkpoint
        if checkpoint:
            await redis_cache.delete(f"pdf_checkpoint:{document_id}", CacheLevel.APPLICATION)

        processing_result['status'] = 'completed'
        current_task.update_state(state='PROGRESS', meta={'progress': 100, 'status': 'Processing complete'})

        logger.info(f"PDF processing completed for document {document_id}: {processing_result['pages_processed']} pages")
        return processing_result

    except Exception as e:
        processing_result['status'] = 'failed'
        processing_result['errors'].append(str(e))
        logger.error(f"PDF processing failed for {document_id}: {e}")
        raise

async def _validate_pdf_file(file_path: str) -> Dict[str, Any]:
    """Validate PDF file before processing"""
    try:
        file_path = Path(file_path)

        # Check if file exists
        if not file_path.exists():
            return {'valid': False, 'error': f"File not found: {file_path}"}

        # Check file size
        file_size = file_path.stat().st_size
        if file_size > settings.PDF_MAX_FILE_SIZE:
            return {
                'valid': False,
                'error': f"File too large: {file_size / 1024 / 1024:.1f}MB > {settings.PDF_MAX_FILE_SIZE / 1024 / 1024:.1f}MB"
            }

        # Check if it's a PDF file
        if file_path.suffix.lower() != '.pdf':
            return {'valid': False, 'error': f"Not a PDF file: {file_path.suffix}"}

        # Try to open and get basic info
        try:
            import PyPDF2
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)

                if num_pages > settings.PDF_MAX_PAGES:
                    return {
                        'valid': False,
                        'error': f"Too many pages: {num_pages} > {settings.PDF_MAX_PAGES}"
                    }

                return {
                    'valid': True,
                    'file_size': file_size,
                    'num_pages': num_pages,
                    'file_path': str(file_path)
                }
        except Exception as e:
            return {'valid': False, 'error': f"Cannot read PDF: {str(e)}"}

    except Exception as e:
        return {'valid': False, 'error': f"Validation error: {str(e)}"}

@celery_app.task(base=BaseKOOTask, bind=True, name='koo.tasks.pdf_processing.extract_text')
def extract_pdf_text(self, file_path: str) -> Dict[str, Any]:
    """
    Extract text from PDF file with memory optimization

    Args:
        file_path: Path to the PDF file

    Returns:
        Extracted text content
    """
    try:
        current_task.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Starting text extraction'})

        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Generate unique document ID for this extraction
            document_id = f"extract_{int(datetime.now().timestamp())}"

            text_content = loop.run_until_complete(
                _extract_pdf_text_optimized(file_path, document_id)
            )

            current_task.update_state(state='PROGRESS', meta={'progress': 100, 'status': 'Text extraction complete'})

            return text_content

        finally:
            loop.close()

    except Exception as e:
        logger.error(f"PDF text extraction task failed: {e}")
        raise

@celery_app.task(base=BaseKOOTask, bind=True, name='koo.tasks.pdf_processing.index_content')
def index_pdf_content(self, document_id: str, content: Dict[str, Any]) -> Dict[str, Any]:
    """
    Index PDF content for search with memory optimization

    Args:
        document_id: Document identifier
        content: Extracted content to index

    Returns:
        Indexing results
    """
    try:
        current_task.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Starting content indexing'})

        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            index_result = loop.run_until_complete(
                _index_pdf_content_optimized(document_id, content)
            )

            current_task.update_state(state='PROGRESS', meta={'progress': 100, 'status': 'Indexing complete'})

            return index_result

        finally:
            loop.close()

    except Exception as e:
        logger.error(f"PDF indexing task failed: {e}")
        raise

@celery_app.task(base=BaseKOOTask, bind=True, name='koo.tasks.pdf_processing.batch_processing')
def batch_pdf_processing(self, file_paths: List[str], batch_size: Optional[int] = None) -> Dict[str, Any]:
    """
    Process multiple PDF files in batch with memory optimization

    Args:
        file_paths: List of PDF file paths to process
        batch_size: Number of files to process in each batch (default: from settings)

    Returns:
        Batch processing results
    """
    try:
        current_task.update_state(state='PROGRESS', meta={'progress': 5, 'status': 'Initializing batch processing'})

        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                _batch_pdf_processing_async(file_paths, batch_size or settings.PDF_PAGE_BATCH_SIZE)
            )
            return result
        finally:
            loop.close()

    except Exception as e:
        logger.error(f"Batch PDF processing task failed: {e}")
        raise

async def _batch_pdf_processing_async(file_paths: List[str], batch_size: int) -> Dict[str, Any]:
    """Async implementation of batch PDF processing"""
    batch_results = {
        'total_files': len(file_paths),
        'processed_files': 0,
        'failed_files': 0,
        'skipped_files': 0,
        'results': [],
        'memory_stats': [],
        'timestamp': datetime.now().isoformat()
    }

    try:
        # Process files in batches to manage memory
        for batch_start in range(0, len(file_paths), batch_size):
            batch_end = min(batch_start + batch_size, len(file_paths))
            batch_files = file_paths[batch_start:batch_end]

            progress = int((batch_start / len(file_paths)) * 90) + 10
            current_task.update_state(
                state='PROGRESS',
                meta={
                    'progress': progress,
                    'status': f'Processing batch {batch_start//batch_size + 1} ({batch_start+1}-{batch_end}/{len(file_paths)})'
                }
            )

            # Check memory before processing batch
            if check_memory_pressure():
                logger.warning("Memory pressure detected, performing cleanup")
                await cleanup_memory()

                # If still under pressure, skip remaining files
                if check_memory_pressure():
                    logger.error("Severe memory pressure, skipping remaining files")
                    for file_path in file_paths[batch_start:]:
                        batch_results['results'].append({
                            'file_path': file_path,
                            'status': 'skipped',
                            'error': 'Memory pressure - processing skipped'
                        })
                        batch_results['skipped_files'] += 1
                    break

            # Process batch concurrently with limited concurrency
            semaphore = asyncio.Semaphore(2)  # Limit concurrent processing
            batch_tasks = []

            for i, file_path in enumerate(batch_files):
                document_id = f"batch_{int(datetime.now().timestamp())}_{batch_start + i}"
                task = _process_single_pdf_with_semaphore(semaphore, file_path, document_id)
                batch_tasks.append(task)

            # Wait for batch completion
            batch_task_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Process results
            for i, result in enumerate(batch_task_results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to process {batch_files[i]}: {result}")
                    batch_results['results'].append({
                        'file_path': batch_files[i],
                        'status': 'failed',
                        'error': str(result)
                    })
                    batch_results['failed_files'] += 1
                else:
                    batch_results['results'].append(result)
                    if result['status'] == 'completed':
                        batch_results['processed_files'] += 1
                    else:
                        batch_results['failed_files'] += 1

            # Record memory stats after batch
            memory_stats = get_memory_stats()
            batch_results['memory_stats'].append({
                'batch': batch_start // batch_size + 1,
                'memory_mb': memory_stats.process_memory_mb,
                'memory_percent': memory_stats.memory_percent
            })

            # Cleanup between batches
            await cleanup_memory()

        current_task.update_state(state='PROGRESS', meta={'progress': 100, 'status': 'Batch processing complete'})

        logger.info(
            f"Batch PDF processing completed: {batch_results['processed_files']} successful, "
            f"{batch_results['failed_files']} failed, {batch_results['skipped_files']} skipped"
        )
        return batch_results

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise

async def _process_single_pdf_with_semaphore(semaphore: asyncio.Semaphore, file_path: str, document_id: str) -> Dict[str, Any]:
    """Process a single PDF with semaphore for concurrency control"""
    async with semaphore:
        return await _process_single_pdf_optimized(file_path, document_id)

# Optimized helper functions

async def _extract_pdf_text_optimized(
    file_path: str,
    document_id: str,
    start_page: int = 0,
    existing_text: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Extract text from PDF file with memory optimization and streaming

    Args:
        file_path: Path to the PDF file
        document_id: Document identifier for checkpointing
        start_page: Page to start extraction from (for resuming)
        existing_text: Previously extracted text (for resuming)

    Returns:
        Extracted text content with memory statistics
    """
    try:
        import PyPDF2

        text_content = {
            'file_path': file_path,
            'document_id': document_id,
            'pages': existing_text or [],
            'total_pages': 0,
            'extraction_method': 'PyPDF2_optimized',
            'timestamp': datetime.now().isoformat(),
            'memory_stats': [],
            'processing_stats': {
                'start_page': start_page,
                'pages_processed': 0,
                'errors': 0,
                'checkpoints_saved': 0
            }
        }

        # Get parser from pool
        parser_class = await pdf_parser_pool.acquire()

        try:
            # Open file and get total pages
            with open(file_path, 'rb') as file:
                pdf_reader = parser_class(file)
                text_content['total_pages'] = len(pdf_reader.pages)

                # Process pages in batches with memory monitoring
                for page_num in range(start_page, text_content['total_pages']):
                    try:
                        # Check memory pressure before processing each page
                        if page_num % settings.PDF_MEMORY_CHECK_INTERVAL == 0:
                            if check_memory_pressure():
                                logger.warning(f"Memory pressure detected at page {page_num + 1}")
                                await cleanup_memory()

                                # If still under pressure, save checkpoint and stop
                                if check_memory_pressure():
                                    logger.error(f"Severe memory pressure, stopping at page {page_num + 1}")
                                    checkpoint = ProcessingCheckpoint(
                                        document_id=document_id,
                                        file_path=file_path,
                                        pages_processed=page_num,
                                        total_pages=text_content['total_pages'],
                                        extracted_text=text_content['pages'],
                                        timestamp=datetime.now(),
                                        memory_stats=get_memory_stats()
                                    )
                                    await save_checkpoint(checkpoint)
                                    text_content['processing_stats']['checkpoints_saved'] += 1
                                    break

                        # Extract text from current page
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()

                        # Add page data
                        page_data = {
                            'page_number': page_num + 1,
                            'text': page_text,
                            'word_count': len(page_text.split()) if page_text else 0,
                            'char_count': len(page_text) if page_text else 0
                        }

                        text_content['pages'].append(page_data)
                        text_content['processing_stats']['pages_processed'] += 1

                        # Update progress
                        progress = int(((page_num - start_page + 1) / (text_content['total_pages'] - start_page)) * 50) + 20
                        current_task.update_state(
                            state='PROGRESS',
                            meta={
                                'progress': progress,
                                'status': f'Extracting text: page {page_num + 1}/{text_content["total_pages"]}'
                            }
                        )

                        # Save checkpoint periodically
                        if (page_num + 1) % settings.PDF_CHECKPOINT_INTERVAL == 0:
                            checkpoint = ProcessingCheckpoint(
                                document_id=document_id,
                                file_path=file_path,
                                pages_processed=page_num + 1,
                                total_pages=text_content['total_pages'],
                                extracted_text=text_content['pages'],
                                timestamp=datetime.now(),
                                memory_stats=get_memory_stats()
                            )
                            await save_checkpoint(checkpoint)
                            text_content['processing_stats']['checkpoints_saved'] += 1
                            logger.debug(f"Checkpoint saved at page {page_num + 1}")

                        # Record memory stats periodically
                        if page_num % 20 == 0:
                            memory_stats = get_memory_stats()
                            text_content['memory_stats'].append({
                                'page': page_num + 1,
                                'memory_mb': memory_stats.process_memory_mb,
                                'memory_percent': memory_stats.memory_percent,
                                'timestamp': memory_stats.timestamp.isoformat()
                            })

                        # Yield control to allow other tasks
                        if page_num % 5 == 0:
                            await asyncio.sleep(0.01)

                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                        text_content['pages'].append({
                            'page_number': page_num + 1,
                            'text': '',
                            'word_count': 0,
                            'char_count': 0,
                            'error': str(e)
                        })
                        text_content['processing_stats']['errors'] += 1

        finally:
            # Return parser to pool
            await pdf_parser_pool.release(parser_class)

        # Final memory stats
        final_memory = get_memory_stats()
        text_content['memory_stats'].append({
            'page': 'final',
            'memory_mb': final_memory.process_memory_mb,
            'memory_percent': final_memory.memory_percent,
            'timestamp': final_memory.timestamp.isoformat()
        })

        logger.info(
            f"Text extraction completed for {document_id}: "
            f"{text_content['processing_stats']['pages_processed']} pages, "
            f"{text_content['processing_stats']['errors']} errors"
        )

        return text_content

    except ImportError:
        # Fallback to alternative method if PyPDF2 not available
        logger.warning("PyPDF2 not available, using fallback text extraction")
        return {
            'file_path': file_path,
            'document_id': document_id,
            'pages': existing_text or [],
            'total_pages': 0,
            'extraction_method': 'fallback',
            'error': 'PyPDF2 not available',
            'timestamp': datetime.now().isoformat(),
            'memory_stats': [],
            'processing_stats': {'start_page': start_page, 'pages_processed': 0, 'errors': 1}
        }
    except Exception as e:
        logger.error(f"PDF text extraction failed: {e}")
        raise

async def _index_pdf_content_optimized(document_id: str, content: Dict[str, Any]) -> Dict[str, Any]:
    """Index PDF content for search with memory optimization"""
    try:
        pages = content.get('pages', [])

        index_result = {
            'document_id': document_id,
            'success': True,
            'indexed_pages': len(pages),
            'total_words': sum(page.get('word_count', 0) for page in pages),
            'total_chars': sum(page.get('char_count', 0) for page in pages),
            'timestamp': datetime.now().isoformat(),
            'memory_optimized': True
        }

        # Build searchable text in chunks to avoid memory issues
        searchable_chunks = []
        chunk_size = 1000000  # 1MB chunks
        current_chunk = []
        current_size = 0

        for page in pages:
            page_text = page.get('text', '')
            if page_text:
                current_chunk.append(page_text)
                current_size += len(page_text)

                # If chunk is large enough, process it
                if current_size >= chunk_size:
                    searchable_chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_size = 0

                    # Yield control and check memory
                    await asyncio.sleep(0.01)
                    if check_memory_pressure():
                        await cleanup_memory()

        # Add remaining text
        if current_chunk:
            searchable_chunks.append(' '.join(current_chunk))

        # Cache indexed content in chunks to avoid large memory allocations
        for i, chunk in enumerate(searchable_chunks):
            chunk_key = f"search_index:{document_id}:chunk_{i}"
            await redis_cache.set(
                chunk_key,
                {
                    'chunk_index': i,
                    'total_chunks': len(searchable_chunks),
                    'searchable_text': chunk,
                    'document_id': document_id
                },
                ttl=settings.PDF_CACHE_TTL,
                level=CacheLevel.APPLICATION
            )

        # Cache metadata separately
        metadata_key = f"search_index:{document_id}:metadata"
        await redis_cache.set(
            metadata_key,
            {
                'document_id': document_id,
                'total_pages': content.get('total_pages', 0),
                'total_chunks': len(searchable_chunks),
                'file_path': content.get('file_path'),
                'extraction_method': content.get('extraction_method'),
                'indexed_at': datetime.now().isoformat(),
                'processing_stats': content.get('processing_stats', {}),
                'memory_stats': content.get('memory_stats', [])
            },
            ttl=settings.PDF_CACHE_TTL,
            level=CacheLevel.APPLICATION
        )

        logger.info(f"PDF content indexed for {document_id}: {len(searchable_chunks)} chunks")
        return index_result

    except Exception as e:
        logger.error(f"PDF content indexing failed: {e}")
        return {
            'document_id': document_id,
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

async def _cache_pdf_content(document_id: str, content: Dict[str, Any]) -> None:
    """Cache PDF content with memory optimization"""
    try:
        # Cache content in chunks to avoid memory issues
        pages = content.get('pages', [])
        chunk_size = 100  # Pages per chunk

        for i in range(0, len(pages), chunk_size):
            chunk_pages = pages[i:i + chunk_size]
            chunk_key = f"pdf_content:{document_id}:chunk_{i // chunk_size}"

            chunk_data = {
                'document_id': document_id,
                'chunk_index': i // chunk_size,
                'total_chunks': (len(pages) + chunk_size - 1) // chunk_size,
                'pages': chunk_pages,
                'page_range': {
                    'start': i + 1,
                    'end': min(i + chunk_size, len(pages))
                }
            }

            await redis_cache.set(
                chunk_key,
                chunk_data,
                ttl=settings.PDF_CACHE_TTL,
                level=CacheLevel.APPLICATION
            )

        # Cache summary metadata
        summary_key = f"pdf_content:{document_id}:summary"
        summary_data = {
            'document_id': document_id,
            'file_path': content.get('file_path'),
            'total_pages': content.get('total_pages', 0),
            'total_chunks': (len(pages) + chunk_size - 1) // chunk_size,
            'extraction_method': content.get('extraction_method'),
            'timestamp': content.get('timestamp'),
            'processing_stats': content.get('processing_stats', {}),
            'memory_stats_summary': {
                'peak_memory_mb': max(
                    (stat.get('memory_mb', 0) for stat in content.get('memory_stats', [])),
                    default=0
                ),
                'avg_memory_percent': sum(
                    stat.get('memory_percent', 0) for stat in content.get('memory_stats', [])
                ) / max(len(content.get('memory_stats', [])), 1)
            }
        }

        await redis_cache.set(
            summary_key,
            summary_data,
            ttl=settings.PDF_CACHE_TTL,
            level=CacheLevel.APPLICATION
        )

        logger.debug(f"PDF content cached for {document_id}")

    except Exception as e:
        logger.error(f"Failed to cache PDF content: {e}")

async def _process_single_pdf_optimized(file_path: str, document_id: str) -> Dict[str, Any]:
    """Process a single PDF file with memory optimization"""
    try:
        # Validate file first
        validation_result = await _validate_pdf_file(file_path)
        if not validation_result['valid']:
            return {
                'document_id': document_id,
                'file_path': file_path,
                'status': 'failed',
                'error': f"Validation failed: {validation_result['error']}",
                'timestamp': datetime.now().isoformat()
            }

        # Extract text with optimization
        text_content = await _extract_pdf_text_optimized(file_path, document_id)

        # Index content with optimization
        index_result = await _index_pdf_content_optimized(document_id, text_content)

        # Cache content with optimization
        await _cache_pdf_content(document_id, text_content)

        # Get final memory stats
        final_memory = get_memory_stats()

        return {
            'document_id': document_id,
            'file_path': file_path,
            'status': 'completed',
            'pages_processed': len(text_content.get('pages', [])),
            'total_pages': text_content.get('total_pages', 0),
            'indexed': index_result['success'],
            'extraction_method': text_content.get('extraction_method'),
            'processing_stats': text_content.get('processing_stats', {}),
            'memory_stats': {
                'peak_memory_mb': max(
                    (stat.get('memory_mb', 0) for stat in text_content.get('memory_stats', [])),
                    default=final_memory.process_memory_mb
                ),
                'final_memory_mb': final_memory.process_memory_mb,
                'memory_samples': len(text_content.get('memory_stats', []))
            },
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to process PDF {file_path}: {e}")
        return {
            'document_id': document_id,
            'file_path': file_path,
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

# Additional utility tasks for PDF processing management

@celery_app.task(base=BaseKOOTask, bind=True, name='koo.tasks.pdf_processing.cleanup_memory')
def cleanup_pdf_memory(self) -> Dict[str, Any]:
    """Clean up PDF processing memory and resources"""
    try:
        current_task.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Starting memory cleanup'})

        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(_cleanup_pdf_memory_async())
            return result
        finally:
            loop.close()

    except Exception as e:
        logger.error(f"PDF memory cleanup task failed: {e}")
        raise

async def _cleanup_pdf_memory_async() -> Dict[str, Any]:
    """Async implementation of PDF memory cleanup"""
    try:
        initial_memory = get_memory_stats()

        # Clean up parser pool
        await pdf_parser_pool.cleanup()

        # Force garbage collection
        await cleanup_memory()

        # Clear temporary files if any
        temp_dir = Path(tempfile.gettempdir())
        pdf_temp_files = list(temp_dir.glob("pdf_processing_*"))
        cleaned_files = 0

        for temp_file in pdf_temp_files:
            try:
                temp_file.unlink()
                cleaned_files += 1
            except Exception as e:
                logger.warning(f"Failed to remove temp file {temp_file}: {e}")

        final_memory = get_memory_stats()
        memory_freed = initial_memory.process_memory_mb - final_memory.process_memory_mb

        current_task.update_state(state='PROGRESS', meta={'progress': 100, 'status': 'Memory cleanup complete'})

        return {
            'success': True,
            'memory_freed_mb': memory_freed,
            'temp_files_cleaned': cleaned_files,
            'initial_memory_mb': initial_memory.process_memory_mb,
            'final_memory_mb': final_memory.process_memory_mb,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Memory cleanup failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

@celery_app.task(base=BaseKOOTask, bind=True, name='koo.tasks.pdf_processing.resume_processing')
def resume_pdf_processing(self, document_id: str) -> Dict[str, Any]:
    """Resume interrupted PDF processing from checkpoint"""
    try:
        current_task.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Loading checkpoint'})

        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(_resume_pdf_processing_async(document_id))
            return result
        finally:
            loop.close()

    except Exception as e:
        logger.error(f"PDF resume processing task failed: {e}")
        raise

async def _resume_pdf_processing_async(document_id: str) -> Dict[str, Any]:
    """Async implementation of PDF processing resume"""
    try:
        # Load checkpoint
        checkpoint = await load_checkpoint(document_id)
        if not checkpoint:
            return {
                'success': False,
                'error': 'No checkpoint found for document',
                'document_id': document_id,
                'timestamp': datetime.now().isoformat()
            }

        logger.info(f"Resuming PDF processing for {document_id} from page {checkpoint.pages_processed}")

        # Resume processing
        result = await _process_pdf_document_async(
            checkpoint.file_path,
            document_id,
            resume=True
        )

        return {
            'success': True,
            'resumed_from_page': checkpoint.pages_processed,
            'processing_result': result,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to resume PDF processing: {e}")
        return {
            'success': False,
            'error': str(e),
            'document_id': document_id,
            'timestamp': datetime.now().isoformat()
        }
