"""
Reference Library Service
Process and manage PDF textbook chapters for the KOO Platform
"""

import asyncio
import json
import logging
import os
import re
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict

import PyPDF2
import aiofiles
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, text
from sqlalchemy.orm import selectinload

from ..models.references import Textbook, BookChapter, ChapterCitation, ContentReference, ReferenceSearchIndex
from ..core.config import settings
from ..core.exceptions import ValidationError, ResourceNotFoundError, ExternalServiceError
from ..core.database import db_manager
from .hybrid_ai_manager import query_ai
from ..utils.text_processing import extract_keywords, clean_text, extract_medical_terms, extract_citations

logger = logging.getLogger(__name__)

@dataclass
class ProcessingStats:
    """Statistics for processing operations"""
    textbooks_found: int = 0
    textbooks_processed: int = 0
    chapters_found: int = 0
    chapters_processed: int = 0
    chapters_failed: int = 0
    processing_time_seconds: float = 0.0
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

@dataclass
class ChapterSearchResult:
    """Search result for chapter content"""
    chapter_id: str
    textbook_title: str
    chapter_title: str
    relevance_score: float
    matching_text: str
    file_path: str
    chapter_number: Optional[int] = None
    page_reference: Optional[str] = None

class ReferenceLibraryService:
    """Service for managing PDF textbook references"""

    def __init__(self):
        self.textbooks_root = Path(settings.TEXTBOOKS_PATH) if hasattr(settings, 'TEXTBOOKS_PATH') else Path("textbooks")
        self.supported_formats = ['.pdf']
        self.chunk_size = 1000  # Characters per search chunk
        self.max_file_size_mb = 100  # Maximum PDF file size

        # Chapter number patterns
        self.chapter_patterns = [
            r'^(\d+)[.\-_]\s*(.+)\.pdf$',           # "01-cerebral-anatomy.pdf"
            r'^chapter[_\-\s]*(\d+)[.\-_]\s*(.+)\.pdf$',  # "chapter-01-anatomy.pdf"
            r'^ch[_\-\s]*(\d+)[.\-_]\s*(.+)\.pdf$',       # "ch01-anatomy.pdf"
            r'^([A-Z]\d+)[.\-_]\s*(.+)\.pdf$',            # "A12-appendix.pdf"
        ]

    async def scan_textbooks_folder(self, force_rescan: bool = False) -> ProcessingStats:
        """
        Scan the textbooks folder and process all found textbooks

        Args:
            force_rescan: If True, reprocess already processed textbooks

        Returns:
            ProcessingStats with processing results
        """
        start_time = datetime.now()
        stats = ProcessingStats()

        try:
            if not self.textbooks_root.exists():
                raise ValidationError(f"Textbooks folder not found: {self.textbooks_root}")

            logger.info(f"Scanning textbooks folder: {self.textbooks_root}")

            # Find all textbook folders
            textbook_folders = [
                folder for folder in self.textbooks_root.iterdir()
                if folder.is_dir() and not folder.name.startswith('.')
            ]

            stats.textbooks_found = len(textbook_folders)
            logger.info(f"Found {stats.textbooks_found} textbook folders")

            # Process each textbook folder
            for folder in textbook_folders:
                try:
                    textbook = await self._process_textbook_folder(folder, force_rescan)
                    if textbook:
                        stats.textbooks_processed += 1
                        stats.chapters_found += len(textbook.chapters)
                        stats.chapters_processed += len([c for c in textbook.chapters if c.is_processed])
                        stats.chapters_failed += len([c for c in textbook.chapters if not c.is_processed and c.processing_error])

                except Exception as e:
                    error_msg = f"Error processing textbook folder {folder.name}: {e}"
                    logger.error(error_msg)
                    stats.errors.append(error_msg)

            # Calculate processing time
            end_time = datetime.now()
            stats.processing_time_seconds = (end_time - start_time).total_seconds()

            logger.info(f"Textbook scanning completed: {stats.textbooks_processed}/{stats.textbooks_found} textbooks processed")
            return stats

        except Exception as e:
            logger.error(f"Error scanning textbooks folder: {e}")
            raise ExternalServiceError(f"Failed to scan textbooks: {e}")

    async def _process_textbook_folder(self, folder_path: Path, force_rescan: bool = False) -> Optional[Textbook]:
        """Process a single textbook folder"""

        async with db_manager.get_session() as session:
            try:
                # Check if textbook already exists
                existing_textbook = await session.execute(
                    select(Textbook).where(Textbook.name == folder_path.name)
                )
                textbook = existing_textbook.scalar_one_or_none()

                if textbook and textbook.is_processed and not force_rescan:
                    logger.info(f"Textbook '{folder_path.name}' already processed, skipping")
                    return textbook

                # Load metadata if available
                metadata_file = folder_path / "metadata.json"
                metadata = {}
                if metadata_file.exists():
                    async with aiofiles.open(metadata_file, 'r', encoding='utf-8') as f:
                        content = await f.read()
                        metadata = json.loads(content)

                # Create or update textbook record
                if not textbook:
                    textbook = Textbook(
                        name=folder_path.name,
                        title=metadata.get('textbook', {}).get('title', folder_path.name),
                        edition=metadata.get('textbook', {}).get('edition'),
                        authors=metadata.get('textbook', {}).get('authors', []),
                        publisher=metadata.get('textbook', {}).get('publisher'),
                        publication_year=metadata.get('textbook', {}).get('year'),
                        isbn=metadata.get('textbook', {}).get('isbn'),
                        specialty=metadata.get('textbook', {}).get('specialty', 'neurosurgery'),
                        folder_path=str(folder_path),
                        metadata=metadata,
                        processing_started_at=datetime.utcnow()
                    )
                    session.add(textbook)
                else:
                    # Update existing textbook
                    textbook.processing_started_at = datetime.utcnow()
                    textbook.is_processed = False
                    textbook.processing_error = None

                await session.commit()

                # Find and process PDF chapters
                pdf_files = list(folder_path.glob("*.pdf"))
                logger.info(f"Found {len(pdf_files)} PDF files in {folder_path.name}")

                chapters_metadata = metadata.get('chapters', [])
                chapters_by_file = {ch.get('file', ''): ch for ch in chapters_metadata}

                for pdf_file in pdf_files:
                    try:
                        await self._process_chapter_pdf(session, textbook.id, pdf_file, chapters_by_file.get(pdf_file.name, {}))
                    except Exception as e:
                        logger.error(f"Error processing chapter {pdf_file.name}: {e}")

                # Mark textbook as processed
                textbook.is_processed = True
                textbook.processing_completed_at = datetime.utcnow()
                await session.commit()

                # Load with chapters for return
                result = await session.execute(
                    select(Textbook).options(selectinload(Textbook.chapters)).where(Textbook.id == textbook.id)
                )
                return result.scalar_one()

            except Exception as e:
                logger.error(f"Error processing textbook folder {folder_path}: {e}")
                if textbook:
                    textbook.processing_error = str(e)
                    await session.commit()
                raise

    async def _process_chapter_pdf(
        self,
        session: AsyncSession,
        textbook_id: str,
        pdf_file: Path,
        chapter_metadata: Dict[str, Any]
    ) -> Optional[Chapter]:
        """Process a single PDF chapter file"""

        try:
            # Check file size
            file_size_mb = pdf_file.stat().st_size / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                logger.warning(f"PDF file {pdf_file.name} is too large ({file_size_mb:.1f}MB), skipping")
                return None

            # Check if chapter already exists
            existing_chapter = await session.execute(
                select(BookChapter).where(BookChapter.file_path == str(pdf_file))
            )
            chapter = existing_chapter.scalar_one_or_none()

            # Parse chapter information from filename
            chapter_number, title = self._parse_chapter_filename(pdf_file.name)

            # Use metadata if available
            if chapter_metadata:
                title = chapter_metadata.get('title', title)
                chapter_number = chapter_metadata.get('chapter_number', chapter_number)

            # Extract text content from PDF
            content_text, page_count = await self._extract_pdf_content(pdf_file)

            if not content_text.strip():
                logger.warning(f"No text content extracted from {pdf_file.name}")
                return None

            # Process content with AI
            summary = await self._generate_chapter_summary(title, content_text)
            keywords = extract_keywords(content_text, max_keywords=30)
            medical_terms = extract_medical_terms(content_text)

            # Create or update chapter record
            if not chapter:
                chapter = BookChapter(
                    textbook_id=textbook_id,
                    file_name=pdf_file.name,
                    file_path=str(pdf_file),
                    file_size_mb=file_size_mb,
                    chapter_number=chapter_number,
                    title=title,
                    content_text=content_text,
                    summary=summary,
                    page_count=page_count,
                    keywords=keywords,
                    medical_terms=medical_terms,
                    word_count=len(content_text.split()),
                    reading_time_minutes=max(1, len(content_text.split()) // 200),  # ~200 words per minute
                    is_processed=True,
                    processed_at=datetime.utcnow()
                )
                session.add(chapter)
            else:
                # Update existing chapter
                chapter.content_text = content_text
                chapter.summary = summary
                chapter.keywords = keywords
                chapter.medical_terms = medical_terms
                chapter.word_count = len(content_text.split())
                chapter.reading_time_minutes = max(1, len(content_text.split()) // 200)
                chapter.is_processed = True
                chapter.processed_at = datetime.utcnow()
                chapter.processing_error = None

            await session.commit()

            # Process citations
            await self._extract_and_store_citations(session, chapter.id, content_text)

            # Create search index chunks
            await self._create_search_index(session, chapter.id, content_text, title)

            logger.info(f"Successfully processed chapter: {title}")
            return chapter

        except Exception as e:
            logger.error(f"Error processing PDF chapter {pdf_file.name}: {e}")
            if chapter:
                chapter.processing_error = str(e)
                chapter.is_processed = False
                await session.commit()
            return None

    def _parse_chapter_filename(self, filename: str) -> Tuple[Optional[int], str]:
        """Parse chapter number and title from filename"""

        for pattern in self.chapter_patterns:
            match = re.match(pattern, filename, re.IGNORECASE)
            if match:
                chapter_num_str = match.group(1)
                title = match.group(2)

                # Extract numeric part from chapter number
                chapter_num = None
                if chapter_num_str.isdigit():
                    chapter_num = int(chapter_num_str)
                elif chapter_num_str[0].isalpha() and chapter_num_str[1:].isdigit():
                    chapter_num = int(chapter_num_str[1:])  # Skip letter prefix

                # Clean up title
                title = title.replace('-', ' ').replace('_', ' ').strip()
                title = ' '.join(word.capitalize() for word in title.split())

                return chapter_num, title

        # No pattern matched, use filename as title
        title = filename.replace('.pdf', '').replace('-', ' ').replace('_', ' ')
        title = ' '.join(word.capitalize() for word in title.split())
        return None, title

    async def _extract_pdf_content(self, pdf_file: Path) -> Tuple[str, int]:
        """Extract text content from PDF file"""

        try:
            content_text = ""
            page_count = 0

            with open(pdf_file, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                page_count = len(reader.pages)

                for page in reader.pages:
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            content_text += page_text + "\n"
                    except Exception as e:
                        logger.warning(f"Error extracting text from page in {pdf_file.name}: {e}")

            # Clean and normalize text
            content_text = clean_text(content_text)

            return content_text, page_count

        except Exception as e:
            logger.error(f"Error extracting PDF content from {pdf_file}: {e}")
            return "", 0

    async def _generate_chapter_summary(self, title: str, content: str) -> str:
        """Generate AI summary of chapter content"""

        # Limit content for AI processing
        content_excerpt = content[:4000] if len(content) > 4000 else content

        prompt = f"""
        Create a concise medical summary for this neurosurgical textbook chapter.

        Chapter Title: {title}
        Content: {content_excerpt}

        Provide a professional 3-4 sentence summary focusing on:
        1. Key medical concepts and procedures
        2. Clinical significance and applications
        3. Main findings or recommendations

        Keep it clinically accurate and appropriate for medical professionals.
        """

        try:
            summary = await query_ai(prompt, max_tokens=300)
            return clean_text(summary)
        except Exception as e:
            logger.warning(f"AI summary generation failed for '{title}': {e}")
            return f"Summary for chapter: {title}"

    async def _extract_and_store_citations(self, session: AsyncSession, chapter_id: str, content: str) -> List[ChapterCitation]:
        """Extract citations from chapter content and store them"""

        try:
            citations_text = extract_citations(content)
            stored_citations = []

            for citation_text in citations_text:
                # Determine citation type
                citation_type = self._classify_citation(citation_text)

                # Extract additional metadata
                doi = self._extract_doi(citation_text)
                pmid = self._extract_pmid(citation_text)

                citation = ChapterCitation(
                    chapter_id=chapter_id,
                    citation_text=citation_text,
                    citation_type=citation_type,
                    doi=doi,
                    pmid=pmid,
                    confidence_score=0.8  # Default confidence
                )

                session.add(citation)
                stored_citations.append(citation)

            await session.commit()
            logger.debug(f"Stored {len(stored_citations)} citations for chapter {chapter_id}")
            return stored_citations

        except Exception as e:
            logger.error(f"Error extracting citations for chapter {chapter_id}: {e}")
            return []

    def _classify_citation(self, citation_text: str) -> str:
        """Classify the type of citation"""

        if 'PMID:' in citation_text or 'pubmed' in citation_text.lower():
            return 'pubmed'
        elif 'DOI:' in citation_text or '10.' in citation_text:
            return 'journal'
        elif 'Fig' in citation_text or 'Figure' in citation_text:
            return 'figure'
        elif 'Table' in citation_text:
            return 'table'
        elif citation_text.startswith(('http', 'www')):
            return 'url'
        else:
            return 'reference'

    def _extract_doi(self, text: str) -> Optional[str]:
        """Extract DOI from citation text"""
        doi_pattern = r'10\.\d+\/[^\s]+'
        match = re.search(doi_pattern, text)
        return match.group(0) if match else None

    def _extract_pmid(self, text: str) -> Optional[int]:
        """Extract PMID from citation text"""
        pmid_pattern = r'PMID:\s*(\d+)'
        match = re.search(pmid_pattern, text)
        return int(match.group(1)) if match else None

    async def _create_search_index(self, session: AsyncSession, chapter_id: str, content: str, title: str) -> None:
        """Create search index chunks for the chapter"""

        try:
            # Split content into chunks
            words = content.split()
            chunks = []

            for i in range(0, len(words), self.chunk_size // 5):  # Overlap chunks
                chunk_words = words[i:i + self.chunk_size // 5]
                chunk_text = ' '.join(chunk_words)

                if len(chunk_text.strip()) > 50:  # Only store meaningful chunks
                    chunks.append((i, chunk_text))

            # Store chunks in search index
            for position, chunk_text in chunks:
                search_chunk = ReferenceSearchIndex(
                    chapter_id=chapter_id,
                    text_chunk=chunk_text,
                    chunk_position=position,
                    chunk_size=len(chunk_text),
                    section_title=title  # Could be enhanced to detect section titles
                )
                session.add(search_chunk)

            await session.commit()
            logger.debug(f"Created {len(chunks)} search index chunks for chapter {chapter_id}")

        except Exception as e:
            logger.error(f"Error creating search index for chapter {chapter_id}: {e}")

    async def search_chapters(
        self,
        query: str,
        textbook_id: Optional[str] = None,
        specialty: Optional[str] = None,
        limit: int = 20
    ) -> List[ChapterSearchResult]:
        """Search chapters by content"""

        async with db_manager.get_session() as session:
            try:
                # Build search query
                search_query = select(
                    BookChapter.id,
                    BookChapter.title,
                    BookChapter.file_path,
                    BookChapter.chapter_number,
                    Textbook.title.label('textbook_title'),
                    ReferenceSearchIndex.text_chunk,
                    ReferenceSearchIndex.chunk_position
                ).select_from(
                    BookChapter.__table__.join(Textbook.__table__)
                    .join(ReferenceSearchIndex.__table__)
                ).where(
                    and_(
                        BookChapter.is_processed == True,
                        or_(
                            BookChapter.content_text.ilike(f'%{query}%'),
                            BookChapter.title.ilike(f'%{query}%'),
                            BookChapter.keywords.any(func.lower(func.unnest(BookChapter.keywords)).like(f'%{query.lower()}%')),
                            ReferenceSearchIndex.text_chunk.ilike(f'%{query}%')
                        )
                    )
                )

                # Add filters
                if textbook_id:
                    search_query = search_query.where(BookChapter.textbook_id == textbook_id)
                if specialty:
                    search_query = search_query.where(Textbook.specialty == specialty)

                search_query = search_query.limit(limit)

                result = await session.execute(search_query)
                rows = result.fetchall()

                # Process results
                search_results = []
                for row in rows:
                    # Calculate relevance score (simple implementation)
                    relevance_score = self._calculate_relevance_score(query, row.text_chunk, row.title)

                    search_result = ChapterSearchResult(
                        chapter_id=row.id,
                        textbook_title=row.textbook_title,
                        chapter_title=row.title,
                        relevance_score=relevance_score,
                        matching_text=row.text_chunk[:300] + "..." if len(row.text_chunk) > 300 else row.text_chunk,
                        file_path=row.file_path,
                        chapter_number=row.chapter_number
                    )
                    search_results.append(search_result)

                # Sort by relevance score
                search_results.sort(key=lambda x: x.relevance_score, reverse=True)

                return search_results

            except Exception as e:
                logger.error(f"Error searching chapters: {e}")
                raise ExternalServiceError(f"Chapter search failed: {e}")

    def _calculate_relevance_score(self, query: str, text: str, title: str) -> float:
        """Calculate simple relevance score for search results"""

        query_lower = query.lower()
        text_lower = text.lower()
        title_lower = title.lower()

        score = 0.0

        # Title matches are highly relevant
        if query_lower in title_lower:
            score += 1.0

        # Count query word matches in text
        query_words = query_lower.split()
        text_words = text_lower.split()

        matches = sum(1 for word in query_words if word in text_words)
        score += (matches / len(query_words)) * 0.8

        # Boost for medical terms
        medical_terms = extract_medical_terms(text)
        if any(term.lower() in query_lower for term in medical_terms):
            score += 0.3

        return min(score, 1.0)

    async def get_chapter_by_id(self, chapter_id: str) -> Optional[Chapter]:
        """Get a specific chapter by ID"""

        async with db_manager.get_session() as session:
            result = await session.execute(
                select(BookChapter).options(
                    selectinload(BookChapter.textbook),
                    selectinload(BookChapter.citations)
                ).where(BookChapter.id == chapter_id)
            )
            return result.scalar_one_or_none()

    async def get_textbooks(self) -> List[Textbook]:
        """Get all textbooks"""

        async with db_manager.get_session() as session:
            result = await session.execute(
                select(Textbook).options(selectinload(Textbook.chapters))
            )
            return result.scalars().all()

# Global service instance
reference_library = ReferenceLibraryService()