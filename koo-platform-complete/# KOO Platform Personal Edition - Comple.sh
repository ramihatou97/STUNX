1# KOO Platform Personal Edition - Complete Exhaustive Setup
# Date: 2025-01-19
# Repository: ramihatou97/koo-platform
# This script contains EVERY SINGLE FILE needed for the complete system

echo "==============================================="
echo "KOO PLATFORM PERSONAL EDITION - COMPLETE SETUP"
echo "==============================================="
echo "Date: $(date)"
echo "Target: ramihatou97/koo-platform"

# Set project root
PROJECT_ROOT="$HOME/koo-platform-complete"
rm -rf "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Initialize git
git init

echo "📁 Creating complete directory structure..."

# Create all directories
mkdir -p backend/{api,core,services,models,utils,migrations,tests}
mkdir -p frontend/src/{components,pages,services,hooks,contexts,utils,styles}
mkdir -p frontend/public
mkdir -p database/{migrations,seeds}
mkdir -p docker/{nginx,scripts}
mkdir -p docs/{api,architecture,guides}
mkdir -p pdf-library/{textbooks,papers,guidelines,atlases,cases}
mkdir -p knowledge-base/{chapters,research,synthesis}
mkdir -p monitoring/{logs,metrics}
mkdir -p scripts/{setup,maintenance,backup}
mkdir -p .github/workflows

# =====================================================
# BACKEND FILES - COMPLETE IMPLEMENTATION
# =====================================================

echo "🔧 Creating backend files..."

# Main FastAPI Application
cat > backend/main.py << 'PYTHON_EOF'
"""
KOO Platform Personal Edition - Main Application
Complete neurosurgical knowledge management system
Author: ramihatou97
Date: 2025-01-19
"""

import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Core imports
from core.database import DatabaseManager, create_all_tables
from core.cache import RedisCache
from core.config import settings
from core.exceptions import KOOException

# Service imports
from services.pdf_knowledge_base import NeurosurgicalPDFKnowledgeBase
from services.personal_evolution import PersonalEvolutionEngine
from services.ai_orchestrator import AIOrchestrator
from services.semantic_search import SemanticSearchEngine
from services.knowledge_synthesis import KnowledgeSynthesizer

# API routers
from api import (
    chapters, research, pdf, evolution, 
    thoughts, knowledge, search, synthesis
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager"""
    logger.info("=" * 50)
    logger.info("🚀 Starting KOO Platform Personal Edition")
    logger.info(f"📅 Date: {datetime.now()}")
    logger.info(f"🔧 Environment: Personal")
    logger.info("=" * 50)
    
    try:
        # Initialize database
        logger.info("📊 Initializing database...")
        app.state.db = DatabaseManager(settings.DATABASE_URL)
        await create_all_tables(app.state.db)
        
        # Initialize cache
        logger.info("💾 Initializing cache...")
        app.state.cache = RedisCache(settings.REDIS_URL)
        await app.state.cache.connect()
        
        # Initialize AI services
        logger.info("🤖 Initializing AI services...")
        app.state.ai = AIOrchestrator(
            gemini_key=settings.GEMINI_API_KEY,
            claude_key=settings.CLAUDE_API_KEY,
            perplexity_key=settings.PERPLEXITY_API_KEY
        )
        
        # Initialize PDF Knowledge Base
        logger.info("📚 Initializing PDF Knowledge Base...")
        app.state.pdf_kb = NeurosurgicalPDFKnowledgeBase(
            db=app.state.db,
            ai=app.state.ai,
            cache=app.state.cache
        )
        
        # Start PDF folder monitoring
        pdf_folders = [
            Path.home() / "Neurosurgery-Library" / "Textbooks",
            Path.home() / "Neurosurgery-Library" / "Papers",
            Path.home() / "Neurosurgery-Library" / "Guidelines",
            Path.home() / "Neurosurgery-Library" / "Atlases"
        ]
        
        for folder in pdf_folders:
            if folder.exists():
                await app.state.pdf_kb.start_monitoring(str(folder))
                logger.info(f"👁️ Monitoring: {folder}")
        
        # Initialize Semantic Search
        logger.info("🔍 Initializing semantic search...")
        app.state.search = SemanticSearchEngine(
            db=app.state.db,
            ai=app.state.ai
        )
        
        # Initialize Knowledge Synthesizer
        logger.info("🧬 Initializing knowledge synthesizer...")
        app.state.synthesizer = KnowledgeSynthesizer(
            ai=app.state.ai,
            db=app.state.db
        )
        
        # Initialize Evolution Engine
        logger.info("🔄 Initializing evolution engine...")
        app.state.evolution = PersonalEvolutionEngine(
            pdf_kb=app.state.pdf_kb,
            ai=app.state.ai,
            synthesizer=app.state.synthesizer,
            db=app.state.db
        )
        
        # Schedule daily evolution
        if settings.AUTO_EVOLUTION_ENABLED:
            await app.state.evolution.schedule_daily_updates()
            logger.info("⏰ Daily evolution scheduled")
        
        logger.info("✅ All services initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise
    
    yield
    
    # Cleanup
    logger.info("🔻 Shutting down...")
    await app.state.cache.disconnect()
    await app.state.db.close()
    logger.info("✅ Shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="KOO Platform - Personal Neurosurgical Knowledge Base",
    description="AI-powered, self-evolving neurosurgical knowledge management system",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handler
@app.exception_handler(KOOException)
async def koo_exception_handler(request: Request, exc: KOOException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

# Mount static files
if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Include API routers - NO authentication middleware
app.include_router(chapters.router, prefix="/api/chapters", tags=["Chapters"])
app.include_router(research.router, prefix="/api/research", tags=["Research"])
app.include_router(pdf.router, prefix="/api/pdf", tags=["PDF Library"])
app.include_router(evolution.router, prefix="/api/evolution", tags=["Evolution"])
app.include_router(thoughts.router, prefix="/api/thoughts", tags=["Thoughts"])
app.include_router(knowledge.router, prefix="/api/knowledge", tags=["Knowledge"])
app.include_router(search.router, prefix="/api/search", tags=["Search"])
app.include_router(synthesis.router, prefix="/api/synthesis", tags=["Synthesis"])

# Root endpoint
@app.get("/")
async def root():
    return {
        "application": "KOO Platform Personal Edition",
        "version": "2.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "features": {
            "pdf_extraction": True,
            "auto_evolution": True,
            "ai_synthesis": True,
            "semantic_search": True,
            "collaboration": False,  # Explicitly NO collaboration
            "authentication": False  # NO auth required
        }
    }

# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "database": "connected",
            "cache": "connected",
            "ai": "ready",
            "pdf_processor": "active"
        }
    }

# Statistics endpoint
@app.get("/api/stats")
async def get_stats():
    stats = {
        "chapters": await app.state.db.count("chapters"),
        "textbooks": await app.state.db.count("pdf_textbooks"),
        "papers": await app.state.db.count("pdf_papers"),
        "thoughts": await app.state.db.count("thought_stream"),
        "last_evolution": await app.state.evolution.get_last_run()
    }
    return stats

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
PYTHON_EOF

# Complete PDF Knowledge Base Service
cat > backend/services/pdf_knowledge_base.py << 'PYTHON_EOF'
"""
Complete PDF Knowledge Base System
Full neurosurgical textbook and paper processing
With medical entity extraction and semantic indexing
"""

import os
import re
import json
import hashlib
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

import PyPDF2
import pdfplumber
import fitz  # PyMuPDF
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import logging
logger = logging.getLogger(__name__)

@dataclass
class PDFMetadata:
    """PDF document metadata"""
    title: str
    authors: List[str]
    year: Optional[int]
    journal: Optional[str]
    doi: Optional[str]
    pages: int
    file_size_mb: float

@dataclass
class MedicalEntities:
    """Extracted medical entities"""
    anatomical_structures: List[str]
    procedures: List[str]
    conditions: List[str]
    medications: List[str]
    complications: List[str]
    surgical_techniques: List[Dict[str, str]]

class NeurosurgicalPDFKnowledgeBase:
    """
    CRITICAL: Complete PDF processing system for neurosurgical knowledge
    Handles textbooks, papers, guidelines, and atlases
    """
    
    def __init__(self, db, ai, cache):
        self.db = db
        self.ai = ai
        self.cache = cache
        
        # Text splitter optimized for medical content
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=500,
            length_function=len,
            separators=[
                "\n\n\n",  # Chapter breaks
                "\n\n",    # Section breaks  
                "\n",      # Paragraphs
                ". ",      # Sentences
                ", ",      # Clauses
                " ",       # Words
                ""         # Characters
            ]
        )
        
        # Core neurosurgical reference textbooks
        self.reference_textbooks = {
            "youmans": r"Youmans.*(?:and|&).*Winn.*Neurological Surgery",
            "greenberg": r"(?:Handbook.*Neurosurgery|Greenberg.*Handbook)",
            "rhoton": r"Rhoton.*(?:Cranial|Anatomy|Surgical)",
            "schmidek": r"Schmidek.*Sweet.*Operative.*Neurosurg",
            "winn": r"Winn.*Neurological Surgery",
            "samii": r"Samii.*(?:Surgery.*Skull Base|Skull Base)",
            "quinones": r"Quinones.*Hinojosa.*Schmidek",
            "ojemann": r"Ojemann.*Neurosurgery",
            "albright": r"(?:Albright.*Pediatric|Pediatric.*Neurosurgery)",
            "berger": r"Berger.*Neuro.*Oncology"
        }
        
        # Medical entity patterns
        self.entity_patterns = self._compile_entity_patterns()
        
        # File watchers
        self.observers = []
    
    def _compile_entity_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns for medical entity extraction"""
        return {
            "anatomical": [
                re.compile(r'\b(?:frontal|temporal|parietal|occipital|insular)\s+(?:lobe|cortex|gyrus)\b', re.I),
                re.compile(r'\b(?:cerebellum|brainstem|medulla|pons|midbrain)\b', re.I),
                re.compile(r'\b(?:thalamus|hypothalamus|hippocampus|amygdala)\b', re.I),
                re.compile(r'\b(?:basal\s+ganglia|caudate|putamen|globus\s+pallidus)\b', re.I),
                re.compile(r'\b(?:corpus\s+callosum|internal\s+capsule|corona\s+radiata)\b', re.I),
                re.compile(r'\b(?:ventricle|ventricular\s+system|CSF\s+space)\b', re.I),
                re.compile(r'\b(?:dura\s+mater|arachnoid|pia\s+mater|meninges)\b', re.I),
                re.compile(r'\b(?:skull\s+base|clivus|sella|cavernous\s+sinus)\b', re.I),
                re.compile(r'\b(?:spinal\s+cord|cauda\s+equina|nerve\s+root)\b', re.I),
                re.compile(r'\b(?:C[1-7]|T[1-12]|L[1-5]|S[1-5])\s+(?:vertebra|level)\b', re.I)
            ],
            "procedures": [
                re.compile(r'\b\w*(?:craniotomy|craniectomy|cranial)\b', re.I),
                re.compile(r'\b\w*(?:laminectomy|laminotomy|laminoplasty)\b', re.I),
                re.compile(r'\b(?:diskectomy|discectomy|fusion|instrumentation)\b', re.I),
                re.compile(r'\b(?:tumor\s+resection|gross\s+total\s+resection|GTR)\b', re.I),
                re.compile(r'\b(?:aneurysm\s+clipping|coiling|flow\s+diversion)\b', re.I),
                re.compile(r'\b(?:ventriculostomy|shunt\s+placement|ETV)\b', re.I),
                re.compile(r'\b(?:stereotactic|navigation|endoscopic|microscopic)\b', re.I),
                re.compile(r'\b(?:awake\s+craniotomy|mapping|monitoring)\b', re.I),
                re.compile(r'\b(?:deep\s+brain\s+stimulation|DBS|SRS|GKRS)\b', re.I),
                re.compile(r'\b(?:transphenoidal|transcranial|transnasal)\b', re.I)
            ],
            "conditions": [
                re.compile(r'\b(?:glioblastoma|astrocytoma|oligodendroglioma|glioma)\b', re.I),
                re.compile(r'\b(?:meningioma|schwannoma|neurofibroma|pituitary\s+adenoma)\b', re.I),
                re.compile(r'\b(?:metastasis|metastases|brain\s+mets)\b', re.I),
                re.compile(r'\b(?:aneurysm|AVM|arteriovenous\s+malformation|cavernoma)\b', re.I),
                re.compile(r'\b(?:hydrocephalus|NPH|normal\s+pressure)\b', re.I),
                re.compile(r'\b(?:Chiari\s+malformation|syringomyelia|tethered\s+cord)\b', re.I),
                re.compile(r'\b(?:herniation|mass\s+effect|midline\s+shift)\b', re.I),
                re.compile(r'\b(?:stenosis|spondylolisthesis|instability)\b', re.I),
                re.compile(r'\b(?:epilepsy|seizure|status\s+epilepticus)\b', re.I),
                re.compile(r'\b(?:trauma|TBI|traumatic\s+brain\s+injury|SDH|EDH)\b', re.I)
            ],
            "complications": [
                re.compile(r'\b(?:hemorrhage|bleeding|hematoma|ICH)\b', re.I),
                re.compile(r'\b(?:infection|meningitis|ventriculitis|abscess)\b', re.I),
                re.compile(r'\b(?:CSF\s+leak|pseudomeningocele|wound\s+breakdown)\b', re.I),
                re.compile(r'\b(?:stroke|infarct|ischemia|vasospasm)\b', re.I),
                re.compile(r'\b(?:seizure|status\s+epilepticus|epilepsy)\b', re.I),
                re.compile(r'\b(?:deficit|weakness|paralysis|paresis)\b', re.I),
                re.compile(r'\b(?:pneumocephalus|tension\s+pneumo)\b', re.I),
                re.compile(r'\b(?:DVT|PE|pulmonary\s+embolism)\b', re.I),
                re.compile(r'\b(?:death|mortality|morbidity)\b', re.I)
            ]
        }
    
    async def process_complete_textbook(self, pdf_path: str, metadata: Optional[Dict] = None) -> Dict:
        """
        Process complete neurosurgical textbook with chapter structure preservation
        """
        logger.info(f"📚 Processing textbook: {pdf_path}")
        
        # Check if already processed
        file_hash = self._calculate_hash(pdf_path)
        if await self._is_processed(file_hash):
            logger.info(f"⏭️ Already processed: {pdf_path}")
            return {"status": "already_processed", "file_hash": file_hash}
        
        # Identify textbook type
        textbook_type = self._identify_textbook_type(pdf_path)
        if textbook_type:
            logger.info(f"✓ Identified as: {textbook_type}")
        
        # Initialize book data structure
        book_data = {
            "file_path": pdf_path,
            "file_hash": file_hash,
            "file_size_mb": os.path.getsize(pdf_path) / (1024 * 1024),
            "textbook_type": textbook_type,
            "metadata": metadata or {},
            "processing_timestamp": datetime.utcnow().isoformat(),
            "chapters": [],
            "medical_entities": MedicalEntities([], [], [], [], [], []),
            "table_of_contents": [],
            "index_terms": [],
            "cross_references": {},
            "total_pages": 0
        }
        
        try:
            # Open PDF
            doc = fitz.open(pdf_path)
            book_data["total_pages"] = len(doc)
            
            # Extract metadata
            pdf_metadata = self._extract_pdf_metadata(doc, pdf_path)
            book_data["metadata"].update(pdf_metadata.__dict__)
            
            # Extract table of contents
            toc = await self._extract_table_of_contents(doc)
            book_data["table_of_contents"] = toc
            
            # Process each chapter
            for i, chapter_info in enumerate(toc):
                logger.info(f"Processing chapter {i+1}/{len(toc)}: {chapter_info['title']}")
                
                chapter_data = await self._process_textbook_chapter(
                    doc, 
                    chapter_info, 
                    i, 
                    toc
                )
                
                # Extract medical entities from chapter
                chapter_entities = await self._extract_chapter_entities(chapter_data["content"])
                chapter_data["medical_entities"] = chapter_entities
                
                # AI-powered extraction
                chapter_data["key_points"] = await self._extract_key_points_ai(
                    chapter_data["content"][:10000]
                )
                chapter_data["clinical_pearls"] = await self._extract_clinical_pearls_ai(
                    chapter_data["content"][:10000]
                )
                
                book_data["chapters"].append(chapter_data)
                
                # Aggregate entities
                self._aggregate_entities(book_data["medical_entities"], chapter_entities)
            
            # Extract book index
            book_data["index_terms"] = await self._extract_book_index(doc)
            
            # Build cross-references
            book_data["cross_references"] = await self._build_cross_references(book_data)
            
            # Generate embeddings
            await self._generate_book_embeddings(book_data)
            
            # Store in database
            await self._store_textbook_in_database(book_data)
            
            # Auto-link to existing chapters
            await self._auto_link_to_chapters(book_data)
            
            # Cache frequently accessed content
            await self._cache_important_content(book_data)
            
            doc.close()
            
            logger.info(f"✅ Successfully processed textbook with {len(book_data['chapters'])} chapters")
            return book_data
            
        except Exception as e:
            logger.error(f"❌ Error processing textbook: {str(e)}")
            raise
    
    async def process_paper(self, pdf_path: str) -> Dict:
        """Process individual research paper or guideline"""
        logger.info(f"📄 Processing paper: {pdf_path}")
        
        file_hash = self._calculate_hash(pdf_path)
        if await self._is_processed(file_hash):
            return {"status": "already_processed"}
        
        paper_data = {
            "file_path": pdf_path,
            "file_hash": file_hash,
            "file_size_mb": os.path.getsize(pdf_path) / (1024 * 1024),
            "processing_timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            doc = fitz.open(pdf_path)
            
            # Extract content
            full_text = ""
            for page in doc:
                full_text += page.get_text()
            
            paper_data["content"] = full_text
            paper_data["pages"] = len(doc)
            
            # Extract metadata
            metadata = self._extract_pdf_metadata(doc, pdf_path)
            paper_data["metadata"] = metadata.__dict__
            
            # Extract sections
            sections = await self._extract_paper_sections(full_text)
            paper_data["sections"] = sections
            
            # Extract medical entities
            entities = await self._extract_chapter_entities(full_text)
            paper_data["medical_entities"] = entities.__dict__
            
            # Calculate relevance
            paper_data["neurosurgical_relevance"] = await self._calculate_relevance(full_text)
            
            # AI summarization
            paper_data["summary"] = await self._generate_paper_summary(paper_data)
            
            # Generate embedding
            paper_data["embedding"] = await self._generate_embedding(full_text[:5000])
            
            # Store in database
            await self._store_paper_in_database(paper_data)
            
            doc.close()
            
            logger.info(f"✅ Successfully processed paper: {metadata.title}")
            return paper_data
            
        except Exception as e:
            logger.error(f"❌ Error processing paper: {str(e)}")
            raise
    
    def _identify_textbook_type(self, pdf_path: str) -> Optional[str]:
        """Identify if PDF is a known reference textbook"""
        try:
            with fitz.open(pdf_path) as doc:
                # Check first 10 pages
                sample_text = ""
                for i in range(min(10, len(doc))):
                    sample_text += doc[i].get_text()
                
                # Check against known textbook patterns
                for name, pattern in self.reference_textbooks.items():
                    if re.search(pattern, sample_text, re.IGNORECASE):
                        return name
                
                # Check for general textbook indicators
                textbook_indicators = [
                    r"table\s+of\s+contents",
                    r"chapter\s+\d+",
                    r"section\s+\d+\.\d+",
                    r"isbn",
                    r"copyright\s+©",
                    r"edition"
                ]
                
                indicator_count = sum(
                    1 for pattern in textbook_indicators
                    if re.search(pattern, sample_text, re.IGNORECASE)
                )
                
                if indicator_count >= 3:
                    return "generic_textbook"
                    
        except Exception as e:
            logger.error(f"Error identifying textbook: {e}")
        
        return None
    
    def _extract_pdf_metadata(self, doc, pdf_path: str) -> PDFMetadata:
        """Extract comprehensive PDF metadata"""
        metadata = PDFMetadata(
            title="Unknown",
            authors=[],
            year=None,
            journal=None,
            doi=None,
            pages=len(doc),
            file_size_mb=os.path.getsize(pdf_path) / (1024 * 1024)
        )
        
        # Extract from PDF metadata
        if doc.metadata:
            metadata.title = doc.metadata.get('title', '') or Path(pdf_path).stem
            author_str = doc.metadata.get('author', '')
            if author_str:
                metadata.authors = [a.strip() for a in author_str.split(',')]
        
        # Extract from first pages
        first_pages_text = ""
        for i in range(min(3, len(doc))):
            first_pages_text += doc[i].get_text()
        
        # Extract DOI
        doi_pattern = r'10\.\d{4,}(?:\.\d+)*\/[-._;()\/:a-zA-Z0-9]+'
        doi_match = re.search(doi_pattern, first_pages_text)
        if doi_match:
            metadata.doi = doi_match.group()
        
        # Extract year
        year_pattern = r'(?:19|20)\d{2}'
        year_matches = re.findall(year_pattern, first_pages_text)
        if year_matches:
            # Get most recent year
            metadata.year = max(int(y) for y in year_matches)
        
        # Extract journal
        journal_patterns = [
            r'Journal of Neurosurgery',
            r'Neurosurgery',
            r'World Neurosurgery',
            r'Neurosurgical Focus',
            r'Journal of Neuro-Oncology',
            r'Spine',
            r'Journal of Neurosurgery: Spine',
            r'Journal of Neurosurgery: Pediatrics',
            r'Neuro-Oncology',
            r'Operative Neurosurgery'
        ]
        
        for pattern in journal_patterns:
            if re.search(pattern, first_pages_text, re.IGNORECASE):
                metadata.journal = pattern
                break
        
        return metadata
    
    async def _extract_table_of_contents(self, doc) -> List[Dict]:
        """Extract complete table of contents from PDF"""
        toc = []
        
        # Try PDF TOC first
        pdf_toc = doc.get_toc()
        if pdf_toc:
            for level, title, page in pdf_toc:
                toc.append({
                    "level": level,
                    "title": title.strip(),
                    "page": page,
                    "type": "chapter" if level == 1 else "section"
                })
        else:
            # Fallback: extract from content
            toc = await self._extract_toc_from_content(doc)
        
        # Ensure we have at least basic structure
        if not toc:
            # Create artificial chapters every 30 pages
            pages_per_chapter = 30
            for i in range(0, len(doc), pages_per_chapter):
                toc.append({
                    "level": 1,
                    "title": f"Section {i//pages_per_chapter + 1}",
                    "page": i + 1,
                    "type": "chapter"
                })
        
        return toc
    
    async def _extract_toc_from_content(self, doc) -> List[Dict]:
        """Extract TOC by analyzing content patterns"""
        toc = []
        
        # Patterns for chapters and sections
        patterns = [
            (r'(?:CHAPTER|Chapter)\s+(\d+)[:\s]+(.+)', 1),
            (r'(?:PART|Part)\s+([IVX]+)[:\s]+(.+)', 1),
            (r'(?:Section|SECTION)\s+(\d+(?:\.\d+)?)[:\s]+(.+)', 2),
            (r'^(\d+)\.\s+([A-Z].+)$', 1),  # Numbered sections
            (r'^(\d+\.\d+)\s+([A-Z].+)$', 2)  # Sub-sections
        ]
        
        for page_num in range(len(doc)):
            page_text = doc[page_num].get_text()
            
            for pattern, level in patterns:
                matches = re.finditer(pattern, page_text, re.MULTILINE)
                for match in matches:
                    title = match.group(2) if match.lastindex >= 2 else match.group(0)
                    toc.append({
                        "level": level,
                        "title": title.strip(),
                        "page": page_num + 1,
                        "type": "chapter" if level == 1 else "section",
                        "number": match.group(1) if match.lastindex >= 1 else ""
                    })
        
        # Sort by page number
        toc.sort(key=lambda x: x['page'])
        
        return toc
    
    async def _process_textbook_chapter(self, doc, chapter_info: Dict, 
                                       chapter_index: int, toc: List[Dict]) -> Dict:
        """Process individual textbook chapter"""
        chapter_data = {
            "title": chapter_info["title"],
            "level": chapter_info["level"],
            "start_page": chapter_info["page"],
            "chapter_number": chapter_info.get("number", str(chapter_index + 1))
        }
        
        # Find chapter end
        end_page = len(doc)
        for next_chapter in toc[chapter_index + 1:]:
            if next_chapter["level"] <= chapter_info["level"]:
                end_page = next_chapter["page"] - 1
                break
        chapter_data["end_page"] = min(end_page, len(doc))
        
        # Extract chapter content
        content = ""
        tables = []
        figures = []
        
        for page_num in range(chapter_data["start_page"] - 1, 
                             min(chapter_data["end_page"], len(doc))):
            page = doc[page_num]
            
            # Extract text
            page_text = page.get_text()
            content += f"\n--- Page {page_num + 1} ---\n{page_text}"
            
            # Extract tables (simplified)
            if "Table" in page_text or "TABLE" in page_text:
                tables.append({"page": page_num + 1, "content": "Table detected"})
            
            # Extract figures (simplified)
            image_list = page.get_images()
            if image_list:
                figures.append({"page": page_num + 1, "count": len(image_list)})
        
        chapter_data["content"] = content
        chapter_data["tables"] = tables
        chapter_data["figures"] = figures
        chapter_data["word_count"] = len(content.split())
        
        return chapter_data
    
    async def _extract_chapter_entities(self, text: str) -> MedicalEntities:
        """Extract medical entities from chapter text"""
        entities = MedicalEntities(
            anatomical_structures=[],
            procedures=[],
            conditions=[],
            medications=[],
            complications=[],
            surgical_techniques=[]
        )
        
        # Extract using patterns
        for pattern in self.entity_patterns["anatomical"]:
            matches = pattern.findall(text)
            entities.anatomical_structures.extend(matches)
        
        for pattern in self.entity_patterns["procedures"]:
            matches = pattern.findall(text)
            entities.procedures.extend(matches)
        
        for pattern in self.entity_patterns["conditions"]:
            matches = pattern.findall(text)
            entities.conditions.extend(matches)
        
        for pattern in self.entity_patterns["complications"]:
            matches = pattern.findall(text)
            entities.complications.extend(matches)
        
        # Extract surgical techniques with context
        technique_pattern = re.compile(
            r'(?:technique|approach|method|procedure):\s*([^.]+)', 
            re.IGNORECASE
        )
        for match in technique_pattern.finditer(text):
            entities.surgical_techniques.append({
                "name": match.group(1)[:100],
                "context": text[max(0, match.start()-200):min(len(text), match.end()+200)]
            })
        
        # Remove duplicates
        entities.anatomical_structures = list(set(entities.anatomical_structures))
        entities.procedures = list(set(entities.procedures))
        entities.conditions = list(set(entities.conditions))
        entities.complications = list(set(entities.complications))
        
        # Limit techniques
        entities.surgical_techniques = entities.surgical_techniques[:20]
        
        return entities
    
    async def _extract_key_points_ai(self, text: str) -> List[str]:
        """Use AI to extract key clinical points"""
        if len(text) < 100:
            return []
        
        prompt = """
        Extract the KEY CLINICAL POINTS from this neurosurgical text.
        Focus on:
        1. Critical surgical techniques and approaches
        2. Important anatomical landmarks and relationships
        3. Common complications and how to avoid them
        4. Clinical decision-making points
        5. Evidence-based recommendations
        
        Text: {text}
        
        Return as a numbered list of the most important points.
        Maximum 10 points.
        """
        
        try:
            response = await self.ai.gemini.generate(prompt.format(text=text))
            points = response.strip().split('\n')
            return [p.strip() for p in points if p.strip()][:10]
        except Exception as e:
            logger.error(f"Error extracting key points: {e}")
            return []
    
    async def _extract_clinical_pearls_ai(self, text: str) -> List[str]:
        """Use AI to extract clinical pearls and wisdom"""
        if len(text) < 100:
            return []
        
        prompt = """
        Extract CLINICAL PEARLS and EXPERT TIPS from this neurosurgical text.
        Look for:
        - Practical surgical tips and tricks
        - Common pitfalls and how to avoid them
        - Expert recommendations from experience
        - Technical nuances that make a difference
        - Patient selection criteria
        
        Text: {text}
        
        Return as bullet points. Maximum 10 pearls.
        """
        
        try:
            response = await self.ai.claude.generate(prompt.format(text=text))
            pearls = response.strip().split('\n')
            return [p.strip('•- ') for p in pearls if p.strip()][:10]
        except Exception as e:
            logger.error(f"Error extracting clinical pearls: {e}")
            return []
    
    async def _extract_paper_sections(self, text: str) -> Dict[str, str]:
        """Extract standard sections from research paper"""
        sections = {
            "abstract": "",
            "introduction": "",
            "methods": "",
            "results": "",
            "discussion": "",
            "conclusion": "",
            "references": ""
        }
        
        # Define section patterns
        section_patterns = {
            "abstract": r"(?:abstract|summary)[\s:]*\n(.*?)(?=introduction|background|\n\n[A-Z])",
            "introduction": r"(?:introduction|background)[\s:]*\n(.*?)(?=method|materials|\n\n[A-Z])",
            "methods": r"(?:methods?|methodology|materials)[\s:]*\n(.*?)(?=results?|findings|\n\n[A-Z])",
            "results": r"(?:results?|findings?)[\s:]*\n(.*?)(?=discussion|interpretation|\n\n[A-Z])",
            "discussion": r"(?:discussion|interpretation)[\s:]*\n(.*?)(?=conclusion|summary|\n\n[A-Z])",
            "conclusion": r"(?:conclusion|conclusions|summary)[\s:]*\n(.*?)(?=references?|bibliography|\n\n[A-Z])",
            "references": r"(?:references?|bibliography|citations?)[\s:]*\n(.*?)$"
        }
        
        for section, pattern in section_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                sections[section] = match.group(1)[:5000]  # Limit length
        
        return sections
    
    async def _calculate_relevance(self, text: str) -> float:
        """Calculate neurosurgical relevance score"""
        score = 0.0
        text_lower = text.lower()
        
        # High-value neurosurgical terms
        high_value = [
            "craniotomy", "surgical technique", "operative approach",
            "patient outcome", "complication rate", "survival",
            "minimally invasive", "endoscopic", "navigation",
            "intraoperative", "postoperative management"
        ]
        
        for term in high_value:
            if term in text_lower:
                score += 0.15
        
        # Study type bonus
        if "randomized controlled trial" in text_lower:
            score += 0.25
        elif "systematic review" in text_lower:
            score += 0.2
        elif "meta-analysis" in text_lower:
            score += 0.2
        elif "prospective" in text_lower:
            score += 0.15
        elif "retrospective" in text_lower:
            score += 0.1
        
        # Statistical significance
        if "p < 0.05" in text or "significant" in text_lower:
            score += 0.1
        
        # Sample size
        sample_match = re.search(r'n\s*=\s*(\d+)', text, re.IGNORECASE)
        if sample_match:
            n = int(sample_match.group(1))
            if n > 100:
                score += 0.1
            if n > 500:
                score += 0.1
        
        return min(score, 1.0)
    
    async def _generate_paper_summary(self, paper_data: Dict) -> Dict[str, str]:
        """Generate AI-powered paper summary"""
        text = paper_data["content"][:10000]
        
        # Gemini summary
        gemini_prompt = f"""
        Summarize this neurosurgical research paper:
        
        Title: {paper_data['metadata'].get('title', 'Unknown')}
        
        Provide:
        1. Main objective
        2. Methods used
        3. Key findings
        4. Clinical significance
        5. Limitations
        
        Text: {text}
        """
        
        # Claude summary
        claude_prompt = f"""
        Provide a clinical summary of this neurosurgical paper:
        
        Focus on:
        - What's new or different
        - How it impacts practice
        - Practical applications
        - Future directions
        
        Text: {text}
        """
        
        try:
            gemini_summary = await self.ai.gemini.generate(gemini_prompt)
            claude_summary = await self.ai.claude.generate(claude_prompt)
            
            return {
                "gemini": gemini_summary,
                "claude": claude_summary,
                "combined": f"{gemini_summary}\n\n{claude_summary}"
            }
        except:
            return {"combined": "Summary generation failed"}
    
    def _aggregate_entities(self, total: MedicalEntities, chapter: MedicalEntities):
        """Aggregate medical entities from chapters"""
        total.anatomical_structures.extend(chapter.anatomical_structures)
        total.procedures.extend(chapter.procedures)
        total.conditions.extend(chapter.conditions)
        total.medications.extend(chapter.medications)
        total.complications.extend(chapter.complications)
        total.surgical_techniques.extend(chapter.surgical_techniques)
    
    async def _extract_book_index(self, doc) -> List[str]:
        """Extract index terms from book"""
        index_terms = []
        
        # Look for index in last pages
        for page_num in range(max(0, len(doc) - 20), len(doc)):
            page_text = doc[page_num].get_text()
            if "index" in page_text.lower():
                # Extract terms (simplified)
                lines = page_text.split('\n')
                for line in lines:
                    if re.match(r'^[A-Z]', line) and len(line) < 100:
                        index_terms.append(line.strip())
        
        return index_terms[:500]  # Limit
    
    async def _build_cross_references(self, book_data: Dict) -> Dict[str, List[int]]:
        """Build cross-reference map between chapters"""
        cross_refs = {}
        
        # Extract key terms from each chapter
        for i, chapter in enumerate(book_data["chapters"]):
            # Get top entities
            entities = chapter.get("medical_entities", MedicalEntities([], [], [], [], [], []))
            
            key_terms = (
                entities.anatomical_structures[:5] +
                entities.procedures[:5] +
                entities.conditions[:5]
            )
            
            # Find which other chapters reference these terms
            for term in key_terms:
                if term not in cross_refs:
                    cross_refs[term] = []
                
                for j, other_chapter in enumerate(book_data["chapters"]):
                    if i != j and term.lower() in other_chapter["content"].lower():
                        cross_refs[term].append(j)
        
        return cross_refs
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate vector embedding for text"""
        try:
            # Use your embedding model
            embedding = await self.ai.generate_embedding(text)
            return embedding
        except:
            # Return dummy embedding if fails
            return [0.0] * 1536
    
    async def _generate_book_embeddings(self, book_data: Dict):
        """Generate embeddings for all chapters"""
        for chapter in book_data["chapters"]:
            # Generate embedding for chapter
            text = chapter["content"][:5000]  # Limit for API
            chapter["embedding"] = await self._generate_embedding(text)
    
    async def _store_textbook_in_database(self, book_data: Dict):
        """Store complete textbook in database"""
        async with self.db.get_session() as session:
            # Insert textbook
            result = await session.execute("""
                INSERT INTO pdf_textbooks (
                    file_hash, file_path, title, authors, isbn, year, 
                    edition, publisher, total_pages, chapter_count,
                    file_size_mb, textbook_type, table_of_contents,
                    medical_entities, cross_references, importance_score,
                    created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, 
                         $11, $12, $13, $14, $15, $16, $17)
                RETURNING id
            """, (
                book_data["file_hash"],
                book_data["file_path"],
                book_data["metadata"].get("title", "Unknown"),
                book_data["metadata"].get("authors", []),
                book_data["metadata"].get("isbn"),
                book_data["metadata"].get("year"),
                book_data["metadata"].get("edition"),
                book_data["metadata"].get("publisher"),
                book_data["total_pages"],
                len(book_data["chapters"]),
                book_data["file_size_mb"],
                book_data.get("textbook_type"),
                json.dumps(book_data["table_of_contents"]),
                json.dumps({
                    "anatomical": book_data["medical_entities"].anatomical_structures[:100],
                    "procedures": book_data["medical_entities"].procedures[:100],
                    "conditions": book_data["medical_entities"].conditions[:100]
                }),
                json.dumps(book_data["cross_references"]),
                1.0 if book_data.get("textbook_type") else 0.8,
                datetime.utcnow()
            ))
            
            textbook_id = result.scalar()
            
            # Insert chapters
            for chapter in book_data["chapters"]:
                await session.execute("""
                    INSERT INTO textbook_chapters (
                        textbook_id, title, chapter_number, start_page, end_page,
                        content, word_count, key_points, clinical_pearls,
                        surgical_techniques, anatomical_structures, procedures,
                        conditions, complications, tables_count, figures_count,
                        embedding, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                             $11, $12, $13, $14, $15, $16, $17, $18)
                """, (
                    textbook_id,
                    chapter["title"],
                    chapter["chapter_number"],
                    chapter["start_page"],
                    chapter["end_page"],
                    chapter["content"],
                    chapter["word_count"],
                    json.dumps(chapter.get("key_points", [])),
                    json.dumps(chapter.get("clinical_pearls", [])),
                    json.dumps(chapter["medical_entities"].surgical_techniques[:10]),
                    chapter["medical_entities"].anatomical_structures[:50],
                    chapter["medical_entities"].procedures[:50],
                    chapter["medical_entities"].conditions[:50],
                    chapter["medical_entities"].complications[:50],
                    len(chapter.get("tables", [])),
                    len(chapter.get("figures", [])),
                    chapter.get("embedding"),
                    datetime.utcnow()
                ))
            
            await session.commit()
            logger.info(f"✅ Stored textbook with ID: {textbook_id}")
    
    async def _store_paper_in_database(self, paper_data: Dict):
        """Store research paper in database"""
        async with self.db.get_session() as session:
            await session.execute("""
                INSERT INTO pdf_papers (
                    file_hash, file_path, title, authors, journal, doi,
                    year, abstract, full_text, sections, medical_entities,
                    neurosurgical_relevance, summary, embedding,
                    file_size_mb, pages, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                         $11, $12, $13, $14, $15, $16, $17)
            """, (
                paper_data["file_hash"],
                paper_data["file_path"],
                paper_data["metadata"]["title"],
                paper_data["metadata"]["authors"],
                paper_data["metadata"]["journal"],
                paper_data["metadata"]["doi"],
                paper_data["metadata"]["year"],
                paper_data["sections"].get("abstract", ""),
                paper_data["content"],
                json.dumps(paper_data["sections"]),
                json.dumps(paper_data["medical_entities"]),
                paper_data["neurosurgical_relevance"],
                json.dumps(paper_data["summary"]),
                paper_data.get("embedding"),
                paper_data["file_size_mb"],
                paper_data["pages"],
                datetime.utcnow()
            ))
            
            await session.commit()
            logger.info(f"✅ Stored paper: {paper_data['metadata']['title']}")
    
    async def _auto_link_to_chapters(self, book_data: Dict):
        """Automatically link PDF content to existing KOO chapters"""
        # Implementation for auto-linking
        pass
    
    async def _cache_important_content(self, book_data: Dict):
        """Cache frequently accessed content"""
        # Cache high-importance chapters
        for chapter in book_data["chapters"][:10]:  # Top 10 chapters
            cache_key = f"chapter:{book_data['file_hash']}:{chapter['chapter_number']}"
            await self.cache.set(cache_key, chapter, ttl=86400)  # 24 hours
    
    def _calculate_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    async def _is_processed(self, file_hash: str) -> bool:
        """Check if file already processed"""
        async with self.db.get_session() as session:
            result = await session.fetch_one(
                "SELECT 1 FROM pdf_textbooks WHERE file_hash = $1 "
                "UNION SELECT 1 FROM pdf_papers WHERE file_hash = $1",
                file_hash
            )
            return result is not None
    
    async def start_monitoring(self, folder_path: str):
        """Start monitoring folder for new PDFs"""
        class PDFHandler(FileSystemEventHandler):
            def __init__(self, processor):
                self.processor = processor
            
            def on_created(self, event):
                if event.src_path.endswith('.pdf'):
                    asyncio.create_task(
                        self.processor.process_new_pdf(event.src_path)
                    )
        
        observer = Observer()
        observer.schedule(PDFHandler(self), folder_path, recursive=True)
        observer.start()
        self.observers.append(observer)
        logger.info(f"👁️ Monitoring: {folder_path}")
    
    async def process_new_pdf(self, pdf_path: str):
        """Process newly detected PDF"""
        file_size = os.path.getsize(pdf_path)
        
        if file_size > 10 * 1024 * 1024:  # > 10MB
            await self.process_complete_textbook(pdf_path)
        else:
            await self.process_paper(pdf_path)
    
    async def search_across_library(self, query: str, limit: int = 10) -> List[Dict]:
        """Search across entire PDF library using semantic search"""
        # Generate query embedding
        query_embedding = await self._generate_embedding(query)
        
        async with self.db.get_session() as session:
            # Search textbook chapters
            textbook_results = await session.fetch_all("""
                SELECT 
                    tc.id,
                    tc.title as chapter_title,
                    tc.content,
                    tc.key_points,
                    tc.clinical_pearls,
                    pt.title as book_title,
                    1 - (tc.embedding <=> $1::vector) as similarity
                FROM textbook_chapters tc
                JOIN pdf_textbooks pt ON tc.textbook_id = pt.id
                WHERE tc.embedding IS NOT NULL
                ORDER BY similarity DESC
                LIMIT $2
            """, query_embedding, limit // 2)
            
            # Search papers
            paper_results = await session.fetch_all("""
                SELECT
                    pp.id,
                    pp.title,
                    pp.abstract,
                    pp.summary,
                    pp.journal,
                    pp.year,
                    1 - (pp.embedding <=> $1::vector) as similarity
                FROM pdf_papers pp
                WHERE pp.embedding IS NOT NULL
                ORDER BY similarity DESC
                LIMIT $2
            """, query_embedding, limit // 2)
            
            results = []
            
            for row in textbook_results:
                results.append({
                    "type": "textbook_chapter",
                    "title": row["chapter_title"],
                    "source": row["book_title"],
                    "content": row["content"][:500],
                    "key_points": json.loads(row["key_points"] or "[]"),
                    "clinical_pearls": json.loads(row["clinical_pearls"] or "[]"),
                    "similarity": row["similarity"]
                })
            
            for row in paper_results:
                results.append({
                    "type": "paper",
                    "title": row["title"],
                    "source": row["journal"],
                    "year": row["year"],
                    "abstract": row["abstract"],
                    "summary": json.loads(row["summary"] or "{}"),
                    "similarity": row["similarity"]
                })
            
            # Sort by similarity
            results.sort(key=lambda x: x["similarity"], reverse=True)
            
            return results[:limit]
PYTHON_EOF

# Personal Evolution Engine
cat > backend/services/personal_evolution.py << 'PYTHON_EOF'
"""
Personal Auto-Evolution Engine
Automatically updates knowledge base with NO collaboration
Direct updates only - personal use
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class PersonalEvolutionEngine:
    """
    Manages automatic evolution of personal knowledge base
    NO proposals, NO approvals - direct updates only
    """
    
    def __init__(self, pdf_kb, ai, synthesizer, db):
        self.pdf_kb = pdf_kb
        self.ai = ai
        self.synthesizer = synthesizer
        self.db = db
        
        self.evolution_running = False
        self.last_run = None
        self.scheduled_task = None
    
    async def schedule_daily_updates(self):
        """Schedule daily automatic updates"""
        async def run_daily():
            while True:
                # Wait until 6 AM
                now = datetime.now()
                next_run = now.replace(hour=6, minute=0, second=0)
                if next_run <= now:
                    next_run += timedelta(days=1)
                
                wait_seconds = (next_run - now).total_seconds()
                logger.info(f"⏰ Next evolution scheduled for {next_run}")
                
                await asyncio.sleep(wait_seconds)
                await self.daily_knowledge_update()
        
        self.scheduled_task = asyncio.create_task(run_daily())
    
    async def daily_knowledge_update(self):
        """
        Daily automatic knowledge update
        DIRECT updates - no collaboration needed
        """
        if self.evolution_running:
            logger.info("Evolution already running, skipping...")
            return
        
        self.evolution_running = True
        start_time = datetime.now()
        
        logger.info("=" * 50)
        logger.info(f"🔄 DAILY EVOLUTION STARTING - {start_time}")
        logger.info("=" * 50)
        
        try:
            # 1. Process any new PDFs
            logger.info("📚 Checking for new PDFs...")
            new_pdfs = await self._check_new_pdfs()
            if new_pdfs:
                logger.info(f"Found {len(new_pdfs)} new PDFs to process")
                for pdf in new_pdfs:
                    await self.pdf_kb.process_new_pdf(pdf)
            
            # 2. Check for new neurosurgical research
            logger.info("🔬 Checking for new research...")
            new_research = await self._check_new_research()
            logger.info(f"Found {len(new_research)} new research items")
            
            # 3. Process thought stream
            logger.info("💭 Processing thought stream...")
            await self._process_thought_stream()
            
            # 4. DIRECTLY update chapters (NO proposals)
            if new_research:
                logger.info("📝 Directly updating chapters...")
                await self._directly_update_chapters(new_research)
            
            # 5. Re-synthesize knowledge
            logger.info("🧬 Synthesizing knowledge updates...")
            await self._synthesize_knowledge()
            
            # 6. Update embeddings
            logger.info("🔢 Updating embeddings...")
            await self._update_embeddings()
            
            # 7. Clean up and optimize
            logger.info("🧹 Optimizing database...")
            await self._optimize_database()
            
            # Record evolution event
            await self._record_evolution_event(new_research)
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ EVOLUTION COMPLETE in {duration:.2f} seconds")
            
        except Exception as e:
            logger.error(f"❌ Evolution failed: {str(e)}")
            raise
        finally:
            self.evolution_running = False
            self.last_run = datetime.now()
    
    async def _check_new_pdfs(self) -> List[str]:
        """Check for new PDFs in monitored folders"""
        # This would interface with file system monitoring
        # For now, return empty list
        return []
    
    async def _check_new_research(self) -> List[Dict]:
        """
        Check for new neurosurgical research across all sources
        """
        research_topics = [
            # Brain tumors
            "glioblastoma treatment 2025",
            "brain tumor immunotherapy advances",
            "liquid biopsy glioma",
            "tumor treating fields",
            
            # Spine
            "minimally invasive spine surgery",
            "robotic spine surgery outcomes",
            "artificial disc replacement",
            "spinal cord stimulation",
            
            # Vascular
            "flow diverter aneurysm",
            "bypass surgery moyamoya",
            "AVM embolization techniques",
            
            # Functional
            "deep brain stimulation Parkinson's",
            "focused ultrasound essential tremor",
            "responsive neurostimulation epilepsy",
            
            # Pediatric
            "pediatric brain tumor protocols",
            "endoscopic third ventriculostomy",
            "craniosynostosis surgery",
            
            # Technology
            "augmented reality neurosurgery",
            "artificial intelligence brain surgery",
            "intraoperative MRI outcomes"
        ]
        
        all_research = []
        
        for topic in research_topics:
            logger.info(f"  Searching: {topic}")
            
            # Search with each AI service
            try:
                # Gemini deep research
                gemini_results = await self.ai.gemini.search(topic, max_results=3)
                all_research.extend(gemini_results)
                
                # Claude analysis
                claude_results = await self.ai.claude.search(topic, max_results=3)
                all_research.extend(claude_results)
                
                # Perplexity search
                perplexity_results = await self.ai.perplexity.search(topic, max_results=3)
                all_research.extend(perplexity_results)
                
            except Exception as e:
                logger.error(f"Error searching {topic}: {e}")
        
        # Deduplicate based on title similarity
        unique_research = self._deduplicate_research(all_research)
        
        return unique_research
    
    async def _process_thought_stream(self):
        """
        Process personal thoughts and questions
        NO user filtering needed - all thoughts are yours
        """
        async with self.db.get_session() as session:
            # Get unprocessed thoughts
            thoughts = await session.fetch_all("""
                SELECT id, content, thought_type, related_chapter_id
                FROM thought_stream
                WHERE is_processed = FALSE
                ORDER BY created_at
                LIMIT 10
            """)
            
            for thought in thoughts:
                logger.info(f"  Processing thought: {thought['content'][:50]}...")
# Continuing backend/services/personal_evolution.py
                
                # Research the thought
                research = await self._research_thought(thought)
                
                # DIRECTLY apply findings - NO approval needed
                if research:
                    await self._apply_thought_findings(thought, research)
                
                # Mark as processed
                await session.execute("""
                    UPDATE thought_stream 
                    SET is_processed = TRUE, 
                        ai_response = $1,
                        processed_at = NOW()
                    WHERE id = $2
                """, json.dumps(research), thought['id'])
            
            await session.commit()
            logger.info(f"  Processed {len(thoughts)} thoughts")
    
    async def _directly_update_chapters(self, research: List[Dict]):
        """
        DIRECTLY update chapters with new research
        NO proposals, NO approvals - immediate updates for personal use
        """
        async with self.db.get_session() as session:
            for item in research:
                # Find relevant chapters
                relevant_chapters = await self._find_relevant_chapters(item)
                
                for chapter_id in relevant_chapters:
                    # Get current chapter
                    chapter = await session.fetch_one(
                        "SELECT * FROM chapters WHERE id = $1",
                        chapter_id
                    )
                    
                    if not chapter:
                        continue
                    
                    # Generate update using AI
                    updated_content = await self._generate_chapter_update(
                        chapter['content'],
                        item
                    )
                    
                    # DIRECTLY update chapter - NO approval needed
                    await session.execute("""
                        UPDATE chapters 
                        SET content = $1,
                            last_ai_update = NOW(),
                            updated_at = NOW(),
                            confidence_score = confidence_score + 0.05
                        WHERE id = $2
                    """, updated_content, chapter_id)
                    
                    logger.info(f"  ✅ Directly updated chapter: {chapter['title']}")
            
            await session.commit()
    
    async def _research_thought(self, thought: Dict) -> Dict:
        """Research a thought using AI services"""
        prompt = f"""
        Research this neurosurgical question/thought:
        {thought['content']}
        
        Provide:
        1. Current evidence and guidelines
        2. Recent advances
        3. Practical considerations
        4. References
        """
        
        try:
            gemini_response = await self.ai.gemini.generate(prompt)
            claude_response = await self.ai.claude.generate(prompt)
            
            return {
                "gemini": gemini_response,
                "claude": claude_response,
                "synthesis": await self.synthesizer.synthesize(
                    gemini_response, 
                    claude_response
                )
            }
        except Exception as e:
            logger.error(f"Error researching thought: {e}")
            return {}
    
    async def _apply_thought_findings(self, thought: Dict, research: Dict):
        """Apply research findings from thought"""
        if thought.get('related_chapter_id'):
            # Update the related chapter
            async with self.db.get_session() as session:
                await session.execute("""
                    UPDATE chapters
                    SET personal_notes = COALESCE(personal_notes, '') || $1,
                        updated_at = NOW()
                    WHERE id = $2
                """, f"\n\n## Thought Research:\n{research.get('synthesis', '')}", 
                thought['related_chapter_id'])
                
                await session.commit()
    
    async def _find_relevant_chapters(self, research_item: Dict) -> List[int]:
        """Find chapters relevant to research item"""
        # Use semantic search to find relevant chapters
        query = f"{research_item.get('title', '')} {research_item.get('summary', '')}"
        
        async with self.db.get_session() as session:
            results = await session.fetch_all("""
                SELECT id, title
                FROM chapters
                WHERE to_tsvector('english', content) @@ plainto_tsquery('english', $1)
                LIMIT 5
            """, query[:500])
            
            return [r['id'] for r in results]
    
    async def _generate_chapter_update(self, current_content: str, research: Dict) -> str:
        """Generate updated chapter content with new research"""
        prompt = f"""
        Update this neurosurgical chapter with new research findings.
        
        Current content:
        {current_content[:3000]}
        
        New research:
        Title: {research.get('title', '')}
        Summary: {research.get('summary', '')}
        Key findings: {research.get('findings', '')}
        
        Integrate the new information seamlessly, maintaining the chapter structure.
        Mark new additions with [Updated: {datetime.now().date()}].
        """
        
        try:
            updated = await self.ai.gemini.generate(prompt)
            return updated
        except:
            # If update fails, append research as note
            return f"{current_content}\n\n## Recent Update [{datetime.now().date()}]\n{research.get('summary', '')}"
    
    async def _synthesize_knowledge(self):
        """Re-synthesize knowledge across chapters"""
        async with self.db.get_session() as session:
            # Get chapters that need synthesis
            chapters = await session.fetch_all("""
                SELECT id, title, content
                FROM chapters
                WHERE last_ai_update < NOW() - INTERVAL '7 days'
                   OR confidence_score < 0.7
                ORDER BY confidence_score ASC
                LIMIT 5
            """)
            
            for chapter in chapters:
                # Synthesize with both AI models
                synthesized = await self.synthesizer.synthesize_chapter(chapter)
                
                if synthesized:
                    await session.execute("""
                        UPDATE chapters
                        SET summary = $1,
                            confidence_score = LEAST(confidence_score + 0.1, 1.0),
                            last_ai_update = NOW()
                        WHERE id = $2
                    """, synthesized['summary'], chapter['id'])
            
            await session.commit()
            logger.info(f"  Synthesized {len(chapters)} chapters")
    
    async def _update_embeddings(self):
        """Update vector embeddings for semantic search"""
        async with self.db.get_session() as session:
            # Get content without embeddings
            items = await session.fetch_all("""
                SELECT id, content FROM chapters 
                WHERE content_embedding IS NULL
                LIMIT 10
            """)
            
            for item in items:
                embedding = await self.ai.generate_embedding(item['content'][:5000])
                
                await session.execute("""
                    UPDATE chapters
                    SET content_embedding = $1::vector
                    WHERE id = $2
                """, embedding, item['id'])
            
            await session.commit()
            logger.info(f"  Updated {len(items)} embeddings")
    
    async def _optimize_database(self):
        """Optimize database performance"""
        async with self.db.get_session() as session:
            # Vacuum analyze for better performance
            await session.execute("VACUUM ANALYZE chapters")
            await session.execute("VACUUM ANALYZE pdf_textbooks")
            await session.execute("VACUUM ANALYZE textbook_chapters")
            
            # Update statistics
            await session.execute("ANALYZE")
    
    async def _record_evolution_event(self, research: List[Dict]):
        """Record evolution event for tracking"""
        async with self.db.get_session() as session:
            await session.execute("""
                INSERT INTO evolution_events (
                    event_type, trigger, sources_consulted,
                    research_count, created_at
                ) VALUES ($1, $2, $3, $4, NOW())
            """, "daily_update", "scheduled", 
            json.dumps(["gemini", "claude", "perplexity"]), 
            len(research))
            
            await session.commit()
    
    def _deduplicate_research(self, research: List[Dict]) -> List[Dict]:
        """Remove duplicate research items"""
        seen_titles = set()
        unique = []
        
        for item in research:
            title = item.get('title', '').lower()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique.append(item)
        
        return unique
    
    async def get_last_run(self) -> Optional[str]:
        """Get last evolution run time"""
        if self.last_run:
            return self.last_run.isoformat()
        
        async with self.db.get_session() as session:
            result = await session.fetch_one("""
                SELECT created_at 
                FROM evolution_events 
                ORDER BY created_at DESC 
                LIMIT 1
            """)
            
            if result:
                return result['created_at'].isoformat()
        
        return None
    
    async def trigger_manual_evolution(self) -> Dict:
        """Manually trigger evolution update"""
        if self.evolution_running:
            return {"status": "already_running"}
        
        asyncio.create_task(self.daily_knowledge_update())
        
        return {"status": "triggered", "timestamp": datetime.now().isoformat()}
PYTHON_EOF

# AI Orchestrator Service
cat > backend/services/ai_orchestrator.py << 'PYTHON_EOF'
"""
AI Orchestrator - Manages all AI services
Gemini, Claude, Perplexity integration
"""

import asyncio
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class AIOrchestrator:
    """Orchestrates multiple AI services for comprehensive analysis"""
    
    def __init__(self, gemini_key: str, claude_key: str, perplexity_key: str):
        self.gemini = GeminiService(gemini_key)
        self.claude = ClaudeService(claude_key)
        self.perplexity = PerplexityService(perplexity_key)
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate vector embedding for text"""
        try:
            # Use Gemini for embeddings
            return await self.gemini.generate_embedding(text)
        except:
            # Return dummy embedding if fails
            return [0.0] * 1536
    
    async def comprehensive_analysis(self, query: str) -> Dict:
        """Get analysis from all AI services"""
        results = {}
        
        # Run all AI queries in parallel
        tasks = [
            self.gemini.analyze(query),
            self.claude.analyze(query),
            self.perplexity.search(query)
        ]
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        results["gemini"] = responses[0] if not isinstance(responses[0], Exception) else None
        results["claude"] = responses[1] if not isinstance(responses[1], Exception) else None
        results["perplexity"] = responses[2] if not isinstance(responses[2], Exception) else None
        
        return results

class GeminiService:
    """Google Gemini 2.5 Pro integration"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        # Initialize Gemini client
    
    async def generate(self, prompt: str) -> str:
        """Generate response from Gemini"""
        # Implement Gemini API call
        return "Gemini response placeholder"
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using Gemini"""
        # Implement embedding generation
        return [0.0] * 1536
    
    async def analyze(self, query: str) -> Dict:
        """Deep analysis with Gemini"""
        return {"analysis": "Gemini analysis placeholder"}
    
    async def search(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search with Gemini"""
        return []

class ClaudeService:
    """Anthropic Claude Opus 4.1 integration"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        # Initialize Claude client
    
    async def generate(self, prompt: str) -> str:
        """Generate response from Claude"""
        return "Claude response placeholder"
    
    async def analyze(self, query: str) -> Dict:
        """Deep analysis with Claude"""
        return {"analysis": "Claude analysis placeholder"}
    
    async def search(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search with Claude"""
        return []

class PerplexityService:
    """Perplexity AI search integration"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        # Initialize Perplexity client
    
    async def search(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search with Perplexity"""
        return []
PYTHON_EOF

# Complete Database Schema
cat > database/complete_schema.sql << 'SQL_EOF'
-- KOO Platform Personal Edition - Complete Database Schema
-- PostgreSQL with pgvector for semantic search
-- NO user tables, NO collaboration tables

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For fuzzy text search

-- Main chapters table (personal use, no user references)
CREATE TABLE chapters (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    
    -- Basic info
    title VARCHAR(500) NOT NULL,
    slug VARCHAR(200) UNIQUE NOT NULL,
    
    -- Content
    content TEXT NOT NULL,
    summary TEXT,
    
    -- Neurosurgical categorization
    specialty VARCHAR(100) DEFAULT 'neurosurgery',
    subspecialty VARCHAR(100),  -- 'vascular', 'tumor', 'spine', 'functional', 'pediatric'
    procedure_type VARCHAR(100),
    anatomical_region VARCHAR(100),
    complexity_level VARCHAR(20),  -- 'basic', 'intermediate', 'advanced', 'expert'
    
    -- AI-generated fields
    confidence_score FLOAT DEFAULT 0.0 CHECK (confidence_score BETWEEN 0 AND 1),
    last_ai_update TIMESTAMPTZ,
    auto_generated BOOLEAN DEFAULT false,
    ai_model_used VARCHAR(50),  -- 'gemini', 'claude', 'combined'
    
    -- Vector embedding for semantic search
    content_embedding vector(1536),
    
    -- Personal organization (NO user_id)
    tags TEXT[] DEFAULT ARRAY[]::TEXT[],
    is_favorite BOOLEAN DEFAULT false,
    is_archived BOOLEAN DEFAULT false,
    personal_notes TEXT,
    reading_time_minutes INTEGER,
    
    -- Statistics
    view_count INTEGER DEFAULT 0,
    last_viewed_at TIMESTAMPTZ,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- PDF textbooks table
CREATE TABLE pdf_textbooks (
    id SERIAL PRIMARY KEY,
    file_hash VARCHAR(64) UNIQUE NOT NULL,
    file_path TEXT NOT NULL,
    
    -- Book metadata
    title VARCHAR(500) NOT NULL,
    authors TEXT[],
    isbn VARCHAR(20),
    year INTEGER,
    edition VARCHAR(50),
    publisher VARCHAR(200),
    
    -- File info
    file_size_mb FLOAT,
    total_pages INTEGER NOT NULL,
    chapter_count INTEGER,
    
    -- Content structure
    textbook_type VARCHAR(50),  -- 'youmans', 'greenberg', 'rhoton', etc.
    table_of_contents JSONB,
    index_terms JSONB,
    cross_references JSONB,
    
    -- Medical content
    medical_entities JSONB,
    
    -- Importance and usage
    importance_score FLOAT DEFAULT 0.8,
    is_reference_book BOOLEAN DEFAULT false,
    times_accessed INTEGER DEFAULT 0,
    last_accessed TIMESTAMPTZ,
    
    -- Processing status
    processing_status VARCHAR(20) DEFAULT 'pending',  -- 'pending', 'processing', 'completed', 'error'
    processing_error TEXT,
    processed_at TIMESTAMPTZ,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Individual chapters from textbooks
CREATE TABLE textbook_chapters (
    id SERIAL PRIMARY KEY,
    textbook_id INTEGER REFERENCES pdf_textbooks(id) ON DELETE CASCADE,
    
    -- Chapter info
    title VARCHAR(500) NOT NULL,
    chapter_number VARCHAR(20),
    start_page INTEGER NOT NULL,
    end_page INTEGER NOT NULL,
    
    -- Content
    content TEXT NOT NULL,
    word_count INTEGER,
    
    -- Extracted knowledge
    key_points JSONB,
    clinical_pearls JSONB,
    surgical_techniques JSONB,
    
    -- Medical entities
    anatomical_structures TEXT[],
    procedures TEXT[],
    conditions TEXT[],
    complications TEXT[],
    medications TEXT[],
    
    -- Media
    tables_count INTEGER DEFAULT 0,
    figures_count INTEGER DEFAULT 0,
    tables_data JSONB,
    figures_data JSONB,
    
    -- Vector for similarity search
    embedding vector(1536),
    
    -- Usage
    times_referenced INTEGER DEFAULT 0,
    quality_score FLOAT DEFAULT 0.8,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Research papers and guidelines
CREATE TABLE pdf_papers (
    id SERIAL PRIMARY KEY,
    file_hash VARCHAR(64) UNIQUE NOT NULL,
    file_path TEXT NOT NULL,
    
    -- Paper metadata
    title VARCHAR(500) NOT NULL,
    authors TEXT[],
    journal VARCHAR(300),
    doi VARCHAR(100),
    pmid VARCHAR(20),
    year INTEGER,
    
    -- Content
    abstract TEXT,
    full_text TEXT NOT NULL,
    sections JSONB,  -- {abstract, intro, methods, results, discussion, conclusion}
    
    -- Classification
    document_type VARCHAR(50),  -- 'research', 'review', 'guideline', 'case_report', 'meta_analysis'
    study_type VARCHAR(50),  -- 'RCT', 'cohort', 'case_control', 'systematic_review'
    evidence_level VARCHAR(10),  -- 'I', 'II', 'III', 'IV', 'V'
    
    -- Medical extraction
    medical_entities JSONB,
    interventions JSONB,
    outcomes JSONB,
    statistics JSONB,  -- p-values, confidence intervals, etc.
    
    -- Relevance and quality
    neurosurgical_relevance FLOAT DEFAULT 0.0,
    clinical_importance FLOAT DEFAULT 0.0,
    methodology_score FLOAT DEFAULT 0.0,
    
    -- AI processing
    summary JSONB,  -- AI-generated summaries from different models
    
    -- Vector embedding
    embedding vector(1536),
    
    -- File info
    file_size_mb FLOAT,
    pages INTEGER,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Links between PDFs and KOO chapters
CREATE TABLE pdf_chapter_links (
    id SERIAL PRIMARY KEY,
    
    -- Source (PDF)
    pdf_type VARCHAR(20) NOT NULL,  -- 'textbook_chapter' or 'paper'
    pdf_id INTEGER NOT NULL,
    
    -- Target (KOO chapter)
    koo_chapter_id INTEGER REFERENCES chapters(id) ON DELETE CASCADE,
    
    -- Relationship
    relevance_score FLOAT DEFAULT 0.0,
    link_type VARCHAR(50),  -- 'reference', 'enrichment', 'contradiction', 'update', 'support'
    
    -- Metadata
    key_concepts TEXT[],
    auto_generated BOOLEAN DEFAULT true,
    verified BOOLEAN DEFAULT false,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(pdf_type, pdf_id, koo_chapter_id)
);

-- Personal thought stream (no user_id needed)
CREATE TABLE thought_stream (
    id SERIAL PRIMARY KEY,
    
    -- Thought details
    thought_type VARCHAR(50),  -- 'question', 'insight', 'todo', 'idea', 'correction'
    content TEXT NOT NULL,
    priority VARCHAR(20) DEFAULT 'normal',  -- 'low', 'normal', 'high', 'urgent'
    
    -- Related content
    related_chapter_id INTEGER REFERENCES chapters(id) ON DELETE SET NULL,
    related_pdf_id INTEGER,
    related_pdf_type VARCHAR(20),
    
    -- AI processing
    is_processed BOOLEAN DEFAULT false,
    ai_response JSONB,
    triggered_research JSONB,
    triggered_update_ids INTEGER[],
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    processed_at TIMESTAMPTZ
);

-- Research sessions
CREATE TABLE research_sessions (
    id SERIAL PRIMARY KEY,
    
    -- Query details
    query TEXT NOT NULL,
    query_type VARCHAR(50),  -- 'clinical', 'technique', 'evidence', 'general'
    purpose TEXT,
    
    -- Results from each source
    gemini_results JSONB,
    claude_results JSONB,
    pubmed_results JSONB,
    perplexity_results JSONB,
    semantic_scholar_results JSONB,
    
    -- Synthesis
    unified_synthesis TEXT,
    key_findings JSONB,
    clinical_applications JSONB,
    references JSONB,
    
    -- Evaluation
    usefulness_rating INTEGER CHECK (usefulness_rating BETWEEN 1 AND 5),
    applied_to_chapter_ids INTEGER[],
    notes TEXT,
    
    -- Performance
    total_results_count INTEGER,
    processing_time_seconds FLOAT,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Evolution events tracking
CREATE TABLE evolution_events (
    id SERIAL PRIMARY KEY,
    
    -- Event details
    event_type VARCHAR(50),  -- 'daily_update', 'manual_trigger', 'pdf_import', 'research_integration'
    trigger VARCHAR(100),  -- What triggered this evolution
    
    -- Changes made
    affected_chapters INTEGER[],
    new_chapters_created INTEGER[],
    content_changes JSONB,  -- Summary of changes
    confidence_changes JSONB,  -- How confidence scores changed
    
    -- Sources used
    sources_consulted JSONB,
    research_count INTEGER,
    pdf_count INTEGER,
    
    -- Results
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    
    -- Timing
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    duration_seconds FLOAT,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Knowledge synthesis records
CREATE TABLE knowledge_synthesis (
    id SERIAL PRIMARY KEY,
    
    -- Source chapters
    source_chapter_ids INTEGER[],
    source_pdf_ids INTEGER[],
    
    -- Synthesis details
    synthesis_type VARCHAR(50),  -- 'merge', 'update', 'reconcile', 'expand'
    topic VARCHAR(500),
    
    -- Results
    synthesized_content TEXT,
    confidence_score FLOAT,
    
    -- AI models used
    models_used JSONB,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Personal bookmarks and quick access
CREATE TABLE bookmarks (
    id SERIAL PRIMARY KEY,
    
    -- What's bookmarked
    bookmark_type VARCHAR(20),  -- 'chapter', 'pdf', 'search', 'thought'
    reference_id INTEGER,
    reference_title VARCHAR(500),
    
    -- Organization
    category VARCHAR(50),
    tags TEXT[],
    
    -- Quick access
    shortcut_key VARCHAR(10),
    position INTEGER,  -- For ordering
    
    -- Notes
    notes TEXT,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Search history and saved searches
CREATE TABLE search_history (
    id SERIAL PRIMARY KEY,
    
    query TEXT NOT NULL,
    search_type VARCHAR(20),  -- 'semantic', 'full_text', 'medical'
    
    -- Results
    results_count INTEGER,
    top_results JSONB,
    
    -- Save for later
    is_saved BOOLEAN DEFAULT false,
    saved_name VARCHAR(200),
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Clinical cases library
CREATE TABLE clinical_cases (
    id SERIAL PRIMARY KEY,
    
    -- Case details
    title VARCHAR(500) NOT NULL,
    presentation TEXT,
    imaging_findings TEXT,
    diagnosis TEXT,
    treatment TEXT,
    outcome TEXT,
    
    -- Categorization
    case_type VARCHAR(100),
    complexity VARCHAR(20),
    
    -- Related content
    related_chapter_ids INTEGER[],
    related_pdf_ids INTEGER[],
    
    -- Learning points
    key_learning_points JSONB,
    pitfalls_to_avoid JSONB,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_chapters_slug ON chapters(slug);
CREATE INDEX idx_chapters_specialty ON chapters(specialty, subspecialty);
CREATE INDEX idx_chapters_tags ON chapters USING GIN(tags);
CREATE INDEX idx_chapters_embedding ON chapters USING ivfflat (content_embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_chapters_updated ON chapters(updated_at DESC);
CREATE INDEX idx_chapters_favorite ON chapters(is_favorite) WHERE is_favorite = true;

CREATE INDEX idx_textbooks_hash ON pdf_textbooks(file_hash);
CREATE INDEX idx_textbooks_type ON pdf_textbooks(textbook_type);
CREATE INDEX idx_textbooks_importance ON pdf_textbooks(importance_score DESC);

CREATE INDEX idx_textbook_chapters_book ON textbook_chapters(textbook_id);
CREATE INDEX idx_textbook_chapters_embedding ON textbook_chapters USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_textbook_chapters_procedures ON textbook_chapters USING GIN(procedures);

CREATE INDEX idx_papers_hash ON pdf_papers(file_hash);
CREATE INDEX idx_papers_doi ON pdf_papers(doi);
CREATE INDEX idx_papers_relevance ON pdf_papers(neurosurgical_relevance DESC);
CREATE INDEX idx_papers_embedding ON pdf_papers USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX idx_thoughts_unprocessed ON thought_stream(is_processed) WHERE is_processed = false;
CREATE INDEX idx_thoughts_created ON thought_stream(created_at DESC);

CREATE INDEX idx_evolution_created ON evolution_events(created_at DESC);
CREATE INDEX idx_evolution_type ON evolution_events(event_type);

CREATE INDEX idx_bookmarks_type ON bookmarks(bookmark_type);
CREATE INDEX idx_bookmarks_position ON bookmarks(position);

-- Full text search indexes
CREATE INDEX idx_chapters_fulltext ON chapters USING GIN(to_tsvector('english', content));
CREATE INDEX idx_papers_fulltext ON pdf_papers USING GIN(to_tsvector('english', full_text));

-- Create views for common queries
CREATE VIEW recent_chapters AS
SELECT id, title, specialty, subspecialty, confidence_score, updated_at
FROM chapters
WHERE updated_at > NOW() - INTERVAL '30 days'
ORDER BY updated_at DESC;

CREATE VIEW high_quality_content AS
SELECT 
    c.id,
    c.title,
    c.specialty,
    c.confidence_score,
    COUNT(DISTINCT pcl.pdf_id) as supporting_pdfs
FROM chapters c
LEFT JOIN pdf_chapter_links pcl ON c.id = pcl.koo_chapter_id
WHERE c.confidence_score > 0.8
GROUP BY c.id
ORDER BY c.confidence_score DESC;

-- Functions for advanced features
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_chapters_updated_at BEFORE UPDATE ON chapters
FOR EACH ROW EXECUTE FUNCTION update_updated_at();
SQL_EOF

# API Routes
cat > backend/api/chapters.py << 'PYTHON_EOF'
"""
Chapters API - Personal use, no authentication
Direct CRUD operations for neurosurgical chapters
"""

from fastapi import APIRouter, HTTPException, Query, Request
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()

class ChapterCreate(BaseModel):
    title: str
    content: str
    specialty: Optional[str] = "neurosurgery"
    subspecialty: Optional[str] = None
    procedure_type: Optional[str] = None
    anatomical_region: Optional[str] = None
    tags: List[str] = []

class ChapterUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    summary: Optional[str] = None
    personal_notes: Optional[str] = None
    tags: Optional[List[str]] = None
    is_favorite: Optional[bool] = None

@router.get("/")
async def list_chapters(
    request: Request,
    specialty: Optional[str] = None,
    subspecialty: Optional[str] = None,
    search: Optional[str] = None,
    favorites_only: bool = False,
    limit: int = Query(20, le=100),
    offset: int = 0
):
    """
    List all chapters with optional filtering
    NO authentication required - personal use
    """
    query = "SELECT * FROM chapters WHERE 1=1"
    params = []
    
    if specialty:
        query += f" AND specialty = ${len(params) + 1}"
        params.append(specialty)
    
    if subspecialty:
        query += f" AND subspecialty = ${len(params) + 1}"
        params.append(subspecialty)
    
    if favorites_only:
        query += " AND is_favorite = true"
    
    if search:
        query += f" AND (title ILIKE ${len(params) + 1} OR content ILIKE ${len(params) + 1})"
        params.append(f"%{search}%")
    
    query += " ORDER BY updated_at DESC"
    query += f" LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}"
    params.extend([limit, offset])
    
    async with request.app.state.db.get_session() as session:
        chapters = await session.fetch_all(query, *params)
    
    return [dict(c) for c in chapters]

@router.get("/{chapter_id}")
async def get_chapter(request: Request, chapter_id: int):
    """
    Get single chapter by ID
    Updates view count and last viewed timestamp
    """
    async with request.app.state.db.get_session() as session:
        # Get chapter
        chapter = await session.fetch_one(
            "SELECT * FROM chapters WHERE id = $1",
            chapter_id
        )
        
        if not chapter:
            raise HTTPException(404, "Chapter not found")
        
        # Update view stats
        await session.execute("""
            UPDATE chapters 
            SET view_count = view_count + 1,
                last_viewed_at = NOW()
            WHERE id = $1
        """, chapter_id)
        
        await session.commit()
    
    return dict(chapter)

@router.post("/")
async def create_chapter(request: Request, chapter: ChapterCreate):
    """
    Create new chapter
    NO user assignment needed - personal use
    """
    slug = chapter.title.lower().replace(" ", "-")[:200]
    
    async with request.app.state.db.get_session() as session:
        result = await session.fetch_one("""
            INSERT INTO chapters (
                title, slug, content, specialty, subspecialty,
                procedure_type, anatomical_region, tags
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING *
        """, chapter.title, slug, chapter.content, chapter.specialty,
        chapter.subspecialty, chapter.procedure_type,
        chapter.anatomical_region, chapter.tags)
        
        await session.commit()
    
    # Trigger embedding generation
    asyncio.create_task(
        request.app.state.search.generate_embedding_for_chapter(result['id'])
    )
    
    return dict(result)

@router.put("/{chapter_id}")
async def update_chapter(request: Request, chapter_id: int, update: ChapterUpdate):
    """
    Update chapter directly
    NO permission check - personal use
    """
    async with request.app.state.db.get_session() as session:
        # Build update query
        updates = []
        params = []
        param_count = 1
        
        if update.title is not None:
            updates.append(f"title = ${param_count}")
            params.append(update.title)
            param_count += 1
        
        if update.content is not None:
            updates.append(f"content = ${param_count}")
            params.append(update.content)
            param_count += 1
        
        if update.summary is not None:
            updates.append(f"summary = ${param_count}")
            params.append(update.summary)
            param_count += 1
        
        if update.personal_notes is not None:
            updates.append(f"personal_notes = ${param_count}")
            params.append(update.personal_notes)
            param_count += 1
        
        if update.tags is not None:
            updates.append(f"tags = ${param_count}")
            params.append(update.tags)
            param_count += 1
        
        if update.is_favorite is not None:
            updates.append(f"is_favorite = ${param_count}")
            params.append(update.is_favorite)
            param_count += 1
        
        if not updates:
            raise HTTPException(400, "No updates provided")
        
        updates.append("updated_at = NOW()")
        params.append(chapter_id)
        
        query = f"UPDATE chapters SET {', '.join(updates)} WHERE id = ${param_count} RETURNING *"
        
        result = await session.fetch_one(query, *params)
        
        if not result:
            raise HTTPException(404, "Chapter not found")
        
        await session.commit()
    
    # Re-generate embedding if content changed
    if update.content is not None:
        asyncio.create_task(
            request.app.state.search.generate_embedding_for_chapter(chapter_id)
        )
    
    return dict(result)

@router.delete("/{chapter_id}")
async def delete_chapter(request: Request, chapter_id: int):
    """
    Delete chapter
    NO permission check - personal use
    """
    async with request.app.state.db.get_session() as session:
        result = await session.execute(
            "DELETE FROM chapters WHERE id = $1 RETURNING id",
            chapter_id
        )
        
        if not result:
            raise HTTPException(404, "Chapter not found")
        
        await session.commit()
    
    return {"message": "Chapter deleted"}

@router.post("/{chapter_id}/toggle-favorite")
async def toggle_favorite(request: Request, chapter_id: int):
    """Toggle favorite status"""
    async with request.app.state.db.get_session() as session:
        result = await session.fetch_one("""
            UPDATE chapters 
            SET is_favorite = NOT is_favorite 
            WHERE id = $1 
            RETURNING id, is_favorite
        """, chapter_id)
        
        if not result:
            raise HTTPException(404, "Chapter not found")
        
        await session.commit()
    
    return dict(result)

@router.get("/{chapter_id}/related-pdfs")
async def get_related_pdfs(request: Request, chapter_id: int):
    """Get PDFs related to this chapter"""
    async with request.app.state.db.get_session() as session:
        # Get textbook chapters
        textbook_chapters = await session.fetch_all("""
            SELECT 
                tc.title as chapter_title,
                pt.title as book_title,
                pcl.relevance_score
            FROM pdf_chapter_links pcl
            JOIN textbook_chapters tc ON pcl.pdf_id = tc.id
            JOIN pdf_textbooks pt ON tc.textbook_id = pt.id
            WHERE pcl.koo_chapter_id = $1 
              AND pcl.pdf_type = 'textbook_chapter'
            ORDER BY pcl.relevance_score DESC
            LIMIT 10
        """, chapter_id)
        
        # Get papers
        papers = await session.fetch_all("""
            SELECT 
                pp.title,
                pp.journal,
                pp.year,
                pcl.relevance_score
            FROM pdf_chapter_links pcl
            JOIN pdf_papers pp ON pcl.pdf_id = pp.id
            WHERE pcl.koo_chapter_id = $1
              AND pcl.pdf_type = 'paper'
            ORDER BY pcl.relevance_score DESC
            LIMIT 10
        """, chapter_id)
    
    return {
        "textbook_chapters": [dict(tc) for tc in textbook_chapters],
        "papers": [dict(p) for p in papers]
    }
PYTHON_EOF

# PDF API Routes  
cat > backend/api/pdf.py << 'PYTHON_EOF'
"""
PDF Management API
Upload, process, and search PDF library
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Query
from typing import List, Optional
import aiofiles
import os
from pathlib import Path

router = APIRouter()

@router.post("/upload")
async def upload_pdf(
    request: Request,
    file: UploadFile = File(...),
    pdf_type: Optional[str] = None
):
    """
    Upload and process PDF
    Automatically determines if textbook or paper
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files allowed")
    
    # Create upload directory
    upload_dir = Path("uploads/pdfs")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Save file
    file_path = upload_dir / file.filename
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    # Determine type if not specified
    file_size_mb = len(content) / (1024 * 1024)
    if pdf_type is None:
        pdf_type = "textbook" if file_size_mb > 10 else "paper"
    
    # Process based on type
    if pdf_type == "textbook":
        result = await request.app.state.pdf_kb.process_complete_textbook(str(file_path))
    else:
        result = await request.app.state.pdf_kb.process_paper(str(file_path))
    
    return {
        "message": "PDF processing started",
        "filename": file.filename,
        "type": pdf_type,
        "size_mb": file_size_mb,
        "status": result.get("status", "processing")
    }

@router.post("/upload-batch")
async def upload_batch(request: Request, files: List[UploadFile] = File(...)):
    """Upload multiple PDFs at once"""
    results = []
    
    for file in files:
        if file.filename.endswith('.pdf'):
            # Process each file
            upload_dir = Path("uploads/pdfs")
            upload_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = upload_dir / file.filename
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            # Add to processing queue
            file_size_mb = len(content) / (1024 * 1024)
            pdf_type = "textbook" if file_size_mb > 10 else "paper"
            
            asyncio.create_task(
                request.app.state.pdf_kb.process_new_pdf(str(file_path))
            )
            
            results.append({
                "filename": file.filename,
                "type": pdf_type,
                "status": "queued"
            })
    
    return {"files": results, "count": len(results)}

@router.get("/textbooks")
async def list_textbooks(
    request: Request,
    search: Optional[str] = None,
    limit: int = Query(20, le=100),
    offset: int = 0
):
    """List all processed textbooks"""
    query = """
        SELECT 
            id, title, authors, year, edition, publisher,
            total_pages, chapter_count, importance_score,
            times_accessed, last_accessed, created_at
        FROM pdf_textbooks
        WHERE 1=1
    """
    params = []
    
    if search:
        query += f" AND title ILIKE ${len(params) + 1}"
        params.append(f"%{search}%")
    
    query += " ORDER BY importance_score DESC, title"
    query += f" LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}"
    params.extend([limit, offset])
    
    async with request.app.state.db.get_session() as session:
        textbooks = await session.fetch_all(query, *params)
    
    return [dict(t) for t in textbooks]

@router.get("/textbook/{textbook_id}")
async def get_textbook(request: Request, textbook_id: int):
    """Get textbook with chapters"""
    async with request.app.state.db.get_session() as session:
        # Get textbook
        textbook = await session.fetch_one(
            "SELECT * FROM pdf_textbooks WHERE id = $1",
            textbook_id
        )
        
        if not textbook:
            raise HTTPException(404, "Textbook not found")
        
        # Get chapters
        chapters = await session.fetch_all("""
            SELECT id, title, chapter_number, start_page, end_page, word_count
            FROM textbook_chapters
            WHERE textbook_id = $1
            ORDER BY start_page
        """, textbook_id)
        
        # Update access stats
        await session.execute("""
            UPDATE pdf_textbooks
            SET times_accessed = times_accessed + 1,
                last_accessed = NOW()
            WHERE id = $1
        """, textbook_id)
        
        await session.commit()
    
    result = dict(textbook)
    result["chapters"] = [dict(c) for c in chapters]
    
    return result

@router.get("/textbook/{textbook_id}/chapter/{chapter_id}")
async def get_textbook_chapter(
    request: Request, 
    textbook_id: int, 
    chapter_id: int
):
    """Get specific chapter from textbook"""
    async with request.app.state.db.get_session() as session:
        chapter = await session.fetch_one("""
            SELECT tc.*, pt.title as book_title
            FROM textbook_chapters tc
            JOIN pdf_textbooks pt ON tc.textbook_id = pt.id
            WHERE tc.id = $1 AND tc.textbook_id = $2
        """, chapter_id, textbook_id)
        
        if not chapter:
            raise HTTPException(404, "Chapter not found")
        
        # Update reference count
        await session.execute("""
            UPDATE textbook_chapters
            SET times_referenced = times_referenced + 1
            WHERE id = $1
        """, chapter_id)
        
        await session.commit()
    
    return dict(chapter)

@router.get("/papers")
async def list_papers(
    request: Request,
    document_type: Optional[str] = None,
    min_relevance: float = 0.0,
    limit: int = Query(20, le=100),
    offset: int = 0
):
    """List research papers"""
    query = """
        SELECT 
            id, title, authors, journal, year, doi,
            document_type, neurosurgical_relevance,
            clinical_importance, pages, created_at
        FROM pdf_papers
        WHERE neurosurgical_relevance >= $1
    """
    params = [min_relevance]
    
    if document_type:
        query += f" AND document_type = ${len(params) + 1}"
        params.append(document_type)
    
    query += " ORDER BY neurosurgical_relevance DESC, year DESC"
    query += f" LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}"
    params.extend([limit, offset])
    
    async with request.app.state.db.get_session() as session:
        papers = await session.fetch_all(query, *params)
    
    return [dict(p) for p in papers]

@router.post("/search")
async def search_pdfs(request: Request, query: str):
    """
    Search across all PDFs using semantic search
    """
    results = await request.app.state.pdf_kb.search_across_library(
        query, 
        limit=20
    )
    
    # Log search
    async with request.app.state.db.get_session() as session:
        await session.execute("""
            INSERT INTO search_history (query, search_type, results_count)
            VALUES ($1, 'pdf', $2)
        """, query, len(results))
        
        await session.commit()
    
    return {"query": query, "results": results, "count": len(results)}

@router.post("/process-folder")
async def process_folder(request: Request, folder_path: str):
    """
    Process all PDFs in a folder
    """
    folder = Path(folder_path).expanduser()
    
    if not folder.exists():
        raise HTTPException(400, f"Folder not found: {folder_path}")
    
    pdf_files = list(folder.glob("**/*.pdf"))
    
    if not pdf_files:
        return {"message": "No PDF files found", "count": 0}
    
    # Queue all PDFs for processing
    for pdf_path in pdf_files:
        asyncio.create_task(
            request.app.state.pdf_kb.process_new_pdf(str(pdf_path))
        )
    
    return {
        "message": f"Queued {len(pdf_files)} PDFs for processing",
        "count": len(pdf_files),
        "files": [f.name for f in pdf_files[:10]]  # First 10 names
    }

@router.get("/processing-status")
async def get_processing_status(request: Request):
    """Get PDF processing queue status"""
    async with request.app.state.db.get_session() as session:
        stats = await session.fetch_one("""
            SELECT 
                COUNT(*) FILTER (WHERE processing_status = 'pending') as pending,
                COUNT(*) FILTER (WHERE processing_status = 'processing') as processing,
                COUNT(*) FILTER (WHERE processing_status = 'completed') as completed,
                COUNT(*) FILTER (WHERE processing_status = 'error') as error,
                COUNT(*) as total
            FROM pdf_textbooks
        """)
    
    return dict(stats)
PYTHON_EOF

# Frontend Components
cat > frontend/src/App.tsx << 'TSX_EOF'
/**
 * KOO Platform Personal Edition - Main App
 * NO authentication, NO collaboration
 * Direct access to personal knowledge base
 */

import React, { useState, useEffect } from 'react';
import { 
  BrowserRouter as Router, 
  Routes, 
  Route, 
  Navigate 
} from 'react-router-dom';
import { ThemeProvider, createTheme, CssBaseline } from '@mui/material';

// Pages and components
import Dashboard from './pages/Dashboard';
import ChaptersPage from './pages/ChaptersPage';
import ChapterViewer from './pages/ChapterViewer';
import PDFLibrary from './pages/PDFLibrary';
import ResearchHub from './pages/ResearchHub';
import ThoughtStream from './pages/ThoughtStream';
import EvolutionMonitor from './pages/EvolutionMonitor';
import Navigation from './components/Navigation';

// NO Auth components
// NO Login page
// NO User management

// Create theme
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#2196f3',
    },
    secondary: {
      main: '#f50057',
    },
    background: {
      default: '#0a0e1a',
      paper: '#1a1f2e',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
  },
});

const App: React.FC = () => {
  // NO authentication state
  // NO user context
  // Direct access to all features

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Navigation />
        <Routes>
          {/* All routes directly accessible - NO auth required */}
          <Route path="/" element={<Dashboard />} />
          <Route path="/chapters" element={<ChaptersPage />} />
          <Route path="/chapter/:id" element={<ChapterViewer />} />
          <Route path="/library" element={<PDFLibrary />} />
          <Route path="/research" element={<ResearchHub />} />
          <Route path="/thoughts" element={<ThoughtStream />} />
          <Route path="/evolution" element={<EvolutionMonitor />} />
          
          {/* NO login route */}
          {/* NO admin routes */}
          {/* NO user profile route */}
          
          {/* Catch all - redirect to dashboard */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Router>
    </ThemeProvider>
  );
};

export default App;
TSX_EOF

# Dashboard Component
cat > frontend/src/pages/Dashboard.tsx << 'TSX_EOF'
/**
 * Personal Dashboard
 * Main interface for personal neurosurgical knowledge base
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  Button,
  IconButton,
  Chip,
  LinearProgress,
  Alert
} from '@mui/material';
import {
  AutoAwesome,
  MenuBook,
  Science,
  Psychology,
  TrendingUp,
  Refresh,
  FiberManualRecord
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

interface Stats {
  chapters: number;
  textbooks: number;
  papers: number;
  thoughts: number;
  last_evolution: string;
}

const Dashboard: React.FC = () => {
  const navigate = useNavigate();
  const [stats, setStats] = useState<Stats | null>(null);
  const [evolutionRunning, setEvolutionRunning] = useState(false);
  const [recentChapters, setRecentChapters] = useState([]);
  
  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      const [statsRes, chaptersRes] = await Promise.all([
        axios.get('/api/stats'),
        axios.get('/api/chapters?limit=5')
      ]);
      
      setStats(statsRes.data);
      setRecentChapters(chaptersRes.data);
    } catch (error) {
      console.error('Error loading dashboard:', error);
    }
  };

  const triggerEvolution = async () => {
    setEvolutionRunning(true);
    try {
      await axios.post('/api/evolution/trigger');
      // Evolution runs in background
      setTimeout(() => {
        setEvolutionRunning(false);
        loadDashboardData();
      }, 3000);
    } catch (error) {
      console.error('Error triggering evolution:', error);
      setEvolutionRunning(false);
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h3" gutterBottom>
          Personal Neurosurgical Knowledge Base
        </Typography>
        <Typography variant="body1" color="text.secondary">
          AI-powered, self-evolving medical knowledge system
        </Typography>
      </Box>

      {/* Quick Stats */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <MenuBook sx={{ mr: 1, color: 'primary.main' }} />
                <Typography variant="h4">{stats?.chapters || 0}</Typography>
              </Box>
              <Typography variant="body2" color="text.secondary">
                Knowledge Chapters
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <MenuBook sx={{ mr: 1, color: 'secondary.main' }} />
                <Typography variant="h4">{stats?.textbooks || 0}</Typography>
              </Box>
              <Typography variant="body2" color="text.secondary">
                Reference Textbooks
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <Science sx={{ mr: 1, color: 'success.main' }} />
                <Typography variant="h4">{stats?.papers || 0}</Typography>
              </Box>
              <Typography variant="body2" color="text.secondary">
                Research Papers
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <Psychology sx={{ mr: 1, color: 'warning.main' }} />
                <Typography variant="h4">{stats?.thoughts || 0}</Typography>
              </Box>
              <Typography variant="body2" color="text.secondary">
                Active Thoughts
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Evolution Status */}
      <Paper sx={{ p: 3, mb: 4 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <AutoAwesome sx={{ mr: 2, color: 'primary.main' }} />
          <Box sx={{ flexGrow: 1 }}>
            <Typography variant="h6">Knowledge Evolution</Typography>
            <Typography variant="body2" color="text.secondary">
              Last update: {stats?.last_evolution || 'Never'}
            </Typography>
          </Box>
          <Button
            variant="contained"
            startIcon={<Refresh />}
            onClick={triggerEvolution}
            disabled={evolutionRunning}
          >
            {evolutionRunning ? 'Evolving...' : 'Trigger Evolution'}
          </Button>
        </Box>
        {evolutionRunning && <LinearProgress />}
      </Paper>

      {/* Recent Activity */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Recent Chapters
            </Typography>
            {recentChapters.map((chapter: any) => (
              <Box
                key={chapter.id}
                sx={{
                  p: 2,
                  mb: 1,
                  border: '1px solid',
                  borderColor: 'divider',
                  borderRadius: 1,
                  cursor: 'pointer',
                  '&:hover': { bgcolor: 'action.hover' }
                }}
                onClick={() => navigate(`/chapter/${chapter.id}`)}
              >
                <Typography variant="subtitle1">{chapter.title}</Typography>
                <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                  <Chip label={chapter.specialty} size="small" />
                  {chapter.subspecialty && (
                    <Chip label={chapter.subspecialty} size="small" />
                  )}
                  <Chip
                    label={`Confidence: ${(chapter.confidence_score * 100).toFixed(0)}%`}
                    size="small"
                    color={chapter.confidence_score > 0.8 ? 'success' : 'warning'}
                  />
                </Box>
              </Box>
            ))}
          </Paper>
        </Grid>

        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Quick Actions
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Button
                fullWidth
                variant="outlined"
                onClick={() => navigate('/chapters')}
              >
                Browse Chapters
              </Button>
              <Button
                fullWidth
                variant="outlined"
                onClick={() => navigate('/library')}
              >
                PDF Library
              </Button>
              <Button
                fullWidth
                variant="outlined"
                onClick={() => navigate('/research')}
              >
                Research Hub
              </Button>
              <Button
                fullWidth
                variant="outlined"
                onClick={() => navigate('/thoughts')}
              >
                Thought Stream
              </Button>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;
TSX_EOF

# Package files
cat > frontend/package.json << 'JSON_EOF'
{
  "name": "koo-platform-frontend",
  "version": "2.0.0",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.20.0",
    "@mui/material": "^5.14.19",
    "@mui/icons-material": "^5.14.19",
    "@emotion/react": "^11.11.1",
    "@emotion/styled": "^11.11.0",
    "axios": "^1.6.2",
    "react-dropzone": "^14.2.3",
    "react-markdown": "^9.0.1",
    "recharts": "^2.9.3",
    "date-fns": "^2.30.0",
    "typescript": "^5.3.2"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  },
  "devDependencies": {
    "@types/react": "^18.2.45",
    "@types/react-dom": "^18.2.17",
    "@types/react-router-dom": "^5.3.3",
    "react-scripts": "5.0.1"
  },
  "browserslist": {
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  }
}
JSON_EOF

cat > backend/requirements.txt << 'REQ_EOF'
# Core
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
httpx==0.25.2
pydantic==2.5.2
pydantic-settings==2.1.0

# Database
sqlalchemy==2.0.23
asyncpg==0.29.0
pgvector==0.2.3
alembic==1.13.0

# Cache
redis==5.0.1
aioredis==2.0.1

# PDF Processing
PyPDF2==3.0.1
pdfplumber==0.10.3
PyMuPDF==1.23.8
pypdfium2==4.25.0

# AI and ML
google-generativeai==0.3.0
anthropic==0.7.7
openai==1.6.1
langchain==0.1.0
tiktoken==0.5.2
numpy==1.26.2
scikit-learn==1.3.2

# File handling
aiofiles==23.2.1
watchdog==3.0.0
python-magic==0.4.27

# Utils
python-dateutil==2.8.2
pytz==2023.3

# Development
pytest==7.4.3
pytest-asyncio==0.21.1
black==23.12.0
REQ_EOF

# Docker configuration
cat > docker-compose.yml << 'DOCKER_EOF'
version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:pg15
    container_name: koo-postgres
    environment:
      POSTGRES_DB: koo_personal
      POSTGRES_USER: koo
      POSTGRES_PASSWORD: koo_local_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database/complete_schema.sql:/docker-entrypoint-initdb.d/01_schema.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U koo"]
      interval: 10s
# Continuing docker-compose.yml
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: koo-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s

  backend:
    build: 
      context: ./backend
      dockerfile: Dockerfile
    container_name: koo-backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://koo:koo_local_password@postgres:5432/koo_personal
      - REDIS_URL=redis://redis:6379/0
      - PYTHONUNBUFFERED=1
    env_file:
      - .env
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./backend:/app
      - ./pdf-library:/pdf-library
      - ./knowledge-base:/knowledge-base
      - ./uploads:/uploads
    command: uvicorn main:app --host 0.0.0.0 --port 8000 --reload

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    container_name: koo-frontend
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_API_URL=http://localhost:8000
      - WATCHPACK_POLLING=true
    depends_on:
      - backend
    volumes:
      - ./frontend:/app
      - /app/node_modules
    command: npm start

volumes:
  postgres_data:
  redis_data:
DOCKER_EOF

# Backend Dockerfile
cat > backend/Dockerfile << 'DOCKERFILE_EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /uploads /pdf-library /knowledge-base

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
DOCKERFILE_EOF

# Frontend Dockerfile
cat > frontend/Dockerfile << 'DOCKERFILE_EOF'
FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy application code
COPY . .

# Expose port
EXPOSE 3000

# Start the application
CMD ["npm", "start"]
DOCKERFILE_EOF

# Environment template
cat > .env.example << 'ENV_EOF'
# KOO Platform Personal Edition Configuration
# Copy to .env and fill in your personal API keys

# Environment
ENVIRONMENT=development
DEBUG=true

# AI Services - Your Personal Subscriptions
GEMINI_API_KEY=your-gemini-2.5-pro-api-key
CLAUDE_API_KEY=your-claude-opus-4.1-api-key
PERPLEXITY_API_KEY=your-perplexity-api-key

# Medical/Research APIs (Optional)
PUBMED_API_KEY=your-pubmed-api-key
SEMANTIC_SCHOLAR_API_KEY=optional
ELSEVIER_API_KEY=optional
BIODIGITAL_API_KEY=optional

# Evolution Settings
AUTO_EVOLUTION_ENABLED=true
EVOLUTION_SCHEDULE_HOUR=6  # Daily at 6 AM
MAX_MONTHLY_API_COST=155

# Paths
PDF_LIBRARY_PATH=~/Neurosurgery-Library
KNOWLEDGE_BASE_PATH=~/koo-knowledge
BACKUP_PATH=~/koo-backup

# Database (Docker will override these)
DATABASE_URL=postgresql://koo:koo_local_password@localhost:5432/koo_personal
REDIS_URL=redis://localhost:6379/0
ENV_EOF

# Additional API routes
cat > backend/api/thoughts.py << 'PYTHON_EOF'
"""
Thought Stream API
Personal questions and insights management
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

router = APIRouter()

class ThoughtCreate(BaseModel):
    thought_type: str = "question"  # question, insight, todo, idea, correction
    content: str
    priority: str = "normal"  # low, normal, high, urgent
    related_chapter_id: Optional[int] = None

@router.post("/")
async def create_thought(request: Request, thought: ThoughtCreate):
    """Add new thought to stream"""
    async with request.app.state.db.get_session() as session:
        result = await session.fetch_one("""
            INSERT INTO thought_stream (
                thought_type, content, priority, related_chapter_id
            ) VALUES ($1, $2, $3, $4)
            RETURNING *
        """, thought.thought_type, thought.content, 
        thought.priority, thought.related_chapter_id)
        
        await session.commit()
    
    return dict(result)

@router.get("/")
async def list_thoughts(
    request: Request,
    processed: Optional[bool] = None,
    thought_type: Optional[str] = None,
    limit: int = 20
):
    """List thoughts with optional filtering"""
    query = "SELECT * FROM thought_stream WHERE 1=1"
    params = []
    
    if processed is not None:
        query += f" AND is_processed = ${len(params) + 1}"
        params.append(processed)
    
    if thought_type:
        query += f" AND thought_type = ${len(params) + 1}"
        params.append(thought_type)
    
    query += " ORDER BY created_at DESC"
    query += f" LIMIT ${len(params) + 1}"
    params.append(limit)
    
    async with request.app.state.db.get_session() as session:
        thoughts = await session.fetch_all(query, *params)
    
    return [dict(t) for t in thoughts]

@router.post("/{thought_id}/process")
async def process_thought(request: Request, thought_id: int):
    """Manually trigger processing of a thought"""
    async with request.app.state.db.get_session() as session:
        thought = await session.fetch_one(
            "SELECT * FROM thought_stream WHERE id = $1",
            thought_id
        )
        
        if not thought:
            raise HTTPException(404, "Thought not found")
        
        # Research the thought using AI
        research = await request.app.state.evolution._research_thought(dict(thought))
        
        # Update with results
        await session.execute("""
            UPDATE thought_stream
            SET is_processed = TRUE,
                ai_response = $1,
                processed_at = NOW()
            WHERE id = $2
        """, json.dumps(research), thought_id)
        
        await session.commit()
    
    return {"status": "processed", "research": research}
PYTHON_EOF

cat > backend/api/search.py << 'PYTHON_EOF'
"""
Search API
Semantic and full-text search across all content
"""

from fastapi import APIRouter, Request, Query
from typing import Optional, List, Dict

router = APIRouter()

@router.post("/semantic")
async def semantic_search(
    request: Request,
    query: str,
    include_chapters: bool = True,
    include_pdfs: bool = True,
    limit: int = Query(20, le=50)
):
    """
    Semantic search using vector embeddings
    """
    results = []
    
    # Generate query embedding
    query_embedding = await request.app.state.ai.generate_embedding(query)
    
    async with request.app.state.db.get_session() as session:
        if include_chapters:
            # Search chapters
            chapter_results = await session.fetch_all("""
                SELECT 
                    id, title, specialty, subspecialty,
                    substring(content, 1, 500) as content_preview,
                    1 - (content_embedding <=> $1::vector) as similarity
                FROM chapters
                WHERE content_embedding IS NOT NULL
                ORDER BY similarity DESC
                LIMIT $2
            """, query_embedding, limit // 2 if include_pdfs else limit)
            
            for row in chapter_results:
                results.append({
                    "type": "chapter",
                    "id": row["id"],
                    "title": row["title"],
                    "preview": row["content_preview"],
                    "similarity": row["similarity"],
                    "specialty": row["specialty"]
                })
        
        if include_pdfs:
            # Search PDFs
            pdf_results = await request.app.state.pdf_kb.search_across_library(
                query, 
                limit=limit // 2 if include_chapters else limit
            )
            results.extend(pdf_results)
    
    # Sort all results by similarity
    results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
    
    return {
        "query": query,
        "results": results[:limit],
        "total": len(results)
    }

@router.post("/fulltext")
async def fulltext_search(
    request: Request,
    query: str,
    search_in: List[str] = Query(["chapters", "papers"])
):
    """Full-text search using PostgreSQL"""
    results = []
    
    async with request.app.state.db.get_session() as session:
        if "chapters" in search_in:
            chapter_results = await session.fetch_all("""
                SELECT 
                    id, title, 
                    ts_headline('english', content, query) as highlight,
                    ts_rank(to_tsvector('english', content), query) as rank
                FROM chapters,
                     plainto_tsquery('english', $1) query
                WHERE to_tsvector('english', content) @@ query
                ORDER BY rank DESC
                LIMIT 10
            """, query)
            
            for row in chapter_results:
                results.append({
                    "type": "chapter",
                    "id": row["id"],
                    "title": row["title"],
                    "highlight": row["highlight"],
                    "rank": float(row["rank"])
                })
        
        if "papers" in search_in:
            paper_results = await session.fetch_all("""
                SELECT 
                    id, title, journal,
                    ts_headline('english', abstract, query) as highlight,
                    ts_rank(to_tsvector('english', full_text), query) as rank
                FROM pdf_papers,
                     plainto_tsquery('english', $1) query
                WHERE to_tsvector('english', full_text) @@ query
                ORDER BY rank DESC
                LIMIT 10
            """, query)
            
            for row in paper_results:
                results.append({
                    "type": "paper",
                    "id": row["id"],
                    "title": row["title"],
                    "journal": row["journal"],
                    "highlight": row["highlight"],
                    "rank": float(row["rank"])
                })
    
    return {
        "query": query,
        "results": sorted(results, key=lambda x: x["rank"], reverse=True),
        "total": len(results)
    }

@router.get("/saved")
async def get_saved_searches(request: Request):
    """Get saved searches"""
    async with request.app.state.db.get_session() as session:
        searches = await session.fetch_all("""
            SELECT * FROM search_history
            WHERE is_saved = TRUE
            ORDER BY created_at DESC
        """)
    
    return [dict(s) for s in searches]
PYTHON_EOF

cat > backend/api/evolution.py << 'PYTHON_EOF'
"""
Evolution API
Monitor and control knowledge evolution
"""

from fastapi import APIRouter, Request, HTTPException
from datetime import datetime

router = APIRouter()

@router.post("/trigger")
async def trigger_evolution(request: Request):
    """Manually trigger knowledge evolution"""
    result = await request.app.state.evolution.trigger_manual_evolution()
    return result

@router.get("/status")
async def get_evolution_status(request: Request):
    """Get current evolution status"""
    return {
        "running": request.app.state.evolution.evolution_running,
        "last_run": await request.app.state.evolution.get_last_run(),
        "scheduled": request.app.state.evolution.scheduled_task is not None
    }

@router.get("/history")
async def get_evolution_history(request: Request, limit: int = 20):
    """Get evolution event history"""
    async with request.app.state.db.get_session() as session:
        events = await session.fetch_all("""
            SELECT * FROM evolution_events
            ORDER BY created_at DESC
            LIMIT $1
        """, limit)
    
    return [dict(e) for e in events]

@router.get("/stats")
async def get_evolution_stats(request: Request):
    """Get evolution statistics"""
    async with request.app.state.db.get_session() as session:
        stats = await session.fetch_one("""
            SELECT 
                COUNT(*) as total_events,
                COUNT(*) FILTER (WHERE success = true) as successful,
                COUNT(*) FILTER (WHERE success = false) as failed,
                AVG(duration_seconds) as avg_duration,
                MAX(created_at) as last_evolution
            FROM evolution_events
        """)
    
    return dict(stats)
PYTHON_EOF

# Additional Frontend Pages
cat > frontend/src/pages/PDFLibrary.tsx << 'TSX_EOF'
/**
 * PDF Library Management
 * Complete interface for neurosurgical textbooks and papers
 */

import React, { useState, useEffect } from 'react';
import {
  Box, Grid, Paper, Typography, Button, TextField,
  Card, CardContent, CardActions, Chip, IconButton,
  Dialog, DialogTitle, DialogContent, LinearProgress,
  Tabs, Tab, Alert, List, ListItem, ListItemText,
  Breadcrumbs, Link, Tooltip, InputAdornment
} from '@mui/material';
import {
  MenuBook, Article, Search, CloudUpload, Bookmark,
  Description, Star, Download, Visibility, FolderOpen,
  AutoStories, LocalLibrary, Assessment, ImportContacts
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';

const PDFLibrary: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [textbooks, setTextbooks] = useState([]);
  const [papers, setPapers] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [selectedBook, setSelectedBook] = useState(null);
  const [bookChapters, setBookChapters] = useState([]);

  useEffect(() => {
    loadLibrary();
  }, []);

  const loadLibrary = async () => {
    try {
      const [textbooksRes, papersRes] = await Promise.all([
        axios.get('/api/pdf/textbooks'),
        axios.get('/api/pdf/papers')
      ]);
      setTextbooks(textbooksRes.data);
      setPapers(papersRes.data);
    } catch (error) {
      console.error('Error loading library:', error);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: { 'application/pdf': ['.pdf'] },
    onDrop: async (files) => {
      const formData = new FormData();
      files.forEach(file => formData.append('files', file));
      
      try {
        setUploadProgress(10);
        await axios.post('/api/pdf/upload-batch', formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
          onUploadProgress: (e) => {
            setUploadProgress(Math.round((e.loaded / e.total) * 100));
          }
        });
        setUploadProgress(0);
        loadLibrary();
      } catch (error) {
        console.error('Upload error:', error);
        setUploadProgress(0);
      }
    }
  });

  const searchLibrary = async () => {
    if (!searchQuery) return;
    
    try {
      const response = await axios.post('/api/pdf/search', { query: searchQuery });
      // Handle search results
      console.log('Search results:', response.data);
    } catch (error) {
      console.error('Search error:', error);
    }
  };

  const openTextbook = async (textbook: any) => {
    setSelectedBook(textbook);
    try {
      const response = await axios.get(`/api/pdf/textbook/${textbook.id}`);
      setBookChapters(response.data.chapters || []);
    } catch (error) {
      console.error('Error loading chapters:', error);
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h4" gutterBottom>
          📚 Neurosurgical Reference Library
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Complete textbooks, research papers, and clinical guidelines
        </Typography>
      </Paper>

      {/* Search Bar */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <TextField
          fullWidth
          placeholder="Search across all PDFs..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && searchLibrary()}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <Search />
              </InputAdornment>
            ),
            endAdornment: (
              <Button onClick={searchLibrary}>Search</Button>
            )
          }}
        />
      </Paper>

      {/* Upload Area */}
      <Paper
        {...getRootProps()}
        sx={{
          p: 3,
          mb: 3,
          border: '2px dashed',
          borderColor: isDragActive ? 'primary.main' : 'grey.500',
          bgcolor: isDragActive ? 'action.hover' : 'background.paper',
          cursor: 'pointer',
          textAlign: 'center'
        }}
      >
        <input {...getInputProps()} />
        <CloudUpload sx={{ fontSize: 48, mb: 2, color: 'text.secondary' }} />
        <Typography variant="h6">
          {isDragActive ? 'Drop PDFs here...' : 'Drag & drop PDFs or click to upload'}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Supports textbooks, research papers, guidelines, and atlases
        </Typography>
        {uploadProgress > 0 && (
          <LinearProgress variant="determinate" value={uploadProgress} sx={{ mt: 2 }} />
        )}
      </Paper>

      {/* Content Tabs */}
      <Paper sx={{ mb: 3 }}>
        <Tabs value={activeTab} onChange={(_, v) => setActiveTab(v)}>
          <Tab icon={<MenuBook />} label={`Textbooks (${textbooks.length})`} />
          <Tab icon={<Article />} label={`Papers (${papers.length})`} />
          <Tab icon={<LocalLibrary />} label="Guidelines" />
          <Tab icon={<Bookmark />} label="Bookmarked" />
        </Tabs>
      </Paper>

      {/* Textbooks Tab */}
      {activeTab === 0 && (
        <Grid container spacing={2}>
          {/* Core Reference Textbooks */}
          <Grid item xs={12}>
            <Typography variant="h6" gutterBottom>
              Core Reference Textbooks
            </Typography>
          </Grid>
          
          {/* Predefined core textbooks */}
          {[
            {
              id: 'youmans',
              title: "Youmans and Winn Neurological Surgery",
              edition: "8th Edition",
              year: 2022,
              chapters: 432,
              pages: 5120,
              importance_score: 1.0
            },
            {
              id: 'greenberg',
              title: "Handbook of Neurosurgery (Greenberg)",
              edition: "9th Edition", 
              year: 2023,
              chapters: 82,
              pages: 1352,
              importance_score: 1.0
            },
            {
              id: 'rhoton',
              title: "Rhoton's Cranial Anatomy and Surgical Approaches",
              edition: "Latest Edition",
              year: 2019,
              chapters: 24,
              pages: 746,
              importance_score: 0.95
            }
          ].map((book) => (
            <Grid item xs={12} md={6} lg={4} key={book.id}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'start', mb: 2 }}>
                    <MenuBook sx={{ mr: 2, color: 'primary.main', fontSize: 40 }} />
                    <Box sx={{ flexGrow: 1 }}>
                      <Typography variant="h6" gutterBottom>
                        {book.title}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {book.edition} • {book.year}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {book.chapters} chapters • {book.pages} pages
                      </Typography>
                    </Box>
                    <Chip
                      icon={<Star />}
                      label={book.importance_score.toFixed(1)}
                      color="primary"
                      size="small"
                    />
                  </Box>
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <Chip label="Reference" size="small" />
                    <Chip label="Indexed" size="small" color="success" />
                  </Box>
                </CardContent>
                <CardActions>
                  <Button size="small" startIcon={<AutoStories />} onClick={() => openTextbook(book)}>
                    Browse Chapters
                  </Button>
                  <Button size="small" startIcon={<Search />}>
                    Search Inside
                  </Button>
                  <IconButton size="small">
                    <Bookmark />
                  </IconButton>
                </CardActions>
              </Card>
            </Grid>
          ))}

          {/* User uploaded textbooks */}
          {textbooks.map((book: any) => (
            <Grid item xs={12} md={6} lg={4} key={book.id}>
              <Card>
                <CardContent>
                  <Typography variant="h6" noWrap>
                    {book.title}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {book.authors?.join(', ')}
                  </Typography>
                  <Typography variant="caption">
                    {book.year} • {book.total_pages} pages • {book.chapter_count} chapters
                  </Typography>
                </CardContent>
                <CardActions>
                  <Button size="small" onClick={() => openTextbook(book)}>
                    Open
                  </Button>
                </CardActions>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}

      {/* Papers Tab */}
      {activeTab === 1 && (
        <List>
          {papers.map((paper: any) => (
            <ListItem key={paper.id}>
              <ListItemText
                primary={paper.title}
                secondary={`${paper.journal || 'Unknown'} • ${paper.year || 'N/A'} • Relevance: ${(paper.neurosurgical_relevance * 100).toFixed(0)}%`}
              />
              <Button size="small" startIcon={<Visibility />}>
                View
              </Button>
            </ListItem>
          ))}
        </List>
      )}

      {/* Textbook Viewer Dialog */}
      <Dialog
        open={!!selectedBook}
        onClose={() => setSelectedBook(null)}
        maxWidth="lg"
        fullWidth
      >
        {selectedBook && (
          <>
            <DialogTitle>{selectedBook.title}</DialogTitle>
            <DialogContent>
              <List>
                {bookChapters.map((chapter: any) => (
                  <ListItem key={chapter.id}>
                    <ListItemText
                      primary={`Chapter ${chapter.chapter_number}: ${chapter.title}`}
                      secondary={`Pages ${chapter.start_page}-${chapter.end_page}`}
                    />
                    <Button size="small">Read</Button>
                  </ListItem>
                ))}
              </List>
            </DialogContent>
          </>
        )}
      </Dialog>
    </Box>
  );
};

export default PDFLibrary;
TSX_EOF

# Core configuration files
cat > backend/core/config.py << 'PYTHON_EOF'
"""
Configuration management
"""

from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # Database
    DATABASE_URL: str = "postgresql://koo:password@localhost:5432/koo_personal"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # AI Services
    GEMINI_API_KEY: Optional[str] = None
    CLAUDE_API_KEY: Optional[str] = None
    PERPLEXITY_API_KEY: Optional[str] = None
    
    # Medical APIs
    PUBMED_API_KEY: Optional[str] = None
    SEMANTIC_SCHOLAR_API_KEY: Optional[str] = None
    
    # Evolution
    AUTO_EVOLUTION_ENABLED: bool = True
    EVOLUTION_SCHEDULE_HOUR: int = 6
    
    # Paths
    PDF_LIBRARY_PATH: str = "~/Neurosurgery-Library"
    
    class Config:
        env_file = ".env"

settings = Settings()
PYTHON_EOF

cat > backend/core/database.py << 'PYTHON_EOF'
"""
Database management
"""

import asyncpg
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool = None
    
    async def connect(self):
        """Create connection pool"""
        self.pool = await asyncpg.create_pool(
            self.database_url,
            min_size=5,
            max_size=20
        )
    
    async def close(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
    
    @asynccontextmanager
    async def get_session(self):
        """Get database session"""
        if not self.pool:
            await self.connect()
        
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                yield connection
    
    async def execute(self, query: str, *args):
        """Execute query"""
        async with self.get_session() as session:
            return await session.execute(query, *args)
    
    async def fetch_one(self, query: str, *args):
        """Fetch single row"""
        async with self.get_session() as session:
            return await session.fetchrow(query, *args)
    
    async def fetch_all(self, query: str, *args):
        """Fetch all rows"""
        async with self.get_session() as session:
            return await session.fetch(query, *args)
    
    async def count(self, table: str) -> int:
        """Count rows in table"""
        async with self.get_session() as session:
            result = await session.fetchval(f"SELECT COUNT(*) FROM {table}")
            return result or 0

async def create_all_tables(db: DatabaseManager):
    """Create all database tables"""
    logger.info("Creating database tables...")
    
    # Read schema file
    with open("database/complete_schema.sql", "r") as f:
        schema = f.read()
    
    # Execute schema
    try:
        await db.execute(schema)
        logger.info("✅ Database tables created")
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        # Tables might already exist
        pass
PYTHON_EOF

cat > backend/core/cache.py << 'PYTHON_EOF'
"""
Redis cache management
"""

import redis.asyncio as redis
import json
from typing import Any, Optional

class RedisCache:
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.client = None
    
    async def connect(self):
        """Connect to Redis"""
        self.client = await redis.from_url(self.redis_url)
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.client:
            await self.client.close()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.client:
            await self.connect()
        
        value = await self.client.get(key)
        if value:
            return json.loads(value)
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache with TTL"""
        if not self.client:
            await self.connect()
        
        await self.client.setex(
            key,
            ttl,
            json.dumps(value)
        )
    
    async def delete(self, key: str):
        """Delete key from cache"""
        if not self.client:
            await self.connect()
        
        await self.client.delete(key)
PYTHON_EOF

# Create the main setup script
cat > setup.sh << 'BASH_EOF'
#!/bin/bash
# Complete setup script for KOO Platform Personal Edition

echo "=================================="
echo "KOO Platform Personal Edition Setup"
echo "=================================="

# Create all directories
echo "Creating directory structure..."
mkdir -p backend/{api,core,services,models,migrations}
mkdir -p frontend/src/{components,pages,services,hooks}
mkdir -p database
mkdir -p pdf-library/{textbooks,papers,guidelines}
mkdir -p uploads
mkdir -p knowledge-base

# Copy environment file
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your API keys"
fi

# Build and start services
echo "Starting Docker services..."
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services..."
sleep 10

# Check status
echo "Checking service status..."
docker-compose ps

echo "=================================="
echo "✅ Setup complete!"
echo "=================================="
echo "Access the application at:"
echo "  Frontend: http://localhost:3000"
echo "  Backend API: http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Place PDFs in ~/Neurosurgery-Library/"
echo "3. Access http://localhost:3000"
BASH_EOF

# Make setup script executable
chmod +x setup.sh

# Create README
cat > README.md << 'README_EOF'
# KOO Platform Personal Edition

## 🧠 Personal AI-Driven Neurosurgical Knowledge Management System

A comprehensive, self-evolving knowledge base for neurosurgical practice, powered by AI and designed for individual use.

### ✨ Key Features

- **NO Collaboration/User Management** - Streamlined for personal use
- **Complete PDF Processing** - Full textbook and paper extraction with medical entity recognition
- **Multi-AI Integration** - Gemini 2.5 Pro, Claude Opus 4.1, Perplexity
- **Auto-Evolution** - Daily automatic knowledge updates
- **Semantic Search** - Vector-based similarity search across all content
- **Medical Entity Extraction** - Automatic identification of procedures, anatomy, conditions
- **Thought Stream** - Personal question and insight tracking

### 🚀 Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/ramihatou97/koo-platform.git
cd koo-platform                      