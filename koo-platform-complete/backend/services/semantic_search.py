"""
Semantic Search Service
Advanced semantic search across medical literature and chapters using AI embeddings
"""

import asyncio
import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import re
from collections import defaultdict
import hashlib

from .hybrid_ai_manager import query_ai
from .advanced_pubmed_analytics import AdvancedPubMedAnalytics
from core.config import settings
from core.exceptions import ExternalServiceError

logger = logging.getLogger(__name__)

class SearchType(Enum):
    """Types of semantic search"""
    GENERAL = "general"
    CLINICAL = "clinical"
    RESEARCH = "research"
    DIAGNOSTIC = "diagnostic"
    THERAPEUTIC = "therapeutic"
    PROCEDURAL = "procedural"

class ContentType(Enum):
    """Types of content to search"""
    CHAPTERS = "chapters"
    RESEARCH_PAPERS = "research_papers"
    CASE_STUDIES = "case_studies"
    GUIDELINES = "guidelines"
    ALL = "all"

@dataclass
class SearchQuery:
    """Semantic search query"""
    query_text: str
    search_type: SearchType
    content_types: List[ContentType]
    specialty_filter: Optional[List[str]] = None
    date_range: Optional[Tuple[str, str]] = None
    quality_threshold: float = 0.0
    max_results: int = 50
    include_synonyms: bool = True
    expand_medical_terms: bool = True

@dataclass
class SearchResult:
    """Individual search result"""
    content_id: str
    title: str
    content_type: ContentType
    excerpt: str
    full_content: str
    relevance_score: float
    semantic_similarity: float
    keyword_matches: List[str]
    medical_concepts: List[str]
    source: str
    metadata: Dict[str, Any]
    highlighted_text: str

@dataclass
class SemanticSearchResponse:
    """Complete search response"""
    query: SearchQuery
    results: List[SearchResult]
    total_results: int
    search_time_ms: int
    query_expansion: List[str]
    related_concepts: List[str]
    suggested_filters: Dict[str, List[str]]
    semantic_clusters: List[Dict[str, Any]]

class MedicalConceptExtractor:
    """Extract medical concepts and terminology"""

    def __init__(self):
        # Neurosurgical terminology categories
        self.medical_categories = {
            "neuroanatomy": [
                "frontal lobe", "parietal lobe", "temporal lobe", "occipital lobe",
                "cerebellum", "brainstem", "midbrain", "pons", "medulla",
                "thalamus", "hypothalamus", "basal ganglia", "hippocampus",
                "sylvian fissure", "central sulcus", "motor cortex", "sensory cortex",
                "corpus callosum", "ventricular system", "CSF pathway"
            ],
            "spinal_anatomy": [
                "cervical spine", "thoracic spine", "lumbar spine", "sacrum", "coccyx",
                "vertebral body", "facet joint", "pedicle", "lamina", "spinous process",
                "spinal cord", "nerve root", "dura mater", "ligamentum flavum",
                "anterior longitudinal ligament", "posterior longitudinal ligament"
            ],
            "surgical_approaches": [
                "pterional", "frontotemporal", "orbitozygomatic", "retrosigmoid",
                "subtemporal", "interhemispheric", "transcallosal", "transpetrosal",
                "presigmoid", "endoscopic endonasal", "transsphenoidal",
                "anterior cervical", "posterior cervical", "lateral approach"
            ],
            "surgical_techniques": [
                "microsurgical dissection", "awake craniotomy", "intraoperative monitoring",
                "stereotactic surgery", "image-guided surgery", "endoscopic surgery",
                "minimally invasive", "keyhole surgery", "brain mapping",
                "cortical stimulation", "subcortical stimulation", "temporary clipping"
            ],
            "neurosurgical_conditions": [
                "glioblastoma", "meningioma", "pituitary adenoma", "acoustic neuroma",
                "cerebral aneurysm", "arteriovenous malformation", "cavernoma",
                "cervical myelopathy", "lumbar stenosis", "disc herniation",
                "hydrocephalus", "Chiari malformation", "epilepsy", "trigeminal neuralgia"
            ],
            "neurosurgical_procedures": [
                "craniotomy", "craniectomy", "aneurysm clipping", "AVM resection",
                "tumor resection", "laminectomy", "laminoplasty", "discectomy",
                "fusion", "shunt placement", "deep brain stimulation",
                "gamma knife", "stereotactic radiosurgery", "embolization"
            ],
            "imaging_techniques": [
                "MRI", "CT", "CTA", "MRA", "DSA", "PET", "SPECT", "DTI",
                "functional MRI", "intraoperative ultrasound", "fluorescein",
                "5-ALA", "indocyanine green", "neurophysiological monitoring"
            ],
            "clinical_assessment": [
                "Glasgow Coma Scale", "Karnofsky Performance Scale", "mRS",
                "Oswestry Disability Index", "JOA score", "Nurick grade",
                "Hunt-Hess grade", "Fisher grade", "WFNS grade"
            ]
        }

        # Neurosurgical synonyms and terminology
        self.medical_synonyms = {
            "brain tumor": ["brain neoplasm", "intracranial tumor", "cerebral tumor", "CNS neoplasm"],
            "aneurysm": ["cerebral aneurysm", "intracranial aneurysm", "berry aneurysm"],
            "AVM": ["arteriovenous malformation", "cerebral AVM", "brain AVM"],
            "craniotomy": ["cranial opening", "bone flap", "skull opening"],
            "disc herniation": ["herniated disc", "disc prolapse", "ruptured disc"],
            "spinal stenosis": ["canal stenosis", "spinal narrowing", "neural foraminal stenosis"],
            "hydrocephalus": ["water on brain", "ventricular enlargement", "CSF accumulation"],
            "myelopathy": ["spinal cord dysfunction", "cord compression", "spinal cord syndrome"],
            "radiculopathy": ["nerve root compression", "pinched nerve", "nerve impingement"],
            "glioblastoma": ["GBM", "grade IV astrocytoma", "malignant glioma"],
            "meningioma": ["meningeal tumor", "dural-based tumor"],
            "pituitary adenoma": ["pituitary tumor", "sellar mass", "pituitary neoplasm"]
        }

    def extract_concepts(self, text: str) -> Dict[str, List[str]]:
        """Extract medical concepts from text"""
        text_lower = text.lower()
        concepts = defaultdict(list)

        for category, terms in self.medical_categories.items():
            for term in terms:
                if term in text_lower:
                    concepts[category].append(term)

        return dict(concepts)

    def expand_query(self, query: str) -> List[str]:
        """Expand query with medical synonyms and related terms"""
        expanded_terms = [query]
        query_lower = query.lower()

        # Add synonyms
        for term, synonyms in self.medical_synonyms.items():
            if term in query_lower:
                expanded_terms.extend(synonyms)

        # Add related anatomical terms
        if any(anat in query_lower for anat in self.medical_categories["anatomy"]):
            expanded_terms.extend(["neuroanatomy", "brain structure"])

        return list(set(expanded_terms))

class SemanticSearchEngine:
    """Advanced semantic search with medical concept understanding"""

    def __init__(self):
        self.analytics_service = AdvancedPubMedAnalytics()
        self.concept_extractor = MedicalConceptExtractor()

        # Simulated vector database for chapters and content
        self.content_index = {}
        self.concept_embeddings = {}

        # Initialize with sample content
        self._initialize_content_index()

    def _initialize_content_index(self):
        """Initialize content index with neurosurgical knowledge content"""
        neurosurgical_chapters = [
            {
                "id": "anatomy_001",
                "title": "Cranial Anatomy and Surgical Approaches",
                "content": "The skull consists of frontal, parietal, temporal, and occipital bones. Surgical approaches include pterional, frontotemporal, retrosigmoid, and interhemispheric. Key anatomical landmarks for the pterional approach include the sphenoid ridge, sylvian fissure, and frontal horn of lateral ventricle.",
                "type": "anatomy",
                "specialty": ["cranial anatomy", "surgical approaches"],
                "concepts": ["pterional approach", "craniotomy", "anatomical landmarks", "surgical anatomy"]
            },
            {
                "id": "technique_001",
                "title": "Microsurgical Clipping of Anterior Communicating Artery Aneurysms",
                "content": "Surgical technique for ACoA aneurysm clipping involves pterional approach, sylvian fissure dissection, identification of A1 segments, temporary clipping if needed, and clip application perpendicular to aneurysm neck. Key steps include CSF drainage, brain relaxation, and careful preservation of perforating arteries.",
                "type": "surgical_technique",
                "specialty": ["vascular neurosurgery", "aneurysm surgery"],
                "concepts": ["microsurgical clipping", "pterional approach", "sylvian dissection", "ACoA aneurysm"]
            },
            {
                "id": "disease_001",
                "title": "Glioblastoma: Epidemiology, Imaging, and Surgical Management",
                "content": "Glioblastoma (WHO Grade IV) affects 3-4 per 100,000 annually, peak age 55-65. MRI shows heterogeneous enhancement, necrosis, and peritumoral edema. Surgical goals include maximal safe resection with intraoperative neuromonitoring, awake craniotomy for eloquent areas, and extent of resection correlation with survival.",
                "type": "disease_knowledge",
                "specialty": ["neuro-oncology", "brain tumors"],
                "concepts": ["glioblastoma", "surgical resection", "intraoperative monitoring", "survival outcomes"]
            },
            {
                "id": "anatomy_002",
                "title": "Spinal Anatomy and Surgical Corridors",
                "content": "Cervical spine anatomy includes vertebral bodies C1-C7, facet joints, ligamentous structures. Surgical approaches: anterior (Smith-Robinson), posterior (laminectomy, laminoplasty), lateral (XLIF). Key anatomical considerations include vertebral artery course, nerve root anatomy, and spinal cord dimensions.",
                "type": "anatomy",
                "specialty": ["spinal anatomy", "spine surgery"],
                "concepts": ["cervical spine", "surgical approaches", "vertebral artery", "spinal cord anatomy"]
            },
            {
                "id": "technique_002",
                "title": "Endoscopic Endonasal Transsphenoidal Surgery",
                "content": "Endoscopic endonasal approach for pituitary tumors involves nasal preparation, middle turbinate identification, sphenoidotomy, sellar opening, and tumor removal. Key anatomical landmarks include sphenoid rostrum, carotid prominences, and optic prominences. Closure techniques include fat graft, fascia lata, and vascularized nasoseptal flap.",
                "type": "surgical_technique",
                "specialty": ["pituitary surgery", "skull base surgery"],
                "concepts": ["endoscopic surgery", "transsphenoidal", "pituitary adenoma", "skull base anatomy"]
            },
            {
                "id": "disease_002",
                "title": "Cervical Spondylotic Myelopathy: Pathophysiology and Treatment",
                "content": "CSM results from dynamic and static compression of spinal cord. Pathophysiology involves mechanical compression, ischemia, and inflammatory cascade. Clinical manifestations include myelopathic gait, upper extremity dysfunction, and hyperreflexia. Surgical options include anterior discectomy/fusion, laminoplasty, or combined approaches.",
                "type": "disease_knowledge",
                "specialty": ["spine surgery", "degenerative spine"],
                "concepts": ["cervical myelopathy", "spinal cord compression", "surgical decompression", "clinical outcomes"]
            }
        ]

        for chapter in neurosurgical_chapters:
            self.content_index[chapter["id"]] = chapter

    async def semantic_search(self, query: SearchQuery) -> SemanticSearchResponse:
        """Perform semantic search across medical content"""
        logger.info(f"Performing semantic search for: {query.query_text}")

        start_time = datetime.now()

        try:
            # Step 1: Expand query with medical concepts
            expanded_query = await self._expand_medical_query(query)

            # Step 2: Extract medical concepts from query
            query_concepts = self.concept_extractor.extract_concepts(query.query_text)

            # Step 3: Search across different content types
            search_results = await self._search_content(query, expanded_query, query_concepts)

            # Step 4: Rank and score results
            ranked_results = await self._rank_results(search_results, query, query_concepts)

            # Step 5: Generate semantic clusters
            clusters = await self._generate_semantic_clusters(ranked_results)

            # Step 6: Generate suggestions
            suggestions = await self._generate_suggestions(query, ranked_results)

            search_time = (datetime.now() - start_time).total_seconds() * 1000

            response = SemanticSearchResponse(
                query=query,
                results=ranked_results[:query.max_results],
                total_results=len(ranked_results),
                search_time_ms=int(search_time),
                query_expansion=expanded_query,
                related_concepts=list(query_concepts.keys()),
                suggested_filters=suggestions["filters"],
                semantic_clusters=clusters
            )

            logger.info(f"Search completed in {search_time:.0f}ms with {len(ranked_results)} results")
            return response

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            raise ExternalServiceError("search", f"Search failed: {str(e)}")

    async def _expand_medical_query(self, query: SearchQuery) -> List[str]:
        """Expand query using medical terminology and AI"""
        expanded_terms = [query.query_text]

        if query.expand_medical_terms:
            # Use concept extractor
            concept_expansion = self.concept_extractor.expand_query(query.query_text)
            expanded_terms.extend(concept_expansion)

            # Use AI for semantic expansion
            try:
                expansion_prompt = f"""Expand this medical search query with related clinical terms and synonyms:

Query: "{query.query_text}"
Search Type: {query.search_type.value}

Provide 5-8 related medical terms that would help find relevant content. Include:
- Medical synonyms
- Related procedures
- Associated conditions
- Relevant anatomy

Respond with a comma-separated list of terms."""

                ai_expansion = await query_ai("gemini", expansion_prompt, max_tokens=200)
                ai_terms = [term.strip() for term in ai_expansion.split(",")]
                expanded_terms.extend(ai_terms)

            except Exception as e:
                logger.warning(f"AI query expansion failed: {e}")

        return list(set(expanded_terms))

    async def _search_content(self, query: SearchQuery, expanded_query: List[str], query_concepts: Dict[str, List[str]]) -> List[SearchResult]:
        """Search across different content types"""
        results = []

        # Search chapters
        if ContentType.CHAPTERS in query.content_types or ContentType.ALL in query.content_types:
            chapter_results = await self._search_chapters(query, expanded_query, query_concepts)
            results.extend(chapter_results)

        # Search research papers
        if ContentType.RESEARCH_PAPERS in query.content_types or ContentType.ALL in query.content_types:
            research_results = await self._search_research_papers(query, expanded_query, query_concepts)
            results.extend(research_results)

        return results

    async def _search_chapters(self, query: SearchQuery, expanded_query: List[str], query_concepts: Dict[str, List[str]]) -> List[SearchResult]:
        """Search through chapters"""
        results = []

        for chapter_id, chapter in self.content_index.items():
            if chapter["type"] != "chapters":
                continue

            # Calculate relevance scores
            relevance_score = self._calculate_text_relevance(
                query.query_text, chapter["content"], expanded_query
            )

            concept_score = self._calculate_concept_similarity(
                query_concepts, chapter.get("concepts", [])
            )

            if query.specialty_filter:
                specialty_match = any(
                    spec in chapter.get("specialty", [])
                    for spec in query.specialty_filter
                )
                if not specialty_match:
                    continue

            overall_score = (relevance_score * 0.7) + (concept_score * 0.3)

            if overall_score >= query.quality_threshold:
                # Generate excerpt
                excerpt = self._generate_excerpt(chapter["content"], query.query_text)

                # Highlight text
                highlighted = self._highlight_matches(excerpt, [query.query_text] + expanded_query[:3])

                result = SearchResult(
                    content_id=chapter_id,
                    title=chapter["title"],
                    content_type=ContentType.CHAPTERS,
                    excerpt=excerpt,
                    full_content=chapter["content"],
                    relevance_score=overall_score,
                    semantic_similarity=concept_score,
                    keyword_matches=self._find_keyword_matches(chapter["content"], expanded_query),
                    medical_concepts=chapter.get("concepts", []),
                    source="KOO Platform Chapters",
                    metadata={
                        "specialty": chapter.get("specialty", []),
                        "chapter_id": chapter_id
                    },
                    highlighted_text=highlighted
                )

                results.append(result)

        return results

    async def _search_research_papers(self, query: SearchQuery, expanded_query: List[str], query_concepts: Dict[str, List[str]]) -> List[SearchResult]:
        """Search through research papers using PubMed"""
        try:
            # Use our enhanced PubMed search
            papers = await self.analytics_service.pubmed_service.neurosurgical_search(
                topic=query.query_text,
                max_results=min(query.max_results, 30),
                years_back=5
            )

            results = []
            for paper in papers:
                # Calculate relevance
                content = getattr(paper, 'abstract', '') or getattr(paper, 'title', '')
                relevance_score = self._calculate_text_relevance(
                    query.query_text, content, expanded_query
                )

                if relevance_score >= query.quality_threshold:
                    excerpt = self._generate_excerpt(content, query.query_text)
                    highlighted = self._highlight_matches(excerpt, [query.query_text])

                    result = SearchResult(
                        content_id=getattr(paper, 'pmid', str(hash(paper.title))),
                        title=getattr(paper, 'title', ''),
                        content_type=ContentType.RESEARCH_PAPERS,
                        excerpt=excerpt,
                        full_content=content,
                        relevance_score=relevance_score,
                        semantic_similarity=relevance_score,
                        keyword_matches=self._find_keyword_matches(content, expanded_query),
                        medical_concepts=getattr(paper, 'keywords', [])[:5],
                        source=getattr(paper, 'journal', 'PubMed'),
                        metadata={
                            "authors": getattr(paper, 'authors', [])[:3],
                            "publication_date": getattr(paper, 'publication_date', ''),
                            "journal": getattr(paper, 'journal', ''),
                            "pmid": getattr(paper, 'pmid', '')
                        },
                        highlighted_text=highlighted
                    )

                    results.append(result)

            return results

        except Exception as e:
            logger.error(f"Research paper search failed: {e}")
            return []

    async def _rank_results(self, results: List[SearchResult], query: SearchQuery, query_concepts: Dict[str, List[str]]) -> List[SearchResult]:
        """Rank search results by relevance and semantic similarity"""

        # Apply additional scoring factors
        for result in results:
            # Boost recent content
            if result.content_type == ContentType.RESEARCH_PAPERS:
                pub_date = result.metadata.get("publication_date", "")
                if pub_date and "2023" in pub_date or "2024" in pub_date:
                    result.relevance_score *= 1.1

            # Boost content with more medical concepts
            if len(result.medical_concepts) > 3:
                result.relevance_score *= 1.05

            # Apply search type boosting
            if query.search_type == SearchType.CLINICAL:
                if any(term in result.full_content.lower() for term in ["clinical", "patient", "treatment"]):
                    result.relevance_score *= 1.1

        # Sort by relevance score
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results

    async def _generate_semantic_clusters(self, results: List[SearchResult]) -> List[Dict[str, Any]]:
        """Group results into semantic clusters"""
        clusters = []

        # Simple clustering by medical concepts
        concept_groups = defaultdict(list)

        for result in results:
            primary_concept = result.medical_concepts[0] if result.medical_concepts else "general"
            concept_groups[primary_concept].append(result)

        for concept, cluster_results in concept_groups.items():
            if len(cluster_results) >= 2:  # Minimum cluster size
                clusters.append({
                    "concept": concept,
                    "result_count": len(cluster_results),
                    "avg_relevance": sum(r.relevance_score for r in cluster_results) / len(cluster_results),
                    "content_types": list(set(r.content_type.value for r in cluster_results))
                })

        return sorted(clusters, key=lambda x: x["avg_relevance"], reverse=True)

    async def _generate_suggestions(self, query: SearchQuery, results: List[SearchResult]) -> Dict[str, Any]:
        """Generate search suggestions and filters"""
        suggestions = {
            "filters": {
                "specialties": [],
                "content_types": [],
                "concepts": [],
                "date_ranges": []
            },
            "related_queries": []
        }

        # Extract filter suggestions from results
        if results:
            # Specialty filters
            specialties = set()
            for result in results:
                if "specialty" in result.metadata:
                    specialties.update(result.metadata["specialty"])
            suggestions["filters"]["specialties"] = list(specialties)[:5]

            # Concept filters
            concepts = []
            for result in results:
                concepts.extend(result.medical_concepts)
            concept_counts = Counter(concepts)
            suggestions["filters"]["concepts"] = [concept for concept, count in concept_counts.most_common(5)]

            # Content type distribution
            content_types = [result.content_type.value for result in results]
            suggestions["filters"]["content_types"] = list(set(content_types))

        return suggestions

    def _calculate_text_relevance(self, query: str, content: str, expanded_terms: List[str]) -> float:
        """Calculate text relevance score"""
        try:
            query_lower = query.lower()
            content_lower = content.lower()

            # Exact query match
            exact_score = 1.0 if query_lower in content_lower else 0.0

            # Term frequency scoring
            term_scores = []
            for term in [query] + expanded_terms:
                term_lower = term.lower()
                if term_lower in content_lower:
                    # Simple TF calculation
                    tf = content_lower.count(term_lower) / len(content_lower.split())
                    term_scores.append(min(tf * 100, 1.0))  # Normalize

            avg_term_score = sum(term_scores) / len(term_scores) if term_scores else 0.0

            # Combine scores
            relevance = (exact_score * 0.4) + (avg_term_score * 0.6)
            return min(relevance, 1.0)

        except Exception:
            return 0.0

    def _calculate_concept_similarity(self, query_concepts: Dict[str, List[str]], content_concepts: List[str]) -> float:
        """Calculate concept similarity score"""
        try:
            if not query_concepts or not content_concepts:
                return 0.0

            query_terms = []
            for concept_list in query_concepts.values():
                query_terms.extend(concept_list)

            if not query_terms:
                return 0.0

            # Calculate Jaccard similarity
            query_set = set(term.lower() for term in query_terms)
            content_set = set(term.lower() for term in content_concepts)

            intersection = len(query_set.intersection(content_set))
            union = len(query_set.union(content_set))

            return intersection / union if union > 0 else 0.0

        except Exception:
            return 0.0

    def _generate_excerpt(self, content: str, query: str, max_length: int = 300) -> str:
        """Generate relevant excerpt from content"""
        try:
            sentences = content.split('. ')
            query_lower = query.lower()

            # Find sentences containing query terms
            relevant_sentences = []
            for sentence in sentences:
                if query_lower in sentence.lower():
                    relevant_sentences.append(sentence)

            if relevant_sentences:
                excerpt = '. '.join(relevant_sentences[:2])
            else:
                # Fallback to first few sentences
                excerpt = '. '.join(sentences[:2])

            # Truncate if too long
            if len(excerpt) > max_length:
                excerpt = excerpt[:max_length] + "..."

            return excerpt

        except Exception:
            return content[:max_length] + "..." if len(content) > max_length else content

    def _highlight_matches(self, text: str, terms: List[str]) -> str:
        """Highlight matching terms in text"""
        try:
            highlighted = text
            for term in terms:
                if term and len(term) > 2:  # Avoid highlighting very short terms
                    pattern = re.compile(re.escape(term), re.IGNORECASE)
                    highlighted = pattern.sub(f"<mark>{term}</mark>", highlighted)
            return highlighted
        except Exception:
            return text

    def _find_keyword_matches(self, content: str, terms: List[str]) -> List[str]:
        """Find which terms match in the content"""
        matches = []
        content_lower = content.lower()

        for term in terms:
            if term and term.lower() in content_lower:
                matches.append(term)

        return matches

    async def get_search_suggestions(self, partial_query: str, limit: int = 10) -> List[str]:
        """Get search suggestions for autocomplete"""
        try:
            # Medical term suggestions based on partial query
            suggestions = []
            partial_lower = partial_query.lower()

            # Search in medical categories
            for category, terms in self.concept_extractor.medical_categories.items():
                for term in terms:
                    if partial_lower in term.lower() and term not in suggestions:
                        suggestions.append(term)

            # Add common medical phrases
            common_phrases = [
                "brain tumor treatment",
                "epilepsy surgery",
                "aneurysm clipping",
                "spine fusion",
                "glioblastoma prognosis",
                "stroke rehabilitation",
                "hydrocephalus shunt"
            ]

            for phrase in common_phrases:
                if partial_lower in phrase.lower() and phrase not in suggestions:
                    suggestions.append(phrase)

            return suggestions[:limit]

        except Exception as e:
            logger.error(f"Search suggestions failed: {e}")
            return []

    async def get_trending_searches(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get trending search topics"""
        try:
            # Simulated trending searches
            trending = [
                {"query": "glioblastoma immunotherapy", "count": 45, "growth": "+23%"},
                {"query": "minimally invasive spine surgery", "count": 38, "growth": "+15%"},
                {"query": "epilepsy surgery outcomes", "count": 32, "growth": "+18%"},
                {"query": "aneurysm endovascular treatment", "count": 29, "growth": "+12%"},
                {"query": "brain stimulation therapy", "count": 25, "growth": "+20%"}
            ]

            return trending[:limit]

        except Exception as e:
            logger.error(f"Trending searches failed: {e}")
            return []

# Global instance
semantic_search_engine = SemanticSearchEngine()