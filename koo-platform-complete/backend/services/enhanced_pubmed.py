"""
Enhanced PubMed Integration for KOO Platform
Neurosurgical-specialized literature search with 10X improvements
"""

import asyncio
import aiohttp
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging
import re
import json
from urllib.parse import quote_plus

from core.config import settings
from core.cache import cache
from utils.text_processing import extract_keywords, clean_text

logger = logging.getLogger(__name__)

@dataclass
class NeurosurgicalPaper:
    """Enhanced paper data structure with neurosurgical focus"""
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return asdict(self)


class PubMedNeurosurgicalResearch:
    """Enhanced PubMed integration specialized for neurosurgical research"""

    def __init__(self, api_key: Optional[str] = None, email: Optional[str] = None):
        self.api_key = api_key or settings.PUBMED_API_KEY
        self.email = email or settings.PUBMED_EMAIL
        self.tool = settings.PUBMED_TOOL
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

        # Enhanced neurosurgical MeSH terms hierarchy
        self.neurosurgical_mesh = {
            "core": [
                "Neurosurgery", "Neurosurgical Procedures", "Neurosurgeons",
                "Brain Surgery", "Spinal Surgery", "Craniotomy"
            ],
            "conditions": [
                "Brain Neoplasms", "Brain Tumors", "Glioblastoma", "Meningioma",
                "Brain Aneurysm", "Arteriovenous Malformations",
                "Spinal Cord Injuries", "Brain Injuries, Traumatic",
                "Hydrocephalus", "Epilepsy", "Movement Disorders"
            ],
            "procedures": [
                "Stereotactic Techniques", "Deep Brain Stimulation",
                "Endoscopic Surgery", "Microsurgery", "Radiosurgery",
                "Laminectomy", "Diskectomy", "Spinal Fusion",
                "Ventriculostomy", "Burr Holes", "Shunt Surgery"
            ],
            "anatomy": [
                "Brain", "Spinal Cord", "Cerebral Cortex", "Brainstem",
                "Cerebellum", "Meninges", "Ventricular System",
                "Blood-Brain Barrier", "Cerebrospinal Fluid"
            ]
        }

        # Evidence hierarchy with neurosurgical focus
        self.evidence_hierarchy = {
            "systematic review": 1.0,
            "meta-analysis": 1.0,
            "randomized controlled trial": 0.95,
            "multicenter trial": 0.9,
            "clinical trial": 0.85,
            "prospective cohort": 0.8,
            "retrospective cohort": 0.75,
            "case-control study": 0.7,
            "case series": 0.6,
            "case report": 0.5,
            "expert opinion": 0.4,
            "review": 0.35
        }

        # High-impact neurosurgical journals
        self.high_impact_journals = {
            "Journal of Neurosurgery": 4.8,
            "Neurosurgery": 4.3,
            "Journal of Neuro-Oncology": 3.4,
            "World Neurosurgery": 2.9,
            "Neurosurgical Focus": 3.9,
            "Acta Neurochirurgica": 2.4,
            "Journal of Neurosurgical Sciences": 1.8,
            "Surgical Neurology International": 1.2,
            "Brain and Spine": 2.1,
            "Operative Neurosurgery": 2.5
        }

        # Study type patterns for classification
        self.study_type_patterns = {
            "systematic review": [
                r"systematic review", r"meta-analysis", r"systematic literature review"
            ],
            "randomized controlled trial": [
                r"randomized controlled trial", r"randomized clinical trial",
                r"rct", r"randomized", r"randomised"
            ],
            "cohort study": [
                r"cohort study", r"longitudinal study", r"prospective study",
                r"retrospective study", r"follow-up study"
            ],
            "case-control": [
                r"case-control", r"case control study"
            ],
            "case series": [
                r"case series", r"case study", r"case report",
                r"retrospective analysis", r"retrospective review"
            ],
            "clinical trial": [
                r"clinical trial", r"phase [i]+", r"multicenter"
            ]
        }

    async def neurosurgical_search(
        self,
        topic: str,
        max_results: int = 50,
        years_back: int = 5,
        evidence_filter: Optional[str] = None,
        include_clinical_trials: bool = True,
        min_impact_factor: float = 0.0,
        specialty_focus: Optional[str] = None
    ) -> List[NeurosurgicalPaper]:
        """
        Perform specialized neurosurgical literature search with enhanced filtering
        """
        try:
            logger.info(f"Starting neurosurgical search for: {topic}")

            # Check cache first
            cache_key = f"pubmed_search:{self._hash_query(topic, max_results, years_back, evidence_filter)}"
            cached_result = await cache.get(cache_key)
            if cached_result:
                logger.info("Returning cached PubMed results")
                return [NeurosurgicalPaper(**paper) for paper in cached_result]

            # Build specialized query
            query = self._build_surgical_query(
                topic, years_back, evidence_filter,
                include_clinical_trials, specialty_focus
            )

            # Search PubMed
            pmids = await self._search_pubmed(query, max_results)

            if not pmids:
                logger.warning(f"No results found for query: {topic}")
                return []

            # Fetch detailed paper information
            papers = await self._fetch_paper_details(pmids)

            # Enhance with neurosurgical analysis
            enhanced_papers = await self._analyze_surgical_relevance(papers, topic)

            # Filter by impact factor if specified
            if min_impact_factor > 0:
                enhanced_papers = [
                    p for p in enhanced_papers
                    if p.impact_factor >= min_impact_factor
                ]

            # Sort by comprehensive relevance score
            enhanced_papers.sort(
                key=lambda p: (
                    p.neurosurgical_score * 0.4 +
                    p.surgical_relevance * 0.3 +
                    p.clinical_significance * 0.2 +
                    self.evidence_hierarchy.get(p.evidence_level.lower(), 0.1) * 0.1
                ),
                reverse=True
            )

            # Cache results
            await cache.set(
                cache_key,
                [paper.to_dict() for paper in enhanced_papers],
                ttl=3600  # 1 hour cache
            )

            logger.info(f"Retrieved {len(enhanced_papers)} neurosurgical papers for: {topic}")
            return enhanced_papers

        except Exception as e:
            logger.error(f"Neurosurgical search failed: {str(e)}")
            raise

    def _hash_query(self, *args) -> str:
        """Generate cache key hash"""
        import hashlib
        query_str = str(args)
        return hashlib.md5(query_str.encode()).hexdigest()

    def _build_surgical_query(
        self,
        topic: str,
        years_back: int,
        evidence_filter: Optional[str] = None,
        include_clinical_trials: bool = True,
        specialty_focus: Optional[str] = None
    ) -> str:
        """Build optimized query for neurosurgical research"""

        # Clean and prepare topic
        topic_clean = clean_text(topic)
        topic_keywords = extract_keywords(topic_clean)

        # Base topic query with synonyms
        topic_variations = self._generate_topic_variations(topic_clean)
        base_query = f"({' OR '.join([f'"{var}"' for var in topic_variations])})"

        # Add neurosurgical context based on specialty focus
        if specialty_focus:
            neurosurg_terms = self.neurosurgical_mesh.get(specialty_focus,
                                                        self.neurosurgical_mesh["core"])
        else:
            # Use all terms for broad search
            neurosurg_terms = (
                self.neurosurgical_mesh["core"] +
                self.neurosurgical_mesh["conditions"] +
                self.neurosurgical_mesh["procedures"]
            )

        neurosurg_context = f"({' OR '.join([f'"{term}"' for term in neurosurg_terms[:15]])})"

        # Add surgical technique terms
        surgical_terms = (
            '("surgical technique" OR "operative technique" OR "surgical approach" OR '
            '"operative procedure" OR "surgical outcome" OR "operative outcome" OR '
            '"surgical management" OR "operative management")'
        )

        # Date filter
        current_year = datetime.now().year
        start_year = current_year - years_back
        date_filter = f"({start_year}:{current_year}[dp])"

        # Evidence type filter
        evidence_terms = ""
        if evidence_filter:
            evidence_map = {
                "rct": '("randomized controlled trial"[pt] OR "clinical trial"[pt] OR "randomized"[tiab])',
                "systematic_review": '("systematic review"[pt] OR "meta-analysis"[pt] OR "review"[pt])',
                "clinical": '("clinical study"[pt] OR "clinical trial"[pt] OR "multicenter study"[pt])',
                "case_series": '("case reports"[pt] OR "case series"[tiab])',
                "cohort": '("cohort studies"[mh] OR "longitudinal studies"[mh] OR "follow-up studies"[mh])'
            }
            evidence_terms = f" AND {evidence_map.get(evidence_filter, '')}"

        # Clinical trials inclusion
        clinical_trial_terms = ""
        if include_clinical_trials:
            clinical_trial_terms = ' OR "clinical trial"[pt] OR "controlled clinical trial"[pt]'

        # Quality filters
        quality_filters = (
            ' AND ("humans"[MeSH Terms]) AND ("english"[lang]) AND ("adult"[MeSH Terms] OR "child"[MeSH Terms])'
        )

        # Exclude certain publication types
        exclusions = (
            ' NOT ("editorial"[pt] OR "letter"[pt] OR "comment"[pt] OR '
            '"news"[pt] OR "newspaper article"[pt])'
        )

        # Combine all parts
        full_query = (
            f"({base_query} AND {neurosurg_context} AND {surgical_terms})"
            f" AND {date_filter}"
            f"{evidence_terms}"
            f"{clinical_trial_terms}"
            f"{quality_filters}"
            f"{exclusions}"
        )

        logger.debug(f"Built neurosurgical query: {full_query}")
        return full_query

    def _generate_topic_variations(self, topic: str) -> List[str]:
        """Generate topic variations and synonyms"""
        variations = [topic]

        # Common medical synonyms
        synonyms = {
            "tumor": ["tumour", "neoplasm", "mass", "lesion"],
            "surgery": ["surgical", "operative", "procedure"],
            "treatment": ["therapy", "management", "intervention"],
            "brain": ["cerebral", "intracranial", "CNS"],
            "spinal": ["spine", "vertebral", "rachis"],
            "outcome": ["result", "prognosis", "efficacy"]
        }

        topic_lower = topic.lower()
        for key, syns in synonyms.items():
            if key in topic_lower:
                for syn in syns:
                    variations.append(topic.replace(key, syn))
                    variations.append(topic.replace(key.capitalize(), syn.capitalize()))

        return list(set(variations))  # Remove duplicates

    async def _search_pubmed(self, query: str, max_results: int) -> List[str]:
        """Enhanced PubMed search with error handling and retries"""
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": max_results,
            "sort": "relevance",
            "field": "title,abstract"
        }

        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["email"] = self.email
        if self.tool:
            params["tool"] = self.tool

        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/esearch.fcgi"

            for attempt in range(3):  # Retry logic
                try:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            pmids = data.get("esearchresult", {}).get("idlist", [])
                            logger.info(f"Found {len(pmids)} PMIDs")
                            return pmids
                        elif response.status == 429:  # Rate limit
                            wait_time = 2 ** attempt
                            logger.warning(f"Rate limited, waiting {wait_time}s")
                            await asyncio.sleep(wait_time)
                        else:
                            logger.error(f"PubMed search failed: {response.status}")
                            break

                except Exception as e:
                    logger.error(f"Search attempt {attempt + 1} failed: {e}")
                    if attempt == 2:  # Last attempt
                        raise
                    await asyncio.sleep(1)

        return []

    async def _fetch_paper_details(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """Fetch detailed information for papers with batch processing"""
        if not pmids:
            return []

        # Process in batches to avoid URL length limits
        batch_size = 100
        all_papers = []

        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i + batch_size]
            batch_papers = await self._fetch_batch_details(batch_pmids)
            all_papers.extend(batch_papers)

            # Rate limiting between batches
            if i + batch_size < len(pmids):
                await asyncio.sleep(0.5)

        return all_papers

    async def _fetch_batch_details(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """Fetch details for a batch of PMIDs"""
        pmid_str = ",".join(pmids)
        params = {
            "db": "pubmed",
            "id": pmid_str,
            "retmode": "xml",
            "rettype": "abstract"
        }

        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["email"] = self.email
        if self.tool:
            params["tool"] = self.tool

        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/efetch.fcgi"

            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        xml_content = await response.text()
                        return self._parse_pubmed_xml(xml_content)
                    else:
                        logger.error(f"PubMed fetch failed: {response.status}")
                        return []

            except Exception as e:
                logger.error(f"Batch fetch failed: {e}")
                return []

    def _parse_pubmed_xml(self, xml_content: str) -> List[Dict[str, Any]]:
        """Enhanced XML parsing with error handling"""
        papers = []

        try:
            root = ET.fromstring(xml_content)

            for article in root.findall(".//PubmedArticle"):
                paper_data = self._extract_enhanced_paper_data(article)
                if paper_data:
                    papers.append(paper_data)

        except ET.ParseError as e:
            logger.error(f"XML parsing failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in XML parsing: {str(e)}")

        return papers

    def _extract_enhanced_paper_data(self, article) -> Optional[Dict[str, Any]]:
        """Extract comprehensive paper data from XML element"""
        try:
            # Basic identifiers
            pmid_elem = article.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else ""

            pmc_elem = article.find(".//ArticleId[@IdType='pmc']")
            pmc_id = pmc_elem.text if pmc_elem is not None else None

            # Title and abstract
            title_elem = article.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else ""
            title = clean_text(title) if title else ""

            abstract_elems = article.findall(".//AbstractText")
            abstract_parts = []
            for elem in abstract_elems:
                label = elem.get("Label", "")
                text = elem.text or ""
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
            abstract = " ".join(abstract_parts)
            abstract = clean_text(abstract) if abstract else ""

            # Authors
            authors = self._extract_authors_with_affiliations(article)

            # Journal information
            journal_elem = article.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None else ""

            # Publication date
            pub_date = self._extract_publication_date(article)

            # MeSH terms
            mesh_terms = self._extract_mesh_terms(article)

            # Publication types
            pub_types = self._extract_publication_types(article)

            # DOI
            doi_elem = article.find(".//ELocationID[@EIdType='doi']")
            doi = doi_elem.text if doi_elem is not None else None

            # Keywords
            keywords = self._extract_keywords(article, title, abstract)

            return {
                "pmid": pmid,
                "pmc_id": pmc_id,
                "title": title,
                "authors": authors,
                "journal": journal,
                "publication_date": pub_date,
                "abstract": abstract,
                "mesh_terms": mesh_terms,
                "publication_types": pub_types,
                "doi": doi,
                "keywords": keywords
            }

        except Exception as e:
            logger.error(f"Failed to extract paper data: {str(e)}")
            return None

    def _extract_authors_with_affiliations(self, article) -> List[str]:
        """Extract authors with their affiliations"""
        authors = []

        for author in article.findall(".//Author"):
            try:
                lastname = author.find(".//LastName")
                firstname = author.find(".//ForeName")
                initials = author.find(".//Initials")

                if lastname is not None:
                    name_parts = [lastname.text]
                    if firstname is not None:
                        name_parts.insert(0, firstname.text)
                    elif initials is not None:
                        name_parts.insert(0, initials.text)

                    # Get affiliation if available
                    affiliation = author.find(".//Affiliation")
                    if affiliation is not None and affiliation.text:
                        author_str = f"{' '.join(name_parts)} ({affiliation.text[:100]})"
                    else:
                        author_str = ' '.join(name_parts)

                    authors.append(author_str)

            except Exception as e:
                logger.error(f"Failed to extract author: {e}")
                continue

        return authors

    def _extract_publication_date(self, article) -> str:
        """Extract publication date"""
        try:
            pub_date = article.find(".//PubDate")
            if pub_date is not None:
                year = pub_date.find("Year")
                month = pub_date.find("Month")
                day = pub_date.find("Day")

                parts = []
                if year is not None:
                    parts.append(year.text)
                if month is not None:
                    parts.append(month.text)
                if day is not None:
                    parts.append(day.text)

                return "-".join(parts) if parts else ""
        except:
            pass
        return ""

    def _extract_mesh_terms(self, article) -> List[str]:
        """Extract MeSH terms"""
        mesh_terms = []
        for mesh in article.findall(".//MeshHeading/DescriptorName"):
            if mesh.text:
                mesh_terms.append(mesh.text)
        return mesh_terms

    def _extract_publication_types(self, article) -> List[str]:
        """Extract publication types"""
        pub_types = []
        for pub_type in article.findall(".//PublicationType"):
            if pub_type.text:
                pub_types.append(pub_type.text)
        return pub_types

    def _extract_keywords(self, article, title: str, abstract: str) -> List[str]:
        """Extract keywords from various sources"""
        keywords = []

        # Extract from keyword list if available
        for keyword in article.findall(".//Keyword"):
            if keyword.text:
                keywords.append(keyword.text)

        # Extract from title and abstract using NLP
        if title:
            keywords.extend(extract_keywords(title))
        if abstract:
            keywords.extend(extract_keywords(abstract)[:10])  # Limit abstract keywords

        return list(set(keywords))  # Remove duplicates

    async def _analyze_surgical_relevance(self, papers: List[Dict[str, Any]], topic: str) -> List[NeurosurgicalPaper]:
        """Analyze surgical relevance and create enhanced paper objects"""
        enhanced_papers = []

        for paper in papers:
            try:
                # Calculate scores
                surgical_relevance = self._calculate_surgical_relevance(paper, topic)
                clinical_significance = self._calculate_clinical_significance(paper)
                neurosurgical_score = self._calculate_neurosurgical_score(paper)
                evidence_level = self._determine_evidence_level(paper)

                # Get impact factor
                impact_factor = self.high_impact_journals.get(paper.get("journal", ""), 0.0)

                # Create enhanced paper object
                enhanced_paper = NeurosurgicalPaper(
                    pmid=paper.get("pmid", ""),
                    title=paper.get("title", ""),
                    authors=paper.get("authors", []),
                    journal=paper.get("journal", ""),
                    publication_date=paper.get("publication_date", ""),
                    abstract=paper.get("abstract", ""),
                    mesh_terms=paper.get("mesh_terms", []),
                    evidence_level=evidence_level,
                    surgical_relevance=surgical_relevance,
                    clinical_significance=clinical_significance,
                    neurosurgical_score=neurosurgical_score,
                    keywords=paper.get("keywords", []),
                    doi=paper.get("doi"),
                    pmc_id=paper.get("pmc_id"),
                    impact_factor=impact_factor
                )

                enhanced_papers.append(enhanced_paper)

            except Exception as e:
                logger.error(f"Failed to analyze paper {paper.get('pmid', 'unknown')}: {e}")
                continue

        return enhanced_papers

    def _calculate_surgical_relevance(self, paper: Dict[str, Any], topic: str) -> float:
        """Calculate surgical relevance score"""
        score = 0.0
        content = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()

        # Check for surgical terms
        surgical_terms = [
            "surgery", "surgical", "operative", "operation", "procedure",
            "technique", "approach", "treatment", "management"
        ]

        for term in surgical_terms:
            if term in content:
                score += 0.1

        # Check for neurosurgical specificity
        for category, terms in self.neurosurgical_mesh.items():
            for term in terms:
                if term.lower() in content:
                    score += 0.15

        # Topic relevance
        topic_words = topic.lower().split()
        for word in topic_words:
            if word in content:
                score += 0.05

        return min(score, 1.0)

    def _calculate_clinical_significance(self, paper: Dict[str, Any]) -> float:
        """Calculate clinical significance score"""
        score = 0.0
        content = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()

        # Check for clinical outcome terms
        clinical_terms = [
            "outcome", "survival", "mortality", "morbidity", "efficacy",
            "safety", "complication", "recovery", "prognosis", "quality of life"
        ]

        for term in clinical_terms:
            if term in content:
                score += 0.1

        # Check for study size indicators
        if any(term in content for term in ["multicenter", "large cohort", "population"]):
            score += 0.2

        return min(score, 1.0)

    def _calculate_neurosurgical_score(self, paper: Dict[str, Any]) -> float:
        """Calculate neurosurgical specificity score"""
        score = 0.0

        # Journal impact factor contribution
        journal = paper.get("journal", "")
        if journal in self.high_impact_journals:
            score += self.high_impact_journals[journal] / 10.0

        # MeSH terms contribution
        mesh_terms = paper.get("mesh_terms", [])
        neurosurg_mesh = sum(self.neurosurgical_mesh.values(), [])

        for mesh_term in mesh_terms:
            if mesh_term in neurosurg_mesh:
                score += 0.1

        return min(score, 1.0)

    def _determine_evidence_level(self, paper: Dict[str, Any]) -> str:
        """Determine evidence level from publication types and content"""
        pub_types = [pt.lower() for pt in paper.get("publication_types", [])]
        content = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()

        # Check publication types first
        for level, score in self.evidence_hierarchy.items():
            if any(level_term in pub_types for level_term in level.split()):
                return level

        # Check content patterns
        for study_type, patterns in self.study_type_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    return study_type

        return "review"  # Default

    async def health_check(self) -> Dict[str, Any]:
        """Health check for PubMed service"""
        try:
            # Simple search to test connectivity
            params = {
                "db": "pubmed",
                "term": "neurosurgery",
                "retmax": 1,
                "retmode": "json"
            }

            if self.api_key:
                params["api_key"] = self.api_key

            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/esearch.fcgi"
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "status": "healthy",
                            "api_key_valid": bool(self.api_key),
                            "search_working": bool(data.get("esearchresult", {}).get("idlist")),
                            "response_time_ms": response.headers.get("X-Response-Time", "unknown")
                        }
                    else:
                        return {
                            "status": "unhealthy",
                            "error": f"HTTP {response.status}"
                        }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }