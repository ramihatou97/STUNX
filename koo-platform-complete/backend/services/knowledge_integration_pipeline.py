"""
KOO Platform Knowledge Integration Pipeline
Orchestrates multi-source research workflow for automated knowledge synthesis
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import uuid

from .hybrid_ai_manager import hybrid_ai_manager, query_ai
from .enhanced_pubmed import enhanced_pubmed_service
from .advanced_pubmed_analytics import AdvancedPubMedAnalytics
from .pubmed_service import PubMedService
from ..core.exceptions import ExternalServiceError, ValidationError
from ..core.config import settings

logger = logging.getLogger(__name__)

class PipelineStage(Enum):
    """Pipeline execution stages"""
    INITIATED = "initiated"
    RESEARCH_GATHERING = "research_gathering"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    QUALITY_CHECK = "quality_check"
    CHAPTER_UPDATE = "chapter_update"
    COMPLETED = "completed"
    FAILED = "failed"

class ResearchSource(Enum):
    """Research data sources"""
    PUBMED = "pubmed"
    GEMINI = "gemini"
    CLAUDE = "claude"
    PERPLEXITY = "perplexity"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    ELSEVIER = "elsevier"

@dataclass
class ResearchQuery:
    """Research query configuration"""
    topic: str
    neurosurgical_focus: List[str]
    mesh_terms: List[str]
    date_range: Optional[Tuple[str, str]] = None
    max_results: int = 50
    quality_threshold: float = 0.7
    include_citations: bool = True

@dataclass
class ResearchResult:
    """Individual research result"""
    source: ResearchSource
    title: str
    content: str
    url: Optional[str] = None
    doi: Optional[str] = None
    authors: List[str] = None
    publication_date: Optional[str] = None
    journal: Optional[str] = None
    citations: int = 0
    quality_score: float = 0.0
    relevance_score: float = 0.0
    neurosurgical_relevance: float = 0.0
    extracted_at: datetime = None

@dataclass
class KnowledgeSynthesis:
    """Synthesized knowledge output"""
    topic: str
    executive_summary: str
    key_findings: List[str]
    clinical_implications: List[str]
    surgical_techniques: List[str]
    evidence_quality: str
    conflicting_findings: List[str]
    research_gaps: List[str]
    recommendations: List[str]
    sources_used: List[str]
    confidence_score: float
    last_updated: datetime

@dataclass
class PipelineExecution:
    """Pipeline execution tracking"""
    execution_id: str
    query: ResearchQuery
    stage: PipelineStage
    results: List[ResearchResult]
    synthesis: Optional[KnowledgeSynthesis] = None
    started_at: datetime = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = None

class KnowledgeIntegrationPipeline:
    """Main pipeline orchestrator"""

    def __init__(self):
        self.active_executions: Dict[str, PipelineExecution] = {}
        self.completed_executions: List[PipelineExecution] = []
        self.neurosurgical_mesh_terms = [
            "Neurosurgery", "Brain Neoplasms", "Craniotomy", "Neurosurgical Procedures",
            "Brain Injuries", "Spinal Cord Injuries", "Cerebrovascular Disorders",
            "Epilepsy Surgery", "Deep Brain Stimulation", "Stereotactic Techniques",
            "Minimally Invasive Surgical Procedures", "Neuronavigation",
            "Intraoperative Monitoring", "Brain Mapping", "Tumor Resection"
        ]

        # Initialize enhanced analytics service
        self.pubmed_service = PubMedService()
        self.analytics_service = AdvancedPubMedAnalytics(self.pubmed_service)

    async def execute_research_pipeline(self, query: ResearchQuery) -> str:
        """Execute complete research pipeline"""
        execution_id = str(uuid.uuid4())
        execution = PipelineExecution(
            execution_id=execution_id,
            query=query,
            stage=PipelineStage.INITIATED,
            results=[],
            started_at=datetime.now()
        )

        self.active_executions[execution_id] = execution
        logger.info(f"Starting research pipeline {execution_id} for topic: {query.topic}")

        try:
            # Stage 1: Research Gathering
            await self._stage_research_gathering(execution)

            # Stage 2: Analysis and Scoring
            await self._stage_analysis(execution)

            # Stage 3: Multi-AI Synthesis
            await self._stage_synthesis(execution)

            # Stage 4: Quality Check
            await self._stage_quality_check(execution)

            # Stage 5: Chapter Update
            await self._stage_chapter_update(execution)

            # Mark as completed
            execution.stage = PipelineStage.COMPLETED
            execution.completed_at = datetime.now()

            logger.info(f"Pipeline {execution_id} completed successfully")

        except Exception as e:
            execution.stage = PipelineStage.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.now()
            logger.error(f"Pipeline {execution_id} failed: {e}")

        finally:
            # Move to completed executions
            if execution_id in self.active_executions:
                self.completed_executions.append(self.active_executions.pop(execution_id))

        return execution_id

    async def _stage_research_gathering(self, execution: PipelineExecution):
        """Stage 1: Gather research from multiple sources with enhanced analytics"""
        execution.stage = PipelineStage.RESEARCH_GATHERING
        logger.info(f"Pipeline {execution.execution_id}: Starting research gathering")

        # Enhance query with neurosurgical context
        enhanced_query = self._enhance_query_with_neurosurgical_context(execution.query)

        # Gather from PubMed with enhanced analytics
        pubmed_results = await self._gather_from_pubmed(enhanced_query)
        execution.results.extend(pubmed_results)

        # Add citation network analysis for the topic
        try:
            citation_analysis = await self.analytics_service.analyze_citation_network(
                topic=execution.query.topic,
                max_papers=30,
                years_back=3
            )

            # Add citation insights to execution metadata
            if not hasattr(execution, 'metadata'):
                execution.metadata = {}
            execution.metadata['citation_analysis'] = citation_analysis

            logger.info(f"Pipeline {execution.execution_id}: Added citation network analysis")
        except Exception as e:
            logger.warning(f"Citation analysis failed: {e}")

        # Gather from Perplexity (real-time research)
        perplexity_results = await self._gather_from_perplexity(enhanced_query)
        execution.results.extend(perplexity_results)

        # Use Gemini for deep research analysis
        gemini_results = await self._gather_from_gemini(enhanced_query)
        execution.results.extend(gemini_results)

        # Get research trend insights
        try:
            trend_analysis = await self.analytics_service.analyze_research_trends(
                specialty=execution.query.topic,
                years=5
            )
            execution.metadata['trend_analysis'] = trend_analysis
            logger.info(f"Pipeline {execution.execution_id}: Added trend analysis")
        except Exception as e:
            logger.warning(f"Trend analysis failed: {e}")

        logger.info(f"Pipeline {execution.execution_id}: Gathered {len(execution.results)} research results")

    async def _stage_analysis(self, execution: PipelineExecution):
        """Stage 2: Analyze and score research results"""
        execution.stage = PipelineStage.ANALYSIS
        logger.info(f"Pipeline {execution.execution_id}: Starting analysis")

        for result in execution.results:
            # Calculate quality score
            result.quality_score = await self._calculate_quality_score(result)

            # Calculate neurosurgical relevance
            result.neurosurgical_relevance = await self._calculate_neurosurgical_relevance(
                result, execution.query
            )

            # Calculate overall relevance
            result.relevance_score = (result.quality_score + result.neurosurgical_relevance) / 2

        # Sort by relevance score
        execution.results.sort(key=lambda x: x.relevance_score, reverse=True)

        # Filter by quality threshold
        execution.results = [
            r for r in execution.results
            if r.relevance_score >= execution.query.quality_threshold
        ]

        logger.info(f"Pipeline {execution.execution_id}: Filtered to {len(execution.results)} high-quality results")

    async def _stage_synthesis(self, execution: PipelineExecution):
        """Stage 3: Multi-AI synthesis of knowledge"""
        execution.stage = PipelineStage.SYNTHESIS
        logger.info(f"Pipeline {execution.execution_id}: Starting synthesis")

        # Prepare source material for synthesis
        source_material = self._prepare_source_material(execution.results)

        # Use Claude for comprehensive synthesis
        synthesis_prompt = self._create_synthesis_prompt(execution.query, source_material)
        claude_synthesis = await query_ai("claude", synthesis_prompt, max_tokens=4000)

        # Use Gemini for technical analysis
        technical_prompt = self._create_technical_analysis_prompt(execution.query, source_material)
        gemini_analysis = await query_ai("gemini", technical_prompt, max_tokens=3000)

        # Combine and structure synthesis
        execution.synthesis = await self._create_knowledge_synthesis(
            execution.query, claude_synthesis, gemini_analysis, execution.results
        )

        logger.info(f"Pipeline {execution.execution_id}: Synthesis completed")

    async def _stage_quality_check(self, execution: PipelineExecution):
        """Stage 4: Quality check and validation"""
        execution.stage = PipelineStage.QUALITY_CHECK
        logger.info(f"Pipeline {execution.execution_id}: Starting quality check")

        if execution.synthesis:
            # Check for factual consistency
            consistency_score = await self._check_factual_consistency(execution.synthesis)

            # Check for completeness
            completeness_score = await self._check_completeness(execution.synthesis, execution.query)

            # Update confidence score
            execution.synthesis.confidence_score = (consistency_score + completeness_score) / 2

            # Flag any quality issues
            if execution.synthesis.confidence_score < 0.7:
                logger.warning(f"Pipeline {execution.execution_id}: Low confidence score {execution.synthesis.confidence_score}")

        logger.info(f"Pipeline {execution.execution_id}: Quality check completed")

    async def _stage_chapter_update(self, execution: PipelineExecution):
        """Stage 5: Update knowledge base chapters"""
        execution.stage = PipelineStage.CHAPTER_UPDATE
        logger.info(f"Pipeline {execution.execution_id}: Starting chapter update")

        if execution.synthesis:
            # Generate chapter update
            chapter_content = await self._generate_chapter_content(execution.synthesis)

            # Store synthesis for chapter integration
            await self._store_knowledge_synthesis(execution.synthesis, chapter_content)

        logger.info(f"Pipeline {execution.execution_id}: Chapter update completed")

    def _enhance_query_with_neurosurgical_context(self, query: ResearchQuery) -> ResearchQuery:
        """Enhance query with neurosurgical-specific terms"""
        enhanced_mesh_terms = query.mesh_terms.copy()

        # Add relevant neurosurgical MeSH terms
        for term in self.neurosurgical_mesh_terms:
            if any(focus.lower() in term.lower() for focus in query.neurosurgical_focus):
                if term not in enhanced_mesh_terms:
                    enhanced_mesh_terms.append(term)

        return ResearchQuery(
            topic=query.topic,
            neurosurgical_focus=query.neurosurgical_focus,
            mesh_terms=enhanced_mesh_terms,
            date_range=query.date_range,
            max_results=query.max_results,
            quality_threshold=query.quality_threshold,
            include_citations=query.include_citations
        )

    async def _gather_from_pubmed(self, query: ResearchQuery) -> List[ResearchResult]:
        """Gather research from PubMed"""
        try:
            search_query = f"{query.topic} AND ({' OR '.join(query.mesh_terms)})"

            results = await enhanced_pubmed_service.advanced_search(
                query=search_query,
                max_results=query.max_results,
                date_range=query.date_range,
                mesh_terms=query.mesh_terms
            )

            pubmed_results = []
            for result in results.get("papers", []):
                pubmed_results.append(ResearchResult(
                    source=ResearchSource.PUBMED,
                    title=result.get("title", ""),
                    content=result.get("abstract", ""),
                    url=result.get("url"),
                    doi=result.get("doi"),
                    authors=result.get("authors", []),
                    publication_date=result.get("publication_date"),
                    journal=result.get("journal"),
                    citations=result.get("citation_count", 0),
                    extracted_at=datetime.now()
                ))

            return pubmed_results

        except Exception as e:
            logger.error(f"Failed to gather from PubMed: {e}")
            return []

    async def _gather_from_perplexity(self, query: ResearchQuery) -> List[ResearchResult]:
        """Gather real-time research from Perplexity"""
        try:
            perplexity_query = f"Latest neurosurgical research on {query.topic}: {' '.join(query.neurosurgical_focus)}"

            response = await query_ai("perplexity", perplexity_query, max_tokens=2000)

            return [ResearchResult(
                source=ResearchSource.PERPLEXITY,
                title=f"Current Research: {query.topic}",
                content=response,
                extracted_at=datetime.now(),
                quality_score=0.8  # Default for real-time research
            )]

        except Exception as e:
            logger.error(f"Failed to gather from Perplexity: {e}")
            return []

    async def _gather_from_gemini(self, query: ResearchQuery) -> List[ResearchResult]:
        """Gather deep analysis from Gemini"""
        try:
            gemini_query = f"""Provide a comprehensive analysis of {query.topic} in neurosurgery, focusing on:
            {', '.join(query.neurosurgical_focus)}

            Include recent developments, surgical techniques, outcomes, and clinical implications."""

            response = await query_ai("gemini", gemini_query, max_tokens=3000)

            return [ResearchResult(
                source=ResearchSource.GEMINI,
                title=f"Deep Analysis: {query.topic}",
                content=response,
                extracted_at=datetime.now(),
                quality_score=0.85  # Default for AI analysis
            )]

        except Exception as e:
            logger.error(f"Failed to gather from Gemini: {e}")
            return []

    async def _calculate_quality_score(self, result: ResearchResult) -> float:
        """Calculate enhanced quality score using analytics service"""
        try:
            # Convert ResearchResult to format expected by analytics service
            article_data = {
                'title': result.title,
                'abstract': result.content,
                'journal': result.journal,
                'pub_date': result.publication_date,
                'pmid': getattr(result, 'pmid', None)
            }

            # Use enhanced quality scoring from analytics service
            enhanced_score = await self.analytics_service.calculate_quality_score(article_data)

            # Fallback to original scoring if analytics fails
            if enhanced_score is None or enhanced_score == 0:
                return await self._fallback_quality_score(result)

            return enhanced_score

        except Exception as e:
            logger.warning(f"Enhanced quality scoring failed, using fallback: {e}")
            return await self._fallback_quality_score(result)

    async def _fallback_quality_score(self, result: ResearchResult) -> float:
        """Fallback quality score calculation"""
        score = 0.0

        # Citation count factor
        if result.citations > 100:
            score += 0.3
        elif result.citations > 50:
            score += 0.2
        elif result.citations > 10:
            score += 0.1

        # Journal factor
        if result.journal:
            high_impact_journals = ["Nature", "Science", "NEJM", "Lancet", "Journal of Neurosurgery"]
            if any(journal.lower() in result.journal.lower() for journal in high_impact_journals):
                score += 0.3
            else:
                score += 0.1

        # Recency factor
        if result.publication_date:
            try:
                pub_date = datetime.strptime(result.publication_date, "%Y-%m-%d")
                days_old = (datetime.now() - pub_date).days
                if days_old < 365:
                    score += 0.2
                elif days_old < 1825:  # 5 years
                    score += 0.1
            except:
                pass

        # Content quality factor
        if len(result.content) > 500:
            score += 0.2

        return min(score, 1.0)

    async def _calculate_neurosurgical_relevance(self, result: ResearchResult, query: ResearchQuery) -> float:
        """Calculate neurosurgical relevance score"""
        relevance_prompt = f"""Rate the neurosurgical relevance of this research on a scale of 0.0 to 1.0:

        Title: {result.title}
        Content: {result.content[:1000]}...

        Focus areas: {', '.join(query.neurosurgical_focus)}

        Respond with only a decimal number between 0.0 and 1.0"""

        try:
            response = await query_ai("gemini", relevance_prompt, max_tokens=10)
            return float(response.strip())
        except:
            # Fallback: simple keyword matching
            content_lower = result.content.lower()
            title_lower = result.title.lower()

            relevance = 0.0
            for term in query.neurosurgical_focus:
                if term.lower() in content_lower or term.lower() in title_lower:
                    relevance += 0.2

            return min(relevance, 1.0)

    def _prepare_source_material(self, results: List[ResearchResult]) -> str:
        """Prepare source material for synthesis"""
        material = []

        for i, result in enumerate(results[:20], 1):  # Limit to top 20 results
            material.append(f"""
Source {i} ({result.source.value}):
Title: {result.title}
Content: {result.content[:1500]}...
Quality Score: {result.quality_score:.2f}
Neurosurgical Relevance: {result.neurosurgical_relevance:.2f}
---
""")

        return "\n".join(material)

    def _create_synthesis_prompt(self, query: ResearchQuery, source_material: str) -> str:
        """Create synthesis prompt for Claude"""
        return f"""As a neurosurgical knowledge expert, synthesize the following research into a comprehensive analysis of {query.topic}.

Focus areas: {', '.join(query.neurosurgical_focus)}

Please provide:
1. Executive summary (2-3 sentences)
2. Key findings (5-7 bullet points)
3. Clinical implications for neurosurgery
4. Relevant surgical techniques and approaches
5. Evidence quality assessment
6. Any conflicting findings or controversies
7. Research gaps and future directions
8. Clinical recommendations

Source material:
{source_material}

Structure your response as a detailed, professional analysis suitable for a neurosurgical textbook."""

    def _create_technical_analysis_prompt(self, query: ResearchQuery, source_material: str) -> str:
        """Create technical analysis prompt for Gemini"""
        return f"""Provide a technical analysis of the research on {query.topic} with focus on {', '.join(query.neurosurgical_focus)}.

Analyze:
1. Methodological quality of studies
2. Statistical significance and effect sizes
3. Surgical technique variations and outcomes
4. Technology and instrumentation advances
5. Complications and risk factors
6. Long-term follow-up data
7. Cost-effectiveness considerations

Source material:
{source_material[:3000]}

Provide a detailed technical assessment focusing on surgical and clinical aspects."""

    async def _create_knowledge_synthesis(
        self,
        query: ResearchQuery,
        claude_synthesis: str,
        gemini_analysis: str,
        results: List[ResearchResult]
    ) -> KnowledgeSynthesis:
        """Create structured knowledge synthesis"""

        # Parse Claude's synthesis (simplified parsing)
        sections = claude_synthesis.split('\n\n')

        return KnowledgeSynthesis(
            topic=query.topic,
            executive_summary=self._extract_section(claude_synthesis, "executive summary", "Executive Summary"),
            key_findings=self._extract_list(claude_synthesis, "key findings", "Key Findings"),
            clinical_implications=self._extract_list(claude_synthesis, "clinical implications", "Clinical Implications"),
            surgical_techniques=self._extract_list(gemini_analysis, "surgical technique", "Surgical Techniques"),
            evidence_quality=self._extract_section(claude_synthesis, "evidence quality", "Evidence Quality"),
            conflicting_findings=self._extract_list(claude_synthesis, "conflicting", "Conflicting"),
            research_gaps=self._extract_list(claude_synthesis, "research gaps", "Research Gaps"),
            recommendations=self._extract_list(claude_synthesis, "recommendations", "Recommendations"),
            sources_used=[r.title for r in results[:10]],
            confidence_score=0.8,  # Will be updated in quality check
            last_updated=datetime.now()
        )

    def _extract_section(self, text: str, keyword: str, fallback: str) -> str:
        """Extract a section from synthesized text"""
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if keyword.lower() in line.lower():
                # Get next few lines
                return ' '.join(lines[i+1:i+4]).strip()
        return f"{fallback} analysis from synthesis"

    def _extract_list(self, text: str, keyword: str, fallback: str) -> List[str]:
        """Extract a list from synthesized text"""
        lines = text.split('\n')
        items = []
        capturing = False

        for line in lines:
            if keyword.lower() in line.lower():
                capturing = True
                continue
            if capturing:
                if line.strip().startswith(('•', '-', '*', '1.', '2.')):
                    items.append(line.strip().lstrip('•-*0123456789. '))
                elif line.strip() == '' or len(items) >= 5:
                    break

        return items if items else [f"{fallback} from research synthesis"]

    async def _check_factual_consistency(self, synthesis: KnowledgeSynthesis) -> float:
        """Check factual consistency of synthesis"""
        consistency_prompt = f"""Review this neurosurgical knowledge synthesis for factual accuracy and consistency.
        Rate the factual consistency on a scale of 0.0 to 1.0:

        Topic: {synthesis.topic}
        Summary: {synthesis.executive_summary}
        Key Findings: {'; '.join(synthesis.key_findings[:3])}

        Respond with only a decimal number between 0.0 and 1.0"""

        try:
            response = await query_ai("claude", consistency_prompt, max_tokens=10)
            return float(response.strip())
        except:
            return 0.7  # Default consistency score

    async def _check_completeness(self, synthesis: KnowledgeSynthesis, query: ResearchQuery) -> float:
        """Check completeness of synthesis"""
        completeness_score = 0.0

        # Check if key components are present
        if synthesis.executive_summary and len(synthesis.executive_summary) > 50:
            completeness_score += 0.2
        if len(synthesis.key_findings) >= 3:
            completeness_score += 0.2
        if len(synthesis.clinical_implications) >= 2:
            completeness_score += 0.2
        if len(synthesis.surgical_techniques) >= 1:
            completeness_score += 0.2
        if len(synthesis.recommendations) >= 2:
            completeness_score += 0.2

        return completeness_score

    async def _generate_chapter_content(self, synthesis: KnowledgeSynthesis) -> str:
        """Generate formatted chapter content"""
        chapter_prompt = f"""Convert this research synthesis into a well-structured textbook chapter:

        Topic: {synthesis.topic}
        Executive Summary: {synthesis.executive_summary}
        Key Findings: {'; '.join(synthesis.key_findings)}
        Clinical Implications: {'; '.join(synthesis.clinical_implications)}
        Surgical Techniques: {'; '.join(synthesis.surgical_techniques)}
        Recommendations: {'; '.join(synthesis.recommendations)}

        Format as a professional medical textbook chapter with:
        - Clear headings and subheadings
        - Proper medical terminology
        - Clinical pearls and key points
        - Evidence-based recommendations
        - Appropriate academic tone

        Keep it comprehensive but concise."""

        try:
            return await query_ai("claude", chapter_prompt, max_tokens=4000)
        except Exception as e:
            logger.error(f"Failed to generate chapter content: {e}")
            return f"# {synthesis.topic}\n\n{synthesis.executive_summary}\n\n## Key Findings\n\n" + \
                   "\n".join(f"- {finding}" for finding in synthesis.key_findings)

    async def _store_knowledge_synthesis(self, synthesis: KnowledgeSynthesis, chapter_content: str):
        """Store synthesis and chapter content"""
        # This would integrate with your database/storage system
        # For now, we'll log the completion
        logger.info(f"Stored knowledge synthesis for topic: {synthesis.topic}")
        logger.info(f"Chapter content length: {len(chapter_content)} characters")

    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get status of pipeline execution"""
        # Check active executions
        if execution_id in self.active_executions:
            execution = self.active_executions[execution_id]
            return {
                "execution_id": execution_id,
                "stage": execution.stage.value,
                "results_count": len(execution.results),
                "started_at": execution.started_at.isoformat(),
                "is_active": True
            }

        # Check completed executions
        for execution in self.completed_executions:
            if execution.execution_id == execution_id:
                return {
                    "execution_id": execution_id,
                    "stage": execution.stage.value,
                    "results_count": len(execution.results),
                    "started_at": execution.started_at.isoformat(),
                    "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                    "is_active": False,
                    "success": execution.stage == PipelineStage.COMPLETED,
                    "error": execution.error_message,
                    "synthesis": asdict(execution.synthesis) if execution.synthesis else None
                }

        return None

    def get_active_executions(self) -> List[Dict[str, Any]]:
        """Get all active pipeline executions"""
        return [
            {
                "execution_id": exec_id,
                "topic": execution.query.topic,
                "stage": execution.stage.value,
                "started_at": execution.started_at.isoformat(),
                "results_count": len(execution.results)
            }
            for exec_id, execution in self.active_executions.items()
        ]

# Global instance
knowledge_pipeline = KnowledgeIntegrationPipeline()

# Convenience functions
async def start_research_pipeline(
    topic: str,
    neurosurgical_focus: List[str],
    mesh_terms: List[str] = None,
    max_results: int = 50
) -> str:
    """Start a research pipeline execution"""
    query = ResearchQuery(
        topic=topic,
        neurosurgical_focus=neurosurgical_focus,
        mesh_terms=mesh_terms or [],
        max_results=max_results
    )

    return await knowledge_pipeline.execute_research_pipeline(query)

async def get_pipeline_status(execution_id: str) -> Optional[Dict[str, Any]]:
    """Get pipeline execution status"""
    return knowledge_pipeline.get_execution_status(execution_id)