"""
Literature Summarization Engine
AI-powered summarization of medical literature with intelligent synthesis
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import re
from collections import defaultdict, Counter

from .hybrid_ai_manager import query_ai
from .advanced_pubmed_analytics import AdvancedPubMedAnalytics
from ..core.config import settings
from ..core.exceptions import ExternalServiceError

logger = logging.getLogger(__name__)

class SummaryType(Enum):
    """Types of literature summaries"""
    EXECUTIVE = "executive"  # Brief overview (200-300 words)
    COMPREHENSIVE = "comprehensive"  # Detailed analysis (800-1200 words)
    TECHNICAL = "technical"  # Technical/clinical focus
    PATIENT_FRIENDLY = "patient_friendly"  # Lay language summary
    SYSTEMATIC_REVIEW = "systematic_review"  # Systematic review format
    META_ANALYSIS = "meta_analysis"  # Meta-analysis format

class EvidenceLevel(Enum):
    """Evidence levels for medical literature"""
    LEVEL_1A = "1a"  # Systematic review of RCTs
    LEVEL_1B = "1b"  # Individual RCT
    LEVEL_2A = "2a"  # Systematic review of cohort studies
    LEVEL_2B = "2b"  # Individual cohort study
    LEVEL_3A = "3a"  # Systematic review of case-control studies
    LEVEL_3B = "3b"  # Individual case-control study
    LEVEL_4 = "4"    # Case series
    LEVEL_5 = "5"    # Expert opinion

@dataclass
class SummaryRequest:
    """Request for literature summarization"""
    topic: str
    summary_type: SummaryType
    max_papers: int = 50
    years_back: int = 5
    specialty_focus: List[str] = None
    include_preprints: bool = False
    quality_threshold: float = 0.6
    target_audience: str = "clinicians"
    custom_focus: Optional[List[str]] = None

@dataclass
class PaperSummary:
    """Individual paper summary"""
    pmid: str
    title: str
    authors: List[str]
    journal: str
    year: int
    study_type: str
    evidence_level: EvidenceLevel
    sample_size: Optional[int]
    key_findings: List[str]
    clinical_significance: str
    limitations: List[str]
    ai_summary: str
    quality_score: float
    relevance_score: float

@dataclass
class LiteratureSynthesis:
    """Synthesized literature summary"""
    summary_id: str
    topic: str
    summary_type: SummaryType
    executive_summary: str
    detailed_analysis: str
    key_findings: List[str]
    evidence_synthesis: Dict[str, Any]
    clinical_implications: List[str]
    research_gaps: List[str]
    recommendations: List[str]
    papers_included: List[PaperSummary]
    evidence_quality: Dict[str, int]
    conflicts_identified: List[str]
    future_directions: List[str]
    generated_at: datetime
    confidence_score: float

class LiteratureSummarizationEngine:
    """Advanced literature summarization with AI synthesis"""

    def __init__(self):
        self.analytics_service = AdvancedPubMedAnalytics()

        # Study type classification patterns
        self.study_type_patterns = {
            "randomized_controlled_trial": [
                "randomized", "randomised", "rct", "placebo-controlled", "double-blind", "clinical trial"
            ],
            "systematic_review": [
                "systematic review", "meta-analysis", "cochrane", "prisma"
            ],
            "cohort_study": [
                "cohort", "longitudinal", "prospective", "follow-up"
            ],
            "case_control": [
                "case-control", "case control", "retrospective"
            ],
            "case_series": [
                "case series", "case report", "case study"
            ],
            "cross_sectional": [
                "cross-sectional", "cross sectional", "survey"
            ]
        }

        # AI prompts for different summarization tasks
        self.summarization_prompts = {
            "paper_summary": """Summarize this medical research paper for clinical practitioners:

Title: {title}
Abstract: {abstract}
Authors: {authors}
Journal: {journal}

Please provide:
1. Study type and design
2. Sample size and population
3. Main findings (3-5 key points)
4. Clinical significance
5. Study limitations
6. Evidence level assessment

Write in clear, clinical language focusing on practical implications.""",

            "evidence_synthesis": """Synthesize the evidence from these medical studies on {topic}:

{paper_summaries}

Analyze:
1. Consistency of findings across studies
2. Quality of evidence
3. Clinical significance
4. Conflicting results and explanations
5. Gaps in current evidence
6. Strength of recommendations possible

Provide a balanced, evidence-based synthesis suitable for clinical decision-making.""",

            "clinical_implications": """Based on this literature synthesis on {topic}, provide clinical implications:

Evidence Summary:
{evidence_summary}

Please identify:
1. Direct clinical applications
2. Changes to current practice suggested
3. Patient population considerations
4. Risk-benefit assessments
5. Implementation considerations
6. Monitoring requirements

Focus on actionable insights for practicing clinicians.""",

            "research_gaps": """Identify research gaps and future directions from this literature on {topic}:

Current Evidence:
{current_evidence}

Included Studies:
{study_list}

Please identify:
1. Areas with insufficient evidence
2. Methodological improvements needed
3. Population groups understudied
4. Outcome measures lacking
5. Technology/technique innovations needed
6. Collaborative research opportunities

Prioritize gaps by clinical importance and feasibility."""
        }

    async def summarize_literature(self, request: SummaryRequest) -> LiteratureSynthesis:
        """Generate comprehensive literature summary with AI synthesis"""
        logger.info(f"Starting literature summarization for: {request.topic}")

        try:
            # Step 1: Gather relevant papers
            papers = await self._gather_literature(request)

            # Step 2: Analyze individual papers
            paper_summaries = await self._analyze_individual_papers(papers, request)

            # Step 3: Synthesize evidence
            evidence_synthesis = await self._synthesize_evidence(paper_summaries, request)

            # Step 4: Generate clinical implications
            clinical_implications = await self._generate_clinical_implications(evidence_synthesis, request)

            # Step 5: Identify research gaps
            research_gaps = await self._identify_research_gaps(paper_summaries, request)

            # Step 6: Create final synthesis
            synthesis = await self._create_final_synthesis(
                request, paper_summaries, evidence_synthesis,
                clinical_implications, research_gaps
            )

            logger.info(f"Literature summarization completed for {len(paper_summaries)} papers")
            return synthesis

        except Exception as e:
            logger.error(f"Literature summarization failed: {e}")
            raise ExternalServiceError("summarization", f"Failed to summarize literature: {str(e)}")

    async def _gather_literature(self, request: SummaryRequest) -> List[Any]:
        """Gather relevant literature for summarization"""
        logger.info("Gathering literature...")

        try:
            # Use our enhanced PubMed search
            papers = await self.analytics_service.pubmed_service.neurosurgical_search(
                topic=request.topic,
                max_results=request.max_papers,
                years_back=request.years_back
            )

            # Filter by quality threshold if available
            if hasattr(papers[0], 'quality_score'):
                papers = [p for p in papers if getattr(p, 'quality_score', 0) >= request.quality_threshold]

            logger.info(f"Gathered {len(papers)} papers for analysis")
            return papers

        except Exception as e:
            logger.error(f"Literature gathering failed: {e}")
            return []

    async def _analyze_individual_papers(self, papers: List[Any], request: SummaryRequest) -> List[PaperSummary]:
        """Analyze each paper individually using AI"""
        logger.info(f"Analyzing {len(papers)} individual papers...")

        paper_summaries = []

        for i, paper in enumerate(papers):
            try:
                logger.info(f"Analyzing paper {i+1}/{len(papers)}: {paper.title[:50]}...")

                # Extract paper information
                abstract = getattr(paper, 'abstract', '')
                title = getattr(paper, 'title', '')
                authors = getattr(paper, 'authors', [])
                journal = getattr(paper, 'journal', '')
                pmid = getattr(paper, 'pmid', str(i))

                # Determine study type
                study_type = self._classify_study_type(title, abstract)

                # Determine evidence level
                evidence_level = self._determine_evidence_level(study_type, title, abstract)

                # Extract sample size
                sample_size = self._extract_sample_size(abstract)

                # Generate AI summary
                summary_prompt = self.summarization_prompts["paper_summary"].format(
                    title=title,
                    abstract=abstract[:1500],  # Limit abstract length
                    authors=", ".join(authors[:3]),
                    journal=journal
                )

                ai_summary = await query_ai("claude", summary_prompt, max_tokens=800)

                # Extract key findings from AI summary
                key_findings = self._extract_key_findings(ai_summary)

                # Calculate relevance score
                relevance_score = self._calculate_relevance_score(title, abstract, request.topic)

                paper_summary = PaperSummary(
                    pmid=pmid,
                    title=title,
                    authors=authors,
                    journal=journal,
                    year=self._extract_year(getattr(paper, 'publication_date', '')),
                    study_type=study_type,
                    evidence_level=evidence_level,
                    sample_size=sample_size,
                    key_findings=key_findings,
                    clinical_significance=self._extract_clinical_significance(ai_summary),
                    limitations=self._extract_limitations(ai_summary),
                    ai_summary=ai_summary,
                    quality_score=getattr(paper, 'quality_score', 0.5),
                    relevance_score=relevance_score
                )

                paper_summaries.append(paper_summary)

                # Brief pause to avoid rate limiting
                if i % 5 == 4:  # Pause every 5 papers
                    await asyncio.sleep(2)

            except Exception as e:
                logger.error(f"Failed to analyze paper {i}: {e}")
                continue

        logger.info(f"Completed analysis of {len(paper_summaries)} papers")
        return paper_summaries

    async def _synthesize_evidence(self, paper_summaries: List[PaperSummary], request: SummaryRequest) -> Dict[str, Any]:
        """Synthesize evidence across all papers"""
        logger.info("Synthesizing evidence across papers...")

        try:
            # Prepare summaries for AI synthesis
            paper_summaries_text = "\n\n".join([
                f"Study {i+1}: {summary.title}\n"
                f"Type: {summary.study_type}, Evidence Level: {summary.evidence_level.value}\n"
                f"Key Findings: {'; '.join(summary.key_findings)}\n"
                f"Clinical Significance: {summary.clinical_significance}"
                for i, summary in enumerate(paper_summaries[:15])  # Limit to top 15 papers
            ])

            synthesis_prompt = self.summarization_prompts["evidence_synthesis"].format(
                topic=request.topic,
                paper_summaries=paper_summaries_text
            )

            synthesis_response = await query_ai("claude", synthesis_prompt, max_tokens=2000)

            # Analyze evidence quality distribution
            evidence_quality = self._analyze_evidence_quality(paper_summaries)

            # Identify conflicts
            conflicts = self._identify_conflicts(paper_summaries)

            synthesis = {
                "ai_synthesis": synthesis_response,
                "evidence_quality": evidence_quality,
                "conflicts": conflicts,
                "total_papers": len(paper_summaries),
                "high_quality_papers": len([p for p in paper_summaries if p.quality_score > 0.7]),
                "recent_papers": len([p for p in paper_summaries if p.year >= datetime.now().year - 2])
            }

            return synthesis

        except Exception as e:
            logger.error(f"Evidence synthesis failed: {e}")
            return {"error": "Failed to synthesize evidence"}

    async def _generate_clinical_implications(self, evidence_synthesis: Dict[str, Any], request: SummaryRequest) -> List[str]:
        """Generate clinical implications from evidence synthesis"""
        logger.info("Generating clinical implications...")

        try:
            implications_prompt = self.summarization_prompts["clinical_implications"].format(
                topic=request.topic,
                evidence_summary=evidence_synthesis.get("ai_synthesis", "")[:1500]
            )

            response = await query_ai("gemini", implications_prompt, max_tokens=1000)

            # Extract implications from response
            implications = self._extract_list_items(response, "implications")

            return implications

        except Exception as e:
            logger.error(f"Clinical implications generation failed: {e}")
            return ["Clinical implications analysis pending"]

    async def _identify_research_gaps(self, paper_summaries: List[PaperSummary], request: SummaryRequest) -> List[str]:
        """Identify research gaps from literature analysis"""
        logger.info("Identifying research gaps...")

        try:
            # Prepare study list
            study_list = "\n".join([
                f"- {summary.title} ({summary.study_type}, n={summary.sample_size or 'NR'})"
                for summary in paper_summaries[:10]
            ])

            current_evidence = "\n".join([
                f"- {finding}" for summary in paper_summaries[:5]
                for finding in summary.key_findings
            ])

            gaps_prompt = self.summarization_prompts["research_gaps"].format(
                topic=request.topic,
                current_evidence=current_evidence,
                study_list=study_list
            )

            response = await query_ai("claude", gaps_prompt, max_tokens=1000)

            # Extract research gaps from response
            gaps = self._extract_list_items(response, "gaps")

            return gaps

        except Exception as e:
            logger.error(f"Research gaps identification failed: {e}")
            return ["Research gap analysis pending"]

    async def _create_final_synthesis(self, request: SummaryRequest, paper_summaries: List[PaperSummary],
                                    evidence_synthesis: Dict[str, Any], clinical_implications: List[str],
                                    research_gaps: List[str]) -> LiteratureSynthesis:
        """Create final literature synthesis"""
        logger.info("Creating final synthesis...")

        # Generate executive summary
        executive_summary = await self._generate_executive_summary(request, evidence_synthesis)

        # Generate detailed analysis
        detailed_analysis = await self._generate_detailed_analysis(request, evidence_synthesis, paper_summaries)

        # Extract key findings
        key_findings = self._extract_overall_key_findings(paper_summaries)

        # Generate recommendations
        recommendations = await self._generate_recommendations(evidence_synthesis, clinical_implications)

        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(paper_summaries, evidence_synthesis)

        synthesis = LiteratureSynthesis(
            summary_id=f"lit_sum_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            topic=request.topic,
            summary_type=request.summary_type,
            executive_summary=executive_summary,
            detailed_analysis=detailed_analysis,
            key_findings=key_findings,
            evidence_synthesis=evidence_synthesis,
            clinical_implications=clinical_implications,
            research_gaps=research_gaps,
            recommendations=recommendations,
            papers_included=paper_summaries,
            evidence_quality=evidence_synthesis.get("evidence_quality", {}),
            conflicts_identified=evidence_synthesis.get("conflicts", []),
            future_directions=research_gaps,
            generated_at=datetime.now(),
            confidence_score=confidence_score
        )

        return synthesis

    def _classify_study_type(self, title: str, abstract: str) -> str:
        """Classify study type based on title and abstract"""
        text = (title + " " + abstract).lower()

        for study_type, patterns in self.study_type_patterns.items():
            if any(pattern in text for pattern in patterns):
                return study_type

        return "observational_study"

    def _determine_evidence_level(self, study_type: str, title: str, abstract: str) -> EvidenceLevel:
        """Determine evidence level based on study characteristics"""
        text = (title + " " + abstract).lower()

        if study_type == "systematic_review":
            if "randomized" in text or "rct" in text:
                return EvidenceLevel.LEVEL_1A
            else:
                return EvidenceLevel.LEVEL_2A

        elif study_type == "randomized_controlled_trial":
            return EvidenceLevel.LEVEL_1B

        elif study_type == "cohort_study":
            return EvidenceLevel.LEVEL_2B

        elif study_type == "case_control":
            return EvidenceLevel.LEVEL_3B

        elif study_type == "case_series":
            return EvidenceLevel.LEVEL_4

        else:
            return EvidenceLevel.LEVEL_5

    def _extract_sample_size(self, abstract: str) -> Optional[int]:
        """Extract sample size from abstract"""
        try:
            # Look for common sample size patterns
            patterns = [
                r'(\d+)\s+patients?',
                r'(\d+)\s+subjects?',
                r'(\d+)\s+participants?',
                r'n\s*=\s*(\d+)',
                r'sample.*?(\d+)'
            ]

            for pattern in patterns:
                match = re.search(pattern, abstract.lower())
                if match:
                    return int(match.group(1))

            return None

        except Exception:
            return None

    def _extract_key_findings(self, ai_summary: str) -> List[str]:
        """Extract key findings from AI summary"""
        try:
            # Look for numbered lists or bullet points
            findings = []

            # Pattern for numbered findings
            numbered_findings = re.findall(r'\d+\.\s*([^.]+(?:\.[^.]*)?)', ai_summary)
            if numbered_findings:
                findings.extend(numbered_findings[:5])

            # Pattern for bullet points
            bullet_findings = re.findall(r'[-•]\s*([^.]+(?:\.[^.]*)?)', ai_summary)
            if bullet_findings:
                findings.extend(bullet_findings[:3])

            # If no structured findings, extract sentences with keywords
            if not findings:
                key_sentences = re.findall(r'[^.]*(?:found|showed|demonstrated|indicated|concluded)[^.]*\.', ai_summary)
                findings.extend(key_sentences[:3])

            return findings[:5] if findings else ["Key findings analysis pending"]

        except Exception:
            return ["Key findings analysis pending"]

    def _extract_clinical_significance(self, ai_summary: str) -> str:
        """Extract clinical significance from AI summary"""
        try:
            # Look for sections about clinical significance
            significance_patterns = [
                r'clinical[ly]?\s+significant?[ly]?\s*:?\s*([^.]+)',
                r'significance[^.]*([^.]+)',
                r'implications?[^.]*([^.]+)'
            ]

            for pattern in significance_patterns:
                match = re.search(pattern, ai_summary.lower())
                if match:
                    return match.group(1).strip()

            # Fallback: look for conclusion sentences
            conclusion_match = re.search(r'conclusion[s]?[^.]*([^.]+)', ai_summary.lower())
            if conclusion_match:
                return conclusion_match.group(1).strip()

            return "Clinical significance analysis pending"

        except Exception:
            return "Clinical significance analysis pending"

    def _extract_limitations(self, ai_summary: str) -> List[str]:
        """Extract study limitations from AI summary"""
        try:
            limitations = []

            # Look for limitation sections
            limitation_patterns = [
                r'limitations?[^.]*:([^.]+)',
                r'limited by([^.]+)',
                r'weakness(?:es)?[^.]*([^.]+)'
            ]

            for pattern in limitation_patterns:
                matches = re.findall(pattern, ai_summary.lower())
                limitations.extend(matches)

            return limitations[:3] if limitations else ["Study limitations analysis pending"]

        except Exception:
            return ["Study limitations analysis pending"]

    def _extract_year(self, date_string: str) -> int:
        """Extract year from date string"""
        try:
            if date_string:
                year_match = re.search(r'(\d{4})', date_string)
                if year_match:
                    return int(year_match.group(1))
            return datetime.now().year
        except Exception:
            return datetime.now().year

    def _calculate_relevance_score(self, title: str, abstract: str, topic: str) -> float:
        """Calculate relevance score of paper to topic"""
        try:
            text = (title + " " + abstract).lower()
            topic_words = topic.lower().split()

            # Calculate word overlap
            relevance = 0.0
            for word in topic_words:
                if word in text:
                    relevance += 0.2

            # Title relevance has higher weight
            title_relevance = 0.0
            for word in topic_words:
                if word in title.lower():
                    title_relevance += 0.3

            return min(1.0, relevance + title_relevance)

        except Exception:
            return 0.5

    def _analyze_evidence_quality(self, paper_summaries: List[PaperSummary]) -> Dict[str, int]:
        """Analyze distribution of evidence quality"""
        quality_dist = {
            "Level 1": 0,  # Systematic reviews and RCTs
            "Level 2": 0,  # Cohort studies
            "Level 3": 0,  # Case-control studies
            "Level 4": 0,  # Case series
            "Level 5": 0   # Expert opinion
        }

        for summary in paper_summaries:
            level = summary.evidence_level.value
            if level in ["1a", "1b"]:
                quality_dist["Level 1"] += 1
            elif level in ["2a", "2b"]:
                quality_dist["Level 2"] += 1
            elif level in ["3a", "3b"]:
                quality_dist["Level 3"] += 1
            elif level == "4":
                quality_dist["Level 4"] += 1
            else:
                quality_dist["Level 5"] += 1

        return quality_dist

    def _identify_conflicts(self, paper_summaries: List[PaperSummary]) -> List[str]:
        """Identify conflicting findings in the literature"""
        conflicts = []

        # This is a simplified conflict detection
        # In practice, would use more sophisticated NLP analysis
        positive_findings = []
        negative_findings = []

        for summary in paper_summaries:
            for finding in summary.key_findings:
                finding_lower = finding.lower()
                if any(word in finding_lower for word in ["effective", "beneficial", "improved", "significant improvement"]):
                    positive_findings.append(finding)
                elif any(word in finding_lower for word in ["ineffective", "no benefit", "no difference", "not significant"]):
                    negative_findings.append(finding)

        if positive_findings and negative_findings:
            conflicts.append("Conflicting findings regarding treatment effectiveness")

        return conflicts

    def _extract_list_items(self, text: str, item_type: str) -> List[str]:
        """Extract list items from AI response"""
        try:
            items = []

            # Look for numbered lists
            numbered_items = re.findall(r'\d+\.\s*([^.]+(?:\.[^.]*)?)', text)
            items.extend(numbered_items)

            # Look for bullet points
            bullet_items = re.findall(r'[-•]\s*([^.]+(?:\.[^.]*)?)', text)
            items.extend(bullet_items)

            return items[:5] if items else [f"{item_type.title()} analysis pending"]

        except Exception:
            return [f"{item_type.title()} analysis pending"]

    async def _generate_executive_summary(self, request: SummaryRequest, evidence_synthesis: Dict[str, Any]) -> str:
        """Generate executive summary"""
        try:
            prompt = f"""Create a concise executive summary for literature on {request.topic}:

Evidence Synthesis:
{evidence_synthesis.get('ai_synthesis', '')[:1000]}

Requirements:
- 200-300 words maximum
- Focus on key clinical findings
- Include strength of evidence
- Mention any conflicts or limitations

Write for {request.target_audience}."""

            summary = await query_ai("claude", prompt, max_tokens=500)
            return summary

        except Exception as e:
            logger.error(f"Executive summary generation failed: {e}")
            return f"Executive summary of current literature on {request.topic} is being processed."

    async def _generate_detailed_analysis(self, request: SummaryRequest, evidence_synthesis: Dict[str, Any], paper_summaries: List[PaperSummary]) -> str:
        """Generate detailed analysis"""
        try:
            analysis_sections = [
                f"# Literature Analysis: {request.topic}\n",
                f"## Evidence Overview\n{evidence_synthesis.get('ai_synthesis', '')}\n",
                f"## Study Characteristics\n",
                f"- Total studies analyzed: {len(paper_summaries)}\n",
                f"- High-quality studies: {len([p for p in paper_summaries if p.quality_score > 0.7])}\n",
                f"- Recent studies (last 2 years): {len([p for p in paper_summaries if p.year >= datetime.now().year - 2])}\n"
            ]

            return "\n".join(analysis_sections)

        except Exception:
            return f"Detailed analysis of literature on {request.topic}"

    def _extract_overall_key_findings(self, paper_summaries: List[PaperSummary]) -> List[str]:
        """Extract overall key findings from all papers"""
        all_findings = []
        for summary in paper_summaries:
            all_findings.extend(summary.key_findings)

        # Remove duplicates and return top findings
        unique_findings = list(set(all_findings))
        return unique_findings[:10]

    async def _generate_recommendations(self, evidence_synthesis: Dict[str, Any], clinical_implications: List[str]) -> List[str]:
        """Generate clinical recommendations"""
        try:
            recommendations = [
                "Follow evidence-based treatment protocols",
                "Consider patient-specific factors in treatment decisions",
                "Monitor for reported adverse effects",
                "Stay updated with emerging research"
            ]

            # Add specific recommendations based on implications
            recommendations.extend(clinical_implications[:3])

            return recommendations

        except Exception:
            return ["Clinical recommendations pending detailed analysis"]

    def _calculate_confidence_score(self, paper_summaries: List[PaperSummary], evidence_synthesis: Dict[str, Any]) -> float:
        """Calculate overall confidence score for the synthesis"""
        try:
            factors = []

            # Number of high-quality studies
            high_quality = len([p for p in paper_summaries if p.quality_score > 0.7])
            quality_score = min(1.0, high_quality / 10) * 0.3
            factors.append(quality_score)

            # Evidence level distribution
            level_1_studies = len([p for p in paper_summaries if p.evidence_level.value in ["1a", "1b"]])
            evidence_score = min(1.0, level_1_studies / 5) * 0.3
            factors.append(evidence_score)

            # Consistency (inverse of conflicts)
            conflicts = len(evidence_synthesis.get("conflicts", []))
            consistency_score = max(0.0, 1.0 - (conflicts / 5)) * 0.2
            factors.append(consistency_score)

            # Sample size adequacy
            total_sample = sum([p.sample_size or 0 for p in paper_summaries])
            sample_score = min(1.0, total_sample / 1000) * 0.2
            factors.append(sample_score)

            return round(sum(factors), 2)

        except Exception:
            return 0.5

# Global instance
literature_summarizer = LiteratureSummarizationEngine()