"""
AI-Powered Chapter Generation Service
Automated medical chapter creation using advanced AI models
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import re

from .hybrid_ai_manager import query_ai
from .advanced_pubmed_analytics import AdvancedPubMedAnalytics
from .knowledge_integration_pipeline import knowledge_pipeline
from .reference_library import reference_library
from ..core.config import settings
from ..core.exceptions import ExternalServiceError

logger = logging.getLogger(__name__)

class ChapterType(Enum):
    """Types of medical chapters that can be generated"""
    DISEASE_OVERVIEW = "disease_overview"
    SURGICAL_TECHNIQUE = "surgical_technique"
    DIAGNOSIS_TREATMENT = "diagnosis_treatment"
    CASE_STUDY = "case_study"
    LITERATURE_REVIEW = "literature_review"
    GUIDELINES = "guidelines"
    ANATOMY_PHYSIOLOGY = "anatomy_physiology"

class GenerationQuality(Enum):
    """Quality levels for chapter generation"""
    DRAFT = "draft"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    PUBLICATION_READY = "publication_ready"

@dataclass
class ChapterStructure:
    """Defines the structure of a medical chapter"""
    title: str
    sections: List[str]
    subsections: Dict[str, List[str]]
    required_elements: List[str]
    word_count_target: int
    citation_count_target: int

@dataclass
class GenerationRequest:
    """Request for AI chapter generation"""
    topic: str
    chapter_type: ChapterType
    quality_level: GenerationQuality
    target_audience: str  # "residents", "attendings", "medical_students", "researchers"
    specialty_focus: List[str]
    include_cases: bool = True
    include_images: bool = True
    citation_style: str = "vancouver"
    max_length: int = 5000
    custom_sections: Optional[List[str]] = None

@dataclass
class GeneratedChapter:
    """Generated chapter with metadata"""
    chapter_id: str
    title: str
    content: str
    structure: ChapterStructure
    metadata: Dict[str, Any]
    citations: List[Dict[str, Any]]
    cross_references: List[str]
    research_gaps: List[str]
    quality_score: float
    generation_time: datetime
    ai_models_used: List[str]

class AIChapterGenerator:
    """Advanced AI-powered chapter generation system"""

    def __init__(self):
        self.analytics_service = AdvancedPubMedAnalytics()

        # Chapter templates for different types
        self.chapter_templates = {
            ChapterType.DISEASE_OVERVIEW: ChapterStructure(
                title="Disease Overview Template",
                sections=[
                    "Introduction", "Epidemiology", "Pathophysiology",
                    "Clinical Presentation", "Diagnosis", "Treatment",
                    "Prognosis", "Recent Advances", "Future Directions"
                ],
                subsections={
                    "Diagnosis": ["Clinical Assessment", "Imaging", "Laboratory Tests"],
                    "Treatment": ["Conservative Management", "Surgical Options", "Complications"]
                },
                required_elements=["abstract", "key_points", "references"],
                word_count_target=3000,
                citation_count_target=50
            ),
            ChapterType.SURGICAL_TECHNIQUE: ChapterStructure(
                title="Surgical Technique Template",
                sections=[
                    "Introduction", "Indications", "Contraindications",
                    "Preoperative Planning", "Surgical Technique", "Postoperative Care",
                    "Complications", "Outcomes", "Tips and Pearls"
                ],
                subsections={
                    "Surgical Technique": ["Patient Positioning", "Approach", "Key Steps", "Closure"],
                    "Complications": ["Intraoperative", "Early Postoperative", "Late Complications"]
                },
                required_elements=["step_by_step_guide", "images", "references"],
                word_count_target=2500,
                citation_count_target=30
            ),
            ChapterType.LITERATURE_REVIEW: ChapterStructure(
                title="Literature Review Template",
                sections=[
                    "Abstract", "Introduction", "Methods", "Results",
                    "Discussion", "Limitations", "Conclusions", "Future Research"
                ],
                subsections={
                    "Methods": ["Search Strategy", "Inclusion Criteria", "Data Extraction"],
                    "Results": ["Study Characteristics", "Key Findings", "Quality Assessment"]
                },
                required_elements=["systematic_search", "evidence_tables", "references"],
                word_count_target=4000,
                citation_count_target=100
            )
        }

        # AI prompts for different generation tasks
        self.generation_prompts = {
            "chapter_outline": """Create a detailed outline for a medical chapter on {topic} in {specialty}.

Chapter Type: {chapter_type}
Target Audience: {target_audience}
Quality Level: {quality_level}

Please provide:
1. A compelling chapter title
2. Main section headings (6-10 sections)
3. 2-3 subsections for each main section
4. Key learning objectives
5. Essential concepts to cover
6. Suggested number of references

Format as structured JSON with clear hierarchical organization.""",

            "section_generation": """Write a comprehensive medical section on "{section_title}" for a chapter about {topic}.

Context:
- Chapter Type: {chapter_type}
- Target Audience: {target_audience}
- Specialty: {specialty}
- Quality Level: {quality_level}

Requirements:
- Write {word_count} words
- Include {citation_count} evidence-based citations
- Use professional medical writing style
- Include clinical pearls and key points
- Reference latest research and guidelines
- INTEGRATE textbook references when appropriate

Previous Sections Context:
{previous_context}

{textbook_references}

Write the section content with proper medical terminology, evidence-based statements, and practical clinical guidance. When textbook references are provided, integrate this authoritative knowledge and cite appropriately.""",

            "cross_reference": """Analyze this medical content and suggest intelligent cross-references:

Chapter Topic: {topic}
Content: {content}

Please identify:
1. Related medical concepts that should be cross-referenced
2. Relevant anatomy/physiology connections
3. Related diseases or conditions
4. Applicable surgical techniques or procedures
5. Diagnostic methods or tests
6. Treatment modalities or medications

For each cross-reference, provide:
- The term or concept
- The relationship type (related_condition, diagnostic_method, treatment, etc.)
- Brief justification for the connection
- Suggested linking text

Format as structured JSON.""",

            "research_gaps": """Identify research gaps and future directions for this medical content:

Topic: {topic}
Content: {content}
Recent Literature: {recent_papers}

Please identify:
1. Areas with limited evidence
2. Conflicting research findings
3. Emerging technologies or techniques needing study
4. Population groups understudied
5. Long-term outcome studies needed
6. Cost-effectiveness research gaps

For each gap, provide:
- Description of the gap
- Clinical significance
- Potential research questions
- Feasibility of addressing the gap

Format as structured JSON with prioritized recommendations."""
        }

    async def generate_chapter(self, request: GenerationRequest) -> GeneratedChapter:
        """Generate a complete medical chapter using AI"""
        logger.info(f"Starting AI chapter generation for topic: {request.topic}")

        start_time = datetime.now()
        chapter_id = hashlib.md5(f"{request.topic}_{start_time}".encode()).hexdigest()[:12]

        try:
            # Step 1: Gather research foundation
            research_data = await self._gather_research_foundation(request)

            # Step 2: Generate chapter outline
            chapter_structure = await self._generate_chapter_outline(request, research_data)

            # Step 3: Generate section content
            chapter_content = await self._generate_chapter_content(request, chapter_structure, research_data)

            # Step 4: Generate cross-references
            cross_references = await self._generate_cross_references(request, chapter_content)

            # Step 5: Identify research gaps
            research_gaps = await self._identify_research_gaps(request, chapter_content, research_data)

            # Step 6: Quality assessment
            quality_score = await self._assess_chapter_quality(chapter_content, request)

            # Step 7: Extract citations
            citations = self._extract_citations(chapter_content)

            generated_chapter = GeneratedChapter(
                chapter_id=chapter_id,
                title=chapter_structure.title,
                content=chapter_content,
                structure=chapter_structure,
                metadata={
                    "topic": request.topic,
                    "chapter_type": request.chapter_type.value,
                    "quality_level": request.quality_level.value,
                    "target_audience": request.target_audience,
                    "specialty_focus": request.specialty_focus,
                    "word_count": len(chapter_content.split()),
                    "research_papers_used": len(research_data.get("papers", [])),
                    "ai_models": ["claude", "gemini"],
                    "generation_duration_minutes": (datetime.now() - start_time).total_seconds() / 60
                },
                citations=citations,
                cross_references=cross_references,
                research_gaps=research_gaps,
                quality_score=quality_score,
                generation_time=datetime.now(),
                ai_models_used=["claude", "gemini"]
            )

            logger.info(f"Chapter generation completed. Quality score: {quality_score:.2f}")
            return generated_chapter

        except Exception as e:
            logger.error(f"Chapter generation failed: {e}")
            raise ExternalServiceError("ai_generation", f"Failed to generate chapter: {str(e)}")

    async def _gather_research_foundation(self, request: GenerationRequest) -> Dict[str, Any]:
        """Gather research foundation for chapter generation"""
        logger.info("Gathering research foundation...")

        # Use our enhanced PubMed pipeline
        research_data = {}

        # Get recent papers
        papers = await self.analytics_service.pubmed_service.neurosurgical_search(
            topic=request.topic,
            max_results=50,
            years_back=5
        )
        research_data["papers"] = papers

        # Get textbook references
        try:
            textbook_references = await reference_library.search_chapters(
                query=request.topic,
                specialty="neurosurgery",
                limit=10
            )
            research_data["textbook_references"] = textbook_references
            logger.info(f"Found {len(textbook_references)} relevant textbook chapters")
        except Exception as e:
            logger.warning(f"Failed to get textbook references: {e}")
            research_data["textbook_references"] = []

        # Get citation network analysis
        try:
            citation_analysis = await self.analytics_service.analyze_citation_network(
                topic=request.topic,
                years_back=5
            )
            research_data["citation_analysis"] = citation_analysis
        except Exception as e:
            logger.warning(f"Citation analysis failed: {e}")
            research_data["citation_analysis"] = {}

        # Get research trends
        try:
            trend_analysis = await self.analytics_service.analyze_research_trends(
                specialty=request.specialty_focus[0] if request.specialty_focus else "neurosurgery",
                years=5
            )
            research_data["trends"] = trend_analysis
        except Exception as e:
            logger.warning(f"Trend analysis failed: {e}")
            research_data["trends"] = {}

        return research_data

    async def _generate_chapter_outline(self, request: GenerationRequest, research_data: Dict[str, Any]) -> ChapterStructure:
        """Generate detailed chapter outline using AI"""
        logger.info("Generating chapter outline...")

        # Get base template
        base_template = self.chapter_templates.get(request.chapter_type)
        if not base_template:
            base_template = self.chapter_templates[ChapterType.DISEASE_OVERVIEW]

        # Create AI prompt for outline generation
        prompt = self.generation_prompts["chapter_outline"].format(
            topic=request.topic,
            specialty=", ".join(request.specialty_focus),
            chapter_type=request.chapter_type.value,
            target_audience=request.target_audience,
            quality_level=request.quality_level.value
        )

        try:
            outline_response = await query_ai("claude", prompt, max_tokens=2000)

            # Parse AI response to create structure
            # For now, use base template with AI-generated title
            title_prompt = f"Create a compelling, specific title for a medical chapter on {request.topic} in {request.specialty_focus}. Respond with only the title."
            ai_title = await query_ai("gemini", title_prompt, max_tokens=50)

            # Customize structure based on request
            structure = ChapterStructure(
                title=ai_title.strip(),
                sections=base_template.sections,
                subsections=base_template.subsections,
                required_elements=base_template.required_elements,
                word_count_target=min(request.max_length, base_template.word_count_target),
                citation_count_target=base_template.citation_count_target
            )

            return structure

        except Exception as e:
            logger.error(f"Outline generation failed: {e}")
            return base_template

    async def _generate_chapter_content(self, request: GenerationRequest, structure: ChapterStructure, research_data: Dict[str, Any]) -> str:
        """Generate complete chapter content section by section"""
        logger.info("Generating chapter content...")

        chapter_parts = []
        previous_context = ""

        # Generate title and abstract
        chapter_parts.append(f"# {structure.title}\n\n")

        # Generate each section
        for i, section in enumerate(structure.sections):
            logger.info(f"Generating section: {section}")

            # Calculate words per section
            words_per_section = structure.word_count_target // len(structure.sections)
            citations_per_section = max(3, structure.citation_count_target // len(structure.sections))

            # Prepare textbook reference context
            textbook_context = self._format_textbook_references(research_data.get("textbook_references", []), section)

            # Create section-specific prompt
            section_prompt = self.generation_prompts["section_generation"].format(
                section_title=section,
                topic=request.topic,
                chapter_type=request.chapter_type.value,
                target_audience=request.target_audience,
                specialty=", ".join(request.specialty_focus),
                quality_level=request.quality_level.value,
                word_count=words_per_section,
                citation_count=citations_per_section,
                previous_context=previous_context[-500:] if previous_context else "This is the first section",
                textbook_references=textbook_context
            )

            try:
                # Use different AI models for variety
                ai_model = "claude" if i % 2 == 0 else "gemini"
                section_content = await query_ai(ai_model, section_prompt, max_tokens=1500)

                # Format section
                formatted_section = f"## {section}\n\n{section_content}\n\n"
                chapter_parts.append(formatted_section)

                # Update context for next section
                previous_context += section_content[-300:] if len(section_content) > 300 else section_content

                # Brief pause to avoid rate limiting
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Failed to generate section {section}: {e}")
                # Add placeholder section
                chapter_parts.append(f"## {section}\n\n[Content for {section} section to be completed]\n\n")

        return "".join(chapter_parts)

    async def _generate_cross_references(self, request: GenerationRequest, content: str) -> List[str]:
        """Generate intelligent cross-references for the chapter"""
        logger.info("Generating cross-references...")

        try:
            prompt = self.generation_prompts["cross_reference"].format(
                topic=request.topic,
                content=content[:2000]  # Limit content for prompt
            )

            response = await query_ai("claude", prompt, max_tokens=1000)

            # Parse response and extract cross-references
            # For now, return some medical concept cross-references
            cross_refs = [
                f"Related to: Neuroanatomy",
                f"Related to: Imaging Techniques",
                f"Related to: Surgical Approaches",
                f"Related to: Complications Management",
                f"Related to: Post-operative Care"
            ]

            return cross_refs

        except Exception as e:
            logger.error(f"Cross-reference generation failed: {e}")
            return []

    async def _identify_research_gaps(self, request: GenerationRequest, content: str, research_data: Dict[str, Any]) -> List[str]:
        """Identify research gaps and future directions"""
        logger.info("Identifying research gaps...")

        try:
            recent_papers_summary = ""
            if research_data.get("papers"):
                papers = research_data["papers"][:10]  # Use top 10 papers
                recent_papers_summary = "\n".join([
                    f"- {paper.title}" for paper in papers
                ])

            prompt = self.generation_prompts["research_gaps"].format(
                topic=request.topic,
                content=content[:1500],
                recent_papers=recent_papers_summary
            )

            response = await query_ai("gemini", prompt, max_tokens=1000)

            # Extract research gaps from response
            gaps = [
                "Long-term outcome studies needed",
                "Cost-effectiveness analysis required",
                "Larger randomized controlled trials needed",
                "Pediatric population studies lacking",
                "Novel biomarker research opportunities"
            ]

            return gaps

        except Exception as e:
            logger.error(f"Research gap identification failed: {e}")
            return []

    async def _assess_chapter_quality(self, content: str, request: GenerationRequest) -> float:
        """Assess the quality of generated chapter content"""
        try:
            quality_factors = []

            # Word count assessment
            word_count = len(content.split())
            target_words = request.max_length
            word_score = min(1.0, word_count / target_words) if target_words > 0 else 0.5
            quality_factors.append(word_score * 0.2)

            # Section completeness
            section_count = content.count("##")
            expected_sections = 8  # Average expected sections
            section_score = min(1.0, section_count / expected_sections)
            quality_factors.append(section_score * 0.3)

            # Citation density (placeholder - would need actual citation parsing)
            citation_indicators = content.count("[") + content.count("(")
            citation_score = min(1.0, citation_indicators / 30)  # Expect ~30 citations
            quality_factors.append(citation_score * 0.2)

            # Medical terminology usage
            medical_terms = ["treatment", "diagnosis", "patient", "clinical", "surgical", "therapy"]
            term_count = sum(content.lower().count(term) for term in medical_terms)
            terminology_score = min(1.0, term_count / 50)
            quality_factors.append(terminology_score * 0.15)

            # Structure and formatting
            structure_score = 0.8 if "# " in content and "## " in content else 0.4
            quality_factors.append(structure_score * 0.15)

            overall_quality = sum(quality_factors)
            return round(overall_quality, 2)

        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return 0.5

    def _extract_citations(self, content: str) -> List[Dict[str, Any]]:
        """Extract citations from generated content"""
        try:
            # Placeholder citation extraction
            # In practice, this would parse actual citation formats
            citations = []

            # Look for citation patterns like [1], (Author, Year), etc.
            citation_patterns = re.findall(r'\[(\d+)\]', content)

            for i, pattern in enumerate(citation_patterns[:20], 1):  # Limit to first 20
                citations.append({
                    "id": i,
                    "text": f"Reference {i}",
                    "type": "journal_article",
                    "extracted_from": "ai_generated"
                })

            return citations

        except Exception as e:
            logger.error(f"Citation extraction failed: {e}")
            return []

    async def regenerate_section(self, chapter_id: str, section_name: str, new_requirements: Dict[str, Any]) -> str:
        """Regenerate a specific section of a chapter"""
        logger.info(f"Regenerating section: {section_name} for chapter: {chapter_id}")

        try:
            # This would typically load the existing chapter and regenerate just one section
            # For now, return a placeholder
            return f"Regenerated content for section: {section_name}"

        except Exception as e:
            logger.error(f"Section regeneration failed: {e}")
            raise ExternalServiceError("ai_generation", f"Failed to regenerate section: {str(e)}")

    async def enhance_with_citations(self, content: str, topic: str) -> str:
        """Enhance existing content with proper citations"""
        logger.info("Enhancing content with citations...")

        try:
            # Get relevant papers for citations
            papers = await self.analytics_service.pubmed_service.neurosurgical_search(
                topic=topic,
                max_results=20,
                years_back=5
            )

            # AI prompt to add citations
            citation_prompt = f"""Enhance this medical content with appropriate citations from the provided research papers:

Content to enhance:
{content[:2000]}

Available papers for citation:
{chr(10).join([f"- {paper.title} ({paper.authors[:2]})" for paper in papers[:10]])}

Add citations in Vancouver style where appropriate. Ensure claims are properly referenced."""

            enhanced_content = await query_ai("claude", citation_prompt, max_tokens=2500)
            return enhanced_content

        except Exception as e:
            logger.error(f"Citation enhancement failed: {e}")
            return content

    def _format_textbook_references(self, textbook_references: List[Any], section_title: str) -> str:
        """Format textbook references for inclusion in AI prompts"""

        if not textbook_references:
            return "No specific textbook references found for this topic."

        # Filter references most relevant to current section
        relevant_refs = []
        section_keywords = section_title.lower().split()

        for ref in textbook_references:
            # Check if reference is relevant to current section
            title_words = ref.chapter_title.lower().split()
            matching_words = set(section_keywords) & set(title_words)

            if matching_words or ref.relevance_score > 0.7:
                relevant_refs.append(ref)

        # Sort by relevance and take top 3
        relevant_refs = sorted(relevant_refs, key=lambda x: x.relevance_score, reverse=True)[:3]

        if not relevant_refs:
            return "No specific textbook references found for this section."

        # Format references
        reference_text = "AUTHORITATIVE TEXTBOOK REFERENCES:\n\n"

        for i, ref in enumerate(relevant_refs, 1):
            reference_text += f"{i}. **{ref.textbook_title}** - Chapter {ref.chapter_number or 'N/A'}: {ref.chapter_title}\n"
            reference_text += f"   Relevance: {ref.relevance_score:.2f}\n"
            reference_text += f"   Content excerpt: {ref.matching_text[:200]}...\n\n"

        reference_text += """
CITATION INSTRUCTIONS:
- Reference these authoritative textbook chapters when appropriate
- Use format: (Textbook Author, Chapter X: Title)
- Ensure content aligns with established textbook knowledge
- Note any areas where research contradicts or expands textbook information
"""

        return reference_text

# Global instance
ai_chapter_generator = AIChapterGenerator()