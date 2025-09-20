"""
Smart Tagging System for Neurosurgical Knowledge
AI-powered automatic tagging of neurosurgical content with medical concepts
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, Counter

from .hybrid_ai_manager import query_ai
from .semantic_search import MedicalConceptExtractor
from core.config import settings
from core.exceptions import ExternalServiceError

logger = logging.getLogger(__name__)

class TagType(Enum):
    """Types of neurosurgical tags"""
    ANATOMY = "anatomy"
    SURGICAL_TECHNIQUE = "surgical_technique"
    DISEASE_CONDITION = "disease_condition"
    IMAGING_MODALITY = "imaging_modality"
    CLINICAL_ASSESSMENT = "clinical_assessment"
    TREATMENT_APPROACH = "treatment_approach"
    SURGICAL_APPROACH = "surgical_approach"
    COMPLICATIONS = "complications"
    OUTCOMES = "outcomes"
    INSTRUMENTATION = "instrumentation"

class TagConfidence(Enum):
    """Confidence levels for tags"""
    HIGH = "high"          # 0.8-1.0
    MEDIUM = "medium"      # 0.6-0.79
    LOW = "low"           # 0.4-0.59
    UNCERTAIN = "uncertain" # 0.0-0.39

@dataclass
class NeurosurgicalTag:
    """Individual neurosurgical tag"""
    tag_id: str
    tag_name: str
    tag_type: TagType
    confidence: float
    confidence_level: TagConfidence
    context: str
    synonyms: List[str]
    related_concepts: List[str]
    anatomical_location: Optional[str] = None
    pathological_type: Optional[str] = None
    surgical_complexity: Optional[str] = None

@dataclass
class TaggingResult:
    """Result of content tagging"""
    content_id: str
    content_type: str
    tags: List[NeurosurgicalTag]
    primary_topic: str
    anatomical_regions: List[str]
    surgical_procedures: List[str]
    disease_entities: List[str]
    imaging_findings: List[str]
    clinical_significance: str
    tagged_at: datetime
    ai_model_used: str

class NeurosurgicalTagger:
    """Smart tagging system for neurosurgical content"""

    def __init__(self):
        self.concept_extractor = MedicalConceptExtractor()

        # Neurosurgical tag hierarchies
        self.tag_hierarchies = {
            TagType.ANATOMY: {
                "cranial": [
                    "frontal lobe", "parietal lobe", "temporal lobe", "occipital lobe",
                    "cerebellum", "brainstem", "skull base", "ventricular system",
                    "cerebral arteries", "dural sinuses", "cranial nerves"
                ],
                "spinal": [
                    "cervical spine", "thoracic spine", "lumbar spine", "sacrum",
                    "spinal cord", "nerve roots", "facet joints", "intervertebral discs",
                    "spinal ligaments", "epidural space"
                ],
                "peripheral": [
                    "peripheral nerves", "brachial plexus", "lumbosacral plexus",
                    "autonomic nervous system"
                ]
            },
            TagType.SURGICAL_TECHNIQUE: {
                "microsurgical": [
                    "microsurgical dissection", "aneurysm clipping", "tumor resection",
                    "AVM resection", "nerve repair", "bypass surgery"
                ],
                "endoscopic": [
                    "endoscopic endonasal surgery", "neuroendoscopy", "ventricular endoscopy",
                    "spine endoscopy"
                ],
                "stereotactic": [
                    "stereotactic biopsy", "deep brain stimulation", "gamma knife",
                    "LINAC radiosurgery", "frame-based stereotaxy", "frameless stereotaxy"
                ],
                "minimally_invasive": [
                    "keyhole craniotomy", "tubular retractor surgery", "XLIF", "TLIF",
                    "endoscopic discectomy"
                ]
            },
            TagType.DISEASE_CONDITION: {
                "neoplastic": [
                    "glioblastoma", "meningioma", "pituitary adenoma", "acoustic neuroma",
                    "craniopharyngioma", "chordoma", "metastatic disease"
                ],
                "vascular": [
                    "cerebral aneurysm", "arteriovenous malformation", "cavernous malformation",
                    "dural arteriovenous fistula", "moyamoya disease", "cerebral ischemia"
                ],
                "degenerative": [
                    "cervical spondylosis", "lumbar stenosis", "disc herniation",
                    "spondylolisthesis", "cervical myelopathy", "radiculopathy"
                ],
                "functional": [
                    "epilepsy", "trigeminal neuralgia", "hemifacial spasm",
                    "movement disorders", "chronic pain", "spasticity"
                ],
                "traumatic": [
                    "traumatic brain injury", "spinal cord injury", "skull fracture",
                    "epidural hematoma", "subdural hematoma", "penetrating injury"
                ],
                "congenital": [
                    "Chiari malformation", "spina bifida", "craniosynostosis",
                    "arachnoid cyst", "hydrocephalus", "tethered cord"
                ]
            },
            TagType.SURGICAL_APPROACH: {
                "cranial_approaches": [
                    "pterional", "frontotemporal", "orbitozygomatic", "retrosigmoid",
                    "subtemporal", "interhemispheric", "transcallosal", "transpetrosal",
                    "endoscopic endonasal", "eyebrow incision"
                ],
                "spinal_approaches": [
                    "anterior cervical", "posterior cervical", "anterolateral",
                    "posterior lumbar", "lateral lumbar", "minimally invasive",
                    "percutaneous", "robot-assisted"
                ]
            },
            TagType.IMAGING_MODALITY: {
                "structural": [
                    "CT scan", "MRI", "CT angiography", "MR angiography",
                    "digital subtraction angiography", "myelography"
                ],
                "functional": [
                    "functional MRI", "PET scan", "SPECT", "DTI", "perfusion imaging",
                    "MR spectroscopy", "tractography"
                ],
                "intraoperative": [
                    "intraoperative ultrasound", "intraoperative MRI", "fluorescein",
                    "5-ALA", "indocyanine green", "Doppler ultrasound"
                ]
            }
        }

        # Clinical grading scales and assessments
        self.clinical_scales = {
            "neurological": [
                "Glasgow Coma Scale", "Hunt-Hess grade", "Fisher grade", "WFNS grade",
                "mRS", "Karnofsky Performance Scale", "ECOG", "Barthel Index"
            ],
            "spinal": [
                "Oswestry Disability Index", "JOA score", "Nurick grade",
                "NDI", "VAS pain score", "Frankel grade", "ASIA scale"
            ],
            "oncological": [
                "WHO grade", "Karnofsky Performance Scale", "ECOG", "Lansky scale",
                "extent of resection", "progression-free survival"
            ]
        }

        # Tag extraction patterns
        self.extraction_patterns = {
            "anatomy_mentions": r'\b(?:frontal|parietal|temporal|occipital|cerebellum|brainstem|thalamus|basal ganglia)\b',
            "surgical_procedures": r'\b(?:craniotomy|craniectomy|laminectomy|discectomy|fusion|clipping|resection)\b',
            "imaging_findings": r'\b(?:enhancement|mass effect|edema|hemorrhage|infarct|stenosis|occlusion)\b',
            "grading_scales": r'\b(?:WHO grade|Hunt-Hess|Fisher|mRS|Karnofsky|Oswestry|JOA)\b'
        }

    async def tag_content(self, content_id: str, content_text: str, content_type: str = "chapter") -> TaggingResult:
        """Tag neurosurgical content with relevant medical concepts"""
        logger.info(f"Tagging content: {content_id}")

        try:
            # Step 1: Extract neurosurgical concepts
            extracted_concepts = self._extract_neurosurgical_concepts(content_text)

            # Step 2: AI-powered concept identification
            ai_concepts = await self._ai_concept_identification(content_text)

            # Step 3: Generate structured tags
            tags = await self._generate_structured_tags(content_text, extracted_concepts, ai_concepts)

            # Step 4: Analyze content structure
            content_analysis = self._analyze_content_structure(content_text, tags)

            # Step 5: Determine clinical significance
            clinical_significance = await self._assess_clinical_significance(content_text, tags)

            tagging_result = TaggingResult(
                content_id=content_id,
                content_type=content_type,
                tags=tags,
                primary_topic=content_analysis["primary_topic"],
                anatomical_regions=content_analysis["anatomical_regions"],
                surgical_procedures=content_analysis["surgical_procedures"],
                disease_entities=content_analysis["disease_entities"],
                imaging_findings=content_analysis["imaging_findings"],
                clinical_significance=clinical_significance,
                tagged_at=datetime.now(),
                ai_model_used="claude"
            )

            logger.info(f"Generated {len(tags)} tags for content {content_id}")
            return tagging_result

        except Exception as e:
            logger.error(f"Content tagging failed for {content_id}: {e}")
            raise ExternalServiceError("tagging", f"Failed to tag content: {str(e)}")

    def _extract_neurosurgical_concepts(self, content: str) -> Dict[str, List[Tuple[str, float]]]:
        """Extract neurosurgical concepts with confidence scores"""
        content_lower = content.lower()
        extracted_concepts = defaultdict(list)

        # Extract from hierarchical tags
        for tag_type, categories in self.tag_hierarchies.items():
            for category, terms in categories.items():
                for term in terms:
                    if term.lower() in content_lower:
                        # Calculate confidence based on frequency and context
                        frequency = content_lower.count(term.lower())
                        confidence = min(0.5 + (frequency * 0.1), 1.0)
                        extracted_concepts[tag_type.value].append((term, confidence))

        # Extract clinical scales
        for scale_type, scales in self.clinical_scales.items():
            for scale in scales:
                if scale.lower() in content_lower:
                    confidence = 0.8  # High confidence for explicit scale mentions
                    extracted_concepts["clinical_assessment"].append((scale, confidence))

        # Pattern-based extraction
        for pattern_type, pattern in self.extraction_patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                confidence = 0.7  # Medium confidence for pattern matches
                extracted_concepts[pattern_type].append((match, confidence))

        return dict(extracted_concepts)

    async def _ai_concept_identification(self, content: str) -> Dict[str, Any]:
        """Use AI to identify complex neurosurgical concepts"""
        try:
            concept_prompt = f"""Analyze this neurosurgical content and identify key concepts:

Content: {content[:2000]}

Please identify:
1. Primary neurosurgical topic
2. Anatomical structures mentioned
3. Surgical techniques described
4. Disease conditions discussed
5. Imaging modalities referenced
6. Clinical assessments mentioned
7. Treatment approaches outlined
8. Complications or risks discussed

For each concept, provide:
- The concept name
- Type (anatomy, procedure, disease, etc.)
- Confidence level (high/medium/low)
- Context relevance

Format as structured JSON."""

            ai_response = await query_ai("claude", concept_prompt, max_tokens=1000)

            # Parse AI response (simplified parsing)
            ai_concepts = {
                "primary_topic": self._extract_primary_topic(ai_response),
                "anatomical_structures": self._extract_ai_list(ai_response, "anatomical"),
                "surgical_techniques": self._extract_ai_list(ai_response, "surgical"),
                "disease_conditions": self._extract_ai_list(ai_response, "disease"),
                "imaging_modalities": self._extract_ai_list(ai_response, "imaging"),
                "clinical_assessments": self._extract_ai_list(ai_response, "clinical"),
                "treatment_approaches": self._extract_ai_list(ai_response, "treatment"),
                "complications": self._extract_ai_list(ai_response, "complications")
            }

            return ai_concepts

        except Exception as e:
            logger.error(f"AI concept identification failed: {e}")
            return {}

    async def _generate_structured_tags(self, content: str, extracted_concepts: Dict[str, List[Tuple[str, float]]], ai_concepts: Dict[str, Any]) -> List[NeurosurgicalTag]:
        """Generate structured neurosurgical tags"""
        tags = []
        tag_counter = 0

        # Process extracted concepts
        for concept_type, concepts in extracted_concepts.items():
            for concept_name, confidence in concepts:
                tag_counter += 1

                # Determine tag type
                try:
                    tag_type = TagType(concept_type)
                except ValueError:
                    tag_type = TagType.ANATOMY  # Default

                # Determine confidence level
                confidence_level = self._get_confidence_level(confidence)

                # Get context
                context = self._extract_context(content, concept_name)

                # Get synonyms and related concepts
                synonyms = self._get_synonyms(concept_name)
                related_concepts = self._get_related_concepts(concept_name, tag_type)

                # Additional attributes based on type
                additional_attrs = self._get_additional_attributes(concept_name, tag_type, content)

                tag = NeurosurgicalTag(
                    tag_id=f"tag_{tag_counter:04d}",
                    tag_name=concept_name,
                    tag_type=tag_type,
                    confidence=confidence,
                    confidence_level=confidence_level,
                    context=context,
                    synonyms=synonyms,
                    related_concepts=related_concepts,
                    anatomical_location=additional_attrs.get("anatomical_location"),
                    pathological_type=additional_attrs.get("pathological_type"),
                    surgical_complexity=additional_attrs.get("surgical_complexity")
                )

                tags.append(tag)

        # Remove duplicates and sort by confidence
        unique_tags = self._deduplicate_tags(tags)
        unique_tags.sort(key=lambda x: x.confidence, reverse=True)

        return unique_tags[:50]  # Limit to top 50 tags

    def _analyze_content_structure(self, content: str, tags: List[NeurosurgicalTag]) -> Dict[str, Any]:
        """Analyze the structure and focus of the content"""
        content_lower = content.lower()

        # Group tags by type
        tags_by_type = defaultdict(list)
        for tag in tags:
            tags_by_type[tag.tag_type].append(tag.tag_name)

        # Determine primary topic
        primary_topic = "General Neurosurgery"
        if tags_by_type[TagType.DISEASE_CONDITION]:
            primary_topic = tags_by_type[TagType.DISEASE_CONDITION][0]
        elif tags_by_type[TagType.SURGICAL_TECHNIQUE]:
            primary_topic = tags_by_type[TagType.SURGICAL_TECHNIQUE][0]

        return {
            "primary_topic": primary_topic,
            "anatomical_regions": tags_by_type[TagType.ANATOMY][:5],
            "surgical_procedures": tags_by_type[TagType.SURGICAL_TECHNIQUE][:5],
            "disease_entities": tags_by_type[TagType.DISEASE_CONDITION][:5],
            "imaging_findings": tags_by_type[TagType.IMAGING_MODALITY][:5]
        }

    async def _assess_clinical_significance(self, content: str, tags: List[NeurosurgicalTag]) -> str:
        """Assess the clinical significance of the content"""
        try:
            significance_prompt = f"""Assess the clinical significance of this neurosurgical content:

Content: {content[:1000]}

Key concepts identified: {', '.join([tag.tag_name for tag in tags[:10]])}

Please provide a brief assessment (2-3 sentences) of:
1. Clinical relevance
2. Practical applications
3. Target audience (residents, attendings, etc.)

Focus on neurosurgical practice implications."""

            significance = await query_ai("gemini", significance_prompt, max_tokens=300)
            return significance

        except Exception as e:
            logger.error(f"Clinical significance assessment failed: {e}")
            return "Clinical significance assessment pending"

    def _get_confidence_level(self, confidence: float) -> TagConfidence:
        """Convert numeric confidence to confidence level"""
        if confidence >= 0.8:
            return TagConfidence.HIGH
        elif confidence >= 0.6:
            return TagConfidence.MEDIUM
        elif confidence >= 0.4:
            return TagConfidence.LOW
        else:
            return TagConfidence.UNCERTAIN

    def _extract_context(self, content: str, concept: str, window_size: int = 100) -> str:
        """Extract context around a concept mention"""
        try:
            content_lower = content.lower()
            concept_lower = concept.lower()

            start_idx = content_lower.find(concept_lower)
            if start_idx == -1:
                return ""

            # Extract context window
            start = max(0, start_idx - window_size)
            end = min(len(content), start_idx + len(concept) + window_size)

            context = content[start:end].strip()
            return context

        except Exception:
            return ""

    def _get_synonyms(self, concept: str) -> List[str]:
        """Get synonyms for a concept"""
        concept_lower = concept.lower()

        # Check medical synonyms from concept extractor
        for term, synonyms in self.concept_extractor.medical_synonyms.items():
            if concept_lower == term.lower() or concept_lower in [s.lower() for s in synonyms]:
                return synonyms

        return []

    def _get_related_concepts(self, concept: str, tag_type: TagType) -> List[str]:
        """Get related concepts for a given concept and type"""
        related = []

        # Find related concepts within the same hierarchy
        if tag_type in self.tag_hierarchies:
            for category, terms in self.tag_hierarchies[tag_type].items():
                if concept.lower() in [t.lower() for t in terms]:
                    # Add other terms from the same category
                    related.extend([t for t in terms if t.lower() != concept.lower()][:3])

        return related

    def _get_additional_attributes(self, concept: str, tag_type: TagType, content: str) -> Dict[str, Optional[str]]:
        """Get additional attributes based on concept type"""
        attributes = {
            "anatomical_location": None,
            "pathological_type": None,
            "surgical_complexity": None
        }

        # Anatomical location inference
        if tag_type == TagType.ANATOMY:
            if any(region in concept.lower() for region in ["frontal", "parietal", "temporal", "occipital"]):
                attributes["anatomical_location"] = "supratentorial"
            elif any(region in concept.lower() for region in ["cerebellum", "brainstem"]):
                attributes["anatomical_location"] = "infratentorial"
            elif any(region in concept.lower() for region in ["spine", "vertebral"]):
                attributes["anatomical_location"] = "spinal"

        # Pathological type inference
        if tag_type == TagType.DISEASE_CONDITION:
            if any(term in concept.lower() for term in ["glioma", "meningioma", "adenoma"]):
                attributes["pathological_type"] = "neoplastic"
            elif any(term in concept.lower() for term in ["aneurysm", "avm", "stroke"]):
                attributes["pathological_type"] = "vascular"
            elif any(term in concept.lower() for term in ["stenosis", "herniation", "spondylosis"]):
                attributes["pathological_type"] = "degenerative"

        # Surgical complexity inference
        if tag_type == TagType.SURGICAL_TECHNIQUE:
            if any(term in concept.lower() for term in ["microsurgical", "awake", "bypass"]):
                attributes["surgical_complexity"] = "high"
            elif any(term in concept.lower() for term in ["endoscopic", "stereotactic"]):
                attributes["surgical_complexity"] = "medium"
            else:
                attributes["surgical_complexity"] = "standard"

        return attributes

    def _deduplicate_tags(self, tags: List[NeurosurgicalTag]) -> List[NeurosurgicalTag]:
        """Remove duplicate tags based on tag name and type"""
        seen = set()
        unique_tags = []

        for tag in tags:
            tag_key = (tag.tag_name.lower(), tag.tag_type)
            if tag_key not in seen:
                seen.add(tag_key)
                unique_tags.append(tag)

        return unique_tags

    def _extract_primary_topic(self, ai_response: str) -> str:
        """Extract primary topic from AI response"""
        try:
            # Look for primary topic indicators
            lines = ai_response.split('\n')
            for line in lines:
                if "primary" in line.lower() and "topic" in line.lower():
                    # Extract topic name
                    topic_match = re.search(r'[:\-]\s*([^.]+)', line)
                    if topic_match:
                        return topic_match.group(1).strip()

            return "General Neurosurgery"

        except Exception:
            return "General Neurosurgery"

    def _extract_ai_list(self, ai_response: str, concept_type: str) -> List[str]:
        """Extract concept lists from AI response"""
        try:
            concepts = []
            lines = ai_response.split('\n')
            capturing = False

            for line in lines:
                if concept_type.lower() in line.lower():
                    capturing = True
                    continue

                if capturing:
                    if line.strip().startswith(('-', '*', '•')):
                        concept = line.strip().lstrip('-*•').strip()
                        if concept:
                            concepts.append(concept)
                    elif line.strip() == '' or len(concepts) >= 5:
                        break

            return concepts

        except Exception:
            return []

    async def get_tag_suggestions(self, partial_tag: str, tag_type: Optional[TagType] = None) -> List[str]:
        """Get tag suggestions for autocomplete"""
        suggestions = []
        partial_lower = partial_tag.lower()

        # Search in hierarchies
        hierarchies_to_search = [tag_type] if tag_type else list(self.tag_hierarchies.keys())

        for hierarchy_type in hierarchies_to_search:
            if hierarchy_type in self.tag_hierarchies:
                for category, terms in self.tag_hierarchies[hierarchy_type].items():
                    for term in terms:
                        if partial_lower in term.lower():
                            suggestions.append(term)

        return sorted(list(set(suggestions)))[:10]

    async def get_popular_tags(self, tag_type: Optional[TagType] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """Get popular tags for a given type"""
        # Simulated popular tags - in practice would query database
        popular_tags = {
            TagType.ANATOMY: [
                {"tag": "frontal lobe", "usage_count": 45, "trend": "+5%"},
                {"tag": "temporal lobe", "usage_count": 38, "trend": "+3%"},
                {"tag": "cerebellum", "usage_count": 32, "trend": "+8%"}
            ],
            TagType.SURGICAL_TECHNIQUE: [
                {"tag": "craniotomy", "usage_count": 67, "trend": "+12%"},
                {"tag": "microsurgical clipping", "usage_count": 43, "trend": "+7%"},
                {"tag": "endoscopic surgery", "usage_count": 39, "trend": "+15%"}
            ],
            TagType.DISEASE_CONDITION: [
                {"tag": "glioblastoma", "usage_count": 56, "trend": "+9%"},
                {"tag": "cerebral aneurysm", "usage_count": 48, "trend": "+6%"},
                {"tag": "cervical myelopathy", "usage_count": 41, "trend": "+4%"}
            ]
        }

        if tag_type and tag_type in popular_tags:
            return popular_tags[tag_type][:limit]

        # Return all popular tags
        all_popular = []
        for tags in popular_tags.values():
            all_popular.extend(tags)

        return sorted(all_popular, key=lambda x: x["usage_count"], reverse=True)[:limit]

# Global instance
neurosurgical_tagger = NeurosurgicalTagger()