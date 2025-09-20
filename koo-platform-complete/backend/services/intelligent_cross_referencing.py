"""
Intelligent Cross-Referencing System for Neurosurgical Knowledge
AI-powered system for creating intelligent connections between neurosurgical concepts
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict
import networkx as nx

from .hybrid_ai_manager import query_ai
from .smart_tagging import neurosurgical_tagger, TagType
from .semantic_search import semantic_search_engine
from core.config import settings
from core.exceptions import ExternalServiceError

logger = logging.getLogger(__name__)

class CrossReferenceType(Enum):
    """Types of cross-references in neurosurgical knowledge"""
    ANATOMICAL_RELATION = "anatomical_relation"        # Related anatomical structures
    SURGICAL_APPROACH = "surgical_approach"           # Related surgical techniques
    DISEASE_PROGRESSION = "disease_progression"       # Disease progression pathways
    DIAGNOSTIC_PATHWAY = "diagnostic_pathway"         # Diagnostic sequences
    TREATMENT_ALGORITHM = "treatment_algorithm"       # Treatment decision trees
    COMPLICATION_LINK = "complication_link"          # Complication relationships
    IMAGING_CORRELATION = "imaging_correlation"       # Imaging-pathology correlation
    EMBRYOLOGICAL = "embryological"                   # Developmental relationships
    PHYSIOLOGICAL = "physiological"                   # Functional relationships
    PATHOPHYSIOLOGICAL = "pathophysiological"        # Disease mechanism links

class ReferenceStrength(Enum):
    """Strength of cross-reference relationship"""
    STRONG = "strong"          # Direct, essential relationship
    MODERATE = "moderate"      # Important but not essential
    WEAK = "weak"             # Tangential relationship
    CONTEXTUAL = "contextual"  # Context-dependent relationship

@dataclass
class CrossReference:
    """Individual cross-reference connection"""
    reference_id: str
    source_concept: str
    source_chapter_id: str
    target_concept: str
    target_chapter_id: str
    reference_type: CrossReferenceType
    strength: ReferenceStrength
    confidence: float
    description: str
    clinical_relevance: str
    bidirectional: bool
    context_keywords: List[str]
    anatomical_context: Optional[str] = None
    pathological_context: Optional[str] = None

@dataclass
class ConceptGraph:
    """Graph representation of neurosurgical concept relationships"""
    graph_id: str
    central_concept: str
    related_concepts: List[str]
    cross_references: List[CrossReference]
    concept_hierarchy: Dict[str, List[str]]
    clinical_pathways: List[List[str]]
    generated_at: datetime

class IntelligentCrossReferencer:
    """AI-powered cross-referencing system for neurosurgical knowledge"""

    def __init__(self):
        self.tagger = neurosurgical_tagger
        self.search_engine = semantic_search_engine

        # Neurosurgical relationship patterns
        self.relationship_patterns = {
            "anatomical_adjacency": {
                "frontal lobe": ["parietal lobe", "temporal lobe", "corpus callosum"],
                "temporal lobe": ["frontal lobe", "parietal lobe", "hippocampus", "amygdala"],
                "cerebellum": ["brainstem", "fourth ventricle", "vermis"],
                "cervical spine": ["brainstem", "vertebral artery", "spinal cord"],
                "lumbar spine": ["cauda equina", "nerve roots", "facet joints"]
            },
            "surgical_continuity": {
                "pterional approach": ["sylvian fissure dissection", "aneurysm clipping", "tumor resection"],
                "transsphenoidal surgery": ["sellar anatomy", "pituitary adenoma", "CSF leak repair"],
                "laminectomy": ["spinal decompression", "dural repair", "fusion surgery"],
                "craniotomy": ["brain mapping", "tumor resection", "hemostasis"]
            },
            "diagnostic_sequences": {
                "headache": ["neurological examination", "CT scan", "MRI", "lumbar puncture"],
                "seizure": ["EEG", "MRI", "PET scan", "intracranial monitoring"],
                "back pain": ["physical examination", "X-ray", "MRI", "nerve conduction studies"],
                "weakness": ["motor examination", "EMG", "MRI", "evoked potentials"]
            },
            "treatment_pathways": {
                "glioblastoma": ["surgical resection", "radiation therapy", "chemotherapy", "follow-up imaging"],
                "aneurysm": ["imaging evaluation", "surgical clipping", "endovascular coiling", "post-op monitoring"],
                "disc herniation": ["conservative management", "epidural injection", "discectomy", "rehabilitation"],
                "epilepsy": ["medical management", "seizure monitoring", "epilepsy surgery", "outcome assessment"]
            }
        }

        # Clinical correlation patterns
        self.clinical_correlations = {
            "anatomy_pathology": {
                "frontal lobe lesion": ["executive dysfunction", "personality changes", "motor deficits"],
                "temporal lobe lesion": ["memory impairment", "language deficits", "seizures"],
                "brainstem lesion": ["cranial nerve deficits", "ataxia", "consciousness alteration"],
                "spinal cord lesion": ["motor weakness", "sensory loss", "bowel/bladder dysfunction"]
            },
            "imaging_pathology": {
                "ring enhancement": ["glioblastoma", "metastasis", "abscess", "demyelination"],
                "T2 hyperintensity": ["edema", "gliosis", "infarction", "demyelination"],
                "hemorrhage": ["aneurysm", "AVM", "tumor", "trauma"],
                "hydrocephalus": ["obstruction", "overproduction", "malabsorption"]
            }
        }

    async def generate_cross_references(self, chapter_id: str, content: str, existing_chapters: List[Dict[str, Any]]) -> List[CrossReference]:
        """Generate intelligent cross-references for a chapter"""
        logger.info(f"Generating cross-references for chapter: {chapter_id}")

        try:
            # Step 1: Extract concepts from current chapter
            current_concepts = await self._extract_chapter_concepts(content)

            # Step 2: Analyze existing chapters for relationships
            related_chapters = await self._find_related_chapters(current_concepts, existing_chapters)

            # Step 3: Generate cross-references using AI and patterns
            cross_references = []

            # Pattern-based cross-references
            pattern_refs = await self._generate_pattern_based_references(
                chapter_id, current_concepts, related_chapters
            )
            cross_references.extend(pattern_refs)

            # AI-powered cross-references
            ai_refs = await self._generate_ai_cross_references(
                chapter_id, content, current_concepts, related_chapters
            )
            cross_references.extend(ai_refs)

            # Clinical pathway references
            pathway_refs = await self._generate_clinical_pathway_references(
                chapter_id, current_concepts, related_chapters
            )
            cross_references.extend(pathway_refs)

            # Remove duplicates and rank by relevance
            unique_refs = self._deduplicate_references(cross_references)
            ranked_refs = self._rank_references(unique_refs)

            logger.info(f"Generated {len(ranked_refs)} cross-references for chapter {chapter_id}")
            return ranked_refs

        except Exception as e:
            logger.error(f"Cross-reference generation failed: {e}")
            raise ExternalServiceError("cross_referencing", f"Failed to generate cross-references: {str(e)}")

    async def build_concept_graph(self, central_concept: str, depth: int = 2) -> ConceptGraph:
        """Build a concept graph around a central neurosurgical concept"""
        logger.info(f"Building concept graph for: {central_concept}")

        try:
            # Step 1: Find directly related concepts
            direct_relations = await self._find_direct_relations(central_concept)

            # Step 2: Expand to second-degree relationships if needed
            all_concepts = [central_concept] + direct_relations
            if depth > 1:
                for concept in direct_relations:
                    indirect_relations = await self._find_direct_relations(concept)
                    all_concepts.extend(indirect_relations[:3])  # Limit expansion

            # Remove duplicates
            unique_concepts = list(set(all_concepts))

            # Step 3: Generate cross-references between concepts
            cross_references = []
            for i, concept1 in enumerate(unique_concepts):
                for concept2 in unique_concepts[i+1:]:
                    ref = await self._analyze_concept_relationship(concept1, concept2)
                    if ref:
                        cross_references.append(ref)

            # Step 4: Build concept hierarchy
            hierarchy = await self._build_concept_hierarchy(unique_concepts)

            # Step 5: Identify clinical pathways
            pathways = await self._identify_clinical_pathways(unique_concepts)

            concept_graph = ConceptGraph(
                graph_id=f"graph_{central_concept.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                central_concept=central_concept,
                related_concepts=unique_concepts[1:],  # Exclude central concept
                cross_references=cross_references,
                concept_hierarchy=hierarchy,
                clinical_pathways=pathways,
                generated_at=datetime.now()
            )

            return concept_graph

        except Exception as e:
            logger.error(f"Concept graph building failed: {e}")
            raise ExternalServiceError("cross_referencing", f"Failed to build concept graph: {str(e)}")

    async def _extract_chapter_concepts(self, content: str) -> List[str]:
        """Extract key neurosurgical concepts from chapter content"""
        try:
            # Use the tagging system to extract concepts
            tagging_result = await self.tagger.tag_content("temp_chapter", content, "chapter")

            # Extract concept names
            concepts = []
            for tag in tagging_result.tags:
                if tag.confidence >= 0.6:  # Only high-confidence tags
                    concepts.append(tag.tag_name)

            # Add primary topic and anatomical regions
            concepts.append(tagging_result.primary_topic)
            concepts.extend(tagging_result.anatomical_regions)
            concepts.extend(tagging_result.surgical_procedures)
            concepts.extend(tagging_result.disease_entities)

            return list(set(concepts))

        except Exception as e:
            logger.error(f"Concept extraction failed: {e}")
            return []

    async def _find_related_chapters(self, concepts: List[str], existing_chapters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find chapters related to the given concepts"""
        related_chapters = []

        for chapter in existing_chapters:
            chapter_content = chapter.get("content", "")
            chapter_title = chapter.get("title", "")

            # Calculate concept overlap
            overlap_score = 0
            for concept in concepts:
                if concept.lower() in chapter_content.lower() or concept.lower() in chapter_title.lower():
                    overlap_score += 1

            # Consider chapter related if it has significant overlap
            if overlap_score >= 2:
                chapter["relevance_score"] = overlap_score / len(concepts)
                related_chapters.append(chapter)

        # Sort by relevance
        related_chapters.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return related_chapters[:10]  # Limit to top 10 related chapters

    async def _generate_pattern_based_references(self, chapter_id: str, concepts: List[str], related_chapters: List[Dict[str, Any]]) -> List[CrossReference]:
        """Generate cross-references based on predefined patterns"""
        references = []
        ref_counter = 0

        for concept in concepts:
            concept_lower = concept.lower()

            # Check anatomical adjacency patterns
            if concept_lower in self.relationship_patterns["anatomical_adjacency"]:
                adjacent_structures = self.relationship_patterns["anatomical_adjacency"][concept_lower]
                for structure in adjacent_structures:
                    # Find chapters that mention this structure
                    for chapter in related_chapters:
                        if structure.lower() in chapter.get("content", "").lower():
                            ref_counter += 1
                            ref = CrossReference(
                                reference_id=f"ref_{ref_counter:04d}",
                                source_concept=concept,
                                source_chapter_id=chapter_id,
                                target_concept=structure,
                                target_chapter_id=chapter.get("id", ""),
                                reference_type=CrossReferenceType.ANATOMICAL_RELATION,
                                strength=ReferenceStrength.STRONG,
                                confidence=0.8,
                                description=f"{concept} is anatomically adjacent to {structure}",
                                clinical_relevance=f"Understanding {structure} anatomy is crucial for {concept} procedures",
                                bidirectional=True,
                                context_keywords=["anatomy", "surgical approach"],
                                anatomical_context=concept
                            )
                            references.append(ref)

            # Check surgical continuity patterns
            if concept_lower in self.relationship_patterns["surgical_continuity"]:
                related_procedures = self.relationship_patterns["surgical_continuity"][concept_lower]
                for procedure in related_procedures:
                    for chapter in related_chapters:
                        if procedure.lower() in chapter.get("content", "").lower():
                            ref_counter += 1
                            ref = CrossReference(
                                reference_id=f"ref_{ref_counter:04d}",
                                source_concept=concept,
                                source_chapter_id=chapter_id,
                                target_concept=procedure,
                                target_chapter_id=chapter.get("id", ""),
                                reference_type=CrossReferenceType.SURGICAL_APPROACH,
                                strength=ReferenceStrength.MODERATE,
                                confidence=0.7,
                                description=f"{procedure} is commonly performed during {concept}",
                                clinical_relevance=f"Knowledge of {procedure} is important for {concept} surgery",
                                bidirectional=False,
                                context_keywords=["surgical technique", "procedure"],
                                pathological_context=concept
                            )
                            references.append(ref)

        return references

    async def _generate_ai_cross_references(self, chapter_id: str, content: str, concepts: List[str], related_chapters: List[Dict[str, Any]]) -> List[CrossReference]:
        """Generate cross-references using AI analysis"""
        try:
            # Prepare related chapters context
            related_context = "\n".join([
                f"Chapter: {ch.get('title', '')}\nID: {ch.get('id', '')}\nConcepts: {', '.join(concepts[:5])}"
                for ch in related_chapters[:5]
            ])

            cross_ref_prompt = f"""Analyze this neurosurgical content and suggest intelligent cross-references:

Main Content Concepts: {', '.join(concepts[:10])}

Current Chapter: {content[:1500]}

Related Chapters Available:
{related_context}

Please identify 5-8 meaningful cross-references that would help readers understand:
1. Anatomical relationships
2. Surgical technique connections
3. Disease progression pathways
4. Diagnostic correlations
5. Treatment algorithm links

For each cross-reference, provide:
- Source concept
- Target concept
- Relationship type
- Clinical relevance
- Strength (strong/moderate/weak)

Format as structured text with clear sections."""

            ai_response = await query_ai("claude", cross_ref_prompt, max_tokens=1500)

            # Parse AI response to create cross-references
            ai_references = self._parse_ai_cross_references(ai_response, chapter_id, related_chapters)

            return ai_references

        except Exception as e:
            logger.error(f"AI cross-reference generation failed: {e}")
            return []

    async def _generate_clinical_pathway_references(self, chapter_id: str, concepts: List[str], related_chapters: List[Dict[str, Any]]) -> List[CrossReference]:
        """Generate cross-references based on clinical pathways"""
        references = []
        ref_counter = len(references)

        for concept in concepts:
            concept_lower = concept.lower()

            # Check diagnostic pathways
            if concept_lower in self.relationship_patterns["diagnostic_sequences"]:
                sequence = self.relationship_patterns["diagnostic_sequences"][concept_lower]
                for i, step in enumerate(sequence):
                    if i < len(sequence) - 1:
                        next_step = sequence[i + 1]
                        # Find chapters that discuss the next step
                        for chapter in related_chapters:
                            if next_step.lower() in chapter.get("content", "").lower():
                                ref_counter += 1
                                ref = CrossReference(
                                    reference_id=f"ref_{ref_counter:04d}",
                                    source_concept=step,
                                    source_chapter_id=chapter_id,
                                    target_concept=next_step,
                                    target_chapter_id=chapter.get("id", ""),
                                    reference_type=CrossReferenceType.DIAGNOSTIC_PATHWAY,
                                    strength=ReferenceStrength.STRONG,
                                    confidence=0.9,
                                    description=f"{next_step} typically follows {step} in diagnostic workup",
                                    clinical_relevance=f"Sequential diagnostic approach for {concept}",
                                    bidirectional=False,
                                    context_keywords=["diagnosis", "workup", "sequence"]
                                )
                                references.append(ref)

            # Check treatment pathways
            if concept_lower in self.relationship_patterns["treatment_pathways"]:
                pathway = self.relationship_patterns["treatment_pathways"][concept_lower]
                for i, treatment in enumerate(pathway):
                    if i < len(pathway) - 1:
                        next_treatment = pathway[i + 1]
                        for chapter in related_chapters:
                            if next_treatment.lower() in chapter.get("content", "").lower():
                                ref_counter += 1
                                ref = CrossReference(
                                    reference_id=f"ref_{ref_counter:04d}",
                                    source_concept=treatment,
                                    source_chapter_id=chapter_id,
                                    target_concept=next_treatment,
                                    target_chapter_id=chapter.get("id", ""),
                                    reference_type=CrossReferenceType.TREATMENT_ALGORITHM,
                                    strength=ReferenceStrength.MODERATE,
                                    confidence=0.8,
                                    description=f"{next_treatment} is the next step after {treatment}",
                                    clinical_relevance=f"Treatment sequence for {concept} management",
                                    bidirectional=False,
                                    context_keywords=["treatment", "management", "sequence"]
                                )
                                references.append(ref)

        return references

    def _parse_ai_cross_references(self, ai_response: str, chapter_id: str, related_chapters: List[Dict[str, Any]]) -> List[CrossReference]:
        """Parse AI response to extract cross-references"""
        references = []
        try:
            # Simple parsing of AI response
            lines = ai_response.split('\n')
            current_ref = {}
            ref_counter = 0

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                if "source concept" in line.lower():
                    current_ref["source"] = line.split(":")[-1].strip()
                elif "target concept" in line.lower():
                    current_ref["target"] = line.split(":")[-1].strip()
                elif "relationship type" in line.lower():
                    current_ref["type"] = line.split(":")[-1].strip()
                elif "clinical relevance" in line.lower():
                    current_ref["relevance"] = line.split(":")[-1].strip()
                elif "strength" in line.lower():
                    current_ref["strength"] = line.split(":")[-1].strip()

                    # Complete reference when we have all fields
                    if all(key in current_ref for key in ["source", "target", "type", "relevance", "strength"]):
                        ref_counter += 1

                        # Find target chapter
                        target_chapter_id = ""
                        for chapter in related_chapters:
                            if current_ref["target"].lower() in chapter.get("content", "").lower():
                                target_chapter_id = chapter.get("id", "")
                                break

                        # Map relationship type
                        ref_type = CrossReferenceType.ANATOMICAL_RELATION  # Default
                        if "surgical" in current_ref["type"].lower():
                            ref_type = CrossReferenceType.SURGICAL_APPROACH
                        elif "diagnostic" in current_ref["type"].lower():
                            ref_type = CrossReferenceType.DIAGNOSTIC_PATHWAY
                        elif "treatment" in current_ref["type"].lower():
                            ref_type = CrossReferenceType.TREATMENT_ALGORITHM

                        # Map strength
                        strength = ReferenceStrength.MODERATE  # Default
                        if "strong" in current_ref["strength"].lower():
                            strength = ReferenceStrength.STRONG
                        elif "weak" in current_ref["strength"].lower():
                            strength = ReferenceStrength.WEAK

                        ref = CrossReference(
                            reference_id=f"ai_ref_{ref_counter:04d}",
                            source_concept=current_ref["source"],
                            source_chapter_id=chapter_id,
                            target_concept=current_ref["target"],
                            target_chapter_id=target_chapter_id,
                            reference_type=ref_type,
                            strength=strength,
                            confidence=0.7,
                            description=f"AI-identified relationship between {current_ref['source']} and {current_ref['target']}",
                            clinical_relevance=current_ref["relevance"],
                            bidirectional=False,
                            context_keywords=["ai-generated"]
                        )
                        references.append(ref)
                        current_ref = {}  # Reset for next reference

        except Exception as e:
            logger.error(f"AI response parsing failed: {e}")

        return references

    async def _find_direct_relations(self, concept: str) -> List[str]:
        """Find directly related concepts for a given concept"""
        related_concepts = []
        concept_lower = concept.lower()

        # Check all relationship patterns
        for pattern_type, patterns in self.relationship_patterns.items():
            if concept_lower in patterns:
                related_concepts.extend(patterns[concept_lower])

        # Check clinical correlations
        for correlation_type, correlations in self.clinical_correlations.items():
            if concept_lower in correlations:
                related_concepts.extend(correlations[concept_lower])

        return list(set(related_concepts))

    async def _analyze_concept_relationship(self, concept1: str, concept2: str) -> Optional[CrossReference]:
        """Analyze the relationship between two concepts"""
        try:
            # Use AI to analyze relationship
            relationship_prompt = f"""Analyze the relationship between these neurosurgical concepts:

Concept 1: {concept1}
Concept 2: {concept2}

Please determine:
1. Is there a meaningful clinical relationship? (yes/no)
2. What type of relationship? (anatomical, surgical, diagnostic, pathological, etc.)
3. How strong is the relationship? (strong/moderate/weak)
4. What is the clinical relevance?

Respond with: RELATIONSHIP_EXISTS: yes/no, TYPE: [type], STRENGTH: [strength], RELEVANCE: [description]"""

            ai_response = await query_ai("gemini", relationship_prompt, max_tokens=200)

            # Parse response
            if "RELATIONSHIP_EXISTS: yes" in ai_response:
                # Extract relationship details
                ref_type = CrossReferenceType.ANATOMICAL_RELATION  # Default
                strength = ReferenceStrength.MODERATE  # Default
                relevance = "Related neurosurgical concepts"

                if "TYPE:" in ai_response:
                    type_text = ai_response.split("TYPE:")[1].split(",")[0].strip()
                    if "surgical" in type_text.lower():
                        ref_type = CrossReferenceType.SURGICAL_APPROACH
                    elif "diagnostic" in type_text.lower():
                        ref_type = CrossReferenceType.DIAGNOSTIC_PATHWAY

                if "STRENGTH:" in ai_response:
                    strength_text = ai_response.split("STRENGTH:")[1].split(",")[0].strip()
                    if "strong" in strength_text.lower():
                        strength = ReferenceStrength.STRONG
                    elif "weak" in strength_text.lower():
                        strength = ReferenceStrength.WEAK

                if "RELEVANCE:" in ai_response:
                    relevance = ai_response.split("RELEVANCE:")[1].strip()

                return CrossReference(
                    reference_id=f"concept_rel_{hash(concept1 + concept2)}",
                    source_concept=concept1,
                    source_chapter_id="",
                    target_concept=concept2,
                    target_chapter_id="",
                    reference_type=ref_type,
                    strength=strength,
                    confidence=0.6,
                    description=f"Conceptual relationship between {concept1} and {concept2}",
                    clinical_relevance=relevance,
                    bidirectional=True,
                    context_keywords=["conceptual"]
                )

            return None

        except Exception as e:
            logger.error(f"Concept relationship analysis failed: {e}")
            return None

    async def _build_concept_hierarchy(self, concepts: List[str]) -> Dict[str, List[str]]:
        """Build hierarchical relationships between concepts"""
        hierarchy = defaultdict(list)

        # Simple hierarchy based on concept types
        anatomical_concepts = []
        surgical_concepts = []
        disease_concepts = []
        imaging_concepts = []

        for concept in concepts:
            concept_lower = concept.lower()
            if any(anat in concept_lower for anat in ["lobe", "cortex", "spine", "nerve"]):
                anatomical_concepts.append(concept)
            elif any(surg in concept_lower for surg in ["surgery", "approach", "technique"]):
                surgical_concepts.append(concept)
            elif any(dis in concept_lower for dis in ["tumor", "aneurysm", "stenosis"]):
                disease_concepts.append(concept)
            elif any(img in concept_lower for img in ["MRI", "CT", "angiography"]):
                imaging_concepts.append(concept)

        hierarchy["Anatomy"] = anatomical_concepts
        hierarchy["Surgical Techniques"] = surgical_concepts
        hierarchy["Disease Conditions"] = disease_concepts
        hierarchy["Imaging"] = imaging_concepts

        return dict(hierarchy)

    async def _identify_clinical_pathways(self, concepts: List[str]) -> List[List[str]]:
        """Identify clinical pathways through the concepts"""
        pathways = []

        # Look for diagnostic pathways
        diagnostic_steps = []
        treatment_steps = []

        for concept in concepts:
            concept_lower = concept.lower()
            if any(diag in concept_lower for diag in ["CT", "MRI", "examination", "assessment"]):
                diagnostic_steps.append(concept)
            elif any(treat in concept_lower for treat in ["surgery", "treatment", "therapy"]):
                treatment_steps.append(concept)

        if diagnostic_steps:
            pathways.append(diagnostic_steps)
        if treatment_steps:
            pathways.append(treatment_steps)

        return pathways

    def _deduplicate_references(self, references: List[CrossReference]) -> List[CrossReference]:
        """Remove duplicate cross-references"""
        seen = set()
        unique_refs = []

        for ref in references:
            ref_key = (ref.source_concept.lower(), ref.target_concept.lower(), ref.reference_type)
            if ref_key not in seen:
                seen.add(ref_key)
                unique_refs.append(ref)

        return unique_refs

    def _rank_references(self, references: List[CrossReference]) -> List[CrossReference]:
        """Rank references by importance and relevance"""
        # Sort by strength, then confidence
        strength_order = {
            ReferenceStrength.STRONG: 3,
            ReferenceStrength.MODERATE: 2,
            ReferenceStrength.WEAK: 1,
            ReferenceStrength.CONTEXTUAL: 0
        }

        references.sort(
            key=lambda x: (strength_order.get(x.strength, 0), x.confidence),
            reverse=True
        )

        return references[:20]  # Limit to top 20 references

# Global instance
intelligent_cross_referencer = IntelligentCrossReferencer()