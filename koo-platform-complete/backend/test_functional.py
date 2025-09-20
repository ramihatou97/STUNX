#!/usr/bin/env python3
"""
KOO Platform - AI Services Functional Test
Test AI services with mock data and simulated responses
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Mock modules for testing without external dependencies
class MockSettings:
    ENVIRONMENT = "test"
    LOG_LEVEL = "INFO"
    DEBUG = True
    OPENAI_API_KEY = "test-key"
    ANTHROPIC_API_KEY = "test-key"
    GOOGLE_AI_API_KEY = "test-key"
    PERPLEXITY_API_KEY = "test-key"

class MockHybridAI:
    @staticmethod
    async def query_ai(prompt: str, **kwargs) -> str:
        """Mock AI response"""
        if "chapter" in prompt.lower():
            return """# Neurosurgical Chapter: Mock Topic

## Introduction
This is a mock neurosurgical chapter generated for testing purposes.

## Anatomy
The anatomical structures involved include...

## Surgical Technique
The surgical approach involves the following steps:
1. Patient positioning
2. Incision planning
3. Craniotomy
4. Tumor resection

## Conclusion
This concludes the mock chapter content."""

        elif "summarize" in prompt.lower():
            return """## Literature Summary

**Key Findings:**
- Finding 1: Important neurosurgical insight
- Finding 2: Surgical technique improvement
- Finding 3: Patient outcome enhancement

**Evidence Level:** Level 2B
**Number of Studies:** 15 mock studies
**Clinical Implications:** Significant for neurosurgical practice"""

        elif "search" in prompt.lower():
            return """Related neurosurgical concepts:
- Craniotomy techniques
- Tumor resection protocols
- Postoperative care guidelines"""

        elif "tag" in prompt.lower():
            return """Suggested tags:
- neuroanatomy
- surgical-technique
- tumor-resection
- postoperative-care"""

        elif "cross-reference" in prompt.lower():
            return """Cross-references:
- Related Chapter: Advanced Craniotomy Techniques
- Reference Paper: Modern Neurosurgical Approaches
- Clinical Guideline: Postoperative Monitoring"""

        return "Mock AI response for testing"

# Replace imports with mocks
sys.modules['core.config'] = type('MockModule', (), {'settings': MockSettings()})()
sys.modules['core.exceptions'] = type('MockModule', (), {
    'ExternalServiceError': Exception,
    'ValidationError': Exception
})()

# Mock AI manager
async def mock_query_ai(prompt: str, **kwargs) -> str:
    return await MockHybridAI.query_ai(prompt, **kwargs)

# Add backend to path
sys.path.append(str(Path(__file__).parent))

# Mock the hybrid_ai_manager module
class MockAIModule:
    query_ai = staticmethod(mock_query_ai)

sys.modules['services.hybrid_ai_manager'] = MockAIModule()

# Mock analytics modules
class MockPubMedAnalytics:
    async def analyze_literature(self, topic: str, **kwargs):
        return {
            "papers": [
                {"title": "Mock Paper 1", "authors": ["Dr. Test"], "year": 2023},
                {"title": "Mock Paper 2", "authors": ["Dr. Example"], "year": 2024}
            ],
            "total_papers": 2,
            "analysis": "Mock analysis result"
        }

sys.modules['services.advanced_pubmed_analytics'] = type('MockModule', (), {
    'AdvancedPubMedAnalytics': MockPubMedAnalytics
})()

async def test_ai_chapter_generation():
    """Test AI Chapter Generation Service"""
    try:
        from services.ai_chapter_generation import AIChapterGenerator, ChapterType, GenerationRequest

        generator = AIChapterGenerator()
        request = GenerationRequest(
            topic="Mock Craniotomy Technique",
            chapter_type=ChapterType.SURGICAL_TECHNIQUE,
            target_length=1000,
            include_references=True
        )

        result = await generator.generate_chapter(request)

        assert result.title, "Chapter should have a title"
        assert result.content, "Chapter should have content"
        assert len(result.content) > 100, "Chapter content should be substantial"

        print(f"[PASS] AI Chapter Generation: Generated chapter '{result.title}' ({len(result.content)} chars)")
        return True

    except Exception as e:
        print(f"[FAIL] AI Chapter Generation: {e}")
        return False

async def test_literature_summarization():
    """Test Literature Summarization Engine"""
    try:
        from services.literature_summarization import LiteratureSummarizationEngine, SummaryType, SummaryRequest

        engine = LiteratureSummarizationEngine()
        request = SummaryRequest(
            topic="Neurosurgical Outcomes",
            summary_type=SummaryType.TECHNICAL,
            max_papers=10
        )

        result = await engine.summarize_literature(request)

        assert result.summary, "Should have summary content"
        assert result.key_findings, "Should have key findings"
        assert result.evidence_level, "Should have evidence level"

        print(f"[PASS] Literature Summarization: Generated summary with {len(result.key_findings)} findings")
        return True

    except Exception as e:
        print(f"[FAIL] Literature Summarization: {e}")
        return False

async def test_semantic_search():
    """Test Semantic Search Service"""
    try:
        from services.semantic_search import SemanticSearchEngine, SearchType, SearchQuery

        engine = SemanticSearchEngine()
        query = SearchQuery(
            query="craniotomy surgical technique",
            search_type=SearchType.COMPREHENSIVE,
            max_results=5
        )

        result = await engine.search(query)

        assert result.results, "Should have search results"
        assert result.total_results > 0, "Should have positive result count"

        print(f"[PASS] Semantic Search: Found {result.total_results} results")
        return True

    except Exception as e:
        print(f"[FAIL] Semantic Search: {e}")
        return False

async def test_smart_tagging():
    """Test Smart Tagging System"""
    try:
        from services.smart_tagging import NeurosurgicalTagger, TagType, TaggingResult

        tagger = NeurosurgicalTagger()
        content = "This chapter discusses advanced craniotomy techniques for tumor resection in the frontal lobe."

        result = await tagger.tag_content(content, include_hierarchical=True)

        assert result.tags, "Should have generated tags"
        assert any(tag.tag_type == TagType.ANATOMICAL for tag in result.tags), "Should have anatomical tags"

        print(f"[PASS] Smart Tagging: Generated {len(result.tags)} tags")
        return True

    except Exception as e:
        print(f"[FAIL] Smart Tagging: {e}")
        return False

async def test_cross_referencing():
    """Test Intelligent Cross-Referencing"""
    try:
        from services.intelligent_cross_referencing import IntelligentCrossReferencer, CrossReferenceType

        referencer = IntelligentCrossReferencer()
        content = "Advanced craniotomy techniques for frontal lobe tumor resection"

        result = await referencer.generate_cross_references(content)

        assert result.cross_references, "Should have cross-references"
        assert result.concept_map, "Should have concept map"

        print(f"[PASS] Cross-Referencing: Generated {len(result.cross_references)} cross-references")
        return True

    except Exception as e:
        print(f"[FAIL] Cross-Referencing: {e}")
        return False

async def test_knowledge_graph():
    """Test Knowledge Graph Generation"""
    try:
        from services.intelligent_cross_referencing import IntelligentCrossReferencer

        referencer = IntelligentCrossReferencer()
        topics = ["craniotomy", "tumor resection", "frontal lobe", "surgical technique"]

        result = await referencer.generate_concept_graph(topics)

        assert result.nodes, "Should have graph nodes"
        assert result.edges, "Should have graph edges"

        print(f"[PASS] Knowledge Graph: Generated graph with {len(result.nodes)} nodes and {len(result.edges)} edges")
        return True

    except Exception as e:
        print(f"[FAIL] Knowledge Graph: {e}")
        return False

async def main():
    """Main test runner"""
    print("=" * 60)
    print("KOO Platform - AI Services Functional Test")
    print("=" * 60)

    tests = [
        ("AI Chapter Generation", test_ai_chapter_generation),
        ("Literature Summarization", test_literature_summarization),
        ("Semantic Search", test_semantic_search),
        ("Smart Tagging", test_smart_tagging),
        ("Cross-Referencing", test_cross_referencing),
        ("Knowledge Graph", test_knowledge_graph),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\nTesting {test_name}...")
        try:
            if await test_func():
                passed += 1
        except Exception as e:
            print(f"[FAIL] {test_name}: Unexpected error - {e}")

    print("\n" + "=" * 60)
    print("FUNCTIONAL TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n[SUCCESS] All functional tests passed! AI services are working correctly.")
    elif passed >= total * 0.8:
        print("\n[MOSTLY OK] Most tests passed. Minor issues to investigate.")
    else:
        print("\n[ISSUES] Several tests failed. Review the implementation.")

    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)