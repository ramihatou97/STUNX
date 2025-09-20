#!/usr/bin/env python3
"""
KOO Platform - AI Services Structure Test
Simple test to verify code structure and basic functionality
"""

import os
import sys
import json
from pathlib import Path

def test_file_exists(file_path: str, description: str):
    """Test if a file exists"""
    if os.path.exists(file_path):
        print(f"[PASS] {description}: {file_path}")
        return True
    else:
        print(f"[FAIL] {description}: {file_path} - NOT FOUND")
        return False

def test_python_syntax(file_path: str, description: str):
    """Test if Python file has valid syntax"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Basic syntax check by compiling
        compile(content, file_path, 'exec')
        print(f"[PASS] {description}: Valid Python syntax")
        return True
    except SyntaxError as e:
        print(f"[FAIL] {description}: Syntax error at line {e.lineno}: {e.msg}")
        return False
    except Exception as e:
        print(f"[FAIL] {description}: Error - {e}")
        return False

def test_class_definitions(file_path: str, expected_classes: list, description: str):
    """Test if expected classes are defined in the file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        found_classes = []
        for class_name in expected_classes:
            if f"class {class_name}" in content:
                found_classes.append(class_name)

        if len(found_classes) == len(expected_classes):
            print(f"[PASS] {description}: All expected classes found {found_classes}")
            return True
        else:
            missing = [c for c in expected_classes if c not in found_classes]
            print(f"[FAIL] {description}: Missing classes {missing}")
            return False
    except Exception as e:
        print(f"[FAIL] {description}: Error - {e}")
        return False

def test_api_endpoints(file_path: str, description: str):
    """Test if API endpoints are properly defined"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        endpoints = []
        lines = content.split('\n')
        for line in lines:
            if '@router.' in line and ('get' in line or 'post' in line):
                endpoints.append(line.strip())

        if endpoints:
            print(f"[PASS] {description}: Found {len(endpoints)} API endpoints")
            for endpoint in endpoints:
                print(f"    {endpoint}")
            return True
        else:
            print(f"[FAIL] {description}: No API endpoints found")
            return False
    except Exception as e:
        print(f"[FAIL] {description}: Error - {e}")
        return False

def main():
    """Main test runner"""
    print("=" * 60)
    print("KOO Platform - AI Services Structure Test")
    print("=" * 60)

    base_path = Path(__file__).parent

    # Test 1: Core AI Services Files
    print("\n1. Testing AI Services Files:")
    services_tests = [
        (base_path / "services" / "ai_chapter_generation.py", "AI Chapter Generation Service"),
        (base_path / "services" / "literature_summarization.py", "Literature Summarization Engine"),
        (base_path / "services" / "semantic_search.py", "Semantic Search Service"),
        (base_path / "services" / "smart_tagging.py", "Smart Tagging System"),
        (base_path / "services" / "intelligent_cross_referencing.py", "Intelligent Cross-Referencing"),
        (base_path / "services" / "hybrid_ai_manager.py", "Hybrid AI Manager"),
    ]

    services_passed = 0
    for file_path, description in services_tests:
        if test_file_exists(str(file_path), description):
            services_passed += 1

    # Test 2: Python Syntax Validation
    print("\n2. Testing Python Syntax:")
    syntax_passed = 0
    for file_path, description in services_tests:
        if os.path.exists(str(file_path)):
            if test_python_syntax(str(file_path), description):
                syntax_passed += 1

    # Test 3: Class Definitions
    print("\n3. Testing Class Definitions:")
    class_tests = [
        (base_path / "services" / "ai_chapter_generation.py",
         ["AIChapterGenerator", "ChapterType", "GenerationRequest"],
         "AI Chapter Generation Classes"),
        (base_path / "services" / "literature_summarization.py",
         ["LiteratureSummarizationEngine", "SummaryType", "SummaryRequest"],
         "Literature Summarization Classes"),
        (base_path / "services" / "semantic_search.py",
         ["SemanticSearchEngine", "SearchType", "SearchQuery"],
         "Semantic Search Classes"),
        (base_path / "services" / "smart_tagging.py",
         ["NeurosurgicalTagger", "TagType", "TaggingResult"],
         "Smart Tagging Classes"),
        (base_path / "services" / "intelligent_cross_referencing.py",
         ["IntelligentCrossReferencer", "CrossReferenceType", "CrossReference"],
         "Cross-Referencing Classes"),
    ]

    classes_passed = 0
    for file_path, expected_classes, description in class_tests:
        if os.path.exists(str(file_path)):
            if test_class_definitions(str(file_path), expected_classes, description):
                classes_passed += 1

    # Test 4: API Endpoints
    print("\n4. Testing API Endpoints:")
    api_file = base_path / "api" / "ai_knowledge.py"
    api_passed = 0
    if test_file_exists(str(api_file), "AI Knowledge API"):
        if test_api_endpoints(str(api_file), "AI Knowledge API Endpoints"):
            api_passed = 1

    # Test 5: Frontend Component
    print("\n5. Testing Frontend Component:")
    frontend_file = base_path.parent / "frontend" / "src" / "components" / "research" / "KnowledgeGraphVisualization.tsx"
    frontend_passed = 0
    if test_file_exists(str(frontend_file), "Knowledge Graph Visualization Component"):
        if test_python_syntax(str(frontend_file), "TypeScript Syntax Check (basic)"):
            frontend_passed = 1

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Services Files:       {services_passed}/{len(services_tests)}")
    print(f"Python Syntax:        {syntax_passed}/{len(services_tests)}")
    print(f"Class Definitions:    {classes_passed}/{len(class_tests)}")
    print(f"API Endpoints:        {api_passed}/1")
    print(f"Frontend Component:   {frontend_passed}/1")

    total_passed = services_passed + syntax_passed + classes_passed + api_passed + frontend_passed
    total_tests = len(services_tests) + len(services_tests) + len(class_tests) + 1 + 1

    print(f"\nOVERALL:              {total_passed}/{total_tests}")

    if total_passed == total_tests:
        print("\n[SUCCESS] ALL TESTS PASSED! The AI services structure is ready.")
    elif total_passed >= total_tests * 0.8:
        print("\n[MOSTLY OK] Most tests passed. Minor issues to resolve.")
    else:
        print("\n[ISSUES] Several tests failed. Review the issues above.")

    return total_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)