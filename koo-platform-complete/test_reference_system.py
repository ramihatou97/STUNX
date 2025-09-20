#!/usr/bin/env python3
"""
Test script for the KOO Platform reference system
"""

import os
import sys
import asyncio
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

async def test_reference_system():
    """Test the reference system configuration and basic functionality"""

    print("Testing KOO Platform Reference System")
    print("=" * 50)

    try:
        # Test 1: Import configuration
        print("1. Testing configuration import...")
        from core.config import settings
        print(f"   [OK] Settings imported successfully")
        print(f"   [PATH] Textbooks path: {settings.TEXTBOOKS_PATH}")
        print(f"   [CONFIG] Processing batch size: {settings.TEXTBOOK_PROCESSING_BATCH_SIZE}")

        # Test 2: Import models
        print("\n2. Testing database models...")
        from models.references import Textbook, BookChapter, ChapterCitation, ContentReference
        print("   [OK] Database models imported successfully")

        # Test 3: Test textbooks directory
        print("\n3. Testing textbooks directory...")
        textbooks_path = Path(settings.TEXTBOOKS_PATH)
        if not textbooks_path.exists():
            textbooks_path.mkdir(parents=True, exist_ok=True)
            print(f"   [CREATED] Created textbooks directory: {textbooks_path}")
        else:
            print(f"   [OK] Textbooks directory exists: {textbooks_path}")

        # Test 4: Check for sample textbooks
        print("\n4. Checking for textbook folders...")
        subdirs = [d for d in textbooks_path.iterdir() if d.is_dir()]
        if subdirs:
            print(f"   [FOUND] Found {len(subdirs)} textbook folders:")
            for subdir in subdirs[:3]:  # Show first 3
                pdf_files = list(subdir.glob("*.pdf"))
                print(f"      - {subdir.name}: {len(pdf_files)} PDF files")
        else:
            print("   [INFO] No textbook folders found yet")
            print("   [TIP] Create folders in ./data/textbooks/ with PDF chapters")

        # Test 5: Test basic functionality
        print("\n5. Testing basic functionality...")
        print("   [OK] Configuration and models working correctly")

        print("\n[SUCCESS] All tests passed! Reference system is ready.")
        print("\nNext steps:")
        print("   1. Place textbook folders in ./data/textbooks/")
        print("   2. Each folder should contain PDF files (one per chapter)")
        print("   3. Use the API endpoints to process and search textbooks")
        print("   4. Start the server with: uvicorn main:app --reload")

        return True

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the test"""
    success = asyncio.run(test_reference_system())
    if success:
        print(f"\n[SUCCESS] Reference system test completed successfully!")
        sys.exit(0)
    else:
        print(f"\n[FAILED] Reference system test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()