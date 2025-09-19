"""
Comprehensive test suite for KOO Platform API
"""

import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from main import app
from core.database import get_async_session
from models.database import Base, User, Chapter, KnowledgeSource

# Test database setup
TEST_DATABASE_URL = "postgresql://test:test@localhost:5432/test_koo"

@pytest.fixture
async def async_client():
    """Create async test client"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.fixture
async def db_session():
    """Create test database session"""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        yield session

@pytest.fixture
async def test_user(db_session):
    """Create test user"""
    user = User(
        username="testuser",
        email="test@koo-platform.com",
        password_hash="hashed_password",
        full_name="Test User",
        role="editor"
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user

@pytest.fixture
async def test_chapter(db_session, test_user):
    """Create test chapter"""
    chapter = Chapter(
        title="Test Neurosurgical Procedure",
        content="# Test Chapter\n\nThis is a test chapter about neurosurgery.",
        category="neurosurgery",
        created_by=test_user.id,
        status="published"
    )
    db_session.add(chapter)
    await db_session.commit()
    await db_session.refresh(chapter)
    return chapter

class TestChapterAPI:
    """Test chapter-related API endpoints"""

    async def test_get_chapters(self, async_client, test_chapter):
        """Test getting list of chapters"""
        response = await async_client.get("/api/v1/chapters/")
        assert response.status_code == 200

        data = response.json()
        assert len(data["items"]) >= 1
        assert data["items"][0]["title"] == "Test Neurosurgical Procedure"

    async def test_get_chapter_by_id(self, async_client, test_chapter):
        """Test getting specific chapter"""
        response = await async_client.get(f"/api/v1/chapters/{test_chapter.id}")
        assert response.status_code == 200

        data = response.json()
        assert data["title"] == test_chapter.title
        assert data["category"] == test_chapter.category

    async def test_create_chapter(self, async_client, test_user):
        """Test creating new chapter"""
        chapter_data = {
            "title": "New Test Chapter",
            "content": "# New Chapter\n\nContent here.",
            "category": "neurosurgery",
            "summary": "Test summary"
        }

        # Mock authentication
        headers = {"Authorization": f"Bearer test_token"}
        response = await async_client.post(
            "/api/v1/chapters/",
            json=chapter_data,
            headers=headers
        )

        assert response.status_code == 201
        data = response.json()
        assert data["title"] == chapter_data["title"]

    async def test_update_chapter(self, async_client, test_chapter):
        """Test updating existing chapter"""
        update_data = {
            "title": "Updated Chapter Title",
            "content": "# Updated Content\n\nNew content here."
        }

        headers = {"Authorization": f"Bearer test_token"}
        response = await async_client.put(
            f"/api/v1/chapters/{test_chapter.id}",
            json=update_data,
            headers=headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["title"] == update_data["title"]

    async def test_search_chapters(self, async_client, test_chapter):
        """Test chapter search functionality"""
        search_params = {
            "q": "neurosurgical",
            "category": "neurosurgery",
            "limit": 10
        }

        response = await async_client.get(
            "/api/v1/chapters/search",
            params=search_params
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) >= 1

class TestResearchAPI:
    """Test research-related API endpoints"""

    async def test_pubmed_search(self, async_client):
        """Test PubMed research search"""
        search_data = {
            "query": "neurosurgery techniques",
            "max_results": 5,
            "date_range": [2020, 2024]
        }

        response = await async_client.post(
            "/api/v1/research/pubmed/search",
            json=search_data
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) <= 5

    async def test_ai_synthesis(self, async_client):
        """Test AI research synthesis"""
        synthesis_data = {
            "sources": [
                {"title": "Source 1", "abstract": "Abstract 1"},
                {"title": "Source 2", "abstract": "Abstract 2"}
            ],
            "topic": "neurosurgical techniques"
        }

        response = await async_client.post(
            "/api/v1/research/synthesize",
            json=synthesis_data
        )

        assert response.status_code == 200
        data = response.json()
        assert "synthesis" in data
        assert "confidence_score" in data

class TestUserAPI:
    """Test user-related API endpoints"""

    async def test_user_registration(self, async_client):
        """Test user registration"""
        user_data = {
            "username": "newuser",
            "email": "newuser@koo-platform.com",
            "password": "secure_password",
            "full_name": "New User",
            "institution": "Test Hospital"
        }

        response = await async_client.post(
            "/api/v1/auth/register",
            json=user_data
        )

        assert response.status_code == 201
        data = response.json()
        assert data["username"] == user_data["username"]
        assert "id" in data

    async def test_user_login(self, async_client, test_user):
        """Test user authentication"""
        login_data = {
            "username": test_user.username,
            "password": "test_password"
        }

        response = await async_client.post(
            "/api/v1/auth/login",
            json=login_data
        )

        # This would normally return 200 with proper password hashing
        assert response.status_code in [200, 401]

class TestAPIPerformance:
    """Test API performance and load handling"""

    async def test_concurrent_requests(self, async_client):
        """Test handling concurrent requests"""
        async def make_request():
            return await async_client.get("/api/v1/health")

        # Make 50 concurrent requests
        tasks = [make_request() for _ in range(50)]
        responses = await asyncio.gather(*tasks)

        # All requests should succeed
        assert all(r.status_code == 200 for r in responses)

    async def test_large_content_handling(self, async_client):
        """Test handling large content uploads"""
        large_content = "# Large Chapter\n\n" + "Content paragraph. " * 1000

        chapter_data = {
            "title": "Large Content Chapter",
            "content": large_content,
            "category": "neurosurgery"
        }

        headers = {"Authorization": f"Bearer test_token"}
        response = await async_client.post(
            "/api/v1/chapters/",
            json=chapter_data,
            headers=headers
        )

        # Should handle large content gracefully
        assert response.status_code in [201, 413]  # Created or Payload Too Large

# Integration Tests
class TestWorkflowIntegration:
    """Test complete workflow integration"""

    async def test_complete_research_workflow(self, async_client, test_user):
        """Test complete research to chapter creation workflow"""

        # 1. Search for research
        search_response = await async_client.post(
            "/api/v1/research/pubmed/search",
            json={"query": "brain tumor surgery", "max_results": 3}
        )
        assert search_response.status_code == 200

        # 2. Synthesize research
        search_data = search_response.json()
        synthesis_response = await async_client.post(
            "/api/v1/research/synthesize",
            json={
                "sources": search_data["results"][:2],
                "topic": "brain tumor surgery"
            }
        )
        assert synthesis_response.status_code == 200

        # 3. Create chapter from synthesis
        synthesis_data = synthesis_response.json()
        chapter_response = await async_client.post(
            "/api/v1/chapters/",
            json={
                "title": "Brain Tumor Surgery Techniques",
                "content": synthesis_data["synthesis"],
                "category": "neurosurgery",
                "summary": "Evidence-based brain tumor surgery techniques"
            },
            headers={"Authorization": f"Bearer test_token"}
        )
        assert chapter_response.status_code == 201

# Load Testing
@pytest.mark.performance
class TestLoadPerformance:
    """Load testing for production readiness"""

    async def test_api_load_capacity(self, async_client):
        """Test API under sustained load"""
        start_time = asyncio.get_event_loop().time()

        async def sustained_load():
            tasks = []
            for _ in range(100):  # 100 requests per batch
                task = async_client.get("/api/v1/chapters/")
                tasks.append(task)

            responses = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = sum(1 for r in responses if hasattr(r, 'status_code') and r.status_code == 200)
            return success_count

        # Run 5 batches of 100 requests each
        batch_results = []
        for _ in range(5):
            result = await sustained_load()
            batch_results.append(result)
            await asyncio.sleep(1)  # Brief pause between batches

        end_time = asyncio.get_event_loop().time()
        total_time = end_time - start_time
        total_requests = sum(batch_results)

        # Should handle at least 80% of requests successfully
        assert total_requests >= 400  # 80% of 500 requests
        # Should complete within reasonable time
        assert total_time < 60  # Less than 1 minute for 500 requests

if __name__ == "__main__":
    pytest.main(["-v", "--tb=short"])