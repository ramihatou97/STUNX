# Caching & Task Management - Quick Start Guide

This guide provides a quick overview of the enhanced caching layer and background task management features in the KOO Platform.

## 🚀 Quick Start

### 1. Configuration

Add these environment variables to your `.env` file:

```env
# Redis Cache
REDIS_URL=redis://localhost:6379/0
REDIS_MAX_CONNECTIONS=20
CACHE_DEFAULT_TTL=3600

# Celery Tasks
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2
```

### 2. Start Services

```bash
# Start Redis
redis-server

# Start Celery Worker
cd backend
python celery_worker.py worker --loglevel=info

# Start Celery Beat (for scheduled tasks)
celery -A core.task_manager.celery_app beat --loglevel=info

# Start FastAPI
uvicorn main:app --reload
```

## 💾 Caching Examples

### Basic Cache Operations

```python
from core.redis_cache import redis_cache, CacheLevel

# Cache user data
await redis_cache.set("user_123", user_data, ttl=3600, level=CacheLevel.APPLICATION)

# Get cached data
user_data = await redis_cache.get("user_123", CacheLevel.APPLICATION)

# Cache API response
await redis_cache.set("search_results", results, ttl=300, level=CacheLevel.API)
```

### Using Cache Decorators

```python
from core.redis_cache import cached, CacheLevel

@cached(ttl=1800, level=CacheLevel.DATABASE)
async def get_chapter_content(chapter_id: str):
    # Expensive database operation
    return await database.get_chapter(chapter_id)

@cached(ttl=300, level=CacheLevel.API, key_prefix="research")
async def search_papers(query: str):
    # Expensive API call
    return await pubmed_api.search(query)
```

### Cache Management via API

```bash
# Check cache health
curl "http://localhost:8000/api/v1/cache/health"

# Get cache metrics
curl "http://localhost:8000/api/v1/cache/metrics"

# Warm cache with critical data
curl -X POST "http://localhost:8000/api/v1/cache/warm"

# Clean up expired entries
curl -X POST "http://localhost:8000/api/v1/cache/cleanup"
```

## 🔄 Background Tasks

### Submitting Tasks

```python
from core.task_manager import submit_ai_task, submit_pdf_processing_task

# Submit AI query task
task_id = await submit_ai_task(
    "koo.tasks.ai_service.process_query",
    "claude", "Explain brain anatomy"
)

# Submit PDF processing task
task_id = await submit_pdf_processing_task(
    "koo.tasks.pdf_processing.process_document",
    "/path/to/document.pdf", "doc_123"
)
```

### Task Management via API

```bash
# Submit AI query task
curl -X POST "http://localhost:8000/api/v1/tasks/ai/query" \
  -H "Content-Type: application/json" \
  -d '{"service": "claude", "prompt": "What is neuroplasticity?"}'

# Check task status
curl "http://localhost:8000/api/v1/tasks/status/{task_id}"

# Get active tasks
curl "http://localhost:8000/api/v1/tasks/active"

# Get task metrics
curl "http://localhost:8000/api/v1/tasks/metrics"
```

## 📊 Monitoring

### Cache Monitoring

```bash
# Get comprehensive cache metrics
curl "http://localhost:8000/api/v1/cache/metrics"

# Response includes:
{
  "metrics": {
    "global": {
      "hits": 1250,
      "misses": 180,
      "hit_rate": 0.87,
      "avg_response_time_ms": 2.3
    },
    "by_level": {
      "application": {...},
      "database": {...},
      "api": {...}
    }
  }
}
```

### Task Monitoring

```bash
# Get task execution metrics
curl "http://localhost:8000/api/v1/tasks/metrics"

# Response includes:
{
  "metrics": {
    "global": {
      "total_tasks": 450,
      "successful_tasks": 425,
      "failed_tasks": 15,
      "success_rate": 0.94,
      "avg_execution_time": 12.5
    },
    "by_category": {
      "ai_service": {...},
      "pdf_processing": {...}
    }
  }
}
```

## 🔧 Common Operations

### Cache Warming

```python
# Warm frequently accessed data
from core.redis_cache import redis_cache, CacheLevel

# Preload critical chapters
critical_chapters = ["intro", "anatomy", "procedures"]
for chapter_id in critical_chapters:
    chapter_data = await get_chapter_content(chapter_id)
    await redis_cache.set(
        f"chapter:{chapter_id}",
        chapter_data,
        ttl=7200,  # 2 hours
        level=CacheLevel.APPLICATION
    )
```

### Batch Task Processing

```python
# Process multiple documents
from core.task_manager import submit_pdf_processing_task

pdf_files = ["/path/to/doc1.pdf", "/path/to/doc2.pdf"]
task_ids = []

for i, pdf_path in enumerate(pdf_files):
    task_id = await submit_pdf_processing_task(
        "koo.tasks.pdf_processing.process_document",
        pdf_path, f"batch_doc_{i}"
    )
    task_ids.append(task_id)

# Monitor batch progress
for task_id in task_ids:
    status = await task_manager.get_task_status(task_id)
    print(f"Task {task_id}: {status.status}")
```

### Health Checks

```bash
# Run comprehensive health check
curl -X POST "http://localhost:8000/api/v1/tasks/health-check"

# Check specific service health
curl "http://localhost:8000/api/v1/monitoring/health"
```

## 🚨 Troubleshooting

### Cache Issues

```bash
# Check cache health
curl "http://localhost:8000/api/v1/cache/health"

# Clear problematic cache level
curl -X POST "http://localhost:8000/api/v1/cache/clear?level=api"

# Run cache maintenance
curl -X POST "http://localhost:8000/api/v1/cache/maintenance"
```

### Task Issues

```bash
# Check failed tasks
curl "http://localhost:8000/api/v1/tasks/history?status_filter=FAILURE"

# Retry failed task
curl -X POST "http://localhost:8000/api/v1/tasks/retry/{task_id}"

# Clean up old completed tasks
curl -X POST "http://localhost:8000/api/v1/tasks/cleanup?hours_old=24"
```

### Performance Optimization

```python
# Monitor cache hit rates
metrics = await redis_cache.get_metrics()
if metrics['global']['hit_rate'] < 0.8:
    # Consider adjusting TTL values or warming strategies
    pass

# Monitor task execution times
task_metrics = task_manager.get_task_metrics()
if task_metrics['global']['avg_execution_time'] > 30:
    # Consider optimizing task implementations
    pass
```

## 📈 Performance Tips

### Cache Optimization
- Use appropriate TTL values (short for dynamic data, long for static)
- Implement cache warming for frequently accessed data
- Monitor hit rates and adjust strategies accordingly
- Use compression for large datasets

### Task Optimization
- Use appropriate task priorities
- Implement proper error handling and retries
- Monitor task execution times
- Scale workers based on queue length

### Monitoring Best Practices
- Set up alerts for low cache hit rates
- Monitor task failure rates
- Track system resource usage
- Implement automated maintenance schedules

This quick start guide covers the essential operations for the enhanced caching and task management system. For detailed documentation, see [CACHING_AND_TASK_MANAGEMENT.md](./CACHING_AND_TASK_MANAGEMENT.md).
