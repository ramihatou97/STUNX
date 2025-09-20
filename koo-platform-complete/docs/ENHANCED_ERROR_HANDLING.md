# Enhanced Database Connection Pooling and AI Service Error Handling

This document describes the comprehensive error handling and monitoring system implemented for the KOO Platform, including enhanced database connection pooling and AI service error handling with circuit breakers, retry logic, and rate limiting.

## Overview

The enhanced error handling system provides:

- **Database Connection Pooling**: Advanced connection pool management with monitoring, circuit breakers, and automatic recovery
- **AI Service Error Handling**: Comprehensive error handling for AI services with retry logic, circuit breakers, and rate limiting
- **Health Monitoring**: Real-time health monitoring and alerting for all services
- **Metrics Collection**: Detailed metrics collection and reporting for performance analysis
- **Automatic Recovery**: Intelligent recovery mechanisms for transient failures

## Database Connection Pooling

### Features

1. **Enhanced Connection Pool Configuration**
   - Configurable pool size, overflow, and timeout settings
   - Connection pre-ping for health verification
   - Automatic connection recycling
   - Connection event monitoring

2. **Circuit Breaker Pattern**
   - Automatic failure detection and circuit opening
   - Configurable failure thresholds and recovery timeouts
   - Graceful degradation during outages
   - Automatic recovery testing

3. **Retry Logic with Exponential Backoff**
   - Configurable retry attempts and delays
   - Exponential backoff with jitter
   - Smart retry decisions based on error types
   - Maximum delay caps to prevent excessive waits

4. **Connection Pool Metrics**
   - Real-time connection pool status
   - Connection checkout/checkin tracking
   - Average response time monitoring
   - Failure rate tracking

### Configuration

Environment variables for database connection pooling:

```env
# Database Pool Configuration
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20
DATABASE_POOL_TIMEOUT=30
DATABASE_POOL_RECYCLE=3600

# Circuit Breaker Configuration
DB_CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
DB_CIRCUIT_BREAKER_RECOVERY_TIMEOUT=60
DB_CIRCUIT_BREAKER_SUCCESS_THRESHOLD=2

# Retry Configuration
DB_MAX_RETRIES=3
DB_RETRY_BASE_DELAY=1.0
DB_RETRY_MAX_DELAY=30.0
DB_RETRY_BACKOFF_MULTIPLIER=2.0

# Health Check Configuration
DB_HEALTH_CHECK_INTERVAL=30
HEALTH_CHECK_TIMEOUT=5
```

### Usage

#### Basic Session Management

```python
from core.database import db_manager

# Get a session with circuit breaker protection
async with db_manager.session_scope() as session:
    # Your database operations here
    result = await session.execute(query)
```

#### Session with Retry Logic

```python
from core.database import transactional_with_retry

@transactional_with_retry(max_retries=5)
async def my_database_operation(session):
    # This operation will be retried automatically on transient failures
    return await session.execute(query)
```

#### Health Monitoring

```python
from core.database import check_database_health

# Check database health
health_status = await check_database_health()
print(f"Database healthy: {health_status['healthy']}")

# Get detailed metrics
metrics = db_manager.get_pool_metrics()
print(f"Active connections: {metrics['pool_status']['active_connections']}")
```

## AI Service Error Handling

### Features

1. **Circuit Breaker Pattern**
   - Per-service circuit breakers
   - Configurable failure thresholds
   - Automatic recovery testing
   - State transitions (Healthy → Unhealthy → Circuit Open → Degraded → Healthy)

2. **Retry Logic with Exponential Backoff**
   - Smart retry decisions based on error types
   - Exponential backoff with jitter
   - Different retry strategies for different error types
   - Maximum retry limits

3. **Rate Limiting and Cost Management**
   - Per-minute, per-hour, and per-day rate limits
   - Daily budget tracking
   - Cost-per-request monitoring
   - Automatic throttling when limits are reached

4. **Service Health Monitoring**
   - Real-time service status tracking
   - Error classification and counting
   - Response time monitoring
   - Success rate calculation

### Configuration

Environment variables for AI service error handling:

```env
# Circuit Breaker Configuration
AI_CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
AI_CIRCUIT_BREAKER_RECOVERY_TIMEOUT=60
AI_CIRCUIT_BREAKER_SUCCESS_THRESHOLD=2

# Retry Configuration
AI_MAX_RETRIES=3
AI_RETRY_BASE_DELAY=1.0
AI_RETRY_MAX_DELAY=30.0
AI_RETRY_BACKOFF_MULTIPLIER=2.0

# Rate Limiting Configuration
AI_DEFAULT_REQUESTS_PER_MINUTE=60
AI_DEFAULT_REQUESTS_PER_HOUR=1000
AI_DEFAULT_REQUESTS_PER_DAY=10000
AI_DEFAULT_DAILY_BUDGET=10.0

# Health Check Configuration
AI_HEALTH_CHECK_INTERVAL=60
```

### Usage

#### Basic AI Service Query

```python
from services.hybrid_ai_manager import query_ai

# Query with automatic error handling
try:
    response = await query_ai("claude", "Your prompt here")
    print(response)
except ExternalServiceError as e:
    print(f"Service error: {e}")
```

#### Service Health Monitoring

```python
from services.hybrid_ai_manager import get_ai_service_health

# Get service health status
health = get_ai_service_health("claude")
print(f"Service state: {health['state']}")
print(f"Success rate: {health['metrics']['success_rate']}")
```

#### Manual Error Recovery

```python
from services.hybrid_ai_manager import reset_ai_service_errors

# Reset error state for a service
success = reset_ai_service_errors("claude")
if success:
    print("Service errors reset successfully")
```

## Monitoring and Alerting

### API Endpoints

The system provides comprehensive monitoring endpoints:

#### System Health
- `GET /api/v1/monitoring/health` - Overall system health
- `GET /api/v1/monitoring/dashboard` - Comprehensive dashboard data
- `GET /api/v1/monitoring/alerts` - Current system alerts

#### Database Monitoring
- `GET /api/v1/monitoring/database` - Database monitoring data
- `GET /api/v1/monitoring/database/pool-metrics` - Pool metrics
- `POST /api/v1/monitoring/database/health-check` - Active health check
- `POST /api/v1/monitoring/database/reset-circuit-breaker` - Reset circuit breaker

#### AI Services Monitoring
- `GET /api/v1/monitoring/ai-services` - AI services monitoring data
- `GET /api/v1/monitoring/ai-services/{service}/metrics` - Service-specific metrics
- `GET /api/v1/ai/services/health` - AI services health status
- `POST /api/v1/ai/services/{service}/reset-errors` - Reset service errors

### Metrics and Alerts

#### Database Metrics
- Connection pool status (active, idle, overflow connections)
- Connection checkout/checkin rates
- Average response times
- Circuit breaker state
- Failure rates

#### AI Service Metrics
- Request success/failure rates
- Response times
- Error classification and counts
- Rate limit status
- Cost tracking
- Circuit breaker states

#### Alert Types
- **Critical**: Circuit breakers open, service unavailable
- **Warning**: Degraded performance, approaching rate limits
- **Info**: Service recovery, configuration changes

## Error Types and Handling

### Database Errors
- **Connection Errors**: Automatic retry with exponential backoff
- **Timeout Errors**: Circuit breaker activation, retry logic
- **Integrity Errors**: No retry, immediate failure
- **Operational Errors**: Retry with backoff

### AI Service Errors
- **API Key Errors**: No retry, immediate failure
- **Rate Limit Errors**: Extended backoff, automatic throttling
- **Timeout Errors**: Retry with backoff
- **Connection Errors**: Retry with backoff
- **Quota Exceeded**: No retry, budget management
- **Service Unavailable**: Circuit breaker activation

## Best Practices

### Database Operations
1. Use `session_scope()` for automatic transaction management
2. Use `transactional_with_retry()` for operations that may fail transiently
3. Monitor pool metrics regularly
4. Set appropriate timeout values
5. Use health checks for proactive monitoring

### AI Service Operations
1. Handle `ExternalServiceError` exceptions appropriately
2. Monitor service health and error rates
3. Set realistic rate limits and budgets
4. Use fallback strategies when services are unavailable
5. Reset error states when issues are resolved

### Monitoring
1. Set up alerts for critical states
2. Monitor success rates and response times
3. Track cost and usage patterns
4. Review error logs regularly
5. Use dashboard for operational visibility

## Troubleshooting

### Common Issues

#### Database Connection Issues
- Check pool configuration and limits
- Verify database connectivity
- Review circuit breaker state
- Check for connection leaks

#### AI Service Issues
- Verify API keys and quotas
- Check rate limit status
- Review error classifications
- Monitor circuit breaker states

#### Performance Issues
- Monitor response times
- Check pool utilization
- Review retry patterns
- Analyze error rates

### Recovery Procedures

#### Database Recovery
1. Check database server status
2. Reset circuit breaker if needed
3. Verify connection pool health
4. Monitor recovery progress

#### AI Service Recovery
1. Check service provider status
2. Reset service error states
3. Verify API key validity
4. Monitor service health

## Configuration Reference

See the configuration files for complete settings:
- `backend/core/config.py` - Main configuration
- `.env.example` - Environment variable examples

For more detailed information, refer to the source code documentation in:
- `backend/core/database.py` - Database connection pooling
- `backend/core/ai_error_handling.py` - AI service error handling
- `backend/services/hybrid_ai_manager.py` - AI service management
- `backend/api/monitoring.py` - Monitoring endpoints
