# Enhanced Error Handling - Quick Start Guide

This guide provides a quick overview of the enhanced database connection pooling and AI service error handling features implemented in the KOO Platform.

## 🚀 Quick Start

### 1. Configuration

Add these environment variables to your `.env` file:

```env
# Database Connection Pool
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20
DB_CIRCUIT_BREAKER_FAILURE_THRESHOLD=5

# AI Service Error Handling
AI_CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
AI_MAX_RETRIES=3
AI_DEFAULT_DAILY_BUDGET=10.0
```

### 2. Database Operations

```python
# Automatic retry and circuit breaker protection
from core.database import db_manager

async with db_manager.session_scope() as session:
    result = await session.execute(query)
    # Automatic commit/rollback and error handling
```

### 3. AI Service Operations

```python
# Enhanced error handling for AI services
from services.hybrid_ai_manager import query_ai

try:
    response = await query_ai("claude", "Your prompt")
except ExternalServiceError as e:
    # Handle service errors gracefully
    print(f"AI service error: {e}")
```

### 4. Monitoring

```python
# Check system health
from api.monitoring import get_system_health

health = await get_system_health()
print(f"System health: {health['overall_health']}")
```

## 🔧 Key Features

### Database Connection Pooling
- ✅ **Circuit Breaker**: Automatic failure detection and recovery
- ✅ **Retry Logic**: Exponential backoff for transient failures
- ✅ **Pool Monitoring**: Real-time connection metrics
- ✅ **Health Checks**: Proactive health monitoring

### AI Service Error Handling
- ✅ **Circuit Breaker**: Per-service failure protection
- ✅ **Rate Limiting**: Cost and quota management
- ✅ **Smart Retries**: Error-type specific retry logic
- ✅ **Health Monitoring**: Service status tracking

### Monitoring & Alerting
- ✅ **Real-time Metrics**: Performance and health data
- ✅ **Alert System**: Critical and warning notifications
- ✅ **Dashboard API**: Comprehensive monitoring endpoints
- ✅ **Recovery Tools**: Manual reset and recovery options

## 📊 Monitoring Endpoints

### System Health
```bash
# Overall system health
GET /api/v1/monitoring/health

# Monitoring dashboard
GET /api/v1/monitoring/dashboard

# System alerts
GET /api/v1/monitoring/alerts
```

### Database Monitoring
```bash
# Database health and metrics
GET /api/v1/monitoring/database

# Connection pool metrics
GET /api/v1/monitoring/database/pool-metrics

# Reset database circuit breaker
POST /api/v1/monitoring/database/reset-circuit-breaker
```

### AI Services Monitoring
```bash
# AI services health
GET /api/v1/ai/services/health

# Service-specific metrics
GET /api/v1/monitoring/ai-services/{service}/metrics

# Reset service errors
POST /api/v1/ai/services/{service}/reset-errors
```

## 🛠️ Common Operations

### Check Database Health
```python
from core.database import check_database_health

health = await check_database_health()
if not health['healthy']:
    print("Database issues detected!")
```

### Monitor AI Service Status
```python
from services.hybrid_ai_manager import get_ai_service_health

status = get_ai_service_health("claude")
print(f"Claude status: {status['state']}")
print(f"Success rate: {status['metrics']['success_rate']}")
```

### Reset Service Errors
```python
from services.hybrid_ai_manager import reset_ai_service_errors

# Reset errors for a specific service
success = reset_ai_service_errors("claude")
if success:
    print("Claude errors reset successfully")
```

### Get System Metrics
```python
from core.database import db_manager

# Database metrics
db_metrics = db_manager.get_pool_metrics()
print(f"Active connections: {db_metrics['pool_status']['active_connections']}")

# AI service metrics
from core.ai_error_handling import ai_error_handler
ai_metrics = ai_error_handler.get_all_services_status()
```

## 🚨 Alert Types

### Critical Alerts
- 🔴 **Circuit Breaker Open**: Service completely unavailable
- 🔴 **Database Unavailable**: Database connection failures
- 🔴 **Service Down**: AI service completely unresponsive

### Warning Alerts
- 🟡 **Degraded Performance**: High error rates or slow responses
- 🟡 **Rate Limit Approaching**: Near quota limits
- 🟡 **Circuit Breaker Degraded**: Service in recovery mode

### Info Alerts
- 🟢 **Service Recovered**: Service back to healthy state
- 🟢 **Circuit Breaker Closed**: Automatic recovery completed

## 🔄 Circuit Breaker States

### Database Circuit Breaker
1. **Healthy** → Normal operation
2. **Unhealthy** → Errors detected, monitoring closely
3. **Circuit Open** → Too many failures, blocking requests
4. **Degraded** → Testing recovery, limited requests allowed

### AI Service Circuit Breaker
1. **Healthy** → Service operating normally
2. **Degraded** → Some errors, reduced confidence
3. **Unhealthy** → High error rate, monitoring
4. **Circuit Open** → Service blocked due to failures
5. **Rate Limited** → Quota/budget limits reached

## 📈 Performance Metrics

### Database Metrics
- Connection pool utilization
- Average response times
- Error rates and types
- Circuit breaker state changes

### AI Service Metrics
- Request success/failure rates
- Response times per service
- Cost tracking and budget usage
- Rate limit status

## 🔧 Troubleshooting

### Database Issues
```bash
# Check database health
curl -X GET "http://localhost:8000/api/v1/monitoring/database"

# Reset circuit breaker if needed
curl -X POST "http://localhost:8000/api/v1/monitoring/database/reset-circuit-breaker"
```

### AI Service Issues
```bash
# Check AI service health
curl -X GET "http://localhost:8000/api/v1/ai/services/health"

# Reset specific service errors
curl -X POST "http://localhost:8000/api/v1/ai/services/claude/reset-errors"
```

### System Health Check
```bash
# Overall system status
curl -X GET "http://localhost:8000/api/v1/monitoring/health"

# Get all alerts
curl -X GET "http://localhost:8000/api/v1/monitoring/alerts"
```

## 📚 Additional Resources

- [Complete Documentation](./ENHANCED_ERROR_HANDLING.md)
- [Configuration Reference](../backend/core/config.py)
- [API Documentation](../backend/api/monitoring.py)
- [Database Implementation](../backend/core/database.py)
- [AI Error Handling](../backend/core/ai_error_handling.py)

## 🎯 Best Practices

1. **Monitor Regularly**: Check system health and metrics frequently
2. **Set Appropriate Limits**: Configure realistic rate limits and budgets
3. **Handle Errors Gracefully**: Always catch and handle service errors
4. **Use Retry Logic**: Leverage built-in retry mechanisms for transient failures
5. **Reset When Needed**: Manually reset error states after resolving issues
6. **Review Logs**: Regularly check error logs for patterns and issues

## 🔄 Automatic Recovery

The system includes automatic recovery mechanisms:

- **Database**: Automatic connection pool recovery and circuit breaker reset
- **AI Services**: Automatic retry with exponential backoff and circuit breaker recovery
- **Health Checks**: Proactive monitoring and automatic state transitions
- **Rate Limiting**: Automatic quota reset and budget management

For more detailed information, see the [complete documentation](./ENHANCED_ERROR_HANDLING.md).
