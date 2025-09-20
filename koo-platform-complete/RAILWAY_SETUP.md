# Railway Production Setup Guide

## 🚀 Quick Setup Steps

### 1. Add PostgreSQL Database
```bash
# In Railway Dashboard:
1. Click "+ New" → "Database" → "PostgreSQL"
2. Railway automatically sets DATABASE_URL
3. Your app will detect and use PostgreSQL instead of SQLite
```

### 2. Add Redis Cache
```bash
# In Railway Dashboard:
1. Click "+ New" → "Database" → "Redis"
2. Railway automatically sets REDIS_URL
3. Optional: Add Celery variables (see below)
```

### 3. Essential Environment Variables

**Go to Railway Dashboard → Your Project → Variables → Add these:**

```bash
# Security (REQUIRED)
SECRET_KEY=your-long-random-secret-key-here-minimum-32-characters

# Admin Configuration
ADMIN_NAME=Your Name
ADMIN_EMAIL=your-email@domain.com
ADMIN_API_KEY=your-secure-admin-api-key

# AI Service API Keys (add as needed)
GEMINI_API_KEY=your-gemini-api-key-from-google-ai-studio
CLAUDE_API_KEY=your-claude-api-key-from-anthropic
PERPLEXITY_API_KEY=your-perplexity-api-key
PUBMED_API_KEY=your-ncbi-api-key

# Environment
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false
```

### 4. Optional Celery Configuration (for background tasks)
```bash
# Only add if you need background task processing
CELERY_BROKER_URL=${REDIS_URL}/1
CELERY_RESULT_BACKEND=${REDIS_URL}/2
```

### 5. Performance Tuning (Optional)
```bash
# Database optimization
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=40

# Cache optimization
CACHE_DEFAULT_TTL=7200
REDIS_MAX_CONNECTIONS=50

# AI service limits
AI_DEFAULT_REQUESTS_PER_MINUTE=100
AI_DEFAULT_DAILY_BUDGET=50.0
```

## 🔑 Getting API Keys

### Google Gemini
1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Create API key
3. Add as `GEMINI_API_KEY`

### Anthropic Claude
1. Go to [Anthropic Console](https://console.anthropic.com/)
2. Create API key
3. Add as `CLAUDE_API_KEY`

### Perplexity
1. Go to [Perplexity API](https://www.perplexity.ai/settings/api)
2. Create API key
3. Add as `PERPLEXITY_API_KEY`

### PubMed (NCBI)
1. Go to [NCBI API Keys](https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/)
2. Create API key
3. Add as `PUBMED_API_KEY`

## 🚀 Deployment Order

1. **Basic Deployment** - App works with SQLite (current state)
2. **Add PostgreSQL** - Better database performance
3. **Add Redis** - Caching and session storage
4. **Add API Keys** - Enable AI features
5. **Add Celery** - Background task processing (optional)

## 🔍 Verification

After setup, check:
- App health endpoint: `https://your-app.railway.app/health`
- Database status in health response
- Redis status in health response
- API endpoints work: `https://your-app.railway.app/api/v1/`

## 🆘 Troubleshooting

**Database Connection Issues:**
- Verify `DATABASE_URL` is set and starts with `postgresql://`
- Check Railway PostgreSQL service is running

**Redis Connection Issues:**
- Verify `REDIS_URL` is set and starts with `redis://`
- Check Railway Redis service is running

**API Key Issues:**
- Verify API keys are valid and have correct permissions
- Check environment variable names match exactly

## 📊 Monitoring

Railway provides:
- **Metrics** - CPU, Memory, Network usage
- **Logs** - Application logs and errors
- **Deployments** - Build and deployment history
- **Database Metrics** - PostgreSQL and Redis performance