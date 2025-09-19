# 🏗️ KOO Platform - Single User Edition

**AI-Driven Medical Knowledge Management Platform**
*Simplified for personal use with streamlined authentication*

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Node.js 18+ (for development)
- PostgreSQL 14+ (if running locally)

### Environment Setup
1. Copy environment configuration:
```bash
cp .env.example .env
```

2. Update the `.env` file with your personal settings:
```bash
# Your personal configuration
ADMIN_NAME="Your Name"
ADMIN_EMAIL="your-email@example.com"
ADMIN_API_KEY="your-personal-api-key-change-this"

# Add your AI API keys
GEMINI_API_KEY="your_gemini_api_key"
CLAUDE_API_KEY="your_claude_api_key"
PUBMED_API_KEY="your_pubmed_api_key"
```

### Development Mode
```bash
# Start all services
docker-compose up -d

# Or run components separately:
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend
cd frontend
npm install
npm start
```

### Production Deployment
```bash
# Using Docker Compose
docker-compose -f docker-compose.production.yml up -d

# Or using Kubernetes
kubectl apply -f k8s/
```

## 🔧 Architecture

### Simplified Authentication
- **No user registration/login** - you are always authenticated
- **Optional API key** for external access
- **Environment-based configuration** for personal settings
- **Minimal security overhead** while maintaining essential protections

### Core Components
```
📁 koo-platform-complete/
├── 🔧 backend/              # FastAPI application
│   ├── api/                 # Simplified API routes
│   ├── core/                # Configuration & dependencies
│   ├── models/              # Database models
│   ├── services/            # Business logic
│   └── tests/               # Test suite
├── 🎨 frontend/             # React application
│   ├── src/                 # Source code
│   ├── components/          # UI components
│   └── contexts/            # React contexts
├── 🗄️ database/            # Database schema
├── 🚀 infrastructure/       # Deployment configs
└── 📊 monitoring/           # Prometheus config
```

## 🛡️ Security Features

### Essential Security (Kept)
- ✅ HTTPS enforcement in production
- ✅ Input validation and sanitization
- ✅ Rate limiting for external requests
- ✅ Security headers (XSS, CSRF protection)
- ✅ File upload restrictions
- ✅ API key protection for external access

### Removed Complexity
- ❌ User registration/login flows
- ❌ Multi-user authentication
- ❌ Password management
- ❌ Session token rotation
- ❌ Role-based access control
- ❌ User audit trails

## 📚 API Usage

### Authentication
Since you're always authenticated, most endpoints work without headers:

```javascript
// Basic usage (local access)
fetch('/api/v1/chapters/')

// With API key (external access)
fetch('/api/v1/chapters/', {
  headers: {
    'X-API-Key': 'your-api-key'
  }
})
```

### Key Endpoints
```bash
# Health check
GET /health

# Chapters
GET /api/v1/chapters/
POST /api/v1/chapters/
GET /api/v1/chapters/{id}
PUT /api/v1/chapters/{id}
DELETE /api/v1/chapters/{id}

# Research
POST /api/v1/research/search
GET /api/v1/research/trends
POST /api/v1/research/synthesize

# Admin info
GET /api/v1/admin/info
```

## 🔄 Development Workflow

### Making Changes
1. **Backend**: Edit files in `backend/`, server auto-reloads
2. **Frontend**: Edit files in `frontend/src/`, browser auto-updates
3. **Database**: Add migrations in `backend/migrations/`

### Testing
```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test
```

### Building for Production
```bash
# Build frontend
cd frontend
npm run build

# Build backend Docker image
cd backend
docker build -t koo-backend .
```

## 📊 Monitoring

Access monitoring dashboards:
- **Application**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs (development only)
- **Prometheus**: http://localhost:9090 (if enabled)
- **Grafana**: http://localhost:3001 (if enabled)

## 🚀 Deployment Options

### Option 1: Docker Compose (Recommended)
```bash
docker-compose -f docker-compose.production.yml up -d
```

### Option 2: Kubernetes
```bash
kubectl apply -f k8s/deployment.yaml
```

### Option 3: Manual Deployment
```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000

# Frontend
cd frontend
npm run build
# Serve build/ with nginx or similar
```

## 🔧 Configuration

### Environment Variables
Key settings in `.env`:

```bash
# Personal Settings
ADMIN_NAME="Your Name"
ADMIN_EMAIL="your@email.com"
ADMIN_API_KEY="secure-key"

# Database
DATABASE_URL="postgresql://..."

# AI Services
GEMINI_API_KEY="your-key"
CLAUDE_API_KEY="your-key"

# Security
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=60
```

### Database Schema
Initialize with:
```bash
# Apply schema
psql -d koo_platform -f database/schema.sql

# Or use Alembic migrations
cd backend
alembic upgrade head
```

## 📈 Performance Tips

1. **Enable Redis caching** for better performance
2. **Use CDN** for frontend assets in production
3. **Configure database connection pooling**
4. **Enable gzip compression** in nginx
5. **Monitor API usage** to optimize costs

## 🆘 Troubleshooting

### Common Issues

**Database connection failed:**
```bash
# Check PostgreSQL is running
docker-compose ps
# Check database URL in .env
```

**Frontend build fails:**
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

**API key not working:**
```bash
# Verify in .env file
echo $ADMIN_API_KEY
```

### Logs
```bash
# Docker logs
docker-compose logs backend
docker-compose logs frontend

# Application logs
tail -f backend/logs/app.log
```

## 🎯 Next Steps

1. **Set up AI API keys** for research features
2. **Configure backup strategy** for your data
3. **Set up monitoring alerts** (optional)
4. **Customize the UI** to your preferences
5. **Add custom research sources** as needed

---

**Built with ❤️ for personal medical knowledge management**