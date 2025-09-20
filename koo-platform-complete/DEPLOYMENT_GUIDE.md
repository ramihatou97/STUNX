# 🚀 KOO Platform Deployment Guide

## 🎯 Quick Deploy Options

### Option 1: Railway (Recommended - 5 minutes)

#### Step 1: Sign Up for Railway
1. Go to [railway.app](https://railway.app)
2. Sign up with your GitHub account
3. Verify your account

#### Step 2: Deploy Your Repository
1. Click "**New Project**" in Railway dashboard
2. Select "**Deploy from GitHub repo**"
3. Choose your **STUNX** repository
4. Railway will automatically detect your Python app

#### Step 3: Add Environment Variables
In Railway dashboard:
1. Go to your project → **Variables** tab
2. Add these variables:

```env
# Required AI API Keys
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-claude-key
GOOGLE_AI_API_KEY=your-google-ai-key
PERPLEXITY_API_KEY=pplx-your-perplexity-key

# App Settings
ENVIRONMENT=production
DEBUG=false
ADMIN_NAME=Your Name
ADMIN_EMAIL=your.email@example.com
SECRET_KEY=your-generated-secret-key
```

#### Step 4: Add Database
1. In Railway dashboard, click "**+ New**"
2. Select "**Database**" → "**PostgreSQL**"
3. Railway automatically connects it to your app

#### Step 5: Deploy!
1. Railway automatically deploys when you push to GitHub
2. Your app will be available at: `https://your-app-name.railway.app`
3. Visit `/health` to check if it's working

---

### Option 2: Vercel (Frontend) + Railway (Backend)

#### Step 1: Deploy Backend to Railway
Follow Steps 1-5 above for Railway

#### Step 2: Deploy Frontend to Vercel
1. Go to [vercel.com](https://vercel.com)
2. Sign up with GitHub
3. Click "**Import Git Repository**"
4. Select your **STUNX** repository
5. Set **Root Directory** to `frontend`
6. Add environment variable:
   ```
   REACT_APP_API_URL=https://your-backend-url.railway.app
   ```

---

### Option 3: Render (Alternative)

#### Step 1: Create Account
1. Go to [render.com](https://render.com)
2. Sign up with GitHub

#### Step 2: Create Web Service
1. Click "**New +**" → "**Web Service**"
2. Connect your **STUNX** repository
3. Configure:
   - **Name**: koo-platform
   - **Runtime**: Python 3
   - **Build Command**: `cd backend && pip install -r requirements.txt`
   - **Start Command**: `cd backend && python -m uvicorn main:app --host 0.0.0.0 --port $PORT`

#### Step 3: Add Environment Variables
Add the same variables as Railway option

#### Step 4: Add Database
1. Create "**PostgreSQL**" database in Render
2. Copy connection string to `DATABASE_URL`

---

## 🔑 Getting API Keys

### OpenAI (Required)
1. Go to [platform.openai.com](https://platform.openai.com)
2. Create account → **API Keys** → "**Create new secret key**"
3. Copy key (starts with `sk-`)

### Anthropic/Claude (Required)
1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Create account → **API Keys** → "**Create Key**"
3. Copy key (starts with `sk-ant-`)

### Google AI (Required)
1. Go to [makersuite.google.com](https://makersuite.google.com)
2. Create account → **Get API Key**
3. Copy key

### Perplexity (Required)
1. Go to [perplexity.ai](https://perplexity.ai)
2. Create account → **API** → "**Generate**"
3. Copy key (starts with `pplx-`)

---

## ⚡ Quick Start (Railway)

**1-Click Deploy:**

1. **Click**: [Deploy to Railway](https://railway.app/new/template)
2. **Connect**: Your STUNX GitHub repository
3. **Add**: API keys in Variables tab
4. **Wait**: 2-3 minutes for deployment
5. **Visit**: Your app URL → `/docs` to see API docs

**Your KOO Platform will be live at: `https://your-app-name.railway.app`**

---

## 🔧 Post-Deployment Setup

### 1. Test Your Deployment
- Visit: `https://your-app.railway.app/health`
- Should show: `{"status": "healthy", "version": "2.0.0"}`

### 2. Test AI Services
- Visit: `https://your-app.railway.app/docs`
- Try the `/ai/capabilities` endpoint
- Should show all AI services available

### 3. Set Up Custom Domain (Optional)
1. In Railway: Settings → **Domains**
2. Add your custom domain
3. Update DNS records as shown

---

## 🛠️ Troubleshooting

### Common Issues:

#### ❌ Build Fails
**Solution**: Check that all files are committed to GitHub

#### ❌ API Keys Not Working
**Solution**:
1. Verify keys are correct
2. Check API key has credits/usage available
3. Restart deployment after adding keys

#### ❌ Database Connection Issues
**Solution**:
1. Ensure PostgreSQL service is running
2. Check `DATABASE_URL` is set correctly
3. Railway auto-connects database

#### ❌ Import Errors
**Solution**:
1. Check all dependencies in `requirements.txt`
2. Verify Python version compatibility

---

## 🎯 Production Checklist

- [ ] All API keys added and working
- [ ] Database connected and healthy
- [ ] `/health` endpoint returns success
- [ ] `/docs` shows all 11 AI endpoints
- [ ] Custom domain configured (optional)
- [ ] Environment variables set correctly
- [ ] Monitoring/alerts set up

---

## 🚀 You're Ready!

Your AI-powered neurosurgical knowledge platform is now live and ready to help medical professionals worldwide!

**🎊 Congratulations on deploying cutting-edge AI technology!**