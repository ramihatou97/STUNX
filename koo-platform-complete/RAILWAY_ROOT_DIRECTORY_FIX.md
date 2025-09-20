# Railway Root Directory Fix

## 🚨 Problem
Railway is looking for files in `koo-platform-complete/koo-platform-complete/` instead of the repository root, causing configuration files to not be found.

## ✅ Solution

### Step 1: Fix Railway Root Directory Setting

**In Railway Dashboard:**
1. Go to your project
2. Click **Settings** (gear icon)
3. Scroll to **Source** section
4. Find **Root Directory** field
5. **Current setting** (problematic): `koo-platform-complete`
6. **Change to**: `.` (dot for repository root) OR leave **empty**
7. Click **Save**

### Step 2: Verify File Structure

**Expected structure Railway should see:**
```
repository-root/
├── railway.json          ← Configuration file
├── backend/
│   ├── main.py           ← Application entry point
│   └── requirements.txt  ← Python dependencies
├── requirements.txt      ← Root requirements (fallback)
└── other files...
```

### Step 3: Force New Deployment

**After changing root directory:**
1. Go to **Deployments** tab
2. Click **Deploy Latest**
3. Or push a small change to trigger auto-deploy

### Step 4: Verify in Logs

**Check deployment logs for:**
```bash
✅ Found railway.json
✅ Found backend/requirements.txt
✅ Building from repository root
✅ Starting: cd backend && python main.py
```

## 🔧 Alternative Solutions

### Option A: Move Files (NOT recommended)
- Don't move files - fix Railway settings instead

### Option B: Update repository structure
- This is already correct - Railway setting is the issue

### Option C: Use nixpacks.toml instead
```toml
# nixpacks.toml in repository root
[phases.setup]
nixPkgs = ["python3", "gcc"]

[phases.install]
cmds = ["cd backend && pip install -r requirements.txt"]

[start]
cmd = "cd backend && python main.py"
```

## 🔍 Debugging

### Check Current Root Directory
1. Railway Dashboard → Project → Settings
2. Look at **Source** section
3. **Root Directory** field shows current setting

### Verify File Paths
Railway should find these files:
- `./railway.json` ✅
- `./backend/main.py` ✅
- `./backend/requirements.txt` ✅

### Common Issues
- **Root set to**: `koo-platform-complete` ❌
- **Root should be**: `.` or empty ✅
- **Double nesting**: `koo-platform-complete/koo-platform-complete/` ❌

## 📋 Verification Checklist

After fixing:
- [ ] Root Directory setting is `.` or empty
- [ ] New deployment triggered
- [ ] Logs show "Building from repository root"
- [ ] railway.json is found
- [ ] backend/requirements.txt is found
- [ ] Application starts successfully
- [ ] Health check passes at `/health`

## 🆘 If Still Failing

1. **Disconnect and reconnect GitHub repo**
2. **Delete and recreate Railway project**
3. **Use Railway CLI to deploy directly**

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway link
railway up
```