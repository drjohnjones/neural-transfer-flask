# 🚂 Railway Deployment Guide

## Neural Style Transfer Flask App - Deployment Instructions

---

## 📋 Prerequisites

- GitHub account with repo: `neural-transfer-flask`
- Railway account (free or Pro): https://railway.app
- GitHub Personal Access Token (for git push)

---

## 🚀 Quick Deploy (5 Minutes)

### **Step 1: Sign into Railway**

1. Go to https://railway.app
2. Click **"Login"**
3. Choose **"Login with GitHub"**
4. Authorize Railway

---

### **Step 2: Create New Project**

1. Click **"New Project"** (top right)
2. Select **"Deploy from GitHub repo"**
3. Click **"Configure GitHub App"** (if first time)
4. Select repository: **"neural-transfer-flask"**
5. Click **"Deploy Now"**

---

### **Step 3: Wait for Build**

Railway will automatically:
- ✅ Detect Dockerfile
- ✅ Build container (~5-7 minutes)
- ✅ Install dependencies
- ✅ Download VGG19 model
- ✅ Start app

**Watch the logs for:**
```
✅ VGG19 loaded!
🎨 Neural Style Transfer - Production
📱 Running on port: XXXX
```

---

### **Step 4: Generate Public Domain**

1. Click on your deployment
2. Go to **"Settings"** tab
3. Scroll to **"Networking"** section
4. Click **"Generate Domain"**
5. Copy your URL! 🎉

**Your app will be live at:**
```
https://neural-transfer-flask-production.up.railway.app
```

---

## 🔧 Configuration (Optional)

### **Environment Variables:**

Go to **"Variables"** tab and add:

| Variable | Value | Required |
|----------|-------|----------|
| `SECRET_KEY` | `your-random-secret-key` | Optional |
| `PORT` | Auto-set by Railway | Auto |

---

## 💰 Cost Management

### **Expected Costs:**

- **Docker Image:** ~400 MB
- **RAM Usage:** ~4-5 GB (VGG19 model)
- **Processing Time:** 4-5 minutes per image

### **Monthly Estimates:**

| Usage | Cost |
|-------|------|
| Always ON (24/7) | $18-25/month |
| Business Hours (9-5) | $4-5/month |
| Demo Only (10 hrs) | $0.25/month |

### **To Reduce Costs:**

**Option 1: Remove Service When Not Using**
1. Settings → General
2. Scroll to bottom (Danger Zone)
3. Click **"Remove Service"**
4. Redeploy when needed (use this guide)

**Option 2: Downgrade Plan**
- Railway Free Tier: $5 credit/month
- Covers ~500 hours of uptime

---

## 🔄 Redeployment Process

### **If You Removed the Service:**

#### **Method A: From Railway Dashboard**

1. Go to Railway dashboard
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Choose **"neural-transfer-flask"**
5. Wait 5-7 minutes for build

#### **Method B: Using Railway CLI**
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Link to project
railway link

# Deploy
railway up
```

---

### **If Code Changed:**

Railway auto-deploys on every `git push`:
```bash
cd ~/Desktop/projects/52-ai-ml-projects/21-neural-style-transfer

# Make your changes
git add .
git commit -m "Your changes"
git push origin main

# Railway automatically rebuilds!
```

---

## 🧪 Testing Deployment

### **Health Check:**
```bash
curl https://your-app.up.railway.app/health
```

**Expected response:**
```json
{
  "status": "ok",
  "version": "flask-production",
  "device": "cpu"
}
```

### **Manual Test:**

1. Open app URL in browser
2. Upload an image
3. Select a style (e.g., Kandinsky)
4. Click **"Apply Style Transfer"**
5. Wait 4-5 minutes
6. Download result ✅

---

## 🐛 Troubleshooting

### **Build Fails:**

**Error: "Image size exceeded limit"**
- Solution: Ensure `.dockerignore` excludes `venv/`, `data/uploads/`, `data/output/`

**Error: "No module named 'torch'"**
- Solution: Check `requirements.txt` has CPU-only PyTorch:
```
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.2.0+cpu
torchvision==0.17.0+cpu
```

### **App Crashes:**

Check Railway logs:
1. Click deployment
2. View **"Logs"** tab
3. Look for Python errors

**Common issues:**
- Missing style images in `data/style/`
- Port not set (Railway auto-sets PORT)

### **Slow Processing:**

Expected behavior! CPU processing takes 4-5 minutes per image.
- 100 steps: ~2 minutes
- 200 steps: ~3 minutes  
- 300 steps: ~5 minutes

---

## 📊 Monitoring

### **Railway Dashboard:**

**Metrics Tab:**
- Memory usage: ~4-5 GB
- CPU usage: Spikes during processing
- Network: Upload/download bandwidth

**Logs Tab:**
- Real-time application logs
- See each transfer request
- Debug errors

**Deployments Tab:**
- Build history
- Rollback to previous versions

---

## 🔒 Security Notes

### **Environment Variables:**

Never commit secrets to git:
```bash
# Use Railway Variables for:
SECRET_KEY=<random-string>
API_KEY=<if-needed>
```

### **GitHub Token:**

Don't include token in git remote URL:
```bash
# ❌ Bad:
git remote add origin https://TOKEN@github.com/user/repo.git

# ✅ Good:
git remote add origin https://github.com/user/repo.git
# Use credential manager for auth
```

---

## 📱 Custom Domain (Optional)

### **Add Your Domain:**

1. Settings → Networking
2. Click **"Custom Domain"**
3. Enter: `neural-style.yourdomain.com`
4. Update DNS records as instructed:
   - Type: CNAME
   - Name: neural-style
   - Value: (Railway provides)

---

## 🎯 Quick Reference

### **Essential URLs:**

- **Railway Dashboard:** https://railway.app/dashboard
- **GitHub Repo:** https://github.com/drjohnjones/neural-transfer-flask
- **Live App:** https://neural-transfer-flask-production.up.railway.app

### **Key Files:**
```
neural-transfer-flask/
├── app.py              # Flask application
├── Dockerfile          # Container config
├── requirements.txt    # Python dependencies
├── src/
│   └── style_transfer_engine.py
├── templates/
│   └── index.html
└── data/style/         # Art styles
```

### **Railway CLI Commands:**
```bash
railway login              # Authenticate
railway link              # Connect to project
railway up                # Deploy
railway logs              # View logs
railway status            # Check status
railway open              # Open app in browser
```

---

## ✅ Deployment Checklist

Before deploying:

- [ ] Code pushed to GitHub
- [ ] `.gitignore` excludes `venv/`, `data/uploads/`, `data/output/`
- [ ] `.dockerignore` configured
- [ ] `requirements.txt` has CPU-only PyTorch
- [ ] `Dockerfile` is optimized
- [ ] Style images in `data/style/` are < 500KB each
- [ ] Railway account ready
- [ ] GitHub connected to Railway

After deployment:

- [ ] Build completes successfully
- [ ] Health check returns OK
- [ ] Can upload test image
- [ ] Style transfer processes correctly
- [ ] Download works
- [ ] Public domain generated

---

## 📞 Support

**Railway Issues:**
- Documentation: https://docs.railway.app
- Community: https://railway.app/discord

**App Issues:**
- Check logs in Railway dashboard
- Review error messages
- Test locally first: `python app.py`

---

## 🎉 Success!

Your Neural Style Transfer app should now be live at:
```
https://your-app.up.railway.app
```

**Share it with the world!** 🎨✨

---

**Created by:** Dr. John Jones  
**Project:** #21 of 52 AI/ML Projects Challenge  
**Last Updated:** November 2025
