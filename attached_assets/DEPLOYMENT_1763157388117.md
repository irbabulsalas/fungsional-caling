# 🚀 Railway Deployment Guide - Complete Tutorial

**AI Data Analysis Platform by Muhammad Irbabul Salas**

---

## 📋 Pre-Deployment Checklist

Before deploying to Railway, ensure you have:
- ✅ GitHub account
- ✅ Railway account (sign up at https://railway.app)
- ✅ Gemini API key (from https://aistudio.google.com/app/apikey)
- ✅ Code pushed to GitHub repository

---

## 🔑 Step 1: Get Gemini API Key (5 minutes)

1. Visit: **https://aistudio.google.com/app/apikey**
2. Login with your Google account
3. Click **"Create API Key"**
4. Copy the generated API key
5. Save it securely (you'll need it in Step 4)

**Free Tier Limits:**
- 15 requests per minute
- 1,500 requests per day
- 100% FREE forever for Gemini Flash

---

## 📦 Step 2: Push Code to GitHub

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit changes
git commit -m "Initial commit - AI Data Analysis Platform"

# Create repository on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push to GitHub
git branch -M main
git push -u origin main
```

---

## 🚂 Step 3: Create Railway Project

1. Go to **https://railway.app/**
2. Click **"Start a New Project"**
3. Click **"Deploy from GitHub repo"**
4. **Authorize Railway** to access your GitHub
5. **Select your repository** from the list
6. Railway will automatically detect the project

---

## ⚙️ Step 4: Configure Environment Variables

This is **CRITICAL** for the app to work!

1. In Railway dashboard, click your project
2. Go to **"Variables"** tab
3. Click **"+ New Variable"**
4. Add the following:

```
Variable Name: GEMINI_API_KEY
Value: [paste your Gemini API key from Step 1]
```

**DO NOT** add quotes around the API key!

---

## 🔧 Step 5: Configure Deployment Settings

1. Go to **"Settings"** tab
2. Scroll to **"Deploy"** section
3. Ensure **"Start Command"** is:
   ```
   streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
   ```

4. Set **"Restart Policy"**: **ON_FAILURE**
5. Set **"Max Retries"**: **10**

---

## 🌐 Step 6: Generate Public Domain

1. In Railway dashboard, go to **"Settings"** tab
2. Scroll to **"Networking"** section  
3. Click **"Generate Domain"**
4. Railway will provide a URL like: `your-app-name.up.railway.app`
5. Click the URL to open your deployed app!

---

## ✅ Step 7: Verify Deployment

1. Open the generated Railway URL
2. You should see the app loading
3. Check:
   - ✅ Header shows "Muhammad Irbabul Salas"
   - ✅ Profile photo appears
   - ✅ Sidebar navigation works
   - ✅ Can upload data or load sample data
   - ✅ AI chat works (respects rate limiting)

If everything works: **Congratulations!** 🎉

---

## 🔄 Updating Your Deployed App

Railway auto-deploys when you push to GitHub:

```bash
# Make changes to your code
# ...

# Commit changes
git add .
git commit -m "Update: [describe changes]"

# Push to GitHub
git push origin main

# Railway automatically detects and redeploys!
```

---

## 💰 Cost Management

### Free Tier (Trial)
- Railway gives **$5 free credit**
- Enough for ~1 month of light usage
- Auto-sleeps when idle (free tier)

### After Trial
- **Hobby Plan**: $5/month
- Includes $5 usage credit
- No auto-sleep
- Better performance

### Estimated Monthly Cost
```
Gemini API (Flash):     $0    (FREE)
Railway Hobby:          $5    (after trial)
GitHub:                 $0    (FREE)
─────────────────────────────
Total:                  ~$5/month
```

---

## 🎨 Custom Domain (Optional)

Want to use your own domain? (e.g., `dataanalysis.yourdomain.com`)

1. Buy a domain from Namecheap, GoDaddy, etc.
2. In Railway, go to **Settings → Networking**
3. Click **"Custom Domain"**
4. Enter your domain
5. Update DNS records at your domain provider:
   ```
   Type: CNAME
   Name: dataanalysis (or @)
   Value: your-app.up.railway.app
   TTL: 3600
   ```
6. Wait 5-30 minutes for DNS propagation
7. Your app is now live at your custom domain!

**SSL/HTTPS**: Railway provides this automatically for FREE ✅

---

## 📊 Monitoring & Logs

### View Logs
1. Railway Dashboard → Your Project
2. Click **"Deployments"** tab
3. Select latest deployment
4. Click **"View Logs"**
5. Real-time logs appear

### Metrics
1. Go to **"Metrics"** tab
2. View:
   - CPU usage
   - Memory usage
   - Network traffic
   - Request count

---

## 🐛 Troubleshooting

### Issue: Build Failed

**Error**: `Cannot find requirements.txt`

**Solution**: Ensure `Procfile` and `railway.json` are in root directory

---

### Issue: App Won't Start

**Error**: `Application error` or 5xx error

**Solution**:
1. Check logs for specific error
2. Verify `GEMINI_API_KEY` is set correctly
3. Ensure PORT variable is used correctly
4. Restart deployment

---

### Issue: GEMINI_API_KEY Not Working

**Error**: `API key invalid` or `Unauthorized`

**Solution**:
1. Get new API key from Google AI Studio
2. Update Railway environment variable
3. Redeploy

---

### Issue: Rate Limit Errors

**Error**: `429 Too Many Requests`

**Solution**: This is expected! App implements:
- 1 minute cooldown between requests
- 15 requests per hour limit
- Visual countdown timer

Users will see countdown and can continue using other features.

---

## 🔒 Security Best Practices

1. ✅ **Never** commit API keys to GitHub
2. ✅ Always use Railway environment variables
3. ✅ Use `.env.example` for documentation only
4. ✅ Regularly rotate API keys
5. ✅ Monitor usage for unexpected spikes

---

## 📈 Scaling Tips

### For Higher Traffic:

1. **Upgrade Railway Plan**:
   - Pro Plan: $20/month (includes $20 credit)
   - Better resources
   - Priority support

2. **Upgrade Gemini API**:
   - Pay-as-you-go for higher limits
   - Gemini Pro for better performance
   - ~$0.075 per 1K tokens

3. **Optimize Code**:
   - Cache frequent queries
   - Lazy load heavy features
   - Optimize data processing

---

## ✅ Post-Deployment Checklist

```
□ App accessible via Railway URL
□ HTTPS working (padlock icon in browser)
□ Profile photo displays correctly
□ Upload functionality works
□ Sample datasets load successfully
□ AI chat responds (within rate limits)
□ All dashboards render properly
□ Export features work
□ No errors in Railway logs
□ Mobile responsive (test on phone)
□ Dark/Light mode toggle works
```

---

## 🎉 Success!

Your AI Data Analysis Platform is now live and accessible worldwide!

**Share your deployed URL:**
`https://your-app-name.up.railway.app`

---

**Questions or Issues?**
Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) or Railway documentation.

---

**Deployed by Muhammad Irbabul Salas**
*Powered by Railway & Gemini 2.5*
