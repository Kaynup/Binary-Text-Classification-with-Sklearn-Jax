# Production Deployment Guide (v2.0.0)

This guide walks you through deploying the **Sentiment Analyzer (v2.0.0)** with a decoupled architecture:
- **Frontend**: [Vercel](https://vercel.com) (Static Hosting with CSP and XSS Hardening)
- **Backend**: [Railway](https://railway.app) (FastAPI + Pure Scikit-Learn CPU deployment)

---

## Directory Structure

```
├── backend/                       # Deploy to Railway
│   ├── app/                       # Decoupled FastAPI architecture
│   │   ├── main.py                # Application factory & lifespan
│   │   ├── config.py              # Environment variables & rate limits
│   │   ├── logging_config.py      # RotatingFileHandler setup
│   │   ├── api/                   # /predict, /models, /health, /metrics
│   │   ├── middleware/            # Rate limiting & security
│   │   ├── schemas/               # Pydantic request/response models
│   │   └── services/              # Scikit-learn inference service
│   ├── models/sklearn/            # 3.8MB lightweight model
│   ├── Dockerfile                 # Pure CPU Python 3.10-slim image
│   ├── railway.json               # Railway build & healthcheck config
│   ├── Procfile                   # Nixpacks / Procfile entrypoint
│   └── requirements.txt
│
└── deployment-vercel/frontend/    # Deploy to Vercel (or root /frontend)
    ├── index.html                 # Subtle Nordic Slate theme
    ├── styles.css                 # Clean CSS tokens (no neon/gradients)
    ├── app.js                     # XSS-safe DOM rendering & robot states
    ├── config.js                  # Backend API URL configuration
    └── vercel.json                # Security headers & rewrites
```

---

## Step 1: Deploy Backend to Railway (Free Tier)

Railway provides frictionless Python/Docker hosting with generous free resources.

### Option A: Deploy via GitHub & Railway Dashboard
1. Go to [railway.app](https://railway.app) and sign in.
2. Click **"New Project"** -> **"Deploy from GitHub repo"**.
3. Select your repository.
4. In the service settings:
   - **Root Directory**: `backend` (or leave root if building whole repo)
   - Railway will automatically detect the `Dockerfile` and `railway.json`.
5. Under **"Settings"** -> **"Networking"**, click **"Generate Domain"** to get a public URL (e.g. `https://your-api.up.railway.app`).
6. Test your live healthcheck:
   ```bash
   curl https://your-api.up.railway.app/health
   ```

### Option B: Deploy via Railway CLI
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy from backend directory
cd backend
railway login
railway init
railway up
```

---

## Step 2: Configure Frontend with Backend URL

1. Open `deployment-vercel/frontend/config.js` (or `frontend/config.js`).
2. Replace `API_URL` with your Railway URL:
   ```javascript
   window.APP_CONFIG = {
       API_URL: 'https://your-api.up.railway.app/predict',
       APP_NAME: 'Sentiment Analyzer',
       VERSION: '2.0.0',
       FRAMEWORK: 'Scikit-Learn Pure'
   };
   ```

---

## Step 3: Deploy Frontend to Vercel

### Option A: Deploy via Vercel Dashboard
1. Go to [vercel.com](https://vercel.com) and sign in.
2. Click **"Add New..."** -> **"Project"**.
3. Import your GitHub repository.
4. Under **"Project Settings"**:
   - **Root Directory**: `deployment-vercel/frontend` (or `frontend`)
   - **Framework Preset**: `Other`
   - **Build & Output Settings**: Leave defaults.
5. Click **"Deploy"**.

### Option B: Deploy via Vercel CLI
```bash
# Install Vercel CLI
npm install -g vercel

# Deploy from frontend directory
cd deployment-vercel/frontend
vercel --prod
```

---

## Step 4: Verification & Smoke Test

1. Verify backend health and model loading:
   ```bash
   curl -s https://your-api.up.railway.app/health | jq
   ```
2. Verify inference:
   ```bash
   curl -X POST https://your-api.up.railway.app/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "I absolutely loved this experience!"}'
   ```
3. Open your live Vercel URL in your browser, test inputs, and verify robot emotional reactions!
