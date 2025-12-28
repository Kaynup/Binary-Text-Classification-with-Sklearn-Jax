# Sentiment Analyzer - Production Deployment Guide

This guide walks you through deploying the Sentiment Analyzer with a split architecture:
- **Frontend**: Vercel (static hosting)
- **Backend**: Railway/Render (Python API hosting)

## Directory Structure

```
deployment-vercel/
├── frontend/                    # Deploy to Vercel
│   ├── index.html
│   ├── styles.css
│   ├── app.js
│   ├── config.js               # ⚠️ Update with your backend URL
│   └── vercel.json
│
├── backend/                     # Deploy to Railway/Render
│   ├── main.py                  # Unified FastAPI application
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── railway.json
│   ├── render.yaml
│   └── models/
│       ├── sklearn/
│       │   └── logreg-80k.joblib
│       └── jax/
│           └── bag_of_words-with_pooling/
│               └── pooling_batch-300_pipeline.joblib
│
└── README.md
```

---

## Step 1: Deploy Backend to Railway (Recommended)

Railway provides easy Python deployment with good free tier limits.

### Option A: Deploy via Railway Dashboard

1. **Create Account**: Go to [railway.app](https://railway.app) and sign up

2. **Create New Project**:
   - Click "New Project"
   - Select "Deploy from GitHub repo" or "Empty Project"

3. **If using GitHub**:
   - Push the `deployment-vercel/backend` folder to a GitHub repo
   - Connect Railway to your GitHub
   - Select the repository

4. **If deploying manually**:
   - Install Railway CLI: `npm install -g @railway/cli`
   - Navigate to backend: `cd deployment-vercel/backend`
   - Login: `railway login`
   - Initialize: `railway init`
   - Deploy: `railway up`

5. **Get your URL**:
   - Railway will provide a URL like: `https://your-app-name.railway.app`
   - Test it: `curl https://your-app-name.railway.app/health`

### Option B: Deploy via Render

1. Go to [render.com](https://render.com) and sign up

2. Create a new "Web Service"

3. Connect your GitHub repo or use the Docker option

4. Render will auto-detect the Dockerfile

5. Get your URL: `https://your-app-name.onrender.com`

---

## Step 2: Update Frontend Configuration

After deploying the backend, update the frontend config:

### Edit `frontend/config.js`:

```javascript
window.APP_CONFIG = {
  // Replace with your actual backend URL
  API_URL: 'https://YOUR-BACKEND-URL.railway.app/predict',
  
  APP_NAME: 'Sentiment Analyzer',
  VERSION: '1.0.0'
};
```

**Example:**
```javascript
API_URL: 'https://sentiment-api-production.railway.app/predict',
```

---

## Step 3: Deploy Frontend to Vercel

### Option A: Deploy via Vercel Dashboard

1. **Create Account**: Go to [vercel.com](https://vercel.com) and sign up

2. **Import Project**:
   - Click "Add New Project"
   - Import from GitHub, or use Vercel CLI

3. **Configure**:
   - Root Directory: `deployment-vercel/frontend`
   - Framework Preset: "Other"
   - Build Command: (leave empty)
   - Output Directory: `.`

4. **Deploy**: Click "Deploy"

5. **Get your URL**: `https://your-app.vercel.app`

### Option B: Deploy via Vercel CLI

```bash
# Install Vercel CLI
npm install -g vercel

# Navigate to frontend
cd deployment-vercel/frontend

# Login
vercel login

# Deploy
vercel

# For production
vercel --prod
```

---

## Step 4: Test Your Deployment

1. **Test Backend Health**:
   ```bash
   curl https://YOUR-BACKEND-URL.railway.app/health
   ```

2. **Test Prediction**:
   ```bash
   curl -X POST https://YOUR-BACKEND-URL.railway.app/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "I love this product!"}'
   ```

3. **Open Frontend**: 
   Navigate to your Vercel URL and test the full flow

---

## Local Development

### Run Backend Locally:
```bash
cd deployment-vercel/backend
pip install -r requirements.txt
python main.py
# or
uvicorn main:app --reload
```

### Run Frontend Locally:
```bash
cd deployment-vercel/frontend

# Update config.js to use localhost:
# API_URL: 'http://127.0.0.1:8000/predict'

# Serve with any static server
python -m http.server 3000
# or
npx serve .
```

---

## Environment Variables

### Backend (Railway/Render):

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Server port |
| `ENABLE_JAX` | `true` | Enable JAX model (set to `false` for faster cold starts) |

### Frontend:
Configuration is handled via `config.js` file instead of environment variables.

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info and available models |
| `/health` | GET | Health check |
| `/predict` | POST | Predict sentiment (auto model selection) |
| `/predict?model=sklearn` | POST | Force sklearn model |
| `/predict?model=jax` | POST | Force JAX model |
| `/docs` | GET | Interactive API documentation |

---

## Troubleshooting

### CORS Errors
The backend is configured to allow all origins. If you still see CORS errors:
- Check that your backend URL is correct in `config.js`
- Ensure the backend is running and accessible

### Cold Start Times
Railway/Render free tiers may have cold starts. The sklearn model loads faster (~2-3s) than JAX (~10-15s).

To optimize:
- Set `ENABLE_JAX=false` if you only need sklearn
- Consider upgrading to a paid tier for faster instances

### Model Not Loading
Check the backend logs for errors:
```bash
railway logs  # For Railway
```

Ensure model files are present in the `models/` directory.

---

## Notes

- **Free Tier Limits**: Both Railway and Render have usage limits on free tiers
- **Cold Starts**: First request after inactivity may be slow
- **Model Size**: JAX model is ~135MB, sklearn is ~4MB
- **Recommended**: Start with sklearn only (`ENABLE_JAX=false`) for fastest deployment

---

## You're Done!

Your sentiment analyzer should now be live at your Vercel URL, connected to your Railway/Render backend!

Frontend: `https://your-app.vercel.app`
Backend: `https://your-api.railway.app`
