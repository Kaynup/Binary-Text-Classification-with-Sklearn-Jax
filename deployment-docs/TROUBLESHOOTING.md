# Troubleshooting Guide

## Common Deployment Issues

### 1. "Function Crashed" on Vercel

**Symptoms:** API returns 500 error, logs show function crashed

**Causes & Fixes:**

| Cause | Fix |
|-------|-----|
| Model file not included | Ensure `model/logreg-80k.joblib` is in the repo |
| Wrong Python version | Add `"functions": {"api/*.py": {"runtime": "python3.9"}}` to `vercel.json` |
| Missing dependencies | Check `requirements.txt` is in deployment root |

---

### 2. CORS Errors in Browser

**Symptoms:** "Access-Control-Allow-Origin" errors in console

**Fix:** CORS middleware is configured in `api/predict.py`. Verify it's set to `allow_origins=["*"]`

---

### 3. 404 on API Endpoint

**Symptoms:** `/api/predict` returns 404

**Fixes:**
1. Verify `vercel.json` routes are correct
2. Check Vercel dashboard → Set **Root Directory** to `deployment`
3. Ensure file is named `predict.py` (not something else)

---

### 4. Slow First Request (Cold Start)

**Symptoms:** First API call takes 3-5+ seconds

**Explanation:** This is normal. Serverless functions have cold starts where the model is loaded.

**Mitigations:**
- Keep model file size small
- Use Vercel Pro for longer warm periods
- Implement a keep-alive ping

---

### 5. Model Load Failures

**Symptoms:** "Model could not be loaded" error

**Debugging:**
```python
# Check model path in api/predict.py
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(SCRIPT_DIR, "..", "model", "logreg-80k.joblib")
print(f"Looking for model at: {model_path}")
print(f"File exists: {os.path.exists(model_path)}")
```

---

### 6. Static Files Not Loading

**Symptoms:** HTML loads but CSS/JS show 404

**Fixes:**
1. Check `vercel.json` has route: `{"src": "/(.*)", "dest": "/app/$1"}`
2. Verify files are in `deployment/app/` folder
3. Check file names match (case-sensitive)

---

## Local Testing

```bash
# Start FastAPI server
cd deployment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate jaxenv
uvicorn api.predict:app --host 0.0.0.0 --port 8000 --reload

# Test API
curl -X POST "http://localhost:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Test message"}'
```

---

## Getting Help

1. Check Vercel deployment logs in dashboard
2. Review `CHANGELOG.md` for recent changes
3. Compare working vs broken deployments
