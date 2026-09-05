# Binary Text Sentiment Classification with Scikit-Learn (v2.0.0)

[![CI Pipeline](https://github.com/Kaynup/Binary-Text-Classification-with-Sklearn-Jax/actions/workflows/ci.yml/badge.svg)](https://github.com/Kaynup/Binary-Text-Classification-with-Sklearn-Jax/actions)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-Scikit--Learn%201.3+-orange)](https://scikit-learn.org/)
[![API](https://img.shields.io/badge/API-FastAPI%20%26%20Uvicorn-009688)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Release](https://img.shields.io/badge/release-v2.0.0-brightgreen)](CHANGELOG.md)

A production-grade, interpretable **Binary Sentiment Classification** engine engineered with pure **Scikit-Learn**, **FastAPI**, and modern frontend observability.

Version **2.0.0** establishes a decoupled architecture, completely eliminates legacy JAX and Git LFS overhead, hardens security with in-memory IP rate limiting and 100% XSS-safe DOM rendering, introduces rotating file logging, embeds rigorous research metrics, and provides automated orchestration via `Makefile`, Prometheus, and Grafana.

---

## Key Highlights

- **Pure Scikit-Learn Architecture**: Logistic Regression (`SAGA`, $L_2$, $C=1.0$) with TF-IDF N-grams (1–5) across 80,000 features.
- **High Throughput & Low Latency**: $P_{50}$ inference latency of **0.035 ms** on standard CPU; throughput exceeding **26,000 inferences/second**.
- **Research-Grade Metrics**: Evaluated against 1.57M Sentiment140 benchmark samples (**84.26% F1-Score**, **0.9139 ROC-AUC**, **84.07% Accuracy**).
- **Subtle, Tactile UI**: Clean **Studio Slate & Nordic Minimalist** theme (neon gradients removed) with interactive emotional robot states (Neutral, Happy, Sad).
- **Security Hardened**: IP sliding-window rate limiting (HTTP 429), strict Content Security Policy (CSP), defensive headers, and zero raw `innerHTML` string interpolation.
- **Production Deployments**: Decoupled for free [Railway](https://railway.app) (FastAPI backend) and [Vercel](https://vercel.com) (static frontend).
- **Observability**: Prometheus metrics exporter (`/metrics`) and pre-configured Grafana dashboard.

---

## Research Benchmarks

Evaluated on the Adeoluwa Adeboye Sentiment140 benchmark dataset:

| Research Metric | Score | Evaluation Description |
|---|---|---|
| **Accuracy** | **84.07%** | Overall test set correctness |
| **F1-Score** | **84.26%** | Harmonic mean of precision and recall |
| **Precision** | **83.00%** | Positive predictive rate ($TP / [TP + FP]$) |
| **Recall / Sensitivity** | **85.55%** | True positive discovery rate ($TP / [TP + FN]$) |
| **Specificity** | **82.59%** | True negative discovery rate ($TN / [TN + FP]$) |
| **ROC-AUC** | **0.9139** | Discrimination power across probability thresholds |
| **PR-AUC** | **0.9079** | Average Precision under precision-recall curve |
| **$P_{50}$ Latency** | **0.035 ms** | Median inference execution time per item |
| **$P_{95}$ Latency** | **0.047 ms** | 95th percentile execution time |
| **Throughput** | **26,533/s** | Single-core CPU inferences per second |

---

## Architecture

```
Binary-Text-Classification-with-Sklearn-Jax/
├── backend/                       # Production FastAPI Backend (Railway)
│   ├── app/
│   │   ├── main.py                # App factory, lifespan & exception handlers
│   │   ├── config.py              # Environment settings & rate limits
│   │   ├── logging_config.py      # RotatingFileHandler (5MB x 5 backups)
│   │   ├── api/                   # /predict, /models, /health, /benchmarks, /metrics
│   │   ├── middleware/            # Sliding-window rate limiting & security headers
│   │   ├── schemas/               # Pydantic v2 schemas
│   │   └── services/              # ClassifierService singleton & MetricsCollector
│   ├── models/sklearn/            # Serialized 3.8MB Scikit-Learn Pipeline
│   ├── Dockerfile                 # Pure CPU Python 3.10-slim image
│   ├── railway.json               # Railway build & healthcheck config
│   ├── Procfile                   # Process configuration
│   └── requirements.txt
│
├── frontend/                      # Production Frontend (Vercel)
│   ├── index.html                 # Semantic HTML with CSP & Research Modal
│   ├── styles.css                 # Studio Slate & Nordic Minimalist theme
│   ├── app.js                     # XSS-immune DOM rendering & robot emotions
│   ├── config.js                  # Dynamic API URL configuration
│   └── vercel.json                # Security headers & rewrites
│
├── src/                           # Reusable Machine Learning Package
│   ├── eda.py                     # Statistical profiling
│   ├── preprocessing.py           # Tokenization & vectorization
│   ├── training.py                # Pipeline construction & tuning
│   └── evaluation.py              # Comprehensive research metrics suite
│
├── scripts/                       # Modular CLI Automation Scripts
│   ├── eda.py                     # CLI for exploratory data analysis
│   ├── preprocess.py              # CLI for data preparation & split caching
│   ├── train.py                   # CLI for pipeline training & serialization
│   └── evaluate.py                # CLI for research metrics & latency benchmarks
│
├── monitoring/                    # Observability Stack
│   ├── prometheus.yml             # Prometheus scrape configuration
│   ├── docker-compose.monitoring.yml # Local Prometheus + Grafana stack
│   └── grafana/                   # Pre-configured dashboard & datasources
│
├── tests/                         # Test Suite (24 Pytest tests)
├── .github/workflows/ci.yml       # GitHub Actions CI Workflow
└── Makefile                       # Automation recipe suite
```

---

## Quick Start with Makefile

The repository includes a comprehensive `Makefile` for all tasks:

```bash
# View all available recipes
make help

# Run test suite
make test

# Start backend server (http://localhost:8000)
make run-backend

# Start frontend locally (http://localhost:3000)
make run-frontend

# Run model evaluation & latency benchmarks
make evaluate

# Tail active backend logs
make logs-backend

# Launch Prometheus (9090) and Grafana (3001)
make monitoring-up
```

---

## API Documentation

Interactive Swagger documentation is available at `http://localhost:8000/docs`.

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API status, metadata, and available routes |
| `GET` | `/health` | Health probe (model status, uptime, version) |
| `GET` | `/models` | Active model specification & pipeline stages |
| `GET` | `/benchmarks`| Research evaluation metrics & latency benchmarks |
| `POST` | `/predict` | Predict sentiment for submitted text payload |
| `GET` | `/metrics` | Prometheus exposition format metrics |

### Sample Prediction Request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "I absolutely loved this product! Outstanding craftsmanship."}'
```

**Response:**
```json
{
  "input": "I absolutely loved this product! Outstanding craftsmanship.",
  "prediction": 1,
  "sentiment": "Positive",
  "confidence": 0.9782,
  "probabilities": {
    "negative": 0.0218,
    "positive": 0.9782
  },
  "inference_time_ms": 1.42,
  "model_used": "sklearn-logreg",
  "api_version": "2.0.0"
}
```

---

## Deployment Guide

### Backend: Free Railway Deployment
1. Connect your repository to [Railway](https://railway.app).
2. Set root directory to `backend` (Railway auto-detects `Dockerfile` and `railway.json`).
3. Generate a public domain under Networking.
4. Healthcheck: `https://your-app.up.railway.app/health`.

### Frontend: Vercel Deployment
1. Update `frontend/config.js` with your Railway backend URL.
2. Deploy the `frontend/` directory to [Vercel](https://vercel.com).
3. The application is immediately live with HTTPS and edge CDN caching.

---

## Dataset & Citations

- **Dataset**: [Juggernaut Sentiment Analysis by Adeoluwa Adeboye (Kaggle)](https://www.kaggle.com/datasets/adeoluwa/juggernaut-sentiment-analysis).
- **Scikit-Learn**: [Pedregosa et al., JMLR 12, pp. 2825-2830, 2011](https://scikit-learn.org/).
- **FastAPI**: [Sebastián Ramírez](https://fastapi.tiangolo.com/).

---

## Git Staging & Release Instructions (v2.0.0)

To stage, commit, and showcase this release on GitHub without Git LFS:

```bash
# 1. Check current repository status
git status

# 2. Untrack legacy Git LFS filters from your local Git configuration
git lfs untrack "*.pkl" "*.npz" "*.csv" "*.joblib" 2>/dev/null || true

# 3. Stage clean configuration and code
git add .gitattributes .gitignore Makefile README.md CHANGELOG.md requirements.txt requirements-dev.txt
git add backend/ src/ scripts/ tests/ frontend/ monitoring/ .github/ utils.py

# 4. Stage the lightweight 3.8MB model
git add models/sklearn/logreg-80k.joblib

# 5. Commit release v2.0.0
git commit -m "feat: release v2.0.0 - pure scikit-learn migration, modular architecture, railway deploy, security & monitoring"

# 6. Create release tag
git tag -a v2.0.0 -m "Release v2.0.0: Pure Scikit-Learn Sentiment Classification Engine"

# 7. Push to remote repository
git push origin main --tags
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.