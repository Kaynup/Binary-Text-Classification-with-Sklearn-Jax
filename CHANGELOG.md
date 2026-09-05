# Changelog

All notable changes to the Binary Sentiment Classification project will be documented in this file.

## [2.0.0] - 2026-09-05

### Major Architectural Upgrade & Pure Scikit-Learn Migration

#### 1. Framework Migration (Pure Scikit-Learn)
- **Eliminated JAX/Flax/Optax**: Removed all heavy neural models, Flax RNN modules (`src/modelling/rnn`), and JAX dependency overhead.
- **Pure Scikit-Learn Architecture**: Established high-performance, interpretable binary sentiment classification using TF-IDF n-grams (1-5) and regularized Logistic Regression (`solver='saga'`, `penalty='l2'`, `C=1.0`, `80,000` features).
- **Sub-Millisecond Inference**: Achieved single-item P50 inference latency under 0.05 ms on standard CPU with throughput exceeding 25,000 inferences/second.

#### 2. Git LFS Elimination & Storage Optimization
- **Cleaned `.gitattributes`**: Removed all Git LFS filters (`*.pkl`, `*.npz`, `*.csv`, `*.joblib` filter=lfs).
- **Lightweight Repository**: Pruned ~400MB of legacy JAX checkpoints.
- **Configured `.gitignore`**: Strictly ignored raw CSV datasets (157MB) and processed TF-IDF matrix caches (`*.npz`).
- **Production Asset**: Retained only the lightweight 3.8MB trained Scikit-Learn model pipeline (`models/sklearn/logreg-80k.joblib`).

#### 3. Research Evaluation Metrics Everywhere
- Added comprehensive research-grade metrics across training, evaluation, and API endpoints:
  - Accuracy: **84.07%**
  - F1-Score: **84.26%**
  - Precision: **83.00%**
  - Recall / Sensitivity: **85.55%**
  - Specificity: **82.59%**
  - ROC-AUC: **0.9139**
  - PR-AUC: **0.9079**
  - Matthews Correlation Coefficient (MCC): **0.6815**
- Added `/benchmarks` API endpoint and an interactive in-app Research Metrics drawer.

#### 4. Notebooks to Production CLI Scripts
- Refactored Jupyter notebooks into modular, command-line scripts in `scripts/`:
  - `scripts/eda.py`: Statistical dataset profiling, label balance, and token length quantiles.
  - `scripts/preprocess.py`: Stratified train/test splitting and TF-IDF vector fitting.
  - `scripts/train.py`: Pipeline training, hyperparameter cross-validation, and joblib serialization.
  - `scripts/evaluate.py`: Test partition evaluation and latency benchmarking ($P_{50}, P_{95}, P_{99}$).

#### 5. High Cohesion & Low Coupling Backend
- Redesigned `backend/app/` with clear architectural boundaries:
  - `api/`: Route controllers (`/predict`, `/models`, `/health`, `/benchmarks`, `/metrics`).
  - `middleware/`: In-memory sliding-window IP rate limiter, security headers, and request timing.
  - `schemas/`: Pydantic v2 data transfer objects with strict validation.
  - `services/`: Singleton classifier inference service and Prometheus metrics collector.
  - `logging_config.py`: Production logging with `RotatingFileHandler` (5MB max, 5 backups) and ISO console stream.

#### 6. Production Railway & Vercel Deployment
- **Free Railway Backend**: Optimized lightweight `Dockerfile` (Python 3.10-slim), `railway.json` with health check, and `Procfile`.
- **Vercel Frontend**: Streamlined static hosting configuration in `frontend/` with root `vercel.json` and strict security headers.
- **In-Memory Rate Limiting**: Enforced sliding-window limits (default: 60 req/min for predict, 120 req/min general) returning HTTP 429 with `Retry-After`.

#### 7. Frontend Redesign & Security Hardening
- **Studio Slate & Nordic Minimalist Theme**: Eliminated all neon gradients, fuzzy radial blurs, and electric glows in favor of a crisp, tactile, low-fatigue slate palette.
- **Character Animation Intact**: Preserved interactive robot emotion states (Neutral, Happy with heart bursts/blushing cheeks, Sad with tears/rain).
- **100% XSS Prevention**: Eliminated dangerous `innerHTML` string interpolation; all user input rendered safely via dedicated `escapeHtml` and `.textContent`.
- **Content Security Policy (CSP)**: Added strict meta tag and `vercel.json` headers.
- **Tactile UI Refinements**:
  - Live dual counters (word count + character limit) with reactive input feedback.
  - Live engine connectivity badge with automated 30s `/health` heartbeat polling.
  - Calculated continuous polarity index (-1.00 to +1.00) with categorical intensity descriptors.
  - 1-click clipboard result export with floating toast feedback.
  - Cleaned up obsolete legacy JAX screenshot assets from `src/images/`.

#### 8. Observability & Monitoring
- Built Prometheus metrics exposition at `/metrics` tracking request counts, response latency histograms, sentiment distribution, and active connections.
- Included Grafana dashboard JSON (`monitoring/grafana/dashboards/sentiment_analyzer.json`) and Docker Compose stack for one-command local observability.

#### 9. Developer Experience & Automation
- **Makefile**: Comprehensive recipes (`make help`, `make run-backend`, `make run-frontend`, `make test`, `make eda`, `make train`, `make evaluate`, `make logs-backend`, `make docker-build`, etc.).
- **GitHub Actions CI**: Automated multi-version matrix testing (Python 3.10, 3.11, 3.12), flake8 linting, and Docker container verification.
- **Unit & Integration Tests**: Comprehensive 24-test pytest suite in `tests/`.
