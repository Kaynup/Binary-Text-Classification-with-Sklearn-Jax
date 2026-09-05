# ==============================================================================
# Sentiment Classification Makefile (v2.0.0)
# End-to-end automation for local dev, training, testing, logs, and deployment.
# ==============================================================================

PYTHON ?= python3
PORT ?= 8000
FRONTEND_PORT ?= 3000

.PHONY: help install run-backend run-frontend test test-cov lint format clean \
        eda preprocess train evaluate logs-backend logs-errors logs-clear \
        docker-build docker-run monitoring-up monitoring-down railway-deploy-info

help: ## Show this help menu with all available recipes
	@echo "\n======================================================================"
	@echo "       Sentiment Analyzer v2.0.0 — Automation Recipes"
	@echo "======================================================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ------------------------------------------------------------------------------
# Environment & Setup
# ------------------------------------------------------------------------------

install: ## Install production and development dependencies
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-dev.txt

# ------------------------------------------------------------------------------
# Running Locally
# ------------------------------------------------------------------------------

run-backend: ## Run FastAPI server with auto-reload
	PORT=$(PORT) $(PYTHON) -m uvicorn backend.app.main:app --host 0.0.0.0 --port $(PORT) --reload

run-frontend: ## Serve frontend application locally
	@echo "Serving frontend at http://localhost:$(FRONTEND_PORT)"
	$(PYTHON) -m http.server $(FRONTEND_PORT) --directory frontend

dev: ## Run both backend and message for frontend
	@echo "Run 'make run-backend' in one terminal and 'make run-frontend' in another."

# ------------------------------------------------------------------------------
# Testing & Quality
# ------------------------------------------------------------------------------

test: ## Execute unit and integration tests with pytest
	$(PYTHON) -m pytest tests/ -v

test-cov: ## Run test suite with line coverage report
	$(PYTHON) -m pytest tests/ -v --cov=backend/app --cov=src --cov-report=term-missing

lint: ## Run flake8 style and syntax linter
	flake8 backend src scripts tests --count --select=E9,F63,F7,F82 --show-source --statistics

clean: ## Clean up bytecode, cache files, and test coverage artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[cod]" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	rm -rf .coverage htmlcov/ .tox/

# ------------------------------------------------------------------------------
# ML Pipeline Scripts (Converted from Notebooks)
# ------------------------------------------------------------------------------

eda: ## Run exploratory data analysis on raw sentiment dataset
	$(PYTHON) scripts/eda.py

preprocess: ## Preprocess dataset and generate TF-IDF vector features
	$(PYTHON) scripts/preprocess.py

train: ## Train Scikit-Learn Logistic Regression pipeline and save model
	$(PYTHON) scripts/train.py

evaluate: ## Benchmark research metrics and inference latency on model
	$(PYTHON) scripts/evaluate.py

# ------------------------------------------------------------------------------
# Logging & Diagnostics
# ------------------------------------------------------------------------------

logs-backend: ## Tail active rotating backend logs with live follow
	@if [ -f logs/sentiment_analyzer.log ]; then \
		tail -f -n 50 logs/sentiment_analyzer.log; \
	elif [ -f backend/inference.log ]; then \
		tail -f -n 50 backend/inference.log; \
	else \
		echo "No log file found yet. Start the backend to generate logs."; \
	fi

logs-errors: ## Filter logs for ERROR and CRITICAL entries
	@if [ -f logs/sentiment_analyzer.log ]; then \
		grep -E "ERROR|CRITICAL" logs/sentiment_analyzer.log || echo "No errors found."; \
	elif [ -f backend/inference.log ]; then \
		grep -E "ERROR|CRITICAL" backend/inference.log || echo "No errors found."; \
	else \
		echo "No log file found."; \
	fi

logs-clear: ## Clear log files safely
	@mkdir -p logs
	> logs/sentiment_analyzer.log 2>/dev/null || true
	@echo "Logs cleared."

# ------------------------------------------------------------------------------
# Container & Deployment
# ------------------------------------------------------------------------------

docker-build: ## Build lightweight backend Docker container image
	docker build -t sentiment-api:v2.0.0 backend/

docker-run: ## Run backend Docker container locally
	docker run -p $(PORT):8000 -e PORT=8000 --name sentiment_api_container --rm sentiment-api:v2.0.0

# ------------------------------------------------------------------------------
# Monitoring (Prometheus & Grafana)
# ------------------------------------------------------------------------------

monitoring-up: ## Start local Prometheus (9090) and Grafana (3001) stack
	docker compose -f monitoring/docker-compose.monitoring.yml up -d
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana:    http://localhost:3001 (login: admin/admin)"

monitoring-down: ## Stop monitoring stack
	docker compose -f monitoring/docker-compose.monitoring.yml down

# ------------------------------------------------------------------------------
# Deployment Assistance
# ------------------------------------------------------------------------------

railway-deploy-info: ## Show instructions to deploy backend to Railway
	@echo "\n======================================================================"
	@echo "              Railway Backend Deployment Instructions"
	@echo "======================================================================"
	@echo " 1. Install CLI:  npm install -g @railway/cli"
	@echo " 2. Navigate:     cd backend"
	@echo " 3. Login:        railway login"
	@echo " 4. Initialize:   railway init"
	@echo " 5. Deploy:       railway up"
	@echo " 6. Get URL:      railway domain"
	@echo " 7. Healthcheck:  curl https://your-railway-url.up.railway.app/health"
	@echo "======================================================================\n"
