# Sentiment Analysis API - Lightweight Production Dockerfile
# Optimized for Railway Free Tier (Pure CPU Scikit-Learn)

FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app:${PYTHONPATH}"

# Install curl for container healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend application code & models
COPY . .

# Ensure logs directory exists
RUN mkdir -p logs

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Run FastAPI via uvicorn
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
