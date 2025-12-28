# Sentiment Analysis API - Dockerfile
# Optimized for Railway, Render, and Fly.io deployment

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV ENABLE_JAX=false

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (Railway will override with PORT env var)
EXPOSE 8000

# Run the application - Use shell form to allow variable substitution
# Railway provides PORT env variable
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
