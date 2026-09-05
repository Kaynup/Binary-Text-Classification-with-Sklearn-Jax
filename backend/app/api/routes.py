"""
FastAPI Routes for Prediction, Models, Health, and Research Benchmarks.
"""
import time
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Depends
from backend.app.schemas.sentiment import (
    PredictRequest,
    PredictResponse,
    ProbabilityScores,
    ModelInfoResponse,
    HealthResponse,
    ResearchBenchmarkResponse
)
from backend.app.services.classifier_service import classifier_service
from backend.app.services.metrics_collector import metrics_collector
from backend.app.config import settings
from backend.app.logging_config import get_logger

logger = get_logger("sentiment_analyzer.routes")
router = APIRouter()

APP_BOOT_TIME = time.time()


@router.get("/", summary="API Root & Service Information")
def root():
    """Returns general service information and available endpoints."""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "framework": "Scikit-Learn (Pure CPU)",
        "model": "Logistic Regression + TF-IDF (1-5 N-grams)",
        "status": "operational",
        "documentation": "/docs",
        "endpoints": {
            "predict": "POST /predict",
            "health": "GET /health",
            "models": "GET /models",
            "benchmarks": "GET /benchmarks",
            "metrics": "GET /metrics (Prometheus)"
        }
    }


@router.get("/health", response_model=HealthResponse, summary="Health Check")
def health_check():
    """Liveness and readiness probe for Railway / container orchestration."""
    uptime = time.time() - APP_BOOT_TIME
    return HealthResponse(
        status="ok" if classifier_service.is_loaded else "degraded",
        model_loaded=classifier_service.is_loaded,
        model_path=classifier_service._model_path or "unloaded",
        version=settings.APP_VERSION,
        uptime_seconds=round(uptime, 2),
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@router.get("/models", response_model=ModelInfoResponse, summary="Inspect Available Models")
def get_models():
    """Returns active model metadata and pipeline configurations."""
    info = classifier_service.get_model_info()
    return ModelInfoResponse(**info)


@router.get("/benchmarks", response_model=ResearchBenchmarkResponse, summary="Research Benchmark Metrics")
def get_benchmarks():
    """Exposes official research evaluation metrics (Accuracy, F1, ROC-AUC, Latency)."""
    benchmarks = classifier_service.get_research_benchmarks()
    return ResearchBenchmarkResponse(**benchmarks)


@router.post("/predict", response_model=PredictResponse, summary="Predict Sentiment")
def predict_sentiment(payload: PredictRequest):
    """
    Predicts sentiment for submitted text.
    Returns binary classification (0: Negative, 1: Positive), confidence, probabilities, and latency.
    """
    # Sanitize and validate input
    raw_text = payload.text.strip()
    if not raw_text:
        raise HTTPException(status_code=422, detail="Input text cannot be empty or whitespace only.")

    try:
        result = classifier_service.predict(raw_text)

        # Record metric for Prometheus / Grafana
        metrics_collector.record_prediction(result["sentiment"])

        # Truncate input preview for safety in response
        input_preview = raw_text[:200] + ("..." if len(raw_text) > 200 else "")

        response_dict = {
            "input": input_preview,
            "prediction": result["prediction"],
            "sentiment": result["sentiment"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "inference_time_ms": result["inference_time_ms"],
            "model_used": payload.model or "sklearn-logreg",
            "api_version": settings.APP_VERSION,
            # Backwards compatibility with legacy frontend clients
            "Jax model": False
        }

        return response_dict

    except Exception as e:
        logger.error(f"Inference error on input text: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Model inference failed. Please check server logs."
        )
