"""
Integration tests for FastAPI endpoints, responses, and Prometheus scraping.
"""
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert data["version"] == "2.0.0"
    assert "predict" in data["endpoints"]


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("ok", "degraded")
    assert "uptime_seconds" in data
    assert "model_loaded" in data


def test_models_endpoint():
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "available_models" in data
    assert "sklearn-logreg" in data["available_models"]


def test_benchmarks_endpoint():
    response = client.get("/benchmarks")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert data["metrics"]["f1_score"] >= 0.80


def test_predict_positive():
    payload = {"text": "This is simply fantastic, wonderful results and superb delivery!"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 1
    assert data["sentiment"] == "Positive"
    assert data["confidence"] >= 0.5
    assert "inference_time_ms" in data
    assert data["model_used"] == "sklearn-logreg"
    assert data["api_version"] == "2.0.0"


def test_predict_negative():
    payload = {"text": "Awful and useless product, broke right out of the box."}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 0
    assert data["sentiment"] == "Negative"


def test_predict_empty_text_validation_error():
    payload = {"text": "   "}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] is True


def test_predict_missing_text_field():
    payload = {}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_prometheus_metrics_endpoint():
    # Make a prediction to generate metrics
    client.post("/predict", json={"text": "Generating prometheus metric sample"})
    response = client.get("/metrics")
    assert response.status_code == 200
    text = response.text
    assert "http_requests_total" in text
    assert "app_uptime_seconds" in text
    assert "sentiment_predictions_total" in text
