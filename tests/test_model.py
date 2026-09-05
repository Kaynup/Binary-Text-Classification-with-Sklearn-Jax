"""
Unit tests for Scikit-Learn Model Loading, Pipeline Inference, and Confidence Scores.
"""
import pytest
from backend.app.services.classifier_service import classifier_service


@pytest.fixture(scope="module", autouse=True)
def ensure_model_loaded():
    classifier_service.load_model()
    assert classifier_service.is_loaded, f"Model failed to load: {classifier_service._load_error}"


def test_model_loaded_successfully():
    assert classifier_service.is_loaded is True
    assert classifier_service._model is not None


def test_positive_sentiment_prediction():
    sample = "I absolutely loved this product! Outstanding craftsmanship and rapid shipping."
    result = classifier_service.predict(sample)

    assert result["prediction"] == 1
    assert result["sentiment"] == "Positive"
    assert 0.5 <= result["confidence"] <= 1.0
    assert result["probabilities"]["positive"] > result["probabilities"]["negative"]
    assert abs((result["probabilities"]["positive"] + result["probabilities"]["negative"]) - 1.0) < 1e-4
    assert result["inference_time_ms"] > 0


def test_negative_sentiment_prediction():
    sample = "Worst customer experience imaginable. The item arrived broken and defective."
    result = classifier_service.predict(sample)

    assert result["prediction"] == 0
    assert result["sentiment"] == "Negative"
    assert 0.5 <= result["confidence"] <= 1.0
    assert result["probabilities"]["negative"] > result["probabilities"]["positive"]


def test_model_metadata():
    info = classifier_service.get_model_info()
    assert "sklearn-logreg" in info["available_models"]
    assert "Logistic Regression" in info["model_type"]
    assert info["features_count"] > 0
    assert info["version"] == "2.0.0"


def test_research_benchmarks():
    benchmarks = classifier_service.get_research_benchmarks()
    assert "metrics" in benchmarks
    m = benchmarks["metrics"]
    assert m["f1_score"] > 0.80
    assert m["accuracy"] > 0.80
    assert m["roc_auc"] > 0.85
