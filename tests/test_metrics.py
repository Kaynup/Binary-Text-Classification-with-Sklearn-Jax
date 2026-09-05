"""
Unit tests for Research Evaluation Metrics mathematical correctness.
"""
import numpy as np
import pytest
from src.evaluation import evaluate_predictions, benchmark_inference_latency


def test_perfect_predictions_metrics():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.1, 0.2])

    metrics = evaluate_predictions(y_true, y_pred, y_prob)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall_sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["matthews_corrcoef"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"]["true_positive"] == 2
    assert metrics["confusion_matrix"]["true_negative"] == 2
    assert metrics["confusion_matrix"]["false_positive"] == 0
    assert metrics["confusion_matrix"]["false_negative"] == 0


def test_imbalanced_predictions_metrics():
    # 3 TP, 1 FP, 1 FN, 3 TN
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 1, 0, 0, 0])

    metrics = evaluate_predictions(y_true, y_pred)

    assert metrics["accuracy"] == 6 / 8
    # Precision = 3 / (3 + 1) = 0.75
    assert metrics["precision"] == 0.75
    # Recall = 3 / (3 + 1) = 0.75
    assert metrics["recall_sensitivity"] == 0.75
    assert metrics["f1_score"] == 0.75
    assert metrics["sample_count"] == 8


def test_latency_benchmarking_runs():
    dummy_predict_fn = lambda texts: [1] * len(texts)
    sample_texts = ["test one", "test two", "test three"]

    bench = benchmark_inference_latency(dummy_predict_fn, sample_texts, warmup_runs=2, benchmark_runs=5)

    assert "p50_item_ms" in bench
    assert "p95_item_ms" in bench
    assert "throughput_inferences_per_sec" in bench
    assert bench["throughput_inferences_per_sec"] > 0
