"""
Research-Grade Evaluation Metrics Suite for Binary Sentiment Classification.
Includes standard discrimination metrics, threshold-invariant metrics,
loss measures, and inference performance benchmarking.
"""
import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    confusion_matrix,
    log_loss,
    classification_report
)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Computes a comprehensive dictionary of binary classification research metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    total = len(y_true)
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    metrics: Dict[str, Any] = {
        "sample_count": total,
        "class_distribution": {
            "negative_samples": int((y_true == 0).sum()),
            "positive_samples": int((y_true == 1).sum()),
            "positive_ratio": round(float((y_true == 1).mean()), 4) if total > 0 else 0.0,
        },
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall_sensitivity": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "matthews_corrcoef": round(mcc, 4),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        }
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        if y_prob.ndim == 2:
            y_prob_pos = y_prob[:, 1]
        else:
            y_prob_pos = y_prob

        try:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob_pos)), 4)
            metrics["pr_auc"] = round(float(average_precision_score(y_true, y_prob_pos)), 4)
            metrics["log_loss"] = round(float(log_loss(y_true, y_prob_pos)), 4)
        except Exception:
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None
            metrics["log_loss"] = None

    return metrics


def benchmark_inference_latency(
    model_predict_fn,
    sample_texts: List[str],
    warmup_runs: int = 5,
    benchmark_runs: int = 30
) -> Dict[str, float]:
    """
    Benchmarks model inference latency and throughput.
    Returns:
        P50, P95, P99 latency in ms and throughput in inferences/sec.
    """
    # Warmup
    for _ in range(warmup_runs):
        _ = model_predict_fn(sample_texts[:min(10, len(sample_texts))])

    durations: List[float] = []
    total_items = 0

    for _ in range(benchmark_runs):
        t0 = time.perf_counter()
        _ = model_predict_fn(sample_texts)
        t1 = time.perf_counter()
        durations.append((t1 - t0) * 1000)  # ms
        total_items += len(sample_texts)

    durations_arr = np.array(durations)
    per_item_ms = durations_arr / max(1, len(sample_texts))

    return {
        "batch_size": len(sample_texts),
        "mean_batch_ms": round(float(np.mean(durations_arr)), 3),
        "p50_item_ms": round(float(np.percentile(per_item_ms, 50)), 3),
        "p95_item_ms": round(float(np.percentile(per_item_ms, 95)), 3),
        "p99_item_ms": round(float(np.percentile(per_item_ms, 99)), 3),
        "throughput_inferences_per_sec": round(float(total_items / (np.sum(durations) / 1000)), 1)
    }


def format_metrics_report(metrics: Dict[str, Any], title: str = "Sentiment Classification Evaluation") -> str:
    """
    Format metrics into a clean markdown table and summary for documentation and CI.
    """
    cm = metrics.get("confusion_matrix", {})
    roc_auc_val = metrics.get("roc_auc", "N/A")
    pr_auc_val = metrics.get("pr_auc", "N/A")
    log_loss_val = metrics.get("log_loss", "N/A")

    report = f"""# {title}

| Metric | Score | Description |
|---|---|---|
| **Accuracy** | `{metrics.get('accuracy', 0.0):.4f}` | Fraction of all correct predictions |
| **F1-Score** | `{metrics.get('f1_score', 0.0):.4f}` | Harmonic mean of precision and recall |
| **Precision** | `{metrics.get('precision', 0.0):.4f}` | True Positive / (True Positive + False Positive) |
| **Recall / Sensitivity** | `{metrics.get('recall_sensitivity', 0.0):.4f}` | True Positive / (True Positive + False Negative) |
| **Specificity** | `{metrics.get('specificity', 0.0):.4f}` | True Negative / (True Negative + False Positive) |
| **Balanced Accuracy** | `{metrics.get('balanced_accuracy', 0.0):.4f}` | Mean of sensitivity and specificity |
| **Matthews Corr. Coef (MCC)** | `{metrics.get('matthews_corrcoef', 0.0):.4f}` | Balanced correlation metric [-1 to +1] |
| **ROC-AUC** | `{roc_auc_val}` | Area Under Receiver Operating Characteristic Curve |
| **PR-AUC (Avg Precision)** | `{pr_auc_val}` | Area Under Precision-Recall Curve |
| **Log Loss (Cross-Entropy)** | `{log_loss_val}` | Quality of probabilistic predictions |

### Confusion Matrix Breakdown
- **True Negatives (TN)**: {cm.get('true_negative', 0):,}
- **False Positives (FP)**: {cm.get('false_positive', 0):,}
- **False Negatives (FN)**: {cm.get('false_negative', 0):,}
- **True Positives (TP)**: {cm.get('true_positive', 0):,}
- **Total Evaluated Samples**: {metrics.get('sample_count', 0):,}
"""
    return report
