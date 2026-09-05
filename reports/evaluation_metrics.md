# Binary Sentiment Classifier v2.0.0 Evaluation

| Metric | Score | Description |
|---|---|---|
| **Accuracy** | `0.8407` | Fraction of all correct predictions |
| **F1-Score** | `0.8426` | Harmonic mean of precision and recall |
| **Precision** | `0.8300` | True Positive / (True Positive + False Positive) |
| **Recall / Sensitivity** | `0.8555` | True Positive / (True Positive + False Negative) |
| **Specificity** | `0.8259` | True Negative / (True Negative + False Positive) |
| **Balanced Accuracy** | `0.8407` | Mean of sensitivity and specificity |
| **Matthews Corr. Coef (MCC)** | `0.6817` | Balanced correlation metric [-1 to +1] |
| **ROC-AUC** | `0.9139` | Area Under Receiver Operating Characteristic Curve |
| **PR-AUC (Avg Precision)** | `0.9079` | Area Under Precision-Recall Curve |
| **Log Loss (Cross-Entropy)** | `0.3803` | Quality of probabilistic predictions |

### Confusion Matrix Breakdown
- **True Negatives (TN)**: 1,243
- **False Positives (FP)**: 262
- **False Negatives (FN)**: 216
- **True Positives (TP)**: 1,279
- **Total Evaluated Samples**: 3,000

### Latency & Throughput Benchmarks
- **P50 Latency**: 0.035 ms/item
- **P95 Latency**: 0.047 ms/item
- **P99 Latency**: 0.068 ms/item
- **Throughput**: 26533.3 inferences/second
