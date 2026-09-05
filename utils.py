"""
Utilities for Binary Text Sentiment Classification (v2.0.0)
Pure Scikit-Learn & NumPy Implementation (No JAX dependencies)
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import load_npz
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
    log_loss
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_processed(base_dir=None, config_id=None):
    """
    Load preprocessed TF-IDF matrices, labels, vectorizer, and config.
    """
    if base_dir is None:
        base_dir = os.path.join(BASE_DIR, "data", "processed")

    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Processed data directory not found: {base_dir}")

    all_dirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])

    if config_id == "all":
        target_dirs = all_dirs
    elif isinstance(config_id, str):
        target_dirs = [config_id]
    elif isinstance(config_id, list):
        target_dirs = config_id
    else:
        raise ValueError("'config_id' must be 'all', a string, or a list of strings.")

    # Validate target_dirs exist
    for cfg in target_dirs:
        if cfg not in all_dirs:
            raise FileNotFoundError(f"Dataset directory not found: {os.path.join(base_dir, cfg)}")

    def load_single(cfg):
        cfg_dir = os.path.join(base_dir, cfg)
        X_train = load_npz(os.path.join(cfg_dir, "X_train_tfidf.npz"))
        X_test = load_npz(os.path.join(cfg_dir, "X_test_tfidf.npz"))
        y_train = joblib.load(os.path.join(cfg_dir, "y_train.pkl"))
        y_test = joblib.load(os.path.join(cfg_dir, "y_test.pkl"))
        vectorizer = joblib.load(os.path.join(cfg_dir, "vectorizer.pkl"))
        with open(os.path.join(cfg_dir, "config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)
        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "vectorizer": vectorizer,
            "config": config
        }

    if len(target_dirs) == 1:
        ds = load_single(target_dirs[0])
        return ds["X_train"], ds["X_test"], ds["y_train"], ds["y_test"], ds["vectorizer"], ds["config"]
    else:
        datasets = {}
        for cfg in target_dirs:
            datasets[cfg] = load_single(cfg)
        return datasets


def load_raw(base_dir=None, file_name="Juggernaut Sentiment Analysis - by kaggle user Adeoluwa Adeboye.csv"):
    """
    Load raw CSV sentiment data.
    """
    if base_dir is None:
        base_dir = os.path.join(BASE_DIR, "data", "raw", file_name)
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Raw dataset file not found: {base_dir}")
    return pd.read_csv(base_dir, on_bad_lines="skip")


def compute_research_metrics(y_true, y_pred, y_prob=None):
    """
    Compute rigorous research-grade classification metrics for binary sentiment classification.
    Returns:
        dict: Complete metric suite including Accuracy, Precision, Recall, Specificity,
              F1, Balanced Accuracy, MCC, ROC-AUC, PR-AUC, and Confusion Matrix breakdown.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(spec, 4),
        "f1_score": round(f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "matthews_corrcoef": round(mcc, 4),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "total_samples": int(len(y_true))
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        # If 2D probability array passed (e.g. from predict_proba), take positive class
        if y_prob.ndim == 2:
            y_prob = y_prob[:, 1]
        try:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
            metrics["pr_auc"] = round(float(average_precision_score(y_true, y_prob)), 4)
            metrics["log_loss"] = round(float(log_loss(y_true, y_prob)), 4)
        except Exception:
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None
            metrics["log_loss"] = None

    return metrics


def batch_iter(X, y, batch_size=256, shuffle=True):
    """
    Yield batches of features and targets.
    """
    idxs = np.arange(len(X))
    if shuffle:
        np.random.shuffle(idxs)
    for i in range(0, len(X), batch_size):
        b = idxs[i:i + batch_size]
        yield X[b], y[b]