import os
import json
import numpy as np
import joblib
import pandas
from scipy.sparse import load_npz
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_processed(base_dir=None, config_id=None):
    if base_dir is None:
        base_dir = os.path.join(BASE_DIR, "data", "processed")

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
        with open(os.path.join(cfg_dir, "config.json"), "r") as f:
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
            print(f"Loaded dataset: {cfg}")
        return datasets



def load_raw(base_dir=None, file_name='Juggernaut Sentiment Analysis - by kaggle user Adeoluwa Adeboye.csv'):
    if base_dir is None:
        base_dir = os.path.join(BASE_DIR, "data", "raw", file_name)
    return pandas.read_csv(base_dir, on_bad_lines='skip')


def binary_loss(logits, labels):
    # logits: (batch, ), labels: (batch, )
    loss = optax.sigmoid_binary_cross_entropy(logits, labels)
    return loss.mean()


def compute_metrics(logits, labels):
    preds = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.int32)
    acc = (preds == labels).mean()

    # True positives, false positives, false negatives
    tp = jnp.sum((preds == 1) & (labels == 1))
    fp = jnp.sum((preds == 1) & (labels == 0))
    fn = jnp.sum((preds == 0) & (labels == 1))

    # Precision, Recall, F1 - safety from divide-by-zero
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def batch_iter(X, y, batch_size, shuffle=True):
    idxs = np.arange(len(X))
    if shuffle:
        np.random.shuffle(idxs)
    for i in range(0, len(X), batch_size):
        b = idxs[i:i+batch_size]
        yield X[b], y[b]