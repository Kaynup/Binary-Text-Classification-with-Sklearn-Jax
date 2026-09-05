"""
Data Preprocessing and Vectorization Pipeline.
Extracted and modularized from notebooks/data-preprocessing.ipynb.
"""
import os
import json
import time
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import save_npz, load_npz
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from src.tokenizers import custom_tokenizer


def prepare_dataset(
    df: pd.DataFrame,
    text_col: str = "SentimentText",
    label_col: str = "Sentiment",
    test_size: float = 0.15,
    random_state: int = 98
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Standardize column names, remove nulls, and execute a stratified train-test split.
    """
    clean_df = df.dropna(subset=[text_col, label_col]).copy()
    X = clean_df[text_col].astype(str)
    y = clean_df[label_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test


def fit_vectorizer(
    X_train: pd.Series,
    max_features: int = 80000,
    ngram_range: Tuple[int, int] = (1, 5),
    use_custom_tokenizer: bool = False
) -> Tuple[TfidfVectorizer, Any, float]:
    """
    Fit a TfidfVectorizer on the training set and transform it.
    """
    tokenizer = custom_tokenizer if use_custom_tokenizer else None
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        tokenizer=tokenizer,
        token_pattern=None if use_custom_tokenizer else r"(?u)\b\w+\b"
    )
    t0 = time.time()
    X_train_vec = vec.fit_transform(X_train)
    fit_duration = time.time() - t0

    return vec, X_train_vec, fit_duration


def save_processed_split(
    output_dir: str,
    X_train_vec,
    X_test_vec,
    y_train: np.ndarray,
    y_test: np.ndarray,
    vectorizer: TfidfVectorizer,
    metadata: Dict[str, Any]
) -> None:
    """
    Save the processed vector matrices, labels, vectorizer, and metadata configuration.
    """
    os.makedirs(output_dir, exist_ok=True)

    save_npz(os.path.join(output_dir, "X_train_tfidf.npz"), X_train_vec)
    save_npz(os.path.join(output_dir, "X_test_tfidf.npz"), X_test_vec)

    joblib.dump(np.asarray(y_train, dtype=int), os.path.join(output_dir, "y_train.pkl"))
    joblib.dump(np.asarray(y_test, dtype=int), os.path.join(output_dir, "y_test.pkl"))
    joblib.dump(vectorizer, os.path.join(output_dir, "vectorizer.pkl"))

    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
