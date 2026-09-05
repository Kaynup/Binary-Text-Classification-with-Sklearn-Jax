"""
Exploratory Data Analysis (EDA) module for Binary Text Sentiment Classification.
Extracted and refactored from notebooks/brief-intro-analysis.ipynb.
"""
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from collections import Counter
import re


def clean_text_for_stats(text: str) -> List[str]:
    """Simple tokenization for fast word-level analysis."""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    return re.findall(r"\b[a-z]{2,}\b", text)


def run_eda(df: pd.DataFrame, text_col: str = "SentimentText", label_col: str = "Sentiment") -> Dict[str, Any]:
    """
    Run comprehensive statistical analysis on the sentiment dataset.
    """
    total_rows = len(df)
    missing_text = int(df[text_col].isna().sum())
    missing_labels = int(df[label_col].isna().sum())

    clean_df = df.dropna(subset=[text_col, label_col])
    labels = clean_df[label_col].astype(int)

    neg_count = int((labels == 0).sum())
    pos_count = int((labels == 1).sum())

    # Text length stats (characters and words)
    char_lengths = clean_df[text_col].astype(str).str.len()
    word_counts = clean_df[text_col].astype(str).str.split().apply(len)

    char_stats = {
        "mean": round(float(char_lengths.mean()), 2),
        "std": round(float(char_lengths.std()), 2),
        "min": int(char_lengths.min()),
        "p25": int(char_lengths.quantile(0.25)),
        "median": int(char_lengths.median()),
        "p75": int(char_lengths.quantile(0.75)),
        "p95": int(char_lengths.quantile(0.95)),
        "max": int(char_lengths.max()),
    }

    word_stats = {
        "mean": round(float(word_counts.mean()), 2),
        "std": round(float(word_counts.std()), 2),
        "min": int(word_counts.min()),
        "p25": int(word_counts.quantile(0.25)),
        "median": int(word_counts.median()),
        "p75": int(word_counts.quantile(0.75)),
        "p95": int(word_counts.quantile(0.95)),
        "max": int(word_counts.max()),
    }

    return {
        "total_records": total_rows,
        "valid_records": len(clean_df),
        "missing_text_count": missing_text,
        "missing_labels_count": missing_labels,
        "class_distribution": {
            "negative_0": neg_count,
            "positive_1": pos_count,
            "positive_ratio": round(pos_count / len(clean_df), 4) if len(clean_df) > 0 else 0.0,
            "negative_ratio": round(neg_count / len(clean_df), 4) if len(clean_df) > 0 else 0.0,
        },
        "character_length_distribution": char_stats,
        "word_count_distribution": word_stats
    }
