"""
Model Training and Pipeline Serialization.
Extracted and modularized from notebooks/sklearn-lab.ipynb and notebooks/sklearn-model-save.ipynb.
"""
import time
import os
from typing import Dict, Any, Tuple, Optional
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.evaluation import evaluate_predictions


def create_sentiment_pipeline(
    max_features: int = 80000,
    ngram_range: Tuple[int, int] = (1, 5),
    solver: str = "saga",
    penalty: str = "l2",
    max_iter: int = 800,
    C: float = 1.0,
    random_state: int = 42
) -> Pipeline:
    """
    Constructs the production end-to-end sentiment classification pipeline:
    TfidfVectorizer -> LogisticRegression.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True
    )
    classifier = LogisticRegression(
        solver=solver,
        penalty=penalty,
        max_iter=max_iter,
        C=C,
        random_state=random_state,
        n_jobs=-1
    )
    pipeline = Pipeline([
        ("vectorizer", vectorizer),
        ("classifier", classifier)
    ])
    return pipeline


def train_and_evaluate_pipeline(
    pipeline: Pipeline,
    X_train,
    y_train,
    X_test,
    y_test
) -> Tuple[Pipeline, Dict[str, Any], float]:
    """
    Fits the pipeline and computes complete research evaluation metrics on the test set.
    """
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    training_duration = time.time() - t0

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None

    metrics = evaluate_predictions(y_test, y_pred, y_prob)
    metrics["training_duration_seconds"] = round(training_duration, 2)

    return pipeline, metrics, training_duration


def tune_logistic_regression(
    X_train_vec,
    y_train,
    n_iter: int = 10,
    cv_folds: int = 3,
    random_state: int = 42
) -> Tuple[LogisticRegression, Dict[str, Any]]:
    """
    Runs hyperparameter search over Logistic Regression solvers, penalties, and C parameters.
    """
    param_dist = {
        "C": [0.01, 0.1, 1.0, 5.0, 10.0],
        "penalty": ["l2"],
        "solver": ["liblinear", "saga", "lbfgs"],
        "max_iter": [500, 800, 1000]
    }
    base_model = LogisticRegression(random_state=random_state)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="f1",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    search.fit(X_train_vec, y_train)
    return search.best_estimator_, {
        "best_params": search.best_params_,
        "best_score": round(float(search.best_score_), 4)
    }


def save_pipeline(pipeline: Pipeline, output_path: str) -> None:
    """Save pipeline to specified path, ensuring parent directories exist."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    joblib.dump(pipeline, output_path)
