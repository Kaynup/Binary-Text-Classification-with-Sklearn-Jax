#!/usr/bin/env python3
"""
CLI Script: End-to-End Sentiment Classification Model Training and Pipeline Serialization.
Usage:
    python scripts/train.py --sample 50000 --output models/sklearn/logreg-80k.joblib
"""
import argparse
import os
import sys
import json
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import prepare_dataset
from src.training import create_sentiment_pipeline, train_and_evaluate_pipeline, save_pipeline
from src.evaluation import format_metrics_report


def main():
    parser = argparse.ArgumentParser(description="Train end-to-end sentiment classification pipeline")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/Juggernaut Sentiment Analysis - by kaggle user Adeoluwa Adeboye.csv",
        help="Path to raw dataset CSV"
    )
    parser.add_argument("--output", type=str, default="models/sklearn/logreg-80k.joblib", help="Output model path")
    parser.add_argument("--report-output", type=str, default="reports/training_report.md", help="Path for report")
    parser.add_argument("--sample", type=int, default=None, help="Sample size limit for quick training")
    parser.add_argument("--max-features", type=int, default=80000, help="Max vocabulary features")
    parser.add_argument("--c-param", type=float, default=1.0, help="Inverse regularization strength C")
    parser.add_argument("--max-iter", type=int, default=800, help="Max solver iterations")
    parser.add_argument("--solver", type=str, default="saga", help="Solver: saga, liblinear, lbfgs")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Error: Dataset not found at '{args.data_path}'")
        sys.exit(1)

    print(f"Loading data from {args.data_path}...")
    df = pd.read_csv(args.data_path, on_bad_lines="skip")
    if args.sample and args.sample < len(df):
        print(f"Subsampling to {args.sample:,} rows...")
        df = df.sample(n=args.sample, random_state=42)

    print("Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = prepare_dataset(df, test_size=0.15, random_state=98)
    print(f"Train samples: {len(X_train):,}, Test samples: {len(X_test):,}")

    print(f"Constructing end-to-end Pipeline (features={args.max_features:,}, solver={args.solver}, C={args.c_param})...")
    pipeline = create_sentiment_pipeline(
        max_features=args.max_features,
        ngram_range=(1, 5),
        solver=args.solver,
        penalty="l2",
        max_iter=args.max_iter,
        C=args.c_param
    )

    print("Fitting pipeline (TF-IDF Vectorizer + Logistic Regression)...")
    fitted_pipeline, metrics, duration = train_and_evaluate_pipeline(
        pipeline, X_train, y_train, X_test, y_test
    )
    print(f"Training completed in {duration:.2f} seconds.")

    print("\n" + "=" * 60)
    print("                 MODEL TEST EVALUATION METRICS")
    print("=" * 60)
    print(f"Accuracy:          {metrics['accuracy'] * 100:.2f}%")
    print(f"F1-Score:          {metrics['f1_score'] * 100:.2f}%")
    print(f"Precision:         {metrics['precision'] * 100:.2f}%")
    print(f"Recall:            {metrics['recall_sensitivity'] * 100:.2f}%")
    print(f"Specificity:       {metrics['specificity'] * 100:.2f}%")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy'] * 100:.2f}%")
    print(f"ROC-AUC:           {metrics.get('roc_auc', 'N/A')}")
    print(f"PR-AUC:            {metrics.get('pr_auc', 'N/A')}")
    print("=" * 60)

    print(f"Saving model to {args.output}...")
    save_pipeline(fitted_pipeline, args.output)
    print("Model serialized successfully.")

    if args.report_output:
        os.makedirs(os.path.dirname(os.path.abspath(args.report_output)), exist_ok=True)
        report = format_metrics_report(metrics, title="Scikit-Learn Logistic Regression v2.0.0 Benchmark")
        with open(args.report_output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report saved to {args.report_output}")


if __name__ == "__main__":
    main()
