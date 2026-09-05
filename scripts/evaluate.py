#!/usr/bin/env python3
"""
CLI Script: Comprehensive Research Metrics Evaluation and Latency Benchmarking.
Usage:
    python scripts/evaluate.py --model models/sklearn/logreg-80k.joblib
"""
import argparse
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation import evaluate_predictions, benchmark_inference_latency, format_metrics_report
from src.preprocessing import prepare_dataset

BENCHMARK_PROMPTS = [
    "I absolutely loved this product! Exceptional quality and fast shipping.",
    "Worst purchase of my life, completely broke after one day. Terrible.",
    "The customer service was stellar, resolved my issue within minutes.",
    "Extremely disappointed with this experience, total waste of money.",
    "Works surprisingly well for the price. Would buy again.",
    "Awful build quality and misleading description. Do not buy.",
    "Five stars! Exceeded all my expectations.",
    "Hate it. Zero stars if I could.",
    "Very happy with this order, highly recommended to everyone!",
    "Regret buying this, full of bugs and crashes constantly."
]


def main():
    parser = argparse.ArgumentParser(description="Evaluate sentiment classifier research metrics and latency")
    parser.add_argument("--model", type=str, default="models/sklearn/logreg-80k.joblib", help="Model path")
    parser.add_argument("--data-path", type=str, default=None, help="Optional raw CSV data path to evaluate on")
    parser.add_argument("--sample", type=int, default=10000, help="Sample size if evaluating on dataset")
    parser.add_argument("--output-report", type=str, default="reports/evaluation_metrics.md", help="Report path")
    parser.add_argument("--output-json", type=str, default="reports/evaluation_metrics.json", help="JSON path")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model file not found at '{args.model}'")
        sys.exit(1)

    print(f"Loading pipeline from {args.model}...")
    pipeline = joblib.load(args.model)

    print("\n--- Latency and Throughput Benchmarking ---")
    latency_bench = benchmark_inference_latency(pipeline.predict, BENCHMARK_PROMPTS, warmup_runs=10, benchmark_runs=50)
    print(f"P50 Item Latency: {latency_bench['p50_item_ms']} ms")
    print(f"P95 Item Latency: {latency_bench['p95_item_ms']} ms")
    print(f"P99 Item Latency: {latency_bench['p99_item_ms']} ms")
    print(f"Throughput:       {latency_bench['throughput_inferences_per_sec']} inferences/second")

    # If dataset is provided, evaluate metrics
    data_path = args.data_path
    if not data_path:
        default_csv = "data/raw/Juggernaut Sentiment Analysis - by kaggle user Adeoluwa Adeboye.csv"
        if os.path.exists(default_csv):
            data_path = default_csv

    metrics = None
    if data_path and os.path.exists(data_path):
        print(f"\nEvaluating on dataset from {data_path} (sampling {args.sample:,} rows)...")
        df = pd.read_csv(data_path, on_bad_lines="skip")
        if args.sample and args.sample < len(df):
            df = df.sample(n=args.sample, random_state=42)

        _, X_test, _, y_test = prepare_dataset(df, test_size=0.3, random_state=98)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None
        metrics = evaluate_predictions(y_test, y_pred, y_prob)
        metrics["latency_benchmarks"] = latency_bench

        print("\n" + "=" * 60)
        print("                  RESEARCH METRICS SUMMARY")
        print("=" * 60)
        print(f"Accuracy:          {metrics['accuracy'] * 100:.2f}%")
        print(f"F1-Score:          {metrics['f1_score'] * 100:.2f}%")
        print(f"Precision:         {metrics['precision'] * 100:.2f}%")
        print(f"Recall:            {metrics['recall_sensitivity'] * 100:.2f}%")
        print(f"Specificity:       {metrics['specificity'] * 100:.2f}%")
        print(f"ROC-AUC:           {metrics.get('roc_auc', 'N/A')}")
        print(f"PR-AUC:            {metrics.get('pr_auc', 'N/A')}")
        print("=" * 60)
    else:
        print("\nNo dataset evaluated; demonstrating benchmark inferences on sample prompts:")
        preds = pipeline.predict(BENCHMARK_PROMPTS)
        probs = pipeline.predict_proba(BENCHMARK_PROMPTS) if hasattr(pipeline, "predict_proba") else None
        for i, text in enumerate(BENCHMARK_PROMPTS):
            label = "POSITIVE" if preds[i] == 1 else "NEGATIVE"
            conf = f"{probs[i][preds[i]]*100:.1f}%" if probs is not None else "N/A"
            print(f"[{label:8s} | Conf: {conf}] {text}")

    if metrics and args.output_report:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_report)), exist_ok=True)
        report = format_metrics_report(metrics, title="Binary Sentiment Classifier v2.0.0 Evaluation")
        report += f"\n### Latency & Throughput Benchmarks\n"
        report += f"- **P50 Latency**: {latency_bench['p50_item_ms']} ms/item\n"
        report += f"- **P95 Latency**: {latency_bench['p95_item_ms']} ms/item\n"
        report += f"- **P99 Latency**: {latency_bench['p99_item_ms']} ms/item\n"
        report += f"- **Throughput**: {latency_bench['throughput_inferences_per_sec']} inferences/second\n"
        with open(args.output_report, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nMarkdown report saved to {args.output_report}")

    if metrics and args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"JSON metrics saved to {args.output_json}")


if __name__ == "__main__":
    main()
