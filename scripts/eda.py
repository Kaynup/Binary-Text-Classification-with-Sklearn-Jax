#!/usr/bin/env python3
"""
CLI Script: Exploratory Data Analysis for Sentiment Analysis Dataset.
Usage:
    python scripts/eda.py --data-path data/raw/dataset.csv --sample 50000
"""
import argparse
import os
import sys
import json

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eda import run_eda
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Run EDA on sentiment dataset")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/Juggernaut Sentiment Analysis - by kaggle user Adeoluwa Adeboye.csv",
        help="Path to raw dataset CSV"
    )
    parser.add_argument("--text-col", type=str, default="SentimentText", help="Text column name")
    parser.add_argument("--label-col", type=str, default="Sentiment", help="Label column name")
    parser.add_argument("--sample", type=int, default=None, help="Optional sample limit for quick inspection")
    parser.add_argument("--output", type=str, default="reports/eda_summary.json", help="Path to save EDA JSON")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at '{args.data_path}'")
        sys.exit(1)

    print(f"Loading data from {args.data_path}...")
    df = pd.read_csv(args.data_path, on_bad_lines="skip")
    if args.sample and args.sample < len(df):
        print(f"Sampling {args.sample:,} rows...")
        df = df.sample(n=args.sample, random_state=42)

    print("Running exploratory data analysis...")
    results = run_eda(df, text_col=args.text_col, label_col=args.label_col)

    print("\n" + "=" * 60)
    print("           EXPLORATORY DATA ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total Rows Analyzed:    {results['total_records']:,}")
    print(f"Valid Non-Null Rows:    {results['valid_records']:,}")
    dist = results["class_distribution"]
    print(f"Negative (0):           {dist['negative_0']:,} ({dist['negative_ratio']*100:.1f}%)")
    print(f"Positive (1):           {dist['positive_1']:,} ({dist['positive_ratio']*100:.1f}%)")
    c_dist = results["character_length_distribution"]
    print(f"Char Lengths:           Median={c_dist['median']}, Mean={c_dist['mean']}, Max={c_dist['max']}")
    w_dist = results["word_count_distribution"]
    print(f"Word Counts:            Median={w_dist['median']}, Mean={w_dist['mean']}, Max={w_dist['max']}")
    print("=" * 60)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved EDA report to {args.output}")


if __name__ == "__main__":
    main()
