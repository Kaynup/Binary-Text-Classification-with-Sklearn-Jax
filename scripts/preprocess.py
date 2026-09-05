#!/usr/bin/env python3
"""
CLI Script: Data Preprocessing and TF-IDF Feature Extraction.
Usage:
    python scripts/preprocess.py --sample 100000 --max-features 50000
"""
import argparse
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import prepare_dataset, fit_vectorizer, save_processed_split


def main():
    parser = argparse.ArgumentParser(description="Preprocess and vectorize text data")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/Juggernaut Sentiment Analysis - by kaggle user Adeoluwa Adeboye.csv",
        help="Path to raw dataset CSV"
    )
    parser.add_argument("--output-dir", type=str, default="data/processed/dataset_custom", help="Output directory")
    parser.add_argument("--test-size", type=float, default=0.15, help="Test set fraction")
    parser.add_argument("--max-features", type=int, default=80000, help="Max vocabulary features")
    parser.add_argument("--min-ngram", type=int, default=1, help="Min ngram")
    parser.add_argument("--max-ngram", type=int, default=5, help="Max ngram")
    parser.add_argument("--sample", type=int, default=None, help="Sample size limit for speed")
    parser.add_argument("--random-state", type=int, default=98, help="Random state seed")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Error: Dataset not found at '{args.data_path}'")
        sys.exit(1)

    print(f"Loading raw dataset from {args.data_path}...")
    df = pd.read_csv(args.data_path, on_bad_lines="skip")
    if args.sample and args.sample < len(df):
        print(f"Sampling {args.sample:,} rows...")
        df = df.sample(n=args.sample, random_state=args.random_state)

    print(f"Splitting dataset (test_size={args.test_size}, stratify=True)...")
    X_train, X_test, y_train, y_test = prepare_dataset(
        df,
        test_size=args.test_size,
        random_state=args.random_state
    )
    print(f"Train samples: {len(X_train):,}, Test samples: {len(X_test):,}")

    ngram_range = (args.min_ngram, args.max_ngram)
    print(f"Fitting TfidfVectorizer (max_features={args.max_features:,}, ngram_range={ngram_range})...")
    vec, X_train_vec, fit_duration = fit_vectorizer(
        X_train,
        max_features=args.max_features,
        ngram_range=ngram_range
    )
    print(f"Vectorizer fitted in {fit_duration:.2f} seconds.")

    print("Transforming test set...")
    X_test_vec = vec.transform(X_test)

    metadata = {
        "split_params": {
            "test_size": args.test_size,
            "random_state": args.random_state
        },
        "vectorizer_params": {
            "max_features": args.max_features,
            "ngram_range": [args.min_ngram, args.max_ngram]
        },
        "vectorization_time_sec": round(fit_duration, 2),
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "vocab_size": len(vec.vocabulary_)
    }

    print(f"Saving artifacts to {args.output_dir}...")
    save_processed_split(
        args.output_dir,
        X_train_vec,
        X_test_vec,
        y_train.to_numpy(),
        y_test.to_numpy(),
        vec,
        metadata
    )
    print("Preprocessing completed successfully!")


if __name__ == "__main__":
    main()
