#!/usr/bin/env python3
"""
Train a text + tabular classifier using hashed text features and tabular data.

Usage:
    python ml/train.py --csv_path data/processed/all_courses_merged.csv --out_dir ml/outputs --target_class difficulty --hash_features 2000

    or (with auto-discovery):

    python ml/train.py --raw_dir data/processed --out_dir ml/outputs --merge_keys source --target_class difficulty --hash_features 2000

Features:
- Detects numeric, categorical, and text columns.
- Text columns (title, skills): HashingVectorizer with n-grams (1,2).
- Numeric: median imputation.
- Categorical: most_frequent imputation + OneHotEncoder (max_categories=50).
- Model: LinearSVC or SGDClassifier (fallback).
- Metrics: accuracy, macro-F1, classification_report.
- Saves: model_classif.joblib, report_classif.json.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.svm import LinearSVC


def normalize_column_name(col: str) -> str:
    """Normalize column name: lowercase, strip, non-alnum → _, collapse _, trim."""
    normalized = col.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    normalized = normalized.strip("_")
    return normalized


def find_column(df: pd.DataFrame, name: str) -> str:
    """Find a column in df by name or normalized name; exit if not found."""
    if name in df.columns:
        return name
    norm = normalize_column_name(name)
    for c in df.columns:
        if normalize_column_name(c) == norm:
            return c
    print(f"ERROR: Column '{name}' not found. Available: {list(df.columns)}", file=sys.stderr)
    sys.exit(9)


def discover_csvs(raw_dir: str) -> List[Path]:
    """Recursively find all CSV files under raw_dir."""
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        print(f"ERROR: raw_dir does not exist: {raw_dir}", file=sys.stderr)
        sys.exit(1)
    csvs = list(raw_path.rglob("*.csv"))
    if not csvs:
        print(f"ERROR: No CSV files found under {raw_dir}", file=sys.stderr)
        sys.exit(2)
    return csvs


def load_csvs(csv_paths: List[Path]) -> Tuple[List[pd.DataFrame], List[str]]:
    """Load CSVs and return list of DataFrames and their filenames."""
    dfs = []
    names = []
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
            dfs.append(df)
            names.append(p.name)
            print(f"Loaded {p.name}: {df.shape[0]} rows, {df.shape[1]} cols")
        except Exception as e:
            print(f"WARNING: Could not read {p.name}: {e}", file=sys.stderr)
    if not dfs:
        print("ERROR: No CSVs could be loaded.", file=sys.stderr)
        sys.exit(3)
    return dfs, names


def print_common_columns(dfs: List[pd.DataFrame], names: List[str]) -> None:
    """Print common columns across all DataFrames."""
    if not dfs:
        return
    common = set(dfs[0].columns)
    for df in dfs[1:]:
        common &= set(df.columns)
    print(f"\nCommon columns across all {len(dfs)} CSVs: {sorted(common)}")
    if not common:
        print("WARNING: No common columns found across all CSVs. Merging may fail.\n")
    else:
        print(f"Suggested merge keys: {', '.join(sorted(common))}\n")


def merge_dataframes(
    dfs: List[pd.DataFrame], merge_keys: List[str], join_type: str
) -> pd.DataFrame:
    """Merge DataFrames pairwise on merge_keys."""
    if not merge_keys:
        print("ERROR: --merge_keys is required for merging multiple CSVs.", file=sys.stderr)
        sys.exit(4)
    
    for i, df in enumerate(dfs):
        missing = [k for k in merge_keys if k not in df.columns]
        if missing:
            print(
                f"ERROR: Merge keys {missing} not found in CSV #{i+1}. "
                f"Available columns: {list(df.columns)}",
                file=sys.stderr,
            )
            sys.exit(5)
    
    result = dfs[0]
    for i, df in enumerate(dfs[1:], start=1):
        print(f"Merging CSV #{i+1} on {merge_keys} using {join_type} join...")
        result = result.merge(df, on=merge_keys, how=join_type, suffixes=("", f"_dup{i}"))
    
    print(f"Merged DataFrame shape: {result.shape}")
    return result


def find_target_column(df: pd.DataFrame, target: str) -> str:
    """Find target column using find_column (for backwards compatibility)."""
    return find_column(df, target)


def identify_column_types(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str], List[str]]:
    """Identify numeric, categorical, and text columns (excluding target)."""
    X = df.drop(columns=[target_col])
    
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    text_cols = [c for c in ["title", "skills"] if c in X.columns]
    
    categorical_cols = [
        c for c in X.select_dtypes(include=["object", "category", "bool"]).columns
        if c not in text_cols
    ]
    
    print(f"Column types identified:")
    print(f"  Numeric: {numeric_cols if numeric_cols else '(none)'}")
    print(f"  Text: {text_cols if text_cols else '(none)'}")
    print(f"  Categorical: {categorical_cols if categorical_cols else '(none)'}\n")
    
    return numeric_cols, categorical_cols, text_cols


def extract_text_column(X) -> pd.Series:
    """
    Accepts either:
      - pandas DataFrame/Series (single column), or
      - numpy ndarray with shape (n_samples, 1) or (n_samples,)
    Returns a 1D pandas Series of strings (no NaNs).
    """
    if isinstance(X, pd.Series):
        s = X
    elif isinstance(X, pd.DataFrame):
        # take the first column regardless of its name/index
        s = X.iloc[:, 0]
    elif isinstance(X, np.ndarray):
        if X.ndim == 2 and X.shape[1] == 1:
            s = pd.Series(X[:, 0])
        elif X.ndim == 1:
            s = pd.Series(X)
        else:
            raise ValueError(f"Expected 1D or 2D (n,1) array, got shape={X.shape}")
    else:
        # last resort: try to build a Series
        s = pd.Series(X)
    return s.fillna("").astype(str)


def build_preprocessing_pipeline(
    numeric_cols: List[str],
    categorical_cols: List[str],
    text_cols: List[str],
    hash_features: int,
    X: pd.DataFrame | None = None,
) -> ColumnTransformer:
    """Build ColumnTransformer with numeric, categorical, and text branches."""
    transformers = []
    
    # Numeric branch
    if numeric_cols:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])
        transformers.append(("num", numeric_pipeline, numeric_cols))
    
    # Categorical branch
    if categorical_cols:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=50, sparse_output=False)),
        ])
        transformers.append(("cat", categorical_pipeline, categorical_cols))
    
    # Text branches (one per text column)
    for text_col in text_cols:
        # Defensive guard: skip if column not present
        if text_col not in X.columns:
            print(f"Skipping text column '{text_col}' (not present)")
            continue
        
        text_pipeline = Pipeline([
            ("extract", FunctionTransformer(extract_text_column, validate=False)),
            ("hash", HashingVectorizer(
                n_features=hash_features,
                alternate_sign=False,
                ngram_range=(1, 2),
                norm=None,
                lowercase=True,
            )),
        ])
        transformers.append((f"text_{text_col}", text_pipeline, [text_col]))
    
    if not transformers:
        print("ERROR: No columns for preprocessing.", file=sys.stderr)
        sys.exit(7)
    
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor


def prepare_data(
    df: pd.DataFrame, target_col: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate features and target, drop duplicates and missing targets."""
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    if before != after:
        print(f"Dropped {before - after} duplicate rows. Remaining: {after}")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    valid_mask = y.notna()
    if not valid_mask.all():
        print(f"Dropping {(~valid_mask).sum()} rows with missing target values.")
        X = X[valid_mask]
        y = y[valid_mask]
    
    if X.shape[0] == 0:
        print("ERROR: No valid rows remaining after dropping missing targets.", file=sys.stderr)
        sys.exit(8)
    
    return X, y


def train_classifier(
    X_train, X_test, y_train, y_test, preprocessor, class_weight: str = "none", svm_max_iter: int = 5000
) -> Tuple[Pipeline, dict]:
    """Train LinearSVC or fallback to SGDClassifier."""
    try:
        if class_weight == "balanced":
            classifier = LinearSVC(class_weight="balanced", max_iter=svm_max_iter, random_state=42)
        else:
            classifier = LinearSVC(max_iter=svm_max_iter, random_state=42)
        print("Using LinearSVC classifier...")
    except Exception:
        print("LinearSVC failed; falling back to SGDClassifier...")
        classifier = SGDClassifier(loss="hinge", max_iter=svm_max_iter, random_state=42)
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])
    
    print("Training classifier...")
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    # Compute confusion matrix with explicit labels
    labels = np.unique(y_test)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    # Normalize confusion matrix (row-wise, with safety for division)
    with np.errstate(all='ignore'):
        cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    
    # Compute per-class accuracy (recall) from confusion matrix
    per_class_accuracy = {}
    for i, label in enumerate(labels):
        if cm[i].sum() > 0:
            per_class_accuracy[str(label)] = float(cm[i, i] / cm[i].sum())
        else:
            per_class_accuracy[str(label)] = 0.0
    
    # Compute support per class
    support_per_class = {}
    for label in labels:
        support_per_class[str(label)] = int((y_test == label).sum())
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Concise summary
    print(f"\n=== Summary ===")
    print(f"accuracy={accuracy:.4f}, macro_f1={macro_f1:.4f}, sizes: train={X_train.shape[0]}, test={X_test.shape[0]}")
    
    report = {
        "task": "classification_hashed_text",
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "classification_report": report_dict,
        "class_labels": labels.tolist(),
        "support_per_class": support_per_class,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix_shape": list(cm.shape),
    }
    
    return pipeline, report, cm, cm_normalized, labels.tolist()


def save_outputs(pipeline, report: dict, out_dir: str, cm, cm_normalized, class_labels) -> None:
    """Save model, report, and confusion matrices."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    model_path = out_path / "model_classif.joblib"
    report_path = out_path / "report_classif.json"
    cm_path = out_path / "confusion_matrix.csv"
    cm_norm_path = out_path / "confusion_matrix_normalized.csv"
    
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved to: {model_path}")
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to: {report_path}")
    
    # Save raw confusion matrix
    cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
    cm_df.to_csv(cm_path)
    print(f"Confusion matrix saved to: {cm_path}")
    
    # Save normalized confusion matrix
    cm_norm_df = pd.DataFrame(cm_normalized, index=class_labels, columns=class_labels)
    cm_norm_df.to_csv(cm_norm_path)
    print(f"Normalized confusion matrix saved to: {cm_norm_path}")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train text + tabular classifier with hashed text features.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv_path",
        default=None,
        help="(Optional) Path to pre-merged CSV. If omitted and all_courses_merged.csv exists, it will be used.",
    )
    parser.add_argument(
        "--raw_dir",
        default=None,
        help="Directory for CSV discovery (fallback if --csv_path not provided).",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for model_classif.joblib and report_classif.json.",
    )
    parser.add_argument(
        "--target_class",
        required=True,
        help="Target column name for classification (e.g., difficulty).",
    )
    parser.add_argument(
        "--merge_keys",
        default=None,
        help="(Optional) Comma-separated merge keys for raw_dir discovery.",
    )
    parser.add_argument(
        "--join",
        default="inner",
        choices=["inner", "left"],
        help="Join type for merging CSVs (default: inner).",
    )
    parser.add_argument(
        "--hash_features",
        type=int,
        default=2000,
        help="Number of hashing features for text (default: 2000).",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test set size (default: 0.2).",
    )
    parser.add_argument(
        "--filter_nonnull",
        default=None,
        help="(Optional) Column name to filter for non-null values before train/test split.",
    )
    parser.add_argument(
        "--class_weight",
        choices=["none", "balanced"],
        default="none",
        help="Class weight for LinearSVC (default: none).",
    )
    parser.add_argument(
        "--svm_max_iter",
        type=int,
        default=5000,
        help="Max iterations for LinearSVC (default: 5000).",
    )
    
    args = parser.parse_args(argv)
    
    # Startup info
    print(f"hash_features={args.hash_features}, class_weight={args.class_weight}, svm_max_iter={args.svm_max_iter}")
    if args.filter_nonnull:
        print(f"filter_nonnull={args.filter_nonnull}")
    else:
        print("filter_nonnull=(not applied)")
    
    # Auto-detect merged CSV
    merged_csv_path = Path("data") / "processed" / "all_courses_merged.csv"
    if args.csv_path is None and merged_csv_path.exists():
        print(f"Detected merged dataset, using all_courses_merged.csv")
        args.csv_path = str(merged_csv_path)
    
    # Load data
    if args.csv_path:
        csv_file = Path(args.csv_path)
        if not csv_file.exists():
            print(f"ERROR: CSV file not found: {args.csv_path}", file=sys.stderr)
            sys.exit(1)
        try:
            merged = pd.read_csv(csv_file)
            print(f"Loaded CSV: {csv_file} ({merged.shape[0]} rows, {merged.shape[1]} cols)\n")
        except Exception as e:
            print(f"ERROR reading {csv_file}: {e}", file=sys.stderr)
            sys.exit(2)
    else:
        if not args.raw_dir:
            print("ERROR: --raw_dir or --csv_path must be provided.", file=sys.stderr)
            sys.exit(3)
        
        print(f"Discovering CSVs under: {args.raw_dir}")
        csv_paths = discover_csvs(args.raw_dir)
        print(f"Found {len(csv_paths)} CSV file(s).\n")
        
        dfs, names = load_csvs(csv_paths)
        print_common_columns(dfs, names)
        
        if len(dfs) == 1:
            print("Only one CSV found, skipping merge.")
            merged = dfs[0]
        else:
            if not args.merge_keys:
                print("ERROR: --merge_keys required for multiple CSVs.", file=sys.stderr)
                sys.exit(4)
            merge_keys = [k.strip() for k in args.merge_keys.split(",")]
            print(f"Merge keys: {merge_keys}")
            print(f"Join type: {args.join}\n")
            merged = merge_dataframes(dfs, merge_keys, args.join)
    
    # Apply filtering if requested (before prepare_data)
    if args.filter_nonnull:
        filter_col = find_column(merged, args.filter_nonnull)
        before_filter = merged.shape[0]
        merged = merged[merged[filter_col].notna()]
        dropped = before_filter - merged.shape[0]
        print(f"Filtered column '{filter_col}': dropped {dropped} rows, {merged.shape[0]} remaining\n")
    
    # Prepare data
    print(f"Preparing data with target: {args.target_class}")
    target_col = find_target_column(merged, args.target_class)
    X, y = prepare_data(merged, target_col)
    
    # Compute and print class distribution
    if args.target_class in merged.columns or target_col in merged.columns:
        class_counts = y.value_counts().sort_values(ascending=False)
        print(f"Class distribution for '{target_col}':")
        for label, count in class_counts.items():
            print(f"  {label}: {count}")
        
        # Warn if any class has < 3 samples
        if (class_counts < 3).any():
            print(f"\nWARNING: Some classes have fewer than 3 samples. Model may not generalize well.")
        print()
    
    # Identify column types
    numeric_cols, categorical_cols, text_cols = identify_column_types(merged, target_col)
    
    # Split data
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=42, stratify=y
        )
    except Exception:
        # If stratify fails (e.g., too few samples), retry without stratify
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=42
        )
    
    print(f"Train set: {X_train.shape[0]} rows")
    print(f"Test set: {X_test.shape[0]} rows\n")
    
    # Build preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(
        numeric_cols, categorical_cols, text_cols, args.hash_features, X
    )
    
    # Train classifier
    pipeline, report, cm, cm_normalized, class_labels = train_classifier(
        X_train, X_test, y_train, y_test, preprocessor, args.class_weight, args.svm_max_iter
    )
    
    # Save outputs
    save_outputs(pipeline, report, args.out_dir, cm, cm_normalized, class_labels)
    
    print("\n✓ Training complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())


