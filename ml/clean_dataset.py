#!/usr/bin/env python3
"""
Clean data/processed/all_courses_merged.csv and output:
  - data/processed/all_courses_cleaned.csv (cleaned dataset)
  - ml/outputs/clean_report.json (cleaning report)

Operations:
  1. Load merged dataset; normalize column names
  2. Ensure unified schema
  3. Type coercions (numeric, boolean)
  4. Difficulty & subject normalization
  5. Text field standardization
  6. Deduplication
  7. Outlier handling & capping
  8. Class balance insight
  9. Generate report & save outputs

Usage:
  python ml/clean_dataset.py                 # Clean and save
  python ml/clean_dataset.py --preview 10   # Show 10 random rows after cleaning
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


SCRIPT_VERSION = "1.0"

# Subject inference keywords
SUBJECT_KEYWORDS = {
    "Data Science": ["data science", "data analysis", "analytics", "sql", "tableau"],
    "Machine Learning": ["machine learning", "ml", "deep learning", "neural", "nlp", "tensorflow", "keras"],
    "Programming": ["python", "java", "c++", "c#", "golang", "ruby", "php", "coding", "programming"],
    "Web Development": ["web", "react", "node", "html", "css", "django", "flask", "frontend", "backend"],
    "Cloud/DevOps": ["cloud", "aws", "azure", "gcp", "devops", "docker", "kubernetes"],
    "AI/Deep Learning": ["ai", "artificial intelligence", "deep learning", "neural network", "gpt"],
    "Database": ["database", "nosql", "mongodb", "postgres", "mysql"],
    "Business": ["business", "management", "finance", "marketing", "leadership"],
}


def normalize_column_name(col: str) -> str:
    """Normalize column name: lowercase, strip, replace non-alnum with _, collapse _."""
    normalized = col.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    normalized = normalized.strip("_")
    return normalized


def find_column(df: pd.DataFrame, name: str) -> Optional[str]:
    """Find column by exact name or normalized name."""
    if name in df.columns:
        return name
    norm_target = normalize_column_name(name)
    for col in df.columns:
        if normalize_column_name(col) == norm_target:
            return col
    return None


def coerce_boolean(val) -> Optional[bool]:
    """Safely coerce value to boolean."""
    if pd.isna(val):
        return None
    if isinstance(val, bool):
        return val
    val_str = str(val).strip().lower()
    if val_str in ("true", "yes", "1", "t", "y"):
        return True
    elif val_str in ("false", "no", "0", "f", "n"):
        return False
    # Fallback: try numeric
    try:
        return bool(int(float(val_str)))
    except Exception:
        return None


def normalize_difficulty(val) -> str:
    """Normalize difficulty to {Beginner, Intermediate, Advanced, Unknown}."""
    if pd.isna(val):
        return "Unknown"
    val_str = str(val).strip().lower()
    if "beginner" in val_str or "intro" in val_str:
        return "Beginner"
    elif "intermediate" in val_str or "inter" in val_str:
        return "Intermediate"
    elif "advanced" in val_str or "expert" in val_str or "graduate" in val_str:
        return "Advanced"
    elif val_str in ("all levels", ""):
        return "Unknown"
    return "Unknown"


def infer_subject(row: pd.Series) -> str:
    """Infer subject from title + skills + description keywords."""
    text = ""
    for col in ["title", "skills", "description"]:
        if col in row and pd.notna(row[col]):
            text += " " + str(row[col]).lower()
        else:
            text += " "
    
    text = text.lower()
    
    for subject, keywords in SUBJECT_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return subject
    
    return "General"


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Safely coerce series to numeric."""
    return pd.to_numeric(series, errors="coerce")


def load_and_validate(path: Path) -> tuple[pd.DataFrame, int]:
    """Load CSV and validate. Return (df, rows_before)."""
    try:
        df = pd.read_csv(path)
        print(f"✓ Loaded {path}: {len(df)} rows")
        return df, len(df)
    except FileNotFoundError:
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR reading {path}: {e}", file=sys.stderr)
        sys.exit(2)


def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all schema columns exist."""
    schema = [
        "source", "title", "university", "difficulty", "subject", "price", "is_paid",
        "rating", "skills", "description", "url", "popularity", "num_lectures",
        "start_date", "end_date", "published_date"
    ]
    
    for col in schema:
        if col not in df.columns:
            df[col] = None
    
    return df


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce types safely."""
    # Numeric fields
    for col in ["price", "rating", "popularity", "num_lectures"]:
        if col in df.columns:
            df[col] = coerce_numeric(df[col])
    
    # Boolean field
    if "is_paid" in df.columns:
        df["is_paid"] = df.apply(lambda row: coerce_boolean(row.get("is_paid")), axis=1)
    
    return df


def normalize_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize difficulty, subject, and text fields."""
    # Difficulty
    if "difficulty" in df.columns:
        df["difficulty"] = df["difficulty"].apply(normalize_difficulty)
    
    # Subject: normalize existing, infer missing
    if "subject" in df.columns:
        df["subject"] = df["subject"].fillna("").str.lower().str.strip()
        missing_mask = (df["subject"] == "")
        if missing_mask.any():
            df.loc[missing_mask, "subject"] = df[missing_mask].apply(infer_subject, axis=1)
    
    # Text fields: fill missing with ""
    for col in ["title", "skills", "description"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()
    
    # Create convenience text column
    text_parts = []
    for col in ["title", "skills", "description"]:
        if col in df.columns:
            text_parts.append(df[col].astype(str))
        else:
            text_parts.append("")
    df["text"] = (text_parts[0] + " " + text_parts[1] + " " + text_parts[2]).str.strip()
    
    return df


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove duplicates and invalid rows."""
    rows_before = len(df)
    
    # Drop exact duplicates
    df = df.drop_duplicates()
    
    # Drop near-duplicates on [source, title]
    df = df.drop_duplicates(subset=["source", "title"], keep="first")
    
    # Drop rows with empty title
    if "title" in df.columns:
        df = df[df["title"].str.strip() != ""]
    
    rows_after = len(df)
    dupes_dropped = rows_before - rows_after
    
    return df, dupes_dropped


def handle_outliers_caps(df: pd.DataFrame) -> pd.DataFrame:
    """Handle outliers and apply caps."""
    # Price: remove negatives, cap at 1M
    if "price" in df.columns:
        df.loc[df["price"] < 0, "price"] = np.nan
        df["price"] = df["price"].clip(upper=1_000_000)
    
    # Rating: compute capped version
    if "rating" in df.columns:
        df["rating_capped_5"] = df["rating"].clip(upper=5)
    
    # Popularity: normalize to [0, 1]
    if "popularity" in df.columns:
        popularity_min = df["popularity"].min()
        popularity_max = df["popularity"].max()
        if pd.notna(popularity_min) and pd.notna(popularity_max) and popularity_min != popularity_max:
            df["popularity_norm"] = (df["popularity"] - popularity_min) / (popularity_max - popularity_min)
        else:
            df["popularity_norm"] = 0.0
    else:
        df["popularity_norm"] = 0.0
    
    return df


def compute_class_balance(df: pd.DataFrame) -> dict:
    """Compute class balance insights."""
    report = {}
    
    # Difficulty distribution
    if "difficulty" in df.columns:
        report["difficulty_dist"] = df["difficulty"].value_counts(dropna=False).to_dict()
    
    # Subject top 20
    if "subject" in df.columns:
        subject_counts = df["subject"].value_counts(dropna=False).head(20).to_dict()
        report["subject_top20"] = subject_counts
    
    return report


def compute_numeric_summary(df: pd.DataFrame) -> dict:
    """Compute numeric summary for key columns."""
    summary = {}
    
    for col in ["price", "rating", "popularity", "num_lectures"]:
        if col in df.columns:
            summary[col] = {
                "count": int(df[col].notna().sum()),
                "mean": float(df[col].mean()) if df[col].notna().sum() > 0 else None,
                "median": float(df[col].median()) if df[col].notna().sum() > 0 else None,
                "min": float(df[col].min()) if df[col].notna().sum() > 0 else None,
                "max": float(df[col].max()) if df[col].notna().sum() > 0 else None,
            }
    
    return summary


def generate_report(
    rows_before: int,
    rows_after: int,
    dupes_dropped: int,
    df: pd.DataFrame,
    class_balance: dict,
    numeric_summary: dict,
) -> dict:
    """Generate comprehensive cleaning report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "script_version": SCRIPT_VERSION,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "dupes_dropped": dupes_dropped,
        "non_null_counts": {
            "source": int(df["source"].notna().sum()),
            "title": int(df["title"].notna().sum()),
            "university": int(df["university"].notna().sum()),
            "difficulty": int(df["difficulty"].notna().sum()),
            "subject": int(df["subject"].notna().sum()),
            "price": int(df["price"].notna().sum()),
            "is_paid": int(df["is_paid"].notna().sum()),
            "rating": int(df["rating"].notna().sum()),
            "skills": int(df["skills"].notna().sum()),
            "description": int(df["description"].notna().sum()),
            "url": int(df["url"].notna().sum()),
            "popularity": int(df["popularity"].notna().sum()),
            "num_lectures": int(df["num_lectures"].notna().sum()),
        },
        "class_balance": class_balance,
        "numeric_summary": numeric_summary,
    }
    
    return report


def save_report(report: dict, path: Path) -> None:
    """Save report to JSON."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"✓ Report saved to {path}")
    except Exception as e:
        print(f"WARNING: Could not save report: {e}", file=sys.stderr)


def save_cleaned_csv(df: pd.DataFrame, path: Path) -> None:
    """Save cleaned CSV."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Reorder columns logically
        final_cols = [
            "source", "title", "university", "difficulty", "subject", "price", "is_paid",
            "rating", "rating_capped_5", "popularity", "popularity_norm", "skills",
            "description", "url", "num_lectures", "start_date", "end_date",
            "published_date", "text"
        ]
        
        available_cols = [col for col in final_cols if col in df.columns]
        df_final = df[available_cols]
        
        df_final.to_csv(path, index=False, encoding="utf-8")
        print(f"✓ Cleaned dataset saved to {path}")
    except Exception as e:
        print(f"ERROR saving cleaned CSV: {e}", file=sys.stderr)
        sys.exit(3)


def print_summary(df: pd.DataFrame, rows_before: int, dupes_dropped: int) -> None:
    """Print concise summary."""
    print(f"\n{'='*60}")
    print(f"{'CLEANING SUMMARY':<30}")
    print(f"{'='*60}")
    print(f"Rows before:    {rows_before}")
    print(f"Rows after:     {len(df)}")
    print(f"Dupes dropped:  {dupes_dropped}")
    print(f"\nNon-null counts:")
    for col in ["subject", "difficulty", "is_paid"]:
        if col in df.columns:
            count = df[col].notna().sum()
            print(f"  {col:15s}: {count:6d}")
    print(f"\nSource distribution:")
    if "source" in df.columns:
        for source, count in df["source"].value_counts().items():
            print(f"  {source:15s}: {count:6d}")
    print(f"{'='*60}\n")


def preview_data(df: pd.DataFrame, num_rows: int = 10) -> None:
    """Print preview of random rows."""
    if len(df) < num_rows:
        num_rows = len(df)
    
    sample = df.sample(n=num_rows, random_state=42)
    print(f"\n📄 Preview ({num_rows} random rows):\n")
    
    display_cols = ["source", "title", "difficulty", "subject", "price", "rating"]
    available_cols = [col for col in display_cols if col in df.columns]
    
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", 40)
    print(sample[available_cols].to_string())
    pd.reset_option("display.max_columns")
    pd.reset_option("display.width")
    pd.reset_option("display.max_colwidth")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Clean merged course dataset"
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        help="Preview N random rows after cleaning (0 = no preview)",
    )
    
    args = parser.parse_args()
    
    print("🧹 Cleaning course dataset...\n")
    
    # Load
    input_path = Path("data") / "processed" / "all_courses_merged.csv"
    df, rows_before = load_and_validate(input_path)
    
    # Ensure schema
    df = ensure_schema(df)
    
    # Coerce types
    df = coerce_types(df)
    
    # Normalize fields
    df = normalize_fields(df)
    
    # Deduplicate
    df, dupes_dropped = deduplicate(df)
    
    # Handle outliers
    df = handle_outliers_caps(df)
    
    # Compute insights
    class_balance = compute_class_balance(df)
    numeric_summary = compute_numeric_summary(df)
    
    # Generate report
    report = generate_report(rows_before, len(df), dupes_dropped, df, class_balance, numeric_summary)
    
    # Save outputs
    output_csv = Path("data") / "processed" / "all_courses_cleaned.csv"
    output_report = Path("ml") / "outputs" / "clean_report.json"
    
    save_cleaned_csv(df, output_csv)
    save_report(report, output_report)
    
    # Summary
    print_summary(df, rows_before, dupes_dropped)
    
    # Optional preview
    if args.preview > 0:
        preview_data(df, args.preview)
    
    print("✓ Cleaning complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
