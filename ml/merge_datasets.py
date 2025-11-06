#!/usr/bin/env python3
"""
Merge course datasets from Coursera, edX, NPTEL, and Udemy into unified schema.

Reads from data/processed/:
  - coursera_cleaned.csv
  - edx_cleaned.csv
  - nptel_cleaned.csv
  - udemy_cleaned.csv

Outputs to:
  - data/processed/all_courses_merged.csv

Unified Schema:
  [source, title, university, difficulty, rating, price, subject, is_paid,
   skills, description, url, popularity, num_lectures, start_date, end_date,
   published_date]

Usage:
  python ml/merge_datasets.py                    # Merge and save
  python ml/merge_datasets.py --preview 10      # Show 10 random rows
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


def normalize_column_name(col: str) -> str:
    """Normalize column name: lowercase, strip, replace non-alnum with _, collapse _."""
    normalized = col.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    normalized = normalized.strip("_")
    return normalized


def find_column(df: pd.DataFrame, name: str) -> Optional[str]:
    """Find column by exact name or normalized name. Returns actual column name or None."""
    if name in df.columns:
        return name
    norm_target = normalize_column_name(name)
    for col in df.columns:
        if normalize_column_name(col) == norm_target:
            return col
    return None


def infer_subject(title: str) -> str:
    """Infer subject from course title keywords."""
    if not title or pd.isna(title):
        return "General"
    
    title_lower = str(title).lower()
    keywords = {
        "Data Science": ["data science", "data analysis", "analytics"],
        "Machine Learning": ["machine learning", "ml", "deep learning", "neural", "nlp"],
        "Programming": ["python", "java", "c++", "javascript", "coding", "programming"],
        "Web Development": ["web", "react", "django", "flask", "frontend", "backend"],
        "DevOps": ["devops", "docker", "kubernetes", "ci/cd", "jenkins"],
        "Cloud": ["aws", "azure", "gcp", "cloud"],
        "Database": ["sql", "nosql", "database", "mongodb", "postgres"],
        "Business": ["business", "management", "marketing", "finance"],
        "General": ["course", "intro", "basics"],
    }
    
    for subject, keywords_list in keywords.items():
        if any(kw in title_lower for kw in keywords_list):
            return subject
    
    return "General"


def load_coursera(path: Path) -> pd.DataFrame:
    """Load and map Coursera dataset."""
    try:
        df = pd.read_csv(path)
        print(f"✓ Loaded Coursera: {len(df)} rows")
    except Exception as e:
        print(f"⚠ Skipping Coursera: {e}")
        return pd.DataFrame()
    
    result = pd.DataFrame()
    result["source"] = "coursera"
    
    col_map = {normalize_column_name(c): c for c in df.columns}
    
    result["title"] = df[col_map["course_name"]] if "course_name" in col_map else None
    result["university"] = df[col_map["university"]] if "university" in col_map else None
    result["difficulty"] = df[col_map["difficulty_level"]] if "difficulty_level" in col_map else "Unknown"
    result["rating"] = pd.to_numeric(df[col_map["course_rating"]] if "course_rating" in col_map else None, errors="coerce")
    result["url"] = df[col_map["course_url"]] if "course_url" in col_map else None
    result["description"] = df[col_map["course_description"]] if "course_description" in col_map else ""
    result["skills"] = df[col_map["skills"]] if "skills" in col_map else ""
    
    result["price"] = None
    result["subject"] = result["title"].apply(infer_subject)
    result["is_paid"] = None
    result["popularity"] = None
    result["num_lectures"] = None
    result["start_date"] = None
    result["end_date"] = None
    result["published_date"] = None
    
    return result


def load_edx(path: Path) -> pd.DataFrame:
    """Load and map edX dataset."""
    try:
        df = pd.read_csv(path)
        print(f"✓ Loaded edX: {len(df)} rows")
    except Exception as e:
        print(f"⚠ Skipping edX: {e}")
        return pd.DataFrame()
    
    result = pd.DataFrame()
    result["source"] = "edx"
    
    col_map = {normalize_column_name(c): c for c in df.columns}
    
    result["title"] = df[col_map["name"]] if "name" in col_map else None
    result["university"] = df[col_map["university"]] if "university" in col_map else None
    result["difficulty"] = df[col_map["difficulty_level"]] if "difficulty_level" in col_map else "Unknown"
    result["rating"] = None
    result["url"] = df[col_map["link"]] if "link" in col_map else None
    result["description"] = df[col_map["about"]] if "about" in col_map else ""
    
    # Skills: use description if not present
    if "skills" in col_map:
        result["skills"] = df[col_map["skills"]]
    elif "course_description" in col_map:
        result["skills"] = df[col_map["course_description"]]
    else:
        result["skills"] = result["description"]
    
    result["price"] = None
    result["subject"] = result["title"].apply(infer_subject)
    result["is_paid"] = None
    result["popularity"] = None
    result["num_lectures"] = None
    result["start_date"] = None
    result["end_date"] = None
    result["published_date"] = None
    
    return result


def load_nptel(path: Path) -> pd.DataFrame:
    """Load and map NPTEL dataset."""
    try:
        df = pd.read_csv(path)
        print(f"✓ Loaded NPTEL: {len(df)} rows")
    except Exception as e:
        print(f"⚠ Skipping NPTEL: {e}")
        return pd.DataFrame()
    
    result = pd.DataFrame()
    result["source"] = "nptel"
    
    col_map = {normalize_column_name(c): c for c in df.columns}
    
    result["title"] = df[col_map["course_name"]] if "course_name" in col_map else None
    result["university"] = df[col_map["instructor"]] if "instructor" in col_map else None
    result["difficulty"] = "Unknown"
    result["rating"] = None
    result["url"] = None
    result["description"] = df[col_map["abstract"]] if "abstract" in col_map else ""
    
    # Skills: use abstract if not present
    if "skills" in col_map:
        result["skills"] = df[col_map["skills"]]
    else:
        result["skills"] = result["description"]
    
    result["price"] = None
    result["subject"] = result["title"].apply(infer_subject)
    result["is_paid"] = None
    result["popularity"] = None
    result["num_lectures"] = None
    
    result["start_date"] = df[col_map["course_timeline_course_duration_start_date"]] \
        if "course_timeline_course_duration_start_date" in col_map else None
    result["end_date"] = df[col_map["course_timeline_course_duration_end_date"]] \
        if "course_timeline_course_duration_end_date" in col_map else None
    result["published_date"] = None
    
    return result


def load_udemy(path: Path) -> pd.DataFrame:
    """Load and map Udemy dataset."""
    try:
        df = pd.read_csv(path)
        print(f"✓ Loaded Udemy: {len(df)} rows")
    except Exception as e:
        print(f"⚠ Skipping Udemy: {e}")
        return pd.DataFrame()
    
    result = pd.DataFrame()
    result["source"] = "udemy"
    
    col_map = {normalize_column_name(c): c for c in df.columns}
    
    result["title"] = df[col_map["course_title"]] if "course_title" in col_map else None
    result["university"] = None
    result["difficulty"] = df[col_map["level"]] if "level" in col_map else "Unknown"
    result["rating"] = pd.to_numeric(df[col_map["num_reviews"]] if "num_reviews" in col_map else None, errors="coerce")
    result["price"] = pd.to_numeric(df[col_map["price"]] if "price" in col_map else None, errors="coerce")
    result["subject"] = df[col_map["subject"]] if "subject" in col_map else "General"
    
    # Safely convert is_paid to boolean
    if "is_paid" in col_map:
        try:
            result["is_paid"] = df[col_map["is_paid"]].astype(bool)
        except Exception:
            result["is_paid"] = None
    else:
        result["is_paid"] = None
    
    result["skills"] = None
    result["description"] = None
    result["url"] = df[col_map["url"]] if "url" in col_map else None
    result["popularity"] = pd.to_numeric(df[col_map["num_subscribers"]] if "num_subscribers" in col_map else None, errors="coerce")
    result["num_lectures"] = pd.to_numeric(df[col_map["num_lectures"]] if "num_lectures" in col_map else None, errors="coerce")
    result["start_date"] = None
    result["end_date"] = None
    result["published_date"] = df[col_map["published_timestamp"]] if "published_timestamp" in col_map else None
    
    return result


def merge_all() -> pd.DataFrame:
    """Merge all datasets into unified schema."""
    data_dir = Path("data") / "processed"
    
    print("Loading datasets from data/processed/...\n")
    
    coursera_path = data_dir / "coursera_cleaned.csv"
    edx_path = data_dir / "edx_cleaned.csv"
    nptel_path = data_dir / "nptel_cleaned.csv"
    udemy_path = data_dir / "udemy_cleaned.csv"
    
    dfs = []
    
    if coursera_path.exists():
        dfs.append(load_coursera(coursera_path))
    if edx_path.exists():
        dfs.append(load_edx(edx_path))
    if nptel_path.exists():
        dfs.append(load_nptel(nptel_path))
    if udemy_path.exists():
        dfs.append(load_udemy(udemy_path))
    
    if not dfs:
        print("ERROR: No datasets found in data/processed/", file=sys.stderr)
        return pd.DataFrame()
    
    # Concatenate all
    print("\nMerging datasets...")
    merged = pd.concat(dfs, ignore_index=True)
    
    # Ensure unified schema order
    schema = ["source", "title", "university", "difficulty", "rating", "price", "subject",
              "is_paid", "skills", "description", "url", "popularity", "num_lectures",
              "start_date", "end_date", "published_date"]
    
    for col in schema:
        if col not in merged.columns:
            merged[col] = None
    
    merged = merged[schema]
    
    # Fill defaults
    merged["difficulty"] = merged["difficulty"].fillna("Unknown")
    merged["skills"] = merged["skills"].fillna("")
    merged["description"] = merged["description"].fillna("")
    
    # Coerce numerics
    merged["rating"] = pd.to_numeric(merged["rating"], errors="coerce")
    merged["price"] = pd.to_numeric(merged["price"], errors="coerce")
    merged["popularity"] = pd.to_numeric(merged["popularity"], errors="coerce")
    merged["num_lectures"] = pd.to_numeric(merged["num_lectures"], errors="coerce")
    
    return merged


def print_summary(df: pd.DataFrame) -> None:
    """Print merge summary with non-null counts."""
    print(f"\n✅ Merged dataset created: {len(df)} rows × {len(df.columns)} columns\n")
    
    print("Non-null counts:")
    key_cols = ["subject", "difficulty", "rating", "price", "is_paid"]
    for col in key_cols:
        if col in df.columns:
            count = df[col].notna().sum()
            print(f"  {col}={count}")
    
    print("\nSource distribution:")
    for source, count in df["source"].value_counts().items():
        print(f"  {source}: {count}")


def preview_data(df: pd.DataFrame, num_rows: int = 10) -> None:
    """Print preview of random rows."""
    if len(df) < num_rows:
        num_rows = len(df)
    
    sample = df.sample(n=num_rows, random_state=42)
    print(f"\n📄 Preview ({num_rows} random rows):\n")
    
    # Display with selected columns for readability
    display_cols = ["source", "title", "difficulty", "rating", "price", "subject"]
    available_cols = [col for col in display_cols if col in df.columns]
    
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", 50)
    print(sample[available_cols].to_string())
    pd.reset_option("display.max_columns")
    pd.reset_option("display.width")
    pd.reset_option("display.max_colwidth")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Merge course datasets from Coursera, edX, NPTEL, and Udemy"
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        help="Print preview of N random rows (0 = no preview)",
    )
    
    args = parser.parse_args()
    
    # Merge
    merged = merge_all()
    
    if merged.empty:
        print("ERROR: Merge resulted in empty dataset", file=sys.stderr)
        return 1
    
    # Save
    output_path = Path("data") / "processed" / "all_courses_merged.csv"
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Saved to: {output_path}\n")
    except Exception as e:
        print(f"ERROR writing {output_path}: {e}", file=sys.stderr)
        return 2
    
    # Summary
    print_summary(merged)
    
    # Optional preview
    if args.preview > 0:
        preview_data(merged, args.preview)
    
    print("\n✓ Merge complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
