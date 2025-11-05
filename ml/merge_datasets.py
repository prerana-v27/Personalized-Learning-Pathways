#!/usr/bin/env python3
"""
Merge all four cleaned course datasets into a unified CSV.

Reads:
  - data/processed/coursera_cleaned.csv
  - data/processed/edx_cleaned.csv
  - data/processed/nptel_cleaned.csv
  - data/processed/udemy_cleaned.csv

Output:
  - data/processed/all_courses_merged.csv (new file, does not overwrite originals)

Schema (unified):
  title, difficulty, university, price, rating, level, subject, is_paid, skills, source
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd


def normalize_column_name(col: str) -> str:
    """Convert column name to lowercase with underscores, remove non-alphanumeric."""
    import re
    # Lowercase and strip
    normalized = col.strip().lower()
    # Replace non-alphanumeric with underscore
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    # Collapse multiple underscores
    normalized = re.sub(r"_+", "_", normalized)
    # Trim leading/trailing underscores
    normalized = normalized.strip("_")
    return normalized


def map_level_to_difficulty(level: str) -> str | None:
    """Map Udemy level to standardized difficulty."""
    if not level or level is None or (isinstance(level, float) and pd.isna(level)):
        return None
    level_str = str(level).strip().lower()
    if level_str in ("beginner", "все уровни"):  # Handle multiple languages
        return "Beginner"
    elif level_str in ("intermediate",):
        return "Intermediate"
    elif level_str in ("expert", "advanced", "all levels"):
        if level_str == "all levels":
            return None
        return "Advanced"
    return None


def standardize_difficulty(diff: str) -> str | None:
    """Standardize difficulty to title-case {Beginner, Intermediate, Advanced}."""
    if not diff or diff is None or (isinstance(diff, float) and pd.isna(diff)):
        return None
    diff_str = str(diff).strip().lower()
    if "beginner" in diff_str:
        return "Beginner"
    elif "intermediate" in diff_str:
        return "Intermediate"
    elif "advanced" in diff_str or "expert" in diff_str:
        return "Advanced"
    return None


def load_coursera(path: str) -> pd.DataFrame:
    """Load and map Coursera dataset to unified schema."""
    df = pd.read_csv(path)
    # Normalize column names for easier lookup
    col_map = {normalize_column_name(c): c for c in df.columns}
    
    result = pd.DataFrame()
    result["title"] = df.get(col_map.get("course_name"), pd.Series(dtype=object))
    result["rating"] = pd.to_numeric(df.get(col_map.get("course_rating"), pd.Series(dtype=object)), errors="coerce")
    result["difficulty"] = df.get(col_map.get("difficulty_level"), pd.Series(dtype=object)).apply(standardize_difficulty)
    result["university"] = df.get(col_map.get("university"), pd.Series(dtype=object))
    result["skills"] = df.get(col_map.get("skills"), pd.Series(dtype=object))
    result["source"] = "coursera"
    
    print(f"Loaded Coursera: {len(result)} rows")
    return result


def load_edx(path: str) -> pd.DataFrame:
    """Load and map edX dataset to unified schema."""
    df = pd.read_csv(path)
    col_map = {normalize_column_name(c): c for c in df.columns}
    
    result = pd.DataFrame()
    result["title"] = df.get(col_map.get("name"), pd.Series(dtype=object))
    result["difficulty"] = df.get(col_map.get("difficulty_level"), pd.Series(dtype=object)).apply(standardize_difficulty)
    result["university"] = df.get(col_map.get("university"), pd.Series(dtype=object))
    result["source"] = "edx"
    
    print(f"Loaded edX: {len(result)} rows")
    return result


def load_nptel(path: str) -> pd.DataFrame:
    """Load and map NPTEL dataset to unified schema."""
    df = pd.read_csv(path)
    col_map = {normalize_column_name(c): c for c in df.columns}
    
    result = pd.DataFrame()
    result["title"] = df.get(col_map.get("course_name"), pd.Series(dtype=object))
    result["university"] = df.get(col_map.get("instructor"), pd.Series(dtype=object))
    result["source"] = "nptel"
    
    print(f"Loaded NPTEL: {len(result)} rows")
    return result


def load_udemy(path: str) -> pd.DataFrame:
    """Load and map Udemy dataset to unified schema."""
    df = pd.read_csv(path)
    col_map = {normalize_column_name(c): c for c in df.columns}
    
    result = pd.DataFrame()
    result["title"] = df.get(col_map.get("course_title"), pd.Series(dtype=object))
    result["price"] = pd.to_numeric(df.get(col_map.get("price"), pd.Series(dtype=object)), errors="coerce")
    result["level"] = df.get(col_map.get("level"), pd.Series(dtype=object))
    result["subject"] = df.get(col_map.get("subject"), pd.Series(dtype=object))
    result["is_paid"] = df.get(col_map.get("is_paid"), pd.Series(dtype=object)).astype(bool, errors="ignore")
    result["rating"] = pd.to_numeric(df.get(col_map.get("num_reviews"), pd.Series(dtype=object)), errors="coerce")  # approx proxy
    result["source"] = "udemy"
    
    # Map level to difficulty if difficulty is missing
    result["difficulty"] = result["level"].apply(map_level_to_difficulty)
    
    print(f"Loaded Udemy: {len(result)} rows")
    return result


def main() -> int:
    """Load all datasets, merge, and save to unified CSV."""
    # Paths
    data_dir = Path("data") / "processed"
    coursera_path = data_dir / "coursera_cleaned.csv"
    edx_path = data_dir / "edx_cleaned.csv"
    nptel_path = data_dir / "nptel_cleaned.csv"
    udemy_path = data_dir / "udemy_cleaned.csv"
    output_path = data_dir / "all_courses_merged.csv"
    
    # Check that originals exist
    for p in [coursera_path, edx_path, nptel_path, udemy_path]:
        if not p.exists():
            print(f"ERROR: {p} not found", file=sys.stderr)
            return 1
    
    print("Loading datasets...\n")
    
    # Load and map each dataset
    try:
        coursera = load_coursera(str(coursera_path))
        edx = load_edx(str(edx_path))
        nptel = load_nptel(str(nptel_path))
        udemy = load_udemy(str(udemy_path))
    except Exception as e:
        print(f"ERROR loading datasets: {e}", file=sys.stderr)
        return 2
    
    # Concatenate
    print("\nMerging datasets...")
    merged = pd.concat([coursera, edx, nptel, udemy], ignore_index=True)
    
    # Ensure unified schema (all columns present, even if empty)
    schema_cols = ["title", "difficulty", "university", "price", "rating", "level", "subject", "is_paid", "skills", "source"]
    for col in schema_cols:
        if col not in merged.columns:
            merged[col] = None
    
    # Reorder columns
    merged = merged[schema_cols]
    
    # Save
    try:
        merged.to_csv(output_path, index=False, encoding="utf-8")
        print(f"\nMerged dataset saved to: {output_path}")
    except Exception as e:
        print(f"ERROR writing {output_path}: {e}", file=sys.stderr)
        return 3
    
    # Print summary
    print(f"\n=== Merged Dataset Summary ===")
    print(f"Shape: {merged.shape[0]} rows, {merged.shape[1]} columns")
    print(f"\nFirst 5 rows:")
    print(merged.head())
    print(f"\nColumn null counts:")
    print(merged.isnull().sum())
    print(f"\nData types:")
    print(merged.dtypes)
    
    # Quality report
    print(f"\n=== Quality Report ===")
    print(f"\nDifficulty value counts:")
    print(merged["difficulty"].value_counts(dropna=False))
    print(f"\nNon-null percentages:")
    for col in ["title", "skills", "price", "rating", "subject"]:
        non_null_pct = (merged[col].notna().sum() / len(merged)) * 100
        print(f"  {col}: {non_null_pct:.1f}%")
    
    print(f"\n✓ Merge complete! Output: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
