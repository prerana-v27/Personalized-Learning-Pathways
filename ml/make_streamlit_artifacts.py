"""Create artifacts expected by the Streamlit app.

Generates:
- artifacts/tfidf.pkl       (TfidfVectorizer fitted on catalog texts)
- artifacts/X_items.npz     (item TF-IDF matrix, L2-normalized)
- artifacts/id_maps.pkl     (dict with item_map and optional user_map)

Usage:
  python ml\\make_streamlit_artifacts.py --catalog Data\\processed\\all_courses_cleaned.csv --out artifacts
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from scipy import sparse


def build_and_save(catalog_path: str, out_dir: str):
    catalog_path = Path(catalog_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not catalog_path.exists():
        raise FileNotFoundError(catalog_path)

    df = pd.read_csv(catalog_path)

    # Ensure course_id exists
    if 'course_id' not in df.columns:
        df['course_id'] = ['c' + str(i) for i in df.index]

    # Prepare texts (title + subject + description + skills)
    text_cols = []
    for c in ('title', 'course_title', 'course_name'):
        if c in df.columns:
            text_cols.append(c)
            break
    for c in ('subject', 'description', 'skills'):
        if c in df.columns:
            text_cols.append(c)

    if text_cols:
        texts = df[text_cols].fillna('').astype(str).agg(' '.join, axis=1).tolist()
    else:
        texts = df.fillna('').astype(str).agg(' '.join, axis=1).tolist()

    # Fit TF-IDF
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(max_features=8192, ngram_range=(1,2))
    X = tfidf.fit_transform(texts)

    # L2-normalize rows
    from sklearn.preprocessing import normalize
    X = normalize(X, norm='l2', axis=1, copy=False)

    # Save tfidf and X_items
    joblib.dump(tfidf, out_dir / 'tfidf.pkl')
    sparse.save_npz(out_dir / 'X_items.npz', X)

    # Create id_maps: item_map maps course_id -> index
    item_map = {str(cid): idx for idx, cid in enumerate(df['course_id'].astype(str).tolist())}
    id_maps = {'item_map': item_map}

    # If interactions file exists, build user_map
    interactions_path = catalog_path.parent / 'interactions_synth.csv'
    if interactions_path.exists():
        inter = pd.read_csv(interactions_path)
        if 'user_id' in inter.columns:
            users = sorted(inter['user_id'].unique())
            user_map = {str(u): i for i, u in enumerate(users)}
            id_maps['user_map'] = user_map

    joblib.dump(id_maps, out_dir / 'id_maps.pkl')

    print(f"Saved tfidf.pkl, X_items.npz, id_maps.pkl to {out_dir}")


def cli():
    parser = argparse.ArgumentParser(description="Create artifacts for Streamlit app")
    parser.add_argument('--catalog', default='Data/processed/all_courses_cleaned.csv', help='Path to catalog CSV')
    parser.add_argument('--out', default='artifacts', help='Output artifacts directory')
    args = parser.parse_args()
    build_and_save(args.catalog, args.out)


if __name__ == '__main__':
    cli()
