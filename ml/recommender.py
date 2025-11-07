"""Simple content-based recommender for course catalog.

Features:
- Builds course text representations using SentenceTransformers (preferred) or TF-IDF (fallback)
- Indexes with sklearn NearestNeighbors (cosine) and provides top-K recommendations
- CLI to build index and to query by course title or id

Usage examples:
  python ml/recommender.py --csv data/processed/all_courses_merged.csv --build-index --index-out ml/outputs/index.joblib
  python ml/recommender.py --csv data/processed/all_courses_merged.csv --index-in ml/outputs/index.joblib --title "Introduction to Python" --topk 10
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd


def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    return df


def prepare_texts(df: pd.DataFrame) -> List[str]:
    # Use title + subject + (if available) description/skills
    cols = []
    if "title" in df.columns:
        cols.append("title")
    elif "course_title" in df.columns:
        cols.append("course_title")
    # optional columns
    for c in ("subject", "skills", "description", "course_name"):
        if c in df.columns:
            cols.append(c)

    if not cols:
        # fallback: stringify whole row
        texts = df.astype(str).agg(" ".join, axis=1).tolist()
    else:
        texts = df[cols].fillna("").astype(str).agg(" ".join, axis=1).tolist()
    return texts


def build_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> (np.ndarray, str):
    """Try to use sentence-transformers, otherwise use TF-IDF as fallback.

    Returns (embeddings, method_name)
    """
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        emb = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        return emb, f"sbert:{model_name}"
    except Exception:
        # fallback to TF-IDF
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(max_features=8192, ngram_range=(1, 2))
        emb = vec.fit_transform(texts)
        # convert sparse to dense (ok for moderate dataset sizes)
        try:
            arr = emb.toarray()
        except Exception:
            arr = np.asarray(emb.todense())
        return arr, "tfidf"


def build_index(emb: np.ndarray, n_neighbors: int = 50):
    from sklearn.neighbors import NearestNeighbors

    # Use cosine distance via metric='cosine'
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn.fit(emb)
    return nn


def save_index(path: str, index_obj, embeddings: np.ndarray, df: pd.DataFrame, method: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"index": index_obj, "embeddings": embeddings, "df": df, "method": method}, path)


def load_index(path: str):
    return joblib.load(path)


def recommend_by_idx(index_data, idx: int, topk: int = 10):
    index = index_data["index"]
    embeddings = index_data["embeddings"]
    df = index_data["df"]
    if isinstance(embeddings, np.ndarray):
        q = embeddings[idx: idx + 1]
    else:
        q = embeddings[idx]
    dists, inds = index.kneighbors(q, n_neighbors=topk + 1)
    # skip the first if it's the same item
    inds = inds[0].tolist()
    if inds and inds[0] == idx:
        inds = inds[1:topk + 1]
    else:
        inds = inds[:topk]
    return df.iloc[inds]


def recommend_by_text(index_data, text: str, topk: int = 10, model_name: str = "all-MiniLM-L6-v2"):
    method = index_data.get("method", "tfidf")
    if method.startswith("sbert"):
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(method.split(":", 1)[1])
            q_emb = model.encode([text], convert_to_numpy=True)
        except Exception:
            # fallback encode using tfidf vectorizer not available here; raise
            raise RuntimeError("SentenceTransformer unavailable for query embedding")
    else:
        # cannot transform with TFIDF vectorizer since we didn't save the vectorizer
        # so use a simple approximate approach: find row with most token overlap (fallback)
        df = index_data["df"]
        texts = df.index.astype(str).tolist()
        # fallback: return top rows by simple substring match
        candidates = df[df.apply(lambda r: text.lower() in " ".join(r.astype(str)).lower(), axis=1)]
        return candidates.head(topk)

    index = index_data["index"]
    dists, inds = index.kneighbors(q_emb, n_neighbors=topk)
    return index_data["df"].iloc[inds[0]]


def cli(argv: Optional[List[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description="Build simple content-based recommender index or query it")
    parser.add_argument("--csv", required=True, help="Merged courses CSV path")
    parser.add_argument("--build-index", action="store_true", help="Build and save index")
    parser.add_argument("--index-out", default="ml/outputs/recommender_index.joblib", help="Where to save index")
    parser.add_argument("--index-in", help="Load an existing index")
    parser.add_argument("--title", help="Query by title text")
    parser.add_argument("--id", type=int, help="Query by row index in CSV")
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args(argv)

    if args.build_index:
        df = load_data(args.csv)
        texts = prepare_texts(df)
        print(f"Building embeddings for {len(texts)} items...")
        emb, method = build_embeddings(texts)
        print(f"Embeddings shape: {getattr(emb, 'shape', None)}, method={method}")
        nn = build_index(emb, n_neighbors=min(200, len(texts)))
        save_index(args.index_out, nn, emb, df.reset_index(drop=True), method)
        print(f"Index saved to {args.index_out}")
        return 0

    if args.index_in:
        data = load_index(args.index_in)
    else:
        # build in-memory
        df = load_data(args.csv)
        texts = prepare_texts(df)
        emb, method = build_embeddings(texts)
        nn = build_index(emb, n_neighbors=min(200, len(texts)))
        data = {"index": nn, "embeddings": emb, "df": df.reset_index(drop=True), "method": method}

    if args.id is not None:
        res = recommend_by_idx(data, args.id, topk=args.topk)
        print(res.head(args.topk).to_string(index=False))
        return 0

    if args.title:
        res = recommend_by_text(data, args.title, topk=args.topk)
        print(res.head(args.topk).to_string(index=False))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(cli())
