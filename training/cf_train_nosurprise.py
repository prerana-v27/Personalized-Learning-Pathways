#!/usr/bin/env python3
"""
Collaborative Filtering Training Script (No Surprise Library)
Pure scikit-learn/scipy implementation for Python 3.13+ compatibility.
"""

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train collaborative filtering models without surprise library"
    )
    parser.add_argument("--data", required=True, help="Path to CSV file")
    parser.add_argument("--user-col", default="user_id", help="User column name")
    parser.add_argument("--item-col", default="course_id", help="Item column name")
    parser.add_argument("--rating-col", default="rating", help="Rating column name")
    parser.add_argument("--time-col", default="timestamp", help="Timestamp column name")
    parser.add_argument("--k", type=int, default=10, help="Top-K for metrics")
    parser.add_argument("--out", default="artifacts", help="Output directory")
    parser.add_argument("--fast", action="store_true", help="Fast mode: fewer iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_and_prepare_data(args):
    """Load CSV and prepare ratings."""
    logger.info(f"Loading data from {args.data}")
    
    try:
        df = pd.read_csv(args.data)
    except FileNotFoundError:
        logger.error(f"File not found: {args.data}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        sys.exit(1)
    
    # Validate required columns
    required_cols = [args.user_col, args.item_col]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        sys.exit(1)
    
    logger.info(f"Loaded {len(df)} interactions")
    
    # Handle rating column
    if args.rating_col not in df.columns:
        logger.warning(f"Rating column '{args.rating_col}' not found. Creating implicit ratings (1.0)")
        df[args.rating_col] = 1.0
    else:
        # Fill missing ratings with 1.0
        missing_ratings = df[args.rating_col].isna().sum()
        if missing_ratings > 0:
            logger.warning(f"Found {missing_ratings} missing ratings. Filling with 1.0")
            df[args.rating_col] = df[args.rating_col].fillna(1.0)
    
    # Clip ratings to [0, 5]
    df[args.rating_col] = df[args.rating_col].clip(0, 5)
    
    # Handle timestamp column
    if args.time_col not in df.columns:
        logger.warning(f"Timestamp column '{args.time_col}' not found. Using row order as temporal sequence")
        df[args.time_col] = pd.RangeIndex(len(df))
    else:
        # Try to parse as datetime
        try:
            df[args.time_col] = pd.to_datetime(df[args.time_col])
        except Exception as e:
            logger.warning(f"Could not parse timestamp as datetime: {e}. Using as-is for sorting")
    
    # Sort by user and timestamp
    df = df.sort_values([args.user_col, args.time_col]).reset_index(drop=True)
    
    logger.info(f"Users: {df[args.user_col].nunique()}, Items: {df[args.item_col].nunique()}")
    logger.info(f"Rating range: [{df[args.rating_col].min():.2f}, {df[args.rating_col].max():.2f}]")
    
    return df


def temporal_split(df, user_col, item_col, rating_col):
    """
    Per-user temporal split: last interaction to test, rest to train.
    Users with only 1 interaction stay in train but excluded from metrics.
    """
    logger.info("Performing per-user temporal split (last interaction to test)")
    
    train_rows = []
    test_rows = []
    single_interaction_users = set()
    
    for user, group in tqdm(df.groupby(user_col), desc="Splitting users"):
        if len(group) == 1:
            # Keep in train, mark for exclusion from metrics
            train_rows.append(group.iloc[0])
            single_interaction_users.add(user)
        else:
            # Last interaction to test, rest to train
            train_rows.extend(group.iloc[:-1].to_dict('records'))
            test_rows.append(group.iloc[-1])
    
    train_df = pd.DataFrame(train_rows)
    test_df = pd.DataFrame(test_rows) if test_rows else pd.DataFrame()
    
    logger.info(f"Train: {len(train_df)} interactions")
    logger.info(f"Test: {len(test_df)} interactions")
    logger.info(f"Single-interaction users (excluded from metrics): {len(single_interaction_users)}")
    
    return train_df, test_df, single_interaction_users


def build_sparse_matrix(train_df, user_col, item_col, rating_col):
    """Build sparse user-item rating matrix and index mappings."""
    logger.info("Building sparse rating matrix")
    
    # Create user and item index mappings
    users = sorted(train_df[user_col].unique())
    items = sorted(train_df[item_col].unique())
    
    user_to_idx = {u: i for i, u in enumerate(users)}
    item_to_idx = {it: i for i, it in enumerate(items)}
    idx_to_user = {i: u for u, i in user_to_idx.items()}
    idx_to_item = {i: it for it, i in item_to_idx.items()}
    
    # Build sparse matrix
    rows = train_df[user_col].map(user_to_idx).values
    cols = train_df[item_col].map(item_to_idx).values
    data = train_df[rating_col].values
    
    rating_matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(len(users), len(items))
    )
    
    logger.info(f"Sparse matrix shape: {rating_matrix.shape}")
    logger.info(f"Sparsity: {100 * (1 - rating_matrix.nnz / (rating_matrix.shape[0] * rating_matrix.shape[1])):.2f}%")
    
    id_maps = {
        'user_to_idx': user_to_idx,
        'item_to_idx': item_to_idx,
        'idx_to_user': idx_to_user,
        'idx_to_item': idx_to_item
    }
    
    return rating_matrix, id_maps


class PopularityModel:
    """Popularity baseline recommender."""
    
    def __init__(self):
        self.item_popularity = None
        self.popular_items = None
    
    def fit(self, rating_matrix):
        """Fit popularity model."""
        # Count interactions per item
        self.item_popularity = np.array(rating_matrix.sum(axis=0)).flatten()
        # Sort items by popularity (descending)
        self.popular_items = np.argsort(self.item_popularity)[::-1]
        return self
    
    def recommend(self, user_idx, seen_items, k=10, n_items=None):
        """Recommend top-K popular items excluding seen items."""
        if n_items is None:
            n_items = len(self.popular_items)
        
        recommendations = []
        for item_idx in self.popular_items:
            if item_idx >= n_items:
                continue
            if item_idx not in seen_items:
                recommendations.append(item_idx)
                if len(recommendations) >= k:
                    break
        
        return recommendations


class ItemKNNModel:
    """Item-based KNN using cosine similarity."""
    
    def __init__(self, k=40):
        self.k = k
        self.item_similarity = None
        self.rating_matrix = None
    
    def fit(self, rating_matrix):
        """Fit Item-KNN model."""
        logger.info(f"Computing item-item cosine similarity (K={self.k})")
        self.rating_matrix = rating_matrix
        
        # Compute item-item similarity (transpose to get items as rows)
        item_matrix = rating_matrix.T.tocsr()
        self.item_similarity = cosine_similarity(item_matrix, dense_output=False)
        
        logger.info(f"Item similarity matrix shape: {self.item_similarity.shape}")
        return self
    
    def predict_for_user(self, user_idx):
        """Predict ratings for all items for a given user."""
        # Get user's ratings
        user_ratings = self.rating_matrix[user_idx].toarray().flatten()
        
        # For each item, compute weighted sum of similarities with rated items
        predictions = np.zeros(self.rating_matrix.shape[1])
        
        for item_idx in range(self.rating_matrix.shape[1]):
            # Get K most similar items
            sim_scores = self.item_similarity[item_idx].toarray().flatten()
            
            # Only consider items the user has rated
            rated_mask = user_ratings > 0
            
            if not rated_mask.any():
                continue
            
            # Get similarities with rated items
            rated_sims = sim_scores[rated_mask]
            rated_ratings = user_ratings[rated_mask]
            
            # Get top-K most similar
            if len(rated_sims) > self.k:
                top_k_idx = np.argpartition(rated_sims, -self.k)[-self.k:]
                rated_sims = rated_sims[top_k_idx]
                rated_ratings = rated_ratings[top_k_idx]
            
            # Weighted average
            sim_sum = np.abs(rated_sims).sum()
            if sim_sum > 0:
                predictions[item_idx] = (rated_sims * rated_ratings).sum() / sim_sum
        
        return predictions
    
    def recommend(self, user_idx, seen_items, k=10, n_items=None):
        """Recommend top-K items for user."""
        predictions = self.predict_for_user(user_idx)
        
        # Mask out seen items
        predictions[list(seen_items)] = -np.inf
        
        # Get top-K
        if n_items is not None:
            predictions = predictions[:n_items]
        
        top_k_idx = np.argpartition(predictions, -k)[-k:]
        top_k_idx = top_k_idx[np.argsort(predictions[top_k_idx])[::-1]]
        
        return top_k_idx.tolist()


class NMFModel:
    """NMF-based collaborative filtering."""
    
    def __init__(self, n_factors=60, random_state=42, max_iter=200):
        self.n_factors = n_factors
        self.random_state = random_state
        self.max_iter = max_iter
        self.model = None
        self.user_factors = None
        self.item_factors = None
    
    def fit(self, rating_matrix):
        """Fit NMF model."""
        logger.info(f"Training NMF model (n_factors={self.n_factors})")
        
        self.model = NMF(
            n_components=self.n_factors,
            init='random',
            random_state=self.random_state,
            max_iter=self.max_iter,
            verbose=0
        )
        
        # Fit NMF: R ≈ W * H
        self.user_factors = self.model.fit_transform(rating_matrix)
        self.item_factors = self.model.components_
        
        logger.info(f"User factors shape: {self.user_factors.shape}")
        logger.info(f"Item factors shape: {self.item_factors.shape}")
        
        return self
    
    def predict_for_user(self, user_idx):
        """Predict ratings for all items for a given user."""
        return self.user_factors[user_idx] @ self.item_factors
    
    def recommend(self, user_idx, seen_items, k=10, n_items=None):
        """Recommend top-K items for user."""
        predictions = self.predict_for_user(user_idx)
        
        # Mask out seen items
        predictions[list(seen_items)] = -np.inf
        
        # Get top-K
        if n_items is not None:
            predictions = predictions[:n_items]
        
        top_k_idx = np.argpartition(predictions, -k)[-k:]
        top_k_idx = top_k_idx[np.argsort(predictions[top_k_idx])[::-1]]
        
        return top_k_idx.tolist()


def get_user_train_items(train_df, user_col, item_col, id_maps):
    """Get set of item indices each user interacted with in training."""
    user_items = defaultdict(set)
    
    for _, row in train_df.iterrows():
        user = row[user_col]
        item = row[item_col]
        
        if user in id_maps['user_to_idx'] and item in id_maps['item_to_idx']:
            user_idx = id_maps['user_to_idx'][user]
            item_idx = id_maps['item_to_idx'][item]
            user_items[user_idx].add(item_idx)
    
    return user_items


def evaluate_model(model, test_df, user_col, item_col, rating_col, id_maps, 
                   user_train_items, k=10, model_name="Model"):
    """
    Evaluate model with top-K metrics.
    
    Returns:
        dict with precision@k, recall@k, ndcg@k
    """
    logger.info(f"Evaluating {model_name}")
    
    precisions = []
    recalls = []
    ndcgs = []
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Evaluating {model_name}"):
        user = row[user_col]
        true_item = row[item_col]
        
        # Skip if user or item not in training set
        if user not in id_maps['user_to_idx'] or true_item not in id_maps['item_to_idx']:
            continue
        
        user_idx = id_maps['user_to_idx'][user]
        true_item_idx = id_maps['item_to_idx'][true_item]
        
        # Get seen items for this user
        seen_items = user_train_items.get(user_idx, set())
        
        # Skip if true item was in training (shouldn't happen with proper split)
        if true_item_idx in seen_items:
            continue
        
        # Get recommendations
        try:
            top_k_items = model.recommend(user_idx, seen_items, k=k)
        except Exception as e:
            logger.warning(f"Error generating recommendations for user {user}: {e}")
            continue
        
        # Metrics
        hit = 1 if true_item_idx in top_k_items else 0
        precisions.append(hit / min(k, len(top_k_items)) if top_k_items else 0)
        recalls.append(hit)  # Binary: 1 relevant item
        
        # NDCG@K
        if hit:
            rank = top_k_items.index(true_item_idx) + 1
            ndcg = 1.0 / np.log2(rank + 1)
        else:
            ndcg = 0.0
        ndcgs.append(ndcg)
    
    metrics = {
        f'Precision@{k}': np.mean(precisions) if precisions else 0.0,
        f'Recall@{k}': np.mean(recalls) if recalls else 0.0,
        f'NDCG@{k}': np.mean(ndcgs) if ndcgs else 0.0,
    }
    
    logger.info(f"{model_name} - P@{k}: {metrics[f'Precision@{k}']:.4f}, "
                f"R@{k}: {metrics[f'Recall@{k}']:.4f}, "
                f"NDCG@{k}: {metrics[f'NDCG@{k}']:.4f}")
    
    return metrics


def print_metrics_table(results, k):
    """Print formatted metrics table."""
    print("\n" + "="*80)
    print(f"{'Model':<15} {'P@'+str(k):<15} {'R@'+str(k):<15} {'NDCG@'+str(k):<15}")
    print("="*80)
    
    for model_name, metrics in results.items():
        p = metrics.get(f'Precision@{k}', 0.0)
        r = metrics.get(f'Recall@{k}', 0.0)
        n = metrics.get(f'NDCG@{k}', 0.0)
        
        print(f"{model_name:<15} {p:<15.4f} {r:<15.4f} {n:<15.4f}")
    
    print("="*80 + "\n")


def save_artifacts(out_dir, popularity_model, knn_model, nmf_model, id_maps, split_info):
    """Save trained models and artifacts."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving artifacts to {out_dir}")
    
    # Save models
    joblib.dump(popularity_model, out_path / "popular.pkl")
    logger.info("Saved Popularity model to popular.pkl")
    
    joblib.dump(knn_model, out_path / "cf_knn.pkl")
    logger.info("Saved KNN model to cf_knn.pkl")
    
    joblib.dump(nmf_model, out_path / "cf_nmf.pkl")
    logger.info("Saved NMF model to cf_nmf.pkl")
    
    # Save ID maps
    joblib.dump(id_maps, out_path / "id_maps.pkl")
    logger.info("Saved ID mappings to id_maps.pkl")
    
    # Save split info
    joblib.dump(split_info, out_path / "split.pkl")
    logger.info("Saved split information to split.pkl")


def main():
    """Main training pipeline."""
    args = parse_args()
    
    logger.info("="*80)
    logger.info("Collaborative Filtering Training Pipeline (No Surprise)")
    logger.info("="*80)
    logger.info(f"Data: {args.data}")
    logger.info(f"Top-K: {args.k}")
    logger.info(f"Output: {args.out}")
    logger.info(f"Fast mode: {args.fast}")
    logger.info(f"Random seed: {args.seed}")
    logger.info("="*80)
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Load and prepare data
    df = load_and_prepare_data(args)
    
    # Temporal split
    train_df, test_df, single_users = temporal_split(
        df, args.user_col, args.item_col, args.rating_col
    )
    
    if len(test_df) == 0:
        logger.error("No test data available. All users have only 1 interaction.")
        sys.exit(1)
    
    # Build sparse matrix
    rating_matrix, id_maps = build_sparse_matrix(
        train_df, args.user_col, args.item_col, args.rating_col
    )
    
    # Get user train items for evaluation
    user_train_items = get_user_train_items(train_df, args.user_col, args.item_col, id_maps)
    
    # Train models
    logger.info("\n" + "="*80)
    logger.info("MODEL TRAINING")
    logger.info("="*80)
    
    # Popularity baseline
    logger.info("Training Popularity baseline")
    popularity_model = PopularityModel()
    popularity_model.fit(rating_matrix)
    
    # Item-KNN
    knn_k = 20 if args.fast else 40
    knn_model = ItemKNNModel(k=knn_k)
    knn_model.fit(rating_matrix)
    
    # NMF
    nmf_factors = 40 if args.fast else 60
    nmf_iters = 100 if args.fast else 200
    nmf_model = NMFModel(n_factors=nmf_factors, random_state=args.seed, max_iter=nmf_iters)
    nmf_model.fit(rating_matrix)
    
    # Evaluate models
    logger.info("\n" + "="*80)
    logger.info("MODEL EVALUATION")
    logger.info("="*80)
    
    results = {}
    
    # Popularity
    results['Popularity'] = evaluate_model(
        popularity_model, test_df, args.user_col, args.item_col, args.rating_col,
        id_maps, user_train_items, k=args.k, model_name="Popularity"
    )
    
    # Item-KNN
    results['ItemKNN'] = evaluate_model(
        knn_model, test_df, args.user_col, args.item_col, args.rating_col,
        id_maps, user_train_items, k=args.k, model_name="ItemKNN"
    )
    
    # NMF
    results['NMF'] = evaluate_model(
        nmf_model, test_df, args.user_col, args.item_col, args.rating_col,
        id_maps, user_train_items, k=args.k, model_name="NMF"
    )
    
    # Print results table
    print_metrics_table(results, args.k)
    
    # Save artifacts
    split_info = {
        'train_size': len(train_df),
        'test_size': len(test_df),
        'single_interaction_users': single_users,
        'knn_k': knn_k,
        'nmf_factors': nmf_factors,
        'metrics': results
    }
    
    save_artifacts(args.out, popularity_model, knn_model, nmf_model, id_maps, split_info)
    
    logger.info("\n" + "="*80)
    logger.info("Training completed successfully!")
    logger.info("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nTraining failed: {e}", exc_info=True)
        sys.exit(1)
