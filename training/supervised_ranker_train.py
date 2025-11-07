#!/usr/bin/env python3
"""
Supervised Logistic Regression Ranker
Trains a point-wise learning-to-rank model using content and interaction features.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, load_npz
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, normalize
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
        description="Train supervised logistic regression ranker"
    )
    parser.add_argument("--catalog", required=True, help="Path to course catalog CSV")
    parser.add_argument("--interactions", required=True, help="Path to interactions CSV")
    parser.add_argument("--neg-per-pos", type=int, default=5, help="Negative samples per positive")
    parser.add_argument("--k", type=int, default=10, help="Top-K for metrics")
    parser.add_argument("--out", default="artifacts", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_artifacts(out_dir):
    """Load pre-computed artifacts from previous training runs."""
    logger.info("Loading artifacts from previous training runs")
    
    artifacts_path = Path(out_dir)
    
    # Check required files
    required_files = {
        'split.pkl': 'train/test split',
        'id_maps.pkl': 'ID mappings',
        'tfidf.pkl': 'TF-IDF vectorizer'
    }
    
    missing = []
    for file, desc in required_files.items():
        if not (artifacts_path / file).exists():
            missing.append(f"{file} ({desc})")
    
    if missing:
        logger.error(f"Missing required artifacts: {', '.join(missing)}")
        logger.error("Please run content_cluster_train.py first to generate these artifacts")
        sys.exit(1)
    
    # Load artifacts
    split_info = joblib.load(artifacts_path / "split.pkl")
    id_maps = joblib.load(artifacts_path / "id_maps.pkl")
    tfidf_vectorizer = joblib.load(artifacts_path / "tfidf.pkl")
    
    # Try to load TF-IDF matrix
    if (artifacts_path / "X_items.npz").exists():
        X_items = load_npz(artifacts_path / "X_items.npz")
        logger.info(f"Loaded TF-IDF item matrix: {X_items.shape}")
    else:
        logger.error("TF-IDF item matrix (X_items.npz) not found")
        logger.error("Please run content_cluster_train.py first")
        sys.exit(1)
    
    logger.info(f"Loaded artifacts: {len(id_maps['user_to_idx'])} users, {len(id_maps['item_to_idx'])} items")
    
    return split_info, id_maps, tfidf_vectorizer, X_items


def load_split_data(interactions_path, split_info):
    """Load and recreate train/test split from interactions."""
    logger.info(f"Loading interactions from {interactions_path}")
    
    try:
        df = pd.read_csv(interactions_path)
    except FileNotFoundError:
        logger.error(f"Interactions file not found: {interactions_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading interactions: {e}")
        sys.exit(1)
    
    # Parse timestamp and sort
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    
    # Recreate train/test split
    train_rows = []
    test_rows = []
    single_users = split_info.get('single_interaction_users', set())
    
    for user, group in df.groupby('user_id'):
        if user in single_users or len(group) == 1:
            train_rows.append(group.iloc[0])
        else:
            train_rows.extend(group.iloc[:-1].to_dict('records'))
            test_rows.append(group.iloc[-1])
    
    train_df = pd.DataFrame(train_rows)
    test_df = pd.DataFrame(test_rows) if test_rows else pd.DataFrame()
    
    logger.info(f"Split: {len(train_df)} train, {len(test_df)} test")
    
    return train_df, test_df


def build_user_profiles(train_df, X_items, id_maps):
    """Build user profiles as mean of TF-IDF vectors of interacted items."""
    logger.info("Building user profiles from training interactions")
    
    user_to_idx = id_maps['user_to_idx']
    item_to_idx = id_maps['item_to_idx']
    n_users = len(user_to_idx)
    n_features = X_items.shape[1]
    
    # Use sparse accumulation
    user_profiles_data = []
    user_profiles_rows = []
    user_profiles_cols = []
    user_counts = np.zeros(n_users)
    
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Building profiles"):
        user = row['user_id']
        item = row['course_id']
        
        if user not in user_to_idx or item not in item_to_idx:
            continue
        
        user_idx = user_to_idx[user]
        item_idx = item_to_idx[item]
        
        # Get rating weight (default 1.0)
        weight = row.get('rating', 1.0) if 'rating' in row else 1.0
        
        # Get item vector (sparse)
        item_vector = X_items[item_idx]
        
        # Accumulate weighted item features
        for feat_idx in item_vector.indices:
            user_profiles_rows.append(user_idx)
            user_profiles_cols.append(feat_idx)
            user_profiles_data.append(weight * item_vector.data[item_vector.indices == feat_idx][0])
        
        user_counts[user_idx] += weight
    
    # Create sparse matrix and normalize by count
    user_profiles_sparse = csr_matrix(
        (user_profiles_data, (user_profiles_rows, user_profiles_cols)),
        shape=(n_users, n_features)
    )
    
    # Normalize by count
    for i in range(n_users):
        if user_counts[i] > 0:
            user_profiles_sparse.data[user_profiles_sparse.indptr[i]:user_profiles_sparse.indptr[i+1]] /= user_counts[i]
    
    logger.info(f"User profiles shape: {user_profiles_sparse.shape}")
    logger.info(f"Users with profiles: {(user_counts > 0).sum()}/{n_users}")
    
    return user_profiles_sparse


def compute_item_statistics(train_df, id_maps):
    """Compute item popularity and recency from training data."""
    logger.info("Computing item statistics from training data")
    
    item_to_idx = id_maps['item_to_idx']
    n_items = len(id_maps['idx_to_item'])
    
    item_counts = np.zeros(n_items)
    item_last_timestamp = {}
    
    # Parse timestamps
    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
    current_time = train_df['timestamp'].max()
    
    for _, row in train_df.iterrows():
        item = row['course_id']
        if item in item_to_idx:
            idx = item_to_idx[item]
            item_counts[idx] += 1
            
            timestamp = row['timestamp']
            if idx not in item_last_timestamp or timestamp > item_last_timestamp[idx]:
                item_last_timestamp[idx] = timestamp
    
    # Compute recency in days
    item_recency_days = np.zeros(n_items)
    for idx, last_time in item_last_timestamp.items():
        days_since = (current_time - last_time).days
        item_recency_days[idx] = days_since
    
    # For items never seen, set to max
    max_days = item_recency_days.max() if item_recency_days.max() > 0 else 365
    item_recency_days[item_counts == 0] = max_days + 1
    
    # Standardize popularity
    pop_mean = item_counts.mean()
    pop_std = item_counts.std()
    if pop_std > 0:
        item_popularity_z = (item_counts - pop_mean) / pop_std
    else:
        item_popularity_z = np.zeros_like(item_counts)
    
    logger.info(f"Item popularity range: [{item_counts.min():.0f}, {item_counts.max():.0f}]")
    logger.info(f"Item recency range: [{item_recency_days.min():.0f}, {item_recency_days.max():.0f}] days")
    
    return item_popularity_z, item_recency_days, item_counts


def compute_user_item_ratings(train_df, id_maps):
    """Compute mean ratings per user and per item."""
    logger.info("Computing user and item mean ratings")
    
    user_to_idx = id_maps['user_to_idx']
    item_to_idx = id_maps['item_to_idx']
    n_users = len(user_to_idx)
    n_items = len(item_to_idx)
    
    user_rating_sums = np.zeros(n_users)
    user_rating_counts = np.zeros(n_users)
    item_rating_sums = np.zeros(n_items)
    item_rating_counts = np.zeros(n_items)
    
    for _, row in train_df.iterrows():
        user = row['user_id']
        item = row['course_id']
        rating = row.get('rating', 1.0) if 'rating' in row else 1.0
        
        if user in user_to_idx:
            user_idx = user_to_idx[user]
            user_rating_sums[user_idx] += rating
            user_rating_counts[user_idx] += 1
        
        if item in item_to_idx:
            item_idx = item_to_idx[item]
            item_rating_sums[item_idx] += rating
            item_rating_counts[item_idx] += 1
    
    # Compute means (with fallback to global mean)
    global_mean = train_df.get('rating', pd.Series([1.0])).mean() if 'rating' in train_df else 1.0
    
    user_mean_ratings = np.where(
        user_rating_counts > 0,
        user_rating_sums / user_rating_counts,
        global_mean
    )
    
    item_mean_ratings = np.where(
        item_rating_counts > 0,
        item_rating_sums / item_rating_counts,
        global_mean
    )
    
    logger.info(f"User mean rating range: [{user_mean_ratings.min():.2f}, {user_mean_ratings.max():.2f}]")
    logger.info(f"Item mean rating range: [{item_mean_ratings.min():.2f}, {item_mean_ratings.max():.2f}]")
    
    return user_mean_ratings, item_mean_ratings


def sample_negatives(train_df, id_maps, neg_per_pos, rng):
    """Sample negative items for each user."""
    logger.info(f"Sampling {neg_per_pos} negative items per positive")
    
    user_to_idx = id_maps['user_to_idx']
    item_to_idx = id_maps['item_to_idx']
    n_items = len(item_to_idx)
    
    # Get positive items per user
    user_positive_items = {}
    for _, row in train_df.iterrows():
        user = row['user_id']
        item = row['course_id']
        
        if user in user_to_idx and item in item_to_idx:
            user_idx = user_to_idx[user]
            item_idx = item_to_idx[item]
            
            if user_idx not in user_positive_items:
                user_positive_items[user_idx] = set()
            user_positive_items[user_idx].add(item_idx)
    
    # Sample negatives
    negative_pairs = []
    
    for user_idx, pos_items in tqdm(user_positive_items.items(), desc="Sampling negatives"):
        n_pos = len(pos_items)
        n_neg_needed = n_pos * neg_per_pos
        
        # Available items (not in positive set)
        available_items = list(set(range(n_items)) - pos_items)
        
        if len(available_items) < n_neg_needed:
            # Sample with replacement if not enough items
            neg_items = rng.choice(available_items, size=n_neg_needed, replace=True)
        else:
            # Sample without replacement
            neg_items = rng.choice(available_items, size=n_neg_needed, replace=False)
        
        for item_idx in neg_items:
            negative_pairs.append((user_idx, item_idx))
    
    logger.info(f"Sampled {len(negative_pairs)} negative pairs")
    
    return negative_pairs


def build_training_features(train_df, negative_pairs, user_profiles, X_items, 
                           item_popularity_z, item_recency_days,
                           user_mean_ratings, item_mean_ratings, id_maps):
    """Build feature matrix for training."""
    logger.info("Building training feature matrix")
    
    user_to_idx = id_maps['user_to_idx']
    item_to_idx = id_maps['item_to_idx']
    
    # Normalize user profiles and item vectors for cosine computation
    logger.info("Normalizing user profiles and item vectors for content cosine feature")
    user_profiles_norm = normalize(user_profiles, norm='l2', axis=1)
    X_items_norm = normalize(X_items, norm='l2', axis=1)
    
    # Collect positive pairs
    positive_pairs = []
    for _, row in train_df.iterrows():
        user = row['user_id']
        item = row['course_id']
        
        if user in user_to_idx and item in item_to_idx:
            user_idx = user_to_idx[user]
            item_idx = item_to_idx[item]
            positive_pairs.append((user_idx, item_idx))
    
    logger.info(f"Building features for {len(positive_pairs)} positives and {len(negative_pairs)} negatives")
    
    # Combine pairs and labels
    all_pairs = positive_pairs + negative_pairs
    labels = np.array([1] * len(positive_pairs) + [0] * len(negative_pairs))
    
    # Build feature matrix
    user_features_list = []
    item_features_list = []
    scalar_features_list = []
    
    for user_idx, item_idx in tqdm(all_pairs, desc="Extracting features"):
        # User profile (sparse)
        user_features_list.append(user_profiles[user_idx])
        
        # Item vector (sparse)
        item_features_list.append(X_items[item_idx])
        
        # Compute content cosine score (dot product of normalized vectors)
        user_norm = user_profiles_norm[user_idx]
        item_norm = X_items_norm[item_idx]
        content_cosine = user_norm.dot(item_norm.T).toarray()[0, 0]
        
        # Scalar features (including content_cosine)
        scalar_features = [
            item_popularity_z[item_idx],
            -item_recency_days[item_idx],  # Negative so recent is positive
            user_mean_ratings[user_idx],
            item_mean_ratings[item_idx],
            content_cosine  # New feature
        ]
        scalar_features_list.append(scalar_features)
    
    # Stack user profiles
    user_features_sparse = csr_matrix(np.vstack([f.toarray() for f in user_features_list]))
    
    # Stack item vectors
    item_features_sparse = csr_matrix(np.vstack([f.toarray() for f in item_features_list]))
    
    # Stack scalar features
    scalar_features_array = np.array(scalar_features_list)
    
    # Standardize scalar features (including content_cosine)
    scaler = StandardScaler()
    scalar_features_scaled = scaler.fit_transform(scalar_features_array)
    scalar_features_sparse = csr_matrix(scalar_features_scaled)
    
    # Combine all features
    X = hstack([user_features_sparse, item_features_sparse, scalar_features_sparse])
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Feature matrix sparsity: {100 * (1 - X.nnz / (X.shape[0] * X.shape[1])):.2f}%")
    
    feature_dims = {
        'user_profile_dim': user_features_sparse.shape[1],
        'item_vector_dim': item_features_sparse.shape[1],
        'scalar_features_dim': scalar_features_sparse.shape[1]
    }
    
    return X, labels, scaler, feature_dims


def train_ranker(X, y, random_state=42):
    """Train SGD classifier ranker."""
    logger.info("Training SGD classifier ranker")
    
    model = SGDClassifier(
        loss='log_loss',
        alpha=1e-4,
        max_iter=1000,
        class_weight='balanced',
        random_state=random_state,
        verbose=0
    )
    
    model.fit(X, y)
    
    logger.info(f"Training completed. Iterations: {model.n_iter_}")
    logger.info(f"Training accuracy: {model.score(X, y):.4f}")
    
    return model


def get_user_train_items(train_df, id_maps):
    """Get set of item indices each user interacted with."""
    from collections import defaultdict
    
    user_items = defaultdict(set)
    
    for _, row in train_df.iterrows():
        user = row['user_id']
        item = row['course_id']
        
        if user in id_maps['user_to_idx'] and item in id_maps['item_to_idx']:
            user_idx = id_maps['user_to_idx'][user]
            item_idx = id_maps['item_to_idx'][item]
            user_items[user_idx].add(item_idx)
    
    return user_items


def predict_scores(model, user_profiles, X_items, item_popularity_z, item_recency_days,
                   user_mean_ratings, item_mean_ratings, scaler, user_idx, item_indices):
    """Predict scores for a set of items for a given user."""
    if len(item_indices) == 0:
        return np.array([])
    
    # Normalize for content cosine computation
    user_profiles_norm = normalize(user_profiles, norm='l2', axis=1)
    X_items_norm = normalize(X_items, norm='l2', axis=1)
    
    # Build features for all items
    user_profile = user_profiles[user_idx]
    user_profile_norm = user_profiles_norm[user_idx]
    
    # Replicate user profile for all items
    user_features = csr_matrix(np.tile(user_profile.toarray(), (len(item_indices), 1)))
    
    # Stack item vectors
    item_features = X_items[item_indices]
    item_features_norm = X_items_norm[item_indices]
    
    # Compute content cosine scores (batched)
    content_cosines = user_profile_norm.dot(item_features_norm.T).toarray().flatten()
    
    # Scalar features (including content_cosine)
    scalar_features = np.array([
        [item_popularity_z[idx], -item_recency_days[idx], 
         user_mean_ratings[user_idx], item_mean_ratings[idx], content_cosines[i]]
        for i, idx in enumerate(item_indices)
    ])
    
    scalar_features_scaled = scaler.transform(scalar_features)
    scalar_features_sparse = csr_matrix(scalar_features_scaled)
    
    # Combine features
    X = hstack([user_features, item_features, scalar_features_sparse])
    
    # Predict probabilities
    scores = model.predict_proba(X)[:, 1]  # Probability of class 1
    
    return scores


def evaluate_ranker(model, test_df, id_maps, user_train_items, user_profiles, X_items,
                   item_popularity_z, item_recency_days, user_mean_ratings, item_mean_ratings,
                   scaler, k=10):
    """Evaluate ranker on test set."""
    logger.info("Evaluating ranker on test set")
    
    n_items = X_items.shape[0]
    
    precisions = []
    recalls = []
    ndcgs = []
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
        user = row['user_id']
        true_item = row['course_id']
        
        if user not in id_maps['user_to_idx'] or true_item not in id_maps['item_to_idx']:
            continue
        
        user_idx = id_maps['user_to_idx'][user]
        true_item_idx = id_maps['item_to_idx'][true_item]
        
        # Get seen items
        seen_items = user_train_items.get(user_idx, set())
        
        if true_item_idx in seen_items:
            continue
        
        # Get candidate items (all except seen)
        candidate_items = list(set(range(n_items)) - seen_items)
        
        if len(candidate_items) == 0:
            continue
        
        # Predict scores
        try:
            scores = predict_scores(
                model, user_profiles, X_items, item_popularity_z, item_recency_days,
                user_mean_ratings, item_mean_ratings, scaler, user_idx, candidate_items
            )
        except Exception as e:
            logger.warning(f"Error predicting for user {user}: {e}")
            continue
        
        # Get top-K
        if len(scores) < k:
            top_k_local_idx = np.argsort(scores)[::-1]
        else:
            top_k_local_idx = np.argpartition(scores, -k)[-k:]
            top_k_local_idx = top_k_local_idx[np.argsort(scores[top_k_local_idx])[::-1]]
        
        top_k_items = [candidate_items[i] for i in top_k_local_idx]
        
        # Metrics
        hit = 1 if true_item_idx in top_k_items else 0
        precisions.append(hit / min(k, len(top_k_items)))
        recalls.append(hit)
        
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
    
    logger.info(f"SupervisedRanker+ContentCosine - P@{k}: {metrics[f'Precision@{k}']:.4f}, "
                f"R@{k}: {metrics[f'Recall@{k}']:.4f}, "
                f"NDCG@{k}: {metrics[f'NDCG@{k}']:.4f}")
    
    return metrics


def print_metrics_table(metrics, k):
    """Print formatted metrics table."""
    print("\n" + "="*80)
    print(f"{'Model':<35} {'P@'+str(k):<15} {'R@'+str(k):<15} {'NDCG@'+str(k):<15}")
    print("="*80)
    
    p = metrics.get(f'Precision@{k}', 0.0)
    r = metrics.get(f'Recall@{k}', 0.0)
    n = metrics.get(f'NDCG@{k}', 0.0)
    
    print(f"{'SupervisedRanker+ContentCosine':<35} {p:<15.4f} {r:<15.4f} {n:<15.4f}")
    print("="*80 + "\n")


def save_artifacts(out_dir, model, scaler, feature_dims):
    """Save trained model and metadata."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving artifacts to {out_dir}")
    
    # Save model
    joblib.dump(model, out_path / "supervised_ranker.pkl")
    logger.info("Saved ranker model to supervised_ranker.pkl")
    
    # Save feature metadata
    feature_meta = {
        'feature_order': [
            'user_profile',
            'item_vector',
            'item_popularity_z',
            'item_recency_days (negative)',
            'user_mean_rating',
            'item_mean_rating',
            'content_cosine'  # New feature
        ],
        'feature_dims': feature_dims,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'model_type': 'SGDClassifier',
        'model_params': {
            'loss': 'log_loss',
            'alpha': 1e-4,
            'max_iter': 1000,
            'class_weight': 'balanced',
            'random_state': 42
        }
    }
    
    with open(out_path / "feature_meta.json", 'w') as f:
        json.dump(feature_meta, f, indent=2)
    
    logger.info("Saved feature metadata to feature_meta.json")
    
    # Save scaler separately for inference
    joblib.dump(scaler, out_path / "feature_scaler.pkl")
    logger.info("Saved feature scaler to feature_scaler.pkl")


def main():
    """Main training pipeline."""
    args = parse_args()
    
    logger.info("="*80)
    logger.info("Supervised SGD Classifier Ranker Training (with Content Cosine)")
    logger.info("="*80)
    logger.info(f"Catalog:       {args.catalog}")
    logger.info(f"Interactions:  {args.interactions}")
    logger.info(f"Neg per pos:   {args.neg_per_pos}")
    logger.info(f"Top-K:         {args.k}")
    logger.info(f"Output:        {args.out}")
    logger.info(f"Random seed:   {args.seed}")
    logger.info("="*80 + "\n")
    
    # Set random seed
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    
    # Load artifacts
    split_info, id_maps, tfidf_vectorizer, X_items = load_artifacts(args.out)
    
    # Load split data
    train_df, test_df = load_split_data(args.interactions, split_info)
    
    if len(test_df) == 0:
        logger.error("No test data available")
        sys.exit(1)
    
    # Build user profiles
    user_profiles = build_user_profiles(train_df, X_items, id_maps)
    
    # Compute item statistics
    item_popularity_z, item_recency_days, item_counts = compute_item_statistics(train_df, id_maps)
    
    # Compute user/item mean ratings
    user_mean_ratings, item_mean_ratings = compute_user_item_ratings(train_df, id_maps)
    
    # Sample negative pairs
    negative_pairs = sample_negatives(train_df, id_maps, args.neg_per_pos, rng)
    
    # Build training features
    X, y, scaler, feature_dims = build_training_features(
        train_df, negative_pairs, user_profiles, X_items,
        item_popularity_z, item_recency_days,
        user_mean_ratings, item_mean_ratings, id_maps
    )
    
    # Train ranker
    logger.info("\n" + "="*80)
    logger.info("MODEL TRAINING")
    logger.info("="*80)
    
    model = train_ranker(X, y, random_state=args.seed)
    
    # Evaluate ranker
    logger.info("\n" + "="*80)
    logger.info("MODEL EVALUATION")
    logger.info("="*80)
    
    user_train_items = get_user_train_items(train_df, id_maps)
    
    metrics = evaluate_ranker(
        model, test_df, id_maps, user_train_items, user_profiles, X_items,
        item_popularity_z, item_recency_days, user_mean_ratings, item_mean_ratings,
        scaler, k=args.k
    )
    
    # Print results
    print_metrics_table(metrics, args.k)
    
    # Save artifacts
    save_artifacts(args.out, model, scaler, feature_dims)
    
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
