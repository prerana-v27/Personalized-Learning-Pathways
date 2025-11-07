#!/usr/bin/env python3
"""
Content-Based and Clustering Recommender Training
Builds content and cluster-based models using the same split as CF models.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
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
        description="Train content-based and clustering recommenders"
    )
    parser.add_argument("--catalog", required=True, help="Path to course catalog CSV")
    parser.add_argument("--interactions", required=True, help="Path to interactions CSV")
    parser.add_argument("--k", type=int, default=10, help="Top-K for metrics")
    parser.add_argument("--out", default="artifacts", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_or_create_split(interactions_path, out_dir):
    """Load existing split or create new one."""
    split_path = Path(out_dir) / "split.pkl"
    id_maps_path = Path(out_dir) / "id_maps.pkl"
    
    if split_path.exists() and id_maps_path.exists():
        logger.info(f"Loading existing split from {split_path}")
        split_info = joblib.load(split_path)
        id_maps = joblib.load(id_maps_path)
        
        # Load interactions and recreate train/test dataframes
        df = pd.read_csv(interactions_path)
        
        # Parse timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
        
        # Split based on stored info
        train_rows = []
        test_rows = []
        
        for user, group in df.groupby('user_id'):
            if len(group) == 1:
                train_rows.append(group.iloc[0])
            else:
                train_rows.extend(group.iloc[:-1].to_dict('records'))
                test_rows.append(group.iloc[-1])
        
        train_df = pd.DataFrame(train_rows)
        test_df = pd.DataFrame(test_rows) if test_rows else pd.DataFrame()
        
        logger.info(f"Loaded split: {len(train_df)} train, {len(test_df)} test")
        
        return train_df, test_df, id_maps, split_info.get('single_interaction_users', set())
    else:
        logger.info("Creating new split from interactions")
        return create_split(interactions_path, out_dir)


def create_split(interactions_path, out_dir):
    """Create time-aware per-user split."""
    logger.info(f"Loading interactions from {interactions_path}")
    
    try:
        df = pd.read_csv(interactions_path)
    except FileNotFoundError:
        logger.error(f"Interactions file not found: {interactions_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading interactions: {e}")
        sys.exit(1)
    
    if len(df) == 0:
        logger.error("Interactions file is empty")
        sys.exit(1)
    
    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} interactions")
    
    # Temporal split
    logger.info("Performing per-user temporal split")
    train_rows = []
    test_rows = []
    single_interaction_users = set()
    
    for user, group in tqdm(df.groupby('user_id'), desc="Splitting users"):
        if len(group) == 1:
            train_rows.append(group.iloc[0])
            single_interaction_users.add(user)
        else:
            train_rows.extend(group.iloc[:-1].to_dict('records'))
            test_rows.append(group.iloc[-1])
    
    train_df = pd.DataFrame(train_rows)
    test_df = pd.DataFrame(test_rows) if test_rows else pd.DataFrame()
    
    logger.info(f"Train: {len(train_df)} interactions")
    logger.info(f"Test: {len(test_df)} interactions")
    
    # Create ID maps
    users = sorted(df['user_id'].unique())
    items = sorted(df['course_id'].unique())
    
    id_maps = {
        'user_to_idx': {u: i for i, u in enumerate(users)},
        'item_to_idx': {it: i for i, it in enumerate(items)},
        'idx_to_user': {i: u for u, i in enumerate(users)},
        'idx_to_item': {i: it for it, i in enumerate(items)}
    }
    
    # Save split and ID maps
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    split_info = {
        'train_size': len(train_df),
        'test_size': len(test_df),
        'single_interaction_users': single_interaction_users
    }
    
    joblib.dump(split_info, Path(out_dir) / "split.pkl")
    joblib.dump(id_maps, Path(out_dir) / "id_maps.pkl")
    
    logger.info("Saved split and ID maps")
    
    return train_df, test_df, id_maps, single_interaction_users


def load_and_prepare_catalog(catalog_path, id_maps):
    """Load catalog and align with ID maps."""
    logger.info(f"Loading catalog from {catalog_path}")
    
    try:
        catalog_df = pd.read_csv(catalog_path)
    except FileNotFoundError:
        logger.error(f"Catalog file not found: {catalog_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading catalog: {e}")
        sys.exit(1)
    
    if len(catalog_df) == 0:
        logger.error("Catalog is empty")
        sys.exit(1)
    
    logger.info(f"Loaded {len(catalog_df)} courses from catalog")
    
    # Ensure course_id column
    if 'course_id' not in catalog_df.columns:
        logger.info("Creating course_id column")
        catalog_df['course_id'] = ['c' + str(i).zfill(6) for i in range(len(catalog_df))]
    
    # Filter to items in ID maps and align order
    item_to_idx = id_maps['item_to_idx']
    idx_to_item = id_maps['idx_to_item']
    
    # Create mapping from course_id to catalog row
    catalog_dict = {row['course_id']: row for _, row in catalog_df.iterrows()}
    
    # Align catalog to ID maps order
    aligned_rows = []
    missing_items = []
    
    for idx in sorted(idx_to_item.keys()):
        item_id = idx_to_item[idx]
        if item_id in catalog_dict:
            aligned_rows.append(catalog_dict[item_id])
        else:
            # Create placeholder for missing items
            missing_items.append(item_id)
            aligned_rows.append({
                'course_id': item_id,
                'title': '',
                'description': '',
                'skills': '',
                'subject': ''
            })
    
    if missing_items:
        logger.warning(f"Found {len(missing_items)} items in interactions but not in catalog")
    
    aligned_catalog = pd.DataFrame(aligned_rows)
    
    logger.info(f"Aligned catalog: {len(aligned_catalog)} items")
    
    return aligned_catalog


def build_tfidf_matrix(catalog_df, out_dir):
    """Build TF-IDF matrix from catalog."""
    logger.info("Building TF-IDF matrix from catalog")
    
    # Combine text fields
    text_fields = []
    for _, row in catalog_df.iterrows():
        parts = []
        
        for field in ['title', 'description', 'skills', 'subject']:
            if field in row and pd.notna(row[field]):
                parts.append(str(row[field]))
        
        text = ". ".join(parts) if parts else ""
        text_fields.append(text)
    
    # Build TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english',
        norm='l2',
        min_df=1,
        max_df=0.95
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(text_fields)
        logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        logger.info(f"TF-IDF sparsity: {100 * (1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])):.2f}%")
    except Exception as e:
        logger.error(f"Error building TF-IDF matrix: {e}")
        sys.exit(1)
    
    # Save TF-IDF artifacts
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(vectorizer, out_path / "tfidf.pkl")
    save_npz(out_path / "X_items.npz", tfidf_matrix)
    
    logger.info("Saved TF-IDF vectorizer and item matrix")
    
    return tfidf_matrix, vectorizer


def compute_item_stats(train_df, id_maps):
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
    
    logger.info(f"Item popularity range: [{item_counts.min():.0f}, {item_counts.max():.0f}]")
    logger.info(f"Item recency range: [{item_recency_days.min():.0f}, {item_recency_days.max():.0f}] days")
    
    return item_counts, item_recency_days


def build_user_profiles(train_df, tfidf_matrix, id_maps):
    """Build user profiles as mean of TF-IDF vectors of seen items."""
    logger.info("Building user profiles from training interactions")
    
    user_to_idx = id_maps['user_to_idx']
    item_to_idx = id_maps['item_to_idx']
    n_users = len(user_to_idx)
    n_features = tfidf_matrix.shape[1]
    
    # Build sparse user profile matrix
    user_profiles = np.zeros((n_users, n_features))
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
        
        # Add weighted item vector to user profile
        user_profiles[user_idx] += weight * tfidf_matrix[item_idx].toarray().flatten()
        user_counts[user_idx] += weight
    
    # Normalize by count to get mean
    for i in range(n_users):
        if user_counts[i] > 0:
            user_profiles[i] /= user_counts[i]
    
    # Convert to sparse matrix
    user_profiles_sparse = csr_matrix(user_profiles)
    
    logger.info(f"User profiles shape: {user_profiles_sparse.shape}")
    logger.info(f"Users with profiles: {(user_counts > 0).sum()}/{n_users}")
    
    return user_profiles_sparse


class ContentRecommender:
    """Content-based recommender using cosine similarity."""
    
    def __init__(self, user_profiles, item_vectors):
        self.user_profiles = user_profiles
        self.item_vectors = item_vectors
        
        # Pre-compute item norms for efficiency
        self.item_norms = normalize(item_vectors, norm='l2', axis=1)
    
    def recommend(self, user_idx, seen_items, k=10):
        """Recommend top-K items for user based on content similarity."""
        # Get user profile
        user_profile = self.user_profiles[user_idx].toarray().flatten()
        
        # If user has no profile, return empty
        if np.linalg.norm(user_profile) == 0:
            return []
        
        # Normalize user profile
        user_profile_norm = user_profile / np.linalg.norm(user_profile)
        
        # Compute cosine similarity with all items
        scores = self.item_norms.dot(user_profile_norm)
        
        # Mask out seen items
        scores[list(seen_items)] = -np.inf
        
        # Get top-K
        if k > len(scores):
            k = len(scores)
        
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
        
        return top_k_idx.tolist()


class ClusterRecommender:
    """Clustering-based recommender using KMeans."""
    
    def __init__(self, item_vectors, item_popularity, n_clusters=50, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.item_vectors = item_vectors
        self.item_popularity = item_popularity
        
        # Normalize item vectors
        self.item_vectors_norm = normalize(item_vectors, norm='l2', axis=1)
        
        # Fit KMeans
        logger.info(f"Fitting KMeans with {n_clusters} clusters")
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            max_iter=300
        )
        self.kmeans.fit(self.item_vectors_norm)
        
        # Normalize centroids
        self.centroids_norm = normalize(self.kmeans.cluster_centers_, norm='l2', axis=1)
        
        # Standardize popularity for scoring
        pop_mean = self.item_popularity.mean()
        pop_std = self.item_popularity.std()
        if pop_std > 0:
            self.popularity_z = (self.item_popularity - pop_mean) / pop_std
        else:
            self.popularity_z = np.zeros_like(self.item_popularity)
        
        logger.info(f"KMeans fitted. Inertia: {self.kmeans.inertia_:.2f}")
    
    def recommend(self, user_idx, seen_items, user_profile, k=10):
        """Recommend top-K items based on cluster similarity and popularity."""
        # If user has no profile, fall back to popularity
        if np.linalg.norm(user_profile) == 0:
            # Return top popular items not in seen
            pop_scores = self.item_popularity.copy()
            pop_scores[list(seen_items)] = -np.inf
            
            top_k_idx = np.argpartition(pop_scores, -k)[-k:]
            top_k_idx = top_k_idx[np.argsort(pop_scores[top_k_idx])[::-1]]
            return top_k_idx.tolist()
        
        # Normalize user profile
        user_profile_norm = user_profile / np.linalg.norm(user_profile)
        
        # Find closest centroid to user profile
        centroid_sims = self.centroids_norm.dot(user_profile_norm)
        
        # Compute score for each item: centroid_sim × (popularity_z + 1)
        item_clusters = self.kmeans.labels_
        item_centroid_sims = centroid_sims[item_clusters]
        
        scores = item_centroid_sims * (self.popularity_z + 1)
        
        # Mask out seen items
        scores[list(seen_items)] = -np.inf
        
        # Get top-K
        if k > len(scores):
            k = len(scores)
        
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
        
        return top_k_idx.tolist()


def select_best_k_clusters(tfidf_matrix, train_df, id_maps, item_popularity, k_values=[50, 100]):
    """Select best number of clusters using validation NDCG@10."""
    logger.info("Selecting best K for KMeans using validation set")
    
    # Create validation split from training data
    val_rows = []
    train_val_rows = []
    
    df_temp = train_df.copy()
    df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp'])
    df_temp = df_temp.sort_values(['user_id', 'timestamp'])
    
    for user, group in df_temp.groupby('user_id'):
        if len(group) > 1:
            train_val_rows.extend(group.iloc[:-1].to_dict('records'))
            val_rows.append(group.iloc[-1])
        else:
            train_val_rows.append(group.iloc[0])
    
    train_val_df = pd.DataFrame(train_val_rows)
    val_df = pd.DataFrame(val_rows)
    
    logger.info(f"Validation split: {len(train_val_df)} train, {len(val_df)} val")
    
    if len(val_df) == 0:
        logger.warning("No validation data, using K=50")
        return 50
    
    # Build user profiles from train_val
    user_profiles = build_user_profiles(train_val_df, tfidf_matrix, id_maps)
    
    # Get user train items
    user_train_items = get_user_train_items(train_val_df, id_maps)
    
    best_k = k_values[0]
    best_ndcg = -1
    
    for k_clusters in k_values:
        logger.info(f"Evaluating K={k_clusters}")
        
        # Train cluster model
        cluster_model = ClusterRecommender(
            tfidf_matrix, item_popularity, n_clusters=k_clusters, random_state=42
        )
        
        # Evaluate on validation set
        ndcgs = []
        
        for _, row in val_df.iterrows():
            user = row['user_id']
            true_item = row['course_id']
            
            if user not in id_maps['user_to_idx'] or true_item not in id_maps['item_to_idx']:
                continue
            
            user_idx = id_maps['user_to_idx'][user]
            true_item_idx = id_maps['item_to_idx'][true_item]
            
            seen_items = user_train_items.get(user_idx, set())
            
            if true_item_idx in seen_items:
                continue
            
            # Get user profile
            user_profile = user_profiles[user_idx].toarray().flatten()
            
            # Get recommendations
            top_k_items = cluster_model.recommend(user_idx, seen_items, user_profile, k=10)
            
            # Compute NDCG
            if true_item_idx in top_k_items:
                rank = top_k_items.index(true_item_idx) + 1
                ndcg = 1.0 / np.log2(rank + 1)
            else:
                ndcg = 0.0
            
            ndcgs.append(ndcg)
        
        avg_ndcg = np.mean(ndcgs) if ndcgs else 0.0
        logger.info(f"K={k_clusters}: Validation NDCG@10 = {avg_ndcg:.4f}")
        
        if avg_ndcg > best_ndcg:
            best_ndcg = avg_ndcg
            best_k = k_clusters
    
    logger.info(f"Selected K={best_k} (NDCG@10={best_ndcg:.4f})")
    
    return best_k


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


def evaluate_recommender(model, test_df, id_maps, user_train_items, user_profiles=None, k=10, model_name="Model"):
    """Evaluate recommender with top-K metrics."""
    logger.info(f"Evaluating {model_name}")
    
    precisions = []
    recalls = []
    ndcgs = []
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Evaluating {model_name}"):
        user = row['user_id']
        true_item = row['course_id']
        
        if user not in id_maps['user_to_idx'] or true_item not in id_maps['item_to_idx']:
            continue
        
        user_idx = id_maps['user_to_idx'][user]
        true_item_idx = id_maps['item_to_idx'][true_item]
        
        seen_items = user_train_items.get(user_idx, set())
        
        if true_item_idx in seen_items:
            continue
        
        # Get recommendations
        try:
            if isinstance(model, ClusterRecommender):
                # Need user profile for cluster model
                user_profile = user_profiles[user_idx].toarray().flatten() if user_profiles is not None else np.zeros(model.item_vectors.shape[1])
                top_k_items = model.recommend(user_idx, seen_items, user_profile, k=k)
            else:
                top_k_items = model.recommend(user_idx, seen_items, k=k)
        except Exception as e:
            logger.warning(f"Error generating recommendations: {e}")
            continue
        
        # Metrics
        hit = 1 if true_item_idx in top_k_items else 0
        precisions.append(hit / min(k, len(top_k_items)) if top_k_items else 0)
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
    
    logger.info(f"{model_name} - P@{k}: {metrics[f'Precision@{k}']:.4f}, "
                f"R@{k}: {metrics[f'Recall@{k}']:.4f}, "
                f"NDCG@{k}: {metrics[f'NDCG@{k}']:.4f}")
    
    return metrics


def print_metrics_table(results, k):
    """Print formatted metrics table."""
    print("\n" + "="*80)
    print(f"{'Model':<20} {'P@'+str(k):<15} {'R@'+str(k):<15} {'NDCG@'+str(k):<15}")
    print("="*80)
    
    for model_name, metrics in results.items():
        p = metrics.get(f'Precision@{k}', 0.0)
        r = metrics.get(f'Recall@{k}', 0.0)
        n = metrics.get(f'NDCG@{k}', 0.0)
        
        print(f"{model_name:<20} {p:<15.4f} {r:<15.4f} {n:<15.4f}")
    
    print("="*80 + "\n")


def save_artifacts(out_dir, cluster_model, content_index, catalog_df, vectorizer):
    """Save model artifacts."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving artifacts to {out_dir}")
    
    # Save cluster model
    if cluster_model is not None:
        joblib.dump(cluster_model, out_path / "kmeans.pkl")
        logger.info("Saved KMeans model to kmeans.pkl")
    
    # Save content index (cached norms)
    joblib.dump(content_index, out_path / "content_index.pkl")
    logger.info("Saved content index to content_index.pkl")
    
    # Save metadata
    meta = {
        'columns': ['title', 'description', 'skills', 'subject'],
        'tfidf_params': {
            'max_features': 5000,
            'ngram_range': (1, 2),
            'stop_words': 'english',
            'norm': 'l2'
        },
        'n_items': len(catalog_df)
    }
    
    with open(out_path / "content_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)
    
    logger.info("Saved content metadata to content_meta.json")


def main():
    """Main training pipeline."""
    args = parse_args()
    
    logger.info("="*80)
    logger.info("Content-Based and Clustering Recommender Training")
    logger.info("="*80)
    logger.info(f"Catalog:       {args.catalog}")
    logger.info(f"Interactions:  {args.interactions}")
    logger.info(f"Top-K:         {args.k}")
    logger.info(f"Output:        {args.out}")
    logger.info(f"Random seed:   {args.seed}")
    logger.info("="*80 + "\n")
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Load or create split
    train_df, test_df, id_maps, single_users = load_or_create_split(args.interactions, args.out)
    
    if len(test_df) == 0:
        logger.error("No test data available")
        sys.exit(1)
    
    # Load and prepare catalog
    catalog_df = load_and_prepare_catalog(args.catalog, id_maps)
    
    # Build TF-IDF matrix
    tfidf_matrix, vectorizer = build_tfidf_matrix(catalog_df, args.out)
    
    # Compute item statistics
    item_popularity, item_recency = compute_item_stats(train_df, id_maps)
    
    # Build user profiles
    user_profiles = build_user_profiles(train_df, tfidf_matrix, id_maps)
    
    # Get user train items
    user_train_items = get_user_train_items(train_df, id_maps)
    
    # Train models
    logger.info("\n" + "="*80)
    logger.info("MODEL TRAINING")
    logger.info("="*80)
    
    # Content-based recommender
    logger.info("Building Content-based recommender")
    content_model = ContentRecommender(user_profiles, tfidf_matrix)
    
    # Cluster-based recommender
    logger.info("Building Cluster-based recommender")
    best_k = select_best_k_clusters(tfidf_matrix, train_df, id_maps, item_popularity, k_values=[50, 100])
    cluster_model = ClusterRecommender(tfidf_matrix, item_popularity, n_clusters=best_k, random_state=args.seed)
    
    # Evaluate models
    logger.info("\n" + "="*80)
    logger.info("MODEL EVALUATION")
    logger.info("="*80)
    
    results = {}
    
    # Content-based
    results['Content'] = evaluate_recommender(
        content_model, test_df, id_maps, user_train_items,
        user_profiles=None, k=args.k, model_name="Content"
    )
    
    # Cluster-based
    results['Cluster'] = evaluate_recommender(
        cluster_model, test_df, id_maps, user_train_items,
        user_profiles=user_profiles, k=args.k, model_name="Cluster"
    )
    
    # Print results table
    print_metrics_table(results, args.k)
    
    # Save artifacts
    content_index = {
        'item_norms': content_model.item_norms,
        'user_profiles': user_profiles
    }
    
    save_artifacts(args.out, cluster_model, content_index, catalog_df, vectorizer)
    
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
