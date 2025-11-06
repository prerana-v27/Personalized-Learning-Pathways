"""
Hybrid Ensemble Training - Blends all trained models with optimized weights

This script:
1. Loads all trained models (NMF, KNN, Supervised, Content, Cluster, Popularity)
2. Creates a validation split from training data for weight optimization
3. Grid-searches ensemble weights for warm users (≥10 train interactions)
4. Uses fixed weights for cold users (0.7*Content + 0.3*Popularity)
5. Evaluates on official test split with warm/cold breakdown
6. Saves ensemble configuration and performance report

Usage:
    python training/hybrid_train.py \
        --catalog Data/processed/all_courses_cleaned.csv \
        --interactions Data/processed/interactions_synth.csv \
        --k 10 --min-warm 10 --out artifacts

Author: Generated for Personalized Learning Pathways
Date: November 6, 2025
"""

import argparse
import logging
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.metrics import ndcg_score
from tqdm import tqdm

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

np.random.seed(42)


# ========================================
# Dummy Model Classes for Unpickling
# ========================================

class PopularityModel:
    """Dummy class for unpickling popularity model."""
    def __init__(self):
        self.item_popularity = None
        self.popular_items = None

class ItemKNNModel:
    """Dummy class for unpickling item-KNN model."""
    def __init__(self, k=40):
        self.k = k
        self.item_similarity = None
        self.R_train = None

class NMFModel:
    """Dummy class for unpickling NMF model."""
    def __init__(self, n_factors=60, random_state=42, max_iter=200):
        self.n_factors = n_factors
        self.random_state = random_state
        self.max_iter = max_iter
        self.user_factors = None
        self.item_factors = None

class ContentRecommender:
    """Dummy class for unpickling content recommender."""
    pass

class ClusterRecommender:
    """Dummy class for unpickling cluster recommender."""
    pass


# ========================================
# Min-Max Normalization
# ========================================

def minmax_normalize_scores(scores: np.ndarray) -> np.ndarray:
    """
    Min-max normalize scores to [0, 1] range.
    Handles constant scores by returning zeros.
    
    Args:
        scores: Array of scores for a user
        
    Returns:
        Normalized scores in [0, 1]
    """
    min_score = scores.min()
    max_score = scores.max()
    
    if max_score - min_score < 1e-10:
        # All scores are the same - return zeros
        return np.zeros_like(scores)
    
    return (scores - min_score) / (max_score - min_score)


# ========================================
# Model Score Computation
# ========================================

def get_nmf_scores(model, user_id: int, item_indices: np.ndarray, 
                   user_factors: np.ndarray, item_factors: np.ndarray) -> np.ndarray:
    """Get NMF model scores for items."""
    if user_id >= len(user_factors):
        return np.zeros(len(item_indices))
    
    user_vec = user_factors[user_id]
    item_vecs = item_factors[item_indices]
    scores = item_vecs.dot(user_vec)
    return scores


def get_knn_scores(model, user_id: int, item_indices: np.ndarray,
                   R_train: csr_matrix, item_sim: np.ndarray) -> np.ndarray:
    """Get Item-KNN model scores for items."""
    if user_id >= R_train.shape[0]:
        return np.zeros(len(item_indices))
    
    user_ratings = R_train[user_id].toarray().flatten()
    rated_items = np.where(user_ratings > 0)[0]
    
    if len(rated_items) == 0:
        return np.zeros(len(item_indices))
    
    scores = np.zeros(len(item_indices))
    for idx, item_id in enumerate(item_indices):
        if item_id >= item_sim.shape[0]:
            continue
        
        # Get similarities to rated items
        sims = item_sim[item_id, rated_items]
        if np.any(sims > 0):
            # Weighted average of ratings by similarity
            scores[idx] = np.sum(sims * user_ratings[rated_items]) / (np.sum(np.abs(sims)) + 1e-10)
    
    return scores


def get_supervised_scores(model, user_id: int, item_indices: np.ndarray,
                         user_profiles: csr_matrix, X_items: csr_matrix,
                         metadata: Dict, interactions_df: pd.DataFrame,
                         scaler) -> np.ndarray:
    """Get supervised ranker scores for items."""
    from sklearn.preprocessing import normalize
    
    if user_id >= user_profiles.shape[0]:
        return np.zeros(len(item_indices))
    
    # If model or scaler is None, return zeros
    if model is None or scaler is None:
        return np.zeros(len(item_indices))
    
    try:
        # User features
        user_profile = user_profiles[user_id]
        user_interactions = interactions_df[interactions_df['user_id'] == user_id]
        user_mean_rating = user_interactions['rating'].mean() if len(user_interactions) > 0 else 1.0
        
        # Item popularity and recency from metadata
        item_popularity_z = metadata.get('item_popularity_z', {})
        item_recency_days = metadata.get('item_recency_days', {})
        item_mean_rating = metadata.get('item_mean_rating', {})
        
        # Normalize profiles for content cosine
        user_profile_norm = normalize(user_profile, norm='l2')
        X_items_norm = normalize(X_items, norm='l2')
        
        # Build feature matrix
        n_items = len(item_indices)
        n_tfidf = user_profiles.shape[1]
        n_features = 2 * n_tfidf + 5  # user_profile + item_vector + 5 scalars
        
        # Use sparse matrix construction
        features = sparse.lil_matrix((n_items, n_features))
        
        for idx, item_id in enumerate(item_indices):
            if item_id >= X_items.shape[0]:
                continue
            
            # TF-IDF features
            features[idx, :n_tfidf] = user_profile
            features[idx, n_tfidf:2*n_tfidf] = X_items[item_id]
            
            # Scalar features
            pop_z = item_popularity_z.get(item_id, 0.0)
            recency = -item_recency_days.get(item_id, 365.0)
            item_rating = item_mean_rating.get(item_id, 1.0)
            
            # Content cosine
            item_vec_norm = X_items_norm[item_id]
            content_cos = user_profile_norm.dot(item_vec_norm.T).toarray()[0, 0]
            
            features[idx, 2*n_tfidf:2*n_tfidf+5] = [pop_z, recency, user_mean_rating, item_rating, content_cos]
        
        # Convert to CSR for efficient prediction
        features = features.tocsr()
        
        # Scale scalar features
        scalar_features = features[:, 2*n_tfidf:].toarray()
        scalar_features_scaled = scaler.transform(scalar_features)
        
        # Reconstruct feature matrix
        features_scaled = sparse.hstack([
            features[:, :2*n_tfidf],
            csr_matrix(scalar_features_scaled)
        ], format='csr')
        
        # Predict
        scores = model.predict_proba(features_scaled)[:, 1]
        return scores
    except Exception as e:
        # Fallback to zeros if scoring fails
        return np.zeros(len(item_indices))


def get_content_scores(user_id: int, item_indices: np.ndarray,
                      user_profiles: csr_matrix, X_items: csr_matrix) -> np.ndarray:
    """Get content-based scores (cosine similarity)."""
    # user_id is already an integer index (not a string ID)
    if user_id >= user_profiles.shape[0]:
        return np.zeros(len(item_indices))
    
    user_profile = user_profiles[user_id]
    item_vectors = X_items[item_indices]
    
    # Cosine similarity
    scores = item_vectors.dot(user_profile.T).toarray().flatten()
    return scores


def get_cluster_scores(kmeans, user_id: int, item_indices: np.ndarray,
                      user_profiles: csr_matrix, X_items: csr_matrix,
                      item_clusters: np.ndarray) -> np.ndarray:
    """Get cluster-based scores."""
    if user_id >= user_profiles.shape[0]:
        return np.zeros(len(item_indices))
    
    # Predict user's preferred cluster
    user_profile = user_profiles[user_id].toarray()
    user_cluster = kmeans.predict(user_profile)[0]
    
    # Score items by cluster membership
    scores = np.zeros(len(item_indices))
    for idx, item_id in enumerate(item_indices):
        if item_id >= len(item_clusters):
            continue
        
        if item_clusters[item_id] == user_cluster:
            # Items in user's cluster get high score
            scores[idx] = 1.0
        else:
            # Items in other clusters get distance-based score
            item_vec = X_items[item_id].toarray()
            cluster_center = kmeans.cluster_centers_[user_cluster].reshape(1, -1)
            # Negative distance as score
            dist = np.linalg.norm(item_vec - cluster_center)
            scores[idx] = max(0.0, 1.0 - dist / 10.0)
    
    return scores


def get_popularity_scores(item_indices: np.ndarray, 
                         item_popularity: Dict[int, float]) -> np.ndarray:
    """Get popularity-based scores."""
    scores = np.array([item_popularity.get(item_id, 0.0) for item_id in item_indices])
    return scores


# ========================================
# Hybrid Scoring
# ========================================

def hybrid_score_warm_user(user_id: int, item_indices: np.ndarray, 
                          models: Dict, weights: Dict[str, float],
                          normalize: bool = True) -> np.ndarray:
    """
    Compute hybrid scores for warm user using weighted ensemble.
    
    Args:
        user_id: User ID
        item_indices: Array of item IDs to score
        models: Dictionary of loaded models and data
        weights: Dictionary of model weights (nmf, knn, supervised, content)
        normalize: Whether to min-max normalize each model's scores
        
    Returns:
        Hybrid scores for items
    """
    scores_dict = {}
    
    # NMF scores
    if 'nmf' in models and models.get('nmf') and weights.get('nmf', 0) > 0:
        try:
            scores_nmf = get_nmf_scores(
                models['nmf'], user_id, item_indices,
                models['nmf_user_factors'], models['nmf_item_factors']
            )
            if normalize:
                scores_nmf = minmax_normalize_scores(scores_nmf)
            scores_dict['nmf'] = scores_nmf * weights['nmf']
        except Exception as e:
            pass  # Skip this model
    
    # KNN scores
    if 'knn' in models and models.get('knn') and weights.get('knn', 0) > 0:
        try:
            scores_knn = get_knn_scores(
                models['knn'], user_id, item_indices,
                models['R_train'], models['knn_item_sim']
            )
            if normalize:
                scores_knn = minmax_normalize_scores(scores_knn)
            scores_dict['knn'] = scores_knn * weights['knn']
        except Exception as e:
            pass  # Skip this model
    
    # Supervised scores
    if 'supervised' in models and models.get('supervised') and weights.get('supervised', 0) > 0:
        try:
            scores_sup = get_supervised_scores(
                models['supervised'], user_id, item_indices,
                models['user_profiles'], models['X_items'],
                models['supervised_metadata'], models['interactions_df'],
                models['supervised_scaler']
            )
            if normalize:
                scores_sup = minmax_normalize_scores(scores_sup)
            scores_dict['supervised'] = scores_sup * weights['supervised']
        except Exception as e:
            pass  # Skip this model
    
    # Content scores
    if weights.get('content', 0) > 0:
        try:
            scores_content = get_content_scores(
                user_id, item_indices,
                models['user_profiles'], models['X_items']
            )
            if normalize:
                scores_content = minmax_normalize_scores(scores_content)
            scores_dict['content'] = scores_content * weights['content']
        except Exception as e:
            pass  # Skip this model
    
    # Cluster scores (optional)
    if 'kmeans' in models and models.get('kmeans') and weights.get('cluster', 0) > 0:
        try:
            scores_cluster = get_cluster_scores(
                models['kmeans'], user_id, item_indices,
                models['user_profiles'], models['X_items'],
                models['item_clusters']
            )
            if normalize:
                scores_cluster = minmax_normalize_scores(scores_cluster)
            scores_dict['cluster'] = scores_cluster * weights['cluster']
        except Exception as e:
            pass  # Skip this model
    
    # Combine scores
    if len(scores_dict) == 0:
        return np.zeros(len(item_indices))
    
    hybrid_scores = sum(scores_dict.values())
    return hybrid_scores


def hybrid_score_cold_user(user_id: int, item_indices: np.ndarray,
                          models: Dict, recency_boost: float = 0.1) -> np.ndarray:
    """
    Compute hybrid scores for cold user (fixed blend: 0.7*Content + 0.3*Popularity + recency).
    
    Args:
        user_id: User ID
        item_indices: Array of item IDs to score
        models: Dictionary of loaded models and data
        recency_boost: Small boost for recent items
        
    Returns:
        Hybrid scores for items
    """
    # Content scores (70%)
    scores_content = get_content_scores(
        user_id, item_indices,
        models['user_profiles'], models['X_items']
    )
    scores_content = minmax_normalize_scores(scores_content)
    
    # Popularity scores (30%)
    scores_pop = get_popularity_scores(item_indices, models['item_popularity'])
    scores_pop = minmax_normalize_scores(scores_pop)
    
    # Recency boost
    item_recency = models.get('item_recency_days', {})
    recency_scores = np.array([
        recency_boost * max(0, 1.0 - item_recency.get(item_id, 365) / 365.0)
        for item_id in item_indices
    ])
    
    # Combine: 0.7*Content + 0.3*Popularity + recency_boost
    hybrid_scores = 0.7 * scores_content + 0.3 * scores_pop + recency_scores
    return hybrid_scores


# ========================================
# Validation Split Creation
# ========================================

def create_validation_split(interactions_df: pd.DataFrame, 
                           train_pairs: set) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create validation split from training data (last interaction per user).
    
    Args:
        interactions_df: Full interactions dataframe
        train_pairs: Set of (user_id, item_id) training pairs
        
    Returns:
        val_df, train_minus_val_df
    """
    train_df = interactions_df[
        interactions_df.apply(lambda r: (r['user_id'], r['item_id']) in train_pairs, axis=1)
    ].copy()
    
    # Sort by timestamp
    train_df = train_df.sort_values(['user_id', 'timestamp'])
    
    # Last interaction per user -> validation
    val_df = train_df.groupby('user_id').tail(1).reset_index(drop=True)
    
    # Rest -> train_minus_val
    val_pairs = set(zip(val_df['user_id'], val_df['item_id']))
    train_minus_val_df = train_df[
        ~train_df.apply(lambda r: (r['user_id'], r['item_id']) in val_pairs, axis=1)
    ].reset_index(drop=True)
    
    logging.info(f"Validation split: {len(val_df)} interactions from {val_df['user_id'].nunique()} users")
    logging.info(f"Train minus val: {len(train_minus_val_df)} interactions")
    
    return val_df, train_minus_val_df


# ========================================
# Evaluation Metrics
# ========================================

def evaluate_recommendations(test_df: pd.DataFrame, models: Dict, weights: Dict[str, float],
                            min_warm: int, k: int, 
                            train_pairs: set) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Evaluate hybrid model on test set with warm/cold breakdown.
    
    Returns:
        overall_metrics, warm_metrics, cold_metrics
    """
    # Determine warm/cold users
    train_counts = pd.DataFrame(list(train_pairs), columns=['user_id', 'item_id'])
    train_user_counts = train_counts['user_id'].value_counts()
    
    test_users = test_df['user_id'].unique()
    warm_users = set(test_users) & set(train_user_counts[train_user_counts >= min_warm].index)
    cold_users = set(test_users) - warm_users
    
    logging.info(f"Test users: {len(test_users)} (warm: {len(warm_users)}, cold: {len(cold_users)})")
    
    # All items for candidate set
    all_items = np.arange(models['X_items'].shape[0])
    
    # Metrics storage
    metrics = {'overall': [], 'warm': [], 'cold': []}
    
    # Evaluate each user
    for user_id in tqdm(test_users, desc="Evaluating"):
        # Get test items for user
        user_test = test_df[test_df['user_id'] == user_id]
        true_items = set(user_test['item_id'].values)
        
        if len(true_items) == 0:
            continue
        
        # Map user_id to integer index
        if 'user_map' in models and user_id in models['user_map']:
            user_idx = models['user_map'][user_id]
        else:
            user_idx = user_id  # Assume already an integer
        
        # Candidate items (all minus seen in train)
        seen_items = set([item for u, item in train_pairs if u == user_id])
        candidate_items = np.array([item for item in all_items if item not in seen_items])
        
        if len(candidate_items) == 0:
            continue
        
        # Score candidates (use integer index)
        is_warm = user_id in warm_users
        if is_warm:
            scores = hybrid_score_warm_user(user_idx, candidate_items, models, weights)
        else:
            scores = hybrid_score_cold_user(user_idx, candidate_items, models)
        
        # Top-K recommendations
        if len(scores) < k:
            top_k_indices = np.argsort(-scores)
        else:
            top_k_indices = np.argpartition(-scores, k-1)[:k]
            top_k_indices = top_k_indices[np.argsort(-scores[top_k_indices])]
        
        recommended_items = candidate_items[top_k_indices]
        
        # Compute metrics
        hits = len(set(recommended_items) & true_items)
        precision = hits / k
        recall = hits / len(true_items)
        
        # NDCG
        relevance = np.array([1 if item in true_items else 0 for item in recommended_items])
        true_relevance = np.ones(min(k, len(true_items)))
        
        if np.sum(relevance) > 0:
            ndcg = ndcg_score([true_relevance], [relevance], k=k)
        else:
            ndcg = 0.0
        
        # Store metrics
        user_metrics = {'precision': precision, 'recall': recall, 'ndcg': ndcg}
        metrics['overall'].append(user_metrics)
        
        if is_warm:
            metrics['warm'].append(user_metrics)
        else:
            metrics['cold'].append(user_metrics)
    
    # Aggregate metrics
    def aggregate(metric_list):
        if len(metric_list) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'ndcg': 0.0}
        return {
            'precision': np.mean([m['precision'] for m in metric_list]),
            'recall': np.mean([m['recall'] for m in metric_list]),
            'ndcg': np.mean([m['ndcg'] for m in metric_list])
        }
    
    overall_metrics = aggregate(metrics['overall'])
    warm_metrics = aggregate(metrics['warm'])
    cold_metrics = aggregate(metrics['cold'])
    
    return overall_metrics, warm_metrics, cold_metrics


# ========================================
# Weight Optimization
# ========================================

def grid_search_weights(val_df: pd.DataFrame, models: Dict, min_warm: int, k: int,
                       train_minus_val_pairs: set, step: float = 0.25) -> Dict[str, float]:
    """
    Grid search ensemble weights on validation set (warm users only).
    Optimize NDCG@10.
    
    Args:
        val_df: Validation dataframe
        models: Dictionary of loaded models
        min_warm: Minimum interactions for warm user
        k: Top-K for evaluation
        train_minus_val_pairs: Training pairs minus validation
        step: Grid search step size
        
    Returns:
        Best weights dictionary
    """
    logging.info("Starting grid search for ensemble weights...")
    
    # Determine warm validation users
    train_counts = pd.DataFrame(list(train_minus_val_pairs), columns=['user_id', 'item_id'])
    train_user_counts = train_counts['user_id'].value_counts()
    
    val_users = val_df['user_id'].unique()
    warm_val_users = set(val_users) & set(train_user_counts[train_user_counts >= min_warm].index)
    
    logging.info(f"Warm validation users: {len(warm_val_users)} / {len(val_users)}")
    
    if len(warm_val_users) == 0:
        logging.warning("No warm validation users! Using default weights.")
        return {'nmf': 0.25, 'knn': 0.0, 'supervised': 0.50, 'content': 0.25}
    
    # Grid search space (exclude all-zero)
    weight_range = np.arange(0.0, 1.01, step)
    
    # Available models
    available_models = []
    if 'nmf' in models:
        available_models.append('nmf')
    if 'knn' in models:
        available_models.append('knn')
    if 'supervised' in models:
        available_models.append('supervised')
    available_models.append('content')  # Always available
    
    logging.info(f"Available models for ensemble: {available_models}")
    
    # Generate weight combinations
    best_ndcg = -1.0
    best_weights = None
    
    total_combinations = 0
    for w_nmf in weight_range if 'nmf' in available_models else [0.0]:
        for w_knn in weight_range if 'knn' in available_models else [0.0]:
            for w_supervised in weight_range if 'supervised' in available_models else [0.0]:
                for w_content in weight_range:
                    # Skip all-zero
                    if w_nmf + w_knn + w_supervised + w_content < 1e-6:
                        continue
                    
                    total_combinations += 1
    
    logging.info(f"Grid search: {total_combinations} weight combinations")
    
    # All items for candidate set
    all_items = np.arange(models['X_items'].shape[0])
    
    # Evaluate each weight combination
    with tqdm(total=total_combinations, desc="Grid search") as pbar:
        for w_nmf in weight_range if 'nmf' in available_models else [0.0]:
            for w_knn in weight_range if 'knn' in available_models else [0.0]:
                for w_supervised in weight_range if 'supervised' in available_models else [0.0]:
                    for w_content in weight_range:
                        # Skip all-zero
                        if w_nmf + w_knn + w_supervised + w_content < 1e-6:
                            continue
                        
                        weights = {
                            'nmf': w_nmf,
                            'knn': w_knn,
                            'supervised': w_supervised,
                            'content': w_content
                        }
                        
                        # Evaluate on warm validation users
                        ndcg_scores = []
                        
                        for user_id in warm_val_users:
                            # Get validation items
                            user_val = val_df[val_df['user_id'] == user_id]
                            true_items = set(user_val['item_id'].values)
                            
                            if len(true_items) == 0:
                                continue
                            
                            # Map user_id to integer index (user_map is the user_to_idx dictionary)
                            if 'user_map' in models and user_id in models['user_map']:
                                user_idx = models['user_map'][user_id]
                            else:
                                user_idx = user_id  # Assume already an integer
                            
                            # Candidate items
                            seen_items = set([item for u, item in train_minus_val_pairs if u == user_id])
                            candidate_items = np.array([item for item in all_items if item not in seen_items])
                            
                            if len(candidate_items) == 0:
                                continue
                            
                            # Score candidates (use integer index)
                            scores = hybrid_score_warm_user(user_idx, candidate_items, models, weights)
                            
                            # Top-K
                            if len(scores) < k:
                                top_k_indices = np.argsort(-scores)
                            else:
                                top_k_indices = np.argpartition(-scores, k-1)[:k]
                                top_k_indices = top_k_indices[np.argsort(-scores[top_k_indices])]
                            
                            recommended_items = candidate_items[top_k_indices]
                            
                            # NDCG
                            relevance = np.array([1 if item in true_items else 0 for item in recommended_items])
                            if np.sum(relevance) > 0:
                                true_relevance = np.ones(min(k, len(true_items)))
                                ndcg = ndcg_score([true_relevance], [relevance], k=k)
                                ndcg_scores.append(ndcg)
                        
                        # Average NDCG
                        avg_ndcg = np.mean(ndcg_scores) if len(ndcg_scores) > 0 else 0.0
                        
                        # Update best
                        if avg_ndcg > best_ndcg:
                            best_ndcg = avg_ndcg
                            best_weights = weights.copy()
                        
                        pbar.update(1)
    
    logging.info(f"Best weights: {best_weights}")
    logging.info(f"Best validation NDCG@{k}: {best_ndcg:.4f}")
    
    return best_weights


# ========================================
# Main Training Function
# ========================================

def main():
    parser = argparse.ArgumentParser(
        description="Train hybrid ensemble by blending all models"
    )
    parser.add_argument('--catalog', type=str, required=True,
                       help='Path to course catalog CSV')
    parser.add_argument('--interactions', type=str, required=True,
                       help='Path to interactions CSV')
    parser.add_argument('--k', type=int, default=10,
                       help='Top-K for evaluation (default: 10)')
    parser.add_argument('--min-warm', type=int, default=10,
                       help='Minimum train interactions for warm user (default: 10)')
    parser.add_argument('--out', type=str, default='artifacts',
                       help='Output directory for artifacts (default: artifacts)')
    
    args = parser.parse_args()
    
    # Setup paths
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    logging.info("=" * 60)
    logging.info("HYBRID ENSEMBLE TRAINING")
    logging.info("=" * 60)
    logging.info(f"Catalog: {args.catalog}")
    logging.info(f"Interactions: {args.interactions}")
    logging.info(f"K: {args.k}")
    logging.info(f"Min warm interactions: {args.min_warm}")
    logging.info(f"Output directory: {args.out}")
    
    # ========================================
    # 1. Load Required Artifacts
    # ========================================
    
    logging.info("\n" + "="*60)
    logging.info("STEP 1: Loading artifacts...")
    logging.info("="*60)
    
    models = {}
    required_files = ['id_maps.pkl', 'tfidf.pkl', 'X_items.npz', 
                     'content_index.pkl', 'popular.pkl']
    optional_files = ['cf_nmf.pkl', 'cf_knn.pkl', 'supervised_ranker.pkl', 'kmeans.pkl']
    
    # Load required files
    for fname in required_files:
        fpath = out_dir / fname
        if not fpath.exists():
            logging.error(f"Required file missing: {fpath}")
            raise FileNotFoundError(f"Required file missing: {fpath}")
        
        # Special handling for different file types
        if fname == 'X_items.npz':
            models['X_items'] = sparse.load_npz(fpath)
        else:
            # Try joblib first (preferred for sklearn objects), then pickle
            try:
                obj = joblib.load(fpath)
            except:
                try:
                    with open(fpath, 'rb') as f:
                        obj = pickle.load(f)
                except Exception as e:
                    logging.error(f"Failed to load {fname}: {e}")
                    raise
            
            if fname == 'id_maps.pkl':
                models['user_map'] = obj.get('user_map') or obj.get('user_to_idx')
                models['item_map'] = obj.get('item_map') or obj.get('item_to_idx')
            elif fname == 'tfidf.pkl':
                models['tfidf'] = obj
            elif fname == 'content_index.pkl':
                models['content_index'] = obj
            elif fname == 'popular.pkl':
                models['popularity'] = obj
                # Handle both dict and object formats
                if hasattr(obj, 'item_popularity'):
                    models['item_popularity'] = obj.item_popularity
                elif isinstance(obj, dict):
                    models['item_popularity'] = obj['item_popularity']
                else:
                    models['item_popularity'] = obj
        
        logging.info(f"✓ Loaded {fname}")
    
    # Load optional model files
    for fname in optional_files:
        fpath = out_dir / fname
        if not fpath.exists():
            logging.warning(f"Optional file not found: {fpath} (will skip this model)")
            continue
        
        # Try joblib first, then pickle
        try:
            obj = joblib.load(fpath)
        except:
            try:
                with open(fpath, 'rb') as f:
                    obj = pickle.load(f)
            except Exception as e:
                logging.warning(f"Failed to load {fname}: {e}")
                continue
        
        if fname == 'cf_nmf.pkl':
            # Handle both dict and object formats
            if isinstance(obj, dict):
                models['nmf'] = obj.get('model')
                models['nmf_user_factors'] = obj.get('user_factors')
                models['nmf_item_factors'] = obj.get('item_factors')
                models['R_train'] = obj.get('R_train')
            else:
                models['nmf'] = obj
                models['nmf_user_factors'] = obj.user_factors
                models['nmf_item_factors'] = obj.item_factors
                models['R_train'] = obj.R_train if hasattr(obj, 'R_train') else None
            logging.info(f"✓ Loaded NMF model")
        elif fname == 'cf_knn.pkl':
            if isinstance(obj, dict):
                models['knn'] = obj.get('model')
                models['knn_item_sim'] = obj.get('item_similarity')
                if 'R_train' not in models:
                    models['R_train'] = obj.get('R_train')
            else:
                models['knn'] = obj
                models['knn_item_sim'] = obj.item_similarity
                if 'R_train' not in models:
                    models['R_train'] = obj.R_train if hasattr(obj, 'R_train') else None
            logging.info(f"✓ Loaded Item-KNN model")
        elif fname == 'supervised_ranker.pkl':
            if isinstance(obj, dict):
                models['supervised'] = obj.get('model')
                models['supervised_scaler'] = obj.get('scaler')
                models['supervised_metadata'] = obj.get('metadata', {})
            else:
                models['supervised'] = obj
                models['supervised_scaler'] = None
                models['supervised_metadata'] = {}
            logging.info(f"✓ Loaded supervised ranker")
        elif fname == 'kmeans.pkl':
            if isinstance(obj, dict):
                models['kmeans'] = obj.get('model')
                models['item_clusters'] = obj.get('item_clusters')
            else:
                models['kmeans'] = obj
                models['item_clusters'] = None
            logging.info(f"✓ Loaded KMeans clustering model")
    
    # Load supervised ranker metadata separately if needed
    supervised_meta_path = out_dir / 'feature_meta.json'
    if supervised_meta_path.exists() and 'supervised' in models:
        import json
        with open(supervised_meta_path, 'r') as f:
            meta = json.load(f)
            if 'metadata' in meta:
                models['supervised_metadata'].update(meta['metadata'])
        logging.info(f"✓ Loaded supervised metadata")
    
    # ========================================
    # 2. Load Data
    # ========================================
    
    logging.info("\n" + "="*60)
    logging.info("STEP 2: Loading data...")
    logging.info("="*60)
    
    # Load interactions
    interactions_df = pd.read_csv(args.interactions)
    interactions_df = interactions_df.rename(columns={'course_id': 'item_id'})
    
    # Convert timestamp to numeric if needed
    if interactions_df['timestamp'].dtype == 'object':
        interactions_df['timestamp'] = pd.to_numeric(interactions_df['timestamp'], errors='coerce')
    
    logging.info(f"Loaded {len(interactions_df)} interactions")
    
    models['interactions_df'] = interactions_df
    
    # Compute train/test split (temporal: last interaction per user -> test)
    logging.info("Computing train/test split (temporal: last interaction per user)...")
    interactions_df = interactions_df.sort_values(['user_id', 'timestamp'])
    
    # Last interaction per user -> test
    test_df = interactions_df.groupby('user_id').tail(1).reset_index(drop=True)
    models['test_pairs'] = set(zip(test_df['user_id'], test_df['item_id']))
    
    # Rest -> train
    test_pairs_set = models['test_pairs']
    train_df = interactions_df[
        ~interactions_df.apply(lambda r: (r['user_id'], r['item_id']) in test_pairs_set, axis=1)
    ].reset_index(drop=True)
    models['train_pairs'] = set(zip(train_df['user_id'], train_df['item_id']))
    
    logging.info(f"Train: {len(models['train_pairs'])} interactions")
    logging.info(f"Test: {len(models['test_pairs'])} interactions")
    
    # Load catalog
    catalog_df = pd.read_csv(args.catalog)
    logging.info(f"Loaded {len(catalog_df)} courses")
    
    # Build user profiles (TF-IDF from interactions)
    logging.info("Building user profiles from interactions...")
    user_profiles = sparse.lil_matrix((len(models['user_map']), models['X_items'].shape[1]))
    
    for user_id, group in interactions_df.groupby('user_id'):
        item_ids = group['item_id'].values
        # Convert to int if needed and filter valid indices
        valid_items = []
        for i in item_ids:
            try:
                idx = int(i) if isinstance(i, str) else i
                if 0 <= idx < models['X_items'].shape[0]:
                    valid_items.append(idx)
            except (ValueError, TypeError):
                continue
        
        if len(valid_items) > 0:
            user_profiles[user_id] = models['X_items'][valid_items].mean(axis=0)
    
    models['user_profiles'] = user_profiles.tocsr()
    logging.info(f"User profiles: {models['user_profiles'].shape}")
    
    # Build item popularity and recency metadata
    item_counts = interactions_df['item_id'].value_counts().to_dict()
    models['item_popularity'] = {
        item_id: count / len(interactions_df)
        for item_id, count in item_counts.items()
    }
    
    # Recency (days since first interaction)
    first_timestamp = interactions_df['timestamp'].min()
    item_last_ts = interactions_df.groupby('item_id')['timestamp'].max()
    models['item_recency_days'] = {
        item_id: (ts - first_timestamp) / 86400
        for item_id, ts in item_last_ts.items()
    }
    
    # Item mean ratings
    models['item_mean_rating'] = interactions_df.groupby('item_id')['rating'].mean().to_dict()
    
    # ========================================
    # 3. Create Validation Split
    # ========================================
    
    logging.info("\n" + "="*60)
    logging.info("STEP 3: Creating validation split...")
    logging.info("="*60)
    
    val_df, train_minus_val_df = create_validation_split(
        interactions_df, models['train_pairs']
    )
    
    train_minus_val_pairs = set(zip(train_minus_val_df['user_id'], train_minus_val_df['item_id']))
    
    # ========================================
    # 4. Grid Search Weights
    # ========================================
    
    logging.info("\n" + "="*60)
    logging.info("STEP 4: Grid searching ensemble weights...")
    logging.info("="*60)
    
    best_weights = grid_search_weights(
        val_df, models, args.min_warm, args.k,
        train_minus_val_pairs, step=0.25
    )
    
    # ========================================
    # 5. Evaluate on Test Split
    # ========================================
    
    logging.info("\n" + "="*60)
    logging.info("STEP 5: Evaluating on test split...")
    logging.info("="*60)
    
    test_df = interactions_df[
        interactions_df.apply(lambda r: (r['user_id'], r['item_id']) in models['test_pairs'], axis=1)
    ]
    
    logging.info(f"Test set: {len(test_df)} interactions")
    
    overall_metrics, warm_metrics, cold_metrics = evaluate_recommendations(
        test_df, models, best_weights, args.min_warm, args.k, models['train_pairs']
    )
    
    # ========================================
    # 6. Print Results
    # ========================================
    
    logging.info("\n" + "="*60)
    logging.info("EVALUATION RESULTS")
    logging.info("="*60)
    
    logging.info(f"\nOVERALL (all users):")
    logging.info(f"  Precision@{args.k}: {overall_metrics['precision']:.4f}")
    logging.info(f"  Recall@{args.k}:    {overall_metrics['recall']:.4f}")
    logging.info(f"  NDCG@{args.k}:      {overall_metrics['ndcg']:.4f}")
    
    logging.info(f"\nWARM USERS (≥{args.min_warm} train interactions):")
    logging.info(f"  Precision@{args.k}: {warm_metrics['precision']:.4f}")
    logging.info(f"  Recall@{args.k}:    {warm_metrics['recall']:.4f}")
    logging.info(f"  NDCG@{args.k}:      {warm_metrics['ndcg']:.4f}")
    
    logging.info(f"\nCOLD USERS (<{args.min_warm} train interactions):")
    logging.info(f"  Precision@{args.k}: {cold_metrics['precision']:.4f}")
    logging.info(f"  Recall@{args.k}:    {cold_metrics['recall']:.4f}")
    logging.info(f"  NDCG@{args.k}:      {cold_metrics['ndcg']:.4f}")
    
    logging.info(f"\nENSEMBLE WEIGHTS (warm users):")
    for model, weight in best_weights.items():
        if weight > 0:
            logging.info(f"  {model:15s}: {weight:.2f}")
    
    logging.info(f"\nCOLD USER POLICY:")
    logging.info(f"  Content:     0.70")
    logging.info(f"  Popularity:  0.30")
    logging.info(f"  Recency boost: 0.10")
    
    # ========================================
    # 7. Save Artifacts
    # ========================================
    
    logging.info("\n" + "="*60)
    logging.info("STEP 6: Saving artifacts...")
    logging.info("="*60)
    
    # Save hybrid configuration
    hybrid_config = {
        'weights': best_weights,
        'min_warm': args.min_warm,
        'k': args.k,
        'cold_policy': {
            'content': 0.7,
            'popularity': 0.3,
            'recency_boost': 0.1
        },
        'metrics': {
            'overall': overall_metrics,
            'warm': warm_metrics,
            'cold': cold_metrics
        }
    }
    
    hybrid_path = out_dir / 'hybrid.pkl'
    with open(hybrid_path, 'wb') as f:
        pickle.dump(hybrid_config, f)
    logging.info(f"✓ Saved hybrid configuration to {hybrid_path}")
    
    # Save model card
    card_path = out_dir / 'HYBRID_CARD.md'
    with open(card_path, 'w', encoding='utf-8') as f:
        f.write("# 🔮 Hybrid Ensemble Model Card\n\n")
        f.write("## Overview\n\n")
        f.write("This hybrid ensemble blends multiple recommendation models with optimized weights.\n\n")
        
        f.write("## Configuration\n\n")
        f.write(f"- **Min warm interactions**: {args.min_warm}\n")
        f.write(f"- **Evaluation K**: {args.k}\n\n")
        
        f.write("## Ensemble Weights (Warm Users)\n\n")
        f.write("Optimized via grid search on validation split:\n\n")
        f.write("| Model | Weight |\n")
        f.write("|-------|--------|\n")
        for model, weight in sorted(best_weights.items(), key=lambda x: -x[1]):
            if weight > 0:
                f.write(f"| {model.capitalize()} | {weight:.2f} |\n")
        f.write("\n")
        
        f.write("## Cold User Policy\n\n")
        f.write("Fixed blend for users with <{} train interactions:\n\n".format(args.min_warm))
        f.write("- **Content**: 0.70\n")
        f.write("- **Popularity**: 0.30\n")
        f.write("- **Recency boost**: 0.10\n\n")
        
        f.write("## Performance Metrics\n\n")
        f.write(f"### Overall (All Users)\n\n")
        f.write(f"- **Precision@{args.k}**: {overall_metrics['precision']:.4f}\n")
        f.write(f"- **Recall@{args.k}**: {overall_metrics['recall']:.4f}\n")
        f.write(f"- **NDCG@{args.k}**: {overall_metrics['ndcg']:.4f}\n\n")
        
        f.write(f"### Warm Users (≥{args.min_warm} interactions)\n\n")
        f.write(f"- **Precision@{args.k}**: {warm_metrics['precision']:.4f}\n")
        f.write(f"- **Recall@{args.k}**: {warm_metrics['recall']:.4f}\n")
        f.write(f"- **NDCG@{args.k}**: {warm_metrics['ndcg']:.4f}\n\n")
        
        f.write(f"### Cold Users (<{args.min_warm} interactions)\n\n")
        f.write(f"- **Precision@{args.k}**: {cold_metrics['precision']:.4f}\n")
        f.write(f"- **Recall@{args.k}**: {cold_metrics['recall']:.4f}\n")
        f.write(f"- **NDCG@{args.k}**: {cold_metrics['ndcg']:.4f}\n\n")
        
        f.write("## Usage\n\n")
        f.write("```python\n")
        f.write("import pickle\n\n")
        f.write("# Load configuration\n")
        f.write("with open('artifacts/hybrid.pkl', 'rb') as f:\n")
        f.write("    config = pickle.load(f)\n\n")
        f.write("weights = config['weights']\n")
        f.write("min_warm = config['min_warm']\n")
        f.write("```\n\n")
        
        f.write(f"*Generated: November 6, 2025*\n")
    
    logging.info(f"✓ Saved model card to {card_path}")
    
    logging.info("\n" + "="*60)
    logging.info("HYBRID ENSEMBLE TRAINING COMPLETE!")
    logging.info("="*60)


if __name__ == '__main__':
    main()
