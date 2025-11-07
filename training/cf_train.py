#!/usr/bin/env python3
"""
Collaborative Filtering Training Script
Trains Item-KNN and NMF models with per-user temporal split and evaluation.
"""

import argparse
import logging
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from surprise import Dataset, Reader, KNNBaseline, NMF
    from surprise.model_selection import GridSearchCV
    from surprise import Trainset
except ImportError:
    print("ERROR: surprise library not installed. Run: pip install scikit-surprise")
    sys.exit(1)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train collaborative filtering models with temporal split"
    )
    parser.add_argument("--data", required=True, help="Path to CSV file")
    parser.add_argument("--user-col", default="user_id", help="User column name")
    parser.add_argument("--item-col", default="course_id", help="Item column name")
    parser.add_argument("--rating-col", default="rating", help="Rating column name")
    parser.add_argument("--time-col", default="timestamp", help="Timestamp column name")
    parser.add_argument("--k", type=int, default=10, help="Top-K for metrics")
    parser.add_argument("--out", default="artifacts", help="Output directory")
    parser.add_argument("--fast", action="store_true", help="Fast mode: smaller grids")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_and_prepare_data(args):
    """Load CSV, prepare ratings, and create temporal split."""
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
    
    Returns:
        train_df: Training dataframe
        test_df: Test dataframe (users with >1 interaction)
        single_user_mask: Boolean mask for users with only 1 interaction
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


def build_surprise_trainset(train_df, user_col, item_col, rating_col):
    """Build Surprise Trainset and extract ID mappings."""
    logger.info("Building Surprise Trainset")
    
    # Create Surprise Dataset
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(
        train_df[[user_col, item_col, rating_col]], 
        reader
    )
    
    # Build full trainset
    trainset = data.build_full_trainset()
    
    # Extract ID mappings (raw ↔ inner)
    id_maps = {
        'user_raw2inner': dict(trainset._raw2inner_id_users),
        'user_inner2raw': dict(trainset._inner2raw_id_users),
        'item_raw2inner': dict(trainset._raw2inner_id_items),
        'item_inner2raw': dict(trainset._inner2raw_id_items),
    }
    
    logger.info(f"Trainset: {trainset.n_users} users, {trainset.n_items} items, {trainset.n_ratings} ratings")
    
    return trainset, id_maps


def train_knn_baseline(trainset, fast_mode=False):
    """Train Item-KNN with grid search."""
    logger.info("Training Item-KNN Baseline with grid search")
    
    if fast_mode:
        param_grid = {'k': [20, 40]}
    else:
        param_grid = {'k': [20, 40, 60]}
    
    sim_options = {
        'name': 'pearson_baseline',
        'user_based': False
    }
    
    gs = GridSearchCV(
        KNNBaseline,
        param_grid,
        measures=['rmse'],
        cv=3,
        n_jobs=-1,
        joblib_verbose=0
    )
    
    gs.fit(Dataset.load_from_df(
        pd.DataFrame(trainset.all_ratings(), columns=['uid', 'iid', 'rating']),
        Reader(rating_scale=(0, 5))
    ))
    
    logger.info(f"Best KNN params: {gs.best_params['rmse']}")
    logger.info(f"Best RMSE: {gs.best_score['rmse']:.4f}")
    
    # Train final model on full trainset
    best_k = gs.best_params['rmse']['k']
    model = KNNBaseline(k=best_k, sim_options=sim_options)
    model.fit(trainset)
    
    return model, gs.best_params['rmse']


def train_nmf(trainset, fast_mode=False):
    """Train NMF with grid search."""
    logger.info("Training NMF with grid search")
    
    if fast_mode:
        param_grid = {
            'n_factors': [40, 60],
            'reg_pu': [0.02, 0.06],
            'reg_qi': [0.02, 0.06]
        }
    else:
        param_grid = {
            'n_factors': [40, 60, 80],
            'reg_pu': [0.02, 0.06],
            'reg_qi': [0.02, 0.06]
        }
    
    gs = GridSearchCV(
        NMF,
        param_grid,
        measures=['rmse'],
        cv=3,
        n_jobs=-1,
        joblib_verbose=0
    )
    
    gs.fit(Dataset.load_from_df(
        pd.DataFrame(trainset.all_ratings(), columns=['uid', 'iid', 'rating']),
        Reader(rating_scale=(0, 5))
    ))
    
    logger.info(f"Best NMF params: {gs.best_params['rmse']}")
    logger.info(f"Best RMSE: {gs.best_score['rmse']:.4f}")
    
    # Train final model on full trainset
    best_params = gs.best_params['rmse']
    model = NMF(
        n_factors=best_params['n_factors'],
        reg_pu=best_params['reg_pu'],
        reg_qi=best_params['reg_qi'],
        random_state=42
    )
    model.fit(trainset)
    
    return model, gs.best_params['rmse']


def get_user_train_items(train_df, user_col, item_col):
    """Get set of items each user interacted with in training."""
    user_items = defaultdict(set)
    for _, row in train_df.iterrows():
        user_items[row[user_col]].add(row[item_col])
    return user_items


def evaluate_model(model, trainset, test_df, user_col, item_col, rating_col, 
                   id_maps, user_train_items, k=10, model_name="Model"):
    """
    Evaluate model with top-K metrics.
    
    Returns:
        dict with precision@k, recall@k, ndcg@k, rmse
    """
    logger.info(f"Evaluating {model_name}")
    
    all_items = set(id_maps['item_raw2inner'].keys())
    
    precisions = []
    recalls = []
    ndcgs = []
    rmse_errors = []
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Evaluating {model_name}"):
        user = row[user_col]
        true_item = row[item_col]
        true_rating = row[rating_col]
        
        # Skip if user not in trainset
        if user not in id_maps['user_raw2inner']:
            continue
        
        # Get items to score (exclude train items)
        train_items = user_train_items.get(user, set())
        candidate_items = all_items - train_items
        
        # Skip if no candidates or true item not in candidates
        if not candidate_items or true_item not in candidate_items:
            continue
        
        # Score all candidate items
        predictions = []
        for item in candidate_items:
            if item in id_maps['item_raw2inner']:
                pred = model.predict(user, item, verbose=False)
                predictions.append((item, pred.est))
        
        # Sort by predicted rating (descending)
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Top-K recommendations
        top_k_items = [item for item, _ in predictions[:k]]
        
        # Metrics
        hit = 1 if true_item in top_k_items else 0
        precisions.append(hit / k)
        recalls.append(hit)  # Binary: 1 relevant item
        
        # NDCG@K
        if hit:
            rank = top_k_items.index(true_item) + 1
            ndcg = 1.0 / np.log2(rank + 1)
        else:
            ndcg = 0.0
        ndcgs.append(ndcg)
        
        # RMSE
        pred = model.predict(user, true_item, verbose=False)
        rmse_errors.append((pred.est - true_rating) ** 2)
    
    metrics = {
        f'Precision@{k}': np.mean(precisions) if precisions else 0.0,
        f'Recall@{k}': np.mean(recalls) if recalls else 0.0,
        f'NDCG@{k}': np.mean(ndcgs) if ndcgs else 0.0,
        'RMSE': np.sqrt(np.mean(rmse_errors)) if rmse_errors else 0.0
    }
    
    logger.info(f"{model_name} - P@{k}: {metrics[f'Precision@{k}']:.4f}, "
                f"R@{k}: {metrics[f'Recall@{k}']:.4f}, "
                f"NDCG@{k}: {metrics[f'NDCG@{k}']:.4f}, "
                f"RMSE: {metrics['RMSE']:.4f}")
    
    return metrics


def popularity_baseline(train_df, test_df, user_col, item_col, rating_col, 
                       user_train_items, k=10):
    """
    Popularity baseline: recommend most popular items (by interaction count).
    """
    logger.info("Computing Popularity baseline")
    
    # Get item popularity from training data
    item_counts = train_df[item_col].value_counts()
    popular_items = item_counts.index.tolist()
    
    precisions = []
    recalls = []
    ndcgs = []
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating Popularity"):
        user = row[user_col]
        true_item = row[item_col]
        
        # Get items to recommend (exclude train items)
        train_items = user_train_items.get(user, set())
        
        # Top-K popular items not in train
        top_k_items = [item for item in popular_items if item not in train_items][:k]
        
        # Metrics
        hit = 1 if true_item in top_k_items else 0
        precisions.append(hit / k)
        recalls.append(hit)
        
        # NDCG@K
        if hit:
            rank = top_k_items.index(true_item) + 1
            ndcg = 1.0 / np.log2(rank + 1)
        else:
            ndcg = 0.0
        ndcgs.append(ndcg)
    
    metrics = {
        f'Precision@{k}': np.mean(precisions) if precisions else 0.0,
        f'Recall@{k}': np.mean(recalls) if recalls else 0.0,
        f'NDCG@{k}': np.mean(ndcgs) if ndcgs else 0.0,
        'RMSE': 0.0  # N/A for popularity
    }
    
    logger.info(f"Popularity - P@{k}: {metrics[f'Precision@{k}']:.4f}, "
                f"R@{k}: {metrics[f'Recall@{k}']:.4f}, "
                f"NDCG@{k}: {metrics[f'NDCG@{k}']:.4f}")
    
    return metrics


def print_metrics_table(results, k):
    """Print formatted metrics table."""
    print("\n" + "="*80)
    print(f"{'Model':<15} {'P@'+str(k):<12} {'R@'+str(k):<12} {'NDCG@'+str(k):<12} {'RMSE':<12}")
    print("="*80)
    
    for model_name, metrics in results.items():
        p = metrics.get(f'Precision@{k}', 0.0)
        r = metrics.get(f'Recall@{k}', 0.0)
        n = metrics.get(f'NDCG@{k}', 0.0)
        rmse = metrics.get('RMSE', 0.0)
        
        rmse_str = f"{rmse:.4f}" if rmse > 0 else "N/A"
        print(f"{model_name:<15} {p:<12.4f} {r:<12.4f} {n:<12.4f} {rmse_str:<12}")
    
    print("="*80 + "\n")


def save_artifacts(out_dir, knn_model, nmf_model, id_maps, split_info):
    """Save trained models and artifacts."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving artifacts to {out_dir}")
    
    # Save models
    with open(out_path / "cf_knn.pkl", "wb") as f:
        pickle.dump(knn_model, f)
    logger.info("Saved KNN model to cf_knn.pkl")
    
    with open(out_path / "cf_nmf.pkl", "wb") as f:
        pickle.dump(nmf_model, f)
    logger.info("Saved NMF model to cf_nmf.pkl")
    
    # Save ID maps
    with open(out_path / "id_maps.pkl", "wb") as f:
        pickle.dump(id_maps, f)
    logger.info("Saved ID mappings to id_maps.pkl")
    
    # Save split info
    with open(out_path / "split.pkl", "wb") as f:
        pickle.dump(split_info, f)
    logger.info("Saved split information to split.pkl")


def main():
    """Main training pipeline."""
    args = parse_args()
    
    logger.info("="*80)
    logger.info("Collaborative Filtering Training Pipeline")
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
    
    # Build Surprise trainset
    trainset, id_maps = build_surprise_trainset(
        train_df, args.user_col, args.item_col, args.rating_col
    )
    
    # Get user train items for evaluation
    user_train_items = get_user_train_items(train_df, args.user_col, args.item_col)
    
    # Train models
    logger.info("\n" + "="*80)
    logger.info("MODEL TRAINING")
    logger.info("="*80)
    
    knn_model, knn_params = train_knn_baseline(trainset, args.fast)
    nmf_model, nmf_params = train_nmf(trainset, args.fast)
    
    # Evaluate models
    logger.info("\n" + "="*80)
    logger.info("MODEL EVALUATION")
    logger.info("="*80)
    
    results = {}
    
    # Popularity baseline
    results['Popularity'] = popularity_baseline(
        train_df, test_df, args.user_col, args.item_col, args.rating_col,
        user_train_items, k=args.k
    )
    
    # KNN
    results['KNN'] = evaluate_model(
        knn_model, trainset, test_df, args.user_col, args.item_col, args.rating_col,
        id_maps, user_train_items, k=args.k, model_name="KNN"
    )
    
    # NMF
    results['NMF'] = evaluate_model(
        nmf_model, trainset, test_df, args.user_col, args.item_col, args.rating_col,
        id_maps, user_train_items, k=args.k, model_name="NMF"
    )
    
    # Print results table
    print_metrics_table(results, args.k)
    
    # Save artifacts
    split_info = {
        'train_size': len(train_df),
        'test_size': len(test_df),
        'single_interaction_users': single_users,
        'knn_params': knn_params,
        'nmf_params': nmf_params,
        'metrics': results
    }
    
    save_artifacts(args.out, knn_model, nmf_model, id_maps, split_info)
    
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
