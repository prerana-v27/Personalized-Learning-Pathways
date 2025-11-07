#!/usr/bin/env python3
"""
Synthetic User-Course Interaction Generator
Generates realistic interaction data from course catalog for collaborative filtering.
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic user-course interactions from course catalog"
    )
    parser.add_argument("--catalog", required=True, help="Path to course catalog CSV")
    parser.add_argument("--out", required=True, help="Output interactions CSV path")
    parser.add_argument("--n-users", type=int, default=1500, help="Number of users to generate")
    parser.add_argument("--min-per-user", type=int, default=3, help="Minimum interactions per user")
    parser.add_argument("--max-per-user", type=int, default=15, help="Maximum interactions per user")
    parser.add_argument("--days-back", type=int, default=365, help="Date range in days")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_catalog(catalog_path):
    """Load and validate course catalog."""
    logger.info(f"Loading course catalog from {catalog_path}")
    
    try:
        df = pd.read_csv(catalog_path)
    except FileNotFoundError:
        logger.error(f"Catalog file not found: {catalog_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading catalog: {e}")
        sys.exit(1)
    
    if len(df) == 0:
        logger.error("Catalog is empty")
        sys.exit(1)
    
    logger.info(f"Loaded {len(df)} courses")
    
    # Ensure unique course_id
    if 'course_id' not in df.columns:
        logger.info("Creating course_id column")
        df['course_id'] = ['c' + str(i).zfill(6) for i in range(len(df))]
    else:
        # Ensure uniqueness
        if df['course_id'].duplicated().any():
            logger.warning("Found duplicate course_ids, regenerating...")
            df['course_id'] = ['c' + str(i).zfill(6) for i in range(len(df))]
    
    return df


def build_tfidf_matrix(catalog_df):
    """Build TF-IDF matrix from course metadata."""
    logger.info("Building TF-IDF matrix from course metadata")
    
    # Combine text fields
    text_fields = []
    for _, row in catalog_df.iterrows():
        parts = []
        
        # Add available fields
        if 'title' in row and pd.notna(row['title']):
            parts.append(str(row['title']))
        if 'description' in row and pd.notna(row['description']):
            parts.append(str(row['description']))
        if 'skills' in row and pd.notna(row['skills']):
            parts.append(str(row['skills']))
        if 'subject' in row and pd.notna(row['subject']):
            parts.append(str(row['subject']))
        
        # Join with period separator
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
    
    return tfidf_matrix


def get_popularity_prior(catalog_df):
    """Extract popularity prior from catalog."""
    logger.info("Extracting popularity prior")
    
    if 'popularity_norm' in catalog_df.columns:
        popularity = catalog_df['popularity_norm'].fillna(0).values
        
        # Check if we have valid popularity values
        if popularity.sum() > 0:
            # Normalize to ensure sum = 1
            popularity = popularity / popularity.sum()
            logger.info("Using popularity_norm for prior")
        else:
            logger.warning("popularity_norm is all zeros, using uniform distribution")
            popularity = np.ones(len(catalog_df)) / len(catalog_df)
    else:
        logger.warning("popularity_norm not found, using uniform distribution")
        popularity = np.ones(len(catalog_df)) / len(catalog_df)
    
    return popularity


def sample_courses_for_user(
    n_courses, 
    tfidf_matrix, 
    popularity_prior, 
    seed_idx=None,
    content_weight=0.7,
    popularity_weight=0.3,
    n_neighbors=10,
    rng=None
):
    """
    Sample courses for a user using mixture of content similarity and popularity.
    
    Args:
        n_courses: Number of courses to sample
        tfidf_matrix: Course content TF-IDF matrix
        popularity_prior: Popularity distribution
        seed_idx: Optional seed course index
        content_weight: Weight for content-based sampling
        popularity_weight: Weight for popularity-based sampling
        n_neighbors: Number of similar courses to consider
        rng: Random number generator
    
    Returns:
        List of course indices
    """
    n_items = tfidf_matrix.shape[0]
    selected_courses = []
    
    # Select seed course
    if seed_idx is None:
        seed_idx = rng.choice(n_items, p=popularity_prior)
    selected_courses.append(seed_idx)
    
    # Sample remaining courses
    while len(selected_courses) < n_courses:
        # Decide sampling strategy
        if rng.random() < content_weight and len(selected_courses) > 0:
            # Content-based: sample from neighbors of already selected courses
            # Pick a random course from selected
            base_course = rng.choice(selected_courses)
            
            # Compute similarities with base course
            base_vector = tfidf_matrix[base_course:base_course+1]
            similarities = cosine_similarity(base_vector, tfidf_matrix).flatten()
            
            # Mask out already selected courses
            similarities[selected_courses] = -1
            
            # Get top-K similar courses
            if similarities.max() > -1:
                top_k_idx = np.argpartition(similarities, -min(n_neighbors, n_items))[-n_neighbors:]
                top_k_idx = top_k_idx[similarities[top_k_idx] > 0]  # Only positive similarity
                
                if len(top_k_idx) > 0:
                    # Sample from top-K with probability proportional to similarity
                    top_k_sims = similarities[top_k_idx]
                    top_k_probs = top_k_sims / top_k_sims.sum()
                    next_course = rng.choice(top_k_idx, p=top_k_probs)
                else:
                    # Fall back to popularity
                    available_mask = np.ones(n_items, dtype=bool)
                    available_mask[selected_courses] = False
                    available_idx = np.where(available_mask)[0]
                    
                    if len(available_idx) == 0:
                        break
                    
                    available_probs = popularity_prior[available_idx]
                    available_probs = available_probs / available_probs.sum()
                    next_course = rng.choice(available_idx, p=available_probs)
            else:
                # No valid similarities, use popularity
                available_mask = np.ones(n_items, dtype=bool)
                available_mask[selected_courses] = False
                available_idx = np.where(available_mask)[0]
                
                if len(available_idx) == 0:
                    break
                
                available_probs = popularity_prior[available_idx]
                available_probs = available_probs / available_probs.sum()
                next_course = rng.choice(available_idx, p=available_probs)
        else:
            # Popularity-based: sample from global popularity
            available_mask = np.ones(n_items, dtype=bool)
            available_mask[selected_courses] = False
            available_idx = np.where(available_mask)[0]
            
            if len(available_idx) == 0:
                break
            
            available_probs = popularity_prior[available_idx]
            available_probs = available_probs / available_probs.sum()
            next_course = rng.choice(available_idx, p=available_probs)
        
        selected_courses.append(next_course)
    
    return selected_courses


def generate_interactions(catalog_df, tfidf_matrix, popularity_prior, args, rng):
    """Generate synthetic user-course interactions."""
    logger.info(f"Generating interactions for {args.n_users} users")
    
    interactions = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days_back)
    
    for user_idx in range(args.n_users):
        if (user_idx + 1) % 100 == 0:
            logger.info(f"Generated interactions for {user_idx + 1}/{args.n_users} users")
        
        # Generate user ID
        user_id = f"u{str(user_idx + 1).zfill(6)}"
        
        # Sample number of courses for this user
        n_courses = rng.integers(args.min_per_user, args.max_per_user + 1)
        
        # Sample courses
        course_indices = sample_courses_for_user(
            n_courses=n_courses,
            tfidf_matrix=tfidf_matrix,
            popularity_prior=popularity_prior,
            seed_idx=None,
            rng=rng
        )
        
        # Generate timestamps uniformly over date range
        timestamps = []
        for _ in range(len(course_indices)):
            random_days = rng.uniform(0, args.days_back)
            timestamp = start_date + timedelta(days=random_days)
            timestamps.append(timestamp)
        
        # Sort by timestamp
        sorted_pairs = sorted(zip(timestamps, course_indices), key=lambda x: x[0])
        
        # Create interaction records
        for timestamp, course_idx in sorted_pairs:
            interactions.append({
                'user_id': user_id,
                'course_id': catalog_df.iloc[course_idx]['course_id'],
                'rating': 1.0,  # Implicit feedback
                'timestamp': timestamp
            })
    
    logger.info(f"Generated {len(interactions)} total interactions")
    
    return pd.DataFrame(interactions)


def print_dataset_card(interactions_df, catalog_df, args):
    """Print dataset statistics."""
    logger.info("\n" + "="*80)
    logger.info("DATASET CARD")
    logger.info("="*80)
    
    n_users = interactions_df['user_id'].nunique()
    n_items = interactions_df['course_id'].nunique()
    n_interactions = len(interactions_df)
    
    # Calculate statistics
    interactions_per_user = interactions_df.groupby('user_id').size()
    avg_per_user = interactions_per_user.mean()
    median_per_user = interactions_per_user.median()
    
    # Sparsity
    possible_interactions = n_users * len(catalog_df)
    sparsity = 100 * (1 - n_interactions / possible_interactions)
    
    # Date range
    min_date = interactions_df['timestamp'].min()
    max_date = interactions_df['timestamp'].max()
    date_range_days = (max_date - min_date).days
    
    # Users meeting minimum threshold
    users_above_min = (interactions_per_user >= args.min_per_user).sum()
    pct_above_min = 100 * users_above_min / n_users
    
    logger.info(f"Number of users:           {n_users:,}")
    logger.info(f"Number of items (courses): {n_items:,}")
    logger.info(f"Total interactions:        {n_interactions:,}")
    logger.info(f"Avg interactions/user:     {avg_per_user:.2f}")
    logger.info(f"Median interactions/user:  {median_per_user:.0f}")
    logger.info(f"Sparsity:                  {sparsity:.2f}%")
    logger.info(f"Date range:                {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')} ({date_range_days} days)")
    logger.info(f"Users ≥ {args.min_per_user} interactions:   {users_above_min} ({pct_above_min:.1f}%)")
    
    # Rating distribution
    logger.info(f"\nRating distribution:")
    logger.info(f"  Min:  {interactions_df['rating'].min():.2f}")
    logger.info(f"  Max:  {interactions_df['rating'].max():.2f}")
    logger.info(f"  Mean: {interactions_df['rating'].mean():.2f}")
    
    # Top courses
    top_courses = interactions_df['course_id'].value_counts().head(5)
    logger.info(f"\nTop 5 most interacted courses:")
    for course_id, count in top_courses.items():
        course_title = catalog_df[catalog_df['course_id'] == course_id]['title'].values
        title = course_title[0] if len(course_title) > 0 else "Unknown"
        logger.info(f"  {course_id}: {count} interactions - {title[:50]}...")
    
    logger.info("="*80 + "\n")


def main():
    """Main pipeline."""
    args = parse_args()
    
    logger.info("="*80)
    logger.info("Synthetic User-Course Interaction Generator")
    logger.info("="*80)
    logger.info(f"Catalog:       {args.catalog}")
    logger.info(f"Output:        {args.out}")
    logger.info(f"Users:         {args.n_users}")
    logger.info(f"Per user:      {args.min_per_user}-{args.max_per_user}")
    logger.info(f"Days back:     {args.days_back}")
    logger.info(f"Random seed:   {args.seed}")
    logger.info("="*80 + "\n")
    
    # Set random seed
    rng = np.random.default_rng(args.seed)
    
    # Load catalog
    catalog_df = load_catalog(args.catalog)
    
    # Build TF-IDF matrix
    tfidf_matrix = build_tfidf_matrix(catalog_df)
    
    # Get popularity prior
    popularity_prior = get_popularity_prior(catalog_df)
    
    # Generate interactions
    interactions_df = generate_interactions(
        catalog_df, tfidf_matrix, popularity_prior, args, rng
    )
    
    # Clip ratings to [0, 5] (already 1.0, but for safety)
    interactions_df['rating'] = interactions_df['rating'].clip(0, 5)
    
    # Print dataset card
    print_dataset_card(interactions_df, catalog_df, args)
    
    # Save output
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save with exactly the required columns
    output_df = interactions_df[['user_id', 'course_id', 'rating', 'timestamp']]
    output_df.to_csv(output_path, index=False)
    
    logger.info(f"Saved interactions to {args.out}")
    logger.info("Generation completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\nGeneration interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nGeneration failed: {e}", exc_info=True)
        sys.exit(1)
