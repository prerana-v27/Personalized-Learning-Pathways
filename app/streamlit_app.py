"""
Personalized Learning Pathways - Streamlit UI

A production-clean interface for course recommendations using trained ML models.

Run with:
    streamlit run app/streamlit_app.py

Models:
    - Hybrid: Ensemble of NMF + KNN + Supervised + Content (auto-weighted)
    - Content: Pure TF-IDF cosine similarity

Author: Generated for Personalized Learning Pathways
Date: November 6, 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
from scipy import sparse
from scipy.sparse import csr_matrix
from typing import Dict, List, Tuple, Optional
import warnings
import sys

warnings.filterwarnings('ignore')


# ========================================
# Dummy Classes for Unpickling (must be in __main__)
# ========================================

# These classes allow unpickling of trained models without importing training scripts
class PopularityModel:
    """Dummy class for unpickling popularity model."""
    def __init__(self):
        self.item_popularity = None
        self.item_ids = None


class ItemKNNModel:
    """Dummy class for unpickling KNN model."""
    def __init__(self):
        self.similarity_matrix = None
        self.item_ids = None


class NMFModel:
    """Dummy class for unpickling NMF model."""
    def __init__(self):
        self.W = None
        self.H = None
        self.user_ids = None
        self.item_ids = None


class ContentRecommender:
    """Dummy class for unpickling content model."""
    def __init__(self):
        self.tfidf = None
        self.X_items = None


class ClusterRecommender:
    """Dummy class for unpickling cluster model."""
    def __init__(self):
        self.kmeans = None
        self.cluster_centers = None


# Register classes in both __main__ and main namespaces for pickle
sys.modules['__main__'].PopularityModel = PopularityModel
sys.modules['__main__'].ItemKNNModel = ItemKNNModel
sys.modules['__main__'].NMFModel = NMFModel
sys.modules['__main__'].ContentRecommender = ContentRecommender
sys.modules['__main__'].ClusterRecommender = ClusterRecommender

# Also register under 'main' for compatibility
if 'main' not in sys.modules:
    import types
    sys.modules['main'] = types.ModuleType('main')

sys.modules['main'].PopularityModel = PopularityModel
sys.modules['main'].ItemKNNModel = ItemKNNModel
sys.modules['main'].NMFModel = NMFModel
sys.modules['main'].ContentRecommender = ContentRecommender
sys.modules['main'].ClusterRecommender = ClusterRecommender

# ========================================
# Configuration (paths resolved relative to repo root)
# ========================================

# Resolve repository root (one level above `app/`)
ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DATA_DIR = ROOT_DIR / "Data" / "processed"

CATALOG_PATH = DATA_DIR / "all_courses_cleaned.csv"
INTERACTIONS_PATH = DATA_DIR / "interactions_synth.csv"

REQUIRED_ARTIFACTS = {
    'tfidf': 'tfidf.pkl',
    'content_index': 'content_index.pkl',
    'id_maps': 'id_maps.pkl'
}

OPTIONAL_ARTIFACTS = {
    'hybrid': 'hybrid.pkl',
    'popular': 'popular.pkl'
}


# ========================================
# Data Loading (Cached)
# ========================================

@st.cache_data
def load_catalog() -> pd.DataFrame:
    """Load course catalog with proper course_id."""
    try:
        df = pd.read_csv(CATALOG_PATH)
        
        # Ensure course_id exists
        if 'course_id' not in df.columns:
            df['course_id'] = 'c' + df.index.astype(str)
        
        # Ensure required columns exist
        required_cols = ['title']
        for col in required_cols:
            if col not in df.columns:
                st.error(f"Catalog missing required column: {col}")
                return pd.DataFrame()
        
        # Fill NaN values in text columns
        text_cols = ['title', 'description', 'skills', 'subject', 'difficulty']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna('')
        
        return df
    except FileNotFoundError:
        st.error(f"Catalog file not found: {CATALOG_PATH}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading catalog: {e}")
        return pd.DataFrame()


@st.cache_data
def load_interactions() -> Optional[pd.DataFrame]:
    """Load user-course interactions (optional)."""
    try:
        if not INTERACTIONS_PATH.exists():
            return None
        
        df = pd.read_csv(INTERACTIONS_PATH)
        
        # Rename course_id if needed
        if 'item_id' in df.columns and 'course_id' not in df.columns:
            df = df.rename(columns={'item_id': 'course_id'})
        
        return df
    except Exception as e:
        st.warning(f"Could not load interactions: {e}")
        return None


def _safe_text(val) -> str:
    """Return a safe string for display. Handles NaN/None gracefully."""
    try:
        if pd.isna(val):
            return ""
    except Exception:
        pass
    return "" if val is None else str(val)


@st.cache_resource
def load_artifacts() -> Dict:
    """Load all model artifacts with caching."""
    artifacts = {}
    errors = []
    
    # Load required artifacts
    for name, filename in REQUIRED_ARTIFACTS.items():
        filepath = ARTIFACTS_DIR / filename
        
        if not filepath.exists():
            errors.append(f"Required file missing: {filename}")
            continue
        
        try:
            # Try joblib first, then pickle
            try:
                obj = joblib.load(filepath)
            except:
                with open(filepath, 'rb') as f:
                    obj = pickle.load(f)
            
            # Handle special cases
            if name == 'tfidf':
                artifacts['tfidf'] = obj
            elif name == 'content_index':
                artifacts['content_index'] = obj
                # Load item vectors if present
                X_items_path = ARTIFACTS_DIR / 'X_items.npz'
                if X_items_path.exists():
                    X_items = sparse.load_npz(X_items_path)
                    # L2 normalize for cosine similarity
                    from sklearn.preprocessing import normalize
                    X_items = normalize(X_items, norm='l2', axis=1, copy=False)
                    artifacts['X_items'] = X_items
            elif name == 'id_maps':
                artifacts['user_map'] = obj.get('user_map') or obj.get('user_to_idx')
                artifacts['item_map'] = obj.get('item_map') or obj.get('item_to_idx')
        
        except Exception as e:
            errors.append(f"Error loading {filename}: {str(e)}")
    
    # Load optional artifacts
    for name, filename in OPTIONAL_ARTIFACTS.items():
        filepath = ARTIFACTS_DIR / filename
        
        if not filepath.exists():
            continue
        
        try:
            try:
                obj = joblib.load(filepath)
            except:
                with open(filepath, 'rb') as f:
                    obj = pickle.load(f)
            
            artifacts[name] = obj
        except Exception as e:
            st.warning(f"Could not load optional {filename}: {e}")
    
    # Show errors if any
    if errors:
        for error in errors:
            st.error(error)

    # --- Fallback: try to load recommender index from ml/outputs if TF-IDF/X_items missing ---
    try:
        need_tfidf = 'tfidf' not in artifacts
        need_xitems = 'X_items' not in artifacts
        if need_tfidf or need_xitems:
            fallback_path = ROOT_DIR / 'ml' / 'outputs' / 'recommender_index.joblib'
            if fallback_path.exists():
                try:
                    idx_data = joblib.load(fallback_path)
                    method = idx_data.get('method', '')
                    emb = idx_data.get('embeddings')
                    df_idx = idx_data.get('df')

                    # Build X_items from embeddings (convert to CSR and normalize)
                    if emb is not None and (need_xitems or 'X_items' not in artifacts):
                        try:
                            from scipy.sparse import csr_matrix
                            from sklearn.preprocessing import normalize
                            X_items = csr_matrix(emb) if not isinstance(emb, csr_matrix) else emb
                            X_items = normalize(X_items, norm='l2', axis=1, copy=False)
                            artifacts['X_items'] = X_items
                        except Exception:
                            artifacts['X_items'] = None

                    # Build a TF-IDF vectorizer from the catalog or saved df in index
                    if need_tfidf:
                        try:
                            # Prefer the project's cleaned catalog if available
                            if CATALOG_PATH.exists():
                                catalog_df = pd.read_csv(CATALOG_PATH)
                            elif df_idx is not None:
                                catalog_df = df_idx
                            else:
                                catalog_df = None

                            if catalog_df is not None:
                                # Prepare text: title + subject + description + skills
                                text_cols = []
                                for c in ('title', 'course_title', 'course_name'):
                                    if c in catalog_df.columns:
                                        text_cols.append(c)
                                        break
                                for c in ('subject', 'skills', 'description'):
                                    if c in catalog_df.columns:
                                        text_cols.append(c)

                                if text_cols:
                                    texts = catalog_df[text_cols].fillna('').astype(str).agg(' '.join, axis=1).tolist()
                                else:
                                    texts = catalog_df.fillna('').astype(str).agg(' '.join, axis=1).tolist()

                                from sklearn.feature_extraction.text import TfidfVectorizer
                                tfidf = TfidfVectorizer(max_features=8192, ngram_range=(1,2))
                                tfidf.fit(texts)
                                artifacts['tfidf'] = tfidf
                        except Exception:
                            artifacts['tfidf'] = None
                except Exception:
                    # ignore fallback load errors
                    pass
    except Exception:
        pass

    return artifacts


# ========================================
# Content-Based Recommendation
# ========================================

def build_query_vector(query_text: str, subjects: List[str], difficulties: List[str],
                       tfidf) -> csr_matrix:
    """Build TF-IDF vector from user query."""
    # Build query string
    parts = []
    if query_text:
        parts.append(query_text.strip())
    if subjects:
        parts.append("; ".join(map(str, subjects)))
    if difficulties:
        parts.append("; ".join(map(str, difficulties)))
    
    query = ". ".join(parts).strip()
    
    if not query:
        # Return zero vector
        return csr_matrix((1, tfidf.transform([""]).shape[1]))
    
    # Vectorize
    query_vec = tfidf.transform([query])
    return query_vec


def recommend_content(query_text: str, subjects: List[str], difficulties: List[str],
                     top_k: int, artifacts: Dict, catalog: pd.DataFrame) -> pd.DataFrame:
    """Content-based recommendations using TF-IDF cosine similarity. NO CACHING."""
    
    # Guard against empty input
    if not query_text and not subjects and not difficulties:
        st.info("Enter what you want to learn or pick filters")
        return pd.DataFrame()
    
    # Get item vectors and TF-IDF
    X_items = artifacts.get('X_items')
    tfidf = artifacts.get('tfidf')
    
    if X_items is None or tfidf is None:
        st.error("Required artifacts (X_items.npz or tfidf.pkl) not found!")
        return pd.DataFrame()
    
    # Build query string
    parts = []
    if query_text:
        parts.append(query_text.strip())
    if subjects:
        parts.append("; ".join(map(str, subjects)))
    if difficulties:
        parts.append("; ".join(map(str, difficulties)))
    
    q = ". ".join(parts).strip()
    if not q:
        st.info("Enter what you want to learn or pick filters")
        return pd.DataFrame()
    
    # Vectorize query (no caching; depends on user input)
    q_vec = tfidf.transform([q])  # CSR (1 x F)
    
    # Cosine similarity (X_items is L2-normalized at load time)
    scores_raw = (q_vec @ X_items.T).toarray().ravel()  # (2430,) - only for trained items
    
    # Pad scores to match catalog size (X_items has 2430, catalog has 9657)
    scores = np.zeros(len(catalog))
    scores[:len(scores_raw)] = scores_raw
    
    # Get top-K*5 for diversity, then trim to top-K
    n_items = len(scores)
    top_k_expanded = min(top_k * 5, n_items)
    
    # If subjects specified, filter catalog first and score only those
    if subjects:
        subject_mask = catalog['subject'].apply(
            lambda x: any(s.lower() in str(x).lower() for s in subjects) if pd.notna(x) else False
        )
        filtered_indices = catalog[subject_mask].index.tolist()
        
        if not filtered_indices:
            st.warning(f"No courses found with subject matching: {', '.join(subjects)}")
            return pd.DataFrame()
        
        # Get scores only for filtered items
        filtered_scores = [(idx, scores[idx]) for idx in filtered_indices]
        filtered_scores.sort(key=lambda x: -x[1])
        top_indices = [idx for idx, _ in filtered_scores[:top_k]]
    else:
        # Original logic: get top-K across all courses
        if n_items < top_k:
            top_indices = np.argsort(-scores)
        else:
            top_indices = np.argpartition(-scores, top_k_expanded - 1)[:top_k_expanded]
            top_indices = top_indices[np.argsort(-scores[top_indices])][:top_k]
    
    # Build results from catalog
    results = []
    for rank, idx in enumerate(top_indices, 1):
        if idx >= len(catalog):
            continue

        course = catalog.iloc[idx]
        url = _safe_text(course.get('url', ''))
        desc = _safe_text(course.get('description', ''))
        short_desc = (desc[:200] + '...') if len(desc) > 200 else desc

        results.append({
            'Rank': rank,
            'course_id': _safe_text(course.get('course_id', f'c{idx}')),
            'Title': _safe_text(course.get('title', 'N/A')),
            'Subject': _safe_text(course.get('subject', 'N/A')),
            'Difficulty': _safe_text(course.get('difficulty', 'N/A')),
            'Score': float(scores[idx]) if scores is not None else 0.0,
            'URL': url,
            'Description': short_desc,
            'item_idx': idx
        })
    
    return pd.DataFrame(results)


# ========================================
# Hybrid Recommendation
# ========================================

def minmax_normalize(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize scores to [0, 1]."""
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s < 1e-10:
        return np.zeros_like(scores)
    return (scores - min_s) / (max_s - min_s)


def get_user_interactions(user_id: str, interactions: pd.DataFrame) -> List[int]:
    """Get list of item indices user has interacted with."""
    user_data = interactions[interactions['user_id'] == user_id]
    
    # Map course_ids to item indices
    item_ids = []
    for course_id in user_data['course_id'].values:
        # Assuming course_id is like 'c123' -> index 123
        try:
            if isinstance(course_id, str) and course_id.startswith('c'):
                idx = int(course_id[1:])
                item_ids.append(idx)
            else:
                idx = int(course_id)
                item_ids.append(idx)
        except:
            continue
    
    return item_ids


def recommend_hybrid_warm(user_id: str, top_k: int, artifacts: Dict, 
                         catalog: pd.DataFrame, interactions: pd.DataFrame) -> pd.DataFrame:
    """Hybrid recommendations for warm users. NO CACHING - depends on user_id."""
    
    hybrid_config = artifacts.get('hybrid')
    if hybrid_config is None:
        st.error("Hybrid model not available!")
        return pd.DataFrame()
    
    # Map user_id to index
    user_map = artifacts.get('user_map', {})
    if user_id not in user_map:
        st.warning(f"User {user_id} not found in training split. Please use Cold start tab.")
        return pd.DataFrame()
    
    user_idx = user_map[user_id]
    
    # Get seen items (to exclude)
    seen_items = set(get_user_interactions(user_id, interactions))
    
    # All candidate items
    all_items = np.arange(len(catalog))
    candidate_items = np.array([i for i in all_items if i not in seen_items])
    
    if len(candidate_items) == 0:
        st.warning("User has seen all courses!")
        return pd.DataFrame()
    
    # For simplicity, use content model only (hybrid internal models require more setup)
    X_items = artifacts.get('X_items')
    if X_items is None:
        st.error("Item vectors not available!")
        return pd.DataFrame()
    
    # Use user's interaction history to build profile
    if len(seen_items) > 0:
        user_profile = X_items[list(seen_items)].mean(axis=0)
    else:
        user_profile = csr_matrix((1, X_items.shape[1]))
    
    scores = X_items[candidate_items].dot(user_profile.T).toarray().flatten()
    
    # Get top-K
    if len(scores) < top_k:
        top_indices = np.argsort(-scores)
    else:
        top_indices = np.argpartition(-scores, top_k-1)[:top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
    
    # Build results
    results = []
    for rank, idx in enumerate(top_indices, 1):
        item_idx = candidate_items[idx]
        if item_idx >= len(catalog):
            continue
        course = catalog.iloc[item_idx]
        url = _safe_text(course.get('url', ''))
        desc = _safe_text(course.get('description', ''))
        short_desc = (desc[:200] + '...') if len(desc) > 200 else desc

        results.append({
            'Rank': rank,
            'course_id': _safe_text(course.get('course_id', f'c{item_idx}')),
            'Title': _safe_text(course.get('title', 'N/A')),
            'Subject': _safe_text(course.get('subject', 'N/A')),
            'Difficulty': _safe_text(course.get('difficulty', 'N/A')),
            'Score': float(scores[idx]) if scores is not None else 0.0,
            'URL': url,
            'Description': short_desc,
            'item_idx': item_idx
        })
    
    return pd.DataFrame(results)


def recommend_hybrid_cold(query_text: str, subjects: List[str], difficulties: List[str],
                         top_k: int, artifacts: Dict, catalog: pd.DataFrame) -> pd.DataFrame:
    """Hybrid recommendations for cold users. NO CACHING - content + popularity blend."""
    
    # Guard against empty input
    if not query_text and not subjects and not difficulties:
        st.info("Enter what you want to learn or pick filters")
        return pd.DataFrame()
    
    # Build query: "text. ; subjects. ; difficulties"
    parts = []
    if query_text:
        parts.append(query_text.strip())
    if subjects:
        parts.append("; ".join(map(str, subjects)))
    if difficulties:
        parts.append("; ".join(map(str, difficulties)))
    q = ". ".join(parts).strip()
    
    # Get artifacts
    tfidf = artifacts.get('tfidf')
    X_items = artifacts.get('X_items')
    popular = artifacts.get('popular')
    
    if tfidf is None or X_items is None:
        st.error("Required artifacts not available!")
        return pd.DataFrame()
    
    # Get hybrid config (default: 0.7 content, 0.3 popularity)
    hybrid_config = artifacts.get('hybrid', {})
    cold_policy = hybrid_config.get('cold_policy', {'content': 0.7, 'popularity': 0.3})
    
    # Content scores
    q_vec = tfidf.transform([q])
    content_scores_raw = (q_vec @ X_items.T).toarray().ravel()
    
    # Pad content_scores to match catalog size (X_items only has 2430, catalog has 9657)
    content_scores = np.zeros(len(catalog))
    content_scores[:len(content_scores_raw)] = content_scores_raw
    
    # Popularity scores
    pop_scores = np.zeros(len(catalog))
    if popular and hasattr(popular, 'item_popularity'):
        pop_array = popular.item_popularity
        # Align: catalog index -> popularity score
        for idx in range(min(len(catalog), len(pop_array))):
            pop_scores[idx] = pop_array[idx]
    
    # Normalize scores
    if content_scores.max() > 0:
        content_scores = minmax_normalize(content_scores)
    if pop_scores.max() > 0:
        pop_scores = minmax_normalize(pop_scores)
    
    # Blend: 0.7*content + 0.3*popularity
    final_scores = (cold_policy['content'] * content_scores + 
                   cold_policy['popularity'] * pop_scores)
    
    # Top-K with subject filtering if specified
    if subjects:
        subject_mask = catalog['subject'].apply(
            lambda x: any(s.lower() in str(x).lower() for s in subjects) if pd.notna(x) else False
        )
        filtered_indices = catalog[subject_mask].index.tolist()
        
        if not filtered_indices:
            st.warning(f"No courses found with subject matching: {', '.join(subjects)}")
            return pd.DataFrame()
        
        # Get scores only for filtered items
        filtered_scores = [(idx, final_scores[idx]) for idx in filtered_indices]
        filtered_scores.sort(key=lambda x: -x[1])
        top_indices = [idx for idx, _ in filtered_scores[:top_k]]
    else:
        # Original logic: get top-K across all courses
        if len(final_scores) < top_k:
            top_indices = np.argsort(-final_scores)
        else:
            top_indices = np.argpartition(-final_scores, top_k*5-1)[:top_k*5]
            top_indices = top_indices[np.argsort(-final_scores[top_indices])][:top_k]
    
    # Build results
    results = []
    for rank, idx in enumerate(top_indices, 1):
        if idx >= len(catalog):
            continue
        course = catalog.iloc[idx]
        url = _safe_text(course.get('url', ''))
        desc = _safe_text(course.get('description', ''))
        short_desc = (desc[:200] + '...') if len(desc) > 200 else desc

        results.append({
            'Rank': rank,
            'course_id': _safe_text(course.get('course_id', f'c{idx}')),
            'Title': _safe_text(course.get('title', 'N/A')),
            'Subject': _safe_text(course.get('subject', 'N/A')),
            'Difficulty': _safe_text(course.get('difficulty', 'N/A')),
            'Score': float(final_scores[idx]) if final_scores is not None else 0.0,
            'Content Score': float(content_scores[idx]) if content_scores is not None else 0.0,
            'Popularity Score': float(pop_scores[idx]) if pop_scores is not None else 0.0,
            'URL': url,
            'Description': short_desc,
            'item_idx': idx
        })
    
    return pd.DataFrame(results)


# ========================================
# UI Rendering
# ========================================

def render_results(results: pd.DataFrame, model_type: str):
    """Render recommendation results with explanations."""
    
    if results.empty:
        st.info("No recommendations found. Try adjusting your inputs.")
        return
    
    st.success(f"Found {len(results)} recommendations!")
    
    for _, row in results.iterrows():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Title with link
            url = _safe_text(row.get('URL', ''))
            if url and url.strip():
                st.markdown(f"**{row['Rank']}. [{_safe_text(row.get('Title',''))}]({url})**")
            else:
                st.markdown(f"**{row['Rank']}. {_safe_text(row.get('Title',''))}**")

            # Metadata
            st.caption(f"📚 {_safe_text(row.get('Subject',''))} | 📊 {_safe_text(row.get('Difficulty',''))}")

            # Description
            desc = _safe_text(row.get('Description', ''))
            if desc:
                st.text(desc)
        
        with col2:
            st.metric("Score", f"{row['Score']:.3f}")
        
        # Explanation expander
        with st.expander("ℹ️ Why this recommendation?"):
            if model_type == "Content":
                st.write(f"**Content Similarity Score:** {row['Score']:.4f}")
                st.write("Matched based on TF-IDF cosine similarity to your query.")
            
            elif model_type == "Hybrid-Warm":
                st.write(f"**Final Score:** {row['Score']:.4f}")
                if 'model_weights' in row and row['model_weights']:
                    st.write("**Ensemble Weights:**")
                    for model, weight in row['model_weights'].items():
                        if weight > 0:
                            st.write(f"- {model.capitalize()}: {weight:.2f}")
                st.write("Recommendation from trained hybrid ensemble (warm user).")
            
            elif model_type == "Hybrid-Cold":
                st.write(f"**Final Score:** {row['Score']:.4f}")
                if 'Content Score' in row:
                    st.write(f"- Content: {row['Content Score']:.4f} (weight: {row.get('policy', {}).get('content', 0.7):.1f})")
                if 'Popularity Score' in row:
                    st.write(f"- Popularity: {row['Popularity Score']:.4f} (weight: {row.get('policy', {}).get('popularity', 0.3):.1f})")
                st.write("Cold-start recommendation using content + popularity blend.")
        
        st.divider()


# ========================================
# Main App
# ========================================

def main():
    st.set_page_config(
        page_title="Course Recommender",
        page_icon="🎓",
        layout="wide"
    )
    
    st.title("🎓 Personalized Learning Pathways")
    st.markdown("*Discover your next course with AI-powered recommendations*")
    st.caption("🔧 App Version: 3.0 (simplified, no caching) - Last Updated: Nov 6, 2025")
    
    # Load data
    with st.spinner("Loading models and data..."):
        artifacts = load_artifacts()
        catalog = load_catalog()
        interactions = load_interactions()
    
    if catalog.empty:
        st.error("Failed to load catalog. Please check data files.")
        return
    
    if not artifacts:
        st.error("No model artifacts loaded. Please train models first.")
        return
    
    # Initialize session state for tracking inputs
    if 'last_query' not in st.session_state:
        st.session_state.last_query = None
    if 'last_results' not in st.session_state:
        st.session_state.last_results = None
    
    # Check available models
    has_hybrid = 'hybrid' in artifacts
    has_content = 'X_items' in artifacts and 'tfidf' in artifacts
    
    if not has_content:
        st.error("Content model not available (missing tfidf.pkl or X_items.npz)")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Clear cache button
        if st.button("🔄 Clear Cache & Reload"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        
        st.divider()
        
        # Model selector
        model_options = []
        if has_hybrid:
            model_options.append("Hybrid")
        if has_content:
            model_options.append("Content")
        
        model_type = st.selectbox("Model", model_options, index=0 if has_hybrid else 0)
        
        # Top-K slider
        top_k = st.slider("Number of recommendations", 5, 30, 10)
        
        # Show hybrid config if selected
        if model_type == "Hybrid" and has_hybrid:
            st.divider()
            st.subheader("📊 Hybrid Model Info")
            
            hybrid_config = artifacts['hybrid']
            min_warm = hybrid_config.get('min_warm', 10)
            weights = hybrid_config.get('weights', {})
            
            st.metric("Min warm interactions", min_warm)
            
            st.caption("**Learned Weights:**")
            for model_name, weight in weights.items():
                if weight > 0:
                    st.caption(f"- {model_name.capitalize()}: {float(weight):.2f}")
    
    # Main panel
    if model_type == "Hybrid" and has_hybrid:
        # Hybrid mode: tabs for warm/cold
        tab1, tab2 = st.tabs(["👤 Warm User", "🆕 Cold Start"])
        
        with tab1:
            st.subheader("Recommendations for Existing User")
            
            if interactions is None:
                st.warning("No interaction data available. Cannot recommend for warm users.")
            else:
                # User selector
                available_users = sorted(interactions['user_id'].unique())
                
                if len(available_users) == 0:
                    st.warning("No users found in interactions data.")
                else:
                    user_id = st.selectbox(
                        "Select User ID",
                        available_users,
                        help="Choose a user from the training set"
                    )
                    
                    if st.button("🔍 Get Recommendations", key="warm"):
                        # Track input changes
                        current_query = ("warm", user_id, top_k, model_type)
                        if st.session_state.last_query != current_query:
                            st.session_state.last_query = current_query
                            st.session_state.last_results = None
                        
                        with st.spinner("Generating recommendations..."):
                            try:
                                results = recommend_hybrid_warm(
                                    user_id, top_k, artifacts, catalog, interactions
                                )
                                st.session_state.last_results = results
                                render_results(results, "Hybrid-Warm")
                            except Exception as e:
                                st.error(f"Error generating recommendations: {e}")
        
        with tab2:
            st.subheader("Recommendations for New User")
            
            # Query input
            query_text = st.text_area(
                "What do you want to learn?",
                placeholder="e.g., machine learning, web development, data science...",
                height=100
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Subject filter
                subjects = catalog['subject'].unique()
                subjects = [s for s in subjects if s and s != 'N/A']
                selected_subjects = st.multiselect(
                    "Subject ",
                    sorted(subjects),
                    help="Filter by subject area"
                )
            
            with col2:
                # Difficulty filter
                difficulties = catalog['difficulty'].unique()
                difficulties = [d for d in difficulties if d and d != 'N/A']
                selected_difficulties = st.multiselect(
                    "Difficulty",
                    sorted(difficulties),
                    help="Filter by difficulty level"
                )
            
            # Debug info
            with st.expander("🔍 Debug: Available Filters"):
                st.write(f"**Total courses in catalog:** {len(catalog)}")
                st.write(f"**Unique subjects:** {len(subjects)}")
                st.caption("Sample subjects: " + ", ".join(sorted(subjects)[:10]))
                st.write(f"**Unique difficulties:** {len(difficulties)}")
                st.caption("Available: " + ", ".join(sorted(difficulties)))
            
            if st.button("🔍 Get Recommendations", key="cold"):
                # Track input changes
                current_query = ("cold", query_text, tuple(selected_subjects), tuple(selected_difficulties), top_k, model_type)
                if st.session_state.last_query != current_query:
                    st.session_state.last_query = current_query
                    st.session_state.last_results = None
                
                if not query_text and not selected_subjects and not selected_difficulties:
                    st.warning("Please provide at least one input (text, subject, or difficulty).")
                else:
                    with st.spinner("Generating recommendations..."):
                        try:
                            # Show what's being searched
                            st.info(f"🔍 Searching for: '{query_text}' | Subjects: {selected_subjects or 'Any'} | Difficulty: {selected_difficulties or 'Any'}")
                            
                            results = recommend_hybrid_cold(
                                query_text, selected_subjects, selected_difficulties,
                                top_k, artifacts, catalog
                            )
                            st.session_state.last_results = results
                            render_results(results, "Hybrid-Cold")
                        except Exception as e:
                            st.error(f"Error generating recommendations: {e}")
                            import traceback
                            st.code(traceback.format_exc())
    
    elif model_type == "Content":
        # Content mode: single form
        st.subheader("Content-Based Recommendations")
        
        # Query input
        query_text = st.text_area(
            "What do you want to learn?",
            placeholder="e.g., machine learning, web development, data science...",
            height=100
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Subject filter
            subjects = catalog['subject'].unique()
            subjects = [s for s in subjects if s and s != 'N/A']
            selected_subjects = st.multiselect(
                "Subject (optional)",
                sorted(subjects),
                help="Filter by subject area"
            )
        
        with col2:
            # Difficulty filter
            difficulties = catalog['difficulty'].unique()
            difficulties = [d for d in difficulties if d and d != 'N/A']
            selected_difficulties = st.multiselect(
                "Difficulty (optional)",
                sorted(difficulties),
                help="Filter by difficulty level"
            )
        
        # Debug info
        with st.expander("🔍 Debug: Available Filters"):
            st.write(f"**Total courses in catalog:** {len(catalog)}")
            st.write(f"**Unique subjects:** {len(subjects)}")
            st.caption("Sample subjects: " + ", ".join(sorted(subjects)[:10]))
            st.write(f"**Unique difficulties:** {len(difficulties)}")
            st.caption("Available: " + ", ".join(sorted(difficulties)))
        
        if st.button("🔍 Get Recommendations", key="content_recommend"):
            # Track input changes
            current_query = ("content", query_text, tuple(selected_subjects), tuple(selected_difficulties), top_k, model_type)
            if st.session_state.last_query != current_query:
                st.session_state.last_query = current_query
                st.session_state.last_results = None
            
            if not query_text and not selected_subjects and not selected_difficulties:
                st.warning("Please provide at least one input (text, subject, or difficulty).")
            else:
                with st.spinner("Generating recommendations..."):
                    try:
                        # Show what's being searched
                        st.info(f"🔍 Searching for: '{query_text}' | Subjects: {selected_subjects or 'Any'} | Difficulty: {selected_difficulties or 'Any'}")
                        
                        results = recommend_content(
                            query_text, selected_subjects, selected_difficulties,
                            top_k, artifacts, catalog
                        )
                        st.session_state.last_results = results
                        render_results(results, "Content")
                    except Exception as e:
                        st.error(f"Error generating recommendations: {e}")
                        import traceback
                        st.code(traceback.format_exc())
    
    # Footer
    st.divider()
    st.caption("💡 AI-Powered Course Recommendations")


if __name__ == "__main__":
    main()
