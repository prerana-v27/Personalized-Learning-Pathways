# 🔮 Hybrid Ensemble Training - Overview

## What It Does

The `hybrid_train.py` script creates an **ensemble recommender** that intelligently blends multiple trained models to achieve better performance than any single model alone.

## Key Features

### 1. **Smart Model Loading** ✅
- Loads all available trained models from `artifacts/`
- Required: TF-IDF vectorizer, content index, popularity model, ID maps
- Optional: NMF, Item-KNN, Supervised Ranker, KMeans clustering
- Handles both joblib and pickle formats automatically

### 2. **Warm vs Cold User Strategy** 🌡️

**Warm Users** (≥10 train interactions):
- Uses optimized ensemble weights from grid search
- Blends: NMF + KNN + Supervised + Content (+ optional Cluster)
- Weights optimized for NDCG@10 on validation set

**Cold Users** (<10 train interactions):
- Fixed policy: **0.7 × Content + 0.3 × Popularity + small recency boost**
- Relies on content-based filtering for users with little history
- Popularity provides fallback recommendations

### 3. **Weight Optimization** 🎯
- Creates validation split from training data (last interaction per user)
- Grid searches weight combinations with step=0.25
- Excludes all-zero combinations
- Optimizes for NDCG@10 on warm validation users
- Typical search space: ~125-625 combinations

### 4. **Min-Max Normalization** 📊
- Normalizes each model's scores to [0, 1] range per user
- Ensures fair combination of different scoring scales
- Handles edge cases (constant scores → zeros)

### 5. **Comprehensive Evaluation** 📈
- Evaluates on official test split (same as individual models)
- Reports metrics for:
  - **Overall**: All test users
  - **Warm**: Users with ≥10 train interactions  
  - **Cold**: Users with <10 train interactions
- Metrics: Precision@K, Recall@K, NDCG@K

### 6. **Production-Ready Artifacts** 💾
- Saves `hybrid.pkl` with:
  - Optimized weights for warm users
  - Cold user policy parameters
  - Min-warm threshold
  - Performance metrics
- Generates `HYBRID_CARD.md` with:
  - Model card documentation
  - Weight breakdown
  - Performance summary
  - Usage examples

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    HYBRID ENSEMBLE                       │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │           Is User Warm (≥10 interactions)?         │ │
│  └───────────┬──────────────────────────┬─────────────┘ │
│              │                          │                │
│         ┌────▼────┐              ┌──────▼──────┐        │
│         │  WARM   │              │    COLD     │        │
│         │ POLICY  │              │   POLICY    │        │
│         └────┬────┘              └──────┬──────┘        │
│              │                          │                │
│         ┌────▼────────────┐        ┌───▼────────────┐   │
│         │ Optimized Blend │        │  Fixed Blend   │   │
│         │                 │        │                │   │
│         │ α × NMF         │        │ 0.7 × Content  │   │
│         │ β × KNN         │        │ 0.3 × Popular  │   │
│         │ γ × Supervised  │        │ 0.1 × Recency  │   │
│         │ δ × Content     │        │                │   │
│         │ ε × Cluster*    │        │                │   │
│         └─────────────────┘        └────────────────┘   │
│                                                          │
│         * optional if kmeans.pkl available              │
└─────────────────────────────────────────────────────────┘
```

## Usage

### Basic Command

```bash
python training/hybrid_train.py \
    --catalog Data/processed/all_courses_cleaned.csv \
    --interactions Data/processed/interactions_synth.csv \
    --k 10 \
    --min-warm 10 \
    --out artifacts
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--catalog` | str | **required** | Path to course catalog CSV |
| `--interactions` | str | **required** | Path to interactions CSV |
| `--k` | int | 10 | Top-K for evaluation |
| `--min-warm` | int | 10 | Min train interactions for warm user |
| `--out` | str | artifacts | Output directory for artifacts |

### Output Files

1. **`artifacts/hybrid.pkl`** - Ensemble configuration
   ```python
   {
       'weights': {'nmf': 0.25, 'supervised': 0.50, ...},
       'min_warm': 10,
       'k': 10,
       'cold_policy': {...},
       'metrics': {
           'overall': {...},
           'warm': {...},
           'cold': {...}
       }
   }
   ```

2. **`artifacts/HYBRID_CARD.md`** - Model card documentation

## Expected Performance

Based on typical ensemble learning improvements:

| Metric | Individual Best | Hybrid Ensemble | Improvement |
|--------|-----------------|-----------------|-------------|
| **Recall@10 (Overall)** | 29.6% | **30-32%** | +0.4-2.4% |
| **Recall@10 (Warm)** | 29.6% | **31-33%** | +1.4-3.4% |
| **Recall@10 (Cold)** | 26.7% | **27-28%** | +0.3-1.3% |
| **NDCG@10 (Overall)** | 16.4% | **17-18%** | +0.6-1.6% |

### Why Ensemble Works

1. **Diverse Strengths**:
   - NMF: Captures latent collaborative patterns
   - Content: Handles cold-start and topic relevance
   - Supervised: Learns optimal feature combinations
   - KNN: Captures local similarities

2. **Error Diversification**:
   - Models make different mistakes
   - Ensemble averages out errors
   - Reduces overfitting to any single approach

3. **Complementary Coverage**:
   - Content works for new/niche items
   - Collaborative works for popular items
   - Supervised learns when to trust each

## Training Time

Approximate timing (with all models loaded):

| Step | Time | Description |
|------|------|-------------|
| **Loading** | ~5 sec | Load all artifacts and models |
| **Data Prep** | ~10 sec | Build user profiles, split data |
| **Validation Split** | ~1 sec | Create validation set |
| **Grid Search** | **10-30 min** | Search 125-625 weight combinations |
| **Test Evaluation** | ~5-10 min | Evaluate on test split |
| **Saving** | ~1 sec | Save configuration and card |
| **Total** | **15-40 min** | Depends on available models |

## Implementation Details

### Score Combination

For each user-item pair:
```python
# Warm users
score = (α × normalize(nmf_score) +
         β × normalize(knn_score) +
         γ × normalize(supervised_score) +
         δ × normalize(content_score))

# Cold users
score = (0.7 × normalize(content_score) +
         0.3 × normalize(popularity_score) +
         0.1 × recency_boost)
```

### Normalization

```python
def minmax_normalize(scores):
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s < 1e-10:
        return zeros_like(scores)
    return (scores - min_s) / (max_s - min_s)
```

### Grid Search

```python
for w_nmf in [0.0, 0.25, 0.50, 0.75, 1.0]:
    for w_knn in [0.0, 0.25, 0.50, 0.75, 1.0]:
        for w_supervised in [0.0, 0.25, 0.50, 0.75, 1.0]:
            for w_content in [0.0, 0.25, 0.50, 0.75, 1.0]:
                if sum([w_nmf, w_knn, w_supervised, w_content]) == 0:
                    continue  # Skip all-zero
                
                # Evaluate on validation set
                ndcg = evaluate_weights({
                    'nmf': w_nmf,
                    'knn': w_knn,
                    'supervised': w_supervised,
                    'content': w_content
                })
                
                if ndcg > best_ndcg:
                    best_weights = current_weights
```

## Deployment Recommendations

### Option 1: Hybrid Only (Simplest)
```python
# Use hybrid for all recommendations
recommendations = hybrid_ensemble.recommend(user_id, k=10)
```

**Pros**: Single model, optimal performance  
**Cons**: Slower than individual models (~10-20ms per user)

### Option 2: Two-Stage (Fastest)
```python
# Stage 1: Fast retrieval with content (100 candidates)
candidates = content_model.recommend(user_id, k=100)

# Stage 2: Re-rank with hybrid
final_recs = hybrid_ensemble.rerank(candidates, k=10)
```

**Pros**: Fast (<5ms), high quality  
**Cons**: Two models to maintain

### Option 3: Adaptive Policy (Most Flexible)
```python
if user_interaction_count >= 50:
    # Mature users: Use supervised ranker (best single model)
    return supervised_ranker.recommend(user_id, k=10)
elif user_interaction_count >= 10:
    # Warm users: Use hybrid ensemble
    return hybrid_ensemble.recommend(user_id, k=10)
else:
    # Cold users: Use content model (fastest)
    return content_model.recommend(user_id, k=10)
```

**Pros**: Optimized for each user segment  
**Cons**: Most complex, three models

## Troubleshooting

### Grid Search Takes Too Long
- Reduce step size: `--step 0.5` (in code modification)
- Only search 2-3 key models (skip KNN if underperforming)
- Use smaller validation set

### Memory Issues
- Models are loaded in memory during grid search
- Requires ~2-4GB RAM with all models
- Consider disabling optional models (KNN, Cluster)

### Negative Weights
- Grid search only considers non-negative weights
- Ensures interpretable ensemble
- All weights ∈ [0, 1]

## Next Steps

After training hybrid ensemble:

1. ✅ **Compare with Individual Models**
   - Check if ensemble > best single model
   - Analyze warm vs cold performance gap

2. ✅ **A/B Test in Production**
   - Deploy hybrid vs current best model
   - Measure CTR, engagement, diversity

3. ✅ **Monitor Performance**
   - Track metrics over time
   - Re-train when data distribution shifts

4. ✅ **Experiment with Advanced Ensembles**
   - Stacking: Train meta-learner on model outputs
   - Boosting: Sequential ensemble with error correction
   - Context-aware: Different weights per user segment

---

*Created: November 6, 2025*  
*For: Personalized Learning Pathways*  
*Status: PRODUCTION READY*
