# Content-Based & Clustering Recommenders - Results

## Model Comparison

### Performance Summary (Top-10 Recommendations)

| Model | Precision@10 | Recall@10 | NDCG@10 | Notes |
|-------|--------------|-----------|---------|-------|
| **Content-based** | **0.0267** | **0.2669** | **0.1494** | 🏆 Best overall |
| NMF (CF) | 0.0242 | 0.2420 | 0.1330 | Strong collaborative |
| Cluster (K=100) | 0.0166 | 0.1658 | 0.0839 | Good for cold-start |
| Popularity (CF) | 0.0050 | 0.0498 | 0.0275 | Baseline |
| ItemKNN (CF) | 0.0021 | 0.0206 | 0.0093 | Underperformed |

## Key Findings

### 🏆 Winner: Content-Based Recommender
- **26.7% Recall@10** - Highest of all models
- **14.9% NDCG@10** - Best ranking quality
- Leverages rich course metadata (title, description, skills, subject)
- Works well for users with diverse interaction histories

### 🥈 Runner-up: NMF (Collaborative Filtering)
- **24.2% Recall@10** - Very close to Content
- **13.3% NDCG@10** - Strong ranking
- Discovers latent patterns in user-course interactions
- Complementary to content-based approach

### 🎯 Hybrid Potential
Combining Content + NMF could achieve:
- **30%+ Recall@10** (estimated)
- Better coverage for both cold-start and popular items
- Personalization + diversity

---

## Technical Implementation

### Content-Based Model

**Architecture**:
```
User Profile = Mean(TF-IDF vectors of interacted courses)
Score(user, course) = Cosine(user_profile, course_vector)
```

**Features**:
- TF-IDF on combined text: `title + description + skills + subject`
- 5,000 features, bigrams (1,2), English stopwords
- L2-normalized sparse CSR matrices (99.76% sparsity)
- Pre-computed item norms for efficiency

**Strengths**:
- ✅ No cold-start problem (uses course metadata)
- ✅ Interpretable recommendations
- ✅ Fast inference (< 1ms per user)
- ✅ Works with small interaction histories

**Artifacts Saved**:
- `tfidf.pkl` - TF-IDF vectorizer
- `X_items.npz` - Item TF-IDF matrix (2,430 × 5,000)
- `content_index.pkl` - Cached user profiles + item norms
- `content_meta.json` - Configuration metadata

---

### Clustering Model (KMeans)

**Architecture**:
```
1. Cluster items into K groups (K=100, selected via validation)
2. For each user, find closest cluster centroid
3. Score = centroid_similarity × (popularity_z + 1)
```

**Validation Process**:
- Tested K ∈ {50, 100} on internal validation split
- K=50: NDCG@10 = 0.0715
- **K=100: NDCG@10 = 0.0875** ✅ Selected

**Scoring Formula**:
```python
centroid_sim = cosine(user_profile, cluster_centroid)
popularity_z = (item_popularity - mean) / std
score = centroid_sim × (popularity_z + 1)
```

**Strengths**:
- ✅ Balances relevance + popularity
- ✅ Efficient for large catalogs
- ✅ Captures coarse-grained content patterns
- ✅ Good for exploration (intra-cluster diversity)

**Weaknesses**:
- ❌ Lower precision than pure content model
- ❌ Popularity bias can dominate
- ❌ Hard cluster assignment loses nuance

**Artifacts Saved**:
- `kmeans.pkl` - Trained KMeans model (100 clusters)
- Item statistics: popularity, recency

---

## Dataset Statistics

### Training Split (Reused from CF)
```
Train interactions:  11,994
Test interactions:    1,500 (last per user)
Users:                1,500
Items:                2,430 (out of 9,657 catalog)
Sparsity:            99.67%
```

### Item Statistics
```
Popularity range:     [1, 111] interactions
Recency range:        [0, 364] days
Most popular course:  c008808 (128 interactions)
```

### User Profiles
```
All 1,500 users have valid profiles
Profile sparsity:     ~99.0% (5,000 dimensions)
Avg profile norm:     Non-zero for all users
```

---

## Memory & Performance

### Computational Efficiency

| Operation | Time | Memory |
|-----------|------|--------|
| TF-IDF matrix build | 1s | ~60 MB (sparse) |
| User profile build | 3s | ~30 MB |
| KMeans K=100 | 4s | ~120 MB |
| Content eval (1,500 users) | 0.6s | Minimal |
| Cluster eval (1,500 users) | 0.8s | Minimal |

**Total Training Time**: ~30 seconds (laptop-friendly!)

### Scalability Notes
- Sparse matrices throughout (no dense copies)
- Batched similarity computation
- Pre-computed norms for inference speedup
- Can handle 10K+ items with same approach

---

## Recommendations for Production

### 1. **Hybrid Ensemble** (Recommended)
```python
final_score = 0.5 × content_score + 0.3 × nmf_score + 0.2 × cluster_score
```

**Why?**
- Combines strengths of all approaches
- Content: interpretable, cold-start friendly
- NMF: discovers hidden patterns
- Cluster: adds diversity

### 2. **Re-ranking Strategy**
```python
# Stage 1: Content retrieves top-100 candidates
candidates = content_model.recommend(user, k=100)

# Stage 2: Re-rank with NMF
final_recs = nmf_model.rerank(candidates, k=10)
```

**Benefits**:
- Fast content-based retrieval
- Personalized re-ranking with CF
- Best of both worlds

### 3. **Context-Aware Weighting**
```python
if user_interactions < 5:
    # Cold users: favor content
    α_content, α_cf = 0.7, 0.3
else:
    # Warm users: favor CF
    α_content, α_cf = 0.3, 0.7
```

---

## Comparison with CF Models

### Content vs NMF (CF)

| Aspect | Content | NMF |
|--------|---------|-----|
| Data dependency | Catalog only | Interactions required |
| Cold-start | ✅ Excellent | ❌ Poor |
| Personalization | 🟡 Medium | ✅ Strong |
| Diversity | ✅ High | 🟡 Medium |
| Interpretability | ✅ Clear | ❌ Opaque |
| Serendipity | 🟡 Medium | ✅ High |

### When to Use Each

**Content-based**:
- ✅ New users (< 5 interactions)
- ✅ Niche/specific course searches
- ✅ Need explainability
- ✅ Course catalog changes frequently

**NMF (CF)**:
- ✅ Established users (> 10 interactions)
- ✅ Discovery recommendations
- ✅ Stable interaction patterns
- ✅ Large user base

**Clustering**:
- ✅ Very large catalogs (100K+ items)
- ✅ Need fast approximate retrieval
- ✅ Exploration-focused recommendations
- ✅ Category-aware suggestions

---

## Artifacts Directory Structure

```
artifacts/
├── # Collaborative Filtering
│   ├── cf_knn.pkl                  # Item-KNN model
│   ├── cf_nmf.pkl                  # NMF model
│   ├── popular.pkl                 # Popularity baseline
│   ├── id_maps.pkl                 # User/item ID mappings
│   └── split.pkl                   # Train/test split info
│
├── # Content-Based & Clustering
│   ├── tfidf.pkl                   # TF-IDF vectorizer
│   ├── X_items.npz                 # Item TF-IDF matrix (sparse)
│   ├── content_index.pkl           # User profiles + item norms
│   ├── content_meta.json           # Configuration metadata
│   └── kmeans.pkl                  # KMeans clustering model
```

---

## Next Steps

### Immediate Improvements
1. **Test hybrid model** combining Content + NMF
2. **Add diversity metrics** (intra-list diversity, coverage)
3. **A/B test** against popularity baseline
4. **Cold-start analysis** (users with 1-3 interactions)

### Advanced Features
1. **Session-based recommendations** using RNNs/Transformers
2. **Multi-objective optimization** (relevance + diversity + novelty)
3. **Contextual bandits** for online learning
4. **Course sequence recommendations** (learning pathways)

### Model Monitoring
- Track precision/recall over time
- Monitor cold-start vs warm-start performance
- Measure recommendation diversity
- User engagement metrics (CTR, completion rate)

---

## Conclusion

✅ **Content-based model achieves best overall performance** (26.7% Recall@10)

✅ **Both models complement each other** - ideal for hybrid approach

✅ **Production-ready implementation** - fast, memory-efficient, well-documented

✅ **Comprehensive artifact suite** - models, data, and metadata all saved

**Recommendation**: Deploy hybrid model with content-based retrieval + CF re-ranking for optimal results! 🚀
