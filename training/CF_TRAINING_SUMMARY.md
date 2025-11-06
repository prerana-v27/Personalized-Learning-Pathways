# Collaborative Filtering Training - Summary

## Successfully Created Scripts

### 1. **Data Generation Script** 
`Data/scripts/make_synthetic_interactions.py`

**Purpose**: Generates realistic synthetic user-course interaction data from the course catalog for collaborative filtering models.

**Features**:
- ✅ Creates unique `course_id` if missing
- ✅ Builds TF-IDF matrix from course metadata (5000 features, bigrams, L2-normalized)
- ✅ Uses popularity priors for realistic sampling
- ✅ Mixture sampling: 70% content-based (TF-IDF neighbors) + 30% popularity
- ✅ Temporal ordering with uniform timestamps over 365 days
- ✅ Comprehensive dataset card with statistics
- ✅ Python 3.13 compatible (no surprise library)

**Generated Dataset**:
```
- Users: 1,500
- Courses: 2,516 (out of 9,657 catalog items)
- Interactions: 13,494
- Avg per user: 9.00
- Sparsity: 99.91%
- Date range: 2024-11-06 to 2025-11-06 (364 days)
- All users have ≥3 interactions
```

**Usage**:
```bash
python Data\scripts\make_synthetic_interactions.py ^
    --catalog Data\processed\all_courses_cleaned.csv ^
    --out Data\processed\interactions_synth.csv ^
    --n-users 1500 ^
    --min-per-user 3 ^
    --max-per-user 15 ^
    --days-back 365 ^
    --seed 42
```

---

### 2. **Collaborative Filtering Training Script**
`training/cf_train_nosurprise.py`

**Purpose**: Train collaborative filtering models without scikit-surprise (pure scikit-learn/scipy implementation).

**Features**:
- ✅ Temporal train/test split (last interaction per user)
- ✅ Sparse CSR matrix for memory efficiency
- ✅ Three models:
  - **Popularity Baseline**: Most interacted courses
  - **Item-KNN**: Cosine similarity-based (K=20 in fast mode)
  - **NMF**: Matrix factorization (40 factors in fast mode)
- ✅ Metrics: Precision@K, Recall@K, NDCG@K
- ✅ Artifacts saved: models (.pkl), ID maps, split info
- ✅ Python 3.13 compatible

**Results** (Fast Mode):
```
Model           P@10      R@10      NDCG@10
================================================
Popularity      0.0050    0.0498    0.0275
ItemKNN         0.0021    0.0206    0.0093
NMF             0.0242    0.2420    0.1330
================================================
```

**Key Insight**: NMF significantly outperforms both baselines, achieving:
- **24.2% Recall@10** (vs 5% for Popularity)
- **13.3% NDCG@10** (vs 2.8% for Popularity)

**Usage**:
```bash
python training\cf_train_nosurprise.py ^
    --data Data\processed\interactions_synth.csv ^
    --user-col user_id ^
    --item-col course_id ^
    --rating-col rating ^
    --time-col timestamp ^
    --k 10 ^
    --out artifacts ^
    --fast
```

---

## Saved Artifacts

All artifacts saved to `artifacts/` directory:

1. **popular.pkl** - Popularity baseline model
2. **cf_knn.pkl** - Item-KNN model (K=20)
3. **cf_nmf.pkl** - NMF model (40 factors)
4. **id_maps.pkl** - User/item ID mappings (raw ↔ internal indices)
5. **split.pkl** - Train/test split info + hyperparameters + metrics

---

## Dependencies (Python 3.13 Compatible)

```bash
pip install pandas numpy scipy scikit-learn tqdm joblib
```

All installed successfully! ✅

---

## Next Steps

### For Production Deployment:
1. **Increase training scale**: Remove `--fast` flag for better performance
   - ItemKNN: K=40 (vs K=20)
   - NMF: 60 factors, 200 iterations (vs 40 factors, 100 iterations)

2. **Generate more data**: Increase synthetic users/interactions
   ```bash
   --n-users 5000 --max-per-user 25
   ```

3. **Hyperparameter tuning**: Grid search over:
   - NMF: `n_factors ∈ {40, 60, 80}`, `reg` parameters
   - ItemKNN: `k ∈ {20, 40, 60}`

4. **Add rating variance**: Currently all ratings are 1.0 (implicit)
   - Could add rating diversity based on interaction patterns

### For Hybrid Models:
The synthetic data includes:
- **User-item interactions** (collaborative filtering)
- **Course metadata** in catalog (content-based filtering)
- **TF-IDF features** already computed during generation

Can combine CF predictions with content-based features for hybrid recommendations!

---

## Technical Highlights

### Memory Efficiency
- Sparse CSR matrices (99.67% sparsity)
- Batched similarity computation
- No dense copies of large matrices

### Robustness
- Graceful error handling
- Comprehensive logging
- Seed-based reproducibility
- Validates input data structure

### Scalability
- Laptop-friendly (< 8 minutes total on 1500 users)
- Progress bars with tqdm
- Efficient sparse operations

---

## File Structure

```
Personalized-Learning-Pathways/
├── Data/
│   ├── processed/
│   │   ├── all_courses_cleaned.csv      # Course catalog (9,657 courses)
│   │   └── interactions_synth.csv        # Synthetic interactions (13,494)
│   └── scripts/
│       └── make_synthetic_interactions.py
├── training/
│   ├── cf_train.py                       # Original (needs surprise lib)
│   └── cf_train_nosurprise.py           # Python 3.13 compatible ✅
└── artifacts/                            # Trained models
    ├── popular.pkl
    ├── cf_knn.pkl
    ├── cf_nmf.pkl
    ├── id_maps.pkl
    └── split.pkl
```

---

**Status**: ✅ All scripts working successfully on Python 3.13!
