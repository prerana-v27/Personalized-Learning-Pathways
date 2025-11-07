# Quick Reference Card 🚀

## Model Performance Rankings

| 🏆 Rank | Model | Recall@10 | Type |
|---------|-------|-----------|------|
| 🥇 #1 | Content | **26.7%** | Content-based |
| 🥈 #2 | NMF | 24.2% | Collaborative |
| 🥉 #3 | Cluster | 16.6% | Content-based |
| #4 | Popularity | 5.0% | Baseline |
| #5 | Item-KNN | 2.1% | Collaborative |

## Files Created ✅

### Scripts
- `Data/scripts/make_synthetic_interactions.py` - Data generator
- `training/cf_train_nosurprise.py` - CF training (Python 3.13)
- `training/content_cluster_train.py` - Content & clustering

### Models (artifacts/)
- `cf_nmf.pkl` - NMF collaborative filtering
- `cf_knn.pkl` - Item-KNN collaborative filtering
- `popular.pkl` - Popularity baseline
- `tfidf.pkl` - TF-IDF vectorizer
- `kmeans.pkl` - KMeans clustering
- `content_index.pkl` - User profiles + item norms
- `id_maps.pkl` - User/item mappings
- `split.pkl` - Train/test split
- `X_items.npz` - Item TF-IDF matrix
- `content_meta.json` - Metadata

### Documentation
- `training/CF_TRAINING_SUMMARY.md` - CF details
- `training/CONTENT_RESULTS.md` - Content model analysis
- `training/FINAL_SUMMARY.md` - Complete overview

## Command Cheatsheet

### Generate Data
```bash
python Data\scripts\make_synthetic_interactions.py --catalog Data\processed\all_courses_cleaned.csv --out Data\processed\interactions_synth.csv --n-users 1500 --min-per-user 3 --max-per-user 15 --days-back 365 --seed 42
```

### Train CF Models
```bash
python training\cf_train_nosurprise.py --data Data\processed\interactions_synth.csv --user-col user_id --item-col course_id --rating-col rating --time-col timestamp --k 10 --out artifacts --fast
```

### Train Content Models
```bash
python training\content_cluster_train.py --catalog Data\processed\all_courses_cleaned.csv --interactions Data\processed\interactions_synth.csv --k 10 --out artifacts
```

## Load Models (Python)

```python
import joblib
import numpy as np
from scipy.sparse import load_npz

# Load all models
nmf = joblib.load('artifacts/cf_nmf.pkl')
knn = joblib.load('artifacts/cf_knn.pkl')
popularity = joblib.load('artifacts/popular.pkl')
kmeans = joblib.load('artifacts/kmeans.pkl')
tfidf = joblib.load('artifacts/tfidf.pkl')
content_index = joblib.load('artifacts/content_index.pkl')
id_maps = joblib.load('artifacts/id_maps.pkl')
split = joblib.load('artifacts/split.pkl')

# Load sparse matrix
X_items = load_npz('artifacts/X_items.npz')
```

## Key Stats

- **Users**: 1,500
- **Courses**: 2,516 (from 9,657 catalog)
- **Interactions**: 13,494
- **Sparsity**: 99.67%
- **Training time**: ~10 minutes total
- **Python version**: 3.13 compatible

## Next Steps

1. ✅ Implement hybrid model (Content + NMF)
2. ✅ Build REST API for serving
3. ✅ Add A/B testing framework
4. ✅ Monitor model performance
5. ✅ Deploy to production

**Status**: All models trained and ready! 🎉
