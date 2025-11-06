# 🔮 Hybrid Ensemble Model Card

## Overview

This hybrid ensemble blends multiple recommendation models with optimized weights.

## Configuration

- **Min warm interactions**: 10
- **Evaluation K**: 10

## Ensemble Weights (Warm Users)

Optimized via grid search on validation split:

| Model | Weight |
|-------|--------|
| Content | 0.25 |

## Cold User Policy

Fixed blend for users with <10 train interactions:

- **Content**: 0.70
- **Popularity**: 0.30
- **Recency boost**: 0.10

## Performance Metrics

### Overall (All Users)

- **Precision@10**: 0.0000
- **Recall@10**: 0.0000
- **NDCG@10**: 0.0000

### Warm Users (≥10 interactions)

- **Precision@10**: 0.0000
- **Recall@10**: 0.0000
- **NDCG@10**: 0.0000

### Cold Users (<10 interactions)

- **Precision@10**: 0.0000
- **Recall@10**: 0.0000
- **NDCG@10**: 0.0000

## Usage

```python
import pickle

# Load configuration
with open('artifacts/hybrid.pkl', 'rb') as f:
    config = pickle.load(f)

weights = config['weights']
min_warm = config['min_warm']
```

*Generated: November 6, 2025*
