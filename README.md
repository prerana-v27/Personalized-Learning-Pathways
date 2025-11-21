# 🎓 Personalized Learning Pathways

A machine learning-powered platform for course difficulty prediction and personalized learning recommendations across multiple online learning platforms.

## 🎯 Project Overview

This project combines course data from major online learning platforms (Coursera, edX, NPTEL, and Udemy) to:
1. Predict course difficulty levels (Beginner, Intermediate, Advanced)
2. Create personalized learning pathways based on user profiles
3. Provide intelligent course recommendations across platforms

### Current Features
- Multi-source course data integration
- Machine learning-based difficulty classification
- Automated data preprocessing pipeline
- Course metadata normalization
- JSON → CSV conversion utilities

### New: Content-based Recommender

This repository now includes a simple content-based recommender in `ml/recommender.py`.

- Uses SentenceTransformers (SBERT) embeddings when available, with a TF-IDF fallback.
- Indexes course embeddings with nearest-neighbor search for fast retrieval.
- Provides a CLI to build/save an index and query by course title or row id.

Quick examples (PowerShell):
```powershell
# Build and save index (recommended)
python ml\recommender.py --csv Data\processed\all_courses_merged.csv --build-index --index-out ml\outputs\recommender_index.joblib

# Query using saved index by title
python ml\recommender.py --csv Data\processed\all_courses_merged.csv --index-in ml\outputs\recommender_index.joblib --title "Introduction to Python" --topk 10

# Query by row id (0-based)
python ml\recommender.py --csv Data\processed\all_courses_merged.csv --id 42 --topk 10
```

If `sentence-transformers` is installed the recommender will use SBERT (`all-MiniLM-L6-v2`) for best results. Otherwise it falls back to TF-IDF.

See `ml/requirements-ml.txt` for recommended packages.

## 🛠️ Project Structure

```
Personalized-Learning-Pathways/
├── Data/
│   ├── raw/               # Original data files
│   │   ├── Coursera.csv
│   │   ├── EdX.csv
│   │   ├── NPTEL.csv
│   │   └── UdemyCleanedTitle.csv
│   ├── processed/         # Cleaned and merged datasets
│   └── scripts/           # Data processing utilities (jsonToCsv.py)
├── ml/                   # Machine learning and recommender code
│   ├── merge_datasets.py  # Dataset merger
│   ├── train.py           # Model training (difficulty classifier)
│   ├── recommender.py     # Content-based recommender (embeddings + NN)
│   ├── requirements-ml.txt# ML/recommender dependencies
│   └── outputs/           # Trained models, indexes, and reports
└── notebooks/             # Analysis notebooks
```

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- pip package manager

### Installation

1. Clone the repository:
```powershell
git clone https://github.com/prerana-v27/Personalized-Learning-Pathways.git
cd Personalized-Learning-Pathways
```

2. Create and activate a virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # PowerShell
```

3. Install core dependencies:
```powershell
pip install --upgrade pip
pip install -r ml\requirements-ml.txt
```

> Note: `sentence-transformers` is optional but recommended for the recommender.

### Usage

1. Convert any JSON raw data (if present):
```powershell
python Data\scripts\jsonToCsv.py
```

2. Merge datasets (ML pipeline expects a merged CSV):
```powershell
python ml\merge_datasets.py
```

3. Train the difficulty classifier (example):
```powershell
python ml\train.py --csv_path data\processed\all_courses_merged.csv --out_dir ml\outputs --target_class difficulty
```

4. Build and query recommender (see examples above).

## 📈 Model Performance (current baseline)

- Accuracy: ~75.10%
- Macro F1: ~0.7456

These are for the current LinearSVC-based classifier (see `README-ML.md` for full experiment details). The recommender has its own evaluation steps planned (Recall@K / NDCG@K).

## 🧭 Algorithms & where they're used

This section maps the main algorithms and models used in the project to the files and runtime components where they are trained, saved, and consumed.

- TF-IDF (sklearn.feature_extraction.text.TfidfVectorizer)
	- Trained: `ml/make_streamlit_artifacts.py` (or during `ml/recommender.py` index build)
	- Saved artifacts: `artifacts/tfidf.pkl`, `artifacts/X_items.npz`
	- Used by: `app/streamlit_app.py` (content and cold-hybrid scorers), `ml/recommender.py` (fallback embedding)

- SentenceTransformers (SBERT)
	- Trained model: pre-trained `all-MiniLM-L6-v2` (downloaded by `sentence-transformers`)
	- Used by: `ml/recommender.py` when available to produce semantic embeddings for nearest-neighbour retrieval

- HashingVectorizer + LinearSVC (text classifier)
	- Implemented in: `ml/train.py` (preprocessing pipeline described in `README-ML.md`)
	- Saved model: `ml/outputs/model_classif.joblib`
	- Used for: course difficulty classification experiments and batch inference

- ColumnTransformer, SimpleImputer, OneHotEncoder
	- Used in: `ml/train.py` preprocessing pipeline for numeric/categorical/text branches

- Content-based TF-IDF retrieval (cosine similarity)
	- Artifact: `artifacts/X_items.npz` (L2-normalized item vectors)
	- Used by: `app/streamlit_app.py` (recommend_content, hybrid cold), `ml/recommender.py`

- Collaborative Filtering / Neighborhood models (ItemKNN), NMF
	- Trained artifacts: `artifacts/cf_knn.pkl`, `artifacts/cf_nmf.pkl`, `artifacts/kmeans.pkl` (if present)
	- Used by: Hybrid ensemble during warm-user recommendations (the hybrid config is saved in `artifacts/hybrid.pkl`)

- Popularity model
	- Artifact: `artifacts/popular.pkl` (stores item popularity scores)
	- Used by: cold-start blending in `recommend_hybrid_cold()` in `app/streamlit_app.py`

- Hybrid ensemble
	- Artifact: `artifacts/hybrid.pkl` (contains ensemble weights, policy, and metrics)
	- Used by: `app/streamlit_app.py` for warm-user / hybrid recommendation flows

- Supervised reranker (LightGBM / sklearn model)
	- Artifact: `artifacts/supervised_ranker.pkl` (if present)
	- Purpose: rerank candidate lists using structured features (similarity, popularity, price, subject match)
	- Used by: ranking pipeline or custom re-ranking step (not always active in the simplified UI)

Notes:
- Artifacts live in the `artifacts/` directory. The Streamlit app loads these artifacts at startup (`app/streamlit_app.py` -> `load_artifacts()`) and falls back to the recommender index in `ml/outputs` if necessary.
- If you retrain models, re-generate or update the corresponding artifacts and restart the Streamlit app (or re-run `ml/make_streamlit_artifacts.py` for TF-IDF/X_items).


## 📚 Documentation

- `README-ML.md` — Full ML documentation, model architecture, and training instructions.
- `Data/README.md` — Data preprocessing notes (if present).

## 👥 Contributors

- **Prerana V** - Project Lead
- **Vikram Krishna** - ML Pipeline Developer
- **Prakriti** - Web Scraping and EDA
- **Miloni Halkati** - Deployment of App
- **Anil D** - Guide

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

We welcome contributions — please open issues or pull requests. See `CONTRIBUTING.md` if present.

## 📬 Contact

- Project Lead: Prerana V
- Repository: https://github.com/prerana-v27/Personalized-Learning-Pathways

## 🙏 Acknowledgments

- Online learning platforms for providing course data
- Open source ML community for tools and libraries

