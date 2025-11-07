"""Enhanced training script to improve classifier performance.

Main features:
- Load merged CSV (Data/processed/all_courses_merged.csv)
- Clean numeric anomalies (cap ratings to [0,5], handle price/rating outliers)
- Normalize/clean target labels (keep Beginner/Intermediate/Advanced)
- Build ColumnTransformer: numeric (imputer+scaler), categorical (imputer+onehot), text (TfidfVectorizer)
- Train multiple classifiers (LinearSVC with hashing/tfidf text, LogisticRegression (saga), RandomForest)
- Stratified K-fold CV and report accuracy, macro-F1
- Save best model and JSON report to ml/outputs/

Usage:
  python ml\\enhanced_train.py --csv Data\\processed\\all_courses_merged.csv --out ml\\outputs
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Lowercase column names
    df = df.rename(columns=lambda s: s.strip() if isinstance(s, str) else s)

    # Target cleaning: keep only three classes
    if 'difficulty' in df.columns:
        df['difficulty'] = df['difficulty'].astype(str).str.strip()
        df.loc[df['difficulty'].str.lower() == 'unknown', 'difficulty'] = ''
        df.loc[df['difficulty'].isin(['', 'nan', 'None', 'NoneType']), 'difficulty'] = ''
        # Map common variants
        df['difficulty'] = df['difficulty'].replace({
            'beginner': 'Beginner', 'introductory': 'Beginner',
            'intermediate': 'Intermediate',
            'advanced': 'Advanced', 'expert': 'Advanced'
        })

    # Ratings: cap to [0,5]
    if 'rating' in df.columns:
        try:
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
            df.loc[df['rating'] > 5, 'rating'] = 5.0
            df.loc[df['rating'] < 0, 'rating'] = 0.0
        except Exception:
            df['rating'] = df['rating']

    # Price: coerce numeric
    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce')

    # Ensure text cols exist
    for c in ['title', 'skills', 'description', 'subject']:
        if c not in df.columns:
            df[c] = ''

    return df


def build_pipeline() -> Pipeline:
    # Numeric features
    numeric_features = ['price', 'rating']
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical features
    categorical_features = ['university', 'source']
    # OneHotEncoder API differs across sklearn versions: try sparse=False, else sparse_output=False
    try:
        _ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    except TypeError:
        _ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='')),
        ('onehot', _ohe)
    ])

    # ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('text', TfidfVectorizer(max_features=8192, ngram_range=(1,2)), 'title')
    ], remainder='drop')

    return preprocessor


def prepare_X_y(df: pd.DataFrame):
    # Keep only rows with valid target
    if 'difficulty' not in df.columns:
        raise RuntimeError('difficulty column missing')

    df2 = df.copy()
    df2.loc[df2['difficulty'] == '', 'difficulty'] = np.nan
    df2 = df2.dropna(subset=['difficulty'])

    y = df2['difficulty'].astype(str)

    # Text column for vectorizer
    X_text = df2['title'].fillna('') + ' ' + df2['skills'].fillna('')

    # Build feature matrix DataFrame for ColumnTransformer
    X_df = pd.DataFrame({
        'price': df2.get('price', pd.Series([np.nan]*len(df2))).values,
        'rating': df2.get('rating', pd.Series([np.nan]*len(df2))).values,
        'university': df2.get('university', pd.Series(['']*len(df2))).values,
        'source': df2.get('source', pd.Series(['']*len(df2))).values,
        'title': X_text.values
    })

    return X_df, y, df2.index


def evaluate_and_train(X, y, out_dir: Path) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Models to try
    models = {
        'LinearSVC': LinearSVC(class_weight='balanced', max_iter=8000, random_state=42),
        'Logistic': LogisticRegression(max_iter=5000, class_weight='balanced', solver='saga', multi_class='multinomial'),
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    }

    preprocessor = build_pipeline()

    best_score = -1.0
    best_model = None
    best_name = None

    for name, clf in models.items():
        print(f"Training & CV for {name}...")
        pipe = Pipeline([
            ('pre', preprocessor),
            ('clf', clf)
        ])

        cv_res = cross_validate(pipe, X, y, cv=skf, scoring=('accuracy', 'f1_macro'), n_jobs=1, return_train_score=False)
        acc = float(np.mean(cv_res['test_accuracy']))
        f1 = float(np.mean(cv_res['test_f1_macro']))
        print(f"{name}: acc={acc:.4f}, f1_macro={f1:.4f}")

        results[name] = {'accuracy': acc, 'f1_macro': f1}

        if f1 > best_score:
            best_score = f1
            best_model = pipe
            best_name = name

    # Fit best model on full data
    print(f"Fitting best model: {best_name}")
    best_model.fit(X, y)

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / 'model_enhanced.joblib'
    joblib.dump(best_model, model_path)

    # Evaluate on training data (report)
    y_pred = best_model.predict(X)
    acc_full = accuracy_score(y, y_pred)
    f1_full = f1_score(y, y_pred, average='macro')
    creport = classification_report(y, y_pred, output_dict=True)

    report = {
        'best_model': best_name,
        'cv_results': results,
        'train_accuracy': acc_full,
        'train_f1_macro': f1_full,
        'classification_report': creport
    }

    with open(out_dir / 'enhanced_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='Data/processed/all_courses_merged.csv')
    parser.add_argument('--out', default='ml/outputs')
    args = parser.parse_args(argv)

    df = pd.read_csv(args.csv)
    df = clean_dataframe(df)

    X, y, idx = prepare_X_y(df)

    report = evaluate_and_train(X, y, Path(args.out))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
