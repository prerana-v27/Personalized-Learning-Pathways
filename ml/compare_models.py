"""Compare two saved pipelines (baseline and enhanced) on a single stratified test split.

Writes a JSON report to ml/outputs/compare_report.json and prints summary.

Usage:
  python ml/compare_models.py --csv Data/processed/all_courses_merged.csv --baseline ml/outputs/model_classif.joblib --enhanced ml/outputs/model_enhanced.joblib --out ml/outputs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def extract_text_column(df: pd.DataFrame, column: str = 'title') -> pd.Series:
    """Extract text column with fallback to empty string"""
    return df.get(column, pd.Series(['']*len(df))).fillna('').astype(str)

def extract_title_column(df: pd.DataFrame) -> pd.Series:
    """Wrapper for older models that expect this function"""
    return extract_text_column(df, 'title')

def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    
    # Register any custom functions needed by the pickled models
    import sys
    setattr(sys.modules[__name__], 'extract_text_column', extract_text_column)
    setattr(sys.modules[__name__], 'extract_title_column', extract_title_column)
    
    return joblib.load(path)


def prepare_data(csv_path: Path):
    print(f'Reading data from {csv_path}...')
    df = pd.read_csv(csv_path)
    
    # Clean and normalize target first
    print('Cleaning and normalizing difficulty levels...')
    if 'difficulty' not in df.columns:
        raise RuntimeError('difficulty column missing')
    df['difficulty'] = df['difficulty'].astype(str).str.strip()
    df.loc[df['difficulty'].str.lower() == 'unknown', 'difficulty'] = ''
    df['difficulty'] = df['difficulty'].replace({
        'beginner': 'Beginner', 'introductory': 'Beginner',
        'intermediate': 'Intermediate',
        'advanced': 'Advanced', 'expert': 'Advanced'
    })
    
    # Filter to valid difficulties first and reset index
    valid_mask = df['difficulty'].notna() & (df['difficulty'] != '')
    df = df[valid_mask].copy()  # Create copy to avoid SettingWithCopyWarning
    df.reset_index(drop=True, inplace=True)  # Reset index to avoid issues with filtering
    print(f'Found {len(df)} courses with valid difficulty levels')

    X_text = df['title'].fillna('') + ' ' + df.get('skills', '').fillna('')
    # Minimal DataFrame used by pipelines in models
    X = pd.DataFrame({
        'price': pd.to_numeric(df.get('price', pd.Series([None]*len(df))), errors='coerce'),
        'rating': pd.to_numeric(df.get('rating', pd.Series([None]*len(df))), errors='coerce'),
        'university': df.get('university', pd.Series(['']*len(df))).astype(str),
        'source': df.get('source', pd.Series(['']*len(df))).astype(str),
        'title': X_text,
        'skills': df.get('skills', pd.Series(['']*len(df))).fillna('').astype(str),
        'subject': df.get('subject', pd.Series(['']*len(df))).fillna('').astype(str),
        'is_paid': df.get('is_paid', pd.Series([False]*len(df))).fillna(False).astype(bool),
        'level': df.get('level', pd.Series(['']*len(df))).fillna('').astype(str)
    })
    y = df['difficulty'].astype(str)
    return X, y


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1_macro = float(f1_score(y_test, y_pred, average='macro'))
    crep = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()
    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'classification_report': crep,
        'confusion_matrix': cm
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='Data/processed/all_courses_merged.csv')
    parser.add_argument('--baseline', default='ml/outputs/model_classif.joblib')
    parser.add_argument('--enhanced', default='ml/outputs/model_enhanced.joblib')
    parser.add_argument('--out', default='ml/outputs')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--random-state', type=int, default=42)
    args = parser.parse_args()

    X, y = prepare_data(Path(args.csv))
    # stratified split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=args.random_state)

    baseline_model = load_model(Path(args.baseline))
    enhanced_model = load_model(Path(args.enhanced))

    # Evaluate
    print('Evaluating baseline model...')
    baseline_res = evaluate_model(baseline_model, X_test, y_test)
    print('Evaluating enhanced model...')
    enhanced_res = evaluate_model(enhanced_model, X_test, y_test)

    out = {
        'baseline_path': str(args.baseline),
        'enhanced_path': str(args.enhanced),
        'n_test': int(len(y_test)),
        'baseline': baseline_res,
        'enhanced': enhanced_res
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'compare_report.json').write_text(json.dumps(out, indent=2))
    print('Wrote report to', out_dir / 'compare_report.json')
    print('Baseline f1_macro =', baseline_res['f1_macro'])
    print('Enhanced f1_macro =', enhanced_res['f1_macro'])


if __name__ == '__main__':
    main()
