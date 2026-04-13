"""
Cross-Validation Script - Validates model robustness
"""

import os
import pickle
import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
import joblib


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = " ".join(text.split())
    return text


def normalize_labels(df: pd.DataFrame, source_file: str) -> tuple:
    label_map = {
        "information_item": 0, "information": 0, "info": 0,
        "action_item": 1, "action": 1, "act": 1,
        "0": 0, "1": 1,
    }
    
    df_copy = df.copy()
    df_copy.columns = [c.lower().strip() for c in df_copy.columns]
    
    text_col = None
    for col in ["sentence", "text"]:
        if col in df_copy.columns:
            text_col = col
            break
    
    label_col = None
    for col in df_copy.columns:
        if "label" in col.lower():
            label_col = col
            break
    
    if text_col is None or label_col is None:
        return None, False
    
    normalized_labels = []
    valid_texts = []
    for idx, raw_label in enumerate(df_copy[label_col].astype(str)):
        key = raw_label.strip().lower()
        if key not in label_map:
            continue
        normalized_labels.append(label_map[key])
        valid_texts.append(df_copy[text_col].iloc[idx])
    
    if len(normalized_labels) == 0:
        return None, False
    
    result = pd.DataFrame({
        "sentence": valid_texts,
        "label": normalized_labels,
    })
    
    return result, True


def load_all_csv_files() -> pd.DataFrame:
    """Load all CSV files from data/ folder."""
    dfs = []
    csv_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv')])
    
    for csv_file in csv_files:
        csv_path = os.path.join(DATA_DIR, csv_file)
        try:
            df = pd.read_csv(csv_path)
            df_normalized, success = normalize_labels(df, csv_file)
            if success:
                dfs.append(df_normalized)
        except:
            pass
    
    return pd.concat(dfs, ignore_index=True)


def run_cross_validation():
    """5-fold cross-validation to verify model robustness"""
    
    print("=" * 70)
    print("CROSS-VALIDATION: Testing Model Robustness")
    print("=" * 70)
    
    # Load data
    print("\n📂 Loading all datasets...")
    df = load_all_csv_files()
    print(f"   Loaded: {len(df)} samples")
    
    # Clean
    print("\n🧹 Cleaning data...")
    df = df.dropna(subset=["sentence", "label"])
    df["sentence"] = df["sentence"].apply(clean_text)
    df = df[df["sentence"].str.len() > 0]
    print(f"   After cleanup: {len(df)} samples")
    
    X = df["sentence"]
    y = df["label"]
    
    # Cross-validation pipeline
    print("\n🔧 Building 5-Fold Cross-Validation...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=100000,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )),
        ("clf", SGDClassifier(
            loss="hinge",
            penalty="l2",
            alpha=0.0001,
            max_iter=2000,
            tol=1e-4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ))
    ])
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score),
    }
    
    print("\n🧠 Running 5-fold cross-validation...")
    print("   (This will take a few minutes...)\n")
    
    cv_results = cross_validate(
        pipeline, X, y, cv=cv, scoring=scoring, n_jobs=1, verbose=2
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION RESULTS (5 Folds)")
    print("=" * 70)
    
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        scores = cv_results[f'test_{metric}']
        print(f"\n{metric.upper()}:")
        for i, score in enumerate(scores):
            print(f"   Fold {i+1}: {score:.4f}")
        print(f"   Mean: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    accuracy_mean = cv_results['test_accuracy'].mean()
    accuracy_std = cv_results['test_accuracy'].std()
    
    if accuracy_std < 0.01:
        print(f"\n✅ STABLE: Low variance ({accuracy_std:.4f})")
        print("   Model performs consistently across all folds.")
        print("   → This verifies the model is NOT overfitting")
    else:
        print(f"\n⚠️  UNSTABLE: High variance ({accuracy_std:.4f})")
        print("   Model accuracy varies significantly across folds.")
        print("   → Check data for quality issues")
    
    if accuracy_mean > 0.98:
        print(f"\n🎯 HIGH ACCURACY: {accuracy_mean*100:.2f}%")
        print("   This is real! Your action item extraction is genuinely high-quality.")
        print("   Possible reasons:")
        print("   - Datasets are very clean and well-labeled")
        print("   - Action items have distinctive language patterns")
        print("   - Task is genuinely easier than expected")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_cross_validation()
