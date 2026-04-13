"""
Bulk Training Script - Uses ALL Data
Reads every CSV in data/ folder and trains on combined 80-20 split.
"""

import argparse
import os
import pickle
import time
from typing import Tuple, List

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import joblib


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def clean_text(text: str) -> str:
    """
    Clean text while preserving Action Item markers.
    
    Keeps: Stopwords, pronouns (critical for action items like "I will...")
    Removes: Excessive whitespace, leading/trailing spaces
    """
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    # Normalize multiple spaces
    text = " ".join(text.split())
    return text


def normalize_labels(df: pd.DataFrame, source_file: str) -> Tuple[pd.DataFrame, bool]:
    """
    Normalize label values across different datasets.
    Returns (normalized_df, success)
    """
    label_map = {
        "information_item": 0,
        "information": 0,
        "info": 0,
        "action_item": 1,
        "action": 1,
        "act": 1,
        "0": 0,
        "1": 1,
    }

    df_copy = df.copy()
    
    # Standardize column names
    df_copy.columns = [c.lower().strip() for c in df_copy.columns]
    
    # Try to find text column
    text_col = None
    if "sentence" in df_copy.columns:
        text_col = "sentence"
    elif "text" in df_copy.columns:
        text_col = "text"
    else:
        # Try to find any column that might contain sentences
        for col in df_copy.columns:
            if "text" in col.lower() or "sentence" in col.lower() or "content" in col.lower():
                text_col = col
                break
    
    # Try to find label column
    label_col = None
    if "label" in df_copy.columns:
        label_col = "label"
    else:
        for col in df_copy.columns:
            if "label" in col.lower() or "type" in col.lower() or "classification" in col.lower():
                label_col = col
                break
    
    if text_col is None or label_col is None:
        print(f"   ⚠️  Skipped: Missing text or label column. Columns: {list(df_copy.columns)}")
        return None, False
    
    # Clean labels
    normalized_labels = []
    valid_texts = []
    for idx, raw_label in enumerate(df_copy[label_col].astype(str)):
        key = raw_label.strip().lower()
        if key not in label_map:
            print(f"   ⚠️  Warning: Unknown label '{raw_label}' in {source_file}, skipping row")
            continue
        normalized_labels.append(label_map[key])
        valid_texts.append(df_copy[text_col].iloc[idx])
    
    if len(normalized_labels) == 0:
        print(f"   ⚠️  Skipped: No valid labels after normalization")
        return None, False
    
    result = pd.DataFrame({
        "sentence": valid_texts,
        "label": normalized_labels,
    })
    
    return result, True


def load_all_csv_files() -> pd.DataFrame:
    """Load and combine all CSV files from data/ folder."""
    dfs = []
    total_loaded = 0
    total_skipped = 0
    
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")
    
    csv_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv')])
    
    print(f"Found {len(csv_files)} CSV files in {DATA_DIR}\n")
    
    for csv_file in csv_files:
        csv_path = os.path.join(DATA_DIR, csv_file)
        
        try:
            print(f"📂 Loading {csv_file}...")
            df = pd.read_csv(csv_path)
            print(f"   Raw rows: {len(df)}")
            
            df_normalized, success = normalize_labels(df, csv_file)
            
            if success:
                dfs.append(df_normalized)
                total_loaded += len(df_normalized)
                print(f"   ✅ Loaded: {len(df_normalized)} rows")
            else:
                total_skipped += len(df)
                
        except Exception as e:
            print(f"   ❌ Error loading {csv_file}: {e}")
            total_skipped += 1
    
    if not dfs:
        raise ValueError("No valid datasets loaded!")
    
    combined = pd.concat(dfs, ignore_index=True)
    
    print(f"\n{'='*70}")
    print(f"✅ Successfully loaded: {len(combined)} rows from {len(dfs)} files")
    print(f"   Class distribution: {combined['label'].value_counts().to_dict()}")
    print(f"{'='*70}\n")
    
    return combined


def train_bulk_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    max_features: int = 100000,
    max_iter: int = 2000,
    output_path: str = "svm_bulk_model.pkl",
    metrics_path: str = "bulk_training_metrics.pkl",
) -> Tuple[Pipeline, dict]:
    """
    Train bulk SVM model with memory-efficient settings.
    """
    
    print("=" * 70)
    print("BULK TRAINING PHASE")
    print("=" * 70)
    
    # Data cleanup
    print("\n🧹 Cleaning data...")
    df = df.dropna(subset=["sentence", "label"])
    df["sentence"] = df["sentence"].apply(clean_text)
    # Remove empty strings after cleaning
    df = df[df["sentence"].str.len() > 0]
    print(f"   Rows after cleanup: {len(df)}")
    
    X = df["sentence"]
    y = df["label"]
    
    # Train-test split
    print(f"\n📊 Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"   Train samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Build pipeline
    print(f"\n🔧 Building pipeline...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=max_features,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )),
        ("clf", SGDClassifier(
            loss="hinge",
            penalty="l2",
            alpha=0.0001,
            max_iter=max_iter,
            tol=1e-4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=1,
        ))
    ])
    
    # Training phase
    print(f"\n🧠 Starting training on {len(X_train)} samples...")
    print(f"   (This may take several minutes...)")
    start_time = time.time()
    
    pipeline.fit(X_train, y_train)
    
    elapsed_time = time.time() - start_time
    print(f"\n✅ Training complete in {elapsed_time/60:.2f} minutes ({elapsed_time:.0f}s)")
    
    # Evaluation phase
    print("\n" + "=" * 70)
    print("EVALUATION PHASE")
    print("=" * 70)
    
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"\n📈 Accuracy Scores:")
    print(f"   Train Accuracy: {train_acc * 100:.2f}%")
    print(f"   Test Accuracy:  {test_acc * 100:.2f}%")
    
    # Detailed metrics
    print(f"\n📊 Test Set Classification Report:")
    print(classification_report(y_test, y_test_pred, 
                                target_names=["Information Item", "Action Item"]))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\n🎯 Confusion Matrix (Test Set):")
    print(f"   True Negatives:  {cm[0, 0]}")
    print(f"   False Positives: {cm[0, 1]}")
    print(f"   False Negatives: {cm[1, 0]}")
    print(f"   True Positives:  {cm[1, 1]}")
    
    
    # Metrics dictionary
    metrics = {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "precision": precision_score(y_test, y_test_pred),
        "recall": recall_score(y_test, y_test_pred),
        "f1": f1_score(y_test, y_test_pred),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "training_time_seconds": elapsed_time,
        "confusion_matrix": cm.tolist(),
        "class_distribution_train": y_train.value_counts().to_dict(),
        "class_distribution_test": y_test.value_counts().to_dict(),
    }
    
    # Save model
    print(f"\n💾 Saving model to {output_path}...")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    joblib.dump(pipeline, output_path)
    print(f"   ✅ Model saved")
    
    # Save metrics
    print(f"💾 Saving metrics to {metrics_path}...")
    os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)
    with open(metrics_path, "wb") as f:
        pickle.dump(metrics, f)
    print(f"   ✅ Metrics saved")
    
    return pipeline, metrics


def print_summary(metrics: dict):
    """Print final summary."""
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Total samples used: {metrics['train_samples'] + metrics['test_samples']}")
    print(f"Final Test Accuracy: {metrics['test_accuracy'] * 100:.2f}%")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print(f"Training time: {metrics['training_time_seconds']/60:.2f} minutes")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Bulk train using ALL CSV files in data/ folder."
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio (default: 0.2 = 80-20 split)",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=100000,
        help="Max vocabulary size (default: 100000)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=2000,
        help="Training epochs (default: 2000)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(SCRIPT_DIR, "svm_bulk_model.pkl"),
        help="Output model path",
    )
    parser.add_argument(
        "--metrics",
        default=os.path.join(SCRIPT_DIR, "bulk_training_metrics.pkl"),
        help="Output metrics path",
    )
    
    args = parser.parse_args()
    
    try:
        # Load all data
        df = load_all_csv_files()
        
        # Train
        pipeline, metrics = train_bulk_model(
            df,
            test_size=args.test_size,
            max_features=args.max_features,
            max_iter=args.max_iter,
            output_path=args.output,
            metrics_path=args.metrics,
        )
        
        # Summary
        print_summary(metrics)
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
