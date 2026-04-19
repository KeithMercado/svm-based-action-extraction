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
    precision_recall_curve,
)
import joblib
import matplotlib.pyplot as plt

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def clean_text(text: str, lowercase: bool = True) -> str:
    """
    Clean text while preserving Action Item markers.
    
    Keeps: Stopwords, pronouns (critical for action items like "I will...")
    Removes: Excessive whitespace, leading/trailing spaces
    """
    if not isinstance(text, str):
        return ""
    if lowercase:
        text = text.lower()
    text = text.strip()
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


def load_ami_hf_dataset(dataset_id: str) -> pd.DataFrame:
    """Load AMI-style dataset from Hugging Face and normalize to sentence/label."""
    if load_dataset is None:
        raise ImportError(
            "Hugging Face datasets package not installed. Run: pip install datasets"
        )

    print(f"\n🤗 Loading Hugging Face dataset: {dataset_id} ...")
    ds = load_dataset(dataset_id)

    parts = []
    for split_name, split_ds in ds.items():
        split_df = split_ds.to_pandas()
        normalized, ok = normalize_labels(split_df, f"{dataset_id}:{split_name}")
        if ok:
            parts.append(normalized)
            print(f"   ✅ {split_name}: {len(normalized)} rows")
        else:
            print(f"   ⚠️  {split_name}: skipped due to missing compatible columns")

    if not parts:
        raise ValueError(
            f"No compatible text/label columns found in Hugging Face dataset: {dataset_id}"
        )

    combined = pd.concat(parts, ignore_index=True)
    print(f"   ✅ Total loaded from HF: {len(combined)}")
    return combined


def maybe_enable_balanced_weight(y: pd.Series, ratio_threshold: float = 1.5):
    """Enable balanced class weights when class imbalance is detected."""
    counts = y.value_counts().to_dict()
    if len(counts) < 2:
        return "balanced"
    majority = max(counts.values())
    minority = min(counts.values())
    imbalance_ratio = (majority / minority) if minority > 0 else float("inf")
    if imbalance_ratio >= ratio_threshold:
        print(f"   ⚖️  Imbalance detected ({imbalance_ratio:.2f}:1). Using class_weight='balanced'")
        return "balanced"
    print(f"   ℹ️  Near-balanced data ({imbalance_ratio:.2f}:1). Using class_weight=None")
    return None


def tune_recall_threshold(y_true: pd.Series, decision_scores: np.ndarray, target_recall: float = 0.95):
    """
    Find threshold that prioritizes higher recall and reduces false negatives.
    Returns (threshold, tuned_pred, tuned_recall, tuned_precision, tuned_f1_macro).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, decision_scores)

    if thresholds.size == 0:
        default_pred = (decision_scores >= 0.0).astype(int)
        return (
            0.0,
            default_pred,
            recall_score(y_true, default_pred, zero_division=0),
            precision_score(y_true, default_pred, zero_division=0),
            f1_score(y_true, default_pred, average="macro", zero_division=0),
        )

    candidate_idxs = np.where(recalls[:-1] >= target_recall)[0]
    if candidate_idxs.size > 0:
        best_idx = candidate_idxs[np.argmax(precisions[:-1][candidate_idxs])]
    else:
        best_idx = int(np.argmax(recalls[:-1]))

    tuned_threshold = float(thresholds[best_idx])
    tuned_pred = (decision_scores >= tuned_threshold).astype(int)

    tuned_recall = recall_score(y_true, tuned_pred, zero_division=0)
    tuned_precision = precision_score(y_true, tuned_pred, zero_division=0)
    tuned_f1_macro = f1_score(y_true, tuned_pred, average="macro", zero_division=0)

    return tuned_threshold, tuned_pred, tuned_recall, tuned_precision, tuned_f1_macro


def save_confusion_matrix_plot(cm: np.ndarray, output_path: str):
    """Save confusion matrix heatmap image for thesis slide deck."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=["Pred Info", "Pred Action"],
        yticklabels=["True Info", "True Action"],
        title="Confusion Matrix (Test Set)",
        ylabel="True label",
        xlabel="Predicted label",
    )

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_f1_chart(info_f1: float, action_f1: float, macro_f1: float, output_path: str):
    """Save per-class and macro F1 bar chart for thesis slide deck."""
    labels = ["Info F1", "Action F1", "Macro F1"]
    values = [info_f1, action_f1, macro_f1]
    colors = ["#4E79A7", "#E15759", "#59A14F"]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1)
    ax.set_title("F1 Scores")
    ax.set_ylabel("Score")

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def train_bulk_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    max_features: int = 100000,
    max_iter: int = 2000,
    ngram_min: int = 1,
    ngram_max: int = 2,
    lowercase_text: bool = True,
    target_recall: float = 0.95,
    force_balanced: bool = False,
    disable_shuffle: bool = False,
    output_path: str = "svm_bulk_model.pkl",
    metrics_path: str = "bulk_training_metrics.pkl",
    cm_plot_path: str = "bulk_confusion_matrix.png",
    f1_plot_path: str = "bulk_f1_scores.png",
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
    df["sentence"] = df["sentence"].apply(lambda t: clean_text(t, lowercase=lowercase_text))
    # Remove empty strings after cleaning
    df = df[df["sentence"].str.len() > 0]
    print(f"   Rows after cleanup: {len(df)}")
    
    X = df["sentence"]
    y = df["label"]
    
    # Train-test split
    print(f"\n📊 Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
        shuffle=not disable_shuffle,
    )
    print(f"   Train samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")

    class_weight = "balanced" if force_balanced else maybe_enable_balanced_weight(y_train)
    
    # Build pipeline
    print(f"\n🔧 Building pipeline...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(ngram_min, ngram_max),
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
            class_weight=class_weight,
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
    y_test_pred_default = pipeline.predict(X_test)

    # Recall-priority threshold tuning for false-negative reduction.
    decision_scores = pipeline.decision_function(X_test)
    tuned_threshold, y_test_pred, tuned_recall, tuned_precision, tuned_f1_macro = tune_recall_threshold(
        y_test, decision_scores, target_recall=target_recall
    )
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_acc_default = accuracy_score(y_test, y_test_pred_default)
    
    print(f"\n📈 Accuracy Scores:")
    print(f"   Train Accuracy: {train_acc * 100:.2f}%")
    print(f"   Test Accuracy (default threshold): {test_acc_default * 100:.2f}%")
    print(f"   Test Accuracy (recall-tuned):      {test_acc * 100:.2f}%")
    print(f"\n🎯 Recall threshold tuning:")
    print(f"   Selected threshold: {tuned_threshold:.5f}")
    print(f"   Tuned Recall:       {tuned_recall:.4f}")
    print(f"   Tuned Precision:    {tuned_precision:.4f}")
    print(f"   Tuned Macro-F1:     {tuned_f1_macro:.4f}")
    
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
    
    
    info_f1 = f1_score(y_test, y_test_pred, pos_label=0, zero_division=0)
    action_f1 = f1_score(y_test, y_test_pred, pos_label=1, zero_division=0)
    macro_f1 = f1_score(y_test, y_test_pred, average="macro", zero_division=0)

    save_confusion_matrix_plot(cm, cm_plot_path)
    save_f1_chart(info_f1, action_f1, macro_f1, f1_plot_path)
    print(f"\n🖼️  Saved confusion matrix plot: {cm_plot_path}")
    print(f"🖼️  Saved F1 chart: {f1_plot_path}")

    # Metrics dictionary
    metrics = {
        "train_accuracy": train_acc,
        "test_accuracy_default": test_acc_default,
        "test_accuracy": test_acc,
        "precision": precision_score(y_test, y_test_pred, zero_division=0),
        "recall": recall_score(y_test, y_test_pred, zero_division=0),
        "f1": f1_score(y_test, y_test_pred, zero_division=0),
        "f1_macro": macro_f1,
        "f1_info": info_f1,
        "f1_action": action_f1,
        "tuned_threshold": tuned_threshold,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "training_time_seconds": elapsed_time,
        "confusion_matrix": cm.tolist(),
        "class_distribution_train": y_train.value_counts().to_dict(),
        "class_distribution_test": y_test.value_counts().to_dict(),
        "class_weight": class_weight,
        "ngram_range": [ngram_min, ngram_max],
        "lowercase_text": lowercase_text,
        "target_recall": target_recall,
        "cm_plot_path": cm_plot_path,
        "f1_plot_path": f1_plot_path,
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
    print(f"Macro F1-Score: {metrics['f1_macro']:.4f}")
    print(f"Threshold used: {metrics['tuned_threshold']:.5f}")
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
        "--ngram-min",
        type=int,
        default=1,
        help="Minimum n-gram value (default: 1)",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=2,
        help="Maximum n-gram value (default: 2)",
    )
    parser.add_argument(
        "--preserve-case",
        action="store_true",
        help="Keep original casing instead of lowercasing text",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.95,
        help="Target recall for threshold tuning (default: 0.95)",
    )
    parser.add_argument(
        "--force-balanced",
        action="store_true",
        help="Always use class_weight='balanced' even if classes look balanced",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling before train-test split",
    )
    parser.add_argument(
        "--hf-dataset-id",
        default="",
        help="Optional Hugging Face dataset id to include (example: edinburghcstr/ami)",
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
    parser.add_argument(
        "--cm-plot",
        default=os.path.join(SCRIPT_DIR, "bulk_confusion_matrix.png"),
        help="Output confusion matrix PNG path",
    )
    parser.add_argument(
        "--f1-plot",
        default=os.path.join(SCRIPT_DIR, "bulk_f1_scores.png"),
        help="Output F1 chart PNG path",
    )
    
    args = parser.parse_args()
    
    try:
        # Load all data
        df = load_all_csv_files()

        if args.hf_dataset_id:
            hf_df = load_ami_hf_dataset(args.hf_dataset_id)
            df = pd.concat([df, hf_df], ignore_index=True)
            print(f"\n✅ Combined local + HF samples: {len(df)}")
        
        # Train
        pipeline, metrics = train_bulk_model(
            df,
            test_size=args.test_size,
            max_features=args.max_features,
            max_iter=args.max_iter,
            ngram_min=args.ngram_min,
            ngram_max=args.ngram_max,
            lowercase_text=not args.preserve_case,
            target_recall=args.target_recall,
            force_balanced=args.force_balanced,
            disable_shuffle=args.no_shuffle,
            output_path=args.output,
            metrics_path=args.metrics,
            cm_plot_path=args.cm_plot,
            f1_plot_path=args.f1_plot,
        )
        
        # Summary
        print_summary(metrics)
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
