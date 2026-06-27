"""
Comprehensive comparison: SVM vs Naive Bayes for Action Item Detection
Evaluates both models across all datasets and generates comparison metrics.
"""

import argparse
import os
import pickle
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "svm_vs_nb_comparison")

DATASETS: Dict[str, str] = {
    "Formal English (10k)": os.path.join(DATA_FOLDER, "action_items_dataset_10k.csv"),
    "Mixed Taglish (20k)": os.path.join(DATA_FOLDER, "action_items_dataset_20k_taglish.csv"),
    "Tagalog Set": os.path.join(DATA_FOLDER, "tagalog_action_items.csv"),
    "Taglish Set": os.path.join(DATA_FOLDER, "taglish_action_items.csv"),
    "Expanded English": os.path.join(DATA_FOLDER, "english_expanded_dataset.csv"),
    "Comprehensive (12k)": os.path.join(DATA_FOLDER, "comprehensive_thesis_dataset_12k.csv"),
    "Meeting Specific (15k)": os.path.join(DATA_FOLDER, "meeting_specific_dataset_15k.csv"),
    "Expanded Contexts (20k)": os.path.join(DATA_FOLDER, "expanded_meeting_contexts_20k.csv"),
    "Massive Diverse (50k)": os.path.join(DATA_FOLDER, "massive_diverse_dataset_50000.csv"),
    "Ultimate Diversity (50k)": os.path.join(DATA_FOLDER, "ultimate_diversity_dataset_50k.csv"),
    "AMI Corpus Dataset": os.path.join(DATA_FOLDER, "ami_dialogueActs_dataset.csv"),
    "Multilingual Resistant (Refined)": os.path.join(DATA_FOLDER, "ami_multilingual_balanced.csv"),
}

CLASS_NAMES = ["Non-Action", "Action Item"]


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[OK] Output directory: {OUTPUT_DIR}")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    return " ".join(text.split())


def load_and_normalize(file_path: str, source_name: str) -> pd.DataFrame:
    """Load CSV and normalize labels to 0/1."""
    try:
        df = pd.read_csv(file_path)
        df.columns = [col.lower().strip() for col in df.columns]

        if "text" in df.columns and "sentence" not in df.columns:
            df = df.rename(columns={"text": "sentence"})

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

        if "sentence" not in df.columns or "label" not in df.columns:
            return None

        rows = df.dropna(subset=["sentence", "label"]).copy()
        rows["sentence"] = rows["sentence"].astype(str).map(clean_text)
        rows = rows[rows["sentence"].str.len() > 0]

        labels: List[int] = []
        valid_sentences: List[str] = []
        for idx, raw_label in enumerate(rows["label"].astype(str)):
            key = raw_label.strip().lower()
            if key not in label_map:
                continue
            labels.append(label_map[key])
            valid_sentences.append(rows["sentence"].iloc[idx])

        if not labels:
            return None

        return pd.DataFrame(
            {
                "sentence": valid_sentences,
                "label": labels,
                "source_file": source_name,
            }
        )
    except Exception as e:
        print(f"   [!] Error loading {source_name}: {e}")
        return None


def load_all_datasets() -> pd.DataFrame:
    """Load and combine all available datasets."""
    parts = []
    for dataset_name, file_path in DATASETS.items():
        if not os.path.exists(file_path):
            continue
        df = load_and_normalize(file_path, dataset_name)
        if df is not None:
            parts.append(df)
            print(f"   [OK] {dataset_name}: {len(df)} rows")
        else:
            print(f"   [!] {dataset_name}: skipped")

    if not parts:
        raise ValueError("No datasets could be loaded.")

    combined = pd.concat(parts, ignore_index=True)
    combined = combined.dropna(subset=["sentence", "label"])
    combined["label"] = combined["label"].astype(int)
    
    print(f"\n[STATS] Combined Dataset Stats:")
    print(f"   Total samples: {len(combined)}")
    print(f"   Class distribution: {combined['label'].value_counts().to_dict()}")
    print(f"   Action items: {(combined['label'] == 1).sum()} ({(combined['label'] == 1).sum() / len(combined) * 100:.1f}%)")
    
    return combined


def create_svm_model():
    """Create SVM (SGD) pipeline with TF-IDF vectorization."""
    return Pipeline([
        (
            "tfidf",
            FeatureUnion([
                (
                    "word_tfidf",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        max_features=100000,
                        min_df=2,
                        max_df=0.95,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "char_tfidf",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        max_features=60000,
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
            ])
        ),
        (
            "clf",
            SGDClassifier(
                loss="hinge",
                penalty="l2",
                alpha=0.0001,
                max_iter=2000,
                tol=1e-4,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])


def create_naive_bayes_model():
    """Create Naive Bayes pipeline with TF-IDF vectorization."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
        )),
        ("clf", MultinomialNB(alpha=0.1)),
    ])


def evaluate_model(model, X_test, y_test, model_name: str) -> Dict:
    """Evaluate model and return comprehensive metrics."""
    y_pred = model.predict(X_test)
    y_test_arr = np.asarray(y_test, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    
    # Get decision scores for PR/ROC curves
    try:
        # Try decision_function first (for SVM)
        X_test_transformed = model.named_steps["tfidf"].transform(X_test)
        if hasattr(model.named_steps["clf"], "decision_function"):
            decision_scores = model.named_steps["clf"].decision_function(X_test_transformed)
            decision_scores = np.asarray(decision_scores, dtype=float).ravel()
        else:
            # Fallback to predict_proba (for Naive Bayes)
            decision_scores = model.predict_proba(X_test)[:, 1]
            decision_scores = np.asarray(decision_scores, dtype=float).ravel()
    except Exception as e:
        print(f"         [!] Could not get decision scores ({e}), using predict_proba")
        decision_scores = model.predict_proba(X_test)[:, 1]
        decision_scores = np.asarray(decision_scores, dtype=float).ravel()
    
    # Calculate metrics
    acc = accuracy_score(y_test_arr, y_pred_arr)
    prec = precision_score(y_test_arr, y_pred_arr, zero_division=0)
    rec = recall_score(y_test_arr, y_pred_arr, zero_division=0)
    f1 = f1_score(y_test_arr, y_pred_arr, zero_division=0)
    
    # Handle AUC-ROC safely
    try:
        auc_roc = roc_auc_score(y_test_arr, decision_scores)
    except Exception as e:
        print(f"         [!] Could not compute AUC-ROC ({e}), using F1 as proxy")
        auc_roc = f1
    
    metrics = {
        "model": model_name,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc_roc": float(auc_roc),
    }
    
    return metrics, y_pred_arr, decision_scores


def compare_on_single_dataset(dataset_name: str, df: pd.DataFrame) -> Dict:
    """Train and compare SVM vs NB on a single dataset."""
    print(f"\n{'='*70}")
    print(f"[EVAL] Evaluating: {dataset_name} ({len(df)} samples)")
    print(f"{'='*70}")
    
    try:
        # Ensure proper numpy array conversion (not pandas/arrow types)
        X = np.asarray(df["sentence"].values, dtype=object)
        y = np.asarray(df["label"].values, dtype=np.int64)
        
        # Ensure we have minimum samples
        if len(X) < 10:
            print(f"   [!] Dataset too small ({len(X)} samples), skipping")
            return None
        
        # 80-20 split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        results = {"dataset": dataset_name}
        
        # Train SVM
        print("\n   Training SVM (SGD with SVM loss)...")
        svm_model = create_svm_model()
        svm_model.fit(X_train, y_train)
        svm_metrics, svm_pred, svm_scores = evaluate_model(svm_model, X_test, y_test, "SVM")
        results["svm"] = svm_metrics
        
        print(f"      Accuracy:  {svm_metrics['accuracy']:.4f}")
        print(f"      Precision: {svm_metrics['precision']:.4f}")
        print(f"      Recall:    {svm_metrics['recall']:.4f}")
        print(f"      F1:        {svm_metrics['f1']:.4f}")
        print(f"      AUC-ROC:   {svm_metrics['auc_roc']:.4f}")
        
        # Train Naive Bayes
        print("\n   Training Naive Bayes...")
        nb_model = create_naive_bayes_model()
        nb_model.fit(X_train, y_train)
        nb_metrics, nb_pred, nb_scores = evaluate_model(nb_model, X_test, y_test, "Naive Bayes")
        results["naive_bayes"] = nb_metrics
        
        print(f"      Accuracy:  {nb_metrics['accuracy']:.4f}")
        print(f"      Precision: {nb_metrics['precision']:.4f}")
        print(f"      Recall:    {nb_metrics['recall']:.4f}")
        print(f"      F1:        {nb_metrics['f1']:.4f}")
        print(f"      AUC-ROC:   {nb_metrics['auc_roc']:.4f}")
        
        # Calculate differences (SVM - NB)
        print(f"\n   [DIFF] Difference (SVM - NB):")
        for metric in ["accuracy", "precision", "recall", "f1", "auc_roc"]:
            diff = svm_metrics[metric] - nb_metrics[metric]
            symbol = "+" if diff > 0 else "-" if diff < 0 else "="
            print(f"      {symbol} {metric:12s}: {diff:+.4f}")
        
        results["test_size"] = len(X_test)
        
        return results, svm_model, nb_model, X_test, y_test, svm_scores, nb_scores
    
    except Exception as e:
        import traceback
        print(f"   [ERROR] Error: {e}")
        traceback.print_exc()
        return None


def create_comparison_visualizations(all_results: List[Dict], output_dir: str):
    """Create comparison charts."""
    if not all_results:
        print("[!] No results to visualize!")
        return
    
    # Prepare data for plotting
    datasets = [r["dataset"] for r in all_results]
    svm_f1 = [r["svm"]["f1"] for r in all_results]
    nb_f1 = [r["naive_bayes"]["f1"] for r in all_results]
    svm_recall = [r["svm"]["recall"] for r in all_results]
    nb_recall = [r["naive_bayes"]["recall"] for r in all_results]
    svm_precision = [r["svm"]["precision"] for r in all_results]
    nb_precision = [r["naive_bayes"]["precision"] for r in all_results]
    
    # F1 Score Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(datasets))
    width = 0.35
    ax.bar(x - width/2, svm_f1, width, label="SVM", color="#4E79A7")
    ax.bar(x + width/2, nb_f1, width, label="Naive Bayes", color="#E15759")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score Comparison: SVM vs Naive Bayes")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.legend()
    # Zoom into the top-end range to highlight small decimal differences
    ax.set_ylim([0.8, 1.0])
    ax.set_yticks(np.arange(0.80, 1.001, 0.01))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "01_f1_comparison.png"), dpi=300)
    plt.close(fig)
    print(f"   [OK] Saved: 01_f1_comparison.png")
    
    # Recall Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, svm_recall, width, label="SVM", color="#4E79A7")
    ax.bar(x + width/2, nb_recall, width, label="Naive Bayes", color="#E15759")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Recall")
    ax.set_title("Recall Comparison: SVM vs Naive Bayes (Important for Action Item Detection)")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.legend()
    # Zoom into the top-end range to highlight small decimal differences
    ax.set_ylim([0.8, 1.0])
    ax.set_yticks(np.arange(0.80, 1.001, 0.01))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "02_recall_comparison.png"), dpi=300)
    plt.close(fig)
    print(f"   [OK] Saved: 02_recall_comparison.png")
    
    # Precision Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, svm_precision, width, label="SVM", color="#4E79A7")
    ax.bar(x + width/2, nb_precision, width, label="Naive Bayes", color="#E15759")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Precision")
    ax.set_title("Precision Comparison: SVM vs Naive Bayes")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.legend()
    # Zoom into the top-end range to highlight small decimal differences
    ax.set_ylim([0.8, 1.0])
    ax.set_yticks(np.arange(0.80, 1.001, 0.01))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "03_precision_comparison.png"), dpi=300)
    plt.close(fig)
    print(f"   [OK] Saved: 03_precision_comparison.png")
    
    # Create radar chart for average metrics
    from math import pi
    
    avg_svm = {
        "Accuracy": np.mean([r["svm"]["accuracy"] for r in all_results]),
        "Precision": np.mean([r["svm"]["precision"] for r in all_results]),
        "Recall": np.mean([r["svm"]["recall"] for r in all_results]),
        "F1": np.mean([r["svm"]["f1"] for r in all_results]),
        "AUC-ROC": np.mean([r["svm"]["auc_roc"] for r in all_results]),
    }
    
    avg_nb = {
        "Accuracy": np.mean([r["naive_bayes"]["accuracy"] for r in all_results]),
        "Precision": np.mean([r["naive_bayes"]["precision"] for r in all_results]),
        "Recall": np.mean([r["naive_bayes"]["recall"] for r in all_results]),
        "F1": np.mean([r["naive_bayes"]["f1"] for r in all_results]),
        "AUC-ROC": np.mean([r["naive_bayes"]["auc_roc"] for r in all_results]),
    }
    
    categories = list(avg_svm.keys())
    svm_vals = list(avg_svm.values())
    nb_vals = list(avg_nb.values())
    
    angles = [n / len(categories) * 2 * pi for n in range(len(categories))]
    angles += angles[:1]
    
    svm_vals += svm_vals[:1]
    nb_vals += nb_vals[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))
    ax.plot(angles, svm_vals, "o-", linewidth=2, label="SVM", color="#4E79A7")
    ax.fill(angles, svm_vals, alpha=0.25, color="#4E79A7")
    ax.plot(angles, nb_vals, "o-", linewidth=2, label="Naive Bayes", color="#E15759")
    ax.fill(angles, nb_vals, alpha=0.25, color="#E15759")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim([0, 1])
    ax.set_title("Average Metrics: SVM vs Naive Bayes", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "04_metrics_radar.png"), dpi=300)
    plt.close(fig)
    print(f"   [OK] Saved: 04_metrics_radar.png")


def save_comparison_report(all_results: List[Dict], output_dir: str):
    """Save detailed comparison report to CSV."""
    if not all_results:
        print("[ERROR] No results to report!")
        return None
    
    rows = []
    
    for result in all_results:
        row = {
            "Dataset": result["dataset"],
            "Test Size": result["test_size"],
            "SVM Accuracy": result["svm"]["accuracy"],
            "NB Accuracy": result["naive_bayes"]["accuracy"],
            "SVM Precision": result["svm"]["precision"],
            "NB Precision": result["naive_bayes"]["precision"],
            "SVM Recall": result["svm"]["recall"],
            "NB Recall": result["naive_bayes"]["recall"],
            "SVM F1": result["svm"]["f1"],
            "NB F1": result["naive_bayes"]["f1"],
            "SVM AUC-ROC": result["svm"]["auc_roc"],
            "NB AUC-ROC": result["naive_bayes"]["auc_roc"],
            "F1 Difference (SVM-NB)": result["svm"]["f1"] - result["naive_bayes"]["f1"],
            "Recall Difference (SVM-NB)": result["svm"]["recall"] - result["naive_bayes"]["recall"],
        }
        rows.append(row)
    
    df_report = pd.DataFrame(rows)
    report_path = os.path.join(output_dir, "comparison_report.csv")
    df_report.to_csv(report_path, index=False)
    print(f"\n[OK] Saved detailed report: {report_path}")
    
    # Print summary statistics
    print(f"\n{'='*70}")
    print(f"[SUMMARY] SUMMARY STATISTICS")
    print(f"{'='*70}")
    print(f"\nAverage F1 Score:")
    print(f"   SVM:         {df_report['SVM F1'].mean():.4f} (+/-{df_report['SVM F1'].std():.4f})")
    print(f"   Naive Bayes: {df_report['NB F1'].mean():.4f} (+/-{df_report['NB F1'].std():.4f})")
    print(f"   Difference:  {(df_report['SVM F1'].mean() - df_report['NB F1'].mean()):+.4f}")
    
    print(f"\nAverage Recall:")
    print(f"   SVM:         {df_report['SVM Recall'].mean():.4f} (+/-{df_report['SVM Recall'].std():.4f})")
    print(f"   Naive Bayes: {df_report['NB Recall'].mean():.4f} (+/-{df_report['NB Recall'].std():.4f})")
    print(f"   Difference:  {(df_report['SVM Recall'].mean() - df_report['NB Recall'].mean()):+.4f}")
    
    print(f"\nAverage Precision:")
    print(f"   SVM:         {df_report['SVM Precision'].mean():.4f} (+/-{df_report['SVM Precision'].std():.4f})")
    print(f"   Naive Bayes: {df_report['NB Precision'].mean():.4f} (+/-{df_report['NB Precision'].std():.4f})")
    print(f"   Difference:  {(df_report['SVM Precision'].mean() - df_report['NB Precision'].mean()):+.4f}")
    
    print(f"\nAverage AUC-ROC:")
    print(f"   SVM:         {df_report['SVM AUC-ROC'].mean():.4f} (+/-{df_report['SVM AUC-ROC'].std():.4f})")
    print(f"   Naive Bayes: {df_report['NB AUC-ROC'].mean():.4f} (+/-{df_report['NB AUC-ROC'].std():.4f})")
    print(f"   Difference:  {(df_report['SVM AUC-ROC'].mean() - df_report['NB AUC-ROC'].mean()):+.4f}")
    
    print(f"\nDatasets where SVM wins:")
    svm_wins = (df_report['SVM F1'] > df_report['NB F1']).sum()
    print(f"   {svm_wins}/{len(df_report)} datasets have higher SVM F1 score")
    
    print(f"\nDatasets where SVM has better Recall:")
    svm_recall_wins = (df_report['SVM Recall'] > df_report['NB Recall']).sum()
    print(f"   {svm_recall_wins}/{len(df_report)} datasets have higher SVM recall")
    print(f"{'='*70}\n")
    
    return df_report


def main():
    ensure_output_dir()
    
    print("\n" + "="*70)
    print("[START] SVM vs Naive Bayes Comparison for Action Item Detection")
    print("="*70)
    
    # Load all datasets
    print("\n[LOAD] Loading datasets...")
    combined_df = load_all_datasets()
    
    # Split by dataset and evaluate
    all_results = []
    
    for dataset_name, file_path in DATASETS.items():
        if not os.path.exists(file_path):
            continue
        
        df = load_and_normalize(file_path, dataset_name)
        if df is None:
            continue
        
        result_tuple = compare_on_single_dataset(dataset_name, df)
        if result_tuple is not None:
            result, svm_model, nb_model, X_test, y_test, svm_scores, nb_scores = result_tuple
            all_results.append(result)
    
    # Generate reports and visualizations
    print("\n" + "="*70)
    print("[GEN] Generating Comparison Report and Visualizations...")
    print("="*70)
    
    if all_results:
        df_report = save_comparison_report(all_results, OUTPUT_DIR)
        create_comparison_visualizations(all_results, OUTPUT_DIR)
    else:
        print("[ERROR] No successful evaluations to report!")
    
    print("\n" + "="*70)
    print("[DONE] Comparison Complete!")
    print(f"   Results saved to: {OUTPUT_DIR}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
