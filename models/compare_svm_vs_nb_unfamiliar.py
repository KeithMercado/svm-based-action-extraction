"""
Cross-dataset evaluation: Train on half of combined data, test on unfamiliar other half
SVM vs Naive Bayes comparison with temporal/data splitting.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "svm_vs_nb_half_unfamiliar")

DATASETS = {
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

        labels = []
        valid_sentences = []
        for idx, raw_label in enumerate(rows["label"].astype(str)):
            key = raw_label.strip().lower()
            if key not in label_map:
                continue
            labels.append(label_map[key])
            valid_sentences.append(rows["sentence"].iloc[idx])

        if not labels:
            return None

        return pd.DataFrame({
            "sentence": valid_sentences,
            "label": labels,
            "source_file": source_name,
        })
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


def evaluate_model(model, X_test, y_test, model_name: str):
    """Evaluate model and return comprehensive metrics."""
    try:
        y_pred = model.predict(X_test)
        y_test_arr = np.asarray(y_test, dtype=int)
        y_pred_arr = np.asarray(y_pred, dtype=int)
        
        # Get decision scores for ROC curve
        try:
            X_test_transformed = model.named_steps["tfidf"].transform(X_test)
            if hasattr(model.named_steps["clf"], "decision_function"):
                decision_scores = model.named_steps["clf"].decision_function(X_test_transformed)
                decision_scores = np.asarray(decision_scores, dtype=float).ravel()
            else:
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
        
        return metrics, y_pred_arr, decision_scores, y_test_arr
    
    except Exception as e:
        import traceback
        print(f"         [ERROR] Fatal error in evaluate_model: {e}")
        traceback.print_exc()
        raise


def create_confusion_matrix_plot(cm, model_name, output_path):
    """Save confusion matrix visualization."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=["Pred Info", "Pred Action"],
        yticklabels=["True Info", "True Action"],
        title=f"Confusion Matrix - {model_name}",
        ylabel="True label",
        xlabel="Predicted label",
    )

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"   [OK] Saved: {os.path.basename(output_path)}")


def create_roc_curve_plot(y_test, svm_scores, nb_scores, output_path):
    """Save ROC curve comparison."""
    from sklearn.metrics import roc_curve, auc
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_scores)
    fpr_nb, tpr_nb, _ = roc_curve(y_test, nb_scores)
    
    auc_svm = auc(fpr_svm, tpr_svm)
    auc_nb = auc(fpr_nb, tpr_nb)
    
    ax.plot(fpr_svm, tpr_svm, color="#4E79A7", lw=2, label=f"SVM (AUC = {auc_svm:.3f})")
    ax.plot(fpr_nb, tpr_nb, color="#E15759", lw=2, label=f"Naive Bayes (AUC = {auc_nb:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random Classifier")
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison (Unfamiliar Test Data)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"   [OK] Saved: {os.path.basename(output_path)}")


def create_pr_curve_plot(y_test, svm_scores, nb_scores, output_path):
    """Save Precision-Recall curve comparison."""
    from sklearn.metrics import precision_recall_curve, auc
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    prec_svm, rec_svm, _ = precision_recall_curve(y_test, svm_scores)
    prec_nb, rec_nb, _ = precision_recall_curve(y_test, nb_scores)
    
    auc_svm = auc(rec_svm, prec_svm)
    auc_nb = auc(rec_nb, prec_nb)
    
    ax.plot(rec_svm, prec_svm, color="#4E79A7", lw=2, label=f"SVM (AUC = {auc_svm:.3f})")
    ax.plot(rec_nb, prec_nb, color="#E15759", lw=2, label=f"Naive Bayes (AUC = {auc_nb:.3f})")
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (Unfamiliar Test Data)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"   [OK] Saved: {os.path.basename(output_path)}")


def main():
    ensure_output_dir()
    
    print("\n" + "="*70)
    print("[START] Cross-Dataset Training (50% train, 50% unfamiliar test)")
    print("="*70)
    
    # Load all datasets combined
    print("\n[LOAD] Loading all datasets...")
    combined_df = load_all_datasets()
    
    # Prepare data: ensure proper numpy arrays
    X_all = np.asarray(combined_df["sentence"].values, dtype=object)
    y_all = np.asarray(combined_df["label"].values, dtype=np.int64)
    
    # 50-50 split: train on first half, test on second half (unfamiliar)
    # Using stratified split to maintain class balance
    print("\n[SPLIT] Creating 50-50 train/test split with stratification...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.5, random_state=42, stratify=y_all
    )
    
    print(f"   Training set: {len(X_train)} samples")
    print(f"   - Action items: {(y_train == 1).sum()} ({(y_train == 1).sum() / len(y_train) * 100:.1f}%)")
    print(f"   Test set (unfamiliar): {len(X_test)} samples")
    print(f"   - Action items: {(y_test == 1).sum()} ({(y_test == 1).sum() / len(y_test) * 100:.1f}%)")
    
    # Train SVM
    print("\n" + "="*70)
    print("[TRAIN] Training SVM (SGD with SVM loss)...")
    print("="*70)
    svm_model = create_svm_model()
    svm_model.fit(X_train, y_train)
    print("[OK] SVM training complete")
    
    # Train Naive Bayes
    print("\n[TRAIN] Training Naive Bayes...")
    nb_model = create_naive_bayes_model()
    nb_model.fit(X_train, y_train)
    print("[OK] Naive Bayes training complete")
    
    # Evaluate on unfamiliar test data
    print("\n" + "="*70)
    print("[EVAL] Evaluating on unfamiliar test data...")
    print("="*70)
    
    svm_metrics, svm_pred, svm_scores, y_test_arr = evaluate_model(svm_model, X_test, y_test, "SVM")
    nb_metrics, nb_pred, nb_scores, _ = evaluate_model(nb_model, X_test, y_test, "Naive Bayes")
    
    # Display results
    print("\n" + "="*70)
    print("[RESULTS] SVM vs Naive Bayes on Unfamiliar Data")
    print("="*70)
    
    print(f"\nSVM Metrics:")
    print(f"   Accuracy:  {svm_metrics['accuracy']:.4f}")
    print(f"   Precision: {svm_metrics['precision']:.4f}")
    print(f"   Recall:    {svm_metrics['recall']:.4f}")
    print(f"   F1:        {svm_metrics['f1']:.4f}")
    print(f"   AUC-ROC:   {svm_metrics['auc_roc']:.4f}")
    
    print(f"\nNaive Bayes Metrics:")
    print(f"   Accuracy:  {nb_metrics['accuracy']:.4f}")
    print(f"   Precision: {nb_metrics['precision']:.4f}")
    print(f"   Recall:    {nb_metrics['recall']:.4f}")
    print(f"   F1:        {nb_metrics['f1']:.4f}")
    print(f"   AUC-ROC:   {nb_metrics['auc_roc']:.4f}")
    
    print(f"\n[DIFF] Difference (SVM - NB):")
    for metric in ["accuracy", "precision", "recall", "f1", "auc_roc"]:
        diff = svm_metrics[metric] - nb_metrics[metric]
        symbol = "+" if diff > 0 else "-" if diff < 0 else "="
        print(f"   {symbol} {metric:12s}: {diff:+.4f}")
    
    # Classification reports
    print(f"\n[REPORT] SVM Classification Report:")
    print(classification_report(y_test_arr, svm_pred, target_names=["Non-Action", "Action"]))
    
    print(f"\n[REPORT] Naive Bayes Classification Report:")
    print(classification_report(y_test_arr, nb_pred, target_names=["Non-Action", "Action"]))
    
    # Generate visualizations
    print("\n" + "="*70)
    print("[VIZ] Generating visualizations...")
    print("="*70)
    
    # Confusion matrices
    cm_svm = confusion_matrix(y_test_arr, svm_pred)
    cm_nb = confusion_matrix(y_test_arr, nb_pred)
    
    create_confusion_matrix_plot(cm_svm, "SVM", 
        os.path.join(OUTPUT_DIR, "confusion_matrix_svm.png"))
    create_confusion_matrix_plot(cm_nb, "Naive Bayes", 
        os.path.join(OUTPUT_DIR, "confusion_matrix_nb.png"))
    
    # ROC curves
    create_roc_curve_plot(y_test_arr, svm_scores, nb_scores,
        os.path.join(OUTPUT_DIR, "roc_curve_comparison.png"))
    
    # PR curves
    create_pr_curve_plot(y_test_arr, svm_scores, nb_scores,
        os.path.join(OUTPUT_DIR, "pr_curve_comparison.png"))
    
    # Save detailed results
    results_df = pd.DataFrame([
        {
            "Metric": "Accuracy",
            "SVM": svm_metrics["accuracy"],
            "Naive Bayes": nb_metrics["accuracy"],
            "Difference": svm_metrics["accuracy"] - nb_metrics["accuracy"],
        },
        {
            "Metric": "Precision",
            "SVM": svm_metrics["precision"],
            "Naive Bayes": nb_metrics["precision"],
            "Difference": svm_metrics["precision"] - nb_metrics["precision"],
        },
        {
            "Metric": "Recall",
            "SVM": svm_metrics["recall"],
            "Naive Bayes": nb_metrics["recall"],
            "Difference": svm_metrics["recall"] - nb_metrics["recall"],
        },
        {
            "Metric": "F1",
            "SVM": svm_metrics["f1"],
            "Naive Bayes": nb_metrics["f1"],
            "Difference": svm_metrics["f1"] - nb_metrics["f1"],
        },
        {
            "Metric": "AUC-ROC",
            "SVM": svm_metrics["auc_roc"],
            "Naive Bayes": nb_metrics["auc_roc"],
            "Difference": svm_metrics["auc_roc"] - nb_metrics["auc_roc"],
        },
    ])
    
    results_path = os.path.join(OUTPUT_DIR, "unfamiliar_data_comparison.csv")
    results_df.to_csv(results_path, index=False)
    print(f"   [OK] Saved detailed report: unfamiliar_data_comparison.csv")
    
    print("\n" + "="*70)
    print("[DONE] Cross-Dataset Evaluation Complete!")
    print(f"   Results saved to: {OUTPUT_DIR}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
