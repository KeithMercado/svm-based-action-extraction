"""Model evaluation and thesis plot generation for the SVM/SGD classifier."""

import argparse
import os
import pickle
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "model_eval")

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

MODEL_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "svm_model.pkl"),
    os.path.join(SCRIPT_DIR, "svm_model.pkl"),
    os.path.join(PROJECT_ROOT, "svm_bulk_model.pkl"),
    os.path.join(SCRIPT_DIR, "svm_bulk_model.pkl"),
]

CLASS_NAMES = ["Non-Action", "Action Item"]

vectorizer = HashingVectorizer(
    n_features=2**16,
    ngram_range=(1, 3),
    alternate_sign=False,
)


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def resolve_model_path(preferred_model_path: str = "") -> str:
    if preferred_model_path:
        if os.path.exists(preferred_model_path):
            return preferred_model_path
        raise FileNotFoundError(f"Preferred model path not found: {preferred_model_path}")

    for candidate in MODEL_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not find svm_model.pkl or svm_bulk_model.pkl in the project root or models/ folder."
    )


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    return " ".join(text.split())


def load_and_normalize(file_path: str, source_name: str) -> pd.DataFrame:
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
        raise ValueError(f"{file_path} must contain sentence/text and label columns.")

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
        raise ValueError(f"No valid labels found in {file_path}")

    return pd.DataFrame(
        {
            "sentence": valid_sentences,
            "label": labels,
            "source_file": source_name,
        }
    )


def load_all_datasets(selected: List[str]) -> pd.DataFrame:
    parts = []
    for dataset_name in selected:
        file_path = DATASETS[dataset_name]
        if not os.path.exists(file_path):
            print(f"[Skip] {dataset_name}: file not found -> {file_path}")
            continue
        try:
            df = load_and_normalize(file_path, dataset_name)
            parts.append(df)
            print(f"[Load] {dataset_name}: {len(df)} rows")
        except Exception as exc:
            print(f"[Skip] {dataset_name}: {exc}")

    if not parts:
        raise ValueError("No datasets could be loaded.")

    combined = pd.concat(parts, ignore_index=True)
    combined = combined.dropna(subset=["sentence", "label"])
    combined["label"] = combined["label"].astype(int)
    return combined


def split_by_source(df: pd.DataFrame, test_ratio: float, val_ratio: float, random_seed: int):
    """Split by source_file to reduce source leakage into validation/test."""
    unique_sources = sorted(df["source_file"].astype(str).unique().tolist())
    if len(unique_sources) < 3:
        raise ValueError(
            "Need at least 3 distinct source datasets for train/val/test split by source_file."
        )

    train_sources, test_sources = train_test_split(
        unique_sources,
        test_size=test_ratio,
        random_state=random_seed,
    )

    val_share_of_train = val_ratio / max(1e-8, (1.0 - test_ratio))
    val_share_of_train = min(max(val_share_of_train, 0.05), 0.5)

    train_sources, val_sources = train_test_split(
        train_sources,
        test_size=val_share_of_train,
        random_state=random_seed,
    )

    train_df = df[df["source_file"].isin(train_sources)].copy()
    val_df = df[df["source_file"].isin(val_sources)].copy()
    test_df = df[df["source_file"].isin(test_sources)].copy()

    return train_df, val_df, test_df, train_sources, val_sources, test_sources


def get_features(model, texts: pd.Series):
    if hasattr(model, "named_steps"):
        return texts
    return vectorizer.transform(texts)


def get_scores(model, features):
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(features), dtype=float)

    clf = None
    if hasattr(model, "named_steps"):
        clf = model.named_steps.get("clf")
    if clf is not None and hasattr(clf, "decision_function"):
        return np.asarray(clf.decision_function(features), dtype=float)

    raise RuntimeError("Loaded model does not expose decision_function for PR analysis.")


def choose_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold_mode: str,
    target_recall: float,
    target_precision: float,
) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    if thresholds.size == 0:
        return 0.0

    if threshold_mode == "precision":
        candidate_idxs = np.where(precisions[:-1] >= target_precision)[0]
        if candidate_idxs.size > 0:
            best_idx = candidate_idxs[np.argmax(recalls[:-1][candidate_idxs])]
        else:
            best_idx = int(np.argmax(precisions[:-1]))
    else:
        candidate_idxs = np.where(recalls[:-1] >= target_recall)[0]
        if candidate_idxs.size > 0:
            best_idx = candidate_idxs[np.argmax(precisions[:-1][candidate_idxs])]
        else:
            best_idx = int(np.argmax(recalls[:-1]))

    return float(thresholds[best_idx])


def plot_class_performance(report: dict, output_path: str):
    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(metrics))
    width = 0.36

    non_action_key = CLASS_NAMES[0] if CLASS_NAMES[0] in report else "0"
    action_key = CLASS_NAMES[1] if CLASS_NAMES[1] in report else "1"

    non_action = [report[non_action_key][metric] for metric in metrics]
    action = [report[action_key][metric] for metric in metrics]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_1 = ax.bar(x - width / 2, non_action, width, label=CLASS_NAMES[0], color="#486B8A")
    bars_2 = ax.bar(x + width / 2, action, width, label=CLASS_NAMES[1], color="#4AAE7A")

    ax.set_title("Model Performance Metrics by Class")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(["Precision", "Recall", "F1-Score"])
    ax.set_ylim(0, 1.08)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.45)

    for bars in (bars_1, bars_2):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.015,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_precision_recall_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    output_path: str,
    target_recall: float,
    threshold_mode: str,
    target_precision: float,
):
    precisions, recalls, _ = precision_recall_curve(y_true, scores)
    y_pred = (scores >= threshold).astype(int)
    chosen_precision = precision_score(y_true, y_pred, zero_division=0)
    chosen_recall = recall_score(y_true, y_pred, zero_division=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(recalls, precisions, color="#FF8C00", marker="o", markersize=3, linewidth=2, label="SVM (SGD) Curve")
    ax.fill_between(recalls, precisions, alpha=0.15, color="#FFB84D")
    mode_label = "Recall priority" if threshold_mode == "recall" else "Precision priority"
    ax.scatter([chosen_recall], [chosen_precision], s=120, color="red", zorder=5, label=f"Chosen Operating Point ({mode_label})")

    ax.annotate(
        f"threshold={threshold:.4f}\nP={chosen_precision:.2f}, R={chosen_recall:.2f}",
        xy=(chosen_recall, chosen_precision),
        xytext=(min(0.98, chosen_recall + 0.05), min(1.0, chosen_precision + 0.08)),
        arrowprops=dict(arrowstyle="->", color="red", lw=1),
        fontsize=9,
    )

    ax.set_title("Precision-Recall Curve (Action Item Class)")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (Quality)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower left")
    ax.text(
        0.02,
        0.03,
        f"Mode: {mode_label} | Target recall: {target_recall:.2f} | Target precision: {target_precision:.2f}",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="#CCCCCC"),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_normalized_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: str):
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        title="Normalized Confusion Matrix (All Samples)",
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j] * 100.0
            ax.text(
                j,
                i,
                f"{value:.2f}%",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def run_evaluation(
    selected_datasets: List[str],
    target_recall: float,
    threshold_mode: str,
    target_precision: float,
    model_path_override: str,
    test_ratio: float,
    val_ratio: float,
    random_seed: int,
):
    ensure_output_dir()
    model_path = resolve_model_path(model_path_override)

    print(f"[System] Using model: {model_path}")
    with open(model_path, "rb") as handle:
        model = pickle.load(handle)

    print("\n" + "=" * 72)
    print("MODEL EVALUATION")
    print("=" * 72)

    df = load_all_datasets(selected_datasets)
    print(f"[Data] Combined samples: {len(df)}")
    print(f"[Data] Class distribution: {df['label'].value_counts().to_dict()}")

    train_df, val_df, test_df, train_sources, val_sources, test_sources = split_by_source(
        df,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        random_seed=random_seed,
    )

    print(f"[Split] train sources: {len(train_sources)} | val sources: {len(val_sources)} | test sources: {len(test_sources)}")
    print(f"[Split] train rows: {len(train_df)} | val rows: {len(val_df)} | test rows: {len(test_df)}")

    val_features = get_features(model, val_df["sentence"])
    val_y_true = val_df["label"].to_numpy(dtype=int)
    val_scores = get_scores(model, val_features)

    test_features = get_features(model, test_df["sentence"])
    y_true = test_df["label"].to_numpy(dtype=int)
    scores = get_scores(model, test_features)

    default_pred = (scores >= 0.0).astype(int)

    chosen_threshold = choose_threshold(
        val_y_true,
        val_scores,
        threshold_mode=threshold_mode,
        target_recall=target_recall,
        target_precision=target_precision,
    )
    y_pred = (scores >= chosen_threshold).astype(int)

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    accuracy_default = accuracy_score(y_true, default_pred)
    accuracy_recall = accuracy_score(y_true, y_pred)
    precision_recall = precision_score(y_true, y_pred, zero_division=0)
    recall_recall = recall_score(y_true, y_pred, zero_division=0)
    f1_recall = f1_score(y_true, y_pred, zero_division=0)

    print(f"[Threshold] Chosen {threshold_mode}-priority threshold (from validation): {chosen_threshold:.5f}")
    print(f"[Default] Accuracy: {accuracy_default * 100:.2f}%")
    print(f"[Tuned-on-val / Test] Accuracy: {accuracy_recall * 100:.2f}%")
    print(f"[Tuned-on-val / Test] Precision: {precision_recall:.4f}")
    print(f"[Tuned-on-val / Test] Recall: {recall_recall:.4f}")
    print(f"[Tuned-on-val / Test] F1: {f1_recall:.4f}")

    print("\n" + "=" * 72)
    print("PER-CLASS METRICS (TEST SET)")
    print("=" * 72)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

    perf_plot_path = os.path.join(OUTPUT_DIR, "model_performance_by_class.png")
    pr_plot_path = os.path.join(OUTPUT_DIR, "precision_recall_curve.png")
    cm_plot_path = os.path.join(OUTPUT_DIR, "normalized_confusion_matrix_pct.png")
    summary_path = os.path.join(OUTPUT_DIR, "evaluation_summary.csv")

    plot_class_performance(report, perf_plot_path)
    plot_precision_recall_curve(
        y_true,
        scores,
        chosen_threshold,
        pr_plot_path,
        target_recall,
        threshold_mode,
        target_precision,
    )
    plot_normalized_confusion_matrix(y_true, y_pred, cm_plot_path)

    summary_rows = [
        {"metric": "accuracy_default_threshold", "value": accuracy_default},
        {"metric": "accuracy_recall_priority", "value": accuracy_recall},
        {"metric": "precision_recall_priority", "value": precision_recall},
        {"metric": "recall_recall_priority", "value": recall_recall},
        {"metric": "f1_recall_priority", "value": f1_recall},
        {"metric": "chosen_threshold", "value": chosen_threshold},
        {"metric": "threshold_mode", "value": threshold_mode},
        {"metric": "target_recall", "value": target_recall},
        {"metric": "target_precision", "value": target_precision},
        {"metric": "train_samples", "value": len(train_df)},
        {"metric": "validation_samples", "value": len(val_df)},
        {"metric": "test_samples", "value": len(test_df)},
        {"metric": "test_ratio", "value": test_ratio},
        {"metric": "val_ratio", "value": val_ratio},
        {"metric": "random_seed", "value": random_seed},
    ]
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print("\n" + "=" * 72)
    print("SAVED ARTIFACTS")
    print("=" * 72)
    print(f"[Plot] {perf_plot_path}")
    print(f"[Plot] {pr_plot_path}")
    print(f"[Plot] {cm_plot_path}")
    print(f"[Summary] {summary_path}")

    print("\nSuggestion:")
    print("If this feels too busy, keep the class metric chart and normalized confusion matrix for the thesis PDF, and move the full precision-recall curve to an appendix figure.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate model performance plots for the SVM/SGD classifier."
    )
    parser.add_argument(
        "--dataset",
        choices=["all"] + list(DATASETS.keys()),
        default="all",
        help="Choose one dataset or all datasets combined.",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.95,
        help="Recall target used to choose the operating threshold.",
    )
    parser.add_argument(
        "--threshold-mode",
        choices=["recall", "precision"],
        default="recall",
        help="Threshold selection strategy: recall priority or precision priority.",
    )
    parser.add_argument(
        "--target-precision",
        type=float,
        default=0.70,
        help="Precision target when threshold-mode=precision.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Optional model path override (useful for separate experiment PKLs).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.20,
        help="Ratio of source datasets reserved for test split.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.20,
        help="Ratio of source datasets reserved for validation split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for source split.",
    )

    args = parser.parse_args()

    selected = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    run_evaluation(
        selected,
        target_recall=args.target_recall,
        threshold_mode=args.threshold_mode,
        target_precision=args.target_precision,
        model_path_override=args.model_path,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()