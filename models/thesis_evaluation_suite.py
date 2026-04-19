"""
Thesis evaluation suite.

Generates final-days artifacts:
1) Error taxonomy table for false negatives
2) Threshold tradeoff figure (balanced vs high recall)
3) External validity split (hold out one dataset family)
4) Confidence score distribution for TP/FP/FN
5) Ablation study results
6) Precision-recall chart
"""

import argparse
import os
import pickle
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.pipeline import FeatureUnion, Pipeline


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "thesis_eval")


TAGALOG_MARKERS = {
    "ako", "ikaw", "siya", "kami", "tayo", "kayo", "sila", "natin", "namin",
    "gawin", "gagawin", "kailangan", "paki", "po", "na", "nang", "lang", "ulit",
    "pwede", "sige", "mamaya", "bukas", "ngayon", "yung", "yung"
}

ENGLISH_MARKERS = {
    "will", "should", "need", "needs", "assign", "deadline", "tomorrow", "today",
    "please", "action", "follow", "update", "send", "review", "complete", "owner"
}

ASR_NOISE_MARKERS = {
    "uh", "um", "erm", "hmm", "ah", "[inaudible]", "inaudible", "noise", "static"
}

INDIRECT_COMMITMENT_PHRASES = [
    "can we",
    "could we",
    "should we",
    "it would be good",
    "let us",
    "let's",
    "maybe we",
    "we might",
]

PRONOUNS = {"he", "she", "they", "it", "this", "that", "these", "those", "siya", "sila", "ito", "iyan"}


@dataclass
class EvalResult:
    mode_name: str
    threshold: float
    precision: float
    recall: float
    f1_action: float
    f1_macro: float
    fp: int
    fn: int
    tp: int
    tn: int


def clean_text(text: str, lowercase: bool = True) -> str:
    if not isinstance(text, str):
        return ""
    if lowercase:
        text = text.lower()
    text = text.strip()
    text = " ".join(text.split())
    return text


def normalize_labels(df: pd.DataFrame, source_file: str) -> Tuple[Optional[pd.DataFrame], bool]:
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
    df_copy.columns = [c.lower().strip() for c in df_copy.columns]

    text_col = None
    if "sentence" in df_copy.columns:
        text_col = "sentence"
    elif "text" in df_copy.columns:
        text_col = "text"
    else:
        for col in df_copy.columns:
            if "text" in col.lower() or "sentence" in col.lower() or "content" in col.lower():
                text_col = col
                break

    label_col = None
    if "label" in df_copy.columns:
        label_col = "label"
    else:
        for col in df_copy.columns:
            if "label" in col.lower() or "type" in col.lower() or "classification" in col.lower():
                label_col = col
                break

    if text_col is None or label_col is None:
        print(f"Skipped {source_file}: missing text/label columns")
        return None, False

    normalized_labels: List[int] = []
    valid_texts: List[str] = []
    for idx, raw_label in enumerate(df_copy[label_col].astype(str)):
        key = raw_label.strip().lower()
        if key not in label_map:
            continue
        normalized_labels.append(label_map[key])
        valid_texts.append(str(df_copy[text_col].iloc[idx]))

    if not normalized_labels:
        print(f"Skipped {source_file}: no valid labels after normalization")
        return None, False

    result = pd.DataFrame({
        "sentence": valid_texts,
        "label": normalized_labels,
        "source_file": source_file,
    })
    return result, True


def load_labeled_corpus(max_rows_per_file: int = 0) -> pd.DataFrame:
    csv_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])
    parts = []

    for csv_file in csv_files:
        file_path = os.path.join(DATA_DIR, csv_file)
        try:
            df = pd.read_csv(file_path)
            normalized, ok = normalize_labels(df, csv_file)
            if not ok:
                continue
            if max_rows_per_file > 0 and len(normalized) > max_rows_per_file:
                normalized = normalized.sample(n=max_rows_per_file, random_state=42)
            parts.append(normalized)
        except Exception as exc:
            print(f"Failed to load {csv_file}: {exc}")

    if not parts:
        raise ValueError("No labeled CSV rows found in data folder")

    all_data = pd.concat(parts, ignore_index=True)
    all_data = all_data.dropna(subset=["sentence", "label"])
    return all_data


def apply_data_hygiene(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """
    Data hygiene pass:
    - remove near-duplicates by normalized sentence
    - surface conflicting labels for same normalized sentence
    - flag potential indirect-commitment rows for manual consistency check
    """
    cleaned = df.copy()
    cleaned["sentence_raw"] = cleaned["sentence"].astype(str)
    cleaned["sentence"] = cleaned["sentence_raw"].apply(lambda s: clean_text(s, lowercase=True))
    cleaned["sentence_norm"] = cleaned["sentence"].str.replace(r"\s+", " ", regex=True).str.strip()

    conflict_summary = (
        cleaned.groupby("sentence_norm")["label"]
        .nunique()
        .reset_index(name="distinct_labels")
    )
    conflict_keys = conflict_summary[conflict_summary["distinct_labels"] > 1]["sentence_norm"]

    conflicts_df = cleaned[cleaned["sentence_norm"].isin(conflict_keys)].copy()
    if len(conflicts_df) > 0:
        conflicts_df.to_csv(
            os.path.join(output_dir, "data_hygiene_label_conflicts.csv"),
            index=False,
        )

    # Keep majority label for conflicting normalized sentences.
    majority_labels = (
        cleaned.groupby("sentence_norm")["label"]
        .agg(lambda s: int(s.value_counts().idxmax()))
        .to_dict()
    )
    cleaned["label"] = cleaned["sentence_norm"].map(majority_labels).astype(int)

    before = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset=["sentence_norm"], keep="first")
    after = len(cleaned)

    dedup_stats = pd.DataFrame(
        [
            {"metric": "rows_before_dedup", "value": before},
            {"metric": "rows_after_dedup", "value": after},
            {"metric": "duplicates_removed", "value": before - after},
            {"metric": "conflict_groups", "value": int(len(conflict_keys))},
        ]
    )
    dedup_stats.to_csv(os.path.join(output_dir, "data_hygiene_summary.csv"), index=False)

    indirect_review_df = cleaned[
        cleaned["sentence"].str.contains(
            "|".join([
                "can we",
                "could we",
                "should we",
                "it would be good",
                "let us",
                "let's",
                "maybe we",
                "we might",
            ]),
            case=False,
            regex=True,
        )
    ].copy()
    if len(indirect_review_df) > 0:
        indirect_review_df[["sentence", "label", "source_file"]].to_csv(
            os.path.join(output_dir, "label_consistency_indirect_commitment_review.csv"),
            index=False,
        )

    cleaned = cleaned.drop(columns=["sentence_norm"])
    return cleaned


def load_hard_negative_rows(hard_negatives_file: str) -> pd.DataFrame:
    """Load manually labeled hard negatives and normalize to sentence/label schema."""
    hard_df = pd.read_csv(hard_negatives_file)
    if "sentence" not in hard_df.columns:
        raise ValueError("Hard-negatives file must contain a 'sentence' column")

    if "label" in hard_df.columns:
        labels = hard_df["label"].astype(int)
    else:
        labels = pd.Series(np.zeros(len(hard_df), dtype=int))

    result = pd.DataFrame(
        {
            "sentence": hard_df["sentence"].astype(str),
            "sentence_raw": hard_df["sentence"].astype(str),
            "label": labels,
            "source_file": "manual_hard_negative",
        }
    )
    return result


def build_runtime_vectorizer(vectorizer_n_features: int = 2**16, ngram_range: Tuple[int, int] = (1, 3)):
    """Build the same HashingVectorizer configuration used by the deployed app."""
    from sklearn.feature_extraction.text import HashingVectorizer

    return HashingVectorizer(
        n_features=vectorizer_n_features,
        ngram_range=ngram_range,
        alternate_sign=False,
    )


def load_pickled_classifier(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model pickle not found: {model_path}")
    with open(model_path, "rb") as handle:
        return pickle.load(handle)


def get_scoring_texts(df: pd.DataFrame, runtime_mode: bool) -> pd.Series:
    if runtime_mode:
        if "sentence_raw" not in df.columns:
            return df["sentence"].astype(str)
        return df["sentence_raw"].astype(str)
    return df["sentence"].astype(str)


def split_external_validity(df: pd.DataFrame, holdout_keyword: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    holdout_mask = df["source_file"].str.lower().str.contains(holdout_keyword.lower())
    train_df = df[~holdout_mask].copy()
    test_df = df[holdout_mask].copy()

    if len(test_df) == 0:
        raise ValueError(
            f"No held-out rows found. Adjust holdout keyword. Current keyword: {holdout_keyword}"
        )
    if len(train_df) == 0:
        raise ValueError("Holdout split removed all training rows. Choose a narrower holdout keyword")

    return train_df, test_df


def build_pipeline(
    ngram_range: Tuple[int, int] = (1, 2),
    class_weight: Optional[str] = "balanced",
    use_char_ngrams: bool = False,
    char_ngram_range: Tuple[int, int] = (3, 5),
) -> Pipeline:
    if use_char_ngrams:
        vectorizer = FeatureUnion(
            [
                (
                    "word_tfidf",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=ngram_range,
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
                        ngram_range=char_ngram_range,
                        max_features=60000,
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )
    else:
        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=ngram_range,
            max_features=100000,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )

    return Pipeline(
        [
            ("tfidf", vectorizer),
            (
                "clf",
                SGDClassifier(
                    loss="hinge",
                    penalty="l2",
                    alpha=0.0001,
                    max_iter=2000,
                    tol=1e-4,
                    class_weight=class_weight,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def choose_high_recall_threshold(y_true: np.ndarray, scores: np.ndarray, target_recall: float) -> float:
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, scores)
    if thresholds.size == 0:
        return 0.0

    candidate = np.where(recall_vals[:-1] >= target_recall)[0]
    if candidate.size > 0:
        best_idx = candidate[np.argmax(precision_vals[:-1][candidate])]
    else:
        best_idx = int(np.argmax(recall_vals[:-1]))

    return float(thresholds[best_idx])


def evaluate_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float, mode_name: str) -> EvalResult:
    y_pred = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return EvalResult(
        mode_name=mode_name,
        threshold=threshold,
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1_action=f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        f1_macro=f1_score(y_true, y_pred, average="macro", zero_division=0),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
        tn=int(tn),
    )


def _tokenize(sentence: str) -> List[str]:
    return [tok for tok in sentence.lower().replace("/", " ").replace("-", " ").split() if tok]


def classify_fn_error(sentence: str) -> str:
    text = sentence.lower()
    toks = set(_tokenize(sentence))

    has_tagalog = len(toks.intersection(TAGALOG_MARKERS)) > 0
    has_english = len(toks.intersection(ENGLISH_MARKERS)) > 0
    if has_tagalog and has_english:
        return "Taglish mix"

    if any(marker in text for marker in ASR_NOISE_MARKERS):
        return "ASR noise"

    if any(phrase in text for phrase in INDIRECT_COMMITMENT_PHRASES):
        return "Indirect commitment"

    if len(toks.intersection(PRONOUNS)) > 0 and len(text.split()) <= 14:
        return "Pronoun ambiguity"

    if len(text.split()) >= 28 or "as discussed" in text or "as mentioned" in text:
        return "Long-context dependency"

    return "Other"


def build_error_taxonomy(y_true: np.ndarray, y_pred: np.ndarray, test_sentences: pd.Series) -> pd.DataFrame:
    required_categories = [
        "Taglish mix",
        "ASR noise",
        "Indirect commitment",
        "Pronoun ambiguity",
        "Long-context dependency",
        "Other",
    ]

    fn_mask = (y_true == 1) & (y_pred == 0)
    fn_sentences = test_sentences[fn_mask]

    if len(fn_sentences) == 0:
        return pd.DataFrame(
            [{"error_type": c, "count": 0, "percentage": 0.0} for c in required_categories]
        )

    categories = [classify_fn_error(s) for s in fn_sentences.tolist()]
    counts_dict = pd.Series(categories).value_counts().to_dict()

    rows = []
    total = len(fn_sentences)
    for category in required_categories:
        count = int(counts_dict.get(category, 0))
        percentage = round((count / total) * 100.0, 2) if total > 0 else 0.0
        rows.append({
            "error_type": category,
            "count": count,
            "percentage": percentage,
        })

    counts = pd.DataFrame(rows)
    return counts


def export_hard_negative_candidates(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    output_path: str,
    top_k: int = 300,
):
    fp_mask = (y_true == 0) & (y_pred == 1)
    fp_rows = test_df[fp_mask].copy()
    sentence_column = "sentence_raw" if "sentence_raw" in fp_rows.columns else "sentence"
    if len(fp_rows) == 0:
        pd.DataFrame(
            columns=[
                "sentence",
                "source_file",
                "decision_score",
                "true_label",
                "predicted_label",
                "review_label",
                "review_notes",
            ]
        ).to_csv(output_path, index=False)
        return

    fp_rows["decision_score"] = scores[fp_mask]
    fp_rows["true_label"] = 0
    fp_rows["predicted_label"] = 1
    fp_rows = fp_rows.sort_values("decision_score", ascending=False).head(top_k)
    fp_rows["review_label"] = ""
    fp_rows["review_notes"] = ""
    fp_rows[[
        sentence_column,
        "source_file",
        "decision_score",
        "true_label",
        "predicted_label",
        "review_label",
        "review_notes",
    ]].rename(columns={sentence_column: "sentence"}).to_csv(output_path, index=False)


def export_fn_manual_annotation_template(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    output_path: str,
    top_k: int = 120,
):
    fn_mask = (y_true == 1) & (y_pred == 0)
    fn_rows = test_df[fn_mask].copy()
    sentence_column = "sentence_raw" if "sentence_raw" in fn_rows.columns else "sentence"
    if len(fn_rows) == 0:
        pd.DataFrame(
            columns=[
                "sentence",
                "source_file",
                "decision_score",
                "true_label",
                "predicted_label",
                "annotator_category",
                "annotator_notes",
            ]
        ).to_csv(output_path, index=False)
        return

    fn_rows["decision_score"] = scores[fn_mask]
    fn_rows["true_label"] = 1
    fn_rows["predicted_label"] = 0
    fn_rows = fn_rows.sort_values("decision_score", ascending=True).head(top_k)
    fn_rows["annotator_category"] = ""
    fn_rows["annotator_notes"] = ""
    fn_rows[[
        sentence_column,
        "source_file",
        "decision_score",
        "true_label",
        "predicted_label",
        "annotator_category",
        "annotator_notes",
    ]].rename(columns={sentence_column: "sentence"}).to_csv(output_path, index=False)


def save_precision_recall_plot(y_true: np.ndarray, scores: np.ndarray, output_path: str) -> float:
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, scores)
    pr_auc = auc(recall_vals, precision_vals)

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.plot(recall_vals, precision_vals, color="#1f77b4", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve (AUC={pr_auc:.4f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return float(pr_auc)


def save_threshold_tradeoff_figure(results: List[EvalResult], output_path: str):
    labels = [r.mode_name for r in results]
    fn_vals = [r.fn for r in results]
    fp_vals = [r.fp for r in results]
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    bars_fn = ax.bar(x - width / 2, fn_vals, width, label="Missed actions (FN)", color="#e15759")
    bars_fp = ax.bar(x + width / 2, fp_vals, width, label="Extra false alarms (FP)", color="#4e79a7")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Count")
    ax.set_title("Threshold Tradeoff: Missed Actions vs Extra False Alarms")
    ax.legend()

    for i, r in enumerate(results):
        ax.text(x[i], max(fn_vals[i], fp_vals[i]) + 1, f"R={r.recall:.3f} P={r.precision:.3f}", ha="center")

    for bar in list(bars_fn) + list(bars_fp):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4, f"{int(h)}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_score_distribution_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    balanced_threshold: float,
    high_recall_threshold: float,
    output_path: str,
):
    tp_scores = scores[(y_true == 1) & (y_pred == 1)]
    fp_scores = scores[(y_true == 0) & (y_pred == 1)]
    fn_scores = scores[(y_true == 1) & (y_pred == 0)]

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    bins = 40

    if len(tp_scores) > 0:
        ax.hist(tp_scores, bins=bins, alpha=0.55, label="TP score", color="#59a14f", density=True)
    if len(fp_scores) > 0:
        ax.hist(fp_scores, bins=bins, alpha=0.55, label="FP score", color="#4e79a7", density=True)
    if len(fn_scores) > 0:
        ax.hist(fn_scores, bins=bins, alpha=0.55, label="FN score", color="#e15759", density=True)

    ax.axvline(x=balanced_threshold, color="black", linestyle="--", linewidth=1.4, label="Balanced threshold")
    ax.axvline(x=high_recall_threshold, color="#f28e2b", linestyle="--", linewidth=1.4, label="High recall threshold")

    ax.set_title("Confidence Score Distribution (TP/FP/FN)")
    ax.set_xlabel("Decision score")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def run_ablation_study(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_csv_path: str,
) -> pd.DataFrame:
    base_text_train = train_df["sentence"].astype(str)
    base_text_test = test_df["sentence"].astype(str)
    y_train = train_df["label"].values
    y_test = test_df["label"].values

    configs = [
        ("Ablation ngram unigram", {"ngram": (1, 1), "lowercase": True, "class_weight": "balanced"}),
        ("Ablation ngram uni+bi", {"ngram": (1, 2), "lowercase": True, "class_weight": "balanced"}),
        ("Ablation add char 3-5", {"ngram": (1, 2), "lowercase": True, "class_weight": "balanced", "char": True}),
        ("Ablation lowercase on", {"ngram": (1, 2), "lowercase": True, "class_weight": "balanced"}),
        ("Ablation preserve case", {"ngram": (1, 2), "lowercase": False, "class_weight": "balanced"}),
        ("Ablation class weight none", {"ngram": (1, 2), "lowercase": True, "class_weight": None}),
        ("Ablation class weight balanced", {"ngram": (1, 2), "lowercase": True, "class_weight": "balanced"}),
    ]

    rows = []
    for label, cfg in configs:
        x_train = base_text_train.apply(lambda s: clean_text(s, lowercase=cfg["lowercase"]))
        x_test = base_text_test.apply(lambda s: clean_text(s, lowercase=cfg["lowercase"]))

        pipeline = build_pipeline(
            ngram_range=cfg["ngram"],
            class_weight=cfg["class_weight"],
            use_char_ngrams=cfg.get("char", False),
        )
        pipeline.fit(x_train, y_train)

        scores = pipeline.decision_function(x_test)
        y_pred = (scores >= 0.0).astype(int)

        rows.append(
            {
                "config": label,
                "ngram_range": str(cfg["ngram"]),
                "lowercase": cfg["lowercase"],
                "class_weight": str(cfg["class_weight"]),
                "use_char_ngrams": bool(cfg.get("char", False)),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1_action": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
                "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
            }
        )

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_csv_path, index=False)
    return result_df


def save_confusion_matrix_plot(cm: np.ndarray, output_path: str, title: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=["Pred Info", "Pred Action"],
        yticklabels=["True Info", "True Action"],
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    threshold = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_interpretation(results: List[EvalResult], output_txt_path: str):
    by_name = {r.mode_name: r for r in results}
    balanced = by_name.get("Balanced mode")
    high_recall = by_name.get("High recall mode")

    lines = []
    lines.append("Threshold Tradeoff Interpretation")
    lines.append("================================")
    if balanced and high_recall:
        fn_reduction = balanced.fn - high_recall.fn
        fp_increase = high_recall.fp - balanced.fp
        lines.append(
            f"Using high recall mode reduced missed actions (FN) by {fn_reduction} "
            f"but increased extra false alarms (FP) by {fp_increase}."
        )
        lines.append(
            f"Balanced mode: Precision={balanced.precision:.4f}, Recall={balanced.recall:.4f}, "
            f"FN={balanced.fn}, FP={balanced.fp}."
        )
        lines.append(
            f"High recall mode: Precision={high_recall.precision:.4f}, Recall={high_recall.recall:.4f}, "
            f"FN={high_recall.fn}, FP={high_recall.fp}."
        )
        lines.append(
            "Thesis framing: If missing an action item is more costly than reviewing extra alarms, "
            "high recall mode is preferred for deployment."
        )

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate final thesis evaluation artifacts")
    parser.add_argument(
        "--model-mode",
        type=str,
        choices=["runtime", "thesis"],
        default="runtime",
        help="runtime = score the deployed svm_model.pkl; thesis = train/evaluate TF-IDF model",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "svm_model.pkl"),
        help="Path to the deployed pickled model when using runtime mode",
    )
    parser.add_argument(
        "--holdout-keyword",
        type=str,
        default="ami",
        help="Keyword in source filename to hold out for external validity test",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.95,
        help="Target recall for high recall mode threshold selection",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output folder for all artifacts",
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=0,
        help="Optional cap for quick smoke runs (0 disables cap)",
    )
    parser.add_argument(
        "--hard-negatives-file",
        type=str,
        default="",
        help="Optional CSV of manually reviewed hard negatives to append to training data",
    )
    parser.add_argument(
        "--export-top-fp",
        type=int,
        default=300,
        help="How many high-confidence false positives to export for hard-negative mining",
    )
    parser.add_argument(
        "--export-top-fn",
        type=int,
        default=120,
        help="How many false negatives to export for manual taxonomy annotation",
    )
    parser.add_argument(
        "--use-char-ngrams",
        action="store_true",
        help="Use combined word n-grams and char n-grams (3-5) for robustness",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading labeled corpus")
    df = load_labeled_corpus(max_rows_per_file=args.max_rows_per_file)
    df = apply_data_hygiene(df, args.output_dir)

    if args.hard_negatives_file:
        print(f"Loading hard negatives from: {args.hard_negatives_file}")
        hard_df = load_hard_negative_rows(args.hard_negatives_file)
        df = pd.concat([df, hard_df], ignore_index=True)
        print(f"Added hard negatives: {len(hard_df)}")

    print("Applying external validity split")
    train_df, test_df = split_external_validity(df, args.holdout_keyword)

    print(f"Train rows: {len(train_df)}")
    print(f"Held-out rows: {len(test_df)}")
    print("Held-out sources:")
    for src in sorted(test_df["source_file"].unique()):
        print(f"- {src}")

    y_train = train_df["label"].values
    y_test = test_df["label"].values

    runtime_mode = args.model_mode == "runtime"
    x_train = get_scoring_texts(train_df, runtime_mode=runtime_mode)
    x_test = get_scoring_texts(test_df, runtime_mode=runtime_mode)

    if runtime_mode:
        print(f"Loading deployed pickle model: {args.model_path}")
        model = load_pickled_classifier(args.model_path)
        vectorizer = build_runtime_vectorizer()
        x_train_features = vectorizer.transform(x_train)
        x_test_features = vectorizer.transform(x_test)
        scores = model.decision_function(x_test_features)
        train_scores = model.decision_function(x_train_features)
        print("Scoring with runtime HashingVectorizer (n_features=2**16, ngram_range=(1, 3))")
    else:
        x_train = x_train.apply(lambda s: clean_text(s, lowercase=True))
        x_test = x_test.apply(lambda s: clean_text(s, lowercase=True))
        model = build_pipeline(
            ngram_range=(1, 2),
            class_weight="balanced",
            use_char_ngrams=args.use_char_ngrams,
        )
        model.fit(x_train, y_train)
        scores = model.decision_function(x_test)
        train_scores = model.decision_function(x_train)

    print(f"Evaluation mode: {args.model_mode}")

    balanced_threshold = 0.0
    high_recall_threshold = choose_high_recall_threshold(y_test, scores, target_recall=args.target_recall)

    balanced_result = evaluate_at_threshold(y_test, scores, balanced_threshold, "Balanced mode")
    high_recall_result = evaluate_at_threshold(y_test, scores, high_recall_threshold, "High recall mode")

    results_df = pd.DataFrame([
        vars(balanced_result),
        vars(high_recall_result),
    ])
    results_path = os.path.join(args.output_dir, "threshold_operating_points.csv")
    results_df.to_csv(results_path, index=False)

    y_pred_balanced = (scores >= balanced_threshold).astype(int)
    y_pred_high_recall = (scores >= high_recall_threshold).astype(int)

    taxonomy_df = build_error_taxonomy(y_test, y_pred_high_recall, x_test)
    taxonomy_path = os.path.join(args.output_dir, "error_taxonomy_false_negatives.csv")
    taxonomy_df.to_csv(taxonomy_path, index=False)

    export_hard_negative_candidates(
        test_df,
        y_test,
        y_pred_high_recall,
        scores,
        os.path.join(args.output_dir, "hard_negative_candidates_high_recall.csv"),
        top_k=args.export_top_fp,
    )
    export_fn_manual_annotation_template(
        test_df,
        y_test,
        y_pred_high_recall,
        scores,
        os.path.join(args.output_dir, "false_negative_manual_annotation_template.csv"),
        top_k=args.export_top_fn,
    )

    cm_balanced = confusion_matrix(y_test, y_pred_balanced)
    cm_high_recall = confusion_matrix(y_test, y_pred_high_recall)
    save_confusion_matrix_plot(
        cm_balanced,
        os.path.join(args.output_dir, "confusion_matrix_balanced_mode.png"),
        "Confusion Matrix (Balanced mode)",
    )
    save_confusion_matrix_plot(
        cm_high_recall,
        os.path.join(args.output_dir, "confusion_matrix_high_recall_mode.png"),
        "Confusion Matrix (High recall mode)",
    )

    save_threshold_tradeoff_figure(
        [balanced_result, high_recall_result],
        os.path.join(args.output_dir, "threshold_tradeoff_fn_vs_fp.png"),
    )

    pr_auc = save_precision_recall_plot(
        y_test,
        scores,
        os.path.join(args.output_dir, "precision_recall_curve.png"),
    )

    save_score_distribution_plot(
        y_test,
        y_pred_balanced,
        scores,
        balanced_threshold,
        high_recall_threshold,
        os.path.join(args.output_dir, "confidence_score_distribution_tp_fp_fn.png"),
    )

    ablation_df = run_ablation_study(
        train_df,
        test_df,
        os.path.join(args.output_dir, "ablation_study_results.csv"),
    )

    write_interpretation(
        [balanced_result, high_recall_result],
        os.path.join(args.output_dir, "threshold_tradeoff_interpretation.txt"),
    )

    print("\nDone. Artifacts generated:")
    print(f"- {results_path}")
    print(f"- {taxonomy_path}")
    print(f"- {os.path.join(args.output_dir, 'precision_recall_curve.png')}")
    print(f"- PR AUC: {pr_auc:.4f}")
    print(f"- {os.path.join(args.output_dir, 'threshold_tradeoff_fn_vs_fp.png')}")
    print(f"- {os.path.join(args.output_dir, 'confidence_score_distribution_tp_fp_fn.png')}")
    print(f"- {os.path.join(args.output_dir, 'ablation_study_results.csv')}")
    print(f"- {os.path.join(args.output_dir, 'hard_negative_candidates_high_recall.csv')}")
    print(f"- {os.path.join(args.output_dir, 'false_negative_manual_annotation_template.csv')}")
    print(f"- {os.path.join(args.output_dir, 'data_hygiene_summary.csv')}")
    print("\nAblation snapshot:")
    print(ablation_df[["config", "precision", "recall", "f1_action", "f1_macro"]].to_string(index=False))


if __name__ == "__main__":
    main()
