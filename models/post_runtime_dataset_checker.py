import argparse
import os
import pickle
from typing import Dict

import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import accuracy_score, classification_report


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Main.py currently saves to project root. Keep both locations for compatibility.
MODEL_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "svm_model.pkl"),
    os.path.join(SCRIPT_DIR, "svm_model.pkl"),
]

DATASET_MAP: Dict[str, str] = {
    "10k": os.path.join(DATA_DIR, "action_items_dataset_10k.csv"),
    "20k": os.path.join(DATA_DIR, "action_items_dataset_20k_taglish.csv"),
    "svm": os.path.join(DATA_DIR, "action_items_dataset_svm.csv"),
}

vectorizer = HashingVectorizer(n_features=2**10)


def resolve_model_path() -> str:
    for path in MODEL_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "svm_model.pkl not found. Run Main.py first, then run this checker."
    )


def normalize_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip() for c in df.columns]

    if "text" in df.columns and "sentence" not in df.columns:
        df = df.rename(columns={"text": "sentence"})

    if "sentence" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"{csv_path} must contain sentence/text and label columns."
        )

    label_map = {
        "information_item": 0,
        "action_item": 1,
        "0": 0,
        "1": 1,
    }

    rows = df.dropna(subset=["sentence", "label"]).copy()
    labels = []
    for raw in rows["label"].astype(str):
        key = raw.strip().lower()
        if key not in label_map:
            raise ValueError(f"Unsupported label value '{raw}' in {csv_path}")
        labels.append(label_map[key])

    return pd.DataFrame({
        "sentence": rows["sentence"].astype(str),
        "label": labels,
    })


def evaluate_dataset(clf, dataset_name: str, csv_path: str):
    if not os.path.exists(csv_path):
        print(f"[Skip] {dataset_name}: file not found -> {csv_path}")
        return

    df = normalize_dataset(csv_path)
    X = vectorizer.transform(df["sentence"])
    y_true = df["label"]
    y_pred = clf.predict(X)

    acc = accuracy_score(y_true, y_pred)

    print("\n" + "=" * 70)
    print(f"POST-RUNTIME DATASET CHECKER: {dataset_name}")
    print("=" * 70)
    print(f"Dataset Path: {csv_path}")
    print(f"Samples: {len(df)}")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_true, y_pred, target_names=["Information", "Action Item"]))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the latest SVM model (after Main.py runtime) against dataset(s)."
    )
    parser.add_argument(
        "--dataset",
        choices=["10k", "20k", "svm", "all"],
        default="all",
        help="Choose one dataset or all datasets.",
    )

    args = parser.parse_args()

    model_path = resolve_model_path()
    print(f"[System] Using model: {model_path}")

    with open(model_path, "rb") as f:
        clf = pickle.load(f)

    selected = DATASET_MAP.items() if args.dataset == "all" else [(args.dataset, DATASET_MAP[args.dataset])]
    for dataset_name, csv_path in selected:
        evaluate_dataset(clf, dataset_name, csv_path)


if __name__ == "__main__":
    main()
