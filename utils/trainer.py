"""
Model Trainer Module
Handles CSV dataset loading, incremental training, and self-training feedback.
"""

import argparse
import os
import sys
import pandas as pd
from sklearn.utils import shuffle


class ModelTrainer:
    """Manages training data loading, model training, and dataset management."""

    def __init__(self, classifier):
        """
        Initialize the trainer.

        Args:
            classifier: ActionItemClassifier instance
        """
        self.classifier = classifier

    def _is_synthetic_file(self, path):
        base = os.path.basename(path).lower()
        return base.startswith("hard_negative_information_items")

    def train_from_csv(self, file_paths, max_synthetic_ratio=0.25, random_seed=42):
        """
        Train the SVM model from CSV datasets.

        Args:
            file_paths (list): List of CSV file paths to train on
            max_synthetic_ratio (float): Maximum synthetic-to-real ratio in the final training mix
            random_seed (int): Random seed for reproducible sampling and shuffle
        """
        real_parts = []
        synthetic_parts = []

        for path in file_paths:
            if os.path.exists(path):
                print(f"[System] Training on {path}...")
                df = pd.read_csv(path)

                # --- ROBUST COLUMN CLEANING ---
                # Convert all column names to lowercase and remove spaces
                df.columns = [c.lower().strip() for c in df.columns]

                # Check if required columns exist after cleaning
                if "label" not in df.columns or ("text" not in df.columns and "sentence" not in df.columns):
                    print(f"[Error] {path} is missing 'text'/'sentence' or 'label' columns.")
                    print(f"Actual columns found: {df.columns.tolist()}")
                    continue  # Skip this file and move to the next

                # Clean data: Remove empty rows
                text_col = "text" if "text" in df.columns else "sentence"
                df = df.dropna(subset=[text_col, "label"])

                normalized = pd.DataFrame(
                    {
                        "text": df[text_col].astype(str),
                        "label": [1 if "action" in str(l).lower() else 0 for l in df["label"].tolist()],
                    }
                )

                if self._is_synthetic_file(path):
                    synthetic_parts.append(normalized)
                else:
                    real_parts.append(normalized)

                print(f"  - Successfully processed {len(normalized)} rows.")
            else:
                print(f"[Error] File not found: {path}")

        if not real_parts and not synthetic_parts:
            print("[Error] No valid training rows were loaded. Aborting training.")
            return

        real_df = pd.concat(real_parts, ignore_index=True) if real_parts else pd.DataFrame(columns=["text", "label"])
        synthetic_df = pd.concat(synthetic_parts, ignore_index=True) if synthetic_parts else pd.DataFrame(columns=["text", "label"])

        if len(real_df) > 0 and len(synthetic_df) > 0:
            max_synthetic_rows = int(len(real_df) * max_synthetic_ratio)
            if len(synthetic_df) > max_synthetic_rows:
                synthetic_df = synthetic_df.sample(n=max_synthetic_rows, random_state=random_seed)
                print(
                    f"[System] Synthetic cap applied: keeping {len(synthetic_df)} synthetic rows "
                    f"for {len(real_df)} real rows (ratio={max_synthetic_ratio:.2f})."
                )

        train_df = pd.concat([real_df, synthetic_df], ignore_index=True)
        train_df = shuffle(train_df, random_state=random_seed).reset_index(drop=True)

        texts = train_df["text"].tolist()
        labels = train_df["label"].tolist()
        self.classifier.train_on_batch(texts, labels)
        print(f"[System] Final training batch size: {len(train_df)} rows")
        print(
            f"[System] Class distribution: {train_df['label'].value_counts().to_dict()} | "
            f"real={len(real_df)}, synthetic={len(synthetic_df)}"
        )

        # Save the updated model
        self.classifier.save_model()
        print("[System] Training complete. Model saved.")

    def collect_user_corrections(self, correction_data):
        """
        Process user corrections from a review session.

        Args:
            correction_data (list): List of (sentence, prediction) tuples

        Returns:
            list: List of corrections made by user
        """
        new_corrections = []

        for sent, pred in correction_data:
            user_input = (
                input(f"AI marked as {pred}: '{sent}' -> Correct? (y/0/1): ")
                .lower()
                .strip()
            )

            if user_input != "y":
                try:
                    correct_label = int(user_input)
                    # Store for CSV logging
                    new_corrections.append(
                        {
                            "text": sent,
                            "label": (
                                "action_item"
                                if correct_label == 1
                                else "information_item"
                            ),
                        }
                    )
                except ValueError:
                    print(f"[Warning] Invalid input. Skipping correction for: {sent}")

        return new_corrections

    def save_corrections_to_csv(self, corrections, csv_path="user_corrections.csv"):
        """
        Save user corrections to CSV for future training.

        Args:
            corrections (list): List of correction dictionaries
            csv_path (str): Path to save corrections
        """
        if not corrections:
            return

        self.classifier.apply_batch_corrections(
            corrections,
            persist_csv=True,
            persist_model=True,
        )
        print(f"[System] Saved {len(corrections)} corrections to {csv_path}")
        print("[System] Model updated and saved.")

    def get_training_datasets(self):
        """
        Get list of default training datasets.

        Returns:
            list: List of dataset file paths
        """
        datasets = [
            "data/ultimate_diversity_dataset_50k.csv",
            "data/massive_diverse_dataset_50000.csv",
            "data/expanded_meeting_contexts_20k.csv",
            "data/meeting_specific_dataset_15k.csv",
            "data/comprehensive_thesis_dataset_12k.csv",
            "data/ami_multilingual_balanced.csv",
            "data/hard_negative_information_items.csv",
            "data/hard_negative_information_items_20000.csv",
        ]

        # Add user corrections if they exist
        if os.path.exists("user_corrections.csv"):
            datasets.append("user_corrections.csv")
            print("[System] Found 'user_corrections.csv'. Adding to training pool...")

        return datasets


def main():
    """CLI entry point for training without using Main.py."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from core.classifier import ActionItemClassifier

    parser = argparse.ArgumentParser(
        description="Train the SVM classifier from CSV datasets (standalone mode)."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="svm_model_exp_hardneg.pkl",
        help="Output PKL path for the trained model.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional list of dataset CSV paths. If omitted, default dataset pool is used.",
    )
    parser.add_argument(
        "--max-synthetic-ratio",
        type=float,
        default=0.25,
        help="Maximum synthetic-to-real ratio in training mix (default: 0.25).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling/shuffling (default: 42).",
    )

    args = parser.parse_args()

    classifier = ActionItemClassifier(model_path=args.model_path)
    trainer = ModelTrainer(classifier)
    file_paths = args.datasets if args.datasets else trainer.get_training_datasets()

    print("=" * 70)
    print("STANDALONE TRAINING (trainer.py)")
    print("=" * 70)
    print(f"[System] Output model path: {args.model_path}")
    print(f"[System] Datasets to process: {len(file_paths)}")

    trainer.train_from_csv(
        file_paths,
        max_synthetic_ratio=args.max_synthetic_ratio,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
