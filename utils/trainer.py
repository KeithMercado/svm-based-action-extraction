"""
Model Trainer Module
Handles CSV dataset loading, incremental training, and self-training feedback.
"""

import os
import pandas as pd


class ModelTrainer:
    """Manages training data loading, model training, and dataset management."""

    def __init__(self, classifier):
        """
        Initialize the trainer.

        Args:
            classifier: ActionItemClassifier instance
        """
        self.classifier = classifier

    def train_from_csv(self, file_paths):
        """
        Train the SVM model from CSV datasets.

        Args:
            file_paths (list): List of CSV file paths to train on
        """
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
                texts = df[text_col].astype(str).tolist()  # Ensure everything is a string
                labels = [
                    1 if "action" in str(l).lower() else 0
                    for l in df["label"].tolist()
                ]

                # Train
                self.classifier.train_on_batch(texts, labels)
                print(f"  - Successfully processed {len(texts)} rows.")
            else:
                print(f"[Error] File not found: {path}")

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
