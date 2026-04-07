"""
Phase 3: Action Item Classifier Module
Handles SVM model loading, feature engineering, and action item prediction.
"""

import os
import pickle
import spacy
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer


class ActionItemClassifier:
    """Classifies text as action items or information items using SVM."""

    def __init__(self, model_path="svm_model.pkl", vectorizer_n_features=2**16):
        """
        Initialize the classifier.

        Args:
            model_path (str): Path to save/load the SVM model
            vectorizer_n_features (int): Number of features for the vectorizer
        """
        self.model_path = model_path
        self.nlp = spacy.load("en_core_web_sm")

        # High-capacity vectorizer to handle 150k+ rows and Taglish prefixes
        self.vectorizer = HashingVectorizer(
            n_features=vectorizer_n_features,
            ngram_range=(1, 3),
            alternate_sign=False,
        )

        self.clf = self._load_or_init_model()

    def _load_or_init_model(self):
        """Load existing model or initialize a new one."""
        if os.path.exists(self.model_path):
            print(f"[System] Loading existing model from {self.model_path}...")
            with open(self.model_path, "rb") as f:
                return pickle.load(f)
        else:
            print("[System] Initializing new SVM model...")
            model = SGDClassifier(loss="hinge")
            # Initial "seed" data to establish classes [0, 1]
            X_initial = self.vectorizer.transform(
                ["info", "please do this"]
            )
            y_initial = [0, 1]
            model.partial_fit(X_initial, y_initial, classes=[0, 1])
            return model

    def get_features(self, sentence):
        """
        Get vectorized representation of text.

        Args:
            sentence (str): Input text

        Returns:
            sparse matrix: Vectorized representation
        """
        return self.vectorizer.transform([sentence])

    def predict(self, sentence):
        """
        Predict if a sentence is an action item (1) or information (0).

        Args:
            sentence (str): Input sentence

        Returns:
            int: 0 for information, 1 for action item
        """
        features = self.get_features(sentence)
        return self.clf.predict(features)[0]

    def classify_segment(self, segment):
        """
        Classify all sentences in a segment.

        Args:
            segment (list): List of sentences

        Returns:
            dict: Classification results
        """
        classified_sentences = []
        detected_actions = []

        for sent in segment:
            prediction = self.predict(sent)
            status = "[!] Action" if prediction == 1 else "[ ] Info"
            print(f"  {status}: {sent}")

            classified_sentences.append({"sentence": sent, "label": prediction})

            if prediction == 1:
                detected_actions.append(sent)

        return {
            "classified_sentences": classified_sentences,
            "detected_actions": detected_actions,
        }

    def train_on_batch(self, texts, labels):
        """
        Incrementally train the model on a batch of data.

        Args:
            texts (list): List of text samples
            labels (list): List of labels (0 or 1)
        """
        X = self.vectorizer.transform(texts)
        self.clf.partial_fit(X, labels, classes=[0, 1])

    def save_model(self):
        """Save the current model to disk."""
        with open(self.model_path, "wb") as f:
            pickle.dump(self.clf, f)
        print(f"[System] Model saved to {self.model_path}")

    def apply_correction(self, sentence, correct_label):
        """
        Apply user correction for a single sentence (self-training).

        Args:
            sentence (str): The sentence to correct
            correct_label (int): Corrected label (0 or 1)
        """
        features = self.get_features(sentence)
        self.clf.partial_fit(features, [correct_label], classes=[0, 1])

    def get_model_info(self):
        """Get information about the current model."""
        return {
            "model_path": self.model_path,
            "model_classes": self.clf.classes_.tolist() if hasattr(self.clf, 'classes_') else None,
            "vectorizer_features": self.vectorizer.n_features,
        }
