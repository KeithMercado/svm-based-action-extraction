"""
Phase 3: Action Item Classifier Module
Handles SVM model loading, feature engineering, Model-Lllama auditing, and action item prediction.
"""

import json
import math
import os
import pickle
import re

import spacy
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    Groq = None
    GROQ_AVAILABLE = False


class ActionItemClassifier:
    """Classifies text as action items or information items using an incremental SVM."""

    # Operating mode constants for threshold-based inference
    MODE_BALANCED = "balanced"
    MODE_HIGH_RECALL = "high_recall"

    def __init__(
        self,
        model_path="svm_model.pkl",
        vectorizer_n_features=2**16,
        confidence_threshold=0.70,
        corrections_csv_path="user_corrections.csv",
        model_llama_name=None,
    ):
        """
        Initialize the classifier.

        Args:
            model_path (str): Path to save/load the SVM model
            vectorizer_n_features (int): Number of features for the vectorizer
            confidence_threshold (float): Student confidence threshold that triggers audit
            corrections_csv_path (str): CSV path for persisted corrections
            model_llama_name (str | None): Groq model used as Model-Lllama
        """
        self.model_path = model_path
        self.corrections_csv_path = corrections_csv_path
        self.confidence_threshold = confidence_threshold
        self.model_llama_name = model_llama_name or os.getenv(
            "GROQ_MODEL_LLAMA", "llama-3.1-8b-instant"
        )
        self.model_llama_available = GROQ_AVAILABLE
        self.nlp = spacy.load("en_core_web_sm")
        self.information_override_patterns = [
            r"^thank you(?: so much)?$",
            r"^thanks(?: everyone| all)?$",
            r"^appreciate it$",
            r"^noted$",
            r"^received(?: with thanks)?$",
            r"^copy that$",
            r"^got it$",
            r"^roger that$",
            r"^understood$",
            r"^acknowledged$",
            r"^good (?:morning|afternoon|evening)(?: everyone)?$",
            r"^welcome(?: everyone)?$",
            r"^fyi$",
            r"^for your information$",
            r"^no further updates from the team$",
            r"^nothing else to add$",
        ]
        self.mode_thresholds = {
            self.MODE_BALANCED: 0.0,
            self.MODE_HIGH_RECALL: -0.12,
        }
        self.operating_mode = self.MODE_HIGH_RECALL

        self.vectorizer = HashingVectorizer(
            n_features=vectorizer_n_features,
            ngram_range=(1, 3),
            alternate_sign=False,
        )

        self.clf = self._load_or_init_model()

    def _load_dotenv(self):
        """Load environment variables if python-dotenv is available."""
        if load_dotenv is not None:
            load_dotenv()

    def _load_or_init_model(self):
        """Load existing model or initialize a new one."""
        if os.path.exists(self.model_path):
            print(f"[System] Loading existing model from {self.model_path}...")
            with open(self.model_path, "rb") as handle:
                return pickle.load(handle)

        print("[System] Initializing new SVM model...")
        model = SGDClassifier(loss="hinge")
        X_initial = self.vectorizer.transform(["info", "please do this"])
        y_initial = [0, 1]
        model.partial_fit(X_initial, y_initial, classes=[0, 1])
        return model

    def _sigmoid(self, value):
        """Numerically stable sigmoid helper."""
        if value >= 0:
            z = math.exp(-value)
            return 1 / (1 + z)
        z = math.exp(value)
        return z / (1 + z)

    def _decision_to_confidence(self, features):
        """Convert the model margin into a confidence-like score in [0, 1]."""
        margin = self.clf.decision_function(features)
        if hasattr(margin, "__len__"):
            margin_value = float(margin[0])
        else:
            margin_value = float(margin)

        prediction = 1 if margin_value >= 0 else 0
        confidence = self._sigmoid(abs(margin_value))
        return prediction, confidence, margin_value

    def set_operating_mode(self, mode):
        """Set the classification operating mode used for inference decisions."""
        normalized = str(mode).strip().lower()
        if normalized not in self.mode_thresholds:
            raise ValueError(
                f"Invalid operating mode: {mode}. Expected one of: {', '.join(self.mode_thresholds.keys())}"
            )
        self.operating_mode = normalized

    def get_operating_threshold(self):
        """Return the active decision threshold for the current operating mode."""
        return float(self.mode_thresholds.get(self.operating_mode, 0.0))

    def _normalize_label(self, label_text):
        """Normalize label text into 0/1."""
        if label_text is None:
            return None

        normalized = str(label_text).strip().lower()
        if normalized in {"1", "action", "action_item", "action item", "task", "todo"}:
            return 1
        if normalized in {"0", "info", "information", "information_item", "information item"}:
            return 0
        return None

    def _extract_json_object(self, text):
        """Extract a JSON object from a Groq response that may include code fences."""
        if not text:
            return None

        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`").strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].lstrip()

        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None

        candidate = stripped[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return None

    def _split_sentences(self, raw_text):
        """Split transcript into coherent sentence-like units without fragmenting clauses."""
        if not raw_text:
            return []

        normalized = re.sub(r"\s+", " ", str(raw_text)).strip()
        if not normalized:
            return []

        # Prefer spaCy boundaries; they are safer than raw period splitting.
        doc = self.nlp(normalized)
        candidates = [sent.text.strip(" \t\r\n.,;:") for sent in doc.sents if sent.text.strip()]
        if not candidates:
            candidates = [s.strip(" \t\r\n.,;:") for s in re.split(r"[.!?]+", normalized) if s.strip()]

        merged = []
        dangling_starts = (
            "to ",
            "and ",
            "but ",
            "or ",
            "so ",
            "because ",
            "that ",
            "which ",
            "with ",
            "for ",
            "as ",
            "if ",
        )

        for chunk in candidates:
            sentence = chunk.strip()
            if not sentence:
                continue

            lower_sentence = sentence.lower()
            if merged:
                previous = merged[-1]
                previous_lower = previous.lower()

                # Re-attach obvious tails caused by abbreviations/time notation (e.g., "2 p." + "to help...").
                previous_ends_abbrev = bool(re.search(r"\b(?:[ap]|[ap]m|mr|mrs|ms|dr|prof)\.?$", previous_lower))
                previous_time_stub = bool(re.search(r"\b\d{1,2}\s*[ap]$", previous_lower))
                looks_dangling = lower_sentence.startswith(dangling_starts)
                starts_lower = sentence[:1].islower()
                very_short_tail = len(sentence.split()) <= 4

                if previous_ends_abbrev or previous_time_stub or looks_dangling or (starts_lower and very_short_tail):
                    merged[-1] = f"{previous} {sentence}".strip()
                    continue

            merged.append(sentence)

        return [sentence for sentence in merged if len(sentence) > 5]

    def _looks_like_specific_task(self, sentence):
        """Detect whether a sentence is a concrete assigned task with ownership or timing."""
        lower = sentence.lower().strip()

        task_markers = [
            r"\bplease\b",
            r"\bneed(s)? to\b",
            r"\bshould\b",
            r"\bmust\b",
            r"\bassign(ed|ment)?\b",
            r"\bwill\b",
            r"\bcan you\b",
            r"\bcould you\b",
            r"\bby\s+(?:tomorrow|today|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{1,2}(?:[:.]\d{2})?\s*(?:am|pm)?|end of day|eod|next week|next month)\b",
            r"\bdeadline\b",
            r"\bfollow up\b",
            r"\baction item\b",
            r"\bowner\b",
            r"\bresponsible\b",
            r"\btask\b",
            r"\bdeliverable\b",
            r"\bsubmit\b",
            r"\bsend\b",
            r"\bprepare\b",
            r"\breview\b",
            r"\bcomplete\b",
            r"\bfinalize\b",
            r"\bupdate\b",
            r"\bcoordinate\b",
        ]

        ownership_markers = [
            r"\bfor (?:you|us|him|her|them|the team|the group)\b",
            r"\bassigned to\b",
            r"\byou will\b",
            r"\byou need to\b",
            r"\blet's\b",
            r"\bwe need to\b",
            r"\bthis needs to\b",
        ]

        has_task_marker = any(re.search(pattern, lower) for pattern in task_markers)
        has_owner_marker = any(re.search(pattern, lower) for pattern in ownership_markers)
        has_deadline = bool(
            re.search(
                r"\b(by|before|within|due|until)\b.*\b(tomorrow|today|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{1,2}(?:[:.]\d{2})?\s*(?:am|pm)?|eod|end of day|next week|next month)\b",
                lower,
            )
        )

        return has_task_marker or has_owner_marker or has_deadline

    def _format_segment_context(self, segment_context):
        """Serialize segment metadata into a compact context block."""
        if not segment_context:
            return ""

        if isinstance(segment_context, str):
            return segment_context.strip()

        lines = []
        for index, segment in enumerate(segment_context, 1):
            if isinstance(segment, dict):
                gist = segment.get("topical_description") or segment.get("gist") or ""
                raw_text = segment.get("raw_text") or segment.get("text") or ""
                token_count = segment.get("token_count")
                prefix = f"Segment {index}"
                if token_count is not None:
                    prefix += f" ({token_count} tokens)"
                entry = f"{prefix}: {gist}".strip()
                if raw_text:
                    entry = f"{entry}\n{raw_text[:1000].strip()}".strip()
                lines.append(entry)
            else:
                lines.append(str(segment).strip())

        return "\n\n".join(line for line in lines if line)

    def _model_llama_verdict(self, sentence, transcript_context, segment_context=None):
        """Ask Groq Llama to verify a single sentence using transcript and segment context."""
        if not self.model_llama_available:
            return {
                "available": False,
                "label": None,
                "confidence": None,
                "reason": "Groq Model-Lllama is unavailable.",
                "raw": None,
            }

        self._load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return {
                "available": False,
                "label": None,
                "confidence": None,
                "reason": "GROQ_API_KEY not found.",
                "raw": None,
            }

        client = Groq(api_key=api_key)
        system_prompt = (
            "You are Model-Lllama, the transcript-level verifier in an active learning pipeline. "
            "Use the full meeting transcript context, not isolated segments, to classify each Taglish sentence. "
            "If topical segment gists are provided, use them as supporting context for the conversation topic. "
            "Only label action_item when the sentence is a concrete assigned task, request, follow-up, or deliverable that requires completion to move the project forward. "
            "The sentence should usually include ownership, responsibility, a recipient, or a timeframe/deadline. "
            "Do not label generic action words, verbs, or vague mentions of activity as action_item. "
            "Examples that are NOT action_item: 'We discussed the plan', 'We will improve the process', 'They talked about sending updates later'. "
            "Return strict JSON only with keys label, confidence, and reason. "
            "Use action_item only for specific assigned tasks, concrete follow-ups, and deadline-bound work. "
            "Use information_item for status updates, explanations, facts, and general discussion."
        )
        segment_context_block = self._format_segment_context(segment_context)
        user_prompt = "Full transcript context:\n"
        user_prompt += f"{transcript_context[:12000]}\n\n"
        if segment_context_block:
            user_prompt += f"Segment context:\n{segment_context_block[:6000]}\n\n"
        user_prompt += (
            "Sentence to verify:\n"
            f"{sentence}\n\n"
            'Return JSON only, for example: {"label":"action_item","confidence":0.93,"reason":"..."}'
        )

        result = client.chat.completions.create(
            model=self.model_llama_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_completion_tokens=220,
        )

        content = result.choices[0].message.content if result.choices else ""
        payload = self._extract_json_object(content)

        if payload is None:
            lowered = content.lower()
            if "action_item" in lowered:
                label = 1
            elif "information_item" in lowered:
                label = 0
            else:
                return {
                    "available": True,
                    "label": None,
                    "confidence": None,
                    "reason": content.strip(),
                    "raw": content,
                }

            return {
                "available": True,
                "label": label,
                "confidence": None,
                "reason": content.strip(),
                "raw": content,
            }

        label = self._normalize_label(payload.get("label"))
        confidence = payload.get("confidence")
        try:
            confidence = float(confidence) if confidence is not None else None
        except Exception:
            confidence = None

        return {
            "available": True,
            "label": label,
            "confidence": confidence,
            "reason": str(payload.get("reason", "")).strip(),
            "raw": payload,
        }

    def _enforce_action_item_specificity(self, sentence, model_llama_label, model_llama_reason):
        """
        Ensure action_item labels are reserved for specific tasks with ownership or timing.

        If Model-Lllama marks a sentence as action_item but the sentence does not look like a real
        assigned task, downgrade it to information_item to avoid generic action words.
        """
        if int(model_llama_label) != 1:
            return int(model_llama_label)

        if self._looks_like_specific_task(sentence):
            return 1

        reason_text = (model_llama_reason or "").lower()
        specificity_hints = ["assigned", "deadline", "due", "owner", "responsible", "deliverable", "follow-up", "task"]
        if any(hint in reason_text for hint in specificity_hints):
            return 1

        return 0

    def _append_corrections_to_csv(self, corrections):
        """Append a batch of corrections to the persistent CSV log."""
        if not corrections:
            return

        import pandas as pd

        df_new = pd.DataFrame(corrections)
        file_exists = os.path.exists(self.corrections_csv_path)
        df_new.to_csv(
            self.corrections_csv_path,
            mode="a",
            index=False,
            header=not file_exists,
        )

    def _teach_student(self, sentence, correct_label):
        """Incrementally train the student model on one confirmed label."""
        features = self.get_features(sentence)
        self.clf.partial_fit(features, [int(correct_label)], classes=[0, 1])

    def save_model(self):
        """Save the current model to disk."""
        with open(self.model_path, "wb") as handle:
            pickle.dump(self.clf, handle)
        print(f"[System] Model saved to {self.model_path}")

    def get_features(self, sentence):
        """Return the vectorized representation of a sentence."""
        return self.vectorizer.transform([sentence])

    def _looks_like_information_override(self, sentence):
        """Force common acknowledgements and courtesy phrases into the information class."""
        normalized = re.sub(r"\s+", " ", str(sentence or "")).strip().lower().rstrip(".?!,")
        if not normalized:
            return False

        if self._looks_like_specific_task(normalized):
            return False

        if len(normalized.split()) > 12:
            return False

        return any(re.fullmatch(pattern, normalized) for pattern in self.information_override_patterns)

    def predict_with_confidence(self, sentence):
        """Return the student prediction, confidence, and raw score."""
        if self._looks_like_information_override(sentence):
            return {
                "label": 0,
                "confidence": 1.0,
                "score": -1.0,
                "threshold": float(self.get_operating_threshold()),
                "operating_mode": self.operating_mode,
            }

        features = self.get_features(sentence)
        _, confidence, raw_score = self._decision_to_confidence(features)
        threshold = self.get_operating_threshold()
        prediction = 1 if raw_score >= threshold else 0
        return {
            "label": int(prediction),
            "confidence": float(confidence),
            "score": float(raw_score),
            "threshold": float(threshold),
            "operating_mode": self.operating_mode,
        }

    def predict(self, sentence):
        """Predict whether a sentence is an action item or information."""
        return self.predict_with_confidence(sentence)["label"]

    def audit_sentence(self, sentence, transcript_context, persist=False, segment_context=None):
        """
        Audit a single sentence against Model-Lllama using the full transcript context.

        The student triggers audit when the confidence is low or when it predicts action_item.
        """
        student = self.predict_with_confidence(sentence)
        needs_audit = student["confidence"] < self.confidence_threshold

        model_llama = None
        final_label = student["label"]
        label_source = "student"

        if needs_audit:
            model_llama = self._model_llama_verdict(
                sentence,
                transcript_context,
                segment_context=segment_context,
            )
            if model_llama.get("available") and model_llama.get("label") is not None:
                final_label = self._enforce_action_item_specificity(
                    sentence,
                    model_llama["label"],
                    model_llama.get("reason"),
                )
                label_source = "model_llama"

        correction = None
        if final_label != student["label"]:
            correction = {
                "text": sentence,
                "label": "action_item" if final_label == 1 else "information_item",
            }

        result = {
            "sentence": sentence,
            "student_label": int(student["label"]),
            "student_confidence": float(student["confidence"]),
            "student_score": float(student["score"]),
            "audited": bool(needs_audit),
            "model_llama": model_llama,
            "model_llama_label": model_llama.get("label") if model_llama else None,
            "final_label": int(final_label),
            "label_source": label_source,
            "correction": correction,
        }

        if persist and correction is not None:
            self.apply_batch_corrections([correction], persist_csv=True, persist_model=True)

        return result

    def audit_transcript(self, raw_text, persist=False, segment_context=None):
        """
        Audit a whole transcription using the full context for every sentence.

        Returns a transcript-level report and a correction queue that can be applied once.
        """
        sentences = self._split_sentences(raw_text)
        evaluations = []
        corrections = []
        action_items = []
        segment_context_block = self._format_segment_context(segment_context)

        for sentence in sentences:
            result = self.audit_sentence(
                sentence,
                transcript_context=raw_text,
                segment_context=segment_context_block,
                persist=False,
            )
            evaluations.append(result)

            if result["final_label"] == 1:
                action_items.append(sentence)

            if result["correction"] is not None:
                corrections.append(result["correction"])

            status = "[!] Action" if result["final_label"] == 1 else "[ ] Info"
            model_llama_tag = result["label_source"]
            print(
                f"  {status} (student_conf={result['student_confidence']:.2f}, source={model_llama_tag}): {sentence}"
            )
            if result["model_llama_label"] is not None:
                print(f"    -> Model-Lllama label: {result['model_llama_label']}")

        return {
            "transcript": raw_text,
            "sentences": evaluations,
            "corrections": corrections,
            "action_items": action_items,
        }

    def _audit_segment(self, segment_text, segment_label):
        """
        Audit a single segment using Groq Llama.

        This is the fast path: one Llama call per segment instead of one per sentence.
        """
        if not self.model_llama_available:
            return {"label": None, "confidence": None, "reason": "Groq unavailable"}

        self._load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return {"label": None, "confidence": None, "reason": "GROQ_API_KEY not found"}

        try:
            client = Groq(api_key=api_key)
            system_prompt = (
                "You are Model-Lllama, a meeting segment classifier. "
                "Decide whether the segment contains action items that require completion. "
                "Return strict JSON only with keys label, confidence, and reason."
            )
            user_prompt = (
                f"Segment topic: {segment_label}\n\n"
                f"Segment text:\n{segment_text[:12000]}\n\n"
                'Return JSON only, for example: {"label":"action_item","confidence":0.85,"reason":"..."}'
            )

            result = client.chat.completions.create(
                model=self.model_llama_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_completion_tokens=100,
            )

            content = result.choices[0].message.content if result.choices else ""
            payload = self._extract_json_object(content)

            if payload is None:
                lowered = content.lower()
                label = 1 if "action_item" in lowered else 0
                return {"label": label, "confidence": None, "reason": content.strip()}

            label = self._normalize_label(payload.get("label"))
            confidence = payload.get("confidence")
            try:
                confidence = float(confidence) if confidence is not None else None
            except Exception:
                confidence = None

            return {
                "label": label,
                "confidence": confidence,
                "reason": str(payload.get("reason", "")).strip(),
            }
        except Exception as e:
            return {"label": None, "confidence": None, "reason": str(e)}

    def audit_segments_batch(self, raw_text, segment_metadata, persist=False):
        """
        Fast segment-level audit that still respects the student confidence threshold.

        One Groq call is made per segment. Sentences are still scored individually by the
        student model, and Llama only overrides the student when confidence is below threshold.
        """
        if not segment_metadata:
            return self.audit_transcript(raw_text, persist=persist)

        sentences = self._split_sentences(raw_text)
        sentence_segments = {}
        segment_labels = {}

        for segment in segment_metadata:
            segment_id = segment.get("segment_id")
            segment_text = segment.get("raw_text", "")
            segment_description = segment.get("topical_description", "")
            segment_label = segment.get("topic_label", f"Topic {segment_id}")

            segment_audit = self._audit_segment(segment_text, f"{segment_label}: {segment_description}")
            segment_labels[segment_id] = segment_audit

            for sentence in self._split_sentences(segment_text):
                sentence_segments[sentence.lower()[:60]] = segment_id

        evaluations = []
        corrections = []
        action_items = []

        for sentence in sentences:
            student = self.predict_with_confidence(sentence)
            needs_audit = student["confidence"] < self.confidence_threshold

            segment_id = sentence_segments.get(sentence.lower()[:60], 1)
            segment_audit = segment_labels.get(segment_id, {})

            model_llama = None
            final_label = student["label"]
            label_source = "student"

            if needs_audit and segment_audit.get("label") is not None:
                model_llama = segment_audit
                final_label = self._enforce_action_item_specificity(
                    sentence,
                    segment_audit["label"],
                    segment_audit.get("reason"),
                )
                label_source = "segment_llama"

            correction = None
            if final_label != student["label"]:
                correction = {
                    "text": sentence,
                    "label": "action_item" if final_label == 1 else "information_item",
                }

            result = {
                "sentence": sentence,
                "student_label": int(student["label"]),
                "student_confidence": float(student["confidence"]),
                "student_score": float(student["score"]),
                "audited": bool(needs_audit),
                "segment_id": segment_id,
                "model_llama": model_llama,
                "model_llama_label": model_llama.get("label") if model_llama else None,
                "final_label": int(final_label),
                "label_source": label_source,
                "correction": correction,
            }
            evaluations.append(result)

            if result["final_label"] == 1:
                action_items.append(sentence)

            if correction is not None:
                corrections.append(correction)

            status = "[!] Action" if result["final_label"] == 1 else "[ ] Info"
            print(
                f"  {status} (student_conf={student['confidence']:.2f}, source={label_source}, segment={segment_id}): {sentence}"
            )
            if model_llama is not None:
                print(f"    -> Segment Llama label: {model_llama.get('label')}")

        if persist and corrections:
            self.apply_batch_corrections(corrections, persist_csv=True, persist_model=True)

        return {
            "transcript": raw_text,
            "sentences": evaluations,
            "corrections": corrections,
            "action_items": action_items,
        }

    def apply_batch_corrections(self, corrections, persist_csv=True, persist_model=True):
        """
        Teach the student model using a batch of confirmed corrections.

        This is the only write path used after the single end-of-run confirmation.
        """
        if not corrections:
            return

        for correction in corrections:
            label = 1 if correction["label"] == "action_item" else 0
            self._teach_student(correction["text"], label)

        if persist_csv:
            self._append_corrections_to_csv(corrections)

        if persist_model:
            self.save_model()

    def train_on_batch(self, texts, labels):
        """Incrementally train the model on a batch of data."""
        X = self.vectorizer.transform(texts)
        self.clf.partial_fit(X, labels, classes=[0, 1])

    def apply_correction(self, sentence, correct_label):
        """Compatibility helper for a single correction."""
        correction = {
            "text": sentence,
            "label": "action_item" if int(correct_label) == 1 else "information_item",
        }
        self.apply_batch_corrections([correction], persist_csv=True, persist_model=True)

    def classify_segment(self, segment):
        """Classify a segment without Model-Lllama auditing for compatibility."""
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

    def get_model_info(self):
        """Get information about the current model."""
        return {
            "model_path": self.model_path,
            "model_classes": self.clf.classes_.tolist() if hasattr(self.clf, "classes_") else None,
            "vectorizer_features": self.vectorizer.n_features,
            "confidence_threshold": self.confidence_threshold,
            "operating_mode": self.operating_mode,
            "operating_threshold": self.get_operating_threshold(),
            "model_llama_name": self.model_llama_name,
        }
