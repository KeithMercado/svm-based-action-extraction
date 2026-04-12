"""
Phase 2: Segmentation Module
Handles semantic cosine segmentation of transcribed meeting text.
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None

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


class Segmenter:
    """Segments transcribed text into semantic cosine-based topic segments."""

    def __init__(
        self,
        chunk_size=5,
        max_tokens=1024,
        topical_model=None,
        embedding_model_name="all-mpnet-base-v2",
    ):
        """
        Initialize the Segmenter.

        Args:
            chunk_size (int): Backward-compatible fallback size for unlabeled text
            max_tokens (int): Maximum token budget per segment
            topical_model (str | None): Groq model used for topical gist generation
            embedding_model_name (str): Sentence embedding model for cosine segmentation
        """
        self.chunk_size = chunk_size
        self.max_tokens = max_tokens
        self.topical_model = topical_model or os.getenv(
            "GROQ_TOPICAL_MODEL", "llama-3.1-8b-instant"
        )
        self.embedding_model_name = embedding_model_name
        self.embedding_model = None
        self.groq_available = GROQ_AVAILABLE
        self.last_segment_metadata = []

    def _load_dotenv(self):
        """Load environment variables if python-dotenv is available."""
        if load_dotenv is not None:
            load_dotenv()

    def token_counter(self, text):
        """Count tokens with a simple whitespace-based heuristic."""
        if not text:
            return 0
        return len(re.findall(r"\S+", text))

    def _split_sentence_units(self, raw_text):
        """Split text into sentence-like units used for semantic clustering."""
        candidate = raw_text.replace("so ", ". ")
        sentences = re.split(r"(?<=[.!?])\s+", candidate)
        return [sentence.strip().strip(".?!") for sentence in sentences if len(sentence.strip()) > 5]

    def _load_embedding_model(self):
        """Lazily load the sentence embedding model."""
        if self.embedding_model is not None:
            return self.embedding_model

        if SentenceTransformer is None:
            raise RuntimeError(
                "Missing dependency: sentence-transformers. Run: pip install sentence-transformers"
            )

        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        return self.embedding_model

    def _compute_similarity_series(self, sentences):
        """Compute cosine similarity between consecutive sentence embeddings."""
        if len(sentences) < 2:
            return []

        model = self._load_embedding_model()
        embeddings = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
        similarities = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
        return similarities.tolist()

    def _build_segment_record(self, segment_sentences, segment_index):
        raw_text = " ".join(segment_sentences).strip()
        token_count = self.token_counter(raw_text)

        return {
            "segment_id": segment_index,
            "topic_label": f"Topic {segment_index}",
            "raw_text": raw_text,
            "topical_description": "",
            "token_count": token_count,
            "char_count": len(raw_text),
            "sentence_count": len(segment_sentences),
            "sentences": segment_sentences,
        }

    def _fallback_topical_description(self, segment_text):
        """Generate a lightweight fallback topical description."""
        words = re.findall(r"\S+", segment_text)
        if not words:
            return "Untitled meeting segment"
        return " ".join(words[:15]).strip()

    def _generate_topical_description(self, segment_text):
        """Ask Groq Llama for a concise topical gist."""
        if not segment_text.strip():
            return "Untitled meeting segment"

        if not self.groq_available:
            return self._fallback_topical_description(segment_text)

        self._load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return self._fallback_topical_description(segment_text)

        try:
            client = Groq(api_key=api_key)
            system_prompt = "Summarize the core topic of this meeting segment in under 10 words."
            user_prompt = f"Meeting segment:\n{segment_text[:12000]}"

            result = client.chat.completions.create(
                model=self.topical_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_completion_tokens=40,
            )

            content = result.choices[0].message.content if result.choices else ""
            description = content.strip() if content else ""
            if not description:
                return self._fallback_topical_description(segment_text)

            words = description.split()
            if len(words) > 10:
                description = " ".join(words[:10]).strip()

            return description
        except Exception:
            return self._fallback_topical_description(segment_text)

    def _to_context_rich_string(self, segment):
        """Serialize segment metadata into a context-rich string for recursive summarization."""
        topic_label = segment.get("topic_label", "Topic")
        description = segment.get("topical_description", "")
        token_count = segment.get("token_count", 0)
        sentence_count = segment.get("sentence_count", 0)
        raw_text = segment.get("raw_text", "")
        return (
            f"{topic_label}: {description}\n"
            f"Token count: {token_count} | Sentence count: {sentence_count}\n"
            f"Segment text: {raw_text}"
        )

    def segment_text(self, raw_text, max_tokens=None):
        """
        Segment raw transcribed text into semantic topic clusters.

        Args:
            raw_text (str): Raw transcribed text
            max_tokens (int | None): Optional override for the segment token limit

        Returns:
            list[str]: Context-rich topic strings ready for recursive summarization
        """
        token_limit = max_tokens or self.max_tokens
        sentences = self._split_sentence_units(raw_text)
        if not sentences:
            return []

        similarities = self._compute_similarity_series(sentences)

        clusters = []
        current_cluster = []
        current_tokens = 0

        for index, sentence in enumerate(sentences):
            sentence_tokens = self.token_counter(sentence)
            starts_new_topic = False

            if current_cluster:
                similarity_to_previous = similarities[index - 1]
                starts_new_topic = similarity_to_previous < 0

            exceeds_token_limit = current_cluster and (current_tokens + sentence_tokens > token_limit)

            if current_cluster and (starts_new_topic or exceeds_token_limit):
                clusters.append(current_cluster)
                current_cluster = []
                current_tokens = 0

            current_cluster.append(sentence)
            current_tokens += sentence_tokens

        if current_cluster:
            clusters.append(current_cluster)

        segments = [
            self._build_segment_record(cluster_sentences, topic_index)
            for topic_index, cluster_sentences in enumerate(clusters, 1)
        ]

        if not segments:
            return []

        with ThreadPoolExecutor(max_workers=min(4, len(segments))) as executor:
            descriptions = list(
                executor.map(
                    self._generate_topical_description,
                    [segment["raw_text"] for segment in segments],
                )
            )

        for segment, description in zip(segments, descriptions):
            segment["topical_description"] = description
            segment["metadata"] = {
                "token_count": segment["token_count"],
                "char_count": segment["char_count"],
                "sentence_count": segment["sentence_count"],
            }

        self.last_segment_metadata = segments
        return [self._to_context_rich_string(segment) for segment in segments]

    def get_segment_metadata(self):
        """
        Return the metadata dicts from the most recent segmentation run.

        Returns:
            list[dict]: Segment records with raw_text, topical_description, and counts
        """
        return self.last_segment_metadata

    def print_segments(self, topic_segments):
        """
        Pretty-print segmentation results.

        Args:
            topic_segments (list): List of segment records
        """
        for i, segment in enumerate(topic_segments, 1):
            if isinstance(segment, dict):
                topic_label = segment.get("topic_label") or f"Topic {i}"
                topical_description = segment.get("topical_description", "")
                print(f"\n[{topic_label}] {topical_description}")
                print(
                    f"[TOKENS: {segment.get('token_count', 0)} | CHARS: {segment.get('char_count', 0)} | SENTENCES: {segment.get('sentence_count', 0)}]"
                )
                print(segment.get("raw_text", ""))
                continue

            text = str(segment).strip()
            lines = text.splitlines()
            headline = lines[0] if lines else f"Topic {i}"
            print(f"\n[{headline}]")
            if len(lines) > 1:
                print("\n".join(lines[1:]))
