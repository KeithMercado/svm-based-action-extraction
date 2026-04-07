"""
Phase 2: Segmentation Module
Handles topic segmentation of transcribed text.
"""


class Segmenter:
    """Segments transcribed text into topic-based chunks."""

    def __init__(self, chunk_size=5):
        """
        Initialize the Segmenter.

        Args:
            chunk_size (int): Number of sentences per segment
        """
        self.chunk_size = chunk_size

    def segment_text(self, raw_text):
        """
        Segment raw transcribed text into topic-based chunks.

        Args:
            raw_text (str): Raw transcribed text

        Returns:
            list: List of segments, where each segment is a list of sentences
        """
        # Split into sentences
        sentences = [
            s.strip()
            for s in raw_text.replace("so ", ". ").split(".")
            if len(s) > 5
        ]

        # Group sentences into chunks
        topic_segments = [
            sentences[i : i + self.chunk_size]
            for i in range(0, len(sentences), self.chunk_size)
        ]

        return topic_segments

    def print_segments(self, topic_segments):
        """
        Pretty-print segmentation results.

        Args:
            topic_segments (list): List of segments
        """
        for i, segment in enumerate(topic_segments, 1):
            print(f"\n[SEGMENT {i}]:")
            print(" ".join(segment))
