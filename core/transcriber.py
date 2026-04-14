"""
Phase 1: Transcription Module
Handles both Groq and Local Whisper transcriptions.
"""

import os
import time
import tempfile
from faster_whisper import WhisperModel
import scipy.io.wavfile as wav

try:
    from integrations.groq.transcribe import transcribe_with_groq
    GROQ_AVAILABLE = True
    GROQ_IMPORT_ERROR = None
except Exception as e:
    GROQ_AVAILABLE = False
    GROQ_IMPORT_ERROR = str(e)


class Transcriber:
    """Orchestrates transcription using either Groq API or Local Whisper."""

    def __init__(self):
        self.groq_available = GROQ_AVAILABLE
        self.local_model = None

    def _load_local_model(self, model_size="medium"):
        """Lazily load the local Whisper model."""
        if self.local_model is None:
            print(f"[System] Loading local Whisper model ({model_size})...")
            self.local_model = WhisperModel(model_size, device="cpu", compute_type="int8")
        return self.local_model

    def transcribe_with_groq_retries(
        self,
        file_path,
        language="tl",
        retries=3,
        initial_prompt=None,
    ):
        """Transcribe with Groq API, with automatic retries on failure."""
        if not GROQ_AVAILABLE:
            raise RuntimeError(
                f"Groq integration is unavailable: {GROQ_IMPORT_ERROR}"
            )

        last_error = None
        for attempt in range(1, retries + 1):
            try:
                return transcribe_with_groq(
                    file_path,
                    language=language,
                    initial_prompt=initial_prompt,
                )
            except Exception as e:
                last_error = e
                if attempt < retries:
                    wait_s = 2 * attempt
                    print(
                        f"[System] Groq attempt {attempt}/{retries} failed: {e}. "
                        f"Retrying in {wait_s}s..."
                    )
                    time.sleep(wait_s)

        raise RuntimeError(
            f"Groq transcription failed after {retries} attempts: {last_error}"
        )

    def transcribe_file(
        self, filename, engine="local", language="tl", retries=3
    ):
        """
        Transcribe an audio/video file.

        Args:
            filename (str): Path to audio/video file
            engine (str): "local" or "groq"
            language (str): Language code (e.g., "tl" for Tagalog, "en" for English)
            retries (int): Number of retry attempts for Groq

        Returns:
            str: Transcribed text
        """
        if engine == "groq":
            from utils.audio_utils import is_video_file

            if not self.groq_available:
                raise RuntimeError(
                    f"Groq integration is unavailable: {GROQ_IMPORT_ERROR}"
                )

            ext = os.path.splitext(filename)[1].lower()
            size_mb = os.path.getsize(filename) / (1024 * 1024)

            # Use chunked mode for large files or video
            if is_video_file(filename) or size_mb > 12:
                print(
                    f"[System] Using fast chunked Groq mode (size={size_mb:.1f} MB)..."
                )
                from utils.audio_utils import transcribe_with_groq_chunked

                return transcribe_with_groq_chunked(
                    filename, language=language, retries=retries
                )

            return self.transcribe_with_groq_retries(
                filename, language=language, retries=retries
            )

        # Local Whisper
        model = self._load_local_model("medium")
        segments, _ = model.transcribe(filename, language=language)
        segments = list(segments)

        if not segments:
            return ""

        return " ".join([s.text for s in segments])

    def transcribe_recorded_audio(self, audio_file_path, engine="local", language="tl"):
        """
        Transcribe audio from a recorded file (after recording completes).

        Args:
            audio_file_path (str): Path to saved audio file
            engine (str): "local" or "groq"
            language (str): Language code

        Returns:
            str: Transcribed text
        """
        return self.transcribe_file(
            audio_file_path, engine=engine, language=language
        )

    def transcribe_live_buffer_groq(
        self,
        audio_buffer,
        sample_rate,
        language="tl",
        retries=3,
        initial_prompt="This is a Taglish meeting transcript involving technical tasks and action items.",
    ):
        """Transcribe in-memory live audio using Groq without chunked file logic."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_path = tmp.name

        try:
            wav.write(temp_path, sample_rate, audio_buffer)
            return self.transcribe_with_groq_retries(
                temp_path,
                language=language,
                retries=retries,
                initial_prompt=initial_prompt,
            )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
