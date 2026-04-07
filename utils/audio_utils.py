"""
Audio Utilities Module
Handles recording logic, FFmpeg chunking, and file operations.
"""

import os
import glob
import subprocess
import tempfile
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import queue
import time
from faster_whisper import WhisperModel

try:
    from integrations.groq.transcribe import transcribe_with_groq
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False


def is_video_file(filename):
    """Check if file is a video format."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in {".mp4", ".mkv", ".mov", ".avi", ".webm"}


def can_use_ffmpeg():
    """Check if FFmpeg is available on the system."""
    try:
        res = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        return res.returncode == 0
    except Exception:
        return False


def transcribe_with_groq_chunked(
    file_path, language="tl", retries=3, segment_seconds=180
):
    """
    Compress and chunk media for faster and more reliable Groq transcription.

    Args:
        file_path (str): Path to media file
        language (str): Language code
        retries (int): Number of retry attempts
        segment_seconds (int): Duration of each segment in seconds

    Returns:
        str: Concatenated transcription of all chunks
    """
    if not GROQ_AVAILABLE:
        raise RuntimeError("Groq integration is not available")

    if not can_use_ffmpeg():
        print("[System] ffmpeg not found. Using direct Groq upload.")
        from core.transcriber import Transcriber

        transcriber = Transcriber()
        return transcriber.transcribe_with_groq_retries(
            file_path, language=language, retries=retries
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        chunk_pattern = os.path.join(temp_dir, "chunk_%03d.mp3")
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            file_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-b:a",
            "32k",
            "-f",
            "segment",
            "-segment_time",
            str(segment_seconds),
            "-reset_timestamps",
            "1",
            chunk_pattern,
        ]

        print("[System] Preparing compressed chunks for Groq...")
        prep = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=False)
        if prep.returncode != 0:
            print("[System] ffmpeg chunking failed. Falling back to direct Groq upload.")
            from core.transcriber import Transcriber

            transcriber = Transcriber()
            return transcriber.transcribe_with_groq_retries(
                file_path, language=language, retries=retries
            )

        chunk_files = sorted(glob.glob(os.path.join(temp_dir, "chunk_*.mp3")))
        if not chunk_files:
            print("[System] No chunks generated. Falling back to direct Groq upload.")
            from core.transcriber import Transcriber

            transcriber = Transcriber()
            return transcriber.transcribe_with_groq_retries(
                file_path, language=language, retries=retries
            )

        texts = []
        total = len(chunk_files)
        for idx, chunk_file in enumerate(chunk_files, start=1):
            print(
                f"[System] Groq chunk {idx}/{total}: {os.path.basename(chunk_file)}"
            )
            from core.transcriber import Transcriber

            transcriber = Transcriber()
            part = transcriber.transcribe_with_groq_retries(
                chunk_file, language=language, retries=retries
            )
            if part.strip():
                texts.append(part.strip())

        return " ".join(texts)


class AudioRecorder:
    """Handles real-time audio recording and collection."""

    def __init__(self, sample_rate=16000):
        """
        Initialize the audio recorder.

        Args:
            sample_rate (int): Sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.audio_data = []
        self.audio_queue = queue.Queue()
        self.is_recording = False

    def callback(self, indata, frames, time, status):
        """Callback function for audio streaming."""
        self.audio_data.append(indata.copy())
        if len(self.audio_data) % 150 == 0:
            chunk = np.concatenate(self.audio_data[-150:])
            self.audio_queue.put(chunk.flatten())

    def start_recording(self):
        """Start recording audio from the microphone."""
        self.is_recording = True
        self.audio_data = []
        print("[RECORDING] Speak into your mic. Press [ENTER] to stop...\n")

        with sd.InputStream(
            samplerate=self.sample_rate, channels=1, callback=self.callback
        ):
            input()  # Wait for user to press ENTER

        self.is_recording = False
        return self.get_recorded_audio()

    def get_recorded_audio(self):
        """Get the full recorded audio array."""
        if not self.audio_data:
            return np.array([])
        return np.concatenate(self.audio_data)

    def save_recorded_audio(self, output_path):
        """
        Save recorded audio to a WAV file.

        Args:
            output_path (str): Path to save the WAV file
        """
        full_audio = self.get_recorded_audio()
        if full_audio.size == 0:
            print("[Error] No audio data to save.")
            return

        wav.write(output_path, self.sample_rate, full_audio)
        print(f"[System] Audio saved to {output_path}")

    def get_audio_queue(self):
        """Get the audio queue for real-time processing."""
        return self.audio_queue
