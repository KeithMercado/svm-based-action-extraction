import time
import os
import numpy as np
import queue
import threading
import pyaudiowpatch as pyaudio
import scipy.io.wavfile as wav
from datetime import datetime

class AudioHandler:
    def __init__(self):
        self.sample_rate = 16000
        self.capture_rate = self.sample_rate
        self.frames_per_buffer = int(os.getenv("LIVE_FRAMES_PER_BUFFER", "1024"))
        self.live_language = os.getenv("LIVE_TRANSCRIBE_LANGUAGE", "tl")
        self.live_chunk_count = int(os.getenv("LIVE_CHUNK_COUNT", "12"))
        self.live_overlap_count = int(os.getenv("LIVE_OVERLAP_COUNT", "4"))
        self.live_vad_energy_threshold = float(os.getenv("LIVE_VAD_ENERGY_THRESHOLD", "0.008"))
        self.live_silence_seconds = float(os.getenv("LIVE_VAD_SILENCE_SECONDS", "0.7"))
        self.live_force_flush_seconds = float(os.getenv("LIVE_FORCE_TRANSCRIBE_SECONDS", "10.0"))
        self.live_min_transcribe_seconds = float(os.getenv("LIVE_MIN_TRANSCRIBE_SECONDS", "1.2"))

        self.current_volume = 0
        self.audio_queue = queue.Queue(maxsize=120)
        self.text_queue = queue.Queue()
        self._system_chunk_queue = queue.Queue(maxsize=180)
        self._mic_chunk_queue = queue.Queue(maxsize=180)
        self.is_listening = False
        self.live_transcription_enabled = True
        self.live_transcriber = None
        self.all_audio_data = []
        self._last_live_text = ""
        self.start_time = None
        self.elapsed_offset_seconds = 0.0
        self.mic_is_muted = False

        self._state_lock = threading.Lock()
        self._mix_thread = None
        self._transcription_thread = None

        self._audio_interface = None
        self.system_stream = None
        self.mic_stream = None

    def toggle_mic(self, status: bool):
        with self._state_lock:
            self.mic_is_muted = bool(status)
        return self.mic_is_muted

    def _safe_queue_put(self, q, item):
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            pass

        try:
            q.get_nowait()
        except queue.Empty:
            pass

        try:
            q.put_nowait(item)
        except queue.Full:
            pass

    def _system_audio_callback(self, in_data, frame_count, time_info, status_flags):
        if status_flags:
            print(f"[System Audio] {status_flags}")

        if self.is_listening:
            chunk = np.frombuffer(in_data, dtype=np.float32).copy()
            self._safe_queue_put(self._system_chunk_queue, chunk)

        return (None, pyaudio.paContinue)

    def _mic_audio_callback(self, in_data, frame_count, time_info, status_flags):
        if status_flags:
            print(f"[Microphone] {status_flags}")

        if self.is_listening:
            chunk = np.frombuffer(in_data, dtype=np.float32).copy()
            self._safe_queue_put(self._mic_chunk_queue, chunk)

        return (None, pyaudio.paContinue)

    def _normalize_audio_chunk(self, audio_chunk):
        if audio_chunk.size == 0:
            return audio_chunk

        peak = float(np.max(np.abs(audio_chunk)))
        if peak > 1.0:
            audio_chunk = audio_chunk / peak

        return np.clip(audio_chunk, -1.0, 1.0)

    def _align_audio_chunks(self, system_data, mic_data):
        if system_data.size == mic_data.size:
            return system_data, mic_data

        target = max(system_data.size, mic_data.size)
        if system_data.size < target:
            system_data = np.pad(system_data, (0, target - system_data.size))
        if mic_data.size < target:
            mic_data = np.pad(mic_data, (0, target - mic_data.size))
        return system_data, mic_data

    def _resample_to_target_rate(self, audio_chunk, source_rate, target_rate):
        if audio_chunk.size == 0 or source_rate == target_rate:
            return audio_chunk

        target_len = int(round(audio_chunk.size * float(target_rate) / float(source_rate)))
        if target_len <= 1:
            return audio_chunk

        src_x = np.linspace(0.0, 1.0, num=audio_chunk.size, endpoint=False)
        dst_x = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
        return np.interp(dst_x, src_x, audio_chunk).astype(np.float32)

    def _is_input_rate_supported(self, device_index, rate):
        try:
            self._audio_interface.is_format_supported(
                float(rate),
                input_device=int(device_index),
                input_channels=1,
                input_format=pyaudio.paFloat32,
            )
            return True
        except Exception:
            return False

    def _choose_common_capture_rate(self, system_device, mic_device):
        preferred_rate = int(self.sample_rate)
        candidate_rates = [
            preferred_rate,
            int(round(float(system_device.get("defaultSampleRate", 48000)))),
            int(round(float(mic_device.get("defaultSampleRate", 48000)))),
            48000,
            44100,
            32000,
            24000,
            22050,
            16000,
        ]

        unique_candidates = []
        seen = set()
        for rate in candidate_rates:
            if rate <= 0 or rate in seen:
                continue
            seen.add(rate)
            unique_candidates.append(rate)

        system_index = int(system_device["index"])
        mic_index = int(mic_device["index"])

        for rate in unique_candidates:
            if self._is_input_rate_supported(system_index, rate) and self._is_input_rate_supported(mic_index, rate):
                return rate

        raise RuntimeError(
            "No common supported sample rate between system loopback and microphone input devices."
        )

    def _mixing_loop(self):
        while self.is_listening:
            try:
                system_data = self._system_chunk_queue.get(timeout=0.25)
            except queue.Empty:
                continue

            try:
                mic_data = self._mic_chunk_queue.get_nowait()
            except queue.Empty:
                mic_data = np.zeros_like(system_data, dtype=np.float32)

            if self.capture_rate != self.sample_rate:
                system_data = self._resample_to_target_rate(system_data, self.capture_rate, self.sample_rate)
                mic_data = self._resample_to_target_rate(mic_data, self.capture_rate, self.sample_rate)

            system_data, mic_data = self._align_audio_chunks(system_data, mic_data)

            with self._state_lock:
                mic_is_muted = self.mic_is_muted

            if mic_is_muted:
                mixed_chunk = system_data
            else:
                mixed_chunk = (system_data * 0.5) + (mic_data * 0.5)

            mixed_chunk = self._normalize_audio_chunk(mixed_chunk.astype(np.float32))

            self.current_volume = np.linalg.norm(mixed_chunk) * 18
            self._safe_queue_put(self.audio_queue, mixed_chunk.copy())
            self.all_audio_data.append(mixed_chunk.reshape(-1, 1))

    def _create_audio_interface(self):
        if self._audio_interface is None:
            self._audio_interface = pyaudio.PyAudio()

    def _get_system_loopback_device(self):
        env_idx = os.getenv("LIVE_SYSTEM_DEVICE_INDEX")
        if env_idx is not None:
            return self._audio_interface.get_device_info_by_index(int(env_idx))

        default_output = self._audio_interface.get_default_wasapi_loopback()
        if default_output:
            return default_output

        return self._audio_interface.get_default_output_device_info()

    def _get_mic_input_device(self):
        env_idx = os.getenv("LIVE_MIC_DEVICE_INDEX")
        if env_idx is not None:
            return self._audio_interface.get_device_info_by_index(int(env_idx))

        return self._audio_interface.get_default_input_device_info()

    def _drain_queue(self, q):
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            return

    def start_stream(self, live_transcription=True, live_transcriber=None, reset_buffer=True, continue_timing=False):
        if self.is_listening:
            return

        self.is_listening = True
        self.live_transcription_enabled = live_transcription
        self.live_transcriber = live_transcriber

        if reset_buffer:
            self.all_audio_data = []
            self.elapsed_offset_seconds = 0.0
        elif not continue_timing:
            self.elapsed_offset_seconds = 0.0

        self._last_live_text = ""
        self.start_time = time.time() - (self.elapsed_offset_seconds if continue_timing else 0.0)

        self._drain_queue(self.audio_queue)
        self._drain_queue(self.text_queue)
        self._drain_queue(self._system_chunk_queue)
        self._drain_queue(self._mic_chunk_queue)

        self._create_audio_interface()

        system_device = self._get_system_loopback_device()
        mic_device = self._get_mic_input_device()
        self.capture_rate = self._choose_common_capture_rate(system_device, mic_device)

        try:
            self.system_stream = self._audio_interface.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.capture_rate,
                input=True,
                frames_per_buffer=self.frames_per_buffer,
                input_device_index=int(system_device["index"]),
                stream_callback=self._system_audio_callback,
            )

            self.mic_stream = self._audio_interface.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.capture_rate,
                input=True,
                frames_per_buffer=self.frames_per_buffer,
                input_device_index=int(mic_device["index"]),
                stream_callback=self._mic_audio_callback,
            )

            self.system_stream.start_stream()
            self.mic_stream.start_stream()

            self._mix_thread = threading.Thread(target=self._mixing_loop, daemon=True)
            self._mix_thread.start()

            if self.live_transcription_enabled:
                self._transcription_thread = threading.Thread(target=self._transcription_loop, daemon=True)
                self._transcription_thread.start()
        except Exception:
            self.stop_stream(save=False)
            raise

    def _normalize_live_text(self, text):
        cleaned = " ".join(text.split())
        if not cleaned:
            return ""
        cleaned = cleaned[0].upper() + cleaned[1:]
        if cleaned[-1] not in ".!?":
            cleaned += "."
        return cleaned

    def _chunk_has_voice(self, audio_chunk):
        # Lightweight RMS VAD keeps dependencies low and is fast enough for live loops.
        energy = float(np.sqrt(np.mean(np.square(audio_chunk)))) if audio_chunk.size else 0.0
        return energy >= self.live_vad_energy_threshold

    def _transcription_loop(self):
        print("[Debug]: Transcription Loop Started.")
        if self.live_transcriber is None:
            self.text_queue.put("[System] Live transcriber is not configured.")
            return

        recorded_chunks = []
        buffered_samples = 0
        silence_samples = 0
        detected_speech = False
        
        while self.is_listening:
            try:
                audio_chunk = self.audio_queue.get(timeout=1)
                elapsed_time = time.time() - self.start_time
                recorded_chunks.append(audio_chunk)
                chunk_samples = len(audio_chunk)
                buffered_samples += chunk_samples

                has_voice = self._chunk_has_voice(audio_chunk)
                if has_voice:
                    detected_speech = True
                    silence_samples = 0
                else:
                    silence_samples += chunk_samples

                buffered_seconds = buffered_samples / float(self.sample_rate)
                silence_seconds = silence_samples / float(self.sample_rate)

                should_flush = False
                if (
                    detected_speech
                    and buffered_seconds >= self.live_min_transcribe_seconds
                    and silence_seconds >= self.live_silence_seconds
                ):
                    should_flush = True

                # Safety override for continuous speech with no pause.
                if detected_speech and buffered_seconds >= self.live_force_flush_seconds:
                    should_flush = True

                if should_flush:
                    full_buffer = np.concatenate(recorded_chunks)
                    recorded_chunks = []
                    buffered_samples = 0
                    silence_samples = 0
                    detected_speech = False

                    external_text = self.live_transcriber(full_buffer, self.sample_rate)
                    cleaned = self._normalize_live_text(external_text)
                    if cleaned and cleaned != self._last_live_text:
                        self._last_live_text = cleaned
                        mins, secs = divmod(int(elapsed_time), 60)
                        timestamp = f"[{mins:02d}:{secs:02d}]"
                        self.text_queue.put(f"{timestamp} {cleaned}")
            except queue.Empty:
                continue
            except Exception as e:
                self.text_queue.put(f"[System] Live transcription warning: {e}")

        # Flush trailing speech once capture stops.
        if recorded_chunks and detected_speech:
            try:
                full_buffer = np.concatenate(recorded_chunks)
                elapsed_time = time.time() - self.start_time if self.start_time else 0
                external_text = self.live_transcriber(full_buffer, self.sample_rate)
                cleaned = self._normalize_live_text(external_text)
                if cleaned and cleaned != self._last_live_text:
                    self._last_live_text = cleaned
                    mins, secs = divmod(int(elapsed_time), 60)
                    timestamp = f"[{mins:02d}:{secs:02d}]"
                    self.text_queue.put(f"{timestamp} {cleaned}")
            except Exception as e:
                self.text_queue.put(f"[System] Live transcription warning: {e}")

    def stop_stream(self, save=False, clear_buffer=False):
        self.is_listening = False
        if self.start_time is not None:
            self.elapsed_offset_seconds = max(0.0, time.time() - self.start_time)

        for stream_attr in ("system_stream", "mic_stream"):
            stream = getattr(self, stream_attr)
            if stream is None:
                continue

            try:
                if stream.is_active():
                    stream.stop_stream()
            except Exception:
                pass

            try:
                stream.close()
            except Exception:
                pass

            setattr(self, stream_attr, None)

        if self._audio_interface is not None:
            try:
                self._audio_interface.terminate()
            except Exception:
                pass
            self._audio_interface = None
        
        saved_path = None
        if save and self.all_audio_data:
            saved_path = self.save_recorded_audio()
            if clear_buffer:
                self.all_audio_data = []

        return saved_path

    def clear_recording_buffer(self):
        self.all_audio_data = []
        self.elapsed_offset_seconds = 0.0
        self.start_time = None

    def save_recorded_audio(self):
        """Saves the collected audio buffer into the output/videos folder."""
        try:
            # Create directory if it doesn't exist
            output_dir = os.path.join("output", "videos")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Generate filename with timestamp for uniqueness
            current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            label = "Recorded_Audio"
            filename = f"{current_date}_{label}.wav"
            filepath = os.path.join(output_dir, filename)

            # Process the audio data collected during the session
            if self.all_audio_data:
                full_audio = np.concatenate(self.all_audio_data, axis=0)

                # Convert float32 [-1, 1] to int16 for standard WAV compatibility.
                int_audio = np.int16(np.clip(full_audio, -1.0, 1.0) * 32767)
                wav.write(filepath, self.sample_rate, int_audio)
                print(f"[Debug]: Audio saved successfully as {filename}")
                return filepath
            else:
                print("[Debug]: No audio data to save.")
                return None

        except Exception as e:
            print(f"[Error] Failed to save audio file: {e}")
            return None