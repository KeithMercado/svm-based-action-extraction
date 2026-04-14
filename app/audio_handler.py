import time
import os
import sounddevice as sd
import numpy as np
import queue
import threading
from faster_whisper import WhisperModel
import scipy.io.wavfile as wav
from datetime import datetime

# if system audio is needed:

# sounddevice alone cannot record system audio, but it can capture microphone input.
# add soundcard alongside sounddevice
# switch to pyaudiowpatch for better system audio capture

class AudioHandler:
    def __init__(self):
        self.sample_rate = 16000
        self.live_language = os.getenv("LIVE_TRANSCRIBE_LANGUAGE", "tl")
        self.live_chunk_count = int(os.getenv("LIVE_CHUNK_COUNT", "12"))
        self.live_overlap_count = int(os.getenv("LIVE_OVERLAP_COUNT", "4"))
        model_size = os.getenv("LIVE_WHISPER_MODEL", "medium")
        self.live_model_name = model_size

        self.current_volume = 0
        self.stream = None
        self.audio_queue = queue.Queue()  
        self.text_queue = queue.Queue()   
        self.is_listening = False
        self.live_transcription_enabled = True
        self.live_transcriber = None
        self.model = None
        self.all_audio_data = [] # List to store every chunk for saving later
        self._last_live_text = ""
        self.start_time = None
        self.elapsed_offset_seconds = 0.0

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        volume_norm = np.linalg.norm(indata) * 18
        self.current_volume = volume_norm
        
        if self.is_listening:
            data_copy = indata.copy()
            self.audio_queue.put(data_copy.flatten())
            self.all_audio_data.append(data_copy) # Collect data for the final file

    def start_stream(self, live_transcription=True, live_transcriber=None, reset_buffer=True, continue_timing=False):
        self.is_listening = True
        self.live_transcription_enabled = live_transcription
        self.live_transcriber = live_transcriber
        if reset_buffer:
            self.all_audio_data = []
            self.elapsed_offset_seconds = 0.0
        elif not continue_timing:
            self.elapsed_offset_seconds = 0.0
        self._last_live_text = ""
        # Keep timestamps continuous across pause/resume when continue_timing is enabled.
        self.start_time = time.time() - (self.elapsed_offset_seconds if continue_timing else 0.0)
        self.stream = sd.InputStream(samplerate=self.sample_rate, channels=1, callback=self.audio_callback)
        self.stream.start()
        if self.live_transcription_enabled:
            threading.Thread(target=self._transcription_loop, daemon=True).start()

    def _normalize_live_text(self, text):
        cleaned = " ".join(text.split())
        if not cleaned:
            return ""
        cleaned = cleaned[0].upper() + cleaned[1:]
        if cleaned[-1] not in ".!?":
            cleaned += "."
        return cleaned

    def _transcription_loop(self):
        print("[Debug]: Transcription Loop Started.")
        recorded_chunks = [] 

        if self.live_transcriber is None and self.model is None:
            self.model = WhisperModel(self.live_model_name, device="cpu", compute_type="int8")
        
        while self.is_listening:
            try:
                audio_chunk = self.audio_queue.get(timeout=1)
                # Calculate the timestamp for this specific chunk relative to start_time
                elapsed_time = time.time() - self.start_time 
                recorded_chunks.append(audio_chunk)

                # Slightly larger chunk + overlap improves readability/accuracy while staying near real-time.
                if len(recorded_chunks) >= self.live_chunk_count:
                    full_buffer = np.concatenate(recorded_chunks)
                    if self.live_overlap_count > 0:
                        recorded_chunks = recorded_chunks[-self.live_overlap_count:]
                    else:
                        recorded_chunks = []

                    if self.live_transcriber is not None:
                        external_text = self.live_transcriber(full_buffer, self.sample_rate)
                        cleaned = self._normalize_live_text(external_text)
                        if cleaned and cleaned != self._last_live_text:
                            self._last_live_text = cleaned
                            mins, secs = divmod(int(elapsed_time), 60)
                            timestamp = f"[{mins:02d}:{secs:02d}]"
                            self.text_queue.put(f"{timestamp} {cleaned}")
                    else:
                        # Language "tl" for Taglish/Tagalog
                        segments, _ = self.model.transcribe(
                            full_buffer,
                            language=self.live_language,
                            beam_size=5,
                            best_of=5,
                            temperature=0,
                            condition_on_previous_text=True,
                            vad_filter=True,
                            no_speech_threshold=0.6,
                            compression_ratio_threshold=2.4,
                            log_prob_threshold=-1.0,
                        )
                        
                        for segment in segments:
                            cleaned = self._normalize_live_text(segment.text)
                            if cleaned and cleaned != self._last_live_text:
                                self._last_live_text = cleaned
                                # Format the time as [MM:SS]
                                mins, secs = divmod(int(elapsed_time), 60)
                                timestamp = f"[{mins:02d}:{secs:02d}]"
                                
                                # Send both the timestamp and the text
                                self.text_queue.put(f"{timestamp} {cleaned}")
            except queue.Empty:
                continue
            except Exception as e:
                self.text_queue.put(f"[System] Live transcription warning: {e}")

    def stop_stream(self, save=False, clear_buffer=False):
        self.is_listening = False
        if self.start_time is not None:
            self.elapsed_offset_seconds = max(0.0, time.time() - self.start_time)
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        # Call the save method when stopping
        saved_path = None
        if save and self.all_audio_data:
            saved_path = self.save_recorded_audio()
            if clear_buffer:
                self.all_audio_data = [] # Clear for next session

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
                # Stack all recorded chunks into one array
                full_audio = np.concatenate(self.all_audio_data, axis=0)
                
                # Write the file (16000Hz is your defined sample rate)
                wav.write(filepath, self.sample_rate, full_audio)
                print(f"[Debug]: Audio saved successfully as {filename}")
                return filepath
            else:
                print("[Debug]: No audio data to save.")
                return None

        except Exception as e:
            print(f"[Error] Failed to save audio file: {e}")
            return None