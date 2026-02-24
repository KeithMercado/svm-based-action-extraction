import sounddevice as sd
import numpy as np
import queue
import threading
from faster_whisper import WhisperModel

class AudioHandler:
    def __init__(self):
        self.current_volume = 0
        self.stream = None
        self.audio_queue = queue.Queue()  
        self.text_queue = queue.Queue()   
        self.is_listening = False
        self.model = WhisperModel("base", device="cpu", compute_type="int8")

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        volume_norm = np.linalg.norm(indata) * 18
        self.current_volume = volume_norm
        
        if self.is_listening:
            self.audio_queue.put(indata.copy().flatten())

    def start_stream(self):
        self.is_listening = True
        self.stream = sd.InputStream(samplerate=16000, channels=1, callback=self.audio_callback)
        self.stream.start()
        threading.Thread(target=self._transcription_loop, daemon=True).start()

    def _transcription_loop(self):
        print("[Debug]: Transcription Loop Started.")
        recorded_chunks = [] 
        
        while self.is_listening:
            try:
                audio_chunk = self.audio_queue.get(timeout=1)
                recorded_chunks.append(audio_chunk)

                # Wait for ~30 chunks to avoid transcribing tiny fragments
                if len(recorded_chunks) >= 30:
                    full_buffer = np.concatenate(recorded_chunks)
                    print(f"[Debug]: Transcribing large buffer, size: {len(full_buffer)}")
                    
                    recorded_chunks = [] 

                    segments, _ = self.model.transcribe(full_buffer, language="tl", vad_filter=True)
                    
                    for segment in segments:
                        if segment.text.strip():
                            print(f"[Debug]: AI Found Text: {segment.text}")
                            self.text_queue.put(segment.text.strip())
            except queue.Empty:
                continue

    def stop_stream(self):
        self.is_listening = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.current_volume = 0