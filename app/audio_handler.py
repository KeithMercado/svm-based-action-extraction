import time
import os
import wave
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
        self.current_volume = 0
        self.stream = None
        self.audio_queue = queue.Queue()  
        self.text_queue = queue.Queue()   
        self.is_listening = False
        self.model = WhisperModel("small", device="cpu", compute_type="int8")
        self.all_audio_data = [] # List to store every chunk for saving later

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        volume_norm = np.linalg.norm(indata) * 18
        self.current_volume = volume_norm
        
        if self.is_listening:
            data_copy = indata.copy()
            self.audio_queue.put(data_copy.flatten())
            self.all_audio_data.append(data_copy) # Collect data for the final file

    def start_stream(self):
        self.is_listening = True
        # Record the start time of the session
        self.start_time = time.time()
        self.stream = sd.InputStream(samplerate=16000, channels=1, callback=self.audio_callback)
        self.stream.start()
        threading.Thread(target=self._transcription_loop, daemon=True).start()

    def _transcription_loop(self):
        print("[Debug]: Transcription Loop Started.")
        recorded_chunks = [] 
        
        while self.is_listening:
            try:
                audio_chunk = self.audio_queue.get(timeout=1)
                # Calculate the timestamp for this specific chunk relative to start_time
                elapsed_time = time.time() - self.start_time 
                recorded_chunks.append(audio_chunk)

                # REDUCE THIS NUMBER: 30 chunks is about 0.8 seconds of audio.
                # Try 10-15 for a faster "live" feel, or 1 to process every single chunk.
                if len(recorded_chunks) >= 15: 
                    full_buffer = np.concatenate(recorded_chunks)
                    recorded_chunks = [] # Reset buffer

                    # Language "tl" for Taglish/Tagalog
                    segments, _ = self.model.transcribe(full_buffer, language="tl", vad_filter=True)
                    
                    for segment in segments:
                        if segment.text.strip():
                            # Format the time as [MM:SS]
                            mins, secs = divmod(int(elapsed_time), 60)
                            timestamp = f"[{mins:02d}:{secs:02d}]"
                            
                            # Send both the timestamp and the text
                            self.text_queue.put(f"{timestamp} {segment.text.strip()}")
            except queue.Empty:
                continue

    def stop_stream(self):
        self.is_listening = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        # Call the save method when stopping
        if self.all_audio_data:
            self.save_recorded_audio()
            self.all_audio_data = [] # Clear for next session

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
                wav.write(filepath, 16000, full_audio)
                print(f"[Debug]: Audio saved successfully as {filename}")
            else:
                print("[Debug]: No audio data to save.")

        except Exception as e:
            print(f"[Error] Failed to save audio file: {e}")