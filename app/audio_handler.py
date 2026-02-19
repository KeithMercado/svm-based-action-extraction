import sounddevice as sd
import numpy as np

class AudioHandler:
    def __init__(self):
        self.current_volume = 0
        self.stream = None

    def audio_callback(self, indata, frames, time, status):
        """Calculates volume levels from microphone input."""
        if status:
            print(status)
        # Calculate RMS volume
        volume_norm = np.linalg.norm(indata) * 18
        self.current_volume = volume_norm

    def start_stream(self):
        self.stream = sd.InputStream(callback=self.audio_callback)
        self.stream.start()

    def stop_stream(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.current_volume = 0