import threading
import queue
import time
import os
import tempfile
import subprocess
import glob
from tkinter import simpledialog, filedialog

from faster_whisper import WhisperModel
import scipy.io.wavfile as wav
from app.export_service import ExportService

try:
    from integrations.groq.transcribe import transcribe_with_groq
    GROQ_IMPORT_ERROR = None
except Exception as e:
    transcribe_with_groq = None
    GROQ_IMPORT_ERROR = str(e)

# we can add pickle file loading here centralized for the whole app, 
# so if we want to load a saved SVM model or something later we can do it here and pass it to the relevant components (pickle.load(MODEL_PATH) for example)

class AppLogic:
    def __init__(self, view, audio):
        self.view = view
        self.audio = audio
        self.exporter = ExportService()
        self.current_mode = None
        self.current_engine = None
        self.file_transcriber = None

    def _ensure_transcript_ready(self):
        self.view.placeholder_text.place_forget()
        if not self.view.transcript_box.winfo_manager():
            self.view.transcript_box.pack(fill="both", expand=True, padx=5, pady=5)

    def _append_system_text(self, text):
        self._ensure_transcript_ready()
        self.view.transcript_box.configure(state="normal")
        self.view.transcript_box.insert("end", f"[System]: {text}\n")
        self.view.transcript_box.see("end")
        self.view.transcript_box.configure(state="disabled")

    def _prompt_mode_and_engine(self):
        mode = simpledialog.askstring(
            "Mode Selection",
            "Select Mode:\n1 = Live Meeting\n2 = Process File\n3 = Train AI Model"
        )
        if mode is None:
            return None, None

        mode = mode.strip()
        if mode not in {"1", "2", "3"}:
            raise ValueError("Invalid mode. Please enter 1, 2, or 3.")

        engine = None
        if mode in {"1", "2"}:
            engine_input = simpledialog.askstring(
                "Transcription Engine",
                "Select Engine:\n1 = Local Faster-Whisper\n2 = Groq whisper-large-v3-turbo"
            )
            if engine_input is None:
                return None, None

            engine_input = engine_input.strip()
            if engine_input not in {"1", "2"}:
                raise ValueError("Invalid engine. Please enter 1 or 2.")
            engine = "groq" if engine_input == "2" else "local"

        return mode, engine

    def _transcribe_file(self, file_path, engine="local", language="tl"):
        if engine == "groq":
            if transcribe_with_groq is None:
                raise RuntimeError(f"Groq integration unavailable: {GROQ_IMPORT_ERROR}")

            ext = os.path.splitext(file_path)[1].lower()
            is_video = ext in {".mp4", ".mkv", ".mov", ".avi", ".webm"}
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

            if is_video or file_size_mb > 12:
                self._append_system_text(
                    f"Using Groq chunked mode (size={file_size_mb:.1f} MB) for faster, stable processing..."
                )
                return self._transcribe_groq_chunked(file_path, language=language)

            return self._transcribe_groq_with_retry(file_path, language=language)

        if self.file_transcriber is None:
            self.file_transcriber = WhisperModel("medium", device="cpu", compute_type="int8")

        segments, _ = self.file_transcriber.transcribe(file_path, language=language)
        segments = list(segments)
        return " ".join([s.text for s in segments])

    def _can_use_ffmpeg(self):
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _transcribe_groq_with_retry(self, file_path, language="tl", retries=3):
        last_error = None
        for attempt in range(1, retries + 1):
            try:
                return transcribe_with_groq(file_path, language=language)
            except Exception as e:
                last_error = e
                if attempt < retries:
                    wait_s = 2 * attempt
                    self._append_system_text(
                        f"Groq attempt {attempt}/{retries} failed: {e}. Retrying in {wait_s}s..."
                    )
                    time.sleep(wait_s)

        raise RuntimeError(f"Groq failed after {retries} attempts: {last_error}")

    def _transcribe_groq_chunked(self, file_path, language="tl", segment_seconds=180):
        if not self._can_use_ffmpeg():
            self._append_system_text("ffmpeg not found. Falling back to direct Groq upload.")
            return self._transcribe_groq_with_retry(file_path, language=language)

        with tempfile.TemporaryDirectory() as temp_dir:
            chunk_pattern = os.path.join(temp_dir, "chunk_%03d.mp3")

            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", file_path,
                "-vn",
                "-ac", "1",
                "-ar", "16000",
                "-b:a", "32k",
                "-f", "segment",
                "-segment_time", str(segment_seconds),
                "-reset_timestamps", "1",
                chunk_pattern,
            ]

            prep = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=False)
            if prep.returncode != 0:
                self._append_system_text("ffmpeg chunking failed. Falling back to direct Groq upload.")
                return self._transcribe_groq_with_retry(file_path, language=language)

            chunk_files = sorted(glob.glob(os.path.join(temp_dir, "chunk_*.mp3")))
            if not chunk_files:
                self._append_system_text("No chunks produced. Falling back to direct Groq upload.")
                return self._transcribe_groq_with_retry(file_path, language=language)

            transcripts = []
            total = len(chunk_files)

            for idx, chunk_file in enumerate(chunk_files, start=1):
                self._append_system_text(f"Groq chunk {idx}/{total}...")
                text = self._transcribe_groq_with_retry(chunk_file, language=language)
                if text.strip():
                    transcripts.append(text.strip())

            return " ".join(transcripts)

    def _groq_live_transcribe(self, audio_buffer, sample_rate):
        if transcribe_with_groq is None:
            raise RuntimeError(f"Groq integration unavailable: {GROQ_IMPORT_ERROR}")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_path = tmp.name

        try:
            wav.write(temp_path, sample_rate, audio_buffer)
            return transcribe_with_groq(temp_path, language="tl")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _process_file_mode(self, engine):
        file_path = filedialog.askopenfilename(
            title="Select Audio or Video File",
            filetypes=[
                ("Media Files", "*.mp3 *.wav *.m4a *.mp4 *.mkv *.mov *.aac"),
                ("All Files", "*.*"),
            ],
        )

        if not file_path:
            self._append_system_text("No file selected.")
            return

        self._append_system_text(f"Processing file: {file_path}")
        self._append_system_text(f"Engine: {engine}")
        self._append_system_text("Transcribing... please wait.")

        def worker():
            try:
                text = self._transcribe_file(file_path, engine=engine, language="tl")
                if not text.strip():
                    self.view.after(0, self._append_system_text, "No speech detected.")
                    return
                self.view.after(0, self.update_ui_text, f"[00:00] {text}")
            except Exception as e:
                if engine == "groq":
                    self.view.after(0, self._append_system_text, f"Groq failed: {e}")
                    self.view.after(0, self._append_system_text, "Trying local whisper fallback...")
                    try:
                        local_text = self._transcribe_file(file_path, engine="local", language="tl")
                        if local_text.strip():
                            self.view.after(0, self.update_ui_text, f"[00:00] {local_text}")
                            return
                    except Exception as local_err:
                        self.view.after(0, self._append_system_text, f"Local fallback failed: {local_err}")

                self.view.after(0, self._append_system_text, f"File processing failed: {e}")

        threading.Thread(target=worker, daemon=True).start()

    def _start_training_mode(self):
        self._append_system_text("Train AI Mode is available in Main.py CLI flow.")
        self._append_system_text("Open Main.py and select mode 3 for dataset training.")

    def handle_start(self):
        """Initializes audio stream, timer, and UI for recording."""
        try:
            mode, engine = self._prompt_mode_and_engine()
            if mode is None:
                return

            self.current_mode = mode
            self.current_engine = engine

            if mode == "2":
                self._process_file_mode(engine)
                return

            if mode == "3":
                self._start_training_mode()
                return

            # 1. Start the actual AI/Audio engine
            live_transcription = True
            live_transcriber = self._groq_live_transcribe if engine == "groq" else None
            self.audio.start_stream(
                live_transcription=live_transcription,
                live_transcriber=live_transcriber,
            )
            self.view.is_recording = True
            
            # 2. UI TRANSITION: Hide placeholder, Show transcript box
            self._ensure_transcript_ready()
            
            # 3. Update Button and Status
            self.view.status_indicator.configure(text=f"● RECORDING ({engine.upper()})", text_color="#ff4b4b")
            self.view.btn_record.configure(image=self.view.stop_icon, command=self.handle_stop, border_color="#ff4b4b")
            
            # 4. Clear old text and show system message
            self.view.transcript_box.configure(state="normal") # Unlock for system message
            self.view.transcript_box.delete("0.0", "end")
            self.view.transcript_box.insert("end", f"[System]: Mode=Live Meeting | Engine={engine}\n")
            self.view.transcript_box.insert("end", "[System]: Listening...\n")
            if engine == "groq":
                self.view.transcript_box.insert("end", "[System]: Live preview now uses Groq; final pass still runs after stop.\n")
            self.view.transcript_box.configure(state="disabled") # Lock back up
            
            # 5. START THE TIMER LOOP
            self.update_timer_loop(time.time()) 
            
            # 6. Start visualizer and monitor thread
            self.view.animate_bars()
            threading.Thread(target=self.transcription_monitor, daemon=True).start()
            self.update_volume_loop()
            
        except Exception as e:
            # If it fails, show the error in the box
            self.view.transcript_box.configure(state="normal")
            if not self.view.transcript_box.winfo_managed():
                self.view.transcript_box.pack(fill="both", expand=True)
            self.view.transcript_box.insert("end", f"\n[Error]: {e}\n")
            self.view.transcript_box.configure(state="disabled")

    def update_timer_loop(self, start_time):
        """Recursively updates the MM:SS timer label every second."""
        if self.view.is_recording:
            elapsed = int(time.time() - start_time)
            mins, secs = divmod(elapsed, 60)
            self.view.timer_label.configure(text=f"{mins:02d}:{secs:02d}")
            self.view.after(1000, lambda: self.update_timer_loop(start_time))

    def transcription_monitor(self):
        """Threaded monitor that watches for new AI transcription segments."""
        print("[Debug]: UI Monitor Started.")
        while self.view.is_recording:
            try:
                # Check the audio handler's text queue
                new_text = self.audio.text_queue.get(timeout=0.1)
                print(f"[Debug]: UI Received: {new_text}")
                # Safely update the GUI from a background thread
                self.view.after(0, self.update_ui_text, new_text)
            except queue.Empty:
                continue

    def update_ui_text(self, text):
        """Inserts text with Cyan timestamps while keeping the box uneditable."""
        # 1. Unlock the box
        self.view.transcript_box.configure(state="normal")
        
        # 2. Stylized insertion (Timestamp in Cyan, Content in White)
        if "]" in text:
            parts = text.split("]", 1)
            timestamp = parts[0] + "]"
            content = parts[1]
            # Use the 'timestamp' tag defined in gui.py for the cyan color
            self.view.transcript_box.insert("end", timestamp, "timestamp")
            self.view.transcript_box.insert("end", f" {content.strip()}\n")
        else:
            self.view.transcript_box.insert("end", f"{text}\n")
            
        # 3. Auto-scroll and Lock
        self.view.transcript_box.see("end")
        self.view.transcript_box.configure(state="disabled")

    def handle_stop(self):
        """Stops the stream, updates UI status, and locks the transcript."""
        # 1. Stop the audio processing
        saved_path = self.audio.stop_stream()
        self.view.is_recording = False
        
        # 2. Reset UI Status and Button
        self.view.status_indicator.configure(text="● READY TO RECORD", text_color="#4a4d50")
        self.view.btn_record.configure(image=self.view.button_icon, command=self.handle_start, border_color="#00f2ff")
        
        # 3. Insert "Stopped" message safely
        self.view.transcript_box.configure(state="normal")
        self.view.transcript_box.insert("end", "[System]: Stopped.\n")
        self.view.transcript_box.configure(state="disabled")

        # 3b. If using Groq in live mode, transcribe the saved file after stopping.
        if self.current_mode == "1" and self.current_engine == "groq" and saved_path:
            self._append_system_text(f"Sending recording to Groq: {saved_path}")
            self._append_system_text("Transcribing... please wait.")

            def worker():
                try:
                    text = self._transcribe_file(saved_path, engine="groq", language="tl")
                    if not text.strip():
                        self.view.after(0, self._append_system_text, "No speech detected from Groq.")
                        return
                    self.view.after(0, self.update_ui_text, f"[00:00] {text}")
                except Exception as e:
                    self.view.after(0, self._append_system_text, f"Groq transcription failed: {e}")

            threading.Thread(target=worker, daemon=True).start()
        
        # 4. Reset the timer display
        self.view.timer_label.configure(text="00:00")

    def handle_export(self, export_type):
        """Handles PDF generation and Media folder access."""
        content = self.view.transcript_box.get("0.0", "end")
        
        # Unlock temporarily to show system status in the box
        self.view.transcript_box.configure(state="normal")
        
        if export_type == "pdf":
            self.view.transcript_box.insert("end", "[System]: Exporting PDF...\n")
            success = self.exporter.generate_pdf(content)
            if success:
                self.view.transcript_box.insert("end", "[System]: PDF Export Successful.\n")
        
        elif export_type == "video":
            self.view.transcript_box.insert("end", "[System]: Opening Media Folder...\n")
            # You can add os.startfile("output/videos") here if desired
            
        self.view.transcript_box.see("end")
        self.view.transcript_box.configure(state="disabled")
        
        # Hide the pop-up menu
        self.view.toggle_pop_menu()

    def update_volume_loop(self):
        """Syncs the GUI visualizer data with the AudioHandler volume levels."""
        if self.view.is_recording:
            vol = getattr(self.audio, 'current_volume', 0)
            self.view.current_volume = vol if vol is not None else 0
            self.view.after(50, self.update_volume_loop)