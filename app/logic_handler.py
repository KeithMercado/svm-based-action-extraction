import threading
import queue
import time
import os
import tempfile
import subprocess
import glob
import pickle
from tkinter import simpledialog, filedialog, messagebox

from faster_whisper import WhisperModel
import scipy.io.wavfile as wav
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from app.export_service import ExportService

try:
    from integrations.groq.transcribe import transcribe_with_groq
    GROQ_IMPORT_ERROR = None
except Exception as e:
    transcribe_with_groq = None
    GROQ_IMPORT_ERROR = str(e)

try:
    from integrations.groq.summarize import summarize_with_groq
    GROQ_SUMMARY_IMPORT_ERROR = None
except Exception as e:
    summarize_with_groq = None
    GROQ_SUMMARY_IMPORT_ERROR = str(e)

try:
    from transformers import BartForConditionalGeneration, BartTokenizer
    LOCAL_BART_IMPORT_ERROR = None
except Exception as e:
    BartForConditionalGeneration = None
    BartTokenizer = None
    LOCAL_BART_IMPORT_ERROR = str(e)

# we can add pickle file loading here centralized for the whole app, 
# so if we want to load a saved SVM model or something later we can do it here and pass it to the relevant components (pickle.load(MODEL_PATH) for example)

class AppLogic:
    def __init__(self, view, audio):
        self.view = view
        self.audio = audio
        self.exporter = ExportService()
        self.current_mode = None
        self.current_engine = None
        self.current_summary_engine = "groq"
        self.file_transcriber = None
        self.project_root = os.path.dirname(os.path.dirname(__file__))
        self.model_path = os.path.join(self.project_root, "svm_model.pkl")
        self.vectorizer = HashingVectorizer(n_features=2**16, ngram_range=(1, 3), alternate_sign=False)
        self.classifier = self._load_or_init_classifier()
        self.local_bart = None

    def _format_secs(self, seconds):
        return f"{seconds:.1f}s"

    def _load_or_init_classifier(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                return pickle.load(f)

        model = SGDClassifier(loss="hinge")
        X_initial = self.vectorizer.transform(["info", "please do this"])
        y_initial = [0, 1]
        model.partial_fit(X_initial, y_initial, classes=[0, 1])
        return model

    def _get_features(self, sentence):
        return self.vectorizer.transform([sentence])

    def _load_local_bart(self):
        if self.local_bart is not None:
            return self.local_bart
        if BartForConditionalGeneration is None or BartTokenizer is None:
            raise RuntimeError(f"Local BART unavailable: {LOCAL_BART_IMPORT_ERROR}")

        model_name = "facebook/bart-large-cnn"
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)
        self.local_bart = (tokenizer, model)
        return self.local_bart

    def _summarize_with_local_bart(self, text, action_items):
        tokenizer, model = self._load_local_bart()
        input_text = f"Meeting: {text}"
        if action_items:
            input_text += " Tasks: " + " . ".join(action_items)

        inputs = tokenizer([input_text], max_length=1024, return_tensors="pt", truncation=True)
        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=4,
            min_length=10,
            max_length=80,
            no_repeat_ngram_size=3,
            encoder_no_repeat_ngram_size=3,
            early_stopping=True,
            forced_bos_token_id=0,
        )
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

    def _segment_topics(self, raw_text):
        sentences = [s.strip() for s in raw_text.replace("so ", ". ").split(".") if len(s.strip()) > 5]
        return [sentences[i:i + 5] for i in range(0, len(sentences), 5)]

    def _extract_actions(self, topic_segments):
        all_chunks_data = []
        all_actions = []

        for chunk in topic_segments:
            detected_actions = []
            for sentence in chunk:
                prediction = self.classifier.predict(self._get_features(sentence))[0]
                if prediction == 1:
                    detected_actions.append(sentence)
                    all_actions.append(sentence)

            all_chunks_data.append({
                "chunk_text": " ".join(chunk),
                "actions": detected_actions,
            })

        return all_chunks_data, all_actions

    def _summarize_chunk(self, text, actions, engine):
        if engine == "groq":
            if summarize_with_groq is None:
                raise RuntimeError(f"Groq summarizer unavailable: {GROQ_SUMMARY_IMPORT_ERROR}")
            return summarize_with_groq(text, actions)
        return self._summarize_with_local_bart(text, actions)

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

    def _set_processing_state(self, active, base_text="● PROCESSING FILE"):
        if active:
            self._processing_state_active = True
            self._processing_state_base = base_text
            self._processing_state_step = 0
            self._tick_processing_state()
            return

        self._processing_state_active = False
        self.view.status_indicator.configure(text="● READY TO RECORD", text_color="#4a4d50")

    def _tick_processing_state(self):
        if not getattr(self, "_processing_state_active", False):
            return

        dots = "." * ((getattr(self, "_processing_state_step", 0) % 3) + 1)
        self.view.status_indicator.configure(
            text=f"{self._processing_state_base}{dots}",
            text_color="#3a7ebf",
        )
        self._processing_state_step = getattr(self, "_processing_state_step", 0) + 1
        self.view.after(450, self._tick_processing_state)

    def _prompt_mode_and_engine(self):
        mode = simpledialog.askstring(
            "Mode Selection",
            "Select Mode:\n1 = Live Meeting\n2 = Process File\n3 = Train AI Model"
        )
        if mode is None:
            return None, None, None

        mode = mode.strip()
        if mode not in {"1", "2", "3"}:
            raise ValueError("Invalid mode. Please enter 1, 2, or 3.")

        engine = None
        summary_engine = None
        if mode in {"1", "2"}:
            engine_input = simpledialog.askstring(
                "Transcription Engine",
                "Select Engine:\n1 = Local Faster-Whisper\n2 = Groq whisper-large-v3-turbo"
            )
            if engine_input is None:
                return None, None, None

            engine_input = engine_input.strip()
            if engine_input not in {"1", "2"}:
                raise ValueError("Invalid engine. Please enter 1 or 2.")
            engine = "groq" if engine_input == "2" else "local"

            summary_input = simpledialog.askstring(
                "Summarization Engine",
                "Select Engine:\n1 = Groq Llama (fast API)\n2 = Local BART"
            )
            if summary_input is None:
                return None, None, None
            summary_input = summary_input.strip()
            if summary_input not in {"1", "2"}:
                raise ValueError("Invalid summarization engine. Please enter 1 or 2.")
            summary_engine = "groq" if summary_input == "1" else "local"

        return mode, engine, summary_engine

    def _transcribe_file(self, file_path, engine="local", language="tl", quiet=False):
        if engine == "groq":
            if transcribe_with_groq is None:
                raise RuntimeError(f"Groq integration unavailable: {GROQ_IMPORT_ERROR}")

            ext = os.path.splitext(file_path)[1].lower()
            is_video = ext in {".mp4", ".mkv", ".mov", ".avi", ".webm"}
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

            if is_video or file_size_mb > 12:
                if not quiet:
                    self._append_system_text(
                        f"Using Groq chunked mode (size={file_size_mb:.1f} MB) for faster, stable processing..."
                    )
                return self._transcribe_groq_chunked(file_path, language=language, quiet=quiet)

            return self._transcribe_groq_with_retry(file_path, language=language, quiet=quiet)

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

    def _transcribe_groq_with_retry(self, file_path, language="tl", retries=3, quiet=False):
        last_error = None
        for attempt in range(1, retries + 1):
            try:
                return transcribe_with_groq(file_path, language=language)
            except Exception as e:
                last_error = e
                if attempt < retries and not quiet:
                    wait_s = 2 * attempt
                    self._append_system_text(
                        f"Groq attempt {attempt}/{retries} failed: {e}. Retrying in {wait_s}s..."
                    )
                    time.sleep(wait_s)

        raise RuntimeError(f"Groq failed after {retries} attempts: {last_error}")

    def _transcribe_groq_chunked(self, file_path, language="tl", segment_seconds=180, quiet=False):
        if not self._can_use_ffmpeg():
            if not quiet:
                self._append_system_text("ffmpeg not found. Falling back to direct Groq upload.")
            return self._transcribe_groq_with_retry(file_path, language=language, quiet=quiet)

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
                if not quiet:
                    self._append_system_text("ffmpeg chunking failed. Falling back to direct Groq upload.")
                return self._transcribe_groq_with_retry(file_path, language=language, quiet=quiet)

            chunk_files = sorted(glob.glob(os.path.join(temp_dir, "chunk_*.mp3")))
            if not chunk_files:
                if not quiet:
                    self._append_system_text("No chunks produced. Falling back to direct Groq upload.")
                return self._transcribe_groq_with_retry(file_path, language=language, quiet=quiet)

            transcripts = []
            total = len(chunk_files)

            for idx, chunk_file in enumerate(chunk_files, start=1):
                if not quiet:
                    self._append_system_text(f"Groq chunk {idx}/{total}...")
                text = self._transcribe_groq_with_retry(chunk_file, language=language, quiet=quiet)
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

    def _process_file_mode(self):
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

        self._set_processing_state(True)
        self._append_system_text("Currently processing the file...")

        def worker():
            try:
                result = self._process_file_to_pdf(file_path)
                self.view.after(0, self.update_ui_text, f"[00:00] Transcript: {result['text']}")
                self.view.after(0, self._append_system_text, f"PDF generated: {result['pdf_path']}")
                self.view.after(
                    0,
                    self._append_system_text,
                    (
                        "Timing: "
                        f"transcription={self._format_secs(result['timings']['transcription'])}, "
                        f"summarization={self._format_secs(result['timings']['summarization'])}, "
                        f"pdf={self._format_secs(result['timings']['pdf'])}, "
                        f"total={self._format_secs(result['timings']['total'])}"
                    ),
                )
                print(f"[Debug] File processing timings: {result['timings']}")
            except Exception as e:
                self.view.after(0, self._append_system_text, f"File processing failed: {e}")
            finally:
                self.view.after(0, self._set_processing_state, False)

        threading.Thread(target=worker, daemon=True).start()

    def _process_file_to_pdf(self, file_path):
        t_start = time.perf_counter()

        t0 = time.perf_counter()
        text = self._transcribe_file(file_path, engine="groq", language="tl", quiet=True)
        transcribe_time = time.perf_counter() - t0
        if not text.strip():
            raise RuntimeError("No speech detected.")

        topic_segments = self._segment_topics(text)
        all_chunks_data, action_items = self._extract_actions(topic_segments)

        t1 = time.perf_counter()
        segment_summaries = []
        for chunk in all_chunks_data:
            seg_summary = self._summarize_chunk(chunk["chunk_text"], chunk["actions"], "groq")
            chunk["summary"] = seg_summary
            segment_summaries.append(seg_summary)
        summarize_time = time.perf_counter() - t1

        combined_summary = " ".join(segment_summaries) if segment_summaries else "No summary generated."

        unique_actions = []
        seen = set()
        for item in action_items:
            key = item.strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique_actions.append(item.strip())

        t2 = time.perf_counter()
        pdf_path = self.exporter.generate_pdf(
            content=text,
            action_items=unique_actions,
            summary=combined_summary,
            segment_summaries=segment_summaries,
            source_file=file_path,
        )
        pdf_time = time.perf_counter() - t2

        total_time = time.perf_counter() - t_start
        return {
            "text": text,
            "pdf_path": pdf_path,
            "timings": {
                "transcription": transcribe_time,
                "summarization": summarize_time,
                "pdf": pdf_time,
                "total": total_time,
            },
        }

    def process_file_path_for_pdf(self, file_path):
        """Public wrapper for other UI components (e.g. video manager)."""
        return self._process_file_to_pdf(file_path)

    def _start_training_mode(self):
        self._append_system_text("Train AI Mode is available in Main.py CLI flow.")
        self._append_system_text("Open Main.py and select mode 3 for dataset training.")

    def handle_start(self):
        """Starts direct live recording without prompting for other modes."""
        try:
            self.current_mode = "1"
            self.current_engine = "groq"
            self.current_summary_engine = "groq"

            self.audio.start_stream(live_transcription=True, live_transcriber=self._groq_live_transcribe)
            self.view.is_recording = True
            
            # 1. UI TRANSITION: Hide placeholder, Show transcript box
            self._ensure_transcript_ready()
            
            # 2. Update Button and Status
            self.view.status_indicator.configure(text="● RECORDING (GROQ)", text_color="#ff4b4b")
            self.view.btn_record.configure(image=self.view.stop_icon, command=self.handle_stop, border_color="#ff4b4b")
            
            # 3. Clear old text and show system message
            self.view.transcript_box.configure(state="normal") # Unlock for system message
            self.view.transcript_box.delete("0.0", "end")
            self.view.transcript_box.insert("end", "[System]: Mode=Live Meeting | Engine=groq\n")
            self.view.transcript_box.insert("end", "[System]: Listening...\n")
            self.view.transcript_box.configure(state="disabled") # Lock back up
            
            # 4. START THE TIMER LOOP
            self.update_timer_loop(time.time()) 
            
            # 5. Start visualizer and monitor thread
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
        """Pauses recording, then asks whether to generate PDF or continue recording."""
        self.audio.stop_stream(save=False)
        self.view.is_recording = False

        buffered_audio_exists = bool(getattr(self.audio, "all_audio_data", []))
        if not buffered_audio_exists:
            self.view.status_indicator.configure(text="● READY TO RECORD", text_color="#4a4d50")
            self.view.btn_record.configure(image=self.view.button_icon, command=self.handle_start, border_color="#00f2ff")
            self.view.timer_label.configure(text="00:00")
            self._append_system_text("Stopped.")
            return

        self.view.status_indicator.configure(text="● PAUSED", text_color="#d9a23b")
        decision = messagebox.askyesnocancel(
            "Generate PDF",
            "Recording paused. Generate PDF now?\n\nYes = Generate PDF\nNo = Discard recording\nCancel = Continue recording"
        )

        if decision is None:
            self.audio.start_stream(
                live_transcription=True,
                live_transcriber=self._groq_live_transcribe,
                reset_buffer=False,
            )
            self.view.is_recording = True
            self.view.status_indicator.configure(text="● RECORDING (GROQ)", text_color="#ff4b4b")
            self.view.btn_record.configure(image=self.view.stop_icon, command=self.handle_stop, border_color="#ff4b4b")
            self._append_system_text("Resumed recording.")
            self.update_timer_loop(time.time())
            self.view.animate_bars()
            threading.Thread(target=self.transcription_monitor, daemon=True).start()
            self.update_volume_loop()
            return

        self.view.status_indicator.configure(text="● READY TO RECORD", text_color="#4a4d50")
        self.view.btn_record.configure(image=self.view.button_icon, command=self.handle_start, border_color="#00f2ff")
        self.view.timer_label.configure(text="00:00")
        self._append_system_text("Stopped.")

        if decision is False:
            self.audio.clear_recording_buffer()
            self._append_system_text("Recording discarded. No PDF generated.")
            return

        saved_path = self.audio.save_recorded_audio()
        self.audio.clear_recording_buffer()
        if not saved_path:
            self._append_system_text("Could not save recording for PDF generation.")
            return

        self._set_processing_state(True, "● PROCESSING RECORDING")
        self._append_system_text("Currently processing the recording...")

        def worker():
            try:
                result = self._process_file_to_pdf(saved_path)
                self.view.after(0, self.update_ui_text, f"[00:00] Transcript: {result['text']}")
                self.view.after(0, self._append_system_text, f"PDF generated: {result['pdf_path']}")
                self.view.after(
                    0,
                    self._append_system_text,
                    (
                        "Timing: "
                        f"transcription={self._format_secs(result['timings']['transcription'])}, "
                        f"summarization={self._format_secs(result['timings']['summarization'])}, "
                        f"pdf={self._format_secs(result['timings']['pdf'])}, "
                        f"total={self._format_secs(result['timings']['total'])}"
                    ),
                )
                print(f"[Debug] Recording processing timings: {result['timings']}")
            except Exception as e:
                self.view.after(0, self._append_system_text, f"Groq transcription/PDF failed: {e}")
            finally:
                self.view.after(0, self._set_processing_state, False)

        threading.Thread(target=worker, daemon=True).start()

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