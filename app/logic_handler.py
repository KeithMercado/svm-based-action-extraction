import threading
import queue
import time
import os
import re
import tempfile
import subprocess
import glob
import pickle
import wave
from tkinter import simpledialog, filedialog

from faster_whisper import WhisperModel
import scipy.io.wavfile as wav
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from app.export_service import ExportService
from core.segmenter import Segmenter
from src.components.pdf_layout_editor import PDFLayoutEditorDialog

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
        self.segmenter = Segmenter(chunk_size=5, max_tokens=1024)
        self.live_transcribe_prompt = (
            "This is a Taglish meeting transcript involving technical tasks and action items."
        )
        self._last_ui_speaker_label = None

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

    def _looks_like_information_override(self, sentence):
        normalized = re.sub(r"\s+", " ", str(sentence or "")).strip().lower().rstrip(".?!,")
        if not normalized:
            return False

        info_patterns = [
            r"^thank you(?: so much)?$",
            r"^thanks(?: everyone| all)?$",
            r"^appreciate it$",
            r"^noted$",
            r"^received(?: with thanks)?$",
            r"^copy that$",
            r"^got it$",
            r"^roger that$",
            r"^understood$",
            r"^acknowledged$",
            r"^good (?:morning|afternoon|evening)(?: everyone)?$",
            r"^welcome(?: everyone)?$",
            r"^fyi$",
            r"^for your information$",
            r"^no further updates from the team$",
            r"^nothing else to add$",
        ]

        task_markers = [
            r"\bplease\b",
            r"\bneed(s)? to\b",
            r"\bshould\b",
            r"\bmust\b",
            r"\bwill\b",
            r"\bcan you\b",
            r"\bcould you\b",
            r"\bfollow up\b",
            r"\bsend\b",
            r"\bprepare\b",
            r"\breview\b",
            r"\bupdate\b",
            r"\bcomplete\b",
        ]

        if len(normalized.split()) > 12:
            return False

        if any(re.search(pattern, normalized) for pattern in task_markers):
            return False

        return any(re.fullmatch(pattern, normalized) for pattern in info_patterns)

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

    def _summarize_chunk(self, text, actions, engine):
        if engine == "groq":
            if summarize_with_groq is None:
                raise RuntimeError(f"Groq summarizer unavailable: {GROQ_SUMMARY_IMPORT_ERROR}")
            return summarize_with_groq(text, actions)
        return self._summarize_with_local_bart(text, actions)

    def _get_media_duration_seconds(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".wav":
            try:
                with wave.open(file_path, "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    if rate > 0:
                        return float(frames) / float(rate)
            except Exception:
                pass

        try:
            probe_cmd = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                file_path,
            ]
            result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10, check=False)
            if result.returncode == 0:
                value = (result.stdout or "").strip()
                if value:
                    return float(value)
        except Exception:
            pass

        return None

    def _ensure_transcript_ready(self):
        self.view.placeholder_text.place_forget()
        if not self.view.transcript_box.winfo_manager():
            self.view.transcript_box.pack(fill="both", expand=True, padx=5, pady=5)

    def _append_system_text(self, text):
        self._ensure_transcript_ready()
        self.view.transcript_box.configure(state="normal")
        self.view.transcript_box.insert("end", "[System]:", "system")
        self.view.transcript_box.insert("end", f" {text}\n")
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

    def _transcribe_groq_with_retry(
        self,
        file_path,
        language="tl",
        retries=3,
        quiet=False,
        initial_prompt=None,
        auto_fallback_to_local=True,
    ):
        if transcribe_with_groq is None:
            raise RuntimeError(f"Groq integration unavailable: {GROQ_IMPORT_ERROR}")

        last_error = None
        is_rate_limit_error = False
        for attempt in range(1, retries + 1):
            try:
                return transcribe_with_groq(
                    file_path,
                    language=language,
                    initial_prompt=initial_prompt,
                )
            except Exception as e:
                last_error = e
                msg = str(e).lower()
                if "429" in msg or "rate_limit" in msg or "rate limit" in msg:
                    is_rate_limit_error = True

                if attempt < retries and not quiet:
                    wait_s = self._recommended_wait_seconds(e, attempt)
                    self._append_system_text(
                        f"Groq attempt {attempt}/{retries} failed: {e}. Retrying in {wait_s}s..."
                    )
                    time.sleep(wait_s)

        if is_rate_limit_error and auto_fallback_to_local:
            if not quiet:
                self._append_system_text(
                    "Groq rate limit exhausted. Falling back to local Whisper (slower but no quota needed)..."
                )
            try:
                if self.file_transcriber is None:
                    self.file_transcriber = WhisperModel("medium", device="cpu", compute_type="int8")
                segments, _ = self.file_transcriber.transcribe(file_path, language=language)
                segments = list(segments)
                if segments:
                    if not quiet:
                        self._append_system_text(f"Local fallback transcription completed.")
                    return " ".join([s.text for s in segments])
            except Exception as fallback_error:
                if not quiet:
                    self._append_system_text(f"Local fallback also failed: {fallback_error}")

        raise RuntimeError(f"Groq failed after {retries} attempts: {last_error}")

    def _recommended_wait_seconds(self, error, attempt):
        """Derive retry wait from Groq rate-limit messages, with safe fallback."""
        msg = str(error)
        lowered = msg.lower()

        match = re.search(r"try again in\s*(?:(\d+)m)?\s*(\d+(?:\.\d+)?)s", lowered)
        if match:
            minutes = int(match.group(1) or 0)
            seconds = float(match.group(2) or 0)
            return max(1, int(minutes * 60 + seconds + 1))

        if "429" in lowered or "rate_limit" in lowered or "rate limit" in lowered:
            return min(120, 8 * attempt)

        return min(30, 2 * attempt)

    def _transcribe_groq_chunked(self, file_path, language="tl", segment_seconds=120, quiet=False):
        segment_seconds = int(os.getenv("GROQ_SEGMENT_SECONDS", str(segment_seconds)))
        segment_seconds = max(30, min(segment_seconds, 300))

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
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_path = tmp.name

        try:
            wav.write(temp_path, sample_rate, audio_buffer)
            return self._transcribe_groq_with_retry(
                temp_path,
                language="tl",
                retries=3,
                quiet=True,
                initial_prompt=self.live_transcribe_prompt,
            )
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
                if result.get("cancelled"):
                    self.view.after(0, self._append_system_text, "PDF generation cancelled.")
                    return
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

    def _extract_topic_labels_from_metadata(self):
        """Extract Llama-generated topical descriptions from Phase 2 segmentation metadata."""
        metadata = self.segmenter.get_segment_metadata()
        topics = []
        for segment in metadata:
            topical_desc = segment.get("topical_description", "").strip()
            if topical_desc:
                # Truncate if too long
                topic = topical_desc[:80] + "..." if len(topical_desc) > 80 else topical_desc
                topics.append(topic)
        
        return topics if topics else None

    def _prompt_pdf_layout_config(self, action_items):
        """Open a modal layout editor on the UI thread and return export preferences."""
        done_event = threading.Event()
        result_holder = {"result": None}

        def open_dialog():
            try:
                dialog = PDFLayoutEditorDialog(self.view, action_items=action_items)
                result_holder["result"] = dialog.show_modal()
            finally:
                done_event.set()

        self.view.after(0, open_dialog)
        done_event.wait()
        return result_holder["result"]

    def _process_file_to_pdf(self, file_path):
        t_start = time.perf_counter()
        media_duration_seconds = self._get_media_duration_seconds(file_path)

        t0 = time.perf_counter()
        text = self._transcribe_file(file_path, engine="groq", language="tl", quiet=True)
        transcribe_time = time.perf_counter() - t0
        if not text.strip():
            raise RuntimeError("No speech detected.")

        # Phase 2: Use Segmenter for semantic segmentation with topical descriptions
        self.segmenter.segment_text(text)
        segment_metadata = self.segmenter.get_segment_metadata()
        
        # Extract actions from segment raw text
        action_items = []
        for segment in segment_metadata:
            segment_text = segment.get("raw_text", "")
            # Split by sentences and check each for action classification
            sentences = [s.strip() for s in segment_text.split(".") if s.strip()]
            for sentence in sentences:
                if sentence:
                    if self._looks_like_information_override(sentence):
                        prediction = 0
                    else:
                        prediction = self.classifier.predict(self._get_features(sentence))[0]
                    if prediction == 1:
                        action_items.append(sentence)
        
        # Extract topical descriptions for PDF
        topic_labels = self._extract_topic_labels_from_metadata()

        t1 = time.perf_counter()
        combined_summary = self._summarize_chunk(text, action_items, "groq")
        summarize_time = time.perf_counter() - t1

        unique_actions = []
        seen = set()
        for item in action_items:
            key = item.strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique_actions.append(item.strip())

        llama_action_items = self.exporter.formatter.extract_llama_action_items(combined_summary)
        editor_action_items = []
        seen_editor = set()
        for item in unique_actions + llama_action_items:
            key = re.sub(r"\s+", " ", item.strip().lower())
            if key and key not in seen_editor:
                seen_editor.add(key)
                editor_action_items.append(item.strip())

        layout_config = self._prompt_pdf_layout_config(editor_action_items)
        if layout_config is None:
            return {
                "cancelled": True,
                "text": text,
                "timings": {
                    "transcription": transcribe_time,
                    "summarization": summarize_time,
                    "pdf": 0.0,
                    "total": time.perf_counter() - t_start,
                },
            }

        selected_actions = layout_config.get("selected_action_items", unique_actions)
        section_order = layout_config.get("section_order")
        include_sections = layout_config.get("include_sections")

        t2 = time.perf_counter()
        pdf_path = self.exporter.generate_pdf(
            content=text,
            action_items=selected_actions,
            summary=combined_summary,
            duration_seconds=media_duration_seconds,
            source_file=file_path,
            topics=topic_labels,
            section_order=section_order,
            include_sections=include_sections,
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

    # The following methods are for the live recording mode, which can be started directly or resumed after a pause. 
    # The stop handler will prompt the user to either generate a PDF from the buffered audio or discard it and reset.
    def _start_live_recording(self):
        self.audio.start_stream(live_transcription=True, live_transcriber=self._groq_live_transcribe)
        self.view.is_recording = True
        self._last_ui_speaker_label = None
        self._ensure_transcript_ready()
        self.view.status_indicator.configure(text="● RECORDING", text_color="#ff4b4b")
        self.view.btn_record.configure(image=self.view.stop_icon, command=self.handle_stop, border_color="#ff4b4b")
        self.view.transcript_box.configure(state="normal")
        self.view.transcript_box.delete("0.0", "end")
        self.view.transcript_box.configure(state="disabled")
        self._append_system_text("Live Meeting")
        self._append_system_text("Listening...")
        self.update_timer_loop(self.audio.start_time)
        self.view.animate_bars()
        threading.Thread(target=self.transcription_monitor, daemon=True).start()
        self.update_volume_loop()

    def _resume_live_recording(self):
        self.audio.start_stream(
            live_transcription=True,
            live_transcriber=self._groq_live_transcribe,
            reset_buffer=False,
            continue_timing=True,
        )
        self.view.is_recording = True
        self.view.status_indicator.configure(text="● RECORDING", text_color="#ff4b4b")
        self.view.btn_record.configure(image=self.view.stop_icon, command=self.handle_stop, border_color="#ff4b4b")
        self._append_system_text("Resumed recording.")
        self.update_timer_loop(self.audio.start_time)
        self.view.animate_bars()
        threading.Thread(target=self.transcription_monitor, daemon=True).start()
        self.update_volume_loop()

    def _handle_recording_prompt_decision(self, decision):
        if decision is None:
            self._resume_live_recording()
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
                if result.get("cancelled"):
                    self.view.after(0, self._append_system_text, "PDF generation cancelled.")
                    return
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

    def _start_training_mode(self):
        self._append_system_text("Train AI Mode is available in Main.py CLI flow.")
        self._append_system_text("Open Main.py and select mode 3 for dataset training.")

    def handle_start(self):
        """Starts direct live recording without prompting for other modes."""
        try:
            self.current_mode = "1"
            self.current_engine = "groq"
            self.current_summary_engine = "groq"
            self._start_live_recording()
            
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
        if "]" not in text:
            self.view.transcript_box.insert("end", f"{text}\n")
        else:
            parts = text.split("]", 1)
            timestamp = parts[0] + "]"
            content = parts[1].strip()

            speaker_match = re.match(r"^(Speaker\s+\d+|Multiple Speakers)\s*:\s*(.+)$", content)
            if speaker_match:
                speaker_label = speaker_match.group(1).strip()
                transcript_text = speaker_match.group(2).strip()

                # Emit speaker header only when turn changes.
                if speaker_label != self._last_ui_speaker_label:
                    speaker_tag = "speaker_multi"
                    if speaker_label == "Speaker 1":
                        speaker_tag = "speaker_1"
                    elif speaker_label == "Speaker 2":
                        speaker_tag = "speaker_2"
                    self.view.transcript_box.insert("end", f"[{speaker_label}]\n", speaker_tag)
                    self._last_ui_speaker_label = speaker_label

                # Always emit the timestamped transcript line.
                self.view.transcript_box.insert("end", timestamp, "timestamp")
                self.view.transcript_box.insert("end", f" {transcript_text}\n")
            else:
                self._last_ui_speaker_label = None
                self.view.transcript_box.insert("end", timestamp, "timestamp")
                self.view.transcript_box.insert("end", f" {content}\n")
            
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
        self.view.show_recording_prompt(self._handle_recording_prompt_decision)

    def handle_export(self, export_type):
        """Handles PDF generation and Media folder access."""
        content = self.view.transcript_box.get("0.0", "end")

        if export_type == "pdf":
            self._append_system_text("Exporting PDF...")
            success = self.exporter.generate_pdf(content)
            if success:
                self._append_system_text("PDF Export Successful.")
        
        elif export_type == "video":
            self._append_system_text("Opening Media Folder...")
            # You can add os.startfile("output/videos") here if desired
        
        # Hide the pop-up menu
        self.view.toggle_pop_menu()

    def handle_toggle_mic(self):
        """Toggles the microphone mute state."""
        new_mute_state = not self.audio.mic_is_muted
        self.audio.toggle_mic(new_mute_state)
        self.view.update_mic_button(new_mute_state)

    def update_volume_loop(self):
        """Syncs the GUI visualizer data with the AudioHandler volume levels."""
        if self.view.is_recording:
            vol = getattr(self.audio, 'current_volume', 0)
            self.view.current_volume = vol if vol is not None else 0
            self.view.after(50, self.update_volume_loop)