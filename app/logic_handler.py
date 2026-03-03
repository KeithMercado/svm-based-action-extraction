import threading
import queue
import time
from app.export_service import ExportService

# we can add pickle file loading here centralized for the whole app, 
# so if we want to load a saved SVM model or something later we can do it here and pass it to the relevant components (pickle.load(MODEL_PATH) for example)

class AppLogic:
    def __init__(self, view, audio):
        self.view = view
        self.audio = audio
        self.exporter = ExportService()

    def handle_start(self):
        """Initializes audio stream, timer, and UI for recording."""
        try:
            # 1. Start the actual AI/Audio engine
            self.audio.start_stream()
            self.view.is_recording = True
            
            # 2. UI TRANSITION: Hide placeholder, Show transcript box
            self.view.placeholder_text.place_forget() 
            self.view.transcript_box.pack(fill="both", expand=True, padx=5, pady=5)
            
            # 3. Update Button and Status
            self.view.status_indicator.configure(text="● RECORDING", text_color="#ff4b4b")
            self.view.btn_record.configure(image=self.view.stop_icon, command=self.handle_stop, border_color="#ff4b4b")
            
            # 4. Clear old text and show system message
            self.view.transcript_box.configure(state="normal") # Unlock for system message
            self.view.transcript_box.delete("0.0", "end")
            self.view.transcript_box.insert("end", "[System]: Listening...\n")
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
        self.audio.stop_stream()
        self.view.is_recording = False
        
        # 2. Reset UI Status and Button
        self.view.status_indicator.configure(text="● READY TO RECORD", text_color="#4a4d50")
        self.view.btn_record.configure(image=self.view.button_icon, command=self.handle_start, border_color="#00f2ff")
        
        # 3. Insert "Stopped" message safely
        self.view.transcript_box.configure(state="normal")
        self.view.transcript_box.insert("end", "[System]: Stopped.\n")
        self.view.transcript_box.configure(state="disabled")
        
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