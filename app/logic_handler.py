import threading
import queue
from app.export_service import ExportService

class AppLogic:
    def __init__(self, view, audio):
        self.view = view
        self.audio = audio
        self.exporter = ExportService() # Initialize the export specialist

    def handle_start(self):
        try:
            self.audio.start_stream() #
            self.view.is_recording = True #
            self.view.btn_record.configure(image=self.view.stop_icon, command=self.handle_stop) #
            self.view.transcript_box.insert("end", "\n[System]: Listening...\n") #
            
            # Start the monitor thread
            threading.Thread(target=self.transcription_monitor, daemon=True).start() #
            
            self.view.animate_bars() #
            self.update_volume_loop() # Start the sync loop
        except Exception as e:
            self.view.transcript_box.insert("end", f"\n[Error]: {e}\n") #

    def transcription_monitor(self):
        """Updates the GUI whenever new text arrives in the audio handler."""
        print("[Debug]: UI Monitor Started.")
        while self.view.is_recording:
            try:
                # Check the audio handler's text queue
                new_text = self.audio.text_queue.get(timeout=0.1) #
                print(f"[Debug]: UI Received: {new_text}")
                # Safely update the GUI from a background thread
                self.view.after(0, self.update_ui_text, new_text) #
            except queue.Empty:
                continue

    def update_ui_text(self, text):
        self.view.transcript_box.insert("end", f"{text}\n") #
        self.view.transcript_box.see("end") # Auto-scroll to bottom

    def handle_stop(self):
        self.audio.stop_stream() #
        self.view.is_recording = False #
        self.view.btn_record.configure(image=self.view.button_icon, command=self.handle_start) #
        self.view.btn_export.configure(state="normal") #
        self.view.transcript_box.insert("end", "[System]: Stopped.\n") #

    def handle_export(self):
        """Fixed the missing attribute by defining it here."""
        # Pull text from the GUI
        content = self.view.transcript_box.get("0.0", "end") #
        # Delegate the work to the ExportService
        success = self.exporter.generate_pdf(content) #
        if success:
            self.view.transcript_box.insert("end", "[System]: Export Successful.\n") #

    def update_volume_loop(self):
        """Updates the volume bars in the UI."""
        if self.view.is_recording:
            self.view.current_volume = self.audio.current_volume #
            self.view.after(50, self.update_volume_loop) #