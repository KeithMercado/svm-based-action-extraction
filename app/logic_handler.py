import threading
import queue
from app.export_service import ExportService

class AppLogic:
    def __init__(self, view, audio):
        self.view = view
        self.audio = audio
        self.exporter = ExportService()

    def handle_start(self):
        try:
            self.audio.start_stream()
            self.view.is_recording = True
            self.view.btn_record.configure(image=self.view.stop_icon, command=self.handle_stop)
            self.view.transcript_box.insert("end", "\n[System]: Listening...\n")
            self.view.animate_bars()

            # This starts the thread that watches for new text from the AI
            threading.Thread(target=self.transcription_monitor, daemon=True).start()
            
            self.update_volume_loop()
        except Exception as e:
            self.view.transcript_box.insert("end", f"\n[Error]: {e}\n")

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
        self.audio.stop_stream()
        self.view.is_recording = False
        self.view.btn_record.configure(image=self.view.button_icon, command=self.handle_start)
        
        # FIX 2: Use btn_folder_main because btn_export was renamed in your GUI
        self.view.btn_folder_main.configure(state="normal") 
        self.view.transcript_box.insert("end", "[System]: Stopped.\n")

    # FIX 3: Add 'export_type' to accept the string from the lambda in main.py
    def handle_export(self, export_type):
        content = self.view.transcript_box.get("0.0", "end")
        
        if export_type == "pdf":
            print("Opening video folder...")
            self.view.transcript_box.insert("end", "[System]: Opening PDF File Folder...\n")
            success = self.exporter.generate_pdf(content)
            if success:
                self.view.transcript_box.insert("end", "[System]: PDF Export Successful.\n")
        
        elif export_type == "video":
            # Logic for the clapperboard icon
            print("Opening video folder...")
            self.view.transcript_box.insert("end", "[System]: Opening Media Folder...\n")
            
        # Hide the pop-up menu after clicking either button
        self.view.toggle_pop_menu()

    def update_volume_loop(self):
        if self.view.is_recording:
            self.view.current_volume = self.audio.current_volume #
            self.view.after(50, self.update_volume_loop) #