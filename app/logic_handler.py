from .export_service import ExportService

class AppLogic:
    def __init__(self, view, audio):
        self.view = view
        self.audio = audio
        self.exporter = ExportService()

    def handle_start(self):
        try:
            self.audio.start_stream()
            self.view.is_recording = True
            # This line works now because we added stop_icon to gui.py
            self.view.btn_record.configure(image=self.view.stop_icon, command=self.handle_stop)
            self.view.transcript_box.insert("end", "\n[System]: Listening...\n")
            self.view.animate_bars()
            self.update_volume_loop()
        except Exception as e:
            self.view.transcript_box.insert("end", f"\n[Error]: {e}\n")

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
            self.view.current_volume = self.audio.current_volume
            self.view.after(50, self.update_volume_loop)