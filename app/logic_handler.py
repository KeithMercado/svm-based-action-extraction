from .export_service import ExportService

class AppLogic:
    def __init__(self, view, audio):
        self.view = view
        self.audio = audio
        self.exporter = ExportService() # Initialize the export specialist

    def handle_start(self):
        try:
            self.audio.start_stream()
            self.view.is_recording = True
            self.view.btn_record.configure(image=self.view.stop_icon, command=self.handle_stop)
            self.view.transcript_box.insert("end", "\n[System]: Listening...\n")
            self.view.animate_bars()
            self.update_volume_loop() # Start the sync loop
        except Exception as e:
            self.view.transcript_box.insert("end", f"\n[Error]: {e}\n")

    def handle_stop(self):
        self.audio.stop_stream()
        self.view.is_recording = False
        self.view.btn_record.configure(image=self.view.button_icon, command=self.handle_start)
        self.view.btn_export.configure(state="normal")
        self.view.transcript_box.insert("end", "[System]: Stopped.\n")

    def handle_export(self):
        # Pull text from the GUI
        content = self.view.transcript_box.get("0.0", "end")
        # Delegate the work to the ExportService
        success = self.exporter.generate_pdf(content)
        if success:
            self.view.transcript_box.insert("end", "[System]: Export Successful.\n")

    def update_volume_loop(self):
        """Placed here to keep the GUI class 'dumb' and the Logic class 'aware'."""
        if self.view.is_recording:
            self.view.current_volume = self.audio.current_volume
            self.view.after(50, self.update_volume_loop)