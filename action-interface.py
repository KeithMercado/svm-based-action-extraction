from src.gui import CompactActionApp
from app.audio_handler import AudioHandler
from app.logic_handler import AppLogic

def main():
    audio = AudioHandler()
    # a placeholder view with no commands yet, since the Logic will fill those in after it’s created
    view = CompactActionApp(start_cmd=None, stop_cmd=None, export_cmd=None)
    
    logic = AppLogic(view, audio)
    
    # Final wiring
    view.btn_record.configure(command=logic.handle_start)
    
    # These match the "Pop-up" buttons we created earlier
    view.btn_clapper.configure(command=lambda: logic.handle_export("video"))
    view.btn_pdf.configure(command=lambda: logic.handle_export("pdf"))
    
    # Ensure the main window stays open
    view.mainloop()

if __name__ == "__main__":
    main()
    
    
    # TODO: add a file explorer for the clapperboard icon to open the media folder, and also add a "Save Transcript" button that saves the transcript as a text file.