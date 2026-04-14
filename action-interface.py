from src.gui import CompactActionApp
from app.audio_handler import AudioHandler
from app.logic_handler import AppLogic

def main():
    audio = AudioHandler()
    # a placeholder view with no commands yet, since the Logic will fill those in after it’s created
    view = CompactActionApp(start_cmd=None, stop_cmd=None, export_cmd=None)
    
    logic = AppLogic(view, audio)
    view.logic = logic
    
    # Final wiring
    view.btn_record.configure(command=logic.handle_start)
    
    # These match the "Pop-up" buttons we created earlier
    view.btn_clapper.configure(command=view.open_video_manager)
    view.btn_pdf.configure(command=view.open_pdf_manager)
    
    # Ensure the main window stays open
    view.mainloop()

if __name__ == "__main__":
    main()