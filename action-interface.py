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
    view.btn_export.configure(command=logic.handle_export)
    
    view.mainloop()

if __name__ == "__main__":
    main()