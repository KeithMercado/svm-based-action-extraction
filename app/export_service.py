# app/export_service.py

class ExportService:
    def __init__(self):
        # Academic standard margins (in mm)
        # hardcoded for now.
        self.margins = {
            "top": 25.4,    
            "bottom": 25.4,
            "left": 31.75,  
            "right": 25.4
        }

    def generate_pdf(self, content):
        """The logic handler calls THIS name."""
        print(f"Exporting content with margins: {self.margins}")
        # For now it just prints the content and margins
        return True