import customtkinter as ctk
import os
from PIL import Image
import random

class CompactActionApp(ctk.CTk):
    def __init__(self, start_cmd, stop_cmd, export_cmd):
        super().__init__()
        self.is_recording = False
        self.current_volume = 0
        
        # Paths
        self.base_dir = os.path.dirname(os.path.dirname(__file__))
        self.assets_dir = os.path.join(self.base_dir, "assets")

        # Window Config
        self.title("MoM SVM Based Action Item Extractor") # change to something more catchy and less technical for the final version, maybe "MoM Genie" or "Taglish Transcriber"
        self.geometry("300x420")
        self.attributes("-topmost", True)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._create_widgets(start_cmd, stop_cmd, export_cmd)

    def _create_widgets(self, start_cmd, stop_cmd, export_cmd):
        # Textbox
        self.transcript_box = ctk.CTkTextbox(self, font=("Inter", 13), wrap="word", border_width=1)
        self.transcript_box.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.transcript_box.insert("0.0", "Ready to record...\n" + "-"*30 + "\n")

        # Controls Frame
        self.controls = ctk.CTkFrame(self, height=150, corner_radius=10)
        self.controls.grid(row=1, column=0, sticky="ew", padx=10, pady=10)

        # Animation Canvas
        self.anim_canvas = ctk.CTkCanvas(self.controls, height=30, width=200, bg="#2b2b2b", highlightthickness=0)
        self.anim_canvas.pack(pady=(5, 0))
        self.bars = [self.anim_canvas.create_rectangle(50+(i*10), 15, 55+(i*10), 15, fill="#1d7c00", outline="") for i in range(10)]

        # Buttons Container
        self.btn_container = ctk.CTkFrame(self.controls, fg_color="transparent")
        self.btn_container.pack(expand=True)

        # Load Icons
        self.button_icon = ctk.CTkImage(Image.open(os.path.join(self.assets_dir, "button.png")), size=(50, 50))
        self.stop_icon = ctk.CTkImage(Image.open(os.path.join(self.assets_dir, "stop.png")), size=(50, 50))
        self.printer_icon = ctk.CTkImage(Image.open(os.path.join(self.assets_dir, "printer.png")), size=(45, 45))

        # Record Button - Transparent Style
        self.btn_record = ctk.CTkButton(
            self.btn_container, 
            text="", 
            image=self.button_icon, 
            width=30, 
            height=30,
            fg_color="transparent",      # No background color
            hover_color="#2b2b2b",       # Subtle highlight when hovering
            border_width=0,              # Remove the border
            command=start_cmd
        )
        self.btn_record.pack(side="left", padx=20, pady=15)

        # Export Button - Transparent Style
        # TODO: Update the icon to a folder or export symbol, and consider adding a functional button for export options in the future.
        # change to folder icon
        # 2 folder inside, pdf and audio file for reference
        # separate with date and time for uniqueness and based on file category.
        self.btn_export = ctk.CTkButton(
            self.btn_container, 
            text="", 
            image=self.printer_icon, 
            width=30, 
            height=30,
            fg_color="transparent",      # No background color
            hover_color="#2b2b2b",       # Subtle highlight when hovering
            border_width=0,              # Remove the border
            state="disabled", 
            command=export_cmd
        )
        self.btn_export.pack(side="left", padx=20, pady=15)

    def animate_bars(self):
        if self.is_recording:
            for i, bar in enumerate(self.bars):
                h = min(14, self.current_volume * random.uniform(0.8, 1.2)) if self.current_volume > 0.2 else 0
                self.anim_canvas.coords(bar, 50+(i*10), 15-h, 55+(i*10), 15+h)
            self.after(50, self.animate_bars)
        else:
            for i, bar in enumerate(self.bars):
                self.anim_canvas.coords(bar, 50+(i*10), 15, 55+(i*10), 15)