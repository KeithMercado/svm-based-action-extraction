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

        # --- POP-UP MENU SECTION (Initially Hidden) ---
        self.pop_menu = ctk.CTkFrame(self.controls, fg_color="#333333", corner_radius=12)
        
        # Load Pop-up Icons
        self.clapper_icon = ctk.CTkImage(Image.open(os.path.join(self.assets_dir, "clapperboard.png")), size=(35, 35))
        self.pdf_icon = ctk.CTkImage(Image.open(os.path.join(self.assets_dir, "pdf-file.png")), size=(35, 35))

        self.btn_clapper = ctk.CTkButton(
            self.pop_menu, text="", image=self.clapper_icon, width=40, height=40,
            fg_color="transparent", hover_color="#444444", border_width=0
        )
        self.btn_clapper.pack(side="left", padx=10, pady=5)

        self.btn_pdf = ctk.CTkButton(
            self.pop_menu, text="", image=self.pdf_icon, width=40, height=40,
            fg_color="transparent", hover_color="#444444", border_width=0
        )
        self.btn_pdf.pack(side="left", padx=10, pady=5)
        # -----------------------------------------------

        # 4. Main Buttons Container
        self.btn_container = ctk.CTkFrame(self.controls, fg_color="transparent")
        self.btn_container.pack(expand=True)

        # Load Icons
        self.button_icon = ctk.CTkImage(Image.open(os.path.join(self.assets_dir, "button.png")), size=(50, 50))
        self.stop_icon = ctk.CTkImage(Image.open(os.path.join(self.assets_dir, "stop.png")), size=(50, 50)) # Added this
        self.folder_icon = ctk.CTkImage(Image.open(os.path.join(self.assets_dir, "folder.png")), size=(45, 45))

        # Record Button
        self.btn_record = ctk.CTkButton(
            self.btn_container, text="", image=self.button_icon, width=30, height=30,
            fg_color="transparent", hover_color="#2b2b2b", border_width=0, command=start_cmd
        )
        self.btn_record.pack(side="left", padx=20, pady=15)

        # Main Folder Button (Trigger for Pop-up)
        self.btn_folder_main = ctk.CTkButton(
            self.btn_container, text="", image=self.folder_icon, width=30, height=30,
            fg_color="transparent", hover_color="#2b2b2b", border_width=0,
            command=self.toggle_pop_menu 
        )
        self.btn_folder_main.pack(side="left", padx=20, pady=15)

    def toggle_pop_menu(self):
        """Toggles the visibility of the Clapper and PDF icons above the folder."""
        if self.pop_menu.winfo_manager():
            self.pop_menu.pack_forget()
        else:
            # Packs the menu specifically above the button container
            self.pop_menu.pack(before=self.btn_container, pady=(0, 5))

    def animate_bars(self):
        if self.is_recording:
            for i, bar in enumerate(self.bars):
                h = min(14, self.current_volume * random.uniform(0.8, 1.2)) if self.current_volume > 0.2 else 0
                self.anim_canvas.coords(bar, 50+(i*10), 15-h, 55+(i*10), 15+h)
            self.after(50, self.animate_bars)
        else:
            for i, bar in enumerate(self.bars):
                self.anim_canvas.coords(bar, 50+(i*10), 15, 55+(i*10), 15)