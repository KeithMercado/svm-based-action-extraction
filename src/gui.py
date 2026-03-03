import customtkinter as ctk
import os
import time
from PIL import Image
import random
from src.components.video_file_manager import VideoFileManager
from src.components.pdf_file_manager import PDFFileManager

class CompactActionApp(ctk.CTk):
    def __init__(self, start_cmd, stop_cmd, export_cmd):
        super().__init__()
        
        # State & Functionality
        self.is_recording = False
        self.current_volume = 0
        self.video_manager = None
        self.pdf_manager = None
        
        # Paths
        self.base_dir = os.path.dirname(os.path.dirname(__file__))
        self.assets_dir = os.path.join(self.base_dir, "assets")

        # Window Config
        self.title("EchoNotes")
        self.geometry("380x600")
        self.configure(fg_color="#1d2027")
        self.attributes("-topmost", True)

        self._create_widgets(start_cmd, stop_cmd)

    def _create_widgets(self, start_cmd, stop_cmd):
        # --- 1. HEADER (Status & Timer) ---
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.pack(fill="x", padx=20, pady=(15, 5))

        self.status_indicator = ctk.CTkLabel(
            self.header_frame, 
            text="● READY TO RECORD", 
            font=("Inter", 11, "bold"), 
            text_color="#4a4d50"
        )
        self.status_indicator.pack(side="left")

        self.timer_label = ctk.CTkLabel(
            self.header_frame, 
            text="00:00", 
            font=("JetBrains Mono", 14), 
            text_color="#00f2ff"
        )
        self.timer_label.pack(side="right")
        
        # LINE 1: Below Header
        ctk.CTkFrame(self, height=2, fg_color="#22242d").pack(fill="x")

        # --- 2. TRANSCRIPTION AREA ---
        self.transcript_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.transcript_frame.pack(fill="both", expand=True, padx=20, pady=10)

        self.placeholder_text = ctk.CTkLabel(
            self.transcript_frame, 
            text="Press record to begin transcription...", 
            font=("Inter", 13), 
            text_color="#3f4245", 
            wraplength=250
        )
        self.placeholder_text.place(relx=0.5, rely=0.5, anchor="center")

        self.transcript_box = ctk.CTkTextbox(
            self.transcript_frame, 
            fg_color="transparent", 
            font=("Inter", 13), 
            border_width=0, 
            text_color="#e1e1e1", 
            wrap="word", 
            spacing3=5,
            state="disabled" # Starts uneditable
        )
        # Cyan tag for timestamps
        self.transcript_box.tag_config("timestamp", foreground="#00f2ff")

        # --- 3. VISUALIZER & SEPARATORS ---
        # LINE 2: Above Visualizer
        ctk.CTkFrame(self, height=2, fg_color="#22242d").pack(fill="x", pady=(10, 0))

        self.visualizer_frame = ctk.CTkFrame(self, fg_color="transparent", height=50)
        self.visualizer_frame.pack(fill="x", pady=5)
        
        self.anim_canvas = ctk.CTkCanvas(
            self.visualizer_frame, 
            height=30, 
            width=300, 
            bg="#1d2027", 
            highlightthickness=0
        )
        self.anim_canvas.place(relx=0.5, rely=0.5, anchor="center")
        
        # Setup Visualizer Bars
        bar_width, gap = 4, 4
        start_x = (300 - (30 * (bar_width + gap))) / 2
        self.bars = [
            self.anim_canvas.create_rectangle(
                start_x + i*8, 15, start_x + i*8 + 4, 15, 
                fill="#00f2ff", outline=""
            ) for i in range(30)
        ]

        # LINE 3: Below Visualizer / Above Controls
        ctk.CTkFrame(self, height=2, fg_color="#22242d").pack(fill="x", pady=(0, 10))

        # --- 4. CONTROLS ---
        self.controls = ctk.CTkFrame(self, fg_color="transparent")
        self.controls.pack(fill="x", pady=(10, 30))

        # Icons (Renamed to match logic handler's expectations or vice versa)
        self.button_icon = ctk.CTkImage(Image.open(os.path.join(self.assets_dir, "button.png")), size=(25, 25))
        self.stop_icon = ctk.CTkImage(Image.open(os.path.join(self.assets_dir, "stop.png")), size=(25, 25))
        self.folder_icon = ctk.CTkImage(Image.open(os.path.join(self.assets_dir, "folder.png")), size=(22, 22))

        self.btn_record = ctk.CTkButton(self.controls, text="", image=self.button_icon, width=60, height=60, 
                                        corner_radius=30, fg_color="#1e2d2e", border_color="#00f2ff", 
                                        border_width=2, hover_color="#2a3f40", command=start_cmd)
        self.btn_record.pack(side="left", expand=True, padx=(40, 10))

        self.btn_folder_main = ctk.CTkButton(self.controls, text="", image=self.folder_icon, width=50, height=50, 
                                            corner_radius=25, fg_color="#23272f", hover_color="#323538", 
                                            command=self.toggle_pop_menu)
        self.btn_folder_main.pack(side="left", expand=True, padx=(10, 40))

        # --- 5. POP-UP MENU ---
        self.pop_menu = ctk.CTkFrame(self, fg_color="#252729", corner_radius=15, border_width=1, border_color="#323538")
        
        # Icons
        self.clapper_icon = ctk.CTkImage(Image.open(os.path.join(self.assets_dir, "clapperboard.png")), size=(30, 30))
        self.pdf_icon = ctk.CTkImage(Image.open(os.path.join(self.assets_dir, "pdf-file.png")), size=(30, 30))

        # Fix: Assigning to self.btn_clapper
        self.btn_clapper = ctk.CTkButton(
            self.pop_menu, text="", image=self.clapper_icon, 
            width=45, height=45, fg_color="transparent", 
            command=self.open_video_manager
        )
        self.btn_clapper.pack(side="left", padx=10, pady=5)

        # Fix: Assigning to self.btn_pdf
        self.btn_pdf = ctk.CTkButton(
            self.pop_menu, text="", image=self.pdf_icon, 
            width=45, height=45, fg_color="transparent", 
            command=self.open_pdf_manager
        )
        self.btn_pdf.pack(side="left", padx=10, pady=5)

    def toggle_pop_menu(self):
        if self.pop_menu.winfo_manager(): self.pop_menu.place_forget()
        else: self.pop_menu.place(relx=0.5, rely=0.75, anchor="center")

    def animate_bars(self):
        """Updated neon cyan visualizer animation with safety checks."""
        if self.is_recording:
            # Recalculate start_x to ensure bars stay centered
            bar_width = 4
            gap = 4
            start_x = (300 - (30 * (bar_width + gap))) / 2
            
            for i, bar in enumerate(self.bars):
                # Calculate height based on current volume
                # Ensure current_volume exists and is a number
                vol = getattr(self, 'current_volume', 0)
                h = min(14, vol * random.uniform(0.8, 1.2)) if vol > 0.1 else 2
                
                # Calculate static x positions based on index i
                x0 = start_x + i * (bar_width + gap)
                x1 = x0 + bar_width
                
                # Update coordinates: (x0, y0, x1, y1)
                # 15 is the vertical center of the 30px canvas
                self.anim_canvas.coords(bar, x0, 15 - h, x1, 15 + h)
                
            self.after(50, self.animate_bars)
        else:
            # Reset bars to flat line when not recording
            bar_width = 4
            gap = 4
            start_x = (300 - (30 * (bar_width + gap))) / 2
            for i, bar in enumerate(self.bars):
                x0 = start_x + i * (bar_width + gap)
                x1 = x0 + bar_width
                self.anim_canvas.coords(bar, x0, 15, x1, 15)

    def open_video_manager(self):
        if self.video_manager is None or not self.video_manager.winfo_exists():
            self.video_manager = VideoFileManager(self)
        self.video_manager.focus()

    def open_pdf_manager(self):
        if self.pdf_manager is None or not self.pdf_manager.winfo_exists():
            self.pdf_manager = PDFFileManager(self)
        self.pdf_manager.focus()