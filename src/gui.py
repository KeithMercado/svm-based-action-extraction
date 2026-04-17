import customtkinter as ctk
import os
import time
from PIL import Image
import random
from src.components.video_file_manager import VideoFileManager
from src.components.pdf_file_manager import PDFFileManager

class CompactActionApp(ctk.CTk):
    def __init__(self, start_cmd, stop_cmd, export_cmd, mic_toggle_cmd=None):
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
        self.configure(fg_color="#1d2027")
        self.attributes("-topmost", True)
        
        # Dimensions
        min_w, min_h = 380, 600
        self.minsize(min_w, min_h)
        self.resizable(True, True)

        # Calculate coordinates for centering the window
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width // 2) - (min_w // 2)
        y = (screen_height // 2) - (min_h // 2)

        # Set geometry with calculated x and y
        self.geometry(f"{min_w}x{min_h}+{x}+{y}")       

        self._create_widgets(start_cmd, stop_cmd, mic_toggle_cmd)

    def _create_widgets(self, start_cmd, stop_cmd, mic_toggle_cmd):
        # --- 1. HEADER (Status & Timer) ---
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.pack(fill="x", padx=20, pady=(15, 5))

        self.folder_icon_small = ctk.CTkImage(
            Image.open(os.path.join(self.assets_dir, "folder.png")),
            size=(15, 15),
        )
        self.btn_menu_small = ctk.CTkButton(
            self.header_frame,
            text="",
            image=self.folder_icon_small,
            width=30,
            height=30,
            corner_radius=15,
            fg_color="#23272f",
            hover_color="#323538",
            border_color="#23272f",
            border_width=1,
            command=self.toggle_pop_menu,
        )
        self.btn_menu_small.pack(side="left", padx=(0, 8))

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
            activate_scrollbars=False, 
            state="disabled"
        )

        # Cyan tag for timestamps
        self.transcript_box.tag_config("timestamp", foreground="#00f2ff")
        # System messages are shown in red
        self.transcript_box.tag_config("system", foreground="#ff5f5f")
        # mouse wheel scrolling for transcript box
        self.transcript_box.bind("<MouseWheel>", lambda event: self.transcript_box.yview_scroll(int(-1*(event.delta/120)), "units"))

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
        self.mic_icon = ctk.CTkImage(Image.open(os.path.join(self.assets_dir, "microphone.png")), size=(24, 24))
        self.muted_mic_icon = ctk.CTkImage(Image.open(os.path.join(self.assets_dir, "muted-mic.png")), size=(24, 24))

        self.btn_record = ctk.CTkButton(self.controls, text="", image=self.button_icon, width=60, height=60, 
                                        corner_radius=30, fg_color="#1e2d2e", border_color="#00f2ff", 
                                        border_width=2, hover_color="#2a3f40", command=start_cmd)
        self.btn_record.pack(side="left", expand=True, padx=(50, 12))

        self.btn_mic_toggle = ctk.CTkButton(
            self.controls,
            text="",
            image=self.mic_icon,
            width=60,
            height=60,
            corner_radius=30,
            fg_color="#1e5f3d",
            hover_color="#25774d",
            border_color="#4fd190",
            border_width=2,
            command=mic_toggle_cmd,
        )
        self.btn_mic_toggle.pack(side="left", expand=True, padx=(12, 50))

        self.update_mic_button(False)

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
        if self.pop_menu.winfo_manager():
            self.pop_menu.place_forget()
        else:
            x = self.btn_menu_small.winfo_rootx() - self.winfo_rootx()
            y = self.btn_menu_small.winfo_rooty() - self.winfo_rooty() + self.btn_menu_small.winfo_height() + 6
            self.pop_menu.place(x=x, y=y)

    def update_mic_button(self, is_muted):
        if is_muted:
            self.btn_mic_toggle.configure(
                image=self.muted_mic_icon,
                fg_color="#7a2323",
                hover_color="#8f2b2b",
                border_color="#ff8f8f",
            )
        else:
            self.btn_mic_toggle.configure(
                image=self.mic_icon,
                fg_color="#1e5f3d",
                hover_color="#25774d",
                border_color="#4fd190",
            )

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


    # Message Box when stopping recording, asking if they want to generate a PDF summary
    def show_recording_prompt(self, on_decision):
        popup = ctk.CTkToplevel(self)
        popup.title("Generate PDF")
        min_w, min_h = 340, 150
        popup.geometry(f"{min_w}x{min_h}")
        popup.minsize(min_w, min_h)
        popup.wm_minsize(min_w, min_h)
        popup.resizable(True, True)
        popup.configure(fg_color="#1d2027")
        popup.attributes("-topmost", True)
        popup.transient(self)
        popup.grab_set()

        self.update_idletasks()
        x = self.winfo_rootx() + (self.winfo_width() // 2) - 170
        y = self.winfo_rooty() + (self.winfo_height() // 2) - 90
        popup.geometry(f"{min_w}x{min_h}+{x}+{y}")

        # --- OUTER CONTAINER ---
        # expand=True allows this frame to center its children vertically
        outer = ctk.CTkFrame(popup, fg_color="#1d2027", border_width=0)
        outer.pack(fill="both", expand=True, padx=20, pady=20)

        # --- CONTENT AREA (Text) ---
        # We put labels in a sub-frame to keep them grouped in the center
        content_frame = ctk.CTkFrame(outer, fg_color="transparent")
        content_frame.pack(fill="both", expand=True)

        title = ctk.CTkLabel(
            content_frame,
            text="Recording paused",
            font=("Inter", 17, "bold"),
            text_color="#e1e1e1",
        )
        title.pack(pady=(0, 5))

        message = ctk.CTkLabel(
            content_frame,
            text="Would you like to generate a PDF summary of the\ncurrent transcription?",
            font=("Inter", 13),
            justify="center",
            text_color="#b7bcc4",
        )
        message.pack()

        # --- BUTTON AREA ---
        button_row = ctk.CTkFrame(outer, fg_color="transparent")
        button_row.pack(side="bottom", fill="x")

        def close_with(choice):
            try: popup.grab_release()
            except: pass
            popup.destroy()
            on_decision(choice)

        yes_btn = ctk.CTkButton(
            button_row,
            text="Generate PDF",
            width=140, # Adjusted width to fit 340 window with 20px side padding
            height=34,
            fg_color="#1f6aa5",
            hover_color="#1a5280",
            font=("Inter", 13, "bold"),
            command=lambda: close_with(True),
        )
        yes_btn.pack(side="left", expand=True, padx=(0, 5))

        no_btn = ctk.CTkButton(
            button_row,
            text="Discard",
            width=140,
            height=34,
            fg_color="#3a3f46",
            hover_color="#4a4f57",
            font=("Inter", 13, "bold"),
            command=lambda: close_with(False),
        )
        no_btn.pack(side="left", expand=True, padx=(5, 0))

        popup.protocol("WM_DELETE_WINDOW", lambda: close_with(None))
        popup.bind("<Escape>", lambda _event: close_with(None))