import customtkinter as ctk
import os
import time
import re
from tempfile import NamedTemporaryFile
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
        self.transcript_box.tag_config("speaker_1", foreground="#47d17d")
        self.transcript_box.tag_config("speaker_2", foreground="#ffd24a")
        self.transcript_box.tag_config("speaker_multi", foreground="#ffb86b")
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
            try:
                popup.grab_release()
            except Exception:
                pass
            try:
                popup.destroy()
            except Exception:
                pass

            if choice is True:
                _open_preview_and_close()
                return

            on_decision(choice)

        def _open_preview_and_close():
            try:
                popup.grab_release()
            except Exception:
                pass
            # Grab transcript text from the main UI's transcript box
            try:
                transcript_text = self.transcript_box.get("1.0", "end").strip()
            except Exception:
                transcript_text = ""
            popup.destroy()
            # Open the interactive PDF preview/editor
            self.open_pdf_preview(transcript_text)

        yes_btn = ctk.CTkButton(
            button_row,
            text="Generate PDF",
            width=140,
            height=34,
            fg_color="#1f6aa5",
            hover_color="#1a5280",
            font=("Inter", 13, "bold"),
            command=_open_preview_and_close,
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

    def open_pdf_preview(self, transcript_text, source_file=None, summary_text=None, topics=None):
        """Open a preview window that shows analytics and allows include/exclude of action items
        before exporting the final PDF. This is conservative: it will not invent missing details."""
        from app.export_service import ExportService, ReportContentFormatter

        preview = ctk.CTkToplevel(self)
        preview.title("PDF Preview & Export")
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        preview_width = min(1440, max(1160, int(screen_width * 0.94)))
        preview_height = min(900, max(760, int(screen_height * 0.88)))
        preview.geometry(f"{preview_width}x{preview_height}")
        preview.minsize(1160, 760)
        preview.resizable(True, True)
        preview.transient(self)
        preview.grab_set()

        preview.grid_rowconfigure(0, weight=1)
        preview.grid_rowconfigure(1, weight=0)
        preview.grid_columnconfigure(0, weight=0, minsize=360)
        preview.grid_columnconfigure(1, weight=1)

        content = ctk.CTkFrame(preview, fg_color="transparent")
        content.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=12, pady=(12, 6))
        content.grid_rowconfigure(0, weight=1)
        content.grid_columnconfigure(0, weight=0, minsize=360)
        content.grid_columnconfigure(1, weight=1)

        left_panel = ctk.CTkFrame(content, fg_color="transparent")
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=0)
        left_panel.grid_rowconfigure(0, weight=1)
        left_panel.grid_columnconfigure(0, weight=1)

        left = ctk.CTkScrollableFrame(left_panel, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew")

        right_panel = ctk.CTkFrame(content, fg_color="transparent")
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=0)
        right_panel.grid_rowconfigure(0, weight=1)
        right_panel.grid_columnconfigure(0, weight=1)

        right = ctk.CTkScrollableFrame(right_panel, fg_color="transparent")
        right.grid(row=0, column=0, sticky="nsew")

        # Analytics area (right)
        analytics_frame = ctk.CTkFrame(right, fg_color="transparent")
        analytics_frame.pack(fill="x", pady=(0, 8))

        title_label = ctk.CTkLabel(analytics_frame, text="Preview: Minutes of the Meeting", font=("Inter", 16, "bold"))
        title_label.pack(anchor="w")

        stats_label = ctk.CTkLabel(analytics_frame, text="Analyzing transcript...", font=("Inter", 11))
        stats_label.pack(anchor="w", pady=(6, 4))

        analytics_card = ctk.CTkFrame(analytics_frame, fg_color="#20242b", corner_radius=10)
        analytics_card.pack(fill="x", pady=(6, 8))

        chart_label = ctk.CTkLabel(analytics_card, text="Building charts...", text_color="#cdd6e1")
        chart_label.pack(padx=10, pady=(10, 2), anchor="w")

        # Two panels: left for breakdown donut, right for keyword bars
        charts_row = ctk.CTkFrame(analytics_card, fg_color="transparent")
        charts_row.pack(fill="x", padx=10, pady=(4, 8))

        breakdown_frame = ctk.CTkFrame(charts_row, fg_color="#1a1d21", width=160, height=140)
        breakdown_frame.pack(side="left", fill="both", expand=False, padx=(0, 8))
        breakdown_frame.pack_propagate(False)
        breakdown_label = ctk.CTkLabel(breakdown_frame, text="", text_color="#cdd6e1")
        breakdown_label.pack(fill="both", expand=True, padx=8, pady=8)

        keywords_frame = ctk.CTkFrame(charts_row, fg_color="#1a1d21", width=240, height=140)
        keywords_frame.pack(side="left", fill="both", expand=True)
        keywords_frame.pack_propagate(False)
        keywords_label = ctk.CTkLabel(keywords_frame, text="", text_color="#cdd6e1")
        keywords_label.pack(fill="both", expand=True, padx=8, pady=8)

        formatter = ReportContentFormatter()
        clean_transcript_text = formatter.clean_transcript_text(transcript_text)

        exporter = ExportService()
        extraction = exporter.extract_action_items_fast(clean_transcript_text)
        action_items = extraction.get("action_items", [])
        action_details = extraction.get("details", [])
        total_sentences = int(extraction.get("total_sentences", 0))
        clean_transcript_text = extraction.get("clean_transcript", clean_transcript_text)
        analytics = exporter.build_transcript_analytics(clean_transcript_text, action_items=action_items)
        summary_text = exporter.build_preview_summary(clean_transcript_text, action_items=action_items, summary_text=summary_text)
        topics = topics or exporter.build_topic_labels(clean_transcript_text)

        # Build separate analytics images (breakdown + keywords) if available
        bpath = None
        kpath = None
        try:
            bpath, kpath = exporter._build_separate_analytics_charts(analytics)
        except Exception:
            bpath, kpath = (None, None)

        chart_photo = None
        if bpath and os.path.exists(bpath):
            try:
                img = Image.open(bpath)
                photo = ctk.CTkImage(img, size=(160, 140))
                breakdown_label.configure(image=photo, text="")
                breakdown_label.image = photo
            except Exception:
                breakdown_label.configure(text="Breakdown chart unavailable")
        else:
            breakdown_label.configure(text="Breakdown chart unavailable")

        if kpath and os.path.exists(kpath):
            try:
                img2 = Image.open(kpath)
                photo2 = ctk.CTkImage(img2, size=(320, 140))
                keywords_label.configure(image=photo2, text="")
                keywords_label.image = photo2
            except Exception:
                keywords_label.configure(text="Keyword chart unavailable")
        else:
            keywords_label.configure(text="Keyword chart unavailable")

        info_count = max(0, total_sentences - len(action_items)) if total_sentences else None

        # Update stats_label
        if info_count is None:
            stats_text = f"Found {len(action_items)} suggested action item(s)."
        else:
            stats_text = f"Sentences: {total_sentences}  •  Action items: {len(action_items)}  •  Info: {info_count}"
        stats_label.configure(text=stats_text)

        # top_action_keywords is now a list of (word, count, weight)
        top_kw = analytics.get("top_action_keywords", []) or []
        if top_kw:
            keyword_lines = [f"{w} ({c}, {int(round(wt*100))}%)" for w, c, wt in top_kw]
            keyword_text = "  •  ".join(keyword_lines)
        else:
            keyword_text = "No recurring action keywords detected."

        keyword_label = ctk.CTkLabel(
            analytics_card,
            text=f"Top action keywords: {keyword_text}",
            wraplength=700,
            justify="left",
            text_color="#d6dbe3",
            font=("Inter", 11),
        )
        keyword_label.pack(padx=10, pady=(0, 10), anchor="w")

        why_frame = ctk.CTkFrame(right, fg_color="#20242b", corner_radius=10)
        why_frame.pack(fill="x", pady=(0, 8))
        why_title = ctk.CTkLabel(why_frame, text="Why This Was Flagged", font=("Inter", 13, "bold"), text_color="#eef3f8")
        why_title.pack(anchor="w", padx=10, pady=(10, 6))

        why_box = ctk.CTkTextbox(why_frame, height=96, wrap="word")
        why_box.pack(fill="x", padx=10, pady=(0, 10))
        why_box.insert("end", "These are conservative explanations based on the transcript text only.\n\n")
        if action_details:
            for idx, detail in enumerate(action_details, 1):
                item_text = str(detail.get("item", "")).strip()
                reason_text = str(detail.get("reason", "")).strip() or "Matched the action-item pattern."
                why_box.insert("end", f"{idx}. {item_text}\n")
                why_box.insert("end", f"   Why flagged: {reason_text}\n\n")
        else:
            why_box.insert("end", "No action items were detected.\n")
        why_box.configure(state="disabled")

        # Section controls: allow include/exclude and reordering of report sections
        sections_frame = ctk.CTkFrame(left, fg_color="transparent")
        sections_frame.pack(fill="x", pady=(0, 8))

        sections_label = ctk.CTkLabel(sections_frame, text="Report Sections (drag via Up/Down)", font=("Inter", 11, "bold"))
        sections_label.pack(anchor="w", pady=(0, 6))

        sections_list_frame = ctk.CTkFrame(sections_frame, fg_color="transparent")
        sections_list_frame.pack(fill="x")

        # Build an interactive list of sections
        default_sections = exporter.DEFAULT_SECTIONS
        section_rows = []

        def rebuild_sections_ui():
            for child in sections_list_frame.winfo_children():
                child.destroy()
            for idx, entry in enumerate(section_rows):
                row = ctk.CTkFrame(sections_list_frame, fg_color="transparent")
                row.pack(fill="x", pady=2)
                chk = ctk.CTkCheckBox(row, text=entry["label"], variable=entry["var"])
                chk.pack(side="left", anchor="w")
                up_btn = ctk.CTkButton(row, text="↑", width=28, height=22, fg_color="#2b3340", command=lambda i=idx: move_section_up(i))
                up_btn.pack(side="right", padx=(6, 0))
                down_btn = ctk.CTkButton(row, text="↓", width=28, height=22, fg_color="#2b3340", command=lambda i=idx: move_section_down(i))
                down_btn.pack(side="right", padx=(6, 0))

        def move_section_up(i):
            if i <= 0:
                return
            section_rows[i - 1], section_rows[i] = section_rows[i], section_rows[i - 1]
            rebuild_sections_ui()

        def move_section_down(i):
            if i >= len(section_rows) - 1:
                return
            section_rows[i + 1], section_rows[i] = section_rows[i], section_rows[i + 1]
            rebuild_sections_ui()

        for s in default_sections:
            v = ctk.BooleanVar(value=True)
            section_rows.append({"label": s, "var": v})

        rebuild_sections_ui()

        # Left side: list of action items with checkboxes
        list_frame = ctk.CTkFrame(left, fg_color="transparent")
        list_frame.pack(fill="both", expand=True)

        cb_vars = []
        if action_items:
            for idx, item in enumerate(action_items, 1):
                var = ctk.BooleanVar(value=True)
                cb = ctk.CTkCheckBox(list_frame, text=f"{idx}. {item}", variable=var, width=340)
                cb.pack(anchor="w", pady=6)
                cb_vars.append((var, item))
        else:
            none_label = ctk.CTkLabel(list_frame, text="No action items detected.", font=("Inter", 12), text_color="gray")
            none_label.pack(pady=20)

        # Preview text (how PDF will appear)
        preview_label = ctk.CTkLabel(right, text="PDF Preview", font=("Inter", 13, "bold"))
        preview_label.pack(anchor="w")

        preview_box = ctk.CTkTextbox(right, width=340, height=300, state="normal")
        preview_box.pack(fill="both", expand=True, pady=(6, 0))

        def render_preview():
            selected = [item for var, item in cb_vars if var.get()]
            preview_box.delete("1.0", "end")
            preview_box.insert("end", "Minutes of the Meeting\n\n")

            # Determine section order and inclusion from section_rows
            current_order = [r["label"] for r in section_rows]
            included = {r["label"] for r in section_rows if r["var"].get()}

            for sec in current_order:
                if sec not in included:
                    continue

                if sec == "Executive Overview":
                    preview_box.insert("end", "Executive Overview\n")
                    summary_paragraphs = formatter.split_summary_into_paragraphs(summary_text, None)
                    for paragraph in summary_paragraphs:
                        preview_box.insert("end", f"{paragraph}\n\n")

                elif sec == "Topics Discussed":
                    preview_box.insert("end", "Topics Discussed\n")
                    if topics:
                        for idx, topic in enumerate(topics, 1):
                            preview_box.insert("end", f"{idx}. {topic}\n")
                        preview_box.insert("end", "\n")
                    else:
                        preview_box.insert("end", "- No topics were identified.\n\n")

                elif sec == "Action Items":
                    preview_box.insert("end", "Analytics Overview\n")
                    preview_box.insert("end", f"- Sentences detected: {analytics.get('total_sentences', 0)}\n")
                    preview_box.insert("end", f"- Action items: {analytics.get('action_count', 0)}\n")
                    preview_box.insert("end", f"- Information sentences: {analytics.get('info_count', 0)}\n")
                    if analytics.get("top_action_keywords"):
                        kw_list = analytics.get("top_action_keywords", [])
                        preview_box.insert("end", "- Top action keywords: " + ", ".join(
                            f"{w} ({c}, {int(round(wt*100))}%)" for w, c, wt in kw_list
                        ) + "\n")

                    preview_box.insert("end", "Action Items:\n")
                    if selected:
                        for i, itm in enumerate(selected, 1):
                            preview_box.insert("end", f"{i}. {itm}\n")
                            explanation = formatter.build_action_explanation(itm)
                            why = formatter.build_action_flag_reason(itm)
                            if explanation:
                                preview_box.insert("end", f"   Task: {explanation}\n")
                            preview_box.insert("end", f"   Why flagged: {why}\n")
                    else:
                        preview_box.insert("end", "No action items selected.\n")
                    preview_box.insert("end", "\n")

                elif sec == "Full Transcript":
                    preview_box.insert("end", "Full Transcript:\n")
                    preview_box.insert("end", clean_transcript_text[:2000] + ("\n..." if len(clean_transcript_text) > 2000 else ""))

        render_preview()

        # Bottom buttons
        btn_row = ctk.CTkFrame(preview, fg_color="transparent")
        btn_row.grid(row=1, column=0, columnspan=2, sticky="ew", padx=12, pady=(6, 12))

        def do_export():
            selected = [item for var, item in cb_vars if var.get()]
            try:
                # Compute section order and inclusion
                section_order = [r["label"] for r in section_rows]
                include_sections = [r["label"] for r in section_rows if r["var"].get()]

                pdf_path = exporter.generate_pdf(
                    content=clean_transcript_text,
                    action_items=selected,
                    summary=summary_text,
                    source_file=source_file,
                    topics=topics,
                    section_order=section_order,
                    include_sections=include_sections,
                )
                try:
                    os.startfile(pdf_path)
                except Exception:
                    pass
                # Clean up any temporary analytics images that were generated for preview
                try:
                    if bpath and os.path.exists(bpath):
                        os.remove(bpath)
                except Exception:
                    pass
                try:
                    if kpath and os.path.exists(kpath):
                        os.remove(kpath)
                except Exception:
                    pass
                preview.destroy()
            except Exception as e:
                import traceback

                traceback.print_exc()

        export_btn = ctk.CTkButton(btn_row, text="Download PDF", width=160, height=36, fg_color="#1f6aa5", command=do_export)
        export_btn.pack(side="right", padx=(6, 0))

        refresh_btn = ctk.CTkButton(btn_row, text="Refresh Preview", width=140, height=36, fg_color="#3a3f46", command=render_preview)
        refresh_btn.pack(side="right", padx=(0, 6))