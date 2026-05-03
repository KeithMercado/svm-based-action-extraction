import os
import customtkinter as ctk
from PIL import Image

from app.export_service import ExportService, ReportContentFormatter


class PDFPreviewWindow(ctk.CTkToplevel):
    """Preview and export window for meeting minutes PDFs with enhanced analytics."""

    def __init__(self, parent, transcript_text, source_file=None, summary_text=None, topics=None, duration_seconds=None):
        super().__init__(parent)

        self.source_file = source_file
        self.duration_seconds = duration_seconds
        self.formatter = ReportContentFormatter()
        self.exporter = ExportService()
        
        self.clean_transcript_text = self.formatter.clean_transcript_text(transcript_text)
        self.extraction = self.exporter.extract_action_items_fast(self.clean_transcript_text)
        self.action_items = self.extraction.get("action_items", [])
        self.action_details = self.extraction.get("details", [])
        self.model_weights = self.extraction.get("confidence_scores", [])
        self.total_sentences = int(self.extraction.get("total_sentences", 0))
        self.clean_transcript_text = self.extraction.get("clean_transcript", self.clean_transcript_text)
        self.analytics = self.exporter.build_transcript_analytics(self.clean_transcript_text, action_items=self.action_items)
        self.summary_text = self.exporter.build_preview_summary(
            self.clean_transcript_text,
            action_items=self.action_items,
            summary_text=summary_text,
        )
        self.topics = topics or self.exporter.build_topic_labels(self.clean_transcript_text)
        self.weight_analytics = None

        self.breakdown_path = None
        self.keywords_path = None
        self.section_rows = []
        self.cb_vars = []
        self._dragged_idx = None

        self.title("Meeting Analytics & PDF Export")
        self.configure(fg_color="#1d2027")  # Match gui.py color palette
        
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        preview_width = min(1440, max(1200, int(screen_width * 0.94)))
        preview_height = min(900, max(800, int(screen_height * 0.88)))
        self.geometry(f"{preview_width}x{preview_height}")
        self.minsize(1000, 700)
        self.resizable(True, True)
        self.transient(parent)
        self.grab_set()

        self.protocol("WM_DELETE_WINDOW", self._close_window)

        # Configure main grid
        self.grid_rowconfigure(0, weight=0) # Header
        self.grid_rowconfigure(1, weight=0) # Dashboard
        self.grid_rowconfigure(2, weight=1) # Main Content
        self.grid_columnconfigure(0, weight=1)

        self._build_layout()
        self._load_analytics_images()
        self._render_preview()

    def _build_layout(self):
        # 1. Top Header Frame
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(15, 10))
        header_frame.grid_columnconfigure(0, weight=1)
        header_frame.grid_columnconfigure(1, weight=0)

        title_label = ctk.CTkLabel(header_frame, text="Meeting Minutes & Insights", font=("Inter", 24, "bold"), text_color="#e1e1e1")
        title_label.grid(row=0, column=0, sticky="w")
        
        btn_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        btn_frame.grid(row=0, column=1, sticky="e")

        refresh_btn = ctk.CTkButton(btn_frame, text="Refresh Preview", width=140, height=36, fg_color="#3a3f46", hover_color="#4b515a", command=self._render_preview)
        refresh_btn.pack(side="left", padx=(0, 10))

        export_btn = ctk.CTkButton(btn_frame, text="Download PDF", width=160, height=36, fg_color="#1f6aa5", hover_color="#144870", command=self._export_pdf)
        export_btn.pack(side="left")

        # 2. Top Dashboard Frame (Analytics)
        dashboard_frame = ctk.CTkFrame(self, fg_color="#252729", corner_radius=12)
        dashboard_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 15))
        dashboard_frame.grid_rowconfigure(0, weight=1)
        dashboard_frame.grid_columnconfigure(0, weight=1, uniform="dash") # Info
        dashboard_frame.grid_columnconfigure(1, weight=1, uniform="dash") # Breakdown
        dashboard_frame.grid_columnconfigure(2, weight=1, uniform="dash") # Keywords

        # Dashboard: Info
        info_frame = ctk.CTkFrame(dashboard_frame, fg_color="transparent")
        info_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        
        duration_text = self._format_duration(self.duration_seconds)
        ctk.CTkLabel(info_frame, text="Session Intelligence", font=("Inter", 16, "bold"), text_color="#e1e1e1").pack(anchor="w", pady=(0, 10))
        ctk.CTkLabel(info_frame, text=f"Duration: {duration_text}", font=("Inter", 13), text_color="#b7bcc4").pack(anchor="w", pady=(0, 5))
        
        self.stats_label = ctk.CTkLabel(info_frame, text="Analyzing transcript...", font=("Inter", 13), text_color="#b7bcc4")
        self.stats_label.pack(anchor="w", pady=(0, 5))

        all_kw = self.analytics.get("all_action_keywords", []) or []
        keyword_text = "None"
        if all_kw:
            top_kws = [f"{word} ({count})" for word, count, weight in all_kw[:4]]
            keyword_text = " • ".join(top_kws)
        
        ctk.CTkLabel(info_frame, text=f"Top Markers:\n{keyword_text}", font=("Inter", 12), text_color="#b7bcc4", wraplength=350, justify="left").pack(anchor="w", pady=(5, 0))

        # Dashboard: Breakdown Chart
        breakdown_frame = ctk.CTkFrame(dashboard_frame, fg_color="#1d2027", corner_radius=10)
        breakdown_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=15)
        self.breakdown_label = ctk.CTkLabel(breakdown_frame, text="Loading chart...", text_color="#b7bcc4")
        self.breakdown_label.pack(fill="both", expand=True, padx=10, pady=10)

        # Dashboard: Keywords Chart
        keywords_frame = ctk.CTkFrame(dashboard_frame, fg_color="#1d2027", corner_radius=10)
        keywords_frame.grid(row=0, column=2, sticky="nsew", padx=(10, 20), pady=15)
        self.keywords_label = ctk.CTkLabel(keywords_frame, text="Loading chart...", text_color="#b7bcc4")
        self.keywords_label.pack(fill="both", expand=True, padx=10, pady=10)

        # 3. Main Area (Left: Config, Right: Preview)
        # Symmetrical and Balanced Layout
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 20))
        main_frame.grid_rowconfigure(0, weight=1)
        # weight=1 for both makes them equally balanced
        main_frame.grid_columnconfigure(0, weight=1, uniform="main")
        main_frame.grid_columnconfigure(1, weight=1, uniform="main")

        # Config Panel (Left)
        config_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        config_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        config_frame.grid_rowconfigure(1, weight=1) # Sections
        config_frame.grid_rowconfigure(3, weight=1) # Items
        config_frame.grid_columnconfigure(0, weight=1) # THIS WAS MISSING AND CAUSED THE HUGE GAP

        sections_header = ctk.CTkLabel(config_frame, text="Report Structure (Drag to reorder)", font=("Inter", 14, "bold"), text_color="#e1e1e1")
        sections_header.grid(row=0, column=0, sticky="w", pady=(0, 8))

        # Match scrollbar to background for aesthetic appeal
        self.sections_scroll = ctk.CTkScrollableFrame(
            config_frame, fg_color="#252729", corner_radius=10, 
            scrollbar_button_color="#252729", scrollbar_button_hover_color="#323538"
        )
        self.sections_scroll.grid(row=1, column=0, sticky="nsew", pady=(0, 15))

        for section_name in self.exporter.DEFAULT_SECTIONS:
            variable = ctk.BooleanVar(value=True)
            self.section_rows.append({"label": section_name, "var": variable})
        
        self._rebuild_sections_ui()

        items_header = ctk.CTkLabel(config_frame, text="Extracted Action Items", font=("Inter", 14, "bold"), text_color="#e1e1e1")
        items_header.grid(row=2, column=0, sticky="w", pady=(0, 8))

        # Match scrollbar to background for aesthetic appeal
        self.items_scroll = ctk.CTkScrollableFrame(
            config_frame, fg_color="#252729", corner_radius=10,
            scrollbar_button_color="#252729", scrollbar_button_hover_color="#323538"
        )
        self.items_scroll.grid(row=3, column=0, sticky="nsew")

        if self.action_items:
            for idx, item in enumerate(self.action_items, 1):
                variable = ctk.BooleanVar(value=True)
                confidence = self.model_weights[idx - 1] if idx - 1 < len(self.model_weights) else 0.0
                display_text = f"{item} (Conf: {confidence:.0%})" if confidence > 0 else item
                
                item_frame = ctk.CTkFrame(self.items_scroll, fg_color="#1d2027", corner_radius=8)
                item_frame.pack(fill="x", pady=4, padx=4)
                
                checkbox = ctk.CTkCheckBox(item_frame, text=display_text, variable=variable, font=("Inter", 12), hover_color="#323538", text_color="#b7bcc4")
                checkbox.pack(anchor="w", padx=10, pady=10)
                
                self.cb_vars.append((variable, item))
        else:
            none_label = ctk.CTkLabel(self.items_scroll, text="No action items detected.", font=("Inter", 12), text_color="#b7bcc4")
            none_label.pack(pady=20)

        # Preview Panel (Right)
        preview_frame = ctk.CTkFrame(main_frame, fg_color="#252729", corner_radius=12)
        preview_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        preview_frame.grid_rowconfigure(1, weight=1)
        preview_frame.grid_columnconfigure(0, weight=1)

        preview_header = ctk.CTkLabel(preview_frame, text="PDF Content Preview", font=("Inter", 14, "bold"), text_color="#e1e1e1")
        preview_header.grid(row=0, column=0, sticky="w", padx=20, pady=(15, 10))

        # Match text color and background to gui.py styling
        self.preview_box = ctk.CTkTextbox(
            preview_frame, font=("Courier", 13), fg_color="#1d2027", text_color="#e1e1e1",
            scrollbar_button_color="#1d2027", scrollbar_button_hover_color="#323538"
        )
        self.preview_box.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))

    # ---- Drag and Drop Implementation ----
    
    def _rebuild_sections_ui(self):
        """Rebuild section reordering UI with drag and drop."""
        for child in self.sections_scroll.winfo_children():
            child.destroy()

        self.section_widgets = []
        for idx, entry in enumerate(self.section_rows):
            row_frame = ctk.CTkFrame(self.sections_scroll, fg_color="#1d2027", corner_radius=8)
            row_frame.pack(fill="x", pady=4, padx=4)
            row_frame.drag_index = idx  # Store index directly on frame
            
            # Left grip icon
            grip = ctk.CTkLabel(row_frame, text="≡", font=("Inter", 18, "bold"), text_color="#4a4d50", width=30, cursor="hand2")
            grip.pack(side="left", padx=(10, 5))
            
            chk = ctk.CTkCheckBox(row_frame, text=entry["label"], variable=entry["var"], font=("Inter", 12), text_color="#b7bcc4")
            chk.pack(side="left", anchor="w", pady=10, padx=(5, 10))

            # Bind events to the entire row frame for better drag responsiveness
            row_frame.bind("<Button-1>", self._on_drag_start)
            row_frame.bind("<B1-Motion>", self._on_drag_motion)
            row_frame.bind("<ButtonRelease-1>", self._on_drag_release)
            grip.bind("<Button-1>", self._on_drag_start)
            grip.bind("<B1-Motion>", self._on_drag_motion)
            grip.bind("<ButtonRelease-1>", self._on_drag_release)

            self.section_widgets.append(row_frame)

    def _on_drag_start(self, event):
        widget = event.widget
        # Find the parent row frame
        while widget and not hasattr(widget, 'drag_index'):
            widget = widget.master
        
        if widget and hasattr(widget, 'drag_index'):
            self._dragged_idx = widget.drag_index
            widget.configure(fg_color="#3a4c63", border_width=2, border_color="#3b82f6")

    def _on_drag_motion(self, event):
        pass # Visual feedback during drag could be added here

    def _on_drag_release(self, event):
        if self._dragged_idx is None:
            return
        
        widget = event.widget
        # Find the parent row frame
        while widget and not hasattr(widget, 'drag_index'):
            widget = widget.master
        
        # Reset visual state of dragged widget
        if self._dragged_idx < len(self.section_widgets):
            self.section_widgets[self._dragged_idx].configure(fg_color="#1d2027", border_width=0)
        
        if not widget or not hasattr(widget, 'drag_index'):
            self._dragged_idx = None
            return
        
        # Get absolute mouse Y position
        mouse_y = event.x_root + event.y
        
        # Find which row_frame is closest to mouse_y
        closest_idx = self._dragged_idx
        min_dist = float('inf')
        
        for i, row in enumerate(self.section_widgets):
            row_y = row.winfo_rooty()
            row_height = row.winfo_height()
            row_center_y = row_y + (row_height // 2)
            dist = abs(mouse_y - row_center_y)
            
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        old_idx = self._dragged_idx
        if old_idx != closest_idx:
            item = self.section_rows.pop(old_idx)
            self.section_rows.insert(closest_idx, item)
            self._rebuild_sections_ui()
            self._render_preview()
        
        self._dragged_idx = None

    # ---- Charts and Preview Rendering ----

    def _load_analytics_images(self):
        """Load and display analytics chart images."""
        try:
            # Build weight-based analytics instead of frequency-based
            self.weight_analytics = self._build_weight_based_analytics()
            self.breakdown_path, self.keywords_path = self.exporter._build_separate_analytics_charts(self.weight_analytics)
        except Exception:
            self.weight_analytics = None
            self.breakdown_path, self.keywords_path = (None, None)

        if self.breakdown_path and os.path.exists(self.breakdown_path):
            try:
                image = Image.open(self.breakdown_path)
                photo = ctk.CTkImage(image, size=(280, 160))
                self.breakdown_label.configure(image=photo, text="")
                self.breakdown_label.image = photo
            except Exception:
                self.breakdown_label.configure(text="Breakdown chart unavailable")
        else:
            self.breakdown_label.configure(text="Breakdown chart unavailable")

        if self.keywords_path and os.path.exists(self.keywords_path):
            try:
                image = Image.open(self.keywords_path)
                photo = ctk.CTkImage(image, size=(280, 160))
                self.keywords_label.configure(image=photo, text="")
                self.keywords_label.image = photo
            except Exception:
                self.keywords_label.configure(text="Keyword chart unavailable")
        else:
            self.keywords_label.configure(text="Keyword chart unavailable")

        self._bind_chart_preview(self.breakdown_label, self.breakdown_path, "Transcript Breakdown")
        self._bind_chart_preview(self.keywords_label, self.keywords_path, "Action Markers")

        info_count = max(0, self.total_sentences - len(self.action_items)) if self.total_sentences else None
        if info_count is None:
            stats_text = f"Action items: {len(self.action_items)}"
        else:
            stats_text = f"Sentences: {self.total_sentences}  •  Action: {len(self.action_items)}  •  Info: {info_count}"
        self.stats_label.configure(text=stats_text)

    def _bind_chart_preview(self, widget, image_path, title):
        if widget is None: return
        def open_preview(_event=None):
            if image_path and os.path.exists(image_path):
                self._open_chart_preview(image_path, title)
        widget.configure(cursor="hand2")
        widget.bind("<Button-1>", open_preview)

    def _open_chart_preview(self, image_path, title):
        popup = ctk.CTkToplevel(self)
        popup.title(title)
        popup.geometry("980x720")
        popup.configure(fg_color="#1d2027")
        popup.minsize(720, 520)
        popup.transient(self)
        popup.grab_set()

        frame = ctk.CTkFrame(popup, fg_color="#252729")
        frame.pack(fill="both", expand=True, padx=12, pady=12)

        title_label = ctk.CTkLabel(frame, text=title, font=("Inter", 16, "bold"), text_color="#e1e1e1")
        title_label.pack(anchor="w", padx=8, pady=(8, 4))

        image = Image.open(image_path)
        preview_width = 920
        preview_height = 620
        photo = ctk.CTkImage(image, size=(preview_width, preview_height))
        image_label = ctk.CTkLabel(frame, text="", image=photo)
        image_label.image = photo
        image_label.pack(fill="both", expand=True, padx=8, pady=(4, 8))

        popup.bind("<Escape>", lambda _event: popup.destroy())

    def _render_preview(self):
        """Render the PDF preview based on selected sections and items."""
        selected = [item for variable, item in self.cb_vars if variable.get()]
        self.preview_box.delete("1.0", "end")
        self.preview_box.insert("end", "Minutes of the Meeting\n")
        self.preview_box.insert("end", "="*80 + "\n\n")

        current_order = [row["label"] for row in self.section_rows]
        included = {row["label"] for row in self.section_rows if row["var"].get()}

        for section_name in current_order:
            if section_name not in included:
                continue

            if section_name == "Executive Overview":
                self.preview_box.insert("end", "[ Executive Overview ]\n")
                self.preview_box.insert("end", "-"*80 + "\n")
                summary_paragraphs = self.formatter.split_summary_into_paragraphs(self.summary_text, self.duration_seconds)
                for paragraph in summary_paragraphs:
                    self.preview_box.insert("end", f"{paragraph}\n\n")
            elif section_name == "Topics Discussed":
                self.preview_box.insert("end", "[ Topics Discussed ]\n")
                self.preview_box.insert("end", "-"*80 + "\n")
                if self.topics:
                    for idx, topic in enumerate(self.topics, 1):
                        self.preview_box.insert("end", f"{idx}. {topic}\n")
                    self.preview_box.insert("end", "\n")
                else:
                    self.preview_box.insert("end", "- No topics were identified.\n\n")
            elif section_name == "Analytics Overview":
                self.preview_box.insert("end", "[ Analytics Overview ]\n")
                self.preview_box.insert("end", "-"*80 + "\n")
                self.preview_box.insert("end", f"- Duration: {self._format_duration(self.duration_seconds)}\n")
                self.preview_box.insert("end", f"- Sentences detected: {self.analytics.get('total_sentences', 0)}\n")
                self.preview_box.insert("end", f"- Action items: {self.analytics.get('action_count', 0)}\n")
                self.preview_box.insert("end", f"- Information sentences: {self.analytics.get('info_count', 0)}\n")
                if self.analytics.get("top_action_keywords"):
                    kw_list = self.analytics.get("top_action_keywords", [])
                    self.preview_box.insert(
                        "end",
                        "- Top action keywords: "
                        + ", ".join(f"{word} ({count}, {int(round(weight * 100))}%)" for word, count, weight in kw_list)
                        + "\n",
                    )
                self.preview_box.insert("end", "\n(Charts will be included in the exported PDF)\n\n")
            elif section_name == "Action Items":
                self.preview_box.insert("end", "[ Action Items ]\n")
                self.preview_box.insert("end", "-"*80 + "\n")
                if selected:
                    for idx, item in enumerate(selected, 1):
                        confidence = 0.0
                        if item in self.action_items:
                            item_idx = self.action_items.index(item)
                            if item_idx < len(self.model_weights):
                                confidence = self.model_weights[item_idx]
                        self.preview_box.insert("end", f"{idx}. {item}\n")
                        if confidence > 0:
                            self.preview_box.insert("end", f"   Evidence weight: {confidence:.2%}\n")
                        explanation = self.formatter.build_action_explanation(item)
                        why = self.formatter.build_action_flag_reason(item)
                        if explanation:
                            # Temporary commented out detailed explanations 
                            # self.preview_box.insert("end", f"   Task: {explanation}\n")
                            self.preview_box.insert("end", f"{why}\n\n")
                else:
                    self.preview_box.insert("end", "No action items selected.\n\n")
            elif section_name == "Full Transcript":
                self.preview_box.insert("end", "[ Full Transcript ]\n")
                self.preview_box.insert("end", "-"*80 + "\n")
                transcript_excerpt = self.clean_transcript_text[:2000]
                if len(self.clean_transcript_text) > 2000:
                    transcript_excerpt += "\n..."
                self.preview_box.insert("end", transcript_excerpt)

    def _export_pdf(self):
        selected = [item for variable, item in self.cb_vars if variable.get()]
        section_order = [row["label"] for row in self.section_rows]
        include_sections = [row["label"] for row in self.section_rows if row["var"].get()]

        try:
            pdf_path = self.exporter.generate_pdf(
                content=self.clean_transcript_text,
                action_items=selected,
                summary=self.summary_text,
                source_file=self.source_file,
                topics=self.topics,
                section_order=section_order,
                include_sections=include_sections,
                duration_seconds=self.duration_seconds,
                analytics=self.weight_analytics,
            )
            try:
                os.startfile(pdf_path)
            except Exception:
                pass
        finally:
            self._cleanup_temp_charts()
            self._close_window()

    def _cleanup_temp_charts(self):
        for path in (self.breakdown_path, self.keywords_path):
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

    def _close_window(self):
        self._cleanup_temp_charts()
        try:
            self.grab_release()
        except Exception:
            pass
        try:
            self.destroy()
        except Exception:
            pass

    def _build_weight_based_analytics(self):
        """Build analytics dict with weight-based action marker data.
        Focuses on action verbs and initiation markers like: paki, should, do, make, submit, etc.
        """
        import re
        
        # Action markers and verbs that initiate tasks
        action_verbs = {
            'paki', 'pakisend', 'paki-send', 'pakisubmit', 'pakireview', 'pakiprepare',
            'should', 'must', 'need', 'please', 'can', 'could', 'will',
            'do', 'make', 'create', 'prepare', 'submit', 'send', 'draft',
            'review', 'update', 'finalize', 'complete', 'develop', 'build',
            'run', 'execute', 'implement', 'arrange', 'schedule', 'close',
            'due', 'ipasa', 'buksan', 'isara', 'ilagay', 'gawin',
            'document', 'outline', 'provide', 'analyze', 'schedule'
        }
        
        # Extract action verbs from action items, weighted by confidence
        weighted_verbs = {}
        total_weight = 0
        
        for item, confidence in zip(self.action_items, self.model_weights):
            # Extract words and check if they're action verbs
            words = re.findall(r'\b[a-z]+\b', item.lower())
            
            for word in words:
                if word in action_verbs:
                    weighted_verbs[word] = weighted_verbs.get(word, 0) + confidence
                    total_weight += confidence
        
        # Sort by weight and build top keywords list
        top_weighted = sorted(weighted_verbs.items(), key=lambda x: x[1], reverse=True)[:5]
        top_keywords = []
        for verb, weight_sum in top_weighted:
            # Count occurrences for display - use simple substring matching
            count = sum(1 for item in self.action_items if verb in item.lower())
            normalized_weight = weight_sum / total_weight if total_weight > 0 else 0
            top_keywords.append((verb, count, normalized_weight))
        
        # Return modified analytics with weight-based action markers
        analytics_copy = self.analytics.copy()
        analytics_copy['top_action_keywords'] = top_keywords
        return analytics_copy

    @staticmethod
    def _format_duration(seconds):
        if seconds is None: return "Unknown"
        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        return f"{minutes}m {secs}s"
