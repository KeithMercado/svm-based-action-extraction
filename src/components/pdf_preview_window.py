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

        self.breakdown_path = None
        self.keywords_path = None
        self.section_rows = []
        self.cb_vars = []

        self.title("PDF Preview & Export")
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        preview_width = min(1440, max(1160, int(screen_width * 0.94)))
        preview_height = min(900, max(760, int(screen_height * 0.88)))
        self.geometry(f"{preview_width}x{preview_height}")
        self.minsize(1160, 760)
        self.resizable(True, True)
        self.transient(parent)
        self.grab_set()

        self.protocol("WM_DELETE_WINDOW", self._close_window)

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=0, minsize=360)
        self.grid_columnconfigure(1, weight=1)

        self._build_layout()
        self._load_analytics_images()
        self._render_preview()

    def _build_layout(self):
        """Build the main layout with left and right panels."""
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=12, pady=(12, 6))
        content.grid_rowconfigure(0, weight=1)
        content.grid_columnconfigure(0, weight=0, minsize=360)
        content.grid_columnconfigure(1, weight=1)

        left_panel = ctk.CTkFrame(content, fg_color="transparent")
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=0)
        left_panel.grid_rowconfigure(0, weight=1)
        left_panel.grid_columnconfigure(0, weight=1)

        self.left = ctk.CTkScrollableFrame(left_panel, fg_color="transparent")
        self.left.grid(row=0, column=0, sticky="nsew")

        right_panel = ctk.CTkFrame(content, fg_color="transparent")
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=0)
        right_panel.grid_rowconfigure(0, weight=1)
        right_panel.grid_columnconfigure(0, weight=1)

        self.right = ctk.CTkScrollableFrame(right_panel, fg_color="transparent")
        self.right.grid(row=0, column=0, sticky="nsew")

        self._build_right_panel()
        self._build_left_panel()

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.grid(row=1, column=0, columnspan=2, sticky="ew", padx=12, pady=(6, 12))

        export_btn = ctk.CTkButton(btn_row, text="Download PDF", width=160, height=36, fg_color="#1f6aa5", command=self._export_pdf)
        export_btn.pack(side="right", padx=(6, 0))

        refresh_btn = ctk.CTkButton(btn_row, text="Refresh Preview", width=140, height=36, fg_color="#3a3f46", command=self._render_preview)
        refresh_btn.pack(side="right", padx=(0, 6))

    def _build_right_panel(self):
        """Build right panel with analytics, duration, and preview."""
        analytics_frame = ctk.CTkFrame(self.right, fg_color="transparent")
        analytics_frame.pack(fill="x", pady=(0, 8))

        title_label = ctk.CTkLabel(analytics_frame, text="Preview: Minutes of the Meeting", font=("Inter", 16, "bold"))
        title_label.pack(anchor="w")

        # Duration display
        duration_text = self._format_duration(self.duration_seconds)
        duration_label = ctk.CTkLabel(analytics_frame, text=f"Duration: {duration_text}", font=("Inter", 10), text_color="#9ca3af")
        duration_label.pack(anchor="w", pady=(2, 0))

        self.stats_label = ctk.CTkLabel(analytics_frame, text="Analyzing transcript...", font=("Inter", 11))
        self.stats_label.pack(anchor="w", pady=(6, 4))

        analytics_card = ctk.CTkFrame(analytics_frame, fg_color="#20242b", corner_radius=10)
        analytics_card.pack(fill="x", pady=(6, 8))

        chart_label = ctk.CTkLabel(analytics_card, text="Building charts...", text_color="#cdd6e1")
        chart_label.pack(padx=10, pady=(10, 2), anchor="w")

        charts_row = ctk.CTkFrame(analytics_card, fg_color="transparent")
        charts_row.pack(fill="both", expand=True, padx=10, pady=(4, 8))
        charts_row.grid_columnconfigure(0, weight=1, uniform="charts")
        charts_row.grid_columnconfigure(1, weight=1, uniform="charts")

        breakdown_frame = ctk.CTkFrame(charts_row, fg_color="#1a1d21", width=160, height=140)
        breakdown_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        breakdown_frame.pack_propagate(False)
        self.breakdown_label = ctk.CTkLabel(breakdown_frame, text="Click to expand", text_color="#cdd6e1")
        self.breakdown_label.pack(fill="both", expand=True, padx=8, pady=8)

        keywords_frame = ctk.CTkFrame(charts_row, fg_color="#1a1d21", width=240, height=140)
        keywords_frame.grid(row=0, column=1, sticky="nsew")
        keywords_frame.pack_propagate(False)
        self.keywords_label = ctk.CTkLabel(keywords_frame, text="Click to expand", text_color="#cdd6e1")
        self.keywords_label.pack(fill="both", expand=True, padx=8, pady=8)

        # Show ALL action markers
        all_kw = self.analytics.get("all_action_keywords", []) or []
        if all_kw:
            keyword_lines = [f"{word} (w={weight:.2f}, n={count})" for word, count, weight in all_kw]
            keyword_text = "  •  ".join(keyword_lines)
        else:
            keyword_text = "No action markers detected."

        keyword_label = ctk.CTkLabel(
            analytics_card,
            text=f"All action markers: {keyword_text}",
            wraplength=700,
            justify="left",
            text_color="#d6dbe3",
            font=("Inter", 11),
        )
        keyword_label.pack(padx=10, pady=(0, 10), anchor="w")

        self._bind_chart_preview(self.breakdown_label, self.breakdown_path, "Transcript Breakdown")
        self._bind_chart_preview(self.keywords_label, self.keywords_path, "Action Markers")

        why_frame = ctk.CTkFrame(self.right, fg_color="#20242b", corner_radius=10)
        why_frame.pack(fill="x", pady=(8, 8))
        why_title = ctk.CTkLabel(why_frame, text="Why This Was Flagged", font=("Inter", 13, "bold"), text_color="#eef3f8")
        why_title.pack(anchor="w", padx=10, pady=(10, 6))

        why_box = ctk.CTkTextbox(why_frame, height=96, wrap="word")
        why_box.pack(fill="x", padx=10, pady=(0, 10))
        why_box.insert("end", "These are conservative explanations based on the transcript text and model confidence scores.\n\n")
        if self.action_details:
            for idx, detail in enumerate(self.action_details, 1):
                item_text = str(detail.get("item", "")).strip()
                reason_text = str(detail.get("reason", "")).strip() or "Matched the action-item pattern."
                basis_text = str(detail.get("basis", "")).strip()
                confidence = self.model_weights[idx - 1] if idx - 1 < len(self.model_weights) else 0.0
                why_box.insert("end", f"{idx}. {item_text}\n")
                why_box.insert("end", f"   Why flagged: {reason_text}\n")
                if basis_text:
                    why_box.insert("end", f"   Basis: {basis_text}\n")
                if confidence > 0:
                    why_box.insert("end", f"   Evidence weight: {confidence:.2%}\n")
                why_box.insert("end", "\n")
        else:
            why_box.insert("end", "No action items were detected.\n")
        why_box.configure(state="disabled")

    def _build_left_panel(self):
        """Build left panel with sections and action items."""
        sections_frame = ctk.CTkFrame(self.left, fg_color="transparent")
        sections_frame.pack(fill="x", pady=(0, 8))

        sections_label = ctk.CTkLabel(sections_frame, text="Report Sections (drag via Up/Down)", font=("Inter", 11, "bold"))
        sections_label.pack(anchor="w", pady=(0, 6))

        self.sections_list_frame = ctk.CTkFrame(sections_frame, fg_color="transparent")
        self.sections_list_frame.pack(fill="x")

        for section_name in self.exporter.DEFAULT_SECTIONS:
            variable = ctk.BooleanVar(value=True)
            self.section_rows.append({"label": section_name, "var": variable})

        self._rebuild_sections_ui()

        list_frame = ctk.CTkFrame(self.left, fg_color="transparent")
        list_frame.pack(fill="both", expand=True)

        if self.action_items:
            for idx, item in enumerate(self.action_items, 1):
                variable = ctk.BooleanVar(value=True)
                confidence = self.model_weights[idx - 1] if idx - 1 < len(self.model_weights) else 0.0
                display_text = f"{idx}. {item} ({confidence:.0%})" if confidence > 0 else f"{idx}. {item}"
                checkbox = ctk.CTkCheckBox(list_frame, text=display_text, variable=variable, width=340)
                checkbox.pack(anchor="w", pady=6)
                self.cb_vars.append((variable, item))
        else:
            none_label = ctk.CTkLabel(list_frame, text="No action items detected.", font=("Inter", 12), text_color="gray")
            none_label.pack(pady=20)

        preview_label = ctk.CTkLabel(self.right, text="PDF Preview", font=("Inter", 13, "bold"))
        preview_label.pack(anchor="w", pady=(8, 0))

        self.preview_box = ctk.CTkTextbox(self.right, width=340, height=300, state="normal")
        self.preview_box.pack(fill="both", expand=True, pady=(6, 0))

    def _rebuild_sections_ui(self):
        """Rebuild section reordering UI."""
        for child in self.sections_list_frame.winfo_children():
            child.destroy()

        for idx, entry in enumerate(self.section_rows):
            row = ctk.CTkFrame(self.sections_list_frame, fg_color="transparent")
            row.pack(fill="x", pady=2)
            chk = ctk.CTkCheckBox(row, text=entry["label"], variable=entry["var"])
            chk.pack(side="left", anchor="w")
            up_btn = ctk.CTkButton(row, text="↑", width=28, height=22, fg_color="#2b3340", command=lambda i=idx: self._move_section_up(i))
            up_btn.pack(side="right", padx=(6, 0))
            down_btn = ctk.CTkButton(row, text="↓", width=28, height=22, fg_color="#2b3340", command=lambda i=idx: self._move_section_down(i))
            down_btn.pack(side="right", padx=(6, 0))

    def _move_section_up(self, index):
        """Move section up in order."""
        if index <= 0:
            return
        self.section_rows[index - 1], self.section_rows[index] = self.section_rows[index], self.section_rows[index - 1]
        self._rebuild_sections_ui()
        self._render_preview()

    def _move_section_down(self, index):
        """Move section down in order."""
        if index >= len(self.section_rows) - 1:
            return
        self.section_rows[index + 1], self.section_rows[index] = self.section_rows[index], self.section_rows[index + 1]
        self._rebuild_sections_ui()
        self._render_preview()

    def _load_analytics_images(self):
        """Load and display analytics chart images."""
        try:
            self.breakdown_path, self.keywords_path = self.exporter._build_separate_analytics_charts(self.analytics)
        except Exception:
            self.breakdown_path, self.keywords_path = (None, None)

        if self.breakdown_path and os.path.exists(self.breakdown_path):
            try:
                image = Image.open(self.breakdown_path)
                photo = ctk.CTkImage(image, size=(220, 180))
                self.breakdown_label.configure(image=photo, text="")
                self.breakdown_label.image = photo
            except Exception:
                self.breakdown_label.configure(text="Breakdown chart unavailable")
        else:
            self.breakdown_label.configure(text="Breakdown chart unavailable")

        if self.keywords_path and os.path.exists(self.keywords_path):
            try:
                image = Image.open(self.keywords_path)
                photo = ctk.CTkImage(image, size=(360, 180))
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
            stats_text = f"Found {len(self.action_items)} suggested action item(s)."
        else:
            stats_text = f"Sentences: {self.total_sentences}  •  Action items: {len(self.action_items)}  •  Info: {info_count}"
        self.stats_label.configure(text=stats_text)

    def _bind_chart_preview(self, widget, image_path, title):
        """Make a chart widget open a larger preview when clicked."""
        if widget is None:
            return

        def open_preview(_event=None):
            if image_path and os.path.exists(image_path):
                self._open_chart_preview(image_path, title)

        widget.configure(cursor="hand2")
        widget.bind("<Button-1>", open_preview)

    def _open_chart_preview(self, image_path, title):
        """Open a larger chart preview window."""
        popup = ctk.CTkToplevel(self)
        popup.title(title)
        popup.geometry("980x720")
        popup.minsize(720, 520)
        popup.transient(self)
        popup.grab_set()

        frame = ctk.CTkFrame(popup, fg_color="#12161b")
        frame.pack(fill="both", expand=True, padx=12, pady=12)

        title_label = ctk.CTkLabel(frame, text=title, font=("Inter", 16, "bold"), text_color="#eef3f8")
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
        self.preview_box.insert("end", "Minutes of the Meeting\n\n")

        current_order = [row["label"] for row in self.section_rows]
        included = {row["label"] for row in self.section_rows if row["var"].get()}

        for section_name in current_order:
            if section_name not in included:
                continue

            if section_name == "Executive Overview":
                self.preview_box.insert("end", "Executive Overview\n")
                summary_paragraphs = self.formatter.split_summary_into_paragraphs(self.summary_text, self.duration_seconds)
                for paragraph in summary_paragraphs:
                    self.preview_box.insert("end", f"{paragraph}\n\n")
            elif section_name == "Topics Discussed":
                self.preview_box.insert("end", "Topics Discussed\n")
                if self.topics:
                    for idx, topic in enumerate(self.topics, 1):
                        self.preview_box.insert("end", f"{idx}. {topic}\n")
                    self.preview_box.insert("end", "\n")
                else:
                    self.preview_box.insert("end", "- No topics were identified.\n\n")
            elif section_name == "Analytics Overview":
                self.preview_box.insert("end", "Analytics Overview\n")
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
                self.preview_box.insert("end", "\n")
            elif section_name == "Action Items":
                self.preview_box.insert("end", "Action Items:\n")
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
                            self.preview_box.insert("end", f"   Task: {explanation}\n")
                        self.preview_box.insert("end", f"   Why flagged: {why}\n")
                else:
                    self.preview_box.insert("end", "No action items selected.\n")
                self.preview_box.insert("end", "\n")
            elif section_name == "Full Transcript":
                self.preview_box.insert("end", "Full Transcript:\n")
                transcript_excerpt = self.clean_transcript_text[:2000]
                if len(self.clean_transcript_text) > 2000:
                    transcript_excerpt += "\n..."
                self.preview_box.insert("end", transcript_excerpt)

    def _export_pdf(self):
        """Export the PDF with selected sections and items."""
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
            )
            try:
                os.startfile(pdf_path)
            except Exception:
                pass
        finally:
            self._cleanup_temp_charts()
            self._close_window()

    def _cleanup_temp_charts(self):
        """Clean up temporary chart images."""
        for path in (self.breakdown_path, self.keywords_path):
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

    def _close_window(self):
        """Close the preview window safely."""
        self._cleanup_temp_charts()
        try:
            self.grab_release()
        except Exception:
            pass
        try:
            self.destroy()
        except Exception:
            pass

    @staticmethod
    def _format_duration(seconds):
        """Format duration in seconds to HH:MM:SS."""
        if seconds is None:
            return "Unknown"
        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        return f"{minutes}m {secs}s"
