import tkinter as tk
import customtkinter as ctk


class PDFLayoutEditorDialog(ctk.CTkToplevel):
    DEFAULT_SECTION_ORDER = [
        "Executive Overview",
        "Topics Discussed",
        "Action Items",
        "Full Transcript",
    ]

    def __init__(self, parent, action_items=None):
        super().__init__(parent)
        self.title("Customize PDF Layout")
        min_w, min_h = 380, 600
        self.geometry(f"{min_w}x{min_h}")
        self.minsize(min_w, min_h)
        self.configure(fg_color="#1d2027")
        self.transient(parent)
        self.grab_set()

        self.section_items = list(self.DEFAULT_SECTION_ORDER)
        self.result = None
        self._drag_index = None
        self._section_rows = []
        self._drag_anchor_row = None
        self.section_container = None

        self._build_ui()

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        title = ctk.CTkLabel(
            self,
            text="Customize PDF Output",
            font=("Inter", 20, "bold"),
            text_color="#f2f4f8",
        )
        title.grid(row=0, column=0, sticky="w", padx=20, pady=(16, 4))

        subtitle = ctk.CTkLabel(
            self,
            text="Drag the section rows to reorder the PDF layout. Closing this window cancels PDF generation.",
            font=("Inter", 12),
            text_color="#b9c1cc",
            wraplength=340,
            justify="left",
        )
        subtitle.grid(row=1, column=0, sticky="w", padx=20, pady=(0, 12))

        self.section_scroll = ctk.CTkScrollableFrame(
            self,
            fg_color="#181b20",
            corner_radius=10,
        )
        self.section_scroll.grid(row=2, column=0, sticky="nsew", padx=14, pady=(0, 10))

        # Compatibility-safe styling: apply optional scrollbar colors only if supported
        # by the installed CustomTkinter version.
        self._apply_scrollbar_style_safe()

        # Render rows into a dedicated container so internal scroll widgets remain intact.
        self.section_container = ctk.CTkFrame(self.section_scroll, fg_color="transparent")
        self.section_container.pack(fill="x", expand=True)

        # Keep the scrollbar hidden but preserve mouse-wheel scrolling.
        self.section_scroll.bind("<MouseWheel>", self._on_mousewheel_scroll)

        self._render_section_rows()

        controls = ctk.CTkFrame(self, fg_color="transparent")
        controls.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 12))
        controls.grid_columnconfigure(0, weight=1)
        controls.grid_columnconfigure(1, weight=1)

        cancel_btn = ctk.CTkButton(
            controls,
            text="Cancel",
            height=34,
            fg_color="#3f4650",
            hover_color="#525a66",
            command=self._on_cancel,
        )
        cancel_btn.grid(row=0, column=1, padx=(6, 0), sticky="ew")

        generate_btn = ctk.CTkButton(
            controls,
            text="Generate PDF",
            height=34,
            fg_color="#1f6aa5",
            hover_color="#1a5280",
            command=self._on_generate,
        )
        generate_btn.grid(row=0, column=0, padx=(0, 6), sticky="ew")

        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

    def _on_mousewheel_scroll(self, event):
        canvas = getattr(self.section_scroll, "_parent_canvas", None)
        if canvas is None:
            return
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _apply_scrollbar_style_safe(self):
        """Apply scrollbar visual styling only when the current CTk build supports it."""
        try:
            self.section_scroll.configure(
                scrollbar_fg_color="#181b20",
                scrollbar_button_color="#60656d",
                scrollbar_button_hover_color="#757c86",
            )
        except Exception:
            # Older CustomTkinter versions may not support these style keys.
            pass

    def _render_section_rows(self):
        container = self.section_container
        if container is None:
            return

        for widget in container.winfo_children():
            widget.destroy()

        self._section_rows = []

        for index, section in enumerate(self.section_items):
            row = ctk.CTkFrame(container, fg_color="#232830", corner_radius=10)
            row.pack(fill="x", padx=8, pady=5)
            row.grid_columnconfigure(1, weight=1)

            handle = ctk.CTkLabel(
                row,
                text="⋮⋮",
                font=("Inter", 16, "bold"),
                text_color="#7d8b9b",
                width=28,
            )
            handle.grid(row=0, column=0, padx=(10, 6), pady=10, sticky="w")

            label = ctk.CTkLabel(
                row,
                text=section,
                font=("Inter", 13, "bold"),
                text_color="#e7edf5",
                anchor="w",
                justify="left",
            )
            label.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="w")

            for widget in (row, handle, label):
                widget.bind("<ButtonPress-1>", lambda event, current_index=index: self._on_section_drag_start(current_index))
                widget.bind("<B1-Motion>", self._on_section_drag_motion)
                widget.bind("<ButtonRelease-1>", self._on_section_drag_stop)

            self._section_rows.append(row)

    def _on_section_drag_start(self, index):
        self._drag_index = index
        self._drag_anchor_row = self._section_rows[index]

    def _find_section_index_from_y(self, y_root):
        if not self._section_rows:
            return None

        for index, row in enumerate(self._section_rows):
            top = row.winfo_rooty()
            bottom = top + row.winfo_height()
            if y_root < bottom:
                return index
        return len(self._section_rows) - 1

    def _move_section(self, old_index, new_index):
        if old_index == new_index:
            return

        item = self.section_items.pop(old_index)
        self.section_items.insert(new_index, item)
        self._render_section_rows()
        self._drag_index = new_index

    def _on_section_drag_motion(self, event):
        if self._drag_index is None:
            return

        target_index = self._find_section_index_from_y(event.y_root)
        if target_index is None or target_index == self._drag_index:
            return

        self._move_section(self._drag_index, target_index)

    def _on_section_drag_stop(self, _event):
        self._drag_index = None
        self._drag_anchor_row = None

    def _current_section_order(self):
        return list(self.section_items)

    def _on_generate(self):
        current_order = self._current_section_order()
        self.result = {
            "section_order": current_order,
            "selection_order": current_order,
            "include_sections": list(self.section_items),
        }

        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()

    def _on_cancel(self):
        self.result = None
        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()

    def show_modal(self):
        self.wait_window()
        return self.result
