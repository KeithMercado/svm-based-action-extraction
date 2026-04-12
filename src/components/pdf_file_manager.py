import customtkinter as ctk
import os
from datetime import datetime

class PDFFileManager(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        
        # Window Config
        self.title("Meeting Minutes (MoM) Manager")
        self.geometry("560x500")
        self.minsize(560, 500)
        self.maxsize(560, 500)
        self.resizable(False, False)
        
        # Get base directory
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.output_dir = os.path.join(self.base_dir, "output", "pdf")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        self._create_widgets()
        self._load_files()
        
    def _create_widgets(self):
        # Title Frame
        title_frame = ctk.CTkFrame(self, fg_color="transparent")
        title_frame.pack(fill="x", padx=20, pady=(20, 10))
        
        title_label = ctk.CTkLabel(
            title_frame, 
            text="📄 Meeting Minutes (MoM)", 
            font=("Inter", 20, "bold")
        )
        title_label.pack(side="left")
        
        # Search & Filter Frame
        search_frame = ctk.CTkFrame(self, fg_color="transparent")
        search_frame.pack(fill="x", padx=20, pady=(0, 10))
        
        self.search_entry = ctk.CTkEntry(
            search_frame, 
            placeholder_text="Search minutes...",
            height=35
        )
        self.search_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.search_entry.bind("<KeyRelease>", self._on_search)
        
        refresh_btn = ctk.CTkButton(
            search_frame,
            text="🔄 Refresh",
            width=100,
            height=35,
            command=self._load_files
        )
        refresh_btn.pack(side="left")
        
        # Sort Options
        sort_frame = ctk.CTkFrame(self, fg_color="transparent")
        sort_frame.pack(fill="x", padx=20, pady=(0, 10))
        
        sort_label = ctk.CTkLabel(
            sort_frame,
            text="Sort by:",
            font=("Inter", 11)
        )
        sort_label.pack(side="left", padx=(0, 10))
        
        self.sort_var = ctk.StringVar(value="Date (Newest)")
        sort_options = ["Date (Newest)", "Date (Oldest)", "Name (A-Z)", "Size"]
        
        sort_menu = ctk.CTkOptionMenu(
            sort_frame,
            variable=self.sort_var,
            values=sort_options,
            width=150,
            height=30,
            command=lambda x: self._load_files()
        )
        sort_menu.pack(side="left")
        
        # File List Frame with Scrollbar
        list_frame = ctk.CTkFrame(self)
        list_frame.pack(fill="both", expand=True, padx=20, pady=(0, 10))
        
        # Scrollable Frame
        self.scrollable_frame = ctk.CTkScrollableFrame(
            list_frame,
            fg_color="transparent"
        )
        self.scrollable_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Status Bar
        self.status_label = ctk.CTkLabel(
            self,
            text="Loading...",
            font=("Inter", 10),
            text_color="gray"
        )
        self.status_label.pack(fill="x", padx=20, pady=(0, 15))
        
    def _load_files(self):
        # Clear existing widgets
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        # Get PDF files
        pdf_files = []
        
        if os.path.exists(self.output_dir):
            for file in os.listdir(self.output_dir):
                if file.lower().endswith('.pdf'):
                    file_path = os.path.join(self.output_dir, file)
                    file_size = os.path.getsize(file_path)
                    file_modified = os.path.getmtime(file_path)
                    pdf_files.append({
                        'name': file,
                        'path': file_path,
                        'size': file_size,
                        'modified': file_modified
                    })
        
        # Sort based on user selection
        sort_option = self.sort_var.get()
        if sort_option == "Date (Newest)":
            pdf_files.sort(key=lambda x: x['modified'], reverse=True)
        elif sort_option == "Date (Oldest)":
            pdf_files.sort(key=lambda x: x['modified'])
        elif sort_option == "Name (A-Z)":
            pdf_files.sort(key=lambda x: x['name'].lower())
        elif sort_option == "Size":
            pdf_files.sort(key=lambda x: x['size'], reverse=True)
        
        # Display files
        if pdf_files:
            for file_info in pdf_files:
                self._create_file_entry(file_info)
            self.status_label.configure(text=f"Found {len(pdf_files)} meeting minute(s)")
        else:
            no_files_label = ctk.CTkLabel(
                self.scrollable_frame,
                text="No meeting minutes found.\nExported PDFs will appear here.",
                font=("Inter", 14),
                text_color="gray"
            )
            no_files_label.pack(pady=50)
            self.status_label.configure(text="No files found")
    
    def _create_file_entry(self, file_info):
        # File Entry Frame
        entry_frame = ctk.CTkFrame(
            self.scrollable_frame,
            fg_color="#2b2b2b",
            corner_radius=8
        )
        entry_frame.pack(fill="x", pady=5, padx=5)
        
        # Info Frame (Left Side)
        info_frame = ctk.CTkFrame(entry_frame, fg_color="transparent")
        info_frame.pack(side="left", fill="both", expand=True, padx=15, pady=12)
        
        # File Name
        display_name = self._truncate_filename(file_info['name'])
        name_label = ctk.CTkLabel(
            info_frame,
            text=display_name,
            font=("Inter", 13, "bold"),
            anchor="w",
            width=320
        )
        name_label.pack(anchor="w")
        
        # File Details
        size_kb = file_info['size'] / 1024
        modified_date = datetime.fromtimestamp(file_info['modified']).strftime("%B %d, %Y at %I:%M %p")
        details_text = f"📊 {size_kb:.1f} KB  •  📅 {modified_date}"
        
        details_label = ctk.CTkLabel(
            info_frame,
            text=details_text,
            font=("Inter", 10),
            text_color="gray",
            anchor="w"
        )
        details_label.pack(anchor="w", pady=(3, 0))
        
        # Action Indicator
        action_label = ctk.CTkLabel(
            info_frame,
            text="📝 Action Items Extracted",
            font=("Inter", 9),
            text_color="#4CAF50",
            anchor="w"
        )
        action_label.pack(anchor="w", pady=(2, 0))
        
        # Button Frame (Right Side)
        btn_frame = ctk.CTkFrame(entry_frame, fg_color="transparent")
        btn_frame.pack(side="right", padx=10)
        
        # Open Button
        open_btn = ctk.CTkButton(
            btn_frame,
            text="📖 Open",
            width=90,
            height=32,
            fg_color="#1f6aa5",
            hover_color="#1a5280",
            command=lambda: self._open_file(file_info['path'])
        )
        open_btn.pack(side="left", padx=5)
        
        # Open Folder Button
        folder_btn = ctk.CTkButton(
            btn_frame,
            text="📁",
            width=40,
            height=32,
            fg_color="#404040",
            hover_color="#505050",
            command=lambda: self._open_folder(file_info['path'])
        )
        folder_btn.pack(side="left", padx=5)
    
    def _open_file(self, file_path):
        """Open the PDF file with default application"""
        try:
            os.startfile(file_path)
        except Exception as e:
            print(f"Error opening file: {e}")
    
    def _open_folder(self, file_path):
        """Open the folder containing the file"""
        try:
            folder_path = os.path.dirname(file_path)
            os.startfile(folder_path)
        except Exception as e:
            print(f"Error opening folder: {e}")
    
    def _on_search(self, event=None):
        """Filter files based on search query"""
        search_query = self.search_entry.get().lower()
        
        for widget in self.scrollable_frame.winfo_children():
            if isinstance(widget, ctk.CTkFrame):
                # Get the file name from the frame's children
                for child in widget.winfo_children():
                    if isinstance(child, ctk.CTkFrame):
                        for label in child.winfo_children():
                            if isinstance(label, ctk.CTkLabel):
                                file_name = label.cget("text").lower()
                                if search_query in file_name:
                                    widget.pack(fill="x", pady=5, padx=5)
                                else:
                                    widget.pack_forget()
                                break
                        break

    def _truncate_filename(self, file_name, max_chars=45):
        if len(file_name) <= max_chars:
            return file_name
        return file_name[:max_chars - 3] + "..."
