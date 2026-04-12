import customtkinter as ctk
import os
from datetime import datetime
import subprocess
import sys
import threading # for handling subprocess without freezing the UI
import shutil # for handling file copying during upload

class VideoFileManager(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        
        # Window Config
        self.title("Video Files Manager")
        self.geometry("560x500")
        self.minsize(560, 500)
        self.maxsize(560, 500)
        self.resizable(False, False)
        
        # Get base directory
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.output_dir = os.path.join(self.base_dir, "output", "videos")
        self._processing_active = False
        self._processing_base_text = "Processing file"
        self._processing_dot_count = 0
        
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
            text="📹 Video Recordings", 
            font=("Inter", 20, "bold")
        )
        title_label.pack(side="left")
        
        # Search Frame
        search_frame = ctk.CTkFrame(self, fg_color="transparent")
        search_frame.pack(fill="x", padx=20, pady=(0, 10))
        
        self.search_entry = ctk.CTkEntry(
            search_frame, 
            placeholder_text="Search videos...",
            height=35
        )
        self.search_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.search_entry.bind("<KeyRelease>", self._on_search)
        
        # button frames for refresh and upload
        button_frame = ctk.CTkFrame(search_frame, fg_color="transparent")
        button_frame.pack(side="left", padx=(0, 10))
        
        # change to logo buttons for refresh and upload
        refresh_btn = ctk.CTkButton(
            button_frame,
            text="🔄",
            width=40,
            height=35,
            font=("Inter", 18),
            command=self._load_files
        )
        refresh_btn.pack(side="left", padx=5)
        
        # added folder upload button for video manager (logo)
        upload_btn = ctk.CTkButton(
            button_frame,
            text="📂",
            width=40,
            height=35,
            font=("Inter", 18),
            command=self._upload_video
        )
        upload_btn.pack(side="left", padx=5)
        
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
        
        # Get video files
        video_extensions = ('.wav', '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.mp3')
        video_files = []
        
        if os.path.exists(self.output_dir):
            for file in os.listdir(self.output_dir):
                if file.lower().endswith(video_extensions):
                    file_path = os.path.join(self.output_dir, file)
                    file_size = os.path.getsize(file_path)
                    file_modified = os.path.getmtime(file_path)
                    video_files.append({
                        'name': file,
                        'path': file_path,
                        'size': file_size,
                        'modified': file_modified
                    })
        
        # Sort by modified date (newest first)
        video_files.sort(key=lambda x: x['modified'], reverse=True)
        
        # Display files
        if video_files:
            for file_info in video_files:
                self._create_file_entry(file_info)
            self.status_label.configure(text=f"Found {len(video_files)} video file(s)")
        else:
            no_files_label = ctk.CTkLabel(
                self.scrollable_frame,
                text="No video files found.\nRecorded videos will appear here.",
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
        info_frame.pack(side="left", fill="both", expand=True, padx=15, pady=10)
        
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
        size_mb = file_info['size'] / (1024 * 1024)
        modified_date = datetime.fromtimestamp(file_info['modified']).strftime("%Y-%m-%d %H:%M")
        details_text = f"📊 {size_mb:.2f} MB  •  📅 {modified_date}"
        
        details_label = ctk.CTkLabel(
            info_frame,
            text=details_text,
            font=("Inter", 10),
            text_color="gray",
            anchor="w"
        )
        details_label.pack(anchor="w", pady=(3, 0))
        
        # Button Frame (Right Side)
        btn_frame = ctk.CTkFrame(entry_frame, fg_color="transparent")
        btn_frame.pack(side="right", padx=10)
        
        # changed the "Open" button to a logo button and sized it down to fit better with the new design
        open_btn = ctk.CTkButton(
            btn_frame,
            text="▶",
            width=40,
            height=30,
            font=("Inter", 14),
            command=lambda: self._open_file(file_info['path'])
        )
        open_btn.pack(side="left", padx=5)
        
        # changed to a logo button for PDF transfer and sized it down to fit better with the new design
        # this now is a terminal based containing the main.py function
        pdf_btn = ctk.CTkButton(
            btn_frame,
            text="📄",
            width=40,
            height=30,
            font=("Inter", 14),
            command=lambda: self._transfer_to_pdf(file_info['path'])
        )
        pdf_btn.pack(side="left", padx=5)
    
    def _open_file(self, file_path):
        """Open the video file with default application"""
        try:
            os.startfile(file_path)
        except Exception as e:
            print(f"Error opening file: {e}")
    
    # added upload function for video manager, 
    # allowing users to upload their own videos to the output directory 
    # and have them appear in the file manager list
    def _upload_video(self):
        """Allow user to upload a video file without creating a new root"""
        try:
            from tkinter import filedialog
            
            # Using 'self' as parent instead of creating a new ctk.CTk()
            # because filedialog can work with the existing window and won't create a new one
            # this have a heavy performance boost compared to creating a new root every time the upload button is clicked, 
            # which can cause memory leaks and multiple windows if not handled properly
            file_path = filedialog.askopenfilename(
                parent=self,
                title="Select a video file to upload",
                filetypes=[
                    ("Video Files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
                    ("Audio Files", "*.mp3 *.wav"),
                    ("All Files", "*.*")
                ]
            )
            
            if file_path:
                file_name = os.path.basename(file_path)
                destination = os.path.join(self.output_dir, file_name)
                
                # check if the selected file is already in the output directory to avoid unnecessary copying and potential overwriting
                if os.path.abspath(file_path) == os.path.abspath(destination):
                    return

                shutil.copy2(file_path, destination)
                print(f"[System] Video uploaded: {file_name}")
                self._load_files()
                
        except Exception as e:
            print(f"Error uploading video: {e}")

    def _transfer_to_pdf(self, file_path):
        """Extract processes from video using a thread to prevent GUI freezing"""
        
        self._start_processing_status(os.path.basename(file_path))
        app_logic = getattr(self.master, "logic", None)
        if app_logic is not None:
            self.after(
                0,
                lambda: app_logic._append_system_text(
                    f"Video manager: currently processing {os.path.basename(file_path)}..."
                ),
            )

        def run_process():
            started = datetime.now()
            try:
                if app_logic is not None:
                    result_data = app_logic.process_file_path_for_pdf(file_path)
                    elapsed = (datetime.now() - started).total_seconds()
                    print(f"[Debug] Video->PDF in-app processing finished in {elapsed:.1f}s")
                    self.after(
                        0,
                        lambda: self._finish_processing_status(
                            f"Finished in {elapsed:.1f}s: {os.path.basename(file_path)}",
                            "green",
                        ),
                    )
                    self.after(
                        0,
                        lambda: app_logic._append_system_text(
                            f"Video manager PDF generated: {result_data['pdf_path']}"
                        ),
                    )
                    self.after(
                        0,
                        lambda: app_logic._append_system_text(
                            f"Video manager timing: total={elapsed:.1f}s"
                        ),
                    )
                    return

                # Fallback for environments where app logic is unavailable
                result = subprocess.run(
                    [sys.executable, "Main.py", file_path],
                    cwd=self.base_dir,
                    capture_output=True,
                    text=True,
                    check=True
                )
                elapsed = (datetime.now() - started).total_seconds()
                print(f"[Debug] Video->PDF processing finished in {elapsed:.1f}s")
                if result.stdout:
                    print("[Debug] Main.py stdout (tail):", result.stdout[-600:])
                if result.stderr:
                    print("[Debug] Main.py stderr (tail):", result.stderr[-600:])
                
                # used the self.after method to update the status label from the thread, 
                # since we can't update GUI elements directly from a non-main thread
                self.after(0, lambda: self._finish_processing_status(
                    f"Finished in {elapsed:.1f}s: {os.path.basename(file_path)}",
                    "green"
                ))
                if app_logic is not None:
                    self.after(
                        0,
                        lambda: app_logic._append_system_text(
                            f"Video manager processing finished in {elapsed:.1f}s: {os.path.basename(file_path)}"
                        ),
                    )

            except Exception as e:
                elapsed = (datetime.now() - started).total_seconds()
                print(f"[Error] Video->PDF processing failed after {elapsed:.1f}s: {e}")
                self.after(0, lambda: self._finish_processing_status("Error during processing", "red"))
                if app_logic is not None:
                    self.after(
                        0,
                        lambda: app_logic._append_system_text(
                            f"Video manager processing failed after {elapsed:.1f}s: {e}"
                        ),
                    )

        # start the thread as 'daemon' so it closes if the app is closed
        process_thread = threading.Thread(target=run_process, daemon=True)
        process_thread.start()

    def _truncate_filename(self, file_name, max_chars=45):
        if len(file_name) <= max_chars:
            return file_name
        return file_name[:max_chars - 3] + "..."

    def _start_processing_status(self, file_name):
        self._processing_active = True
        self._processing_base_text = f"Processing {file_name}"
        self._processing_dot_count = 0
        self._animate_processing_status()

    def _animate_processing_status(self):
        if not self._processing_active:
            return

        dots = "." * ((self._processing_dot_count % 3) + 1)
        self.status_label.configure(text=f"{self._processing_base_text}{dots}", text_color="#3a7ebf")
        self._processing_dot_count += 1
        self.after(450, self._animate_processing_status)

    def _finish_processing_status(self, text, color):
        self._processing_active = False
        self.status_label.configure(text=text, text_color=color)
    
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
