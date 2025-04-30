import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2

class FrequencyAnalysisGUI(tk.Toplevel):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.title("Frequency Analysis")
        self.state("zoomed")  # Maximize window
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.image_panels = []  # List to track image panels
        self.image_count = 0    # Total images loaded

        # --- Left Control Panel ---
        self.control_frame = tk.Frame(self, padx=10, pady=10)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(self.control_frame, text="Frequency Analysis Parameters", font=("Arial", 14, "bold")).pack(pady=5)
        
        # Checkbox: Use Logarithmic Scale for the magnitude spectrum
        self.use_log_var = tk.IntVar(value=1)
        self.log_check = tk.Checkbutton(self.control_frame, text="Use Log Scale", variable=self.use_log_var)
        self.log_check.pack(anchor="w")

        self.load_button = tk.Button(self.control_frame, text="Select Images", command=self.load_images)
        self.load_button.pack(pady=10, fill=tk.X)

        self.clear_button = tk.Button(self.control_frame, text="Clear Images", command=self.clear_images)
        self.clear_button.pack(pady=5, fill=tk.X)

        # --- Right Display Area (Scrollable) ---
        self.display_frame = tk.Frame(self)
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.display_frame)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar = tk.Scrollbar(self.display_frame, orient="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill="y")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Container inside the canvas for placing image panels in a grid
        self.image_container = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.image_container, anchor="nw")
        self.image_container.bind("<Configure>", self.on_frame_configure)

    def on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.rearrange_grid()

    def load_images(self):
        # Allow user to select image files
        file_paths = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )
        use_log = bool(self.use_log_var.get())
        
        for fp in file_paths:
            try:
                # Load image using OpenCV (in grayscale)
                img_cv = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
                if img_cv is None:
                    raise Exception("Could not load image.")
                
                # Compute FFT and shift zero-frequency component to center
                dft = np.fft.fft2(img_cv)
                dft_shift = np.fft.fftshift(dft)
                magnitude = np.abs(dft_shift)
                
                # Apply logarithmic scaling if requested (avoids log(0) using +1)
                if use_log:
                    magnitude = np.log(magnitude + 1)
                
                # Normalize the magnitude to 0-255 and convert to uint8
                mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
                mag_norm = np.uint8(mag_norm)
                
                # Convert images to PIL format for display
                orig_pil = Image.fromarray(img_cv)
                freq_pil = Image.fromarray(mag_norm)
                
                # Create a combined image: original on the left, frequency response on the right
                orig_w, orig_h = orig_pil.size
                freq_w, freq_h = freq_pil.size
                combined_width = orig_w + freq_w
                combined_height = max(orig_h, freq_h)
                combined = Image.new("L", (combined_width, combined_height))
                combined.paste(orig_pil, (0, 0))
                combined.paste(freq_pil, (orig_w, 0))
                
                # Convert to a Tkinter image
                imgtk = ImageTk.PhotoImage(combined)

                # Create a panel to hold the image and a Close button
                panel_frame = tk.Frame(self.image_container, bd=2, relief=tk.SUNKEN)
                # Place temporarily; rearrange_grid() will handle placement
                panel_frame.grid(row=0, column=self.image_count, padx=5, pady=5)
                label = tk.Label(panel_frame, image=imgtk)
                label.image = imgtk  # Keep reference to avoid GC
                label.pack()
                close_btn = tk.Button(panel_frame, text="Close", command=lambda pf=panel_frame: self.close_panel(pf))
                close_btn.pack()
                self.image_panels.append(panel_frame)
                self.image_count += 1
            except Exception as e:
                print(f"Error processing {fp}: {e}")
        self.rearrange_grid()

    def rearrange_grid(self):
        # Use a fixed number of columns to limit to 10 images per row
        max_cols = 5

        # Rearrange panels: fill horizontally until 10 panels then wrap vertically
        for idx, panel in enumerate(self.image_panels):
            row = idx // max_cols
            col = idx % max_cols
            panel.grid_configure(row=row, column=col, padx=5, pady=5, sticky="nw")

    def close_panel(self, panel):
        panel.destroy()
        if panel in self.image_panels:
            self.image_panels.remove(panel)
        self.rearrange_grid()

    def clear_images(self):
        for panel in self.image_panels:
            panel.destroy()
        self.image_panels = []
        self.image_count = 0

    def on_close(self):
        self.destroy()
        self.main_window.deiconify()
