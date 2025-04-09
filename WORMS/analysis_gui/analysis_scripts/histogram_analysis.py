import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

class HistogramAnalysisGUI(tk.Toplevel):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.title("Histogram Analysis")
        self.state("zoomed")  # Maximize this window
        self.protocol("WM_DELETE_WINDOW", self.on_close)  # Re-show main window on close

        self.image_panels = []  # List to track each panel (image + histogram)
        self.image_count = 0    # Counter for total images

        # Left control panel
        self.control_frame = tk.Frame(self, padx=10, pady=10)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(self.control_frame, text="Histogram Parameters", font=("Arial", 14, "bold")).pack(pady=5)
        
        tk.Label(self.control_frame, text="Bins:").pack(anchor="w")
        self.bins_var = tk.StringVar(value="256")
        self.bins_entry = tk.Entry(self.control_frame, textvariable=self.bins_var)
        self.bins_entry.pack(anchor="w", fill=tk.X)

        self.load_button = tk.Button(self.control_frame, text="Select Images", command=self.load_images)
        self.load_button.pack(pady=10, fill=tk.X)

        self.clear_button = tk.Button(self.control_frame, text="Clear Images", command=self.clear_images)
        self.clear_button.pack(pady=5, fill=tk.X)

        # Right display area (scrollable)
        self.display_frame = tk.Frame(self)
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.display_frame)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar = tk.Scrollbar(self.display_frame, orient="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill="y")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Container for placing panels in a grid
        self.image_container = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.image_container, anchor="nw")
        self.image_container.bind("<Configure>", self.on_frame_configure)

    def on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.rearrange_grid()

    def load_images(self):
        # User selects images
        file_paths = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )
        try:
            bins = int(self.bins_var.get())
        except Exception as e:
            messagebox.showerror("Parameter Error", f"Error parsing bins parameter: {e}")
            return

        for fp in file_paths:
            try:
                # Load image (grayscale)
                img = Image.open(fp).convert("L")
                img_np = np.array(img)
                
                # Plot histogram with matplotlib
                fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
                ax.hist(img_np.ravel(), bins=bins, color='gray', edgecolor='black')
                ax.set_title("Histogram")
                ax.set_xlabel("Pixel Value")
                ax.set_ylabel("Frequency")
                fig.tight_layout()

                # Save plot to memory
                buf = BytesIO()
                fig.savefig(buf, format='png')
                plt.close(fig)
                buf.seek(0)
                hist_img = Image.open(buf)

                # Side-by-side combination (original on left, histogram on right)
                img_w, img_h = img.size
                hist_w, hist_h = hist_img.size
                combined_w = img_w + hist_w
                combined_h = max(img_h, hist_h)

                combined = Image.new("L", (combined_w, combined_h))
                # Paste original image at (0,0)
                combined.paste(img, (0, 0))
                # Paste histogram at (img_w,0)
                combined.paste(hist_img, (img_w, 0))

                # Convert to Tkinter image
                imgtk = ImageTk.PhotoImage(combined)

                # Create a panel to hold the combined image + close button
                panel_frame = tk.Frame(self.image_container, bd=2, relief=tk.SUNKEN)
                panel_frame.grid(row=0, column=self.image_count, padx=5, pady=5)
                label = tk.Label(panel_frame, image=imgtk)
                label.image = imgtk  # Keep reference
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
        max_cols = 3

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
        # Re-show the main window and close this one
        self.main_window.deiconify()
        self.destroy()
