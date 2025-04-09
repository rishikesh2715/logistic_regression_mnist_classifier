import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from skimage import feature

class LBPAnalysisGUI(tk.Toplevel):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.title("LBP Analysis")
        self.state("zoomed")  # Maximize window
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.image_panels = []  # To keep track of image panels
        self.image_count = 0    # Total number of images loaded

        # Left control panel for LBP parameters and buttons
        self.control_frame = tk.Frame(self, padx=10, pady=10)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(self.control_frame, text="LBP Parameters", font=("Arial", 14, "bold")).pack(pady=5)

        tk.Label(self.control_frame, text="Radius:").pack(anchor="w")
        self.radius_var = tk.StringVar(value="1")
        self.radius_entry = tk.Entry(self.control_frame, textvariable=self.radius_var)
        self.radius_entry.pack(anchor="w", fill=tk.X)

        tk.Label(self.control_frame, text="Number of Points:").pack(anchor="w")
        self.points_var = tk.StringVar(value="8")
        self.points_entry = tk.Entry(self.control_frame, textvariable=self.points_var)
        self.points_entry.pack(anchor="w", fill=tk.X)

        tk.Label(self.control_frame, text="Method (e.g., uniform):").pack(anchor="w")
        self.method_var = tk.StringVar(value="uniform")
        self.method_entry = tk.Entry(self.control_frame, textvariable=self.method_var)
        self.method_entry.pack(anchor="w", fill=tk.X)

        self.load_button = tk.Button(self.control_frame, text="Select Images", command=self.load_images)
        self.load_button.pack(pady=10, fill=tk.X)

        self.clear_button = tk.Button(self.control_frame, text="Clear Images", command=self.clear_images)
        self.clear_button.pack(pady=5, fill=tk.X)

        # Right display area: a scrollable canvas for image panels
        self.display_frame = tk.Frame(self)
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.display_frame)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar = tk.Scrollbar(self.display_frame, orient="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill="y")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Container inside the canvas for placing image panels using grid layout
        self.image_container = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.image_container, anchor="nw")
        self.image_container.bind("<Configure>", self.on_frame_configure)

    def on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        # Rearrange panels when the container size changes
        self.rearrange_grid()

    def load_images(self):
        # Allow user to select one or more image files
        file_paths = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )
        try:
            radius = float(self.radius_var.get())
            num_points = int(self.points_var.get())
            method = self.method_var.get().strip()
        except Exception as e:
            messagebox.showerror("Parameter Error", f"Error parsing parameters: {e}")
            return

        # Process each selected image
        for fp in file_paths:
            try:
                # Load and convert to grayscale
                img = Image.open(fp).convert("L")
                img_np = np.array(img)
                # Compute LBP features
                lbp = feature.local_binary_pattern(img_np, num_points, radius, method=method)
                # For uniform LBP, maximum value is num_points + 2; otherwise use the maximum found
                max_val = num_points + 2 if method == "uniform" else lbp.max()
                lbp_disp = (lbp / max_val) * 255.0
                lbp_disp = lbp_disp.astype(np.uint8)
                lbp_image = Image.fromarray(lbp_disp)

                # Create a combined image: original on top, LBP visualization below
                combined = Image.new("L", (img.width, img.height * 2))
                combined.paste(img, (0, 0))
                lbp_resized = lbp_image.resize((img.width, img.height))
                combined.paste(lbp_resized, (0, img.height))

                # Convert for Tkinter display
                imgtk = ImageTk.PhotoImage(combined)

                # Create a panel to hold the image and a "Close" button
                panel_frame = tk.Frame(self.image_container, bd=2, relief=tk.SUNKEN)
                panel_frame.grid(row=0, column=self.image_count, padx=5, pady=5)
                label = tk.Label(panel_frame, image=imgtk)
                label.image = imgtk  # Keep reference to avoid garbage collection
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
        max_cols = 10

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
