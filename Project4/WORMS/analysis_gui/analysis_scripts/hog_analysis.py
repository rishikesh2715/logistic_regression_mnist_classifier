import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from skimage import feature

class HOGAnalysisGUI(tk.Toplevel):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.title("HOG Analysis")
        self.state("zoomed")  # Maximize window
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.image_panels = []  # list to track image panels

        # Left control panel for HOG parameters and buttons
        self.control_frame = tk.Frame(self, padx=10, pady=10)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(self.control_frame, text="HOG Parameters", font=("Arial", 14, "bold")).pack(pady=5)
        
        tk.Label(self.control_frame, text="Orientations:").pack(anchor="w")
        self.orientations_var = tk.StringVar(value="9")
        self.orientations_entry = tk.Entry(self.control_frame, textvariable=self.orientations_var)
        self.orientations_entry.pack(anchor="w", fill=tk.X)

        tk.Label(self.control_frame, text="Pixels per cell (e.g., 8,8):").pack(anchor="w")
        self.pixels_var = tk.StringVar(value="8,8")
        self.pixels_entry = tk.Entry(self.control_frame, textvariable=self.pixels_var)
        self.pixels_entry.pack(anchor="w", fill=tk.X)

        tk.Label(self.control_frame, text="Cells per block (e.g., 3,3):").pack(anchor="w")
        self.cells_var = tk.StringVar(value="3,3")
        self.cells_entry = tk.Entry(self.control_frame, textvariable=self.cells_var)
        self.cells_entry.pack(anchor="w", fill=tk.X)

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

        # Container inside the canvas for placing image panels with grid
        self.image_container = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.image_container, anchor="nw")
        self.image_container.bind("<Configure>", self.on_frame_configure)

        # Counter for total images (used only for tracking additions)
        self.image_count = 0

    def on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        # Re-arrange grid whenever the container size changes
        self.rearrange_grid()

    def load_images(self):
        # Allow user to select one or more image files
        file_paths = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )
        
        # Parse HOG parameters
        try:
            orientations = int(self.orientations_var.get())
            pixels = tuple(int(x.strip()) for x in self.pixels_var.get().split(","))
            cells = tuple(int(x.strip()) for x in self.cells_var.get().split(","))
        except Exception as e:
            messagebox.showerror("Parameter Error", f"Error parsing parameters: {e}")
            return

        # Process each selected image
        for fp in file_paths:
            try:
                # Load and convert image to grayscale
                img = Image.open(fp).convert("L")
                img_np = np.array(img)
                
                # Compute HOG features and visualization
                hog_features, hog_image = feature.hog(
                    img_np,
                    orientations=orientations,
                    pixels_per_cell=pixels,
                    cells_per_block=cells,
                    block_norm='L2-Hys',
                    visualize=True,
                    feature_vector=True
                )
                # Normalize HOG visualization for display
                hog_image_disp = (hog_image - hog_image.min()) / (hog_image.max() - hog_image.min()) * 255.0
                hog_image_disp = hog_image_disp.astype(np.uint8)
                hog_image_pil = Image.fromarray(hog_image_disp)

                # Create a combined image: original on top, HOG visualization below
                combined = Image.new("L", (img.width, img.height * 2))
                combined.paste(img, (0, 0))
                hog_resized = hog_image_pil.resize((img.width, img.height))
                combined.paste(hog_resized, (0, img.height))

                # Convert combined image for Tkinter display
                imgtk = ImageTk.PhotoImage(combined)

                # Create a panel to hold the image and a close button
                panel_frame = tk.Frame(self.image_container, bd=2, relief=tk.SUNKEN)
                # Use grid placement temporarily; rearrange_grid() will position correctly.
                panel_frame.grid(row=0, column=self.image_count, padx=5, pady=5)
                label = tk.Label(panel_frame, image=imgtk)
                label.image = imgtk  # Keep a reference to avoid garbage collection
                label.pack()
                close_btn = tk.Button(panel_frame, text="Close", command=lambda pf=panel_frame: self.close_panel(pf))
                close_btn.pack()
                self.image_panels.append(panel_frame)
                self.image_count += 1
            except Exception as e:
                print(f"Error processing {fp}: {e}")
        # Rearrange after adding new images
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
        # Remove all image panels and reset counter
        for panel in self.image_panels:
            panel.destroy()
        self.image_panels = []
        self.image_count = 0

    def on_close(self):
        # When closing the HOG window, destroy it and re-show the main window
        self.destroy()
        self.main_window.deiconify()
