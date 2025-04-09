import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from skimage import filters

class BinarizationAnalysisGUI(tk.Toplevel):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.title("Binarization Analysis")
        self.state("zoomed")  # Maximize the window
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.image_panels = []  # List to track image panels
        self.image_count = 0    # Counter for total images loaded

        # --- Left Control Panel ---
        self.control_frame = tk.Frame(self, padx=10, pady=10)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(self.control_frame, text="Binarization Parameters", font=("Arial", 14, "bold")).pack(pady=5)

        # Option to apply Gaussian blur
        self.apply_blur_var = tk.IntVar(value=0)
        self.blur_check = tk.Checkbutton(self.control_frame, text="Apply Gaussian Blur", variable=self.apply_blur_var)
        self.blur_check.pack(anchor="w")

        tk.Label(self.control_frame, text="Gaussian Sigma (if blur applied):").pack(anchor="w")
        self.sigma_var = tk.StringVar(value="1.0")
        self.sigma_entry = tk.Entry(self.control_frame, textvariable=self.sigma_var)
        self.sigma_entry.pack(anchor="w", fill=tk.X)

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

        # Container inside the canvas for grid layout of image panels
        self.image_container = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.image_container, anchor="nw")
        self.image_container.bind("<Configure>", self.on_frame_configure)

    def on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.rearrange_grid()

    def load_images(self):
        # Allow the user to select one or more image files
        file_paths = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )
        try:
            sigma = float(self.sigma_var.get())
        except Exception as e:
            messagebox.showerror("Parameter Error", f"Error parsing sigma parameter: {e}")
            return

        apply_blur = bool(self.apply_blur_var.get())

        for fp in file_paths:
            try:
                # Load the image and convert to grayscale
                img = Image.open(fp).convert("L")
                img_np = np.array(img)

                # Optionally apply Gaussian blur
                if apply_blur:
                    img_np = filters.gaussian(img_np, sigma=sigma)

                # Compute Otsu's threshold
                thresh = filters.threshold_otsu(img_np)
                binary = img_np > thresh
                # Convert boolean array to uint8 image (0 or 255)
                binary_img = (binary * 255).astype(np.uint8)
                bin_img_pil = Image.fromarray(binary_img)

                # Create a combined image: original on the left, binarized on the right
                combined_width = img.width + bin_img_pil.width
                combined_height = max(img.height, bin_img_pil.height)
                combined = Image.new("L", (combined_width, combined_height))
                combined.paste(img, (0, 0))
                combined.paste(bin_img_pil, (img.width, 0))

                # Convert to an image for Tkinter
                imgtk = ImageTk.PhotoImage(combined)

                # Create a panel frame to hold the combined image and a Close button
                panel_frame = tk.Frame(self.image_container, bd=2, relief=tk.SUNKEN)
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
        self.rearrange_grid()

    def rearrange_grid(self):
        # Determine the number of columns that fit based on the container width and panel width
        container_width = self.image_container.winfo_width()
        if container_width <= 1:
            container_width = self.winfo_width() - 50  # fallback if not yet measured

        if self.image_panels:
            panel_width = self.image_panels[0].winfo_reqwidth() + 10
        else:
            panel_width = 200  # fallback

        max_cols = max(1, container_width // panel_width)

        for idx, panel in enumerate(self.image_panels):
            row = idx // max_cols
            col = idx % max_cols
            panel.grid_configure(row=row, column=col)

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
        # When closing this window, re-show the main window and destroy this one
        self.destroy()
        self.main_window.deiconify()
