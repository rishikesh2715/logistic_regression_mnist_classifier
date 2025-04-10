import os
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from joblib import load
from skimage.feature import hog
from skimage.measure import regionprops, label

# ----------------- Preprocessing and Feature Extraction ----------------- #
def preprocess_and_mask(gray):
    """
    Preprocessing pipeline:
     - Median blur
     - Canny edge detection and blending
     - Otsu thresholding to get a binary mask
     - Morphological open (1 iteration) and close (1 iteration)
     - Contour filtering: keep only contours with area in [200,4000],
       eccentricity >= 0.8, and solidity >= 0.4.
    Returns the final binary mask.
    """
    # Median blur
    gray_blur = cv2.medianBlur(gray, 3)
    # Canny edges and blend
    edges = cv2.Canny(gray_blur, 50, 150)
    blended = cv2.addWeighted(gray_blur, 0.8, edges, 0.2, 0)
    # Otsu thresholding
    _, mask = cv2.threshold(blended, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Morphological open and close (1 iteration each)
    se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se1, iterations=1)
    se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se2, iterations=1)
    # Contour filtering: convert to uint8 and filter regions
    mask_uint8 = (mask > 0).astype(np.uint8)
    lbl = label(mask_uint8)
    keep = np.zeros_like(mask_uint8)
    for r in regionprops(lbl):
        if (200 <= r.area <= 4000 and 
            r.eccentricity >= 0.8 and 
            r.solidity >= 0.4):
            keep[lbl == r.label] = 255
    return keep

def extract_features(gray):
    """
    Extract features from a grayscale image:
      1. Create a binary mask with preprocess_and_mask.
      2. Compute HOG (with orientations=12, pixels per cell 8×8, cells per block 2×2)
         on the masked image (set to zero where mask==0).
      3. From the mask compute shape features using regionprops:
         area, eccentricity, solidity, and axis ratio.
      4. Concatenate the HOG feature vector with these four scalar features.
    """
    mask = preprocess_and_mask(gray)
    masked = gray.copy()
    masked[mask == 0] = 0
    hog_vec = hog(
        masked,
        orientations=12,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2),
        visualize=False,
        feature_vector=True
    ).astype(np.float32)
    props = regionprops((mask > 0).astype(np.uint8))
    if props:
        rp = props[0]
        shape_feats = np.array([
            rp.area,
            rp.eccentricity,
            rp.solidity,
            rp.major_axis_length / (rp.minor_axis_length + 1e-3)
        ], dtype=np.float32)
    else:
        shape_feats = np.zeros(4, dtype=np.float32)
    return np.concatenate([hog_vec, shape_feats])

# ----------------- GUI for Inference ----------------- #
class PredictWormGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Predict Worm/No-Worm")
        self.geometry("400x300")
        self.model = None
        
        # Persistent reference for Tkinter images to avoid garbage collection.
        self._img_ref = None
        
        # Model selection button
        self.btn_model = tk.Button(self, text="Select Trained Model", font=("Arial", 12),
                                   command=self.select_model)
        self.btn_model.pack(pady=10, fill=tk.X, padx=20)
        
        # Inference buttons (disabled until model loaded)
        self.btn_individual = tk.Button(self, text="Select Individual Images", font=("Arial", 12),
                                        command=self.select_individual, state=tk.DISABLED)
        self.btn_individual.pack(pady=10, fill=tk.X, padx=20)
        self.btn_directory = tk.Button(self, text="Select Directory of Images", font=("Arial", 12),
                                       command=self.select_directory, state=tk.DISABLED)
        self.btn_directory.pack(pady=10, fill=tk.X, padx=20)
        
        # Text widget to display results
        self.txt = tk.Text(self, height=8)
        self.txt.pack(pady=10, padx=10, fill=tk.BOTH)
        
    def select_model(self):
        path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Joblib Files", "*.joblib"), ("All Files", "*.*")]
        )
        if path and os.path.isfile(path):
            try:
                self.model = load(path)
                self.txt.insert(tk.END, f"Model loaded from: {path}\n")
                self.btn_individual.config(state=tk.NORMAL)
                self.btn_directory.config(state=tk.NORMAL)
            except Exception as e:
                messagebox.showerror("Error", f"Could not load model: {e}")
        else:
            messagebox.showinfo("Info", "No model selected.")
    
    def predict_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, None
        feats = extract_features(img)
        feats = feats.reshape(1, -1)
        pred = self.model.predict(feats)[0]
        # If available, get the probability for the predicted class.
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(feats)[0][int(pred)]
        else:
            proba = None
        return pred, proba
    
    def display_prediction(self, image_path, pred, proba):
        # Load original image in color
        disp = cv2.imread(image_path)
        if disp is None:
            return
        # Further shorten text using abbreviations. For example "P" for prediction,
        # and rounding the probability for brevity.
        text = f"P:{pred}"
        if proba is not None:
            text += f"({proba*100:.0f}%)"
        cv2.putText(disp, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Resize image proportionally to fit within a maximum size of 800x600
        max_width, max_height = 800, 600
        h, w = disp.shape[:2]
        scale = min(max_width / w, max_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(disp, (new_w, new_h))
        
        cv2.namedWindow("Prediction", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Prediction", new_w, new_h)
        cv2.imshow("Prediction", resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def predict_and_report(self, image_paths):
        for path in image_paths:
            try:
                pred, proba = self.predict_image(path)
                if pred is None:
                    self.txt.insert(tk.END, f"Failed to load: {path}\n")
                    continue
                result = f"{os.path.basename(path)}: {pred}"
                if proba is not None:
                    result += f" ({proba*100:.1f}%)"
                result += "\n"
                self.txt.insert(tk.END, result)
                self.display_prediction(path, pred, proba)
            except Exception as e:
                self.txt.insert(tk.END, f"Error processing {path}: {e}\n")
    
    def select_individual(self):
        paths = filedialog.askopenfilenames(
            title="Select Image Files",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )
        if paths:
            self.predict_and_report(list(paths))
    
    def select_directory(self):
        directory = filedialog.askdirectory(title="Select Directory")
        if directory:
            paths = [os.path.join(directory, f) for f in os.listdir(directory)
                     if f.lower().endswith(('.png','.jpg','.jpeg'))]
            if paths:
                self.predict_and_report(paths)
            else:
                messagebox.showerror("Error", "No images found in selected directory.")

if __name__ == "__main__":
    app = PredictWormGUI()
    app.mainloop()
