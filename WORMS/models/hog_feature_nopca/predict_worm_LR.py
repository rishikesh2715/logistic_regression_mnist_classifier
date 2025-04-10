import os
import sys
import cv2
import numpy as np
from joblib import load
from skimage.feature import hog, local_binary_pattern
import argparse
import tkinter as tk
from tkinter import filedialog, messagebox

# -------------------------------
# Preprocessing and Feature Extraction
# -------------------------------
def preprocess(img):
    """Enhanced preprocessing pipeline using CLAHE, median blur, and edge blending."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.medianBlur(img, 3)
    edges = cv2.Canny(img, 50, 150)
    return cv2.addWeighted(img, 0.8, edges, 0.2, 0)

def extract_features(img):
    """
    Extract combined HOG + LBP features from a preprocessed grayscale image.
    - HOG: orientations=12, pixels_per_cell=(8,8), cells_per_block=(2,2)
    - LBP: P=8, R=1, method="uniform"; then a 10-bin histogram (normalized).
    """
    # HOG features
    hog_features = hog(
        img,
        orientations=12,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
        transform_sqrt=False,
        feature_vector=True
    )
    # LBP features
    lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-7)
    # Concatenate
    return np.concatenate([hog_features, lbp_hist]).astype(np.float32)

def load_image(image_path):
    """Load an image in grayscale, preprocess it, and return the processed image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image {image_path}")
    img = preprocess(img)
    return img

def predict_image(model, image_path):
    """Run inference on a single image and return prediction (and probability if available)."""
    img = load_image(image_path)
    features = extract_features(img)
    features = features.reshape(1, -1)
    pred = model.predict(features)
    proba = model.predict_proba(features) if hasattr(model, "predict_proba") else None
    return pred[0], proba

def display_prediction(image_path, prediction, probability=None):
    """Display an image with the prediction overlayed using OpenCV."""
    img = cv2.imread(image_path)
    if img is None:
        return
    text = f"Predicted: {prediction}"
    if probability is not None:
        prob = probability[0][int(prediction)]
        text += f" ({prob*100:.1f}%)"
    cv2.putText(img, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -------------------------------
# GUI for Inference Method Selection
# -------------------------------
class InferenceSelector(tk.Tk):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.title("Worm Prediction - Choose Inference Method")
        self.geometry("400x200")
        
        label = tk.Label(self, text="Select inference method:", font=("Arial", 14))
        label.pack(pady=20)
        
        btn_individual = tk.Button(self, text="Individual Images", font=("Arial", 12),
                                   command=self.choose_individual)
        btn_individual.pack(pady=5, fill=tk.X, padx=50)
        
        btn_directory = tk.Button(self, text="Choose Directory", font=("Arial", 12),
                                  command=self.choose_directory)
        btn_directory.pack(pady=5, fill=tk.X, padx=50)
        
        btn_quit = tk.Button(self, text="Quit", font=("Arial", 12), command=self.quit)
        btn_quit.pack(pady=10)
    
    def choose_individual(self):
        file_paths = filedialog.askopenfilenames(
            title="Select Image Files",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )
        if file_paths:
            self.run_inference(list(file_paths))
    
    def choose_directory(self):
        directory = filedialog.askdirectory(title="Select Directory Containing Images")
        if directory:
            files = [os.path.join(directory, f) for f in os.listdir(directory)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if files:
                self.run_inference(files)
            else:
                messagebox.showerror("No Images", "No image files found in the selected directory.")
    
    def run_inference(self, image_files):
        results = {}
        for image_path in image_files:
            try:
                pred, proba = predict_image(self.model, image_path)
                results[image_path] = (pred, proba)
                print(f"Image: {os.path.basename(image_path)} -> Predicted: {pred}")
                display_prediction(image_path, pred, proba)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        # Optionally, save or log results here
        self.destroy()

# -------------------------------
# Main Inference Script
# -------------------------------
def main():
    # First, prompt user to select the trained model file
    root = tk.Tk()
    root.withdraw()
    model_path = filedialog.askopenfilename(
        title="Select Trained Model File",
        filetypes=[("Joblib Files", "*.joblib"), ("All Files", "*.*")]
    )
    if not model_path or not os.path.isfile(model_path):
        print("No model selected. Exiting.")
        sys.exit(1)
    print(f"Loading model from: {model_path}")
    model = load(model_path)
    
    # Now start the inference GUI that asks whether to choose individual images or a directory.
    selector = InferenceSelector(model)
    selector.mainloop()

if __name__ == "__main__":
    main()
 