import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from joblib import load
import tkinter as tk
from tkinter import filedialog
import os

def preprocess(img):
    """Identical preprocessing to training"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.medianBlur(img, 3)
    edges = cv2.Canny(img, 50, 150)
    return cv2.addWeighted(img, 0.8, edges, 0.2, 0)

def extract_features(img):
    """Identical feature extraction to training"""
    img = preprocess(img)
    hog_features = hog(img, 
                     orientations=12,
                     pixels_per_cell=(8, 8),
                     cells_per_block=(2, 2),
                     visualize=False)
    lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10))
    return np.concatenate([hog_features, lbp_hist])

def predict_image(model_path, image_path):
    """Load model and predict"""
    model = load(model_path)
    img = cv2.imread(image_path)
    features = extract_features(img)
    proba = model.predict_proba([features])[0]
    return proba

def main():
    # Create GUI
    root = tk.Tk()
    root.withdraw()  # Hide main window

    # 1. Select model
    print("Select your trained model (.joblib file)")
    model_path = filedialog.askopenfilename(
        title="Select Model",
        filetypes=[("Joblib files", "*.joblib")]
    )
    if not model_path:
        print("No model selected. Exiting.")
        return

    # 2. Select image
    print("\nSelect an image to classify")
    image_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image files", "*.png *.jpg *.jpeg")]
    )
    if not image_path:
        print("No image selected. Exiting.")
        return

    # 3. Predict and show result
    proba = predict_image(model_path, image_path)
    confidence = max(proba)
    prediction = np.argmax(proba)
    
    print("\n=== Prediction ===")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Prediction: {'WORM' if prediction == 1 else 'NO WORM'}")
    print(f"Confidence: {confidence:.1%}")
    print("\nClass Probabilities:")
    print(f"- No Worm: {proba[0]:.1%}")
    print(f"- Worm: {proba[1]:.1%}")

    # 4. Display image
    img = cv2.imread(image_path)
    cv2.imshow("Worm Classifier", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()