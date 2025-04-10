import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from skimage.feature import hog, local_binary_pattern
import time
from sklearn.model_selection import GridSearchCV

def preprocess(img):
    """Enhanced preprocessing pipeline"""
    # CLAHE for adaptive contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    
    # Denoising
    img = cv2.medianBlur(img, 3)
    
    # Edge emphasis
    edges = cv2.Canny(img, 50, 150)
    return cv2.addWeighted(img, 0.8, edges, 0.2, 0)

def extract_features(image_path):
    """Feature extraction with HOG and LBP"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = preprocess(img)
    
    # Enhanced HOG parameters
    hog_features = hog(img, 
                     orientations=12,
                     pixels_per_cell=(8, 8),
                     cells_per_block=(2, 2),
                     visualize=False)
    
    # Texture features (LBP)
    lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10))
    
    return np.concatenate([hog_features, lbp_hist])

def load_dataset(directory):
    """Load and label dataset"""
    X, y = [], []
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, directory)
    
    for label in [0, 1]:
        folder = os.path.join(dataset_dir, str(label))
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Directory not found: {folder}")
            
        for img_file in os.listdir(folder):
            if img_file.endswith('.png'):
                try:
                    features = extract_features(os.path.join(folder, img_file))
                    X.append(features)
                    y.append(label)
                except Exception as e:
                    print(f"Error processing {img_file}: {str(e)}")
    
    return np.array(X), np.array(y)

# Main workflow
print("Loading dataset...")
X, y = load_dataset("")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optimized Random Forest with grid search
print("\nTraining model...")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced', None]
}

model = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1
)

start_time = time.time()
model.fit(X_train, y_train)
train_time = time.time() - start_time

# Evaluation
print("\n=== Evaluation ===")
print(f"Best parameters: {model.best_params_}")
print(f"Training time: {train_time:.2f} seconds")

test_pred = model.predict(X_test)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_pred))

print("\nClassification Report:")
print(classification_report(y_test, test_pred))

# Save model
from joblib import dump
dump(model, 'optimized_worm_detector.joblib')
print("\nModel saved as 'optimized_worm_detector.joblib'")